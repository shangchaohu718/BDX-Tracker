# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity
import torch.nn.functional as F
from torch import autograd
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..fb.agent import FBAgent, FBAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode, grad_norm  # [BFM-DIAG-NAN] grad_norm added by Claude
from ..pytree_utils import tree_get_batch_size
from .model import FBcprModel, FBcprModelConfig


class FBcprAgentTrainConfig(FBAgentTrainConfig):
    lr_discriminator: float = 1e-4
    lr_critic: float = 1e-4
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 1
    scale_reg: bool = True
    # the z distribution for rollouts (when agent.use_mix_rollout=1) and for the mini-batches used in the network updates is:
    # - a fraction of 'expert_asm_ratio' zs from expert trajectory encoding
    # - a fraction of 'train_goal_ratio' zs from goal encoding (goals sampled from the train buffer)
    # - the remaining fraction from the uniform distribution
    expert_asm_ratio: float = 0
    # a fraction of 'relabel_ratio' transitions in each mini-batch are relabeled with a z sampled from the above distribution
    relabel_ratio: float | None = 1
    grad_penalty_discriminator: float = 10.0
    # Commented out as example but not useful
    # grad_penalty_obs_weight: float | None = None  # must be in (0,1)
    weight_decay_discriminator: float = 0.0
    # [BFM-DOMAIN-FIX] bound the raw-logit imitation reward used by the critic.
    # BCE margins grow without bound on any separable pair — even legitimate
    # behavior separation — driving Q toward -c/(1-gamma) saturation.
    # None = unchanged raw logit. Diagnostics keep the unclipped reward.
    reward_clip: float | None = None
    # [BFM-COND-FIX] independent conditional ranking loss teaching D which z
    # pairs with which behavior (activity): softplus(margin - D(o,z+) + D(o,z-)),
    # with z+ the window's own z and z- a z from the opposite-activity half of
    # the batch. Off (0.0) = unchanged. NOT mixed into the expert/fake BCE.
    cond_loss_coeff: float = 0.0
    cond_margin: float = 1.0
    cond_pair_frac: float = 0.25
    # Negative-only policy-domain conditional geometry. Policy samples remain
    # BCE-fake; these terms may only push a worse/mismatched pairing down.
    policy_cond_coeff: float = 0.0
    policy_quality_coeff: float = 0.0
    policy_cond_margin: float = 0.5
    policy_manifold_cos_min: float = 0.5
    policy_mismatch_cos_max: float = 0.85
    policy_quality_command_cos_min: float = 0.8
    policy_quality_gap_min: float = 0.1
    policy_max_abs_dof_vel: float = 10.0
    # [BFM-DOMAIN-FIX] batch-standardize the imitation reward:
    # r = (logit - batch_mean) / (batch_std + eps), applied before reward_clip.
    # Preserves ranking while canceling the BCE margin growth that narrows any
    # fixed clip window over time (clip 1/2/3 A/B: bandwidth all decay to 0).
    # Use reward_clip only as a wide numerical safety valve (e.g. 5).
    reward_norm: bool = False


class FBcprAgentConfig(BaseConfig):
    name: Literal["FBcprAgent"] = "FBcprAgent"
    model: FBcprModelConfig = FBcprModelConfig()
    train: FBcprAgentTrainConfig = FBcprAgentTrainConfig()
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return FBcprAgent(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBcprAgent


class FBcprAgent(FBAgent):
    config_class = FBcprAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBcprAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        # make sure batch size is a multiple of seq_length
        seq_length = cfg.model.seq_length
        batch_size = cfg.train.batch_size
        assert (batch_size / seq_length) == (batch_size // seq_length), "Batch size should be divisable by seq_length"
        del seq_length, batch_size

        self._model: FBcprModel = self.cfg.model.build(obs_space, action_dim)
        # This is just to be sure? I think it should not change since build
        self._model.to(self.device)
        self.setup_training()
        self.setup_compile()
        self.env_idx_with_expert_rollout = None

    @classmethod
    def supported_evaluations(cls):
        return ["reward", "tracking"]

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers.update(
            {
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            }
        )
        return optimizers

    def setup_training(self) -> None:
        super().setup_training()

        # prepare parameter list
        self._critic_map_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_map_paramlist = tuple(x for x in self._model._target_critic.parameters())

        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(),
            lr=self.cfg.train.lr_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(),
            lr=self.cfg.train.lr_discriminator,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_critic = torch.compile(self.update_critic, mode=mode)
            self.update_discriminator = torch.compile(self.update_discriminator, mode=mode)
            self.encode_expert = torch.compile(self.encode_expert, mode=mode, fullgraph=True)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_critic = CudaGraphModule(self.update_critic, warmup=5)
            self.update_discriminator = CudaGraphModule(self.update_discriminator, warmup=5)
            self.encode_expert = CudaGraphModule(self.encode_expert, warmup=5)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor, expert_encodings: torch.Tensor, *args, **kwargs):
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
            p_goal = self.cfg.train.train_goal_ratio
            p_expert_asm = self.cfg.train.expert_asm_ratio
            prob = torch.tensor(
                [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
                dtype=torch.float32,
                device=self.device,
            )
            mix_idxs = torch.multinomial(prob, num_samples=self.cfg.train.batch_size, replacement=True).reshape(-1, 1)

            # zs obtained by encoding train goals
            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            train_goal = tree_map(lambda x: x[perm], train_goal)
            goals = self._model._backward_map(train_goal)
            goals = self._model.project_z(goals)
            z = torch.where(mix_idxs == 0, goals, z)

            # zs obtained by encoding expert trajectories
            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    @torch.no_grad()
    def encode_expert(
        self,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
    ):
        # encode expert trajectories through B
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            B_expert = self._model._backward_map(next_obs).detach()  # batch x d
            B_expert = B_expert.view(
                self.cfg.train.batch_size // self.cfg.model.seq_length,
                self.cfg.model.seq_length,
                B_expert.shape[-1],
            )  # N x L x d
            z_expert = B_expert.mean(dim=1)  # N x d
            z_expert = self._model.project_z(z_expert)
            z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)  # batch x d
        return z_expert

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs = (
            tree_map(lambda x: x.to(self.device), train_batch["observation"]),
            train_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
        )
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs, expert_next_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        # train the discriminator
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_next_obs=train_next_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(
            self.update_actor(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor | dict[str, torch.Tensor],
        real_z: torch.Tensor,
        fake_obs: torch.Tensor | dict[str, torch.Tensor],
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tree_get_batch_size(real_obs)
        alpha = torch.rand(batch_size, 1, device=real_z.device)

        # TODO does not work for nested dicts
        interpolated_obs = {}
        interpolated_obs_list = []
        if isinstance(real_obs, torch.Tensor):
            if real_obs.shape != fake_obs.shape:
                raise ValueError(f"Shape mismatch: {real_obs.shape} vs {fake_obs.shape}")
            interpolated_obs = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
            interpolated_obs_list.append(interpolated_obs)
        else:
            for key in real_obs.keys():
                real_obs_tensor = real_obs[key]
                fake_obs_tensor = fake_obs[key]
                if isinstance(real_obs_tensor, torch.Tensor):
                    interpolated_obs[key] = (alpha * real_obs_tensor + (1 - alpha) * fake_obs_tensor).requires_grad_(True)
                    interpolated_obs_list.append(interpolated_obs[key])
                else:
                    raise ValueError(f"Unsupported type for key {key}: {type(real_obs_tensor)}")

        interpolated_z = alpha * real_z + (1 - alpha) * fake_z
        interpolated_z = interpolated_z.requires_grad_(True)

        d_interpolates = self._model._discriminator.compute_logits(interpolated_obs, interpolated_z)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated_obs_list + [interpolated_z],
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            # obs might contain entries that are not used by discriminator, in which case autograd.grad would complain about unused variables
            # We need to explicitely allow unused variables
            # TODO maybe remove these variables from the inputs completely instead? (we could read self._model._discriminator.filtered_space.keys, but this creates a dependency on the discriminator)
            allow_unused=True,
        )

        # Filter out None's from gradients: if any input is not used by the discriminator, autograd.grad will return None for its gradient
        gradients = [g for g in gradients if g is not None]
        cat_gradients = torch.cat(gradients, dim=1)
        gradient_penalty = ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Example of code with grad_penalty_obs_weight
        # obs_dim = real_obs.shape[-1]
        # grad_obs = gradients[:, :obs_dim]
        # grad_z = gradients[:, obs_dim:]
        # w_obs = self.cfg.train.grad_penalty_obs_weight
        # w_z = 1 - w_obs
        # gradient_penalty_obs = ((grad_obs.norm(2, dim=1) - w_obs) ** 2).mean()
        # gradient_penalty_z = ((grad_z.norm(2, dim=1) - w_z) ** 2).mean()
        # gradient_penalty = 0.5 * (gradient_penalty_obs + gradient_penalty_z)
        return gradient_penalty

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_next_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
            unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
            # these are equivalent to binary cross entropy
            expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
            unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
            loss = torch.mean(expert_loss + unlabeled_loss)

            if grad_penalty is not None:
                wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
                loss += grad_penalty * wgan_gp

            # [BFM-COND-FIX] conditional ranking loss: teach D that a window's
            # observation pairs with its OWN z, not with a z from the opposite
            # activity half of the batch. z is window-level (repeat_interleaved
            # over seq_length rows); activity read from the normalized dof_vel
            # block of state (batch-relative ranking only).
            cond_loss = None
            if self.cfg.train.cond_loss_coeff > 0:
                seq = self.cfg.model.seq_length
                n_win = expert_z.shape[0] // seq
                if n_win >= 4:
                    win_act = bdx_motion_activity(expert_obs["state"]).view(n_win, seq).mean(-1)
                    med = win_act.median()
                    hi = (win_act > med).nonzero(as_tuple=False).flatten()
                    lo = (win_act <= med).nonzero(as_tuple=False).flatten()
                    if len(hi) >= 2 and len(lo) >= 2:
                        n_pairs = max(1, int(self.cfg.train.cond_pair_frac * n_win))
                        sel_hi = hi[torch.randint(len(hi), (n_pairs,), device=hi.device)]
                        sel_lo = lo[torch.randint(len(lo), (n_pairs,), device=lo.device)]
                        ar = torch.arange(seq, device=hi.device)
                        rows_hi = (sel_hi[:, None] * seq + ar[None, :]).flatten()
                        rows_lo = (sel_lo[:, None] * seq + ar[None, :]).flatten()
                        # z is constant within a window: any row of a sampled opposite-half window
                        neg_lo_rows = lo[torch.randint(len(lo), (rows_hi.shape[0],), device=lo.device)] * seq
                        neg_hi_rows = hi[torch.randint(len(hi), (rows_lo.shape[0],), device=hi.device)] * seq
                        z_lo = expert_z[neg_lo_rows]
                        z_hi = expert_z[neg_hi_rows]
                        obs_hi = {k: v[rows_hi] for k, v in expert_obs.items()}
                        obs_lo = {k: v[rows_lo] for k, v in expert_obs.items()}
                        pos_hi = self._model._discriminator.compute_logits(obs=obs_hi, z=expert_z[rows_hi])
                        neg_hi_l = self._model._discriminator.compute_logits(obs=obs_hi, z=z_lo)
                        pos_lo = self._model._discriminator.compute_logits(obs=obs_lo, z=expert_z[rows_lo])
                        neg_lo_l = self._model._discriminator.compute_logits(obs=obs_lo, z=z_hi)
                        m = self.cfg.train.cond_margin
                        cond_hi = torch.nn.functional.softplus(m - pos_hi + neg_hi_l)
                        cond_lo = torch.nn.functional.softplus(m - pos_lo + neg_lo_l)
                        cond_loss = (cond_hi.mean() + cond_lo.mean()) * 0.5
                        loss = loss + self.cfg.train.cond_loss_coeff * cond_loss

            # [BFM-POLICY-NEG-COND] Policy observations ALWAYS remain fake in
            # BCE.  Infer what each policy window actually did with frozen B,
            # then only lower an incompatible commanded pairing relative to a
            # detached achieved-latent reference.  No policy reference logit
            # receives an upward gradient.
            policy_cond_loss = None
            policy_quality_loss = None
            policy_cond_coverage = expert_logits.new_zeros(())
            policy_quality_coverage = expert_logits.new_zeros(())
            policy_cond_margin_value = expert_logits.new_zeros(())
            if self.cfg.train.policy_cond_coeff > 0 or self.cfg.train.policy_quality_coeff > 0:
                seq = self.cfg.model.seq_length
                n_win = train_z.shape[0] // seq
                if n_win >= 4:
                    with torch.no_grad():
                        achieved_b = self._model._backward_map(train_next_obs).detach()
                        achieved_z = self._model.project_z(
                            achieved_b.view(n_win, seq, -1).mean(dim=1)
                        )
                        commanded_z = train_z.view(n_win, seq, -1)[:, 0].detach()
                        expert_win_z = expert_z.view(n_win, seq, -1)[:, 0].detach()
                        achieved_n = torch.nn.functional.normalize(achieved_z.float(), dim=-1)
                        commanded_n = torch.nn.functional.normalize(commanded_z.float(), dim=-1)
                        expert_n = torch.nn.functional.normalize(expert_win_z.float(), dim=-1)
                        manifold_sim, nearest_expert = (achieved_n @ expert_n.T).max(dim=1)
                        command_sim = (achieved_n * commanded_n).sum(dim=-1)
                        manifold_w = ((manifold_sim - self.cfg.train.policy_manifold_cos_min)
                                      / max(1e-6, 1.0 - self.cfg.train.policy_manifold_cos_min)).clamp(0, 1)
                        mismatch_w = ((self.cfg.train.policy_mismatch_cos_max - command_sim)
                                      / max(1e-6, 1.0 + self.cfg.train.policy_mismatch_cos_max)).clamp(0, 1)
                        state_windows = train_obs["state"].view(n_win, seq, -1)
                        finite = torch.isfinite(state_windows).all(dim=(1, 2))
                        stable = bdx_motion_activity(state_windows).amax(dim=1) \
                            < self.cfg.train.policy_max_abs_dof_vel
                        confidence = manifold_w * finite.float() * stable.float()
                        cond_weight = confidence * mismatch_w

                    rows = torch.arange(seq, device=train_z.device)[None, :] \
                        + torch.arange(n_win, device=train_z.device)[:, None] * seq
                    rows_flat = rows.flatten()
                    z_achieved_rows = achieved_z[:, None, :].expand(-1, seq, -1).reshape(-1, achieved_z.shape[-1])
                    commanded_logits = self._model._discriminator.compute_logits(train_obs, train_z).view(n_win, seq).mean(1)
                    with torch.no_grad():
                        achieved_logits = self._model._discriminator.compute_logits(
                            train_obs, z_achieved_rows
                        ).view(n_win, seq).mean(1)
                    active_cond = cond_weight > 0
                    if active_cond.any() and self.cfg.train.policy_cond_coeff > 0:
                        per_window = torch.nn.functional.softplus(
                            self.cfg.train.policy_cond_margin
                            + commanded_logits - achieved_logits.detach()
                        )
                        policy_cond_loss = (per_window * cond_weight).sum() / cond_weight.sum().clamp_min(1e-6)
                        loss = loss + self.cfg.train.policy_cond_coeff * policy_cond_loss
                        policy_cond_coverage = active_cond.float().mean()
                        policy_cond_margin_value = (
                            achieved_logits.detach()[active_cond] - commanded_logits.detach()[active_cond]
                        ).mean()

                    if self.cfg.train.policy_quality_coeff > 0:
                        with torch.no_grad():
                            # KNN activity decoder on this expert batch. It is
                            # used only as a local quality proxy/confidence aid,
                            # never as an expert-positive label.
                            expert_activity = bdx_motion_activity(expert_obs["state"]) \
                                .mean(-1).view(n_win, seq).mean(-1)
                            target_activity = expert_activity[nearest_expert]
                            actual_activity = bdx_motion_activity(state_windows).mean(dim=1)
                            activity_scale = expert_activity.std().clamp_min(0.1)
                            alignment = command_sim - 0.25 * (actual_activity - target_activity).abs() / activity_scale
                            command_pair_sim = commanded_n @ commanded_n.T
                            command_pair_sim.fill_diagonal_(-2.0)
                            nearest_command_sim, partner = command_pair_sim.max(dim=1)
                            partner_alignment = alignment[partner]
                            i_is_worse = alignment < partner_alignment
                            bad = torch.where(i_is_worse, torch.arange(n_win, device=train_z.device), partner)
                            good = torch.where(i_is_worse, partner, torch.arange(n_win, device=train_z.device))
                            quality_gap = (alignment[good] - alignment[bad]).clamp_min(0)
                            quality_weight = confidence[bad] * confidence[good]
                            quality_weight *= ((nearest_command_sim - self.cfg.train.policy_quality_command_cos_min)
                                               / max(1e-6, 1.0 - self.cfg.train.policy_quality_command_cos_min)).clamp(0, 1)
                            quality_weight *= (quality_gap >= self.cfg.train.policy_quality_gap_min).float()
                        active_quality = quality_weight > 0
                        if active_quality.any():
                            bad_logits = commanded_logits[bad]
                            with torch.no_grad():
                                good_logits = commanded_logits[good].detach()
                            q_per = torch.nn.functional.softplus(
                                self.cfg.train.policy_cond_margin + bad_logits - good_logits
                            )
                            policy_quality_loss = (q_per * quality_weight).sum() / quality_weight.sum().clamp_min(1e-6)
                            loss = loss + self.cfg.train.policy_quality_coeff * policy_quality_loss
                            policy_quality_coverage = active_quality.float().mean()

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        disc_grad_norm = grad_norm(self._model._discriminator.parameters()).detach()  # [BFM-DIAG-NAN] added by Claude
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean().detach(),
                "disc_train_loss": unlabeled_loss.detach().mean().detach(),
                "disc_grad_norm": disc_grad_norm,  # [BFM-DIAG-NAN] added by Claude
                "disc_ok": torch.isfinite(loss).float(),  # [BFM-DIAG-NAN]
            }
            if grad_penalty is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()
            if cond_loss is not None:
                output_metrics["disc_cond_loss"] = cond_loss.detach()
            if self.cfg.train.policy_cond_coeff > 0:
                output_metrics["disc_policy_cond_loss"] = (
                    policy_cond_loss.detach() if policy_cond_loss is not None else expert_logits.new_zeros(())
                )
            if self.cfg.train.policy_quality_coeff > 0:
                output_metrics["disc_policy_quality_loss"] = (
                    policy_quality_loss.detach() if policy_quality_loss is not None else expert_logits.new_zeros(())
                )
            output_metrics["diag/policy_cond_coverage"] = policy_cond_coverage.detach()
            output_metrics["diag/policy_quality_coverage"] = policy_quality_coverage.detach()
            output_metrics["diag/policy_cond_margin"] = policy_cond_margin_value.detach()
        return output_metrics

    def update_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            # compute target critic
            with torch.no_grad():
                reward = self._model._discriminator.compute_reward(obs=obs, z=z)
                if self.cfg.train.reward_norm:
                    reward = (reward - reward.mean()) / (reward.std() + 1e-5)
                if self.cfg.train.reward_clip is not None:
                    reward = torch.clamp(reward, -self.cfg.train.reward_clip, self.cfg.train.reward_clip)
                # [BFM-DIAG] distribution of the training (clipped) imitation reward:
                # a stable mean can still hide bounded saturation (all mass at the
                # clip), which would leave the behavior gradient dead.
                _reward_stats = None
                if getattr(self.cfg.train, "diag_reward_stats", False):
                    _r = reward.flatten().float()
                    _qs = torch.quantile(_r, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=_r.device))
                    _reward_stats = {
                        "diag/reward_std": _r.std(),
                        "diag/reward_p05": _qs[0], "diag/reward_p25": _qs[1],
                        "diag/reward_p50": _qs[2], "diag/reward_p75": _qs[3],
                        "diag/reward_p95": _qs[4],
                    }
                    if self.cfg.train.reward_clip is not None:
                        _reward_stats["diag/reward_frac_at_lower_clip"] = (_r <= -self.cfg.train.reward_clip).float().mean()
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_critic(next_obs, z, next_action)  # num_parallel x batch x 1
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.critic_pessimism_penalty)
                target_Q = reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            Qs = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = grad_norm(self._model._critic.parameters()).detach()  # [BFM-DIAG-NAN] added by Claude
        self.critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_Q": target_Q.mean().detach(),
                "Q1": Qs.mean().detach(),
                "mean_next_Q": Q_mean.mean().detach(),
                "unc_Q": Q_unc.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_disc_reward": reward.mean().detach(),
                "critic_grad_norm": critic_grad_norm,  # [BFM-DIAG-NAN] added by Claude
                "critic_ok": torch.isfinite(critic_loss).float(),  # [BFM-DIAG-NAN]
            }
            if _reward_stats is not None:
                output_metrics.update(_reward_stats)
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)

            # compute discriminator reward loss
            Qs_discriminator = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute fb reward loss
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
            Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            actor_loss = -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight - Q_fb.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
            }
        return output_metrics
