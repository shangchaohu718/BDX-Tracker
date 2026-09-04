# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from typing import Dict

import pydantic
import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..fb_cpr.agent import FBcprAgent, FBcprAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode, grad_norm  # [BFM-DIAG-NAN] grad_norm added by Claude
from .model import FBcprAuxModelConfig


class FBcprAuxAgentTrainConfig(FBcprAgentTrainConfig):
    lr_aux_critic: float = 1e-4
    reg_coeff_aux: float = 1.0
    aux_critic_pessimism_penalty: float = 0.5
    # Causal-ablation knob for the actor's FB term only (1.0 = unchanged
    # baseline objective). F/B/critic still train normally at 0.0.
    actor_fb_coeff: float = 1.0
    # Distribution stats of the training (clipped) imitation reward.
    diag_reward_stats: bool = False
    # Motion-activity proxies (is the robot actually moving?): mean |action|
    # (raw) and mean |dof_vel| (from the normalized state block).
    diag_motion_activity: bool = False
    # Expensive, observe-only diagnostics for short causal ablations.
    diag_reward_sources: bool = False
    diag_actor_update_ratio: bool = False


class FBcprAuxAgentConfig(BaseConfig):
    name: tp.Literal["FBcprAuxAgent"] = "FBcprAuxAgent"

    model: FBcprAuxModelConfig = FBcprAuxModelConfig()
    train: FBcprAuxAgentTrainConfig = FBcprAuxAgentTrainConfig()
    aux_rewards: list[str] = pydantic.Field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = pydantic.Field(default_factory=dict)
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim: int) -> "FBcprAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        return FBcprAuxAgent


class FBcprAuxAgent(FBcprAgent):
    config_class = FBcprAuxAgentConfig

    def setup_training(self) -> None:
        super().setup_training()

        # prepare parameter list
        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(),
            lr=self.cfg.train.lr_aux_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer.state_dict()
        return optimizers

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_aux_critic = torch.compile(self.update_aux_critic, mode=mode)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_aux_critic = CudaGraphModule(self.update_aux_critic, warmup=5)

    def _diag_observe_batch_finiteness(self, step, named_batches):
        # [BFM-DIAG-NAN] -- instrumentation added by Claude: OBSERVE-ONLY sampled-batch guard.
        # Logs the FIRST batch fed to update() that contains NaN/inf (the intermittent MuJoCo-Warp
        # sim NaN that is the root cause of training divergence), but does NOT skip the update --
        # the update proceeds unmodified, so training behavior is unchanged. One host-sync when the
        # batch is finite; per-key counts only printed when something is actually non-finite.
        counts = {}

        def _scan(tree, prefix):
            if not isinstance(tree, dict):
                return
            for k, v in tree.items():
                key = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict):
                    _scan(v, key)
                elif torch.is_tensor(v) and v.numel():
                    counts[key] = (torch.isnan(v) | torch.isinf(v)).sum()

        for _name, _tree in named_batches:
            _scan(_tree, _name)
        if not counts:
            return
        total = sum(counts.values()).item()
        if total == 0:
            return
        bad = {k: int(v.item()) for k, v in counts.items() if v.item() > 0}
        self._diag_batch_nf_count = getattr(self, "_diag_batch_nf_count", 0) + 1
        if self._diag_batch_nf_count == 1:
            print(f"[BFM-DIAG-NAN] FIRST non-finite SAMPLED BATCH at step={step}: "
                  f"per-key non-finite counts={bad}", flush=True)
        elif self._diag_batch_nf_count % 50 == 0:
            print(f"[BFM-DIAG-NAN] {self._diag_batch_nf_count} non-finite sampled batches so far "
                  f"(last at step={step})", flush=True)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        # [BFM-DIAG-NAN] observe-only: log the first non-finite sampled batch (no skip / no halt)
        self._diag_observe_batch_finiteness(step, (("train", train_batch), ("expert", expert_batch)))

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
        rollout_z = train_z

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
        if self.cfg.train.diag_motion_activity:
            # state layout: [base_ang_vel(3), projected_gravity(3), dof_pos(14), dof_vel(14)]
            metrics.update({
                "diag/mean_abs_action": train_action.detach().abs().mean(),
                "diag/mean_abs_dof_vel_norm": bdx_motion_activity(train_obs["state"]).detach().abs().mean(),
            })

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        if self.cfg.train.diag_reward_sources:
            # Limit diagnostics to a fixed prefix so four extra D forwards do
            # not dominate short-training throughput.
            m = min(256, self.cfg.train.batch_size)
            with torch.no_grad():
                diag_obs = tree_map(lambda x: x[:m], train_obs)
                diag_next_obs = tree_map(lambda x: x[:m], train_next_obs)
                policy_goal_z = self._model.project_z(self._model._backward_map(diag_next_obs))
                # Preserve the training RNG stream: enabling diagnostics must
                # not alter later minibatches, actor noise, or relabel masks.
                if torch.device(self.device).type == "cuda":
                    rng_state = torch.cuda.get_rng_state(self.device)
                    random_z = self._model.sample_z(m, device=self.device)
                    torch.cuda.set_rng_state(rng_state, self.device)
                else:
                    rng_state = torch.get_rng_state()
                    random_z = self._model.sample_z(m, device=self.device)
                    torch.set_rng_state(rng_state)
                metrics.update(
                    {
                        "diag/reward_rollout_z": self._model._discriminator.compute_reward(diag_obs, rollout_z[:m]).mean(),
                        "diag/reward_used_z": self._model._discriminator.compute_reward(diag_obs, train_z[:m]).mean(),
                        "diag/reward_policy_goal_z": self._model._discriminator.compute_reward(diag_obs, policy_goal_z).mean(),
                        "diag/reward_expert_z": self._model._discriminator.compute_reward(diag_obs, expert_z[:m]).mean(),
                        "diag/reward_random_z": self._model._discriminator.compute_reward(diag_obs, random_z).mean(),
                        "diag/cos_used_rollout_z": torch.nn.functional.cosine_similarity(
                            train_z, rollout_z, dim=-1
                        ).mean(),
                        "diag/relabel_fraction": (train_z != rollout_z).any(dim=-1).float().mean(),
                    }
                )

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
        # compute scalar auxiliary reward as a weighted sum of the auxiliary rewards
        aux_reward = torch.zeros(
            (self.cfg.train.batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        )
        for aux_reward_name in self.cfg.aux_rewards:
            # let's log even this information
            metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(self.device)

        aux_reward = self._model._aux_reward_normalizer(aux_reward)

        metrics.update(
            self.update_aux_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                aux_reward=aux_reward,
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
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.critic.num_parallel
            # compute target critic
            with torch.no_grad():
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_aux_critic(next_obs, z, next_action)  # num_parallel x batch x 1
                # TODO AL: should we have aux_critic parameters here?
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.aux_critic_pessimism_penalty)
                target_Q = aux_reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            Qs = self._model._aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        aux_critic_grad_norm = grad_norm(self._model._aux_critic.parameters()).detach()  # [BFM-DIAG-NAN] added by Claude
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_Q.mean().detach(),
                "auxQ1": Qs.mean().detach(),
                "mean_next_auxQ": Q_mean.mean().detach(),
                "unc_auxQ": Q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
                "aux_critic_grad_norm": aux_critic_grad_norm,  # [BFM-DIAG-NAN] added by Claude
                "aux_critic_ok": torch.isfinite(aux_critic_loss).float(),  # [BFM-DIAG-NAN]
            }
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

            # compute auxiliary reward loss
            Qs_aux = self._model._aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute fb reward loss
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
            Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            # Metrics are reduced by the trainer as tensors.  Keep the disabled
            # ablation on-device too, instead of leaking a Python float into the
            # metrics dictionary.
            weight = (
                Q_fb.abs().mean().detach()
                if self.cfg.train.scale_reg
                else Q_fb.new_ones(())
            )
            actor_loss_disc = -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight
            actor_loss_aux = -Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
            actor_loss_fb = -self.cfg.train.actor_fb_coeff * Q_fb.mean()
            actor_loss = actor_loss_disc + actor_loss_aux + actor_loss_fb

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = grad_norm(self._model._actor.parameters()).detach()  # [BFM-DIAG-NAN] added by Claude
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        actor_grad_norm_post_clip = grad_norm(self._model._actor.parameters()).detach()
        if self.cfg.train.diag_actor_update_ratio:
            actor_params_before = [p.detach().clone() for p in self._model._actor.parameters()]
            actor_param_norm = torch.sqrt(
                sum((p.detach().float() ** 2).sum() for p in self._model._actor.parameters())
            )
            with torch.no_grad():
                mean_action_before = self._model._actor(obs, z, self._model.cfg.actor_std).mean
                mean_Qs_fb_before = (
                    self._model._forward_map(obs, z, mean_action_before) * z
                ).sum(-1)
                _, _, mean_Q_fb_before = self.get_targets_uncertainty(
                    mean_Qs_fb_before, self.cfg.train.actor_pessimism_penalty
                )
        self.actor_optimizer.step()
        if self.cfg.train.diag_actor_update_ratio:
            actor_delta_norm = torch.sqrt(
                sum(
                    ((p.detach().float() - old.float()) ** 2).sum()
                    for p, old in zip(self._model._actor.parameters(), actor_params_before)
                )
            )
            actor_update_ratio = actor_delta_norm / actor_param_norm.clamp_min(1e-12)
            with torch.no_grad():
                mean_action_after = self._model._actor(obs, z, self._model.cfg.actor_std).mean
                mean_Qs_fb_after = (
                    self._model._forward_map(obs, z, mean_action_after) * z
                ).sum(-1)
                _, _, mean_Q_fb_after = self.get_targets_uncertainty(
                    mean_Qs_fb_after, self.cfg.train.actor_pessimism_penalty
                )
                actor_mean_action_delta = (mean_action_after - mean_action_before).abs().mean()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "actor_loss_disc_component": actor_loss_disc.detach(),
                "actor_loss_aux_component": actor_loss_aux.detach(),
                "actor_loss_fb_component": actor_loss_fb.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
                # [BFM-DIAG-NAN] -- instrumentation added by Claude: scale_reg weight, |Q_fb|, actor grad norm, finite flag
                "scale_reg_weight": weight,
                "Q_fb_abs": Q_fb.abs().mean().detach(),
                "actor_grad_norm": actor_grad_norm,
                "actor_grad_norm_post_clip": actor_grad_norm_post_clip,
                "effective_disc_coeff": self.cfg.train.reg_coeff * weight,
                "actor_ok": torch.isfinite(actor_loss).float(),
            }
            if self.cfg.train.diag_actor_update_ratio:
                m = min(256, Q_fb.shape[0])
                diag_obs = tree_map(lambda x: x[:m], obs)
                diag_reward = self._model._discriminator.compute_reward(diag_obs, z[:m]).flatten()
                diag_q_fb = Q_fb[:m].flatten()
                q_centered = diag_q_fb - diag_q_fb.mean()
                r_centered = diag_reward - diag_reward.mean()
                qfb_reward_corr = (q_centered * r_centered).mean() / (
                    q_centered.square().mean().sqrt()
                    * r_centered.square().mean().sqrt()
                    + 1e-8
                )
                q10, q90 = torch.quantile(diag_q_fb, torch.tensor([0.1, 0.9], device=diag_q_fb.device))
                reward_at_qfb_bottom10 = diag_reward[diag_q_fb <= q10].mean()
                reward_at_qfb_top10 = diag_reward[diag_q_fb >= q90].mean()
                output_metrics["actor_param_update_ratio"] = actor_update_ratio.detach()
                output_metrics["diag/Q_fb_actor_mean_before"] = mean_Q_fb_before.mean().detach()
                output_metrics["diag/Q_fb_actor_mean_after"] = mean_Q_fb_after.mean().detach()
                output_metrics["diag/Q_fb_actor_mean_delta"] = (
                    mean_Q_fb_after.mean() - mean_Q_fb_before.mean()
                ).detach()
                output_metrics["diag/actor_mean_action_delta"] = actor_mean_action_delta.detach()
                output_metrics["diag/Q_fb_reward_corr"] = qfb_reward_corr.detach()
                output_metrics["diag/reward_at_Q_fb_bottom10"] = reward_at_qfb_bottom10.detach()
                output_metrics["diag/reward_at_Q_fb_top10"] = reward_at_qfb_top10.detach()
                output_metrics["diag/reward_Q_fb_top_minus_bottom10"] = (
                    reward_at_qfb_top10 - reward_at_qfb_bottom10
                ).detach()
        return output_metrics
