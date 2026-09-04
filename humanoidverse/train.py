# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

from humanoidverse.agents.evaluations.humanoidverse_isaac import (
    HumanoidVerseIsaacTrackingEvaluation,
    HumanoidVerseIsaacTrackingEvaluationConfig,
)
from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib, HumanoidVerseIsaacConfig

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")

import json
import time
import typing as tp
import warnings
from pathlib import Path
from typing import Dict, List
from torch.utils._pytree import tree_map

import exca as xk
import gymnasium
import numpy as np
import pydantic
import torch  # better to use scoped import if we use processes
import tyro
import wandb
from packaging.version import Version
from torch.utils._pytree import tree_map
from tqdm import tqdm


from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.load_data import load_expert_trajectories
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.fb_cpr.agent import FBcprAgentConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"

CHECKPOINT_DIR_NAME = "checkpoint"

_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER = {
    HumanoidVerseIsaacConfig: None,
}



Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseIsaacTrackingEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

Agent = FBcprAgentConfig | FBcprAuxAgentConfig


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None

    env: HumanoidVerseIsaacConfig = pydantic.Field(discriminator="name")

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    seed: int = 0
    online_parallel_envs: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    log_every_updates: int = 100_000
    num_env_steps: int = 30_000_000
    # Note: this is in env steps (multiples of online_parallel_envs)
    update_agent_every: int = 500
    # Note: this is in env steps (multiples of online_parallel_envs)
    num_seed_steps: int = 50_000
    num_agent_updates: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    checkpoint_every_steps: int = 5_000_000
    checkpoint_buffer: bool = True
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"  # ["bin", "exp", "lin"]
    padding_beginning: int = 0
    padding_end: int = 0

    # Buffer
    use_trajectory_buffer: bool = False
    buffer_size: int = 5_000_000

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None

    # misc
    load_isaac_expert_data: bool = True
    buffer_device: str = "cpu"
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True

    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    # Note: this is in env steps (multiples of online_parallel_envs)
    eval_every_steps: int = 1_000_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    # exca
    infra: xk.TaskInfra = xk.TaskInfra(version="1")

    def model_post_init(self, context):
        # TODO prioritization needs tracking eval to work, but this is bit hacky to check for it
        if self.load_isaac_expert_data and not isinstance(self.env, HumanoidVerseIsaacConfig):
            raise ValueError("Loading expert isaac data is only supported for HumanoidVerseIsaacConfig")

        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, (HumanoidVerseIsaacTrackingEvaluationConfig)):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")


        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
            elif self.agent == FBcprAgentConfig:
                # TODO how to do checks like these in pydantic or more systematically?
                raise ValueError("FBcprAgent requires expert data to be provided (motions and motions_root)")

        # Ensure all evaluations have unique log names
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        """In case of cluster run, use exca and process instead of explivit build"""
        return Workspace(self)


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    if checkpoint_dir.exists():
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]

        print(f"Loading the agent at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_time


def init_wandb(cfg: TrainConfig):
    # Offline mode: no internet on remote. Logs locally to ./_wandb/,
    # sync later with `wandb sync _wandb/wandb/offline-run-*` when online.
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", "./_wandb")
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    os.makedirs("./_wandb", exist_ok=True)
    exp_name = "BFM-Zero"
    wandb_name = exp_name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=wandb_name, config=wandb_config, dir="./_wandb")


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        # HACK with Isaac, we can not recreate environments with current code, so we need to
        #      create the environment with desired number of envs here
        if isinstance(cfg.env, HumanoidVerseIsaacConfig):
            from omegaconf import OmegaConf

            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
        else:
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space

        assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        # TODO for backwards consistency, we do not pass "time" to the agent, so we remove it from the obs_space we pass to the agent/model
        #      but would we need it at some point?
        del self.obs_space.spaces["time"]

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if isinstance(cfg.env, HumanoidVerseIsaacConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()}

        if self.cfg.use_wandb:
            init_wandb(self.cfg)

        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

        self.priorization_eval_name = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseIsaacTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        self.training_with_expert_data = True

        self.manager = None

    def train(self):
        self.start_time = time.time()
        self.train_online()

    def _diag_buffer_finiteness(self, storage, prefix: str = "") -> str:
        # [BFM-DIAG-SNAPSHOT] -- instrumentation added by Claude
        # Scan every leaf tensor in a buffer storage tree and report NaN/inf counts.
        lines = []
        totals = {"nan": 0, "inf": 0, "elem": 0}

        def _scan(d, pfx):
            for k, v in d.items():
                key = f"{pfx}/{k}" if pfx else k
                if isinstance(v, dict):
                    _scan(v, key)
                elif torch.is_tensor(v):
                    vt = v.float()
                    n_nan = int(torch.isnan(vt).sum().item())
                    n_inf = int(torch.isinf(vt).sum().item())
                    totals["nan"] += n_nan
                    totals["inf"] += n_inf
                    totals["elem"] += vt.numel()
                    flag = "OK " if (n_nan == 0 and n_inf == 0) else "BAD"
                    has_fin = torch.isfinite(vt).any().item()
                    _mn = float(vt[torch.isfinite(vt)].min().item()) if has_fin else float("nan")
                    _mx = float(vt[torch.isfinite(vt)].max().item()) if has_fin else float("nan")
                    lines.append(f"  [{flag}] {key:38s} shape={str(tuple(v.shape)):20s} NaN={n_nan:>8d} inf={n_inf:>8d} min={_mn:+.4e} max={_mx:+.4e}")

        _scan(storage, prefix)
        header = f"  total elements={totals['elem']}, NaN={totals['nan']}, inf={totals['inf']}"
        return header + "\n" + "\n".join(lines)

    def _diag_do_snapshot(self, t, replay_buffer, label: str):
        # [BFM-DIAG-SNAPSHOT] -- instrumentation added by Claude
        # Save the full pre-update state (train buffer + one fixed expert batch + agent
        # incl. normalizer + RNG) to work_dir/debug_snapshot/, print a finiteness summary, exit.
        import pickle
        import random
        import sys
        snap_dir = self.work_dir / "debug_snapshot"
        snap_dir.mkdir(parents=True, exist_ok=True)
        replay_buffer["train"].save(snap_dir / "train_buffer")  # re-sampleable offline
        with torch.no_grad():  # expert buffer is not serializable -> save one fixed batch
            _expert_batch = replay_buffer["expert_slicer"].sample(self.agent.cfg.train.batch_size)
            _expert_batch = tree_map(lambda x: x.detach().cpu(), _expert_batch)
        torch.save(_expert_batch, snap_dir / "expert_batch.pt")
        self.agent.save(str(snap_dir / "agent"))  # model + optimizers + config + normalizer (pre-update, at init)
        _rng = {
            "t": int(t), "label": label,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "train_buffer_len": len(replay_buffer["train"]),
            "buffer_device": self.cfg.buffer_device,
            "batch_size": self.agent.cfg.train.batch_size,
        }
        with open(snap_dir / "rng_states.pkl", "wb") as _f:
            pickle.dump(_rng, _f)
        _nf = self._diag_buffer_finiteness(replay_buffer["train"].storage)
        print(f"[BFM-DIAG-SNAPSHOT] ({label}) saved pre-update state to {snap_dir} (t={t}, "
              f"train_rows={len(replay_buffer['train'])}, expert_batch={self.agent.cfg.train.batch_size}).", flush=True)
        print(f"[BFM-DIAG-SNAPSHOT] ({label}) raw buffer finiteness:\n{_nf}", flush=True)
        self._diag_snapshot_done = True
        print(f"[BFM-DIAG-SNAPSHOT] ({label}) exiting after snapshot capture.", flush=True)
        sys.exit(0)

    def train_online(self) -> None:
        expert_pool_payload = None
        if self.training_with_expert_data:
            # [BFM-DOMAIN-FIX] optional expert-sim pool: expert positives for the
            # discriminator generated through the simulator observation pipeline
            # (state injection) instead of the kinematic motion-library pipeline.
            # Root cause context: BFM_ZERO_DEBUG_HANDOFF — discriminator
            # domain-separation. Unset -> unchanged kinematic expert data.
            pool_path = os.environ.get("BFM_EXPERT_SIM_POOL")
            if pool_path:
                from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer

                payload = torch.load(pool_path, map_location=self.cfg.buffer_device)
                expert_pool_payload = payload
                expert_buffer = TrajectoryDictBuffer(
                    episodes=payload["episodes"],
                    seq_length=self.agent.cfg.model.seq_length,
                    device=self.cfg.buffer_device,
                )
                print(
                    f"[BFM-EXPERT-SIM-POOL] loaded {len(expert_buffer)} frames / "
                    f"{len(payload['episodes'])} episodes from {pool_path}",
                    flush=True,
                )
            elif self.cfg.load_isaac_expert_data:
                expert_buffer = load_expert_trajectories_from_motion_lib(self.train_env._env, self.cfg.agent, device=self.cfg.buffer_device)
            else:
                print("Loading expert trajectories")
                expert_buffer = load_expert_trajectories(
                    self.cfg.motions,
                    self.cfg.motions_root,
                    seq_length=self.agent.cfg.model.seq_length,
                    device=self.cfg.buffer_device,
                    # TODO data stored in disk does not have dictionary obs, so we need to manually
                    #      define what obs key the data on disk corresponds to
                    obs_dict_mapper=_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER[self.cfg.env.__class__],
                )
        print("Creating the training environment")

        if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)

        # [BFM-REF-Z-ALIGN] The environment already starts each episode from a
        # motion-library state.  Reference-state exploration is only
        # conditionally meaningful when the rollout latent describes that same
        # motion window.  Share the already-loaded pool (no second 471 MB load)
        # with the reset sampler; it records the selected episode/offset so the
        # rollout context can be encoded from the identical window below.
        if (
            expert_pool_payload is not None
            and os.environ.get("BFM_REF_Z_ALIGNED", "0") == "1"
            and hasattr(train_env, "_env")
        ):
            train_env._env._bfm_ref_pool_payload = expert_pool_payload
            print(
                "[BFM-REF-Z-ALIGN] reference reset state and rollout z use the same expert-sim pool window",
                flush=True,
            )

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        if (checkpoint_dir / "buffers/train").exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load(checkpoint_dir / "buffers/train", device=self.cfg.buffer_device)
            else:
                replay_buffer["train"] = DictBuffer.load(checkpoint_dir / "buffers/train", device=self.cfg.buffer_device)
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        else:
            if self.cfg.use_trajectory_buffer:
                output_key_t = ["observation", "action", "z", "terminated", "truncated", "step_count", "reward"]
                # TODO this interface should be more elegant (how to inform buffer what keys are coming in / need to be sampled?)
                if isinstance(self.cfg.agent, (FBcprAuxAgentConfig)):
                    output_key_t.append("aux_rewards")

                replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="truncated",
                    output_key_t=output_key_t,  # TODO(team): fix this. in principle we could avoid to sample qpos, qvel for training but we need them for reward evaluation
                    output_key_tp1=["observation", "terminated"],
                )
            else:
                replay_buffer["train"] = DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer

        # [BFM-COND-FIX] optional fresh-D conditional pretraining bootstrap:
        # load only the discriminator weights (actor/critic/F/B stay fresh).
        pretrained_d = os.environ.get("BDX_PRETRAINED_D")
        if pretrained_d:
            payload_d = torch.load(pretrained_d, map_location="cuda")
            self.agent._model._discriminator.load_state_dict(payload_d["discriminator"])
            print(f"[BFM-PRETRAINED-D] loaded discriminator from {pretrained_d} "
                  f"(stats: {payload_d.get('stats', {}).get('final_margin')})", flush=True)

        print("Starting training")
        progb = tqdm(total=self.cfg.num_env_steps, disable=self.cfg.disable_tqdm)
        td, info = train_env.reset()
        # see https://farama.org/Vector-Autoreset-Mode
        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        total_metrics, context = None, None
        start_time = time.time()
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)
        # [BFM-OPS] resume-without-buffer warmup. When continuing from a model checkpoint that did NOT
        # save its replay buffer (checkpoint_buffer=False), the trajectory buffer starts empty and
        # sample() raises until it holds trajectories >= seq_length. So we push the no-update seed gate
        # out to num_seed_steps PAST the resume point: policy rollouts (we have a trained policy, since
        # t >= num_seed_steps) refill the buffer, then updates resume. No-op for fresh runs
        # (_checkpoint_time == 0) and for resumes that DID load a buffer -- fully backward compatible.
        _buffer_was_loaded = (checkpoint_dir / "buffers" / "train").exists()
        _seed_gate = (
            self._checkpoint_time + self.cfg.num_seed_steps
            if (self._checkpoint_time > 0 and not _buffer_was_loaded)
            else self.cfg.num_seed_steps
        )
        if _seed_gate != self.cfg.num_seed_steps:
            print(
                f"[BFM-OPS] bufferless resume: refilling replay buffer with policy rollouts, updates gated "
                f"until t > {_seed_gate} (resume_point={self._checkpoint_time}, +{self.cfg.num_seed_steps} warmup steps).",
                flush=True,
            )

        eval_instances = []
        for evaluation_name in self.evaluations.keys():
            evaluation = self.evaluations[evaluation_name]
            eval_instances.append(isinstance(evaluation, HumanoidVerseIsaacTrackingEvaluation))
        uses_humanoidverse_eval = True if any(eval_instances) else False

        for t in range(self._checkpoint_time, self.cfg.num_env_steps + self.cfg.online_parallel_envs, self.cfg.online_parallel_envs):
            if (t != self._checkpoint_time) and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer)

            if (self.evaluate and eval_time_checker.check(t)) or (self.evaluate and t == self._checkpoint_time):
                eval_metrics = self.eval(t, replay_buffer=replay_buffer)
                eval_time_checker.update_last_step(t)
                if uses_humanoidverse_eval:
                    # reset if there is a humanoidverse evaluation
                    td, info = train_env.reset()
                    terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)

                if self.cfg.prioritization:
                    assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids), (
                        "Mismatch in number of motions returned by the eval"
                    )
                    # priorities
                    index_in_buffer, name_in_buffer = {}, {}
                    for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                        index_in_buffer[motion_id] = i
                        if hasattr(replay_buffer["expert_slicer"], "file_names"):
                            name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]
                    motions_id, priorities, idxs = [], [], []
                    for _, metr in eval_metrics[self.priorization_eval_name].items():
                        motions_id.append(metr["motion_id"])
                        priorities.append(metr["emd"])
                        idxs.append(index_in_buffer[metr["motion_id"]])
                    priorities = (
                        torch.clamp(
                            torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                            min=self.cfg.prioritization_min_val,
                            max=self.cfg.prioritization_max_val,
                        )
                        * self.cfg.prioritization_scale
                    )

                    if self.cfg.prioritization_mode == "lin":
                        pass
                    elif self.cfg.prioritization_mode == "exp":
                        priorities = 2**priorities
                    elif self.cfg.prioritization_mode == "bin":
                        bins = torch.floor(priorities)
                        for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                            mask = bins == i
                            n = mask.sum().item()
                            if n > 0:
                                priorities[mask] = 1 / n
                    else:
                        raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")

                    train_env._env._motion_lib.update_sampling_weight_by_id(
                        priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer
                    )

                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=priorities.to(self.cfg.buffer_device), idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
                    )

            with torch.no_grad():
                obs = tree_map(lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device), td)
                # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
                step_count = obs.pop("time")

                history_context = None
                if "history" in obs:
                    # this works in inference mode
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(self.cfg.online_parallel_envs)
                    else:
                        history_context = self.agent.history_inference(obs=obs["history"]["observation"], action=obs["history"]["action"])[
                            :, -1
                        ].clone()

                previous_context = context
                context = self.agent.maybe_update_rollout_context(z=context, step_count=step_count, replay_buffer=replay_buffer)
                # Reset rows selected by the reference curriculum carry an
                # episode/offset chosen by LeggedRobotMotions.  Encode exactly
                # that 8-frame pool window, overriding the independently sampled
                # rollout z only for those rows.  This removes the previous
                # (state_from_motion_A, z_from_motion_B) reset mismatch.
                inner_env = getattr(train_env, "_env", None)
                aligned_mask = getattr(inner_env, "_bfm_ref_aligned_mask", None)
                pool_rows = getattr(inner_env, "_bfm_ref_pool_rows", None)
                pool_offsets = getattr(inner_env, "_bfm_ref_pool_offsets", None)
                reset_mask = step_count.reshape(-1) == 0
                if aligned_mask is not None and pool_rows is not None and pool_offsets is not None:
                    use_mask = reset_mask & aligned_mask.to(reset_mask.device)
                    # A reference-aligned episode represents one conditional
                    # task.  Do not let update_z_every_step silently replace
                    # that task every 100 steps; hold z until the next reset.
                    # This was the remaining long-horizon mismatch: at horizon
                    # 500, reset alignment survived only the first 100 steps.
                    keep_mask = (~reset_mask) & aligned_mask.to(reset_mask.device)
                    if previous_context is not None and keep_mask.any():
                        context[keep_mask] = previous_context.to(context.device)[keep_mask]
                    use_ids = use_mask.nonzero(as_tuple=False).flatten()
                    if use_ids.numel() > 0:
                        seq = self.agent.cfg.model.seq_length
                        windows = []
                        for env_i in use_ids.detach().cpu().tolist():
                            ep_i = int(pool_rows[env_i].item())
                            off_i = int(pool_offsets[env_i].item())
                            ep_obs = expert_pool_payload["episodes"][ep_i]["observation"]
                            windows.append({
                                key: value[off_i : off_i + seq]
                                for key, value in ep_obs.items()
                            })
                        stacked = {
                            key: torch.cat([window[key] for window in windows], dim=0).to(self.agent.device)
                            for key in windows[0]
                        }
                        encoded = self.agent._model.backward_map(stacked)
                        encoded = encoded.view(len(windows), seq, -1).mean(dim=1)
                        context[use_ids] = self.agent._model.project_z(encoded)
                if t < self.cfg.num_seed_steps:
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # this works in inference mode
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    # TODO a bit hard-coded -- just to avoid moving stuff from cpu to cuda
                    if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                        action = action.cpu().detach().numpy()
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)

            # we check if at the next iteration we will evaluate
            next_t = t + self.cfg.online_parallel_envs
            if (self.evaluate and eval_time_checker.check(next_t)) or (self.evaluate and next_t == self._checkpoint_time):
                if isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and uses_humanoidverse_eval:
                    # make sure we set truncated since at the next iteration we are forced to reset the environment
                    # after the evaluation. This is because we share the environment with the evaluation
                    new_truncated = np.ones_like(new_truncated, dtype=bool)
                    truncated = np.ones_like(new_truncated, dtype=bool)

            if Version(gymnasium.__version__) >= Version("1.0"):
                if self.cfg.use_trajectory_buffer:
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": terminated[None, ..., None],
                        "truncated": truncated[None, ..., None],
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[None, ...]
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
                else:
                    # We add only transitions corresponding to environments that have not reset in the previous step.
                    # For environments that have reset in the previous step, the new observation corresponds to the state after reset.
                    indexes = ~done

                    real_next_obs = tree_map(lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes], new_td)
                    # TODO again, we need to remove "time" from the observation (to stay consistent with obs_space)
                    _ = real_next_obs.pop("time")
                    _ = real_next_obs.pop("history", None)

                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": real_next_obs,
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[indexes]
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = new_info["qpos"][indexes]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = new_info["qvel"][indexes]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1) for k, v in new_info["aux_rewards"].items() if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            replay_buffer["train"].extend(data)

            # [BFM-DIAG-SNAPSHOT] -- instrumentation added by Claude
            # Capture point A (default): END of the random-action seed phase (pure random rollouts).
            # Active only when BFM_ZERO_SNAPSHOT_AT=seed.
            if (
                os.environ.get("BFM_ZERO_SNAPSHOT", "0") == "1"
                and os.environ.get("BFM_ZERO_SNAPSHOT_AT", "seed") == "seed"
                and not getattr(self, "_diag_snapshot_done", False)
                and t < self.cfg.num_seed_steps
                and (t + self.cfg.online_parallel_envs) >= self.cfg.num_seed_steps
            ):
                self._diag_do_snapshot(t, replay_buffer, label="end-of-seed (pure random)")

            if len(replay_buffer["train"]) > 0 and t > _seed_gate and update_agent_time_checker.check(t):
                update_agent_time_checker.update_last_step(t)
                # [BFM-DIAG-SNAPSHOT] capture point B: right BEFORE the FIRST update (seed + policy data).
                # Active only when BFM_ZERO_SNAPSHOT_AT=first_update. Reproduces the state that NaN'd.
                if (
                    os.environ.get("BFM_ZERO_SNAPSHOT", "0") == "1"
                    and os.environ.get("BFM_ZERO_SNAPSHOT_AT", "seed") == "first_update"
                    and not getattr(self, "_diag_snapshot_done", False)
                ):
                    self._diag_do_snapshot(t, replay_buffer, label="first-update (seed + policy)")
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(replay_buffer, t)
                    # [BFM-DIAG-NAN] -- instrumentation added by Claude: observe-only, print the FIRST non-finite update's breakdown once (no halt)
                    if not getattr(self, "_diag_nan_seen", False):
                        _nf = {}
                        for _k, _v in metrics.items():
                            _m = (_v if torch.is_tensor(_v) else torch.as_tensor(_v)).float().mean()
                            if not torch.isfinite(_m):
                                _nf[_k] = _m.item()
                        if _nf:
                            _full = {k: (v if torch.is_tensor(v) else torch.as_tensor(v)).float().mean().item() for k, v in metrics.items()}
                            print(f"[BFM-DIAG-NAN] FIRST non-finite at step t={t}, update {_ + 1}/{self.cfg.num_agent_updates}. "
                                  f"Non-finite ({len(_nf)}): {_nf}", flush=True)
                            print(f"[BFM-DIAG-NAN] full update metrics: " + ", ".join(f"{k}={round(v, 4)}" for k, v in sorted(_full.items())), flush=True)
                            self._diag_nan_seen = True
                    if total_metrics is None:
                        num_metrics_updates = 1
                        total_metrics = {k: metrics[k].float().clone() for k in metrics.keys()}
                    else:
                        num_metrics_updates += 1
                        total_metrics = {k: total_metrics[k] + metrics[k].float() for k in metrics.keys()}

            if log_time_checker.check(t) and total_metrics is not None:
                log_time_checker.update_last_step(t)
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / num_metrics_updates
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
                m_dict["timestep"] = t
                self.train_logger.log(m_dict)

            progb.update(self.cfg.online_parallel_envs)
            td = new_td
            terminated = new_terminated
            truncated = new_truncated
            done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info
        train_env.close()

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations.keys():
            logger = self.eval_loggers[evaluation_name]
            evaluation = self.evaluations[evaluation_name]

            # NOTE we have this inside the loop so that the agent is not moved to cpu if we don't evaluate
            if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                self.agent._model.to("cpu")
            self.agent._model.train(False)

            if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                # Pass train env
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t, agent_or_model=self.agent, replay_buffer=replay_buffer, logger=logger, env=self.train_env
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )
            # For wandb dict, put it on wandb
            if self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()

        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        print(f"Checkpointing at time {time}")
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))
        if self.cfg.checkpoint_buffer:
            replay_buffer["train"].save(self.work_dir / CHECKPOINT_DIR_NAME / "buffers" / "train")
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time}, f, indent=4)
        # [Gate1-OPS] non-rolling snapshot retention: copy the checkpoint to
        # work_dir/ckpt_<time> so within-run time series (fixed-panel probes)
        # survive the rolling overwrite. Env-gated, model+status only (no buffer).
        # Saves land on step-grid offsets (e.g. 25088), so key on the snapshot
        # INDEX (time // every) rather than exact divisibility.
        snap_every = int(os.environ.get("BFM_SNAPSHOT_EVERY", "0"))
        if snap_every > 0:
            snap_idx = time // snap_every
            # always advance the index (even if the copy already exists) so
            # resume runs don't retry the same slot forever
            should_snap = snap_idx > getattr(self, "_last_snap_idx", -1)
            self._last_snap_idx = max(snap_idx, getattr(self, "_last_snap_idx", -1))
            if should_snap:
                import shutil
                snap = self.work_dir / f"ckpt_{time}"
                if not snap.exists():
                    tmp = self.work_dir / f"ckpt_{time}.tmp"
                    shutil.copytree(
                        self.work_dir / CHECKPOINT_DIR_NAME, tmp,
                        ignore=shutil.ignore_patterns("buffers"),
                    )
                    (tmp / "READY").write_text(str(time))
                    tmp.rename(snap)  # atomic: probes only read ckpt_* dirs (completed)
                    print(f"[Gate1-OPS] retained snapshot {snap}", flush=True)


def train_bfm_zero(profile: str = "h20"):
    """Launch BFM-Zero training.

    Two hardware profiles are available:
      - "h20"    : full paper network for H20-3e 143 GB data-center GPU (production)
      - "rtx5080": small network for 16 GB consumer GPU (local dev/debug)

    Each neural net (forward, backward, actor, critic, discriminator, aux_critic)
    has its own hidden_dim / hidden_layers so they can be tuned independently.
    """
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentTrainConfig
    from humanoidverse.agents.nn_models import ForwardArchiConfig, BackwardArchiConfig, ActorArchiConfig, DiscriminatorArchiConfig, RewardNormalizerConfig
    from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig
    from humanoidverse.agents.nn_filters import DictInputFilterConfig

    # ------------------------------------------------------------------ #
    #  Per-network architecture + env/buffer scale for each profile.
    #  Learning rates, reward weights, and all algorithm hyperparams are
    #  identical (paper values) across profiles.
    # ------------------------------------------------------------------ #
    PROFILES = {
        "h20": {
            # network sizes (paper values)
            "f_dim": 2048, "f_layers": 6,            # forward map
            "b_dim": 256, "b_layers": 1,             # backward map (z head)
            "actor_dim": 2048, "actor_layers": 6,
            "critic_dim": 2048, "critic_layers": 6,
            "disc_dim": 1024, "disc_layers": 3,      # discriminator
            "aux_dim": 2048, "aux_layers": 6,        # aux critic
            # env / buffer scale
            "work_dir": "results/bfmzero-isaac-diag",
            "batch_size": 8192,
            "online_parallel_envs": 8192,
            "eval_num_envs": 1024,
            "buffer_size": 5_120_000,
            "buffer_device": "cuda",
        },
        "rtx5080": {
            # ~1/4 width, shallower → fits 16 GB
            "f_dim": 512, "f_layers": 3,
            "b_dim": 256, "b_layers": 1,
            "actor_dim": 512, "actor_layers": 3,
            "critic_dim": 512, "critic_layers": 3,
            "disc_dim": 512, "disc_layers": 2,
            "aux_dim": 512, "aux_layers": 3,
            "work_dir": "results/bfmzero-isaac-rtx5080",
            "batch_size": 512,
            "online_parallel_envs": 256,
            "eval_num_envs": 256,
            "buffer_size": 1_024_000,
            "buffer_device": "cpu",   # 16 GB: keep replay buffer on host
        },
    }
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose from {list(PROFILES)}")
    p = PROFILES[profile]
    # [BFM-DIAG-SNAPSHOT] -- instrumentation added by Claude
    # When BFM_ZERO_SNAPSHOT=1: skip eval/prioritization, run the random-action seed
    # phase (num_seed_steps = 10*N_env), capture the random-rollout buffer + agent at
    # the end of seeding, then exit -- so the first agent.update() can be debugged
    # offline without re-running the simulator. Unset = normal training (unchanged).
    _snapshot_mode = os.environ.get("BFM_ZERO_SNAPSHOT", "0") == "1"
    if _snapshot_mode:
        print("[BFM-DIAG-SNAPSHOT] SNAPSHOT MODE: will capture the random-rollout buffer and exit before the first update.")
    # [BFM-DIAG-SNAPSHOT] RESUME MODE: continue training from a captured snapshot's buffer+agent
    # (assembled into work_dir/checkpoint/ -- see the snapshot->checkpoint assembly below),
    # WITHOUT re-running the slow seed phase and WITHOUT eval/prioritization. BFM_ZERO_SNAPSHOT
    # is unset so the capture block never fires; training simply resumes and proceeds.
    _resume_mode = os.environ.get("BFM_ZERO_RESUME", "0") == "1"
    _diag_mode = _snapshot_mode or _resume_mode
    if _resume_mode:
        print("[BFM-DIAG-SNAPSHOT] RESUME MODE: will load the snapshot checkpoint (buffer+agent) and continue training.")
    # [BFM-DIAG-SNAPSHOT] use a dedicated work_dir so snapshot/resume runs are isolated from the
    # main training dir. Override with BFM_ZERO_SNAPSHOT_DIR / BFM_ZERO_WORK_DIR respectively.
    _work_dir = p["work_dir"]
    if _snapshot_mode:
        _work_dir = os.environ.get("BFM_ZERO_SNAPSHOT_DIR", _work_dir + "-snapshot")
    elif _resume_mode:
        _work_dir = os.environ.get("BFM_ZERO_WORK_DIR", _work_dir + "-resume")

    # [BFM-DIAG-SNAPSHOT] evaluations disabled in snapshot/resume mode (also disables prioritization)
    _evaluations = [] if _diag_mode else [
        HumanoidVerseIsaacTrackingEvaluationConfig(
            name='HumanoidVerseIsaacTrackingEvaluationConfig', generate_videos=False, videos_dir='videos',
            video_name_prefix='unknown_agent', name_in_logs='humanoidverse_tracking_eval', env=None,
            num_envs=p["eval_num_envs"], n_episodes_per_motion=1,
        )
    ]

    cfg = TrainConfig(
        name='TrainConfig',
        agent=FBcprAuxAgentConfig(
            name='FBcprAuxAgent',
            model=FBcprAuxModelConfig(
                name='FBcprAuxModel',
                device='cuda',
                archi=FBcprAuxModelArchiConfig(
                    name='FBcprAuxModelArchiConfig',
                    z_dim=256,
                    norm_z=True,
                    f=ForwardArchiConfig(name='ForwardArchi', hidden_dim=p["f_dim"], model='residual', hidden_layers=p["f_layers"], embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                    b=BackwardArchiConfig(name='BackwardArchi', hidden_dim=p["b_dim"], hidden_layers=p["b_layers"], norm=True, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    actor=ActorArchiConfig(name='actor', model='residual', hidden_dim=p["actor_dim"], hidden_layers=p["actor_layers"], embedding_layers=2, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'last_action', 'history_actor'])),
                    critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=p["critic_dim"], model='residual', hidden_layers=p["critic_layers"], embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                    discriminator=DiscriminatorArchiConfig(name='DiscriminatorArchi', hidden_dim=p["disc_dim"], hidden_layers=p["disc_layers"], input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    aux_critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=p["aux_dim"], model='residual', hidden_layers=p["aux_layers"], embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor']))
                ),
                obs_normalizer=ObsNormalizerConfig(
                    name='ObsNormalizerConfig',
                    normalizers={
                        'state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'privileged_state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'last_action': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'history_actor': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01)
                    },
                    allow_mismatching_keys=True
                ),
                inference_batch_size=500000,
                seq_length=8,
                actor_std=0.05,
                amp=True,  # [BFM-DIAG-SNAPSHOT] enable PyTorch bfloat16 AMP (~98% of compute is PyTorch; also cuts VRAM)
                norm_aux_reward=RewardNormalizerConfig(name='RewardNormalizer', translate=False, scale=True)
            ),
            train=FBcprAuxAgentTrainConfig(
                name='FBcprAuxAgentTrainConfig',
                lr_f=0.0003,
                lr_b=1e-05,
                lr_actor=0.0003,
                weight_decay=0.0,
                clip_grad_norm=0.0,   # paper default (disabled)
                fb_target_tau=0.01,
                ortho_coef=100.0,
                train_goal_ratio=0.2,
                fb_pessimism_penalty=0.0,
                actor_pessimism_penalty=0.5,
                stddev_clip=0.3,
                q_loss_coef=0.0,
                batch_size=p["batch_size"],
                discount=0.98,
                use_mix_rollout=True,
                update_z_every_step=100,
                z_buffer_size=8192,
                rollout_expert_trajectories=True,
                rollout_expert_trajectories_length=250,
                rollout_expert_trajectories_percentage=0.5,
                lr_discriminator=1e-05,
                lr_critic=0.0003,
                critic_target_tau=0.005,
                critic_pessimism_penalty=0.5,
                reg_coeff=0.05,
                scale_reg=True,
                expert_asm_ratio=0.6,
                relabel_ratio=0.8,
                grad_penalty_discriminator=10.0,
                weight_decay_discriminator=0.0,
                lr_aux_critic=0.0003,
                reg_coeff_aux=0.02,
                aux_critic_pessimism_penalty=0.5
            ),
            aux_rewards=['penalty_torques', 'penalty_action_rate', 'limits_dof_pos', 'limits_torque', 'penalty_undesired_contact', 'penalty_feet_ori', 'penalty_ankle_roll', 'penalty_slippage'],
            aux_rewards_scaling={'penalty_action_rate': -0.1, 'penalty_feet_ori': -0.4, 'penalty_ankle_roll': -4.0, 'limits_dof_pos': -10.0, 'penalty_slippage': -2.0, 'penalty_undesired_contact': -1.0, 'penalty_torques': 0.0, 'limits_torque': 0.0},
            cudagraphs=False,
            compile=False
        ),
        motions='',
        motions_root='',
        env=HumanoidVerseIsaacConfig(
            name='humanoidverse_isaac',
            device='cuda:0',
            # TODO this needs to be updated to point to a path with lafan dataset chunked into 10s clips
            lafan_tail_path='humanoidverse/data/lafan_29dof_10s-clipped.pkl',
            enable_cameras=False,
            camera_render_save_dir='isaac_videos',
            max_episode_length_s=None,
            disable_obs_noise=False,
            disable_domain_randomization=False,
            relative_config_path='exp/bfm_zero/bfm_zero',
            include_last_action=True,
            # [BFM-DIAG-NAN] lie_down_init_prob is env-var overridable for the bad-init-state A/B
            # confirmation run: default 0.3 (baseline); set BFM_ZERO_LIE_DOWN_PROB=0.0 on the remote
            # to spawn all envs upright (no lying-down penetration) and see if the sim NaN vanishes.
            hydra_overrides=['simulator=mujoco_warp', 'robot=g1/g1_29dof_hard_waist', 'robot.control.action_scale=0.25', 'robot.control.action_clip_value=5.0', 'robot.control.normalize_action_to=5.0', 'env.config.lie_down_init=True', f'env.config.lie_down_init_prob={os.environ.get("BFM_ZERO_LIE_DOWN_PROB", "0.3")}'],
            context_length=None,
            include_dr_info=False,
            included_dr_obs_names=None,
            include_history_actor=True,
            include_history_noaction=False,
            make_config_g1env_compatible=False,
            root_height_obs=True
        ),
        work_dir=_work_dir,
        seed=4728,
        online_parallel_envs=p["online_parallel_envs"],
        log_every_updates=8192,
        num_env_steps=384000000,
        update_agent_every=1024,
        num_seed_steps=p["online_parallel_envs"] * 10,  # [BFM-DIAG-SNAPSHOT] 10 random-action seed iterations, scaled with N_env
        num_agent_updates=16,
        # [BFM-OPS] checkpoint every 2M steps (~50 min at ~9.6M-steps/4h throughput) so saves land WELL
        # inside the 4h wall-clock cutoff on this shared host. The old 9.6M value ~= 4h collided with the
        # cutoff and produced a half-written checkpoint (model+buffer written, train_status.json never
        # reached) on every run -- structurally unable to persist progress past the resume point.
        checkpoint_every_steps=2000000,
        # [BFM-OPS] do NOT checkpoint the replay buffer: it is ~13G (5.12M transitions) and on this 94G
        # shared volume it filled the disk and corrupted a half-written save. Model + optimizer state (the
        # learned part) are still checkpointed and resumed; the replay buffer simply re-seeds on each resume
        # (num_seed_steps of random actions), negligible cost over a 384M-step run.
        checkpoint_buffer=False,
        prioritization=(not _diag_mode),  # [BFM-DIAG-SNAPSHOT] off in snapshot/resume mode (no eval)
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode='exp',
        use_trajectory_buffer=True,
        buffer_size=p["buffer_size"],
        use_wandb=True,
        wandb_ename='yitangl',  # your wandb entity (username/team), empty = default from wandb login
        wandb_gname='bfmzero-isaac',  # run group
        wandb_pname='bfmzero-isaac',  # your wandb project name
        load_isaac_expert_data=True,
        buffer_device=p["buffer_device"],
        disable_tqdm=True,
        evaluations=_evaluations,
        eval_every_steps=9600000,
        tags={"profile": profile},
    )
    workspace = cfg.build()
    workspace.train()


if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    #
    # Profile selection via env var so it works under nohup without argparse:
    #   BFM_ZERO_PROFILE=rtx5080 python -m humanoidverse.train   (local dev)
    #   python -m humanoidverse.train                            (defaults to h20)
    profile = os.environ.get("BFM_ZERO_PROFILE", "h20")
    train_bfm_zero(profile=profile)

# uv run --no-cache -m humanoidverse.meta_online_entry_point
