"""BFM-Zero training launcher for the BDX 14-DOF robot on a 16 GB RTX 5080.

Mirrors the hyperparameters of the bfmzero-bdx-100M checkpoint (z_dim 128,
discount 0.95, lr's, aux rewards, trajectory buffer) at a resource scale that
fits 16 GB VRAM + 31 GB host RAM:

  - network: same 512-wide / 3-layer residual stacks as the checkpoint
  - batch 512, 256 parallel envs (mujoco_warp), replay buffer 768k on CPU
  - expert data: stratified subset of the dataset pkl (see convert_dataset_to_pkl)

Modes (env var BDX_MODE):
  bench : 200k env steps, no eval / no wandb -> measures FPS for ETA
  full  : 10M env steps + tracking eval every 2.5M, wandb offline
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
# [BDX-FIX] disable the G1-era hard-joint-limit reset ([BFM-DIAG-NAN] Fix #1). BDX's narrow
# joint ranges (hip_yaw +-0.35 rad) vs the action target range (+-1.25 rad) made it fire on
# ~75% of envs EVERY step -> 1-3-step episodes -> no tracking learned (validated: with it on,
# avg episode length 1-3 steps; with it off, episodes run continuously). NaN protection on
# mujoco_warp is separately covered by the BFM_ZERO_NAN_RESET isfinite guard.
os.environ.setdefault("BFM_ZERO_HARD_LIMIT_RESET", "0")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")  # PyTorch 2.9 CPU Inductor miscompiles trajectory index sampling
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce VRAM fragmentation

from pathlib import Path

import torch

torch.set_float32_matmul_precision("high")

from humanoidverse.agents.evaluations.humanoidverse_isaac import (
    HumanoidVerseIsaacTrackingEvaluation,
    HumanoidVerseIsaacTrackingEvaluationConfig,
)
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.train import TrainConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from humanoidverse.agents.nn_models import ForwardArchiConfig, BackwardArchiConfig, ActorArchiConfig, DiscriminatorArchiConfig, RewardNormalizerConfig
from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig

MODE = os.environ.get("BDX_MODE", "bench")
DATA = os.environ.get("BDX_DATA", str(Path(__file__).parents[1] / "data" / "bdx_14dof_dataset.pkl"))
NUM_ENV_STEPS = 200_000 if MODE == "bench" else int(os.environ.get("BDX_STEPS", 10_000_000))
EVAL_EVERY = int(os.environ.get("BDX_EVAL_EVERY", 2_500_000))
N_ENVS = int(os.environ.get("BDX_ENVS", 256))
WORK_DIR = os.environ.get("BDX_WORK_DIR", f"results/bfmzero-bdx-{MODE}")
SCALE_REG = os.environ.get("BDX_SCALE_REG", "true").lower() in ("1", "true", "yes")
CLIP_GRAD_NORM = float(os.environ.get("BDX_CLIP_GRAD_NORM", "0"))
RELABEL_RATIO = float(os.environ.get("BDX_RELABEL_RATIO", "0.8"))
ACTOR_FB_COEFF = float(os.environ.get("BDX_ACTOR_FB_COEFF", "1.0"))
LR_ACTOR = float(os.environ.get("BDX_LR_ACTOR", "0.0003"))
LR_DISCRIMINATOR = float(os.environ.get("BDX_LR_DISCRIMINATOR", "1e-05"))
REWARD_CLIP = os.environ.get("BDX_REWARD_CLIP", "")
REWARD_CLIP = float(REWARD_CLIP) if REWARD_CLIP else None
REWARD_NORM = os.environ.get("BDX_REWARD_NORM", "0") == "1"
REG_COEFF_AUX = float(os.environ.get("BDX_REG_COEFF_AUX", "0.02"))
COND_LOSS = float(os.environ.get("BDX_COND_LOSS", "0"))
POLICY_COND_LOSS = float(os.environ.get("BDX_POLICY_COND_LOSS", "0"))
POLICY_QUALITY_LOSS = float(os.environ.get("BDX_POLICY_QUALITY_LOSS", "0"))
DIAG_DYNAMICS = os.environ.get("BDX_DIAG_DYNAMICS", "0") == "1"
DIAG_REWARD_STATS = os.environ.get("BDX_DIAG_REWARD_STATS", "0") == "1"
DIAG_MOTION = os.environ.get("BDX_DIAG_MOTION", "0") == "1"
DISCRIMINATOR_CONDITIONING = os.environ.get("BDX_D_CONDITIONING", "concat")
DISABLE_EVAL = os.environ.get("BDX_DISABLE_EVAL", "0") == "1"
REF_STATE_CURRICULUM = os.environ.get("BFM_REF_STATE_CURRICULUM", "0") == "1"

if MODE not in ("bench", "full"):
    raise ValueError(f"unknown BDX_MODE {MODE}")

BENCH = MODE == "bench"

cfg = TrainConfig(
    name="TrainConfig",
    agent=FBcprAuxAgentConfig(
        name="FBcprAuxAgent",
        model=FBcprAuxModelConfig(
            name="FBcprAuxModel",
            device="cuda",
            archi=FBcprAuxModelArchiConfig(
                name="FBcprAuxModelArchiConfig",
                z_dim=128,                       # checkpoint value
                norm_z=True,
                f=ForwardArchiConfig(name="ForwardArchi", hidden_dim=512, model="residual", hidden_layers=3, embedding_layers=2, num_parallel=2, ensemble_mode="batch", input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"])),
                b=BackwardArchiConfig(name="BackwardArchi", hidden_dim=128, hidden_layers=1, norm=True, input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"])),
                actor=ActorArchiConfig(name="actor", model="residual", hidden_dim=512, hidden_layers=3, embedding_layers=2, input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "last_action", "history_actor"])),
                critic=ForwardArchiConfig(name="ForwardArchi", hidden_dim=512, model="residual", hidden_layers=3, embedding_layers=2, num_parallel=2, ensemble_mode="batch", input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"])),
                discriminator=DiscriminatorArchiConfig(name="DiscriminatorArchiConfig", hidden_dim=512, hidden_layers=2, conditioning=DISCRIMINATOR_CONDITIONING, input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"])),
                aux_critic=ForwardArchiConfig(name="ForwardArchi", hidden_dim=512, model="residual", hidden_layers=3, embedding_layers=2, num_parallel=2, ensemble_mode="batch", input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"])),
            ),
            obs_normalizer=ObsNormalizerConfig(
                name="ObsNormalizerConfig",
                normalizers={
                    "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                },
                allow_mismatching_keys=True,
            ),
            inference_batch_size=100000,  # 16G guard: eval/backward-map batch cap (500k OOMs during eval with long episodes)
            seq_length=8,                     # checkpoint value
            actor_std=0.05,
            amp=True,
            norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
        ),
        train=FBcprAuxAgentTrainConfig(
            name="FBcprAuxAgentTrainConfig",
            lr_f=0.0003,
            lr_b=1e-05,
            lr_actor=LR_ACTOR,
            weight_decay=0.0,
            clip_grad_norm=CLIP_GRAD_NORM,
            fb_target_tau=0.01,
            ortho_coef=100.0,
            train_goal_ratio=0.2,
            fb_pessimism_penalty=0.0,
            actor_pessimism_penalty=0.5,
            stddev_clip=0.3,
            q_loss_coef=0.0,
            batch_size=int(os.environ.get("BDX_BATCH", 2048)),  # checkpoint used 4096; 2048 fits 16G (512 caused a ~716mm plateau)
            discount=0.95,                    # checkpoint value
            use_mix_rollout=True,
            update_z_every_step=100,
            z_buffer_size=8192,
            rollout_expert_trajectories=True,
            rollout_expert_trajectories_length=250,
            rollout_expert_trajectories_percentage=0.5,
            lr_discriminator=LR_DISCRIMINATOR,
            lr_critic=0.0003,
            critic_target_tau=0.005,
            critic_pessimism_penalty=0.5,
            reg_coeff=0.05,
            scale_reg=SCALE_REG,
            expert_asm_ratio=0.6,
            relabel_ratio=RELABEL_RATIO,
            actor_fb_coeff=ACTOR_FB_COEFF,
            reward_clip=REWARD_CLIP,
            reward_norm=REWARD_NORM,
            grad_penalty_discriminator=10.0,
            weight_decay_discriminator=0.0,
            cond_loss_coeff=COND_LOSS,
            policy_cond_coeff=POLICY_COND_LOSS,
            policy_quality_coeff=POLICY_QUALITY_LOSS,
            lr_aux_critic=0.0003,
            reg_coeff_aux=REG_COEFF_AUX,
            aux_critic_pessimism_penalty=0.5,
            diag_reward_sources=DIAG_DYNAMICS,
            diag_actor_update_ratio=DIAG_DYNAMICS,
            diag_reward_stats=DIAG_REWARD_STATS,
            diag_motion_activity=DIAG_MOTION,
        ),
        aux_rewards=["penalty_torques", "penalty_action_rate", "limits_dof_pos", "limits_torque", "penalty_undesired_contact", "penalty_feet_ori", "penalty_ankle_roll", "penalty_slippage"],
        aux_rewards_scaling={"penalty_action_rate": -0.1, "penalty_feet_ori": -0.4, "penalty_ankle_roll": 0.0, "limits_dof_pos": -10.0, "penalty_slippage": -2.0, "penalty_undesired_contact": -1.0, "penalty_torques": 0.0, "limits_torque": 0.0},  # ankle_roll 0.0: not applicable to BDX's single-DOF ankle (empty-slice dead code, see _reward_penalty_ankle_roll note)
        cudagraphs=False,
        compile=False,
    ),
    motions="",
    motions_root="",
    env=HumanoidVerseIsaacConfig(
        name="humanoidverse_isaac",
        device="cuda:0",
        lafan_tail_path=DATA,
        enable_cameras=False,
        camera_render_save_dir="isaac_videos",
        max_episode_length_s=None,
        disable_obs_noise=False,
        disable_domain_randomization=False,
        relative_config_path="exp/bfm_zero/bfm_zero",
        include_last_action=True,
        hydra_overrides=[                   # exact set from the bfmzero-bdx-100M checkpoint
            "simulator=mujoco_warp",
            "robot=bdx/bdx_14dof",
            "robot.control.action_scale=0.25",
            "robot.control.action_clip_value=5.0",
            "robot.control.normalize_action_to=5.0",
            # [BDX-FIX] contact-buffer overflow -> dropped rows -> penetration cascade -> NaN
            # (the mujoco_warp.yaml's own documented pathway). trim_collision is the repo's
            # fix #3 (disabled there only for eval fidelity); bigger nconmax/njmax is safe
            # at 512 worlds (the documented OOM was at 8192 worlds).
            "simulator.config.sim.trim_collision=true",
            "simulator.config.sim.nconmax=512",
            "simulator.config.sim.njmax=2048",
        ],
        context_length=None,
        include_dr_info=False,
        included_dr_obs_names=None,
        include_history_actor=True,
        include_history_noaction=False,
        make_config_g1env_compatible=False,
        root_height_obs=True,
    ),
    work_dir=WORK_DIR,
    seed=4728,
    online_parallel_envs=N_ENVS,
    log_every_updates=8192,
    num_env_steps=NUM_ENV_STEPS,
    update_agent_every=1024,
    num_seed_steps=N_ENVS * 10,
    num_agent_updates=16,
    checkpoint_every_steps=int(os.environ.get("BDX_CHECKPOINT_EVERY", "2000000")),
    checkpoint_buffer=False,              # disk guard: replay buffer is ~4 GB, never snapshot it
    prioritization=False,
    use_trajectory_buffer=True,
    buffer_size=768_000,                  # RAM guard: ~4 GB on host
    use_wandb=False,
    wandb_ename=None,
    wandb_gname=None,
    wandb_pname=None,
    load_isaac_expert_data=True,
    buffer_device="cpu",                  # RAM guard: replay buffer on host, VRAM stays for model+motion lib
    disable_tqdm=True,
    evaluations=[] if (BENCH or DISABLE_EVAL) else [
        HumanoidVerseIsaacTrackingEvaluationConfig(
            name="HumanoidVerseIsaacTrackingEvaluationConfig",
            generate_videos=False,
            videos_dir="videos",
            video_name_prefix="bdx_agent",
            name_in_logs="humanoidverse_tracking_eval",
            env=None,
            num_envs=64,
            n_episodes_per_motion=1,
        )
    ],
    eval_every_steps=EVAL_EVERY,
    tags={"mode": MODE, "robot": "bdx_14dof"},
)

if __name__ == "__main__":
    print(
        f"BFM-Zero BDX launcher: mode={MODE} steps={NUM_ENV_STEPS:,} data={DATA} "
        f"work_dir={WORK_DIR} scale_reg={SCALE_REG} clip={CLIP_GRAD_NORM} "
        f"relabel={RELABEL_RATIO} actor_fb={ACTOR_FB_COEFF} lr_actor={LR_ACTOR} lr_disc={LR_DISCRIMINATOR} "
        f"ref_curriculum={REF_STATE_CURRICULUM} diag={DIAG_DYNAMICS}",
        flush=True,
    )
    if not Path(DATA).exists():
        raise FileNotFoundError(f"motion pkl not found: {DATA} — run convert_dataset_to_pkl.py first")
    workspace = cfg.build()
    workspace.train()
