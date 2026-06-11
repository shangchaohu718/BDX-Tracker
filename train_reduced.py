"""BFM-Zero full training with reduced actor model for RTX 5080 (16GB VRAM).

Changes from default config:
- Model hidden_dim: 2048 → 1024, hidden_layers: 6 → 4 (all networks)
- AMP enabled (bfloat16)
- buffer_device: cpu
- online_parallel_envs: 256 (fits in 16GB with reduced model)
- batch_size: 512
"""
import sys
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, ".")

from humanoidverse.train import TrainConfig, Workspace
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from humanoidverse.agents.nn_models import ForwardArchiConfig, BackwardArchiConfig, ActorArchiConfig, DiscriminatorArchiConfig, RewardNormalizerConfig
from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.evaluations.humanoidverse_isaac import HumanoidVerseIsaacTrackingEvaluationConfig

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
                f=ForwardArchiConfig(name='ForwardArchi', hidden_dim=1024, model="residual", hidden_layers=4, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                b=BackwardArchiConfig(name='BackwardArchi', hidden_dim=256, hidden_layers=1, norm=True, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                actor=ActorArchiConfig(name='actor', model='residual', hidden_dim=1024, hidden_layers=4, embedding_layers=2, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'last_action', 'history_actor'])),
                critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=1024, model="residual", hidden_layers=4, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                discriminator=DiscriminatorArchiConfig(name='DiscriminatorArchi', hidden_dim=1024, hidden_layers=3, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                aux_critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=1024, model="residual", hidden_layers=4, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor']))
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
            amp=False,
            norm_aux_reward=RewardNormalizerConfig(name='RewardNormalizer', translate=False, scale=True)
        ),
        train=FBcprAuxAgentTrainConfig(
            name='FBcprAuxAgentTrainConfig',
            lr_f=0.0003,
            lr_b=1e-05,
            lr_actor=0.0003,
            weight_decay=0.0,
            clip_grad_norm=0.0,
            fb_target_tau=0.01,
            ortho_coef=100.0,
            train_goal_ratio=0.2,
            fb_pessimism_penalty=0.0,
            actor_pessimism_penalty=0.5,
            stddev_clip=0.3,
            q_loss_coef=0.0,
            batch_size=512,
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
        compile=False,
    ),
    motions='',
    motions_root='',
    env=HumanoidVerseIsaacConfig(
        name='humanoidverse_isaac',
        device='cuda:0',
        lafan_tail_path='humanoidverse/data/lafan_29dof_10s-clipped.pkl',
        enable_cameras=False,
        camera_render_save_dir='isaac_videos',
        max_episode_length_s=None,
        disable_obs_noise=False,
        disable_domain_randomization=False,
        relative_config_path='exp/bfm_zero/bfm_zero',
        include_last_action=True,
        hydra_overrides=['robot=g1/g1_29dof_hard_waist', 'robot.control.action_scale=0.25', 'robot.control.action_clip_value=5.0', 'robot.control.normalize_action_to=5.0', 'env.config.lie_down_init=True', 'env.config.lie_down_init_prob=0.3'],
        context_length=None,
        include_dr_info=False,
        included_dr_obs_names=None,
        include_history_actor=True,
        include_history_noaction=False,
        make_config_g1env_compatible=False,
        root_height_obs=True
    ),
    work_dir='results/bfmzero-reduced-rtx5080',
    seed=4728,
    online_parallel_envs=256,
    num_env_steps=50_000_000,
    update_agent_every=1024,
    num_seed_steps=10240,
    num_agent_updates=16,
    checkpoint_every_steps=5_000_000,
    checkpoint_buffer=True,
    prioritization=False,
    use_trajectory_buffer=True,
    buffer_size=2_000_000,
    use_wandb=False,
    load_isaac_expert_data=True,
    buffer_device='cpu',
    disable_tqdm=False,
    evaluations=[],
    eval_every_steps=10_000_000,
    log_every_updates=100_000,
    tags={'reduced_model': True, 'rtx5080': True},
)

print("=" * 60)
print("BFM-Zero Training (Reduced Model for RTX 5080)")
print(f"  parallel_envs:  {cfg.online_parallel_envs}")
print(f"  num_env_steps:  {cfg.num_env_steps:,}")
print(f"  batch_size:     {cfg.agent.train.batch_size}")
print(f"  buffer_device:  {cfg.buffer_device}")
print(f"  amp:            {cfg.agent.model.amp}")
print(f"  hidden_dim:     1024 (reduced from 2048)")
print(f"  hidden_layers:  4 (reduced from 6)")
print(f"  work_dir:       {cfg.work_dir}")
print("=" * 60)

workspace = cfg.build()
workspace.train()