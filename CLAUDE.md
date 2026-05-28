# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BFM-Zero is a promptable behavioral foundation model for humanoid control using unsupervised reinforcement learning. A single trained policy can perform diverse behaviors by conditioning on different latent vectors (z). The target robot is the Unitree G1 (29 DOF).

## Commands

### Setup
```bash
uv sync                        # Install all dependencies (requires Python 3.10, CUDA GPU)
```

### Training
```bash
uv run python -m humanoidverse.train
```
Key: change `buffer_device` to `"cuda:0"` if you have enough VRAM. Target: eval/emd < 0.75 after 50-100M steps.

### Inference (all use tyro for CLI args)
```bash
uv run python -m humanoidverse.tracking_inference --help   # Motion tracking + ONNX export
uv run python -m humanoidverse.goal_inference --help        # Goal-reaching inference
uv run python -m humanoidverse.reward_inference --help      # Reward-based task inference
```
Common flags: `--model_folder`, `--simulator mujoco` (no Isaac Lab needed), `--no-headless`, `--save_mp4`.

### Linting
```bash
uv run ruff check humanoidverse/
uv run ruff format humanoidverse/
```
Config: line-length=140, ignores E402/E731, extends with import sorting (I).

## Architecture

### Config System

**Two config systems coexist:**

1. **Pydantic dataclasses** (primary for training/inference code) — `BaseConfig` in `agents/base.py` uses frozen Pydantic models with discriminated unions for polymorphism. All agent, model, and training configs use this pattern. The `name` field serves as the discriminator. Configs have `build()` methods that instantiate the corresponding class.

2. **Hydra YAML** (for environment/simulator setup) — configs in `config/` are organized by concern: `env/`, `robot/`, `simulator/`, `terrain/`, `rewards/`, `obs/`, `domain_rand/`, `exp/`. The experiment config `exp/bfm_zero/bfm_zero.yaml` composes them via Hydra defaults.

### Agent Hierarchy

```
BaseConfig (agents/base.py)
  └─ FB (agents/fb/)          — Forward-Backward base agent
       └─ FBcpr (agents/fb_cpr/)   — Adds contrastive predictive representation + discriminator
            └─ FBcprAux (agents/fb_cpr_aux/)  — Adds auxiliary critic for extra rewards (THE BFM-Zero agent)
```

Each level has an `agent.py` (training logic) and `model.py` (neural net architecture). Models follow the same inheritance chain. `FBcprAuxAgent` is the default agent used by `train.py`.

### Neural Network Components (agents/fb_cpr_aux/model.py)

The model contains:
- **Actor** — policy network conditioned on latent z
- **Critic / Target Critic** — Q-value estimation
- **Forward map** — predicts next state from (obs, z, action)
- **Backward map** — infers latent z from state transitions
- **Discriminator** — distinguishes expert vs learned trajectories
- **Aux Critic** — additional value function for auxiliary rewards (torque, action rate penalties)
- **Normalizers** — running statistics for observations and rewards

### Simulator Abstraction (simulator/)

`BaseSimulator` (abstract base) defines the interface: `setup()`, `load_assets()`, `create_envs()`, `apply_torques_at_dof()`, `simulate_at_each_physics_step()`, `render()`. Implementations: `isaacsim/`, `mujoco/`, `genesis/`, `isaacgym/`.

### Environment Layer (envs/)

```
BaseTask → LeggedRobotBase → LeggedRobotMotions (main training env)
```
- `legged_robot_motions.py` — loads motion library (LAfan dataset), handles resampling, termination, tracking evaluation
- `g1_env_helper/` — G1-specific rewards, collision, robot configuration (29 DOF)
- `gymnasium_wrapper.py` — Gymnasium compatibility for non-Isaac environments

### Training Flow (train.py)

`TrainConfig` → `Workspace.build()` → `Workspace.train()`:
1. Load expert trajectories from motion library (LAfan pkl files)
2. Fill replay buffer with seed data
3. Online training loop: collect rollouts → update agent every N steps → periodic evaluation → checkpoint
4. Optional prioritization based on tracking evaluation scores

### Data

- `humanoidverse/data/lafan_29dof.zip` — manually extract to `humanoidverse/data/` to get the two pkl files below:
  ```bash
  unzip humanoidverse/data/lafan_29dof.zip -d humanoidverse/data/
  ```
- `humanoidverse/data/lafan_29dof.pkl` — evaluation motion data
- `humanoidverse/data/lafan_29dof_10s-clipped.pkl` — training motion data
- If LFS fails: https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/main/data

### Docs

- `docs/papers/` — BFM-Zero paper (ICLR 2026 submission) LaTeX source and figures
- `docs/research/2511.04131/` — Structured research notes: paper overview, methodology, technical innovations, implementation details, experimental results, comparisons, limitations, and reproduction guide

### Key Dependencies

Isaac Sim/Lab (Linux only, via nvidia pypi), MuJoCo, PyTorch, Hydra, Pydantic, tyro, exca, wandb, tensordict.
