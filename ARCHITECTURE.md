# BFM-Zero Architecture

> Analysis of the BFM-Zero model architecture, based on the `fb_cpr_aux` agent variant
> used for training. Source files: `humanoidverse/agents/nn_models.py`, `fb/model.py`,
> `fb/agent.py`, `fb_cpr/agent.py`, `fb_cpr_aux/model.py`, `fb_cpr_aux/agent.py`.

---

## 1. Overview

BFM-Zero learns **promptable** humanoid behaviors via unsupervised RL. A latent
skill vector **z** (256-dim) encodes "what to do"; the policy conditions on z to
produce actions. Training uses no task-specific reward — instead, a Forward-Backward
(FB) bilinear model provides implicit rewards, augmented by a discriminator and
auxiliary critics.

The model consists of **6 independently trained networks** with **no shared
parameters**:

```
                 Normalized Observation
                        │
    ┌──────┬──────┬─────┼──────┬──────────┐
    ▼      ▼      ▼     ▼      ▼          ▼
 Forward  Back   Actor Critic Aux-Crit  Discrim
 (f)      (b)   (π)   (Q_d)  (Q_aux)   (D)
```

---

## 2. Robot & Observation Space

**Robot**: Unitree G1 with 29 DOF + waist joints, 31 rigid bodies.

### Observation components

| Key | Dim | Content |
|-----|-----|---------|
| `base_ang_vel` | 3 | Angular velocity (gyroscope) |
| `projected_gravity` | 3 | Gravity vector in body frame |
| `dof_pos` | 29 | Joint positions |
| `dof_vel` | 29 | Joint velocities |
| `actions` | 29 | Previous step action (last_action) |
| **`state`** | **93** | Sum of above 5 keys |
| `history_actor` | 372 | 4-frame stack of state keys (93 × 4) |
| `privileged_state` / `max_local_self` | 462 | Full-body kinematic state (31 bodies) |
| **Total dict** | **927** | All keys combined |

### privileged_state breakdown (462 dim)

```
31 bodies × (3 pos + 6 rotation-tan-norm + 3 lin_vel + 3 ang_vel)
  = 31 × 15 - 3 (root pos removed) + 1 (root height)
  = 462 + 1 = 463 (with root_height_obs=True)
```

Per body: local position (3), rotation as 6D tan-norm (6), local linear velocity (3),
local angular velocity (3). All transformed into the root heading frame.

### history_actor breakdown (372 dim)

```
4 frames × (3 ang_vel + 3 gravity + 29 dof_pos + 29 dof_vel + 29 actions)
  = 4 × 93 = 372
```

FIFO buffer: newest frame at index 0, oldest at index 3.

### Input filters — which network sees what

| Network | Input filter keys | Effective dim (+ z) |
|---------|-------------------|---------------------|
| **Actor** | `state, last_action, history_actor` | 93 + 29 + 372 = **494** (+ 256 z) |
| **Forward** | `state, privileged_state, last_action, history_actor` | 93 + 462 + 29 + 372 = **956** (+ 256 z) |
| **Critic** | `state, privileged_state, last_action, history_actor` | 93 + 462 + 29 + 372 = **956** (+ 256 z + action) |
| **Aux Critic** | `state, privileged_state, last_action, history_actor` | 93 + 462 + 29 + 372 = **956** (+ 256 z + action) |
| **Discriminator** | `state, privileged_state` | 93 + 462 = **555** (+ 256 z) |
| **Backward** | `state, privileged_state` | 93 + 462 = **555** |

---

## 3. The z Latent

z is a 256-dim skill vector, L2-normalized and scaled by √256:

```python
z = sqrt(z_dim) * F.normalize(randn(batch, z_dim), dim=-1)  # norm_z=True
```

### Sampling strategies

| Method | Use | Formula |
|--------|-----|---------|
| Random | Exploration | `√256 · normalize(randn)` |
| Goal-encoded | Exploit | `backward_map(obs) → √256 · normalize` |
| Mixed (training) | Both | `z = where(rand < 0.2, z_goal, z_random)` (train_goal_ratio=0.2) |
| Tracking (inference) | Deploy | `backward_map(obs_seq)` averaged over sliding window of length `seq_length` |

### z buffer

During training, sampled z values are stored in a FIFO buffer (size 8192) and
reused for rollout diversity (`use_mix_rollout=True`).

---

## 4. Network Architectures

All configs below reflect the **production** (full-size) training setup. The
reduced model used for RTX 5080 testing is noted where different.

### 4.1 Backward Model (B) — z encoder

```
Input:  filtered obs (state + privileged_state) = 555 dim
        │
  Linear(555, 256) → LayerNorm(256) → Tanh
        │
  Linear(256, 256) → ReLU                    ← hidden_layers=1
        │
  Linear(256, 256)                           ← output to z_dim
        │
  √z_dim · L2_normalize → z (batch, 256)
```

Simple MLP. Tiny model (~0.3M params). Trained only through the FB loss (no
direct supervised loss on z).

### 4.2 Forward Model (F) — dynamics predictor

Predicts the next goal embedding z' given (obs, z, action). Uses a **2-ensemble**
(`num_parallel=2`) for uncertainty estimation.

**Full-size config**: hidden_dim=2048, hidden_layers=6, embedding_layers=2, num_parallel=2
**Reduced config** (RTX 5080): hidden_dim=1024, hidden_layers=4

```
Input:  obs (956 dim), z (256 dim), action (29 dim)
        │
        ├────────────────┬──────────────┐
        ▼                ▼              │
  concat(obs, z)    concat(obs, a)      │
    1212-dim           985-dim          │
        │                │              │
   embed_z           embed_sa          │
   (residual_        (residual_        │
    embedding)        embedding)        │
        │                │              │
        ▼                ▼              │
   (2, B, H/2)      (2, B, H/2)       │   ← num_parallel=2 duplicates
        │                │              │
        └── concat ──────┘              │
              │                         │
         (2, B, H)                      │
              │                         │
     ResidualBlock(H) × N_hidden       │   ← N=6 (full) or 4 (reduced)
              │                         │
     Block(H → z_dim, no activation)    │
              │                         │
              ▼                         │
Output: F (2, B, 256)                   │   ← 2 parallel predictions of z'
```

The residual embedding consists of:

```
residual_embedding(input_dim, hidden_dim, embedding_layers=2):
  Block(input_dim → hidden_dim, activation=True)     # project up
  Block(hidden_dim → hidden_dim//2, activation=True)  # project to half
```

Each `Block` is: `LayerNorm → Linear → Mish`. Each `ResidualBlock` is: `LayerNorm → Linear → Mish + skip connection`.

**Parameter count** (full-size): ~15.2M (2× from ensemble)
**Parameter count** (reduced): ~7.7M

### 4.3 Actor (π) — policy network

Same architecture as Forward but **single network** (no ensemble), outputs an
action distribution instead of z'.

```
Input:  obs (494 dim), z (256 dim)
        │
        ├────────────────┬──────────────┐
        ▼                ▼              │
  concat(obs, z)    obs alone           │
    750-dim           494-dim           │
        │                │              │
   embed_z           embed_s            │
   (residual)        (residual)         │
        │                │              │
        └── concat ──────┘              │
              │                         │
         (B, H)                         │   ← num_parallel=1 (no ensemble)
              │                         │
     ResidualBlock(H) × N_hidden       │
              │                         │
     Block(H → action_dim, no activation)│
              │                         │
     Tanh → scale by action_clip_value   │
              │                         │
              ▼                         │
Output: mean (B, 29)                    │
        std  = actor_std (learnable, init=0.05)
        dist  = TruncatedNormal(mean, std, clip=0.3)
```

**Parameter count** (full-size): ~11.2M
**Parameter count** (reduced): ~3.7M

### 4.4 Critic (Q_d) — discriminator reward Q-function

Identical architecture to Forward Model: `num_parallel=2`, residual,
hidden_dim=2048, 6 layers. Takes (obs, z, action) → Q value.

```
Input:  obs (956), z (256), action (29)
        │
  [Same as Forward Model: embed_z + embed_sa → concat → residual blocks → output]
        │
Output: Q (2, B, 1)    ← 2 ensemble predictions, scalar Q-value
```

Trained with TD(0) using rewards from the Discriminator. Uses target network
with polyak averaging (tau=0.005).

### 4.5 Aux Critic (Q_aux) — auxiliary reward Q-function

Identical architecture to Critic. Takes (obs, z, action) → Q_aux value.

Trained with TD(0) using auxiliary task rewards (joint limits, torques, slippage,
feet orientation, ankle roll, action rate, undesired contact).

### 4.6 Discriminator (D) — expert vs agent classifier

WGAN-GP style discriminator. No ensemble.

```
Input:  obs (555), z (256)
        │
  concat(obs, z) = 811 dim
        │
  Linear(811, 1024) → activation
  Linear(1024, 1024) → activation
  Linear(1024, 1024) → activation     ← hidden_layers=3
  Linear(1024, 1)                     ← scalar output
        │
Output: D (B, 1)  — unnormalized score (no sigmoid)
```

**Parameter count**: ~2.8M

---

## 5. Loss Functions

### 5.1 Forward-Backward (FB) Loss — the core

Computes a batch matrix `M = F · Bᵀ` of shape `(num_parallel, batch, batch)`:

```python
F = forward_map(obs, z, action)       # (2, B, 256) — 2 parallel predictions
B = backward_map(goal_obs)             # (B, 256)
M = F @ B.T                           # (2, B, B) — pairwise compatibility
```

Same for targets with next obs:
```python
F_target = target_forward_map(next_obs, z, next_action)
B_target = target_backward_map(goal_obs)
M_target = F_target @ B_target.T
```

**Loss components**:

| Component | Formula | Purpose |
|-----------|---------|---------|
| `fb_offdiag` | `0.5 · MSE(M_offdiag, γ · M_target_offdiag)` | Bellman backup on cross-sample pairs |
| `fb_diag` | `-mean(diag(M))` | Maximize self-prediction (push diagonal → 1) |
| `orth_diag` | `-mean(diag(B·Bᵀ))` | Maximize z diversity (encourage norm = z_dim) |
| `orth_offdiag` | `0.5 · MSE(offdiag(B·Bᵀ))` | Decorrelate z vectors across batch |
| **Total** | `fb_offdiag + fb_diag + ortho_coef × (orth_diag + orth_offdiag)` | |

Where `ortho_coef = 100.0`.

The implicit Q value: `Q_fb = (F · z).sum(-1)` — dot product of forward prediction
with the goal z. No explicit reward needed.

### 5.2 Critic Loss — discriminator reward

Standard TD(0) with pessimistic ensemble:

```python
reward = compute_reward(discriminator_output)  # WGAN-style reward from D
target_Q = reward + discount * target_critic(next_obs, z, next_action)
loss = MSE(critic(obs, z, action), target_Q.detach())
```

### 5.3 Discriminator Loss — WGAN-GP

```python
# Real = expert trajectories, Fake = agent rollouts
D_real = discriminator(expert_obs, z)
D_fake = discriminator(agent_obs, z)

# Gradient penalty
gradient_penalty = λ_gp · (||∇D(interpolated)||_2 - 1)²

loss = D_fake.mean() - D_real.mean() + gradient_penalty
```

The discriminator reward for the Critic is: `reward = -log(1 - sigmoid(D))` or
simply `D` as a continuous signal.

### 5.4 Aux Critic Loss

Same as Critic loss but with auxiliary reward:

```python
aux_reward = Σ (aux_rewards_scaling[k] × aux_reward[k])
# e.g. penalty_torques × 0.0, penalty_action_rate × -0.1, limits_dof_pos × -10.0, ...
```

Aux reward is normalized via `RewardNormalizer` (translate=False, scale=True).

### 5.5 Actor Loss — combined

```python
# Three Q estimates, all with pessimistic penalty
Q_disc = critic(obs, z, action_sampled)         → mean - penalty × uncertainty
Q_aux  = aux_critic(obs, z, action_sampled)     → mean - penalty × uncertainty
Q_fb   = (forward(obs, z, action_sampled) · z)  → mean - penalty × uncertainty

# Weighted combination
weight = |Q_fb|.mean().detach() if scale_reg else 1.0
actor_loss = -Q_fb.mean()
             - Q_disc.mean() × reg_coeff × weight       # reg_coeff = 0.05
             - Q_aux.mean() × reg_coeff_aux × weight     # reg_coeff_aux = 0.02
```

The actor maximizes all three Q values, weighted by the FB Q magnitude.

---

## 6. Training Loop

```
for t in range(num_env_steps):
    1. Collect rollout: env.step(action) → obs, reward, done
    2. Store transition in replay buffer (CPU or CUDA)
    3. If t > num_seed_steps and time to update:
       for _ in range(num_agent_updates):  # default 16
           a. Sample batch from buffer
           b. Sample z (mixed: random + goal-encoded)
           c. Update FB model (f + b)
           d. Update Critic (Q_d)
           e. Update Aux Critic (Q_aux)
           f. Update Discriminator (D)
           g. Update Actor (π) — delayed (every other step)
    4. Soft-update target networks (polyak averaging)
    5. Log metrics every log_every_updates steps
    6. Checkpoint every checkpoint_every_steps steps
```

### Key hyperparameters (production / reduced)

| Param | Production | Reduced (RTX 5080) |
|-------|-----------|-------------------|
| `online_parallel_envs` | 1024 | 256 |
| `num_env_steps` | 384M | 50M |
| `batch_size` | 1024 | 512 |
| `buffer_size` | 5.12M | 2M |
| `buffer_device` | cuda | cpu |
| `hidden_dim` | 2048 | 1024 |
| `hidden_layers` | 6 | 4 |
| `num_parallel` (F, Critic, Aux) | 2 | 2 |
| `amp` | False | False |
| `num_seed_steps` | 10,240 | 10,240 |
| `num_agent_updates` | 16 | 16 |
| `update_agent_every` | 1024 | 1024 |
| `discount` | 0.98 | 0.98 |

---

## 7. Hardware Requirements

### VRAM breakdown (full model, 1024 envs)

| Component | VRAM |
|-----------|------|
| Isaac Sim 4.5 (1024 envs) | ~5 GB |
| Model weights + targets (6 networks) | ~0.5 GB |
| Adam optimizer states (2× params) | ~0.5 GB |
| Training activations (forward + backward pass) | ~4-5 GB |
| CUDA buffer (optional) | ~3 GB |
| **Total (with CUDA buffer)** | **~13-14 GB** |
| **Total (with CPU buffer)** | **~10-11 GB** |

### Recommended GPUs

| GPU | VRAM | Full model | Reduced model |
|-----|------|-----------|--------------|
| RTX 4090 / A6000 | 24 GB | ✅ (1024 envs) | ✅ |
| RTX 5080 | 16 GB | ❌ | ✅ (256 envs, CPU buffer) |
| RTX 4080 | 16 GB | ❌ | ✅ (256 envs, CPU buffer) |
| RTX 3090 | 24 GB | ✅ (512 envs) | ✅ |

---

## 8. File Map

```
humanoidverse/
├── train.py                          # Training loop, TrainConfig, train_bfm_zero()
├── agents/
│   ├── nn_models.py                  # Network architectures (ForwardMap, BackwardMap, Actor, etc.)
│   ├── fb/
│   │   ├── model.py                  # FBModel: orchestrates f, b, actor, obs_normalizer
│   │   └── agent.py                  # FBAgent: FB loss, actor update, tracking z sampling
│   ├── fb_cpr/
│   │   └── agent.py                  # FBcprAgent: extends FBAgent with Critic + Discriminator
│   └── fb_cpr_aux/
│       ├── model.py                  # FBcprAuxModel: adds Aux Critic + reward normalizer
│       └── agent.py                  # FBcprAuxAgent: adds aux critic loss, combined actor loss
├── envs/
│   └── legged_robot_motions/
│       └── legged_robot_motions.py   # compute_humanoid_observations_max (privileged obs)
├── config/
│   ├── obs/bfm_zero_obs.yaml         # Observation config (history length, obs keys)
│   └── robot/g1/g1_29dof_hard_waist.yaml  # Robot config (29 DOF, 31 bodies)
└── utils/helpers.py                  # Observation assembly, history handler
```

---

## 9. Data Flow Diagram

```
                         ┌──────────────────────────────────────────┐
                         │           Isaac Sim / MuJoCo             │
                         │     (1024 parallel environments)         │
                         └──────────────┬───────────────────────────┘
                                        │ raw obs
                                        ▼
                         ┌──────────────────────────┐
                         │     Observation Assembler  │
                         │  - Obs Normalizer (BatchNorm)│
                         │  - History Handler (4 frames)│
                         │  - Privileged state extract  │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
             ┌─────────┐        ┌──────────┐         ┌──────────┐
             │  state   │        │ history  │         │privileged│
             │  (93)    │        │_actor(372)│         │_state(462)│
             └────┬─────┘        └────┬─────┘         └────┬─────┘
                  │                   │                    │
    ┌─────────────┼───────────────────┼────────────────────┼──────────┐
    │             │    Actor Input    │                    │          │
    │             └───────┬───────────┘                    │          │
    │                     │                                │          │
    │    ┌────────────────┴────────────────┐               │          │
    │    │         z (256 dim)             │               │          │
    │    │  ┌ random ∥ goal-encoded ──┐   │               │          │
    │    │  └─────────────────────────┘   │               │          │
    │    └──────────┬─────────────────────┘               │          │
    │               │                                     │          │
    │    ┌──────────▼──────────┐                          │          │
    │    │  Actor (π)          │                          │          │
    │    │  494 + 256 → action │                          │          │
    │    └──────────┬──────────┘                          │          │
    │               │ action (29 dim)                     │          │
    │    ┌──────────▼──────────────────────────────────────▼───┐     │
    │    │           Forward / Critic / Aux Critic             │     │
    │    │           956 + 256 + action → Q values             │     │
    │    └──────────┬──────────────────────────────────────┬───┘     │
    │               │                                      │         │
    │               │ Q_fb, Q_disc, Q_aux                  │         │
    │    ┌──────────▼──────────┐        ┌─────────────────▼──┐      │
    │    │  Actor Loss          │        │  Discriminator (D)  │      │
    │    │  -Q_fb - α·Q_disc   │        │  555 + 256 → score  │      │
    │    │  - β·Q_aux          │        │  expert vs agent    │      │
    │    └─────────────────────┘        └────────────────────┘      │
    │                                                              │
    │    ┌─────────────────────────────────────────────────────┐     │
    │    │                Backward (B)                         │     │
    │    │                555 → z (256)                        │     │
    │    └─────────────────────────────────────────────────────┘     │
    │                                                              │
    │    ┌─────────────────────────────────────────────────────┐     │
    │    │              FB Loss = F·Bᵀ matrix                 │     │
    │    │              diagonal → 1, off-diagonal Bellman     │     │
    │    └─────────────────────────────────────────────────────┘     │
    │                                                              │
    └── Replay Buffer (CPU or CUDA, up to 5M transitions) ────────┘
```