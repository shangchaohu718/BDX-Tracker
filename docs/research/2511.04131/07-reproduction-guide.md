# BFM-Zero 复现指南

> **注意**：BFM-Zero基于FB-CPR框架，实现涉及6个网络组件的离线策略联合训练。本指南基于论文公开信息提供方法论级别的复现指导。

## 前置条件

### 硬件要求

| 组件 | 要求 | 说明 |
|------|------|------|
| GPU | NVIDIA GPU (推荐RTX 4090+) | 训练需要 |
| RAM | 32GB+ | 大回放缓冲区 |
| 存储 | 50GB+ | 数据+模型+日志 |
| 机器人 | Unitree G1 (29-DoF) | 实机部署需要 |

### 软件环境

| 依赖 | 用途 |
|------|------|
| Isaac Lab | 训练仿真 (200Hz) |
| MuJoCo | 评估仿真 |
| PyTorch | 深度学习框架 |
| LAFAN1数据集 | 运动数据 |
| LocoMujoco | 运动重定向(可选) |

---

## 环境搭建

### 1. 安装Isaac Lab

```bash
# Isaac Lab (NVIDIA)
# https://isaac-sim.github.io/IsaacLab/
pip install isaaclab
```

### 2. 安装MuJoCo

```bash
pip install mujoco
```

### 3. 安装其他依赖

```bash
pip install torch numpy
```

### 4. 获取FB-CPR基础代码

BFM-Zero基于FB-CPR算法。FB-CPR的参考实现可在相关论文的代码库中找到。

---

## 运动数据准备

### 数据来源

| 数据集 | 用途 | 说明 |
|--------|------|------|
| LAFAN1 | 主训练数据 | 40条运动 |
| AMASS (CMU+BMLHandball) | 扩展实验 | OOD评估+可选训练 |

### 数据处理流程

```bash
# 1. 下载LAFAN1数据集
# 2. 使用LocoMujoco或其他工具重定向到Unitree G1
# 3. 分割为10秒片段（细粒度优先级）
# 4. 准备无标签轨迹数据：
#    τ = (o_1, s_1, ..., o_T, s_T)
#    - 观测 o: 64维本体感受
#    - 特权状态 s: 463维完整状态
```

---

## 训练

### 训练配置

| 参数 | 值 |
|------|-----|
| 仿真器 | Isaac Lab |
| 仿真频率 | 200Hz |
| 控制频率 | 50Hz |
| 并行环境 | 1,024 |
| 总训练步数 | ~192M |
| UTD比率 | 16 |
| Episode长度 | 500步 |
| 回放缓冲区 | ~5M transitions |

### 关键实现细节

#### 网络组件

```python
# 伪代码 - BFM-Zero的6个核心组件

class BFMZero:
    def __init__(self, d=256):
        # Forward映射 - 后继特征
        self.F = ResNet(
            embed_blocks=4, hidden=2048,
            residual_blocks=6, activation='mish'
        )

        # Backward映射 - 任务编码器
        self.B = MLP(layers=1, hidden=256, activation='relu')

        # 策略 (Actor)
        self.pi = ResNet(
            embed_blocks=4, hidden=2048,
            residual_blocks=6, activation='mish'
        )

        # 风格Critic
        self.Q_D = ResNet(
            embed_blocks=4, hidden=2048,
            residual_blocks=6, activation='mish',
            ensemble=2
        )

        # 辅助Critic
        self.Q_R = ResNet(
            embed_blocks=4, hidden=2048,
            residual_blocks=6, activation='mish',
            ensemble=2
        )

        # 判别器
        self.D = MLP(layers=2, hidden=1024, activation='relu')
```

#### 输入维度

```python
# 观测空间
o_t = concat(
    q_t - q_bar,     # 29维: 归一化关节位置
    q_dot_t,         # 29维: 关节速度
    omega_root_t/4,  # 3维:  根角速度(缩放)
    g_t              # 3维:  根投影重力
)  # 总计64维

# 特权状态
s_t = 463维 (根高度、身体姿态、旋转、线/角速度)

# 历史观测 (H=4)
o_t_H = concat(o_{t-4}, a_{t-4}, ..., o_t)  # 93*4 + 64 = 436维

# 潜在向量
z = 256维 (超球面上的向量)
```

#### FB损失

```python
def fb_loss(F, B, batch):
    o_H, s, a, o_H_next, s_next, z = batch

    # Forward映射
    F_val = F(o_H, s, a, z)          # [batch, d]
    F_next = F(o_H_next, s_next, a_next, z)  # [batch, d]

    # Backward映射
    B_val = B(s, o)                   # [batch, d]
    B_pos = B(s_pos, o_pos)           # 正样本 [batch, d]

    # FB时序差分损失
    td_loss = (F_val @ B_pos.T - gamma * F_next.detach() @ B_pos.detach().T)**2
    fb_term = -2 * (F_val @ B_next.T).mean()

    # 正交性约束
    ortho_loss = (B_val @ B_pos.T)**2 - (B_val * B_val).sum(-1)

    # Q值一致性
    q_loss = (F_val @ z - B_next.detach() @ Sigma_B @ z - gamma * F_next.detach() @ z)**2

    return td_loss.mean() + fb_term + ortho_loss.mean() + q_loss.mean()
```

#### 判别器训练

```python
def discriminator_loss(D, B, expert_batch, agent_batch):
    # 专家数据
    tau = expert_batch  # 运动轨迹
    z_tau = B(tau.states).mean(dim=0)  # 运动的零样本嵌入
    z_tau = z_tau * sqrt(d) / z_tau.norm()

    loss_expert = -D(tau.states, tau.obs, z_tau).log().mean()
    loss_agent = -(1 - D(agent_batch.s, agent_batch.o, agent_batch.z)).log().mean()

    return loss_expert + loss_agent

def discriminator_reward(D, s, o, z):
    d = D(s, o, z)
    return d / (1 - d)
```

#### Actor损失

```python
def actor_loss(F, Q_D, Q_R, pi, batch, alpha_D=0.05, alpha_R=0.02):
    o_H, s, z = batch
    a = pi(o_H, z)  # 策略输出动作

    fb_term = F(o_H, s, a, z) @ z          # 后继特征方向
    style_term = Q_D(o_H, s, a, z)          # 类人行为
    safety_term = Q_R(o_H, s, a, z)         # 安全约束

    return -(fb_term + alpha_D * style_term + alpha_R * safety_term)
```

#### 运动优先级采样

```python
def motion_priority(emd_score):
    """基于EMD的指数优先级"""
    clamped = max(0.5, min(emd_score, 2.0))
    return 2 ** (clamped * 4)
```

#### 域随机化

```python
def apply_domain_randomization(env):
    # 动力学随机化
    env.com_offset = np.random.uniform(-0.02, 0.02, 3)
    env.link_mass *= np.random.uniform(0.95, 1.05)
    env.friction = np.random.uniform(-0.5, 1.25)
    env.default_joint_pos += np.random.uniform(-0.02, 0.02, 29)
    env.push_robot(np.random.uniform(0, 0.5))

    # 观测噪声
    env.obs_noise = {
        'joint_pos': np.random.uniform(-0.01, 0.01, 29),
        'joint_vel': np.random.uniform(-0.5, 0.5, 29),
        'gravity': np.random.uniform(-0.05, 0.05, 3),
        'angular_vel': np.random.uniform(-0.05, 0.05, 3),
    }
```

---

## 零样本推理

### 奖励推断

```python
def reward_inference(B, reward_fn, replay_buffer, n_samples=400000):
    """给定奖励函数，推断最优潜在向量"""
    states = replay_buffer.sample(n_samples)
    rewards = np.array([reward_fn(s) for s in states])
    B_vals = np.array([B(s) for s in states])

    z_r = (rewards @ B_vals) / n_samples
    return z_r
```

### 目标到达

```python
def goal_inference(B, goal_state):
    """给定目标状态，计算潜在向量"""
    return B(goal_state)
```

### 运动跟踪

```python
def tracking_inference(B, motion_sequence, look_ahead=3):
    """给定运动序列，计算跟踪潜在向量序列"""
    z_sequence = []
    for t in range(len(motion_sequence)):
        z_t = sum(B(motion_sequence[t+k]) for k in range(look_ahead))
        z_sequence.append(z_t)
    return z_sequence
```

---

## 少样本适应

### CEM单姿态适应

```python
def cem_adaptation(z_init, env, n_iterations=20, population=50, elite_frac=0.2):
    """CEM优化潜在向量"""
    z = z_init.copy()
    sigma = np.ones_like(z) * 0.1

    for i in range(n_iterations):
        # 采样
        candidates = np.random.randn(population, len(z)) * sigma + z

        # 评估
        scores = [evaluate(c, env) for c in candidates]

        # 选择精英
        elite_idx = np.argsort(scores)[-int(population * elite_frac):]
        elite = candidates[elite_idx]

        # 更新
        z = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-5

    return z
```

---

## 复现检查清单

### 训练

- [ ] Isaac Lab安装
- [ ] LAFAN1数据获取
- [ ] 运动重定向到Unitree G1
- [ ] Forward映射F实现 (ResNet, 6块, 2048维)
- [ ] Backward映射B实现 (MLP, 256维)
- [ ] 策略π实现 (ResNet, 6块, 2048维)
- [ ] 判别器D实现 (MLP, 1024维)
- [ ] 风格Critic Q_D实现
- [ ] 辅助Critic Q_R实现
- [ ] FB损失实现
- [ ] GAN判别器损失实现
- [ ] Actor损失实现 (FB + 判别器 + 辅助)
- [ ] 非对称训练（策略用历史，Critic用特权）
- [ ] 域随机化
- [ ] 观测噪声注入
- [ ] 奖励正则化 (6项)
- [ ] 运动优先级采样 (EMD指数)
- [ ] 大规模并行训练 (1024环境)
- [ ] 离线策略训练 (UTD=16)

### 推理

- [ ] 奖励推断实现 (采样估计)
- [ ] 目标到达推断实现 (B映射)
- [ ] 运动跟踪推断实现 (前瞻求和)
- [ ] 技能插值实现 (SLERP)

### 适应

- [ ] CEM单姿态适应实现
- [ ] 轨迹适应实现 (双退火优化)

### 评估

- [ ] MuJoCo评估环境
- [ ] 24种奖励函数定义
- [ ] 跟踪评估 (E_mpjpe)
- [ ] 目标到达评估 (E_mpjpe)
- [ ] 奖励优化评估 (累积回报)

### 部署

- [ ] Unitree G1机器人
- [ ] Sim-to-real迁移验证
- [ ] 扰动恢复测试

---

## 降低复现门槛的建议

### 参考实现

- FB-CPR: 原论文代码库（虚拟角色动画）
- FastTD3: 大规模离线策略RL的参考实现

### 可从TWIST2复用的组件

| TWIST2组件 | 用途 |
|-----------|------|
| IsaacGym环境 | 训练环境基础 |
| 运动重定向 | GMR到G1 |
| Unitree G1 URDF | 机器人模型 |
| 部署脚本 | 实机框架 |

### 简化方案

| 原始配置 | 简化方案 | 预期效果 |
|---------|---------|---------|
| 440.5M (6块2048 ResNet) | 40.1M (2层1024 MLP) | 验证方法可行性 |
| LAFAN1 + AMASS | 仅LAFAN1 | 验证核心算法 |
| 全部6项正则化 | 关节极限+动作率 | 最低安全约束 |
| CEM + 轨迹优化 | 仅CEM | 验证适应概念 |

---

## 常见问题

### Q1: BFM-Zero vs PPO，训练难度如何？

BFM-Zero的训练复杂度远高于PPO：
- 6个网络组件需要联合训练
- 离线策略训练需要回放缓冲区管理
- FB损失需要仔细的数值稳定性处理
- 判别器训练可能不稳定（GAN相关问题）
- 建议从小模型(40M参数)开始验证

### Q2: 能否用PPO替代FB-CPR？

不行。FB-CPR的核心优势是学习统一的潜在表示，PPO只能学习单任务策略。BFM-Zero的多任务零样本推理完全依赖于FB表示的潜在空间。

### Q3: 与TWIST2项目如何结合？

最有前景的方向是将BFM-Zero作为TWIST2的多功能底层控制器：
1. TWIST2的上层系统（遥操作、视觉、Diffusion Policy）提供参考运动
2. 通过Backward映射编码为潜在向量
3. BFM-Zero策略执行，额外获得目标到达和奖励优化能力

### Q4: 440.5M参数能部署到机器人上吗？

可以。推理时只需要Actor网络(31.9M)和Backward映射(0.2M)，总计约32M参数。其他组件仅在训练时使用。实机控制频率50Hz，在板载计算机上可行。

### Q5: 为什么奖励推断有时失败？

域随机化增加了训练数据的方差，导致小样本估计不稳定。建议：
- 使用运动数据集（非回放缓冲区）进行推断
- 多次推断取最优
- 使用更大样本量

---

## 参考资源

- **论文**: `docs/papers/2511.04131/`
- **项目主页**: https://lecar-lab.github.io/BFM-Zero/
- **基础算法**: FB-CPR (Tirinzoni et al.)
- **训练框架参考**: FastTD3
- **数据集**: LAFAN1, AMASS
- **重定向**: LocoMujoco
- **仿真**: Isaac Lab
- **基线**: GMT, Any2Track, BeyondMimic

[上一篇: 局限性与未来方向](./06-limitations.md) | [返回目录](./README.md)
