# BFM-Zero 研究报告索引

**论文**: BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning
**arXiv**: 2511.04131
**机构**: Carnegie Mellon University + Meta
**投稿**: ICLR 2026
**日期**: 2025年11月

---

## 快速概览

BFM-Zero 是首个基于**无监督强化学习**的**行为基础模型**，在Unitree G1人形机器人上实现了可提示(promptable)的全身控制。通过Forward-Backward表示学习统一的潜在空间，单一策略可零样本执行运动跟踪、目标到达、奖励优化等多种任务。

### 核心数据

| 维度 | 值 |
|------|-----|
| 方法类别 | 离线策略无监督RL |
| 训练数据 | LAFAN1 (40条运动) |
| 模型参数 | 440.5M (含所有组件) |
| 潜在维度 d | 256 |
| 观测维度 | 64维本体感受 / 463维特权信息 |
| 动作维度 | 29 (关节PD目标) |
| 机器人 | Unitree G1 (29-DoF) |
| 仿真器 | Isaac Lab (200Hz) / 控制频率50Hz |

### 核心架构

```
                    预训练阶段
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
 Forward Map F    Backward Map B      Discriminator D
 (后继特征)       (任务编码器)        (运动判别器)
    │                   │                   │
    └─────────┬─────────┘                   │
              │                             │
              ▼                             ▼
        潜在空间 Z ⊆ R^d           Style Critic Q_D
        (d=256维超球面)                    │
              │                             │
              ├─────────────────────────────┤
              │                             │
              ▼                             ▼
        策略 π(o_{t,H}, z)         Aux Critic Q_R
        (历史条件化)                (安全约束)
              │
              ▼
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
零样本推理             少样本适应
├─ 运动跟踪            ├─ CEM单姿态优化
├─ 目标到达            └─ 轨迹级优化
└─ 奖励优化

```

### 与TWIST2的关系

| 维度 | TWIST2 | BFM-Zero |
|------|--------|----------|
| 方法 | 在线策略PPO | 离线策略无监督RL |
| 任务 | 运动跟踪专用 | 通用可提示(跟踪/目标/奖励) |
| 训练范式 | 监督(跟踪奖励) | 无监督(自发现奖励) |
| 潜在空间 | 无 | 256维结构化空间 |
| 运动数据 | ~20k clips | LAFAN1 (40条运动) |
| 扩展性 | 单任务 | 多任务零样本 |

---

## 报告目录

| 编号 | 标题 | 内容 |
|------|------|------|
| [00](./00-paper-overview.md) | 论文概述 | 背景、动机、贡献 |
| [01](./01-methodology.md) | 方法论 | FB表示、预训练、推理、适应 |
| [02](./02-technical-innovations.md) | 技术创新 | 6大创新点分析 |
| [03](./03-implementation-details.md) | 实现细节 | 架构、超参数、域随机化 |
| [04](./04-experimental-results.md) | 实验结果 | 仿真验证、实机部署、适应 |
| [05](./05-comparison.md) | 相关工作比较 | 与PPO/SONIC/TWIST2等对比 |
| [06](./06-limitations.md) | 局限性与未来方向 | 已知局限、扩展路径 |
| [07](./07-reproduction-guide.md) | 复现指南 | 环境搭建、训练配置、检查清单 |
