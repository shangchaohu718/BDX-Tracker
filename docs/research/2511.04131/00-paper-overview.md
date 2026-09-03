# BFM-Zero 论文概述

## 论文信息

| 项目 | 内容 |
|------|------|
| 标题 | BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning |
| arXiv | 2511.04131 |
| 机构 | Carnegie Mellon University + Meta |
| 作者 | Yitang Li*, Zhengyi Luo*, Tonghe Zhang$, Cunxi Dai$, Andrea Tirinzoni, Anssi Kanervisto 等 |
| 项目主页 | https://lecar-lab.github.io/BFM-Zero/ |
| 本地 PDF | `/home/tcl/Documents/RoboticsControl/2511.04131v1 BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning.pdf` |
| 机器人 | Unitree G1 (29-DoF), Booster T1 |

---

## 背景

### 现有人形控制方法的三大局限

1. **任务专用**：大多数策略针对单一任务训练（运动模仿或单一运动/操作任务）
2. **不可适应**：训练后无法轻松微调或组合用于新任务
3. **缺乏统一接口**：没有统一可解释的目标规范和行为组合接口

### 为什么不能直接用行为克隆(VLA)?

对于人形机器人全身控制，存在根本性不匹配：
- 没有现成的执行器级动作标签
- 没有大规模遥操作数据集
- 全身控制需要高频执行(50-200Hz)，远高于操作任务的5-10Hz

### 现有运动跟踪方法的问题

| 方法 | 问题 |
|------|------|
| PPO + 跟踪奖励 | 任务专用，不可迁移 |
| 在线策略训练 | 无法利用离线数据，样本效率低 |
| 显式奖励工程 | 每个任务需重新设计奖励 |

### 无监督RL的潜力

VLA模型在操作领域已证明多任务泛化能力，但全身控制缺乏大规模数据。无监督RL可在不依赖显式奖励的情况下学习通用行为表示。

---

## 核心贡献

### 贡献一：首个面向真实人形机器人的无监督RL

**核心洞察**：离线策略无监督RL可以学习有效的共享潜在表示，将运动、目标和奖励嵌入统一空间。

**突破点**：
- 首次将无监督RL（FB-CPR）成功部署到真实人形机器人
- 单一策略可通过不同提示方式执行多种任务
- 无需任务特定重训练

### 贡献二：多模态零样本推理

**三种零样本推理方式**：
1. **运动跟踪**：给定参考运动序列 → 跟踪执行
2. **目标到达**：给定目标姿态 → 自然过渡并保持
3. **奖励优化**：给定奖励函数 → 自动发现最优行为

### 贡献三：高效少样本适应

- 单姿态适应：CEM优化潜在向量，20次迭代
- 轨迹适应：双退火轨迹优化，减少29.1%跟踪误差
- 无需微调网络参数

### 贡献四：关键sim-to-real设计

- 非对称训练（策略用历史，Critic用特权信息）
- 域随机化（物理参数+扰动+传感器噪声）
- 奖励正则化（关节极限、滑动、自碰撞等）

---

## 论文结构

| 章节 | 内容 |
|------|------|
| Method | 无监督RL+FB表示、预训练、零样本推理、少样本适应 |
| Experiments | 仿真验证、实机部署、适应实验、潜在空间分析 |
| Discussion | 局限性、未来方向 |
| Appendix | 相关工作、训练细节、额外结果、Booster T1实验 |

---

[下一篇: 方法论](./01-methodology.md)
