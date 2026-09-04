# 同向组合补丁数据任务单（P2.10b 实验）

> 2026-08-20。目标：回答"neck 同向混淆是数据问题还是架构问题"的最小 A/B。
> 预期规模：**数百条，不重建数据集**。

## 背景（一段话）

planner 在 4 个 strict-unseen 组合上：速度通道退化已被训练手段修复（×3.44→×1.39），
但**同向组合的 neck 命令崩溃**（turn_l+look_l 类，neck 误差 ~0.7 rad）只缓解了 12%。
已验证训练池 2695 条中同向组合内容为零（连 combo_tag=None 的 2426 条里都没有）。
本补丁用于判定：数据补齐后 neck 误差降到 <0.35 rad → 纯数据问题（不改架构）；
只降到 ~0.6 → 架构绑定问题（上 factorized encoder）。

## 需要的四类组合

| 组合 | neck_yaw 命令 | vyaw 或 vy 命令 | 组数 |
|---|---|---|---|
| turn_l + look_l | +0.6 | vyaw +0.5 | 30-50 个独立 history |
| turn_r + look_r | -0.6 | vyaw -0.5 | 30-50 |
| strafe_l + look_l | +0.6 | vy +0.15 | 30-50 |
| strafe_r + look_r | -0.6 | vy -0.15 | 30-50 |

## 结构要求（信息量最大化的关键）

- **每个 history group 内含多个 neck 分支**（不只是同向）：
  ```
  同一个 turn_l history（≥50 帧稳态）：
    分支 1：neck_yaw = +0.6（同向 ← 本实验核心）
    分支 2：neck_yaw = 0（中性对照）
    分支 3：neck_yaw = -0.6（异向对照，模型已会）
  ```
- group_id 标注（整组同侧切分防泄漏）；这些组**全部进 train**（它们不是 holdout）；
- 命令向量的 neck 档位用现有 look_l/look_r 同款（+0.6/-0.6），保持与既有组合池同分布；
- vx 可自由（延续 history 的自然值即可）。

## 格式

与 planner_v2 池完全一致（相对路径 npz + command_vector + combo_tag）。
combo_tag 建议：`turn_l+look_l_patch`（带 `_patch` 后缀，便于训练侧按来源切片分析）。

## 消费与判据（planner 侧已备好）

训练 = 现有池 + 补丁（同 seed/预算/架构），评估 = 原 4 个 strict-unseen holdout：
- neck 误差 0.70 → **<0.35 rad**：数据问题结案，配方更新进 v1.x；
- neck 误差 → ~0.6：数据已给答案但绑定失败，启动 factorized command encoder（P2.12 设计已定稿）。
