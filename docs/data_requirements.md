# BDX 运动数据集需求单（v3 期 · 2026-08-19）

> 交付对象：数据生成 agent。依据：planner PoC 全链实验结论（见 `docs/planner_poc_report.md`）。
> 一句话：**不需要更多小时数，需要打破"命令与历史的相关性"并补行为覆盖。**

## A. 内容需求（按优先级）

### A1. 干预式命令采样（最重要）
问题：现数据中命令可由历史预测（模型学会无视命令通道；我们在 head 和早期 loco 训练中两次踩坑）。
要求：
- **同一 history 配 3–5 个不同命令**（前进/后退/左转/边走边抬头/减速停等）；同一命令出现在多种不同 history 下；
- 配比：**~60% 独立干预采样 + 20–25% 自然连续运动 + 15–20% 边界切换**（走↔停、前↔后、直行↔急转、头左↔头右、速度突变）；
- 命令采样不要在 7 维矩形（vx∈[−0.35,1.0], vy∈[±0.4], vyaw∈[±1.0], neck 各关节限位内）均匀撒点：**proposal → 程序化 planner / 仿真 rollout → 可行性过滤 → 入集**；
- KPI：**命令覆盖 × 转换覆盖 × 反事实密度**（不以小时数计）。

### A2. 行为覆盖缺口（能力矩阵问号项）
- **持续转向**：各档角速度（0.2/0.4/0.6/0.8 rad/s 量级）的持续转弯弧线行走——现库几乎纯直行+瞬态转向；
- 横向侧移（vy 非零的行走）；
- 起步/停止的完整过渡过程（含速度斜坡）；
- 新 gesture 种类（现仅 10 种源动作、30 条 clip）。

### A3. 修现有导出瑕疵（带实测证据）
- **v2_backward 时间反演**：`_rev` clip 的速度符号错误（GT vx 读数为正）；
- **yaw 回绕伪影**：manifest `vyaw_range` 出现 ±156 rad/s（四元数 yaw 回绕未处理）——污染命令标签；
- 后退走体量偏小（359 train 条 vs 前进 2000+），修完导出后可考虑扩。

### A4. 切分要求
- **gesture 留出**：现 `eval_split.json` 把 30 条 gesture 全放 train（防泄漏分组与 walkhead 派生绑定所致）——补几条**无派生变体**的 gesture-only 留出 clip；
- **命令组合留出**：训练见 (前进,左看)+(后退,右看)，留出 (后退,左看) 这类未见组合，用于测组合泛化；
- 保留现有防泄漏分组（镜像/变速/时间反演/同 rollout 切段）。

## B. 格式要求（沿用 v2 约定，不变更）

- 每版本目录 + `manifest.json` + `clips/*.npz`；50 Hz；四元数 **wxyz**；世界系；运动学 = bdx_v4（body 序 = MJCF 序，17 体）；
- 每帧字段：`joint_pos/joint_vel (T,14)`、`body_pos_w (T,17,3)`、`body_quat_w (T,17,4)`、`body_lin_vel_w (T,17,3)`、`body_ang_vel_w (T,17,3)`、`fps`；
- 保留并继续：`body_names/joint_names` 自描述、`resettable` 掩码、`near_dup` 标签、`captions/tags`、每条 clip 的 command 元数据（teacher 命令向量 + 实现均值——修掉回绕后这个字段对齐训练很关键）；
- QC 口径不变（skate/fell/leg-cross 扫描）；contact 判定阈值统一 **0.08 m / 0.3 m/s**（站立踝高实测 0.0736 m）。

## C. 明确不需要的

- **更多同类行走的时长**（实测 4.6h→22.8h teacher 零收益）；
- 长时程参考高度下沉不需要数据侧处理（开环参考生成特性，部署每秒从真实状态重规划，无累积通道）。

## D. 交付后的消费方式（planner 侧，自动）

```
extract_sparse_bdx.py --full-pool   # 重抽取 ~10 分钟
train_p1 (teacher 12ep) → train_p2 (P2.9c 干预式 graft) → eval 全套对比基线
```
验收基线（v2 现行数字，新数据应不回退）：teacher q 20.6 mrad；planner vx 增益 0.96 / vyaw 0.013 rad/s / neck 0.96；P3 tracker 比值 1.04×。
