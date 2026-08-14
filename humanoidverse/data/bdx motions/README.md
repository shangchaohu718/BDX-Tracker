# BDX Motion Library

BDX v4 双足机器人的**动作库**——把走路、站立、表情手势三类动作录好后，沿时间拼成**一个 npz**（带分段索引），用于训练一个通用的策略网络。

- **文件**：`motion_library.npz`（一个文件，含全部动作）
- **总量**：18 段动作 / **162000 帧 / 3240 秒（54 分钟）**
- **采样率**：50 Hz（四元数 wxyz / MuJoCo 约定）
- **机器人**：BDX v4（14 DOF：10 腿 + 4 脖子）

---

## 1. 有哪些动作 + 每个动作几帧到几帧

下表按**时间顺序**列出每个动作在整个 npz 里的帧区间（左闭右开 `[start, end)`）。用 `motion_id[帧号]` 或 `length_starts` 也能查到某帧属于哪个动作。

| 段号 | 类别 | 动作名 | 帧区间 | 帧数 | 时长 |
|---|---|---|---|---|---|
| 0 | walk | walk_seed0 | 0 – 15000 | 15000 | 300s |
| 1 | walk | walk_seed1 | 15000 – 30000 | 15000 | 300s |
| 2 | walk | walk_seed2 | 30000 – 45000 | 15000 | 300s |
| 3 | walk | walk_seed3 | 45000 – 60000 | 15000 | 300s |
| 4 | walk | walk_seed4 | 60000 – 75000 | 15000 | 300s |
| 5 | stand | stand_seed0 | 75000 – 90000 | 15000 | 300s |
| 6 | stand | stand_seed1 | 90000 – 105000 | 15000 | 300s |
| 7 | stand | stand_seed2 | 105000 – 120000 | 15000 | 300s |
| 8 | stand | stand_seed3 | 120000 – 135000 | 15000 | 300s |
| 9 | stand | stand_seed4 | 135000 – 150000 | 15000 | 300s |
| 10 | gesture | angry_no（生气摇头）| 150000 – 151500 | 1500 | 30s |
| 11 | gesture | angry_yes（生气点头）| 151500 – 153000 | 1500 | 30s |
| 12 | gesture | attention（注意）| 153000 – 154500 | 1500 | 30s |
| 13 | gesture | bored_snore（打瞌睡）| 154500 – 156000 | 1500 | 30s |
| 14 | gesture | laugh_giggle（咯咯笑）| 156000 – 157500 | 1500 | 30s |
| 15 | gesture | relax（放松）| 157500 – 159000 | 1500 | 30s |
| 16 | gesture | relax_v2（放松 v2）| 159000 – 160500 | 1500 | 30s |
| 17 | gesture | shy_yes（害羞点头）| 160500 – 162000 | 1500 | 30s |

**按类别汇总**：
- **walk（直行）**：段 0–4，5 个不同 seed（不同随机种子 → 不同的速度/转向命令序列，增加多样性），各 300s，**共 1500s**。走路的腿在步态库里只有 7 种步态（前进 0.3/0.6、后退 0.2/0.4、原地踏步、原地左/右转身），每个 seed 会把这些步态都扫一遍。
- **stand（站立）**：段 5–9，5 个 seed，各 300s，**共 1500s**。原地站着，但脖子/躯干在跟踪随机姿态命令（偏航、俯仰、侧倾、身高都会变）。
- **gesture（表情手势）**：段 10–17，8 种手势，各 30s（手势动作短，会循环多遍），**共 240s**。站立基础上的上半身/脖子表达。

> 手势的单次动作长度（循环一圈）：angry_no 3.2s、angry_yes 0.7s、attention 2.5s、bored_snore 4.4s、laugh_giggle 2.7s、relax 7.4s、relax_v2 9.2s、shy_yes 2.3s。30s 录制里会循环 3–40 遍。

**关于 seed（为什么 walk/stand 各录 5 个）**

"seed" = 随机数种子（一个整数），录制时用 `--seed N` 传进去，设给 PyTorch / NumPy / 仿真环境三处随机数发生器。**关键是环境那个 seed**：它决定**命令重采样的随机序列**——
- walk：`velocity_cmd` 每 3–4 秒从步态网格里重抽一个 `[vx, omega]`，抽哪一档、何时抽，都由 seed 决定；
- stand：头/躯干姿态命令每 3–6 秒重抽一个随机姿态，同理。

策略本身是**确定**的（同样观测 → 同样动作），所以多样性的唯一来源就是这些被随机抽出来的命令：同一个策略、同一个步态库，喂不同命令序列 → 录到不同轨迹。换句话，**同一个 seed 重录会得到完全一样的数据（可复现），换 seed 才会得到不同的轨迹**。

例：同样前 30 秒，两个 seed 抽到的命令完全不同——

| 时间 | seed 0 | seed 1 |
|---|---|---|
| 0s | 后退 0.40 · 右转 0.90 | 前进 0.40 · 左转 0.90 |
| 3.6s | 原地 · 右转 0.60 | 原地 · 直行 |
| 4.7s | （继续上一条）| 后退 0.30 · 左转 1.00 |

> 手势**没用多 seed**：手势是跟踪固定的参考片段（动作已内嵌在 onnx 里），动作本身确定，多 seed 只在开局重置姿态上有微小差别，所以每个手势只录一段。

---

## 2. 数据里面记录了什么

每一帧都记录了机器人**完整的运动状态**。分 4 类（共 12 个数组字段 + 分段索引）：

**① 躯干（根连杆）在哪、朝哪、怎么动**
- `root_pos`：躯干在世界里的位置 (x, y, z)
- `root_rot`：躯干的朝向，用 4 个数（四元数）表示
- `root_lin_vel` / `root_ang_vel`：躯干的线速度 / 转动角速度

**② 14 个关节的角度**（腿 10 个 + 脖子 4 个；关节 = 能转动的连接处：髋、膝、踝、脖子）
- `joint_pos`：14 个关节各自的角度
- `joint_vel`：14 个关节的角速度
- `joint_target`：控制器让每个关节去到的目标角度

**③ 全身 17 个部位的位姿** —— 把机器人拆成 17 段刚体，每段都记它的位置和朝向
- 17 段 = **1 个机身（底座）+ 两条腿各 5 段（髋 3、膝 1、踝 1）+ 脖子 4 段 + 左右耳各 1 段**
- `body_pos` / `body_rot`：每段在世界里的位置 / 朝向
- `body_lin_vel` / `body_ang_vel`：每段的线速度 / 角速度

**④ 动作录制时策略让关节转到哪**
- `actions`：14 个数，录这个动作时教师策略的输出（每个关节的目标）

**分段索引**（用来知道某帧属于哪个动作）：
- `_motion_num_frames`：`(18,)` 每段的帧数
- `length_starts`：`(18,)` 每段的起始帧
- `motion_id`：`(162000,)` 每一帧属于第几段（0=walk_seed0 … 17=shy_yes）

**标量/元信息**：`fps`(=50)、`meta_json`（含所有段的 task、策略、帧数等说明，可 `json.loads` 读）。

> 名词小贴士：**四元数** = 用 4 个数表示一个朝向；**T** = 总帧数（这里是 162000）。
>
> **注意**：这个库里**没有** `obs`（观测）和命令字段——因为走路、站立、手势三个任务的观测维度不一样（走路 79 / 站立 71 / 手势不同），没法并成一个数组。通用策略走「动作模仿（motion imitation）」路线：策略从上面记录的**运动状态**（躯干+关节+身体）按自己的观测约定重建观测，再用跟踪奖励学这些动作，所以不需要教师任务的原始观测。

---

## 3. 用法

```python
import numpy as np, json

d = dict(np.load("motion_library.npz", allow_pickle=True))
root_pos, joint_pos, actions = d["root_pos"], d["joint_pos"], d["actions"]
meta = json.loads(str(d["meta_json"]))

# 查某帧属于哪个动作：
motion_id = d["motion_id"]            # (162000,)
clip = int(motion_id[160000])         # → 16 = relax_v2
print(meta["clips"][clip])            # 这段的 task / 策略 / 帧数 …

# 取出某个动作的所有帧（以 laugh_giggle 为例，段 14）：
s, e = 156000, 157500
laugh_root = root_pos[s:e]            # (1500, 3)
```

**回放**（snapshot 模式，逐帧把保存的位姿写进 MuJoCo 看）：
```bash
cd /home/tcl/bdx_rl_mjlab_periodic
MUJOCO_GL=glfw uv run python -m bdx_rl.scripts.replay_rollout motion_library.npz --speed 2.0
```
（会从头到尾连续放完 162000 帧；窗口里 ↑/↓ 调速、空格暂停、R 重放、关窗退出。）

---

## 4. 数据来源（可复现）

- **walk**：教师策略 `periodic_model_99999.pt`（周期步态任务 `Mjlab-BDX-Mimic-Periodic-Flat-V4`），5 个 seed 各录 300s 连续直行。
- **stand**：教师策略 `bdx_mimic_perpetual`（站立任务 `Mjlab-BDX-Mimic-Perpetual-Flat-V4`），5 个 seed 各录 300s。
- **gesture**：8 个部署好的 onnx 表情策略（`tracking_bdx` 手势动作跟踪任务），各录 30s。

合并/录制脚本与原始 rollout 见 `bdx_motion_dataset/`（源项目）与两个 bdx_rl 项目。
