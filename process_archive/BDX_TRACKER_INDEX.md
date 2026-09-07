# BDX-Tracker 全流程索引 (Process Archive Index)

本仓库 = BDX tracker 全流程工作归档（BFM-zero 代码库完整历史 + 本目录的过程记录）。
分支 `mjlab_fixed_dance_bdx` 为最新工作线（含全部蒸馏战役与闭环 demo 提交）。

## 1. 全流程四个阶段

| 阶段 | 内容 | 位置 |
|---|---|---|
| ① Tracker RL 训练 | bfmzero-bdx-full：BDX 14-DoF 双足全身跟踪策略（Isaac/MuJoCo，BFM latent 驱动） | `humanoidverse/`（训练框架）、权重见 §3 Release |
| ② Planner 蒸馏战役 | teacher MotionAE → CommandedEncoder 学生（v12 CANONICAL，17 轮闭环 G1–G6 全过） | `humanoidverse/planner/`、`process_archive/distill_campaign/` |
| ③ 闭环接线 demo | 人 5 稀疏点 → v12 planner → z → tracker → MuJoCo（舞蹈版+走路版） | `humanoidverse/scripts/human_imitation_closedloop.py`、`process_archive/demos/human_imitation/` |
| ④ 已知问题登记 | RUNTIME_X_AXIS_TERMINATION_LOOP（runtime 自动 reset 传送循环，独立集成问题） | `ISSUES/RUNTIME_X_AXIS_TERMINATION_LOOP.md` |

## 2. 本目录内容

- `distill_campaign/` — 蒸馏战役权威记录：STATE.md（阶段路线与判据）、
  DISTILL_FINAL_REPORT.md（终报 §1–§12，含 erratum 与用户终裁）、
  BASELINE_REPRODUCE.md / P2_1_SWEEP_EVIDENCE.md / FAILURE_LEDGER.md、
  CURRENT_EVALUATION_AUTHORITY.json（SUPERSEDED 机器可读覆盖指针）。
  （git_repair/ 备份包 90MB 未随档，属一次性修复工件。）
- `planner_canonical_v2combo/` — 正典 planner 包的可入档子集：
  `p2_student_v12.pt`（CANONICAL, sha16 34047cde8f6c5a3b）+
  `p1_ae_l64.pt`（teacher）+ `head_pos.npy`/`splits.json`/`closedloop_dummy.pkl` +
  全部评估/血统/收口 JSON（canonical.json 含 vx 四格与三段式标签）。
  大体积训练数据（sparse_full_v1m.npz 1.2GB、train/holdout npz）未随档，
  由 `planner/dataset.py` + `scripts/build_v12_package.py` 等从 BDX 动作数据集重建。
- `demos/human_imitation/` — 人体模仿闭环两版视频（dance1 首版 +
  fightAndSports1_subject1_clip19 走路段复跑）+ 指标 JSON + README（诚实标签）。
- `demos/backward_smoke_v12/` — 12/12 FAIL 有界烟测全程证据
  （冻结 SMOKE_PLAN/RESULT、DISPOSITION、12 条 trial 视频、探针帧）。
- `demos/forward_diag/` — 前向对照诊断（x 通道仪器 INVALID 的证据）。
- `demos/final_eval_p54/` — P5.4 终评 JSON（v12 采用依据）。
- `demos/demo_planner_v29.mp4` — planner 指令跟踪 demo。

## 3. Tracker 权重下载（>100MB，存于 GitHub Release）

`results/bfmzero-bdx-full/`（255MB）不入 git，从本仓库 Release
`bfmzero-bdx-full-checkpoint` 下载：

```bash
gh release download bfmzero-bdx-full-checkpoint -R shangchaohu718/BDX-Tracker \
  -D /path/to/BFM-zero/results/bfmzero-bdx-full
# 资产: model.safetensors -> checkpoint/model/, optimizers.pth -> checkpoint/,
#        config.json -> 根目录
# 另一资产 lafan_29dof.zip (151MB) -> humanoidverse/data/lafan_29dof.zip
#   （LAFAN 源 mocap 数据，人体模仿管线输入；超 100MB 文件限制故走 Release）
```

还原后运行闭环 demo：
```bash
cd BFM-zero
BDX_PLANNER_DATA=$PWD/humanoidverse/data/bdx_planner_v2combo \
MUJOCO_GL=egl .venv/bin/python humanoidverse/scripts/human_imitation_closedloop.py \
  fightAndSports1_subject1_clip19 250
```
（需自备 `humanoidverse/data/` 下的机器人模型、LAFAN pkl 与动作数据，见各脚本头部说明。）

## 4. 诚实边界（沿用战役终裁口径）

- 正向 vx：completion-form validated / legacy-compatible / closed-loop not
  established in current runtime —— 禁写 "closed-loop verified"。
- 负向 vx：REJECTED_BY_ADAPTER（command_adapter_v13 全程在线）。
- 腿部模仿为 QUALITATIVE；tracker 训练池零舞蹈数据。
- 走路段视频中机器人周期性被拽回 = RUNTIME_X_AXIS_TERMINATION_LOOP（runtime
  集成问题，非策略失败；修复需独立任务 + 对称 forward/backward 复测）。
