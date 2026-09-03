# BFM-Zero G1 / BDX 安装与推理指南

BFM-Zero 是一个可提示（promptable）的人形机器人行为基础模型：单一策略通过不同的隐向量 **z** 条件化，即可执行多样动作（跟踪、目标到达、奖励任务）。本指南覆盖 **G1（29 自由度）** 与 **BDX（14 自由度，迪士尼双足机器人）** 两种机器人的安装、推理与免训练部署，均在本地机器（RTX 5080）完成，训练在远程 H20 上进行。

## 目录

- [安装](#安装)
- [数据准备](#数据准备)
- [推理（torch 路径）](#推理torch-路径)
- [ONNX 导出](#onnx-导出)
- [免 torch 部署运行时](#免-torch-部署运行时)
- [常用指标与注意事项](#常用指标与注意事项)

## 安装

```bash
cd bdx_BFMzero
uv sync          # 需要 Python 3.10+，自动安装 PyTorch/CUDA、MuJoCo、mujoco-warp 等全部依赖
```

MuJoCo 可视化需要 GPU 渲染后端。无头（offscreen）渲染默认 EGL；带实时窗口时改用 glfw：

```bash
MUJOCO_GL=glfw   # 带窗口可视化时
# 默认 EGL 用于 --headless / 视频导出
```

**无需 Isaac Sim**。所有推理均通过 `--simulator mujoco` 在纯 MuJoCo 后端运行。

## 数据准备

| 机器人 | 动作数据 | 说明 |
|---|---|---|
| G1 | `humanoidverse/data/lafan_29dof.pkl`（评测）/ `lafan_29dof_10s-clipped.pkl`（训练） | 仓库自带 zip，需手动解压：`unzip humanoidverse/data/lafan_29dof.zip -d humanoidverse/data/` |
| BDX | `humanoidverse/data/bdx_14dof.pkl`（评测）/ `bdx_14dof_clipped.pkl`（训练） | 由 `bdx_rl_mjlab` 的 NPZ 动作库经 `scripts/convert_bdx_npz_to_pkl.py` 转换生成（全局位姿 → 局部轴角 + FK 回验） |

检查点目录结构（以 `remote_checkpoint_750M/` 为例）：

```
remote_checkpoint_750M/
  config.json              # 训练配置（hydra_overrides 中含 robot= 条目）
  checkpoint/              # torch 权重（model/ + optimizers.pth）
  exported/                # ONNX 导出位置（首次推理时生成）
  tracking_inference/      # zs_<motion>__mujoco.pkl —— 每个 clip 的开环隐向量序列
```

## 推理（torch 路径）

主入口是 `humanoidverse/tracking_inference.py`（tyro CLI）。它在推理的同时完成三件事：导出 ONNX、按 clip 计算 z 序列（`backward_map` 后存 pkl）、滚动策略并输出统计。

### G1（750M 检查点）

```bash
# 带实时窗口
MUJOCO_GL=glfw uv run python -m humanoidverse.tracking_inference \
    --model_folder remote_checkpoint_750M \
    --motion_list 50 \
    --simulator mujoco --no-headless

# 无头 + 视频导出
uv run python -m humanoidverse.tracking_inference \
    --model_folder remote_checkpoint_750M \
    --motion_list 25 \
    --simulator mujoco --save_mp4
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model_folder` | 必填 | 检查点目录（含 `config.json` + `checkpoint/`） |
| `--motion_list` | `[25]` | 评测 clip id 列表（对应动作 pkl 内的索引） |
| `--simulator` | `isaacsim` | 推理请用 `mujoco` |
| `--episode_len` | 0（全长） | 滚动步数 |
| `--data_path` | — | 覆盖动作 pkl 路径 |
| `--save_mp4` | 关 | 导出专家 vs 策略对比视频 |

### BDX（100M / 700M 检查点）

```bash
# 需要指定 BDX 动作数据；BDX 是 14 自由度，脚本会自动识别
BFM_ZERO_METRICS_DOF=14 uv run python -m humanoidverse.tracking_inference \
    --model_folder <bdx_checkpoint_dir> \
    --data_path humanoidverse/data/bdx_14dof.pkl \
    --motion_list 0 \
    --simulator mujoco --headless
```

BDX @100M 基线（`episode_len 500`）：

| clip | mpjpe | 结论 |
|---|---|---|
| stand_seed0 | 5.3 cm | 稳定跟踪 |
| angry_no | 5.5 cm | 稳定跟踪 |
| walk_seed0 | 49 cm（min_root_h 0.08） | 塌倒 |
| walk_seed1 | 106 cm | 严重塌倒 |

准静态动作已学会、步态未收敛 —— 与 G1 的时间线一致（G1 跟踪能力约在 384M 步后才成熟）。

## ONNX 导出

`tracking_inference` 每次运行会自动导出 ONNX（actor + 观测归一化网络 baked in）到 `<model_folder>/exported/FBcprAuxModel.onnx`。也可用 `scripts/export_onnx.py` 单独导出。

导出维度（自动随机器人适配）：

| 机器人 | 输入 | 输出 |
|---|---|---|
| G1 | 721 = actor_obs 465 + z 256 | 29 |
| BDX | 368 = actor_obs 240 + z 128 | 14 |

导出后先做 parity 校验（ONNX 输出 vs torch `model.act`，在随机输入上应 < 0.05）。

## 免 torch 部署运行时

`humanoidverse/scripts/onnx_runtime_deploy.py` 是纯部署路径：**numpy + onnxruntime + mujoco + joblib + pyyaml，零 PyTorch 依赖**（已用系统 python3 无 torch 环境验证）。它在 numpy 中按训练环境语义逐帧组装观测，驱动 ONNX actor，并按与 env 完全一致的 P 控制律做物理仿真。

```bash
# G1 750M，实时窗口
MUJOCO_GL=glfw python3 humanoidverse/scripts/onnx_runtime_deploy.py \
    --model_folder remote_checkpoint_750M --motion 25 --viewer

# BDX 100M
python3 humanoidverse/scripts/onnx_runtime_deploy.py \
    --model_folder <bdx_checkpoint_dir> --motion 0

# 从 ground-truth dump 初始化（逐帧对齐 torch 路径做验证时用）
RT_TRACE=/tmp/rt.npz python3 humanoidverse/scripts/onnx_runtime_deploy.py \
    --model_folder remote_checkpoint_750M --motion 50 --episode_len 299 \
    --init_npz /tmp/obs_gt50_nodr.npz
```

机器人配置（dof 名称、默认角、kp/kd、力矩限制、场景 XML）从 `humanoidverse/config/robot/` 的 yaml 自动解析，按 `config.json` 的 `robot=` 条目或动作维度自动选择 G1/BDX。

该运行时已通过逐帧对齐验证：G1 motion 50 全程 299 步，root 高度与 torch 路径一致到千分位（0.796/0.711/0.765…），策略全程站立。与 torch 路径存在三个必须复刻的隐蔽差异（时间步覆盖、`action_rescale` 缩放、原始力矩限制），详见 `docs/deployment-runtime-notes.md`。

## 常用指标与注意事项

输出 `STATS` 字段：

| 字段 | 含义 |
|---|---|
| `mpjpe_mean` | 全程平均每刚体跟踪误差（米） |
| `min_root_h` | 全程最低根高（米）——物理塌倒信号 |
| `fell_frame` | 首次根高 < 0.3 m 的帧（**G1 尺度阈值**，BDX 站高仅 0.29 m，会误报；BDX 请看 min_root_h < ~0.15） |
| `lost_track_frame` | 首次触发动作远离终止的帧 |

注意事项：

- **域随机化**：推理脚本默认开启 DR，会导致同 clip 跑两次结果不同。做确定性评测/对比时务必 `--disable_dr`。
- **z 是开环的**：每个 clip 的 z 序列来自参考动作（`backward_map(ref_obs)`），推理时按帧索引回放；`z_mode` 参数是死代码，闭环 z 尚未实现。
- **segfault 退出码 139/134**：glfw atexit 析构竞态，发生在结果打印**之后**，加 `PYTHONUNBUFFERED=1` 即可拿到 STATS；不影响结果。
- **BDX 训练状态**：100M 已完成（2026-08-14）；+600M 扩展至 700M 于 2026-08-17 启动，完成后用同一套推理命令对比 100M 基线。
