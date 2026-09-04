# BFM-Zero / BDX 训练崩溃排查交接（2026-08-20）

## 0. 任务目标与当前结论

目标：定位 BDX 上 BFM-Zero（`FBcprAux`）训练早期 imitation reward 崩溃、后期 Q 约为 -185、actor loss 极大的根因，并设计最小因果实验闭环，而不是靠调参碰运气。

**【2026-08-20 晚间最终更新：根因两级定案】**

第一级（2×2 冻结因子实验，当天下午）：

> rollout reward 崩塌由 discriminator 参数更新唯一触发：actor 冻结仍逐点崩溃，D 冻结则完全不崩。

第二级（domain ablation，当天晚间，脚本 `humanoidverse/scripts/train_discriminator_domain_ab.py`，结果 `results/bfmzero-bdx-full/discriminator_domain_ab_v2.json`）：

> **Discriminator domain-separation collapse：D 的训练任务含不可满足成分。** 把 expert 运动的精确状态注入 MuJoCo、用 sim 观测管线读取（物理状态与 dataset expert 完全相同，只有观测生成管线不同），fresh D 仍然训练到 AUC=1.0000、margin≈14.1——与训练时 fresh-D 的 AUC≈1、margin≈14.5 签名一致。即：**即使 policy 完美复现 expert 运动，sim 管线观测仍会被判 fake。**

水印机制（已定位代码）：

1. **expert 刚体线速度仍被 sigma=2 高斯平滑**：`humanoidverse/utils/motion_lib/torch_humanoid_batch.py:233` 调 `_compute_velocity(wbody_pos, dt)` 未传 `guassian_filter=False`（该开关只救了角速度，线速度是无条件平滑的同类未修 bug），sim 侧是瞬时速度。落点：`privileged_state[151:202]` local_body_vel。
2. **水印是分布式的，不止一个通道**：通道交换归因显示 pos/rot/vel 任一通道换成对侧数值后 AUC 仍≥0.999；只有 local_body_ang_vel（唯一已修复帧语义+无平滑差异的通道）换掉后 AUC 降到 0.90。swap-all 对照（AUC 0.66，剩余在 state 通道）验证归因方法有效。**结论：逐通道修补不可能救活任务，必须让两类样本共享观测管线。**
3. **BatchNorm 放大器**：local_body_pos raw 残差 p50 仅 5e-4，但经 obs normalizer（BatchNorm running stats）后 p95 达 0.76、7% 条目 >0.5——低方差通道的微小系统残差被归一化放大成强判别特征。

原始假设链全部证伪/降级：scale_reg、relabel conditional mismatch、FB actor term、actor 更新本身、"actor 先把 occupancy 推坏"的整个框架。

当前无训练进程运行。

## 1. 仓库、环境、数据与入口

- 仓库：`/home/tcl/Desktop/start/BFM-zero`
- Python：`/home/tcl/Desktop/start/BFM-zero/.venv/bin/python`
- 机器人：BDX 14-DOF
- motion 数据：`/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/bdx_14dof_train.pkl`
- 训练入口：`humanoidverse/scripts/train_bfm_zero_bdx.py`
- 算法核心：`humanoidverse/agents/fb_cpr_aux/agent.py`
- 主训练循环：`humanoidverse/train.py`
- 关键旧完整训练：`results/bfmzero-bdx-full`（若目录仍在）

本地 PyTorch/TorchInductor 曾在 trajectory buffer `_get_idxs` 生成无效 C++，因此 BDX launcher 已设置 `BFM_DISABLE_TORCH_COMPILE=1`。不要移除，除非另做独立验证。

## 2. 已确认并修复的 observation correctness bugs

### 2.1 expert-only `local_body_ang_vel` smoothing

原先 expert motion angular velocity 使用 `gaussian_filter1d(sigma=2)`，policy 使用 simulator 瞬时角速度，语义不一致。

修复：

- `humanoidverse/config/robot/bdx/bdx_14dof.yaml`
  - `robot.motion.smooth_angular_velocity: false`
- `humanoidverse/utils/motion_lib/torch_humanoid_batch.py`
  - 从 runtime cfg 读取 `smooth_angular_velocity`

同状态验证：去掉 smoothing 后 angular velocity error 从约 `0.1785 rad/s` 降到 `0.0025 rad/s`（约 98.6%）。

### 2.2 expert `base_ang_vel` world frame vs policy body frame

原 expert 使用 world-frame root angular velocity，policy proprioception 使用 body/root-frame。

修复位置：

- `humanoidverse/agents/envs/humanoidverse_isaac.py`
- `humanoidverse/utils/helpers.py`
- `humanoidverse/agents/evaluations/humanoidverse_isaac.py`

核心修复是：

```python
ref_ang_vel = quat_rotate_inverse(
    base_quat, ref_body_angular_vels[:, 0], w_last=True
)
```

matched-state audit：mean abs error `0.1395`、p99 `1.3133`、max `3.18` 修复到约 `7.5e-9`。

### 2.3 重要结论

两个 bug 都必须保留修复，但 fresh-D 2×2（angular smooth/instant × base world/body）四组均 AUC≈1、margin≈14.5。因此它们不是训练 reward collapse 的主因。

相关脚本/结果：

- `humanoidverse/scripts/audit_matched_state_observations.py`
- `humanoidverse/scripts/train_discriminator_ab.py`
- `results/bfmzero-bdx-full/matched_state_observation_audit.json`
- `results/bfmzero-bdx-full/rigid_body_kinematics_audit.json`
- `results/bfmzero-bdx-full/rigid_body_kinematics_audit_unfiltered.json`

## 3. 已排除或降级的假设

### z-only shortcut：排除

旧 checkpoint 离线 probe：

| 输入 | D reward/logit |
|---|---:|
| Expert obs + paired expert z | +5.87 |
| Expert obs + shuffled z | +0.44 |
| Expert obs + zero z | -0.25 |
| Policy obs + expert z | -4.65 |
| Policy obs + policy-goal z | -1.47 |
| Policy obs + random z | -6.15 |

给 policy obs 换 expert z 不能伪装成 expert，故 `D(o,z)≈D(z)` 不成立。D 会使用 obs-z compatibility，但不是单纯用 z 当 domain label。

结果：`results/bfmzero-bdx-full/z_shortcut_feature_audit.json`

### D label contamination：排除

D 训练发生在 relabel 之前，fake 输入使用 replay 原始 rollout z。不是把 relabeled z 误标为 fake 的常规 GAN bug。

### critic divergence：排除为第一原因

旧 100M 中 `Q prediction≈Q target≈-185`。若每步 reward≈-10、discount≈0.95，则长期 Q≈`-10/(1-.95)=-200`，critic 是在忠实拟合死亡 reward，不是自己发散。

### `scale_reg`：已基本排除为第一推动力

见第 6 节 Run A/B。关闭 scale 后前 50k reward collapse 几乎逐点复现。

### 【新】relabel conditional mismatch、FB actor term、actor 更新本身：全部排除

见第 6 节新增 2×2 实验。崩溃由 D 训练唯一决定。

### 【新·晚间】domain ablation：D 任务不可满足已确认；"D 太强/需要调参"定性为错误框架

`train_discriminator_ab.py` 的 AUC≈1 曾被解读为"policy 太差"。domain ablation（B arm）证明：物理状态完全匹配 expert 时（同一运动注入 MuJoCo 读 obs），fresh D 仍 AUC=1.0000/margin≈14.1。旧 fresh-D 的高分主要来自 domain 残差，不是 behavior gap。"调 D 强度"（lr/GP/batch）不在修复路径上。

## 4. 算法静态审计：真正的结构风险

Replay transition `(o,a,o')` 的 action 由 rollout latent 产生：

```text
a ~ pi(a | o, z_rollout)
```

但 update 中在 D 训练后，以 `relabel_ratio=0.8` 替换 z。最终 train z 组成约为：

- 20% rollout z
- 48% shuffled expert z
- 16% policy-goal z
- 16% random z

同一个 relabeled z 会一致地进入当前 reward、critic 和 next-state bootstrap，所以 Bellman 链内部并非 `z_t != z_{t+1}`。真正风险是：

```text
(o, a generated under z1, o') is trained/evaluated under z2
```

普通 HER/UVFA 可以这样做，但这里 reward 是 learned conditional discriminator `r_D(o,z)`，且 FB critic/actor 会在人工 `(o,z)` 联合分布上外推，风险高于显式 goal reward。

Actor loss：

```python
L_actor = -0.05 * w * Q_D - 0.02 * w * Q_aux - Q_FB
w = mean(abs(Q_FB)).detach()  # scale_reg=True
Q_FB = F(o,a,z)^T z
```

`F` 未归一化；z 和 B norm 固定约 `sqrt(128)=11.3137`。旧训练 F norm 约 15→134、`w` 约 10→419。这个尺度链无显式上界：

```text
FB representation scale -> |Q_FB| -> w -> D/A actor coefficients
```

但它是放大器，不是已证实的起火源。

## 5. 已加入的 observe-only diagnostics

文件：`humanoidverse/agents/fb_cpr_aux/agent.py`

config flags：

- `diag_reward_sources`
- `diag_actor_update_ratio`

记录：

- reward under rollout / used / policy-goal / expert / random z
- cosine(used z, rollout z) 与真实 relabel fraction
- actor loss 的 D/Aux/FB 三项
- actor grad pre/post clip
- effective D coefficient
- actor parameter update ratio
- deterministic actor mean 更新前/后：`Q_FB_before/after/delta`
- deterministic mean action delta
- 同 batch `corr(Q_FB, D reward)`
- Q_FB top/bottom 10% 对应的 D reward 与差值

诊断中的 random z 会保存/恢复 CPU/CUDA RNG state，不改变训练随机流。

注意：`scale_reg=False` 时 weight 必须是 tensor；已修为 `Q_fb.new_ones(())`，否则 trainer 聚合 metrics 会报 `'float' object has no attribute float'`。

先运行：

```bash
.venv/bin/python -m py_compile humanoidverse/agents/fb_cpr_aux/agent.py
```

## 6. 已完成/部分完成的关键因果实验

### Run A：baseline，完整 2M

目录：`results/bfm-dynamics-baseline`

设置：seed 4728、relabel=.8、scale_reg=True、clip=None、diagnostics on、eval off。

checkpoint：

- `results/bfm-dynamics-baseline/checkpoint/train_status.json` = `time: 2000128`
- optimizer state 也已保存。

关键早期时间线：

| step | rollout reward | used-z reward | F norm | Qfb abs | actor grad | Δθ/θ | D loss项 | FB loss项 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | +0.283 | +0.402 | 10.75 | 10.78 | 1.31 | .001299 | +0.12 | -4.56 |
| 16k | -0.115 | +0.143 | 10.18 | 11.94 | 1.04 | .000950 | +0.19 | -7.92 |
| 33k | -1.311 | -0.712 | 15.38 | 17.45 | 2.04 | .001129 | +1.56 | -13.95 |
| 49k | -2.194 | -1.498 | 21.16 | 23.20 | 2.96 | .001306 | +4.71 | -19.61 |
| 98k | -3.772 | -3.133 | 33.13 | 45.49 | 4.49 | .001284 | +30.69 | -43.91 |

D loss component 首次超过 FB component 约在 139,264 steps，但 reward 在 49k 前已明显崩。因此 D/scale dominance 的时间顺序晚于事故启动。

### Run B：scale_reg=False，部分到 819,200

目录：`results/bfm-dynamics-scale-off`

用户下班时手动 Ctrl-C；不是 crash。无正式 checkpoint，但 `train_log.txt` 完整保留到 819,200。

最重要对照：

- Run A 49k rollout reward ≈ `-2.19`
- Run B 49k rollout reward ≈ `-2.19`

Run B 的 effective D coeff 恒为 `0.05`、scale weight 恒为 `1.0`，但早期 collapse 几乎逐点复现。因此 `scale_reg` 基本洗清为第一推动力。

Run B 早期还显示：

- 每次 actor step 后 deterministic `Q_FB` 增加（`delta>0`）
- 同 batch `corr(Q_FB,D reward)>0`
- Q_FB top10% 的 D reward 高于 bottom10%

所以不能写成“FB 高分样本立刻被 D 判得更差”。更精确：actor 在 replay state 上提高 FB proxy，且局部 proxy 与 D 尚一致，但后续真实 rollout occupancy 仍快速恶化。这可能是 transition/temporal distribution shift 或 critic-action extrapolation，而非简单单帧负相关。

### Run C：clip=1

尚未运行。目录可能不存在或为空：`results/bfm-dynamics-clip1`。

由于 Adam 下 raw grad 从约 1→几十时，参数更新比例仍约 0.13%→0.08%，clip 的信息优先级已低于 relabel ablation。

### 【新·晚间】domain ablation：D 任务不可满足的直接证据

脚本 `humanoidverse/scripts/train_discriminator_domain_ab.py`，结果 `results/bfmzero-bdx-full/discriminator_domain_ab_v2.json`（4096 样本、fresh D、lr 1e-5、2000 步、z 两侧相同、obs noise 开启）。

三个 arm：

| Arm | 正类 | 负类 | AUC@500 | AUC@2000 | margin@2000 |
|---|---|---|---:|---:|---:|
| B domain | dataset expert obs | 同一运动状态注入 MuJoCo 的 sim obs | 0.990 | **1.0000** | 14.13 |
| C behavior | expert-sim obs（sim 管线） | 旧 policy rollout obs（sim 管线） | 0.996 | 1.0000 | 14.59 |
| （旧 fresh-D） | dataset expert | 旧 policy rollout | — | ≈1 | ≈14.5 |

B arm 的意义：物理状态完全相同、仅观测生成管线不同 → AUC=1。C arm 双方同为 sim 管线仍可分，但那是旧 policy 真实行为差距（预期内）；关键证据是 B——完美行为也无法通过判别。

通道归因（用 B-arm 训好的 D，交换 privileged_state 单通道）：

| 交换 | AUC |
|---|---:|
| ALL（对照，应→0.5） | 0.659（剩余在 state 通道） |
| root_height | 1.0000 |
| local_body_pos | 0.9997 |
| local_body_rot | 0.9991 |
| local_body_vel | 1.0000 |
| local_body_ang_vel | **0.9034** |

水印分布式存在；唯一已修语义的 ang_vel 携带信号最少（换掉后 AUC 最低），侧证归因有效。

配对残差分解（local_body_pos）：raw p50=5.2e-4 / p95=0.013；归一化后 p50=0.032 / p95=0.762、7.1% 条目>0.5。**BatchNorm running stats 是残差放大器。**

注入机制（复用说明）：`reset_envs_idx(target_states={dof_states, root_states})` 设 `need_to_refresh`，下一次 `step()` 内先物理步进、再应用注入、再计算观测——观测恰在注入状态上、无动力学作用。z 用 expert 侧编码（两侧同 z，z 不携带标签）。

### 【新】2026-08-20 下午完成的 2×2 因果实验（第一级定案）

四个 200k 新 run，seed/数据/超参与 Run A 一致，仅改单变量。判定指标为 49152 步 rollout reward（Run A = -2.19）：

| Run | 目录 | actor | D | 49k reward | 200k 走势 |
|---|---|---|---|---:|---|
| relabel=0 | `results/bfm-dynamics-relabel0-200k` | 训练 | 训练 | -2.21 | -4.86，逐点复现 baseline |
| actor_fb=0 | `results/bfm-dynamics-actorfb0-200k` | 训练(FB项移除) | 训练 | -2.30 | -4.1x，同样崩溃 |
| actor 冻结 | `results/bfm-dynamics-actorfrozen-200k` | **lr=0** | 训练 | -2.19 | **-5.02，崩溃照旧（甚至更差）** |
| D+actor 双冻结 | `results/bfm-dynamics-d-and-actor-frozen-200k` | **lr=0** | **lr=0** | **+0.34** | **+0.31，全程平坦** |
| 仅 D 冻结 | `results/bfm-dynamics-dfrozen-only-200k` | 训练 | **lr=0** | **+0.30** | **+0.26，全程平坦** |

冻结验证：actor-frozen run 每行 `actor_param_update_ratio=0.000000`、`diag/actor_mean_action_delta=0.000000`；D-frozen run 的 `disc_train_loss` 仅随输入分布缓慢漂移（权重不动）。

结论（2×2 完整因果分解）：

1. **relabel mismatch 降级排除**：relabel=0 崩溃逐点复现。它仍可能是放大器，但不是启动原因。
2. **FB actor objective 降级排除**：去掉 FB 项（actor 仅剩 0.05·w·Q_D + 0.02·w·Q_aux）同样崩溃。注意 Adam 尺度不变性把 100 倍梯度差压缩成约 2 倍动作移动差（fb0 actΔ≈0.005-0.026 vs baseline ≈0.014-0.024，upd% 均≈0.1%），"不同 objective、相同崩溃"由此而来。
3. **actor 更新整体排除**：权重完全不动仍崩溃。
4. **D 训练是唯一必要因子**：D 冻结的两格全部平坦；且在双冻结 run 中行为指标仍在漂移（actrate 230→161、limits 0.18→0.67，z 条件分布随 B 训练漂移所致），但固定 D 下 reward 不降——行为漂移对 reward 崩塌贡献≈0。

Step-1 日志分析补充（脚本：`humanoidverse/scripts/analyze_bfm_dynamics_runs.py`）：

- Run A/B 崩溃时点完全相同（32768 首破 -1，49152 首破 -2）。
- Run B 中 `diag/Q_fb_actor_mean_delta>0` 占 100% 日志行（actor 每次 update 都在 replay 状态上提高 FB proxy），同批次 `corr(Q_FB, D reward)≈0.5`，top10% 的 D reward 比 bottom10% 高 0.7→3.0——FB proxy 与 D 在批次内局部一致，但这不构成因果。
- `used-z reward - rollout-z reward ≈ +0.65~0.9` 恒为正：relabel 使训练信号在 relabeled z 下显得更好，而真实 rollout 仍被判更差（mismatch 签名），但实验 1 证明这不是崩塌原因。

## 7. 下一位 agent 应该如何继续（按优先级，2026-08-20 晚间二次修订）

### 新增 launcher 开关（默认值 = Run A 行为，均已 py_compile 验证）

- `BDX_ACTOR_FB_COEFF`（默认 1.0）：actor loss 中 FB 项系数，`actor_fb_coeff`
- `BDX_LR_ACTOR`（默认 0.0003）：actor 学习率（0 = 冻结）
- `BDX_LR_DISCRIMINATOR`（默认 1e-05）：D 学习率（0 = 冻结）

改动位置：`humanoidverse/scripts/train_bfm_zero_bdx.py` 与 `humanoidverse/agents/fb_cpr_aux/agent.py`（`actor_fb_coeff` 字段 + `actor_loss_fb = -coeff * Q_fb.mean()`）。

### Step 0（新最高优先）：让 D 的两类样本共享观测管线 —— 实现进行中，见下方实施日志

原则：D 只应判 behavior，不应判 obs 来源。实现选项（按推荐顺序）：

1. **expert-sim pool（正在实施）**：离线把 expert 状态注入 MuJoCo 读 obs，构建 TrajectoryDictBuffer 替换 expert_slicer。已完成：生成器 `humanoidverse/scripts/generate_expert_sim_pool.py`（warp/cpu 双后端）、`train.py` 挂载点（环境变量 `BFM_EXPERT_SIM_POOL`，未设置=原行为）、验证器 `humanoidverse/scripts/verify_expert_sim_pool.py`。
2. 顺带修线速度平滑水印（ hygiene，单独修不解决问题）：`torch_humanoid_batch.py:233` 加 `guassian_filter=self.cfg.get("smooth_angular_velocity", True)`。
3. 管线统一后 obs noise 自然对称（同一 builder 加同分布噪声）。

#### Step 0 实施日志（2026-08-20 晚，重要勘验结论）

**注入读取的三层坑（按发现顺序）：**

1. **单次注入的 warm-start 残差**：注入后 step() 读 obs，位置通道首次读取带 ~8e-3 残差（重复注入收敛到 5e-5）。探针：`probe_warp_injection_staleness.py`。对策：settle=2（同态注入两次，读第二次）。
2. **warp 的 `mjw.forward` 不重算 `d.cvel` 的线速度部分**：角速度精确（max=0.0），线速度 62% 项在两次相同读取间差异 >0.01、极值 ~11 m/s（= 上一次物理步的陈旧值）。`mjw.step1` 也只部分重算（41% 项 >0.01）。物理步会全量重算 → **policy 观测不受影响，只有注入读取受害**。对策：弃用 warp 精确读取。
3. **CPU `mj_forward` 完整正确**：pos/rot/height 通道读取完全确定（max=0.0），速度通道有首读残差（read2≡read3 精确），settle=2 消除。**最终方案 = CPU 精确读取器 `inject_read_exact_lean`**（写状态→手动 refresh→`mj_forward`→`_pre_compute_observations_callback`→`_compute_observations`→`_get_g1env_observation`，与 policy 观测同代码路径，仅状态来源不同）。

**Gate-1 的两个测试学理（防止伪阳性/伪阴性）：**

- 固定行训练的 fresh D 会**记忆行级噪声**（train AUC 0.99 但 holdout 0.5）；自一致性检验必须用**不相交帧的 holdout + 新鲜重生成负类**。
- holdout 负类若与训练负类同帧生成（近重复），记忆会泄漏到 holdout——必须 disjoint frames。

**warp pool（v2，settle=2）Gate-1 结果**：纯度提升 55 倍（p50 1.5e-4）但 disjoint holdout AUC=0.79——速度通道的 pass 级签名仍在（归因：swap privileged_state 使 AUC 反转 0.79→0.24）。故弃用，转 CPU。

**CPU pool**：17.3 fps 单 env；1500 chunks × 64 帧 = 96,000 帧 ≈ 92 分钟（`results/pool_cpu_gen.log`，输出 `humanoidverse/data/bdx_expert_sim_pool_cpu.pt`）。规模注记：原 kinematic expert buffer 1.16M 帧，pool 96k——D 对正类复用率升高（200k 步约 130 epochs），若 Gate-2 出现 D 过拟合迹象（正类 logit 饱和但行为不改善），先扩 pool 再怀疑方法。

**CPU env 注意**：`num_envs>1` 有 init shape bug 必须 build(1)；env 自身 motion lib 只载 256 条采样，注入前必须先 `load_expert_trajectories_from_motion_lib`（触发全量 1462 条重载），否则 motion_id 越界 CUDA assert。

**【Gate-1 结果：通过】**（`results/bfmzero-bdx-full/expert_sim_pool_gate1_cpu.json`）：

- 纯度（pool vs 独立重生成，local_body_pos raw）：**0.0 逐位相同**
- pool-vs-regen 不相交帧 holdout：**AUC 0.4931 ≈ 0.5**（无 pass 签名）
- 对照 regen1-vs-regen2 holdout：0.4924 ✓
- domain_B（kinematic vs sim regen）对照：AUC 0.9995 / margin 10.6——检验有功效，0.5 是真阴性

**交叉后端注意**（`expert_sim_pool_gate1_cross.json`）：CPU pool vs **warp 注入读取** AUC 0.9988——warp 注入读取的速度通道仍是坏的，不能作为 policy 观测的代表；真正训练对比（CPU pool vs warp 物理演化观测）只能由 Gate-2 实测。

**Pool episode 长度硬约束**：`_sample_tracking_z` 以 `rollout_expert_trajectories_length=250` 采样，episode 必须 ≥251 帧；64 帧 chunks 会 `multinomial sum<=0` 崩溃。生产 pool 用 chunk_len=264。

**【Gate-2 结果（200k，`results/bfm-dynamics-pool-200k`）+ Step1 叠加（`results/bfm-dynamics-pool-clip1-200k`）】**

1. **pool 单独**：raw rollout reward 与 baseline 逐点相同（49k -2.19，196k -4.56）；但 D loss 形态改变——disc_expert/disc_train 稳定在 0.03/0.05（margin ~7 logits），不再是旧的无界饱和。结合 Gate-1（pool 无域签名），解释：domain 修复生效，但**弱初始 policy 的行为级可分性同样驱动 BCE logit 无界增长**——raw-logit reward 的饱和机制在"任何可分对"上都会发生（正是 Step 1 的适用前提）。
2. **pool + reward_clip=1**（新开关 `BDX_REWARD_CLIP`→`FBcprAgentTrainConfig.reward_clip`，只 clip critic 训练用 reward，诊断保持 raw）：Q1 从 -200 吸引子轨迹变为 196k 的 -10.4 且减速；critic 训练 reward 在 98k 后稳定在 ≈-0.93。**reward 侧病理已消除**。
3. **遗留瓶颈（下一阶段）**：actor 的 raw D 评分轨迹与 baseline 完全相同——行为在 200k 内无改善。嫌疑：(a) aux penalty（mean_aux≈-1.8/step）主导 total reward，clip 后 disc 信号（±1）占比更小；(b) |Q_FB| 仍增长到 ~119，actor 目标仍由 FB proxy 支配；(c) 需要更长训练（上游成功训练是 100M 步）。建议顺序：先跑 pool+clip 的 1M 看趋势斜率（raw D reward 是否随时间改善），再考虑 actor 侧 rebalance（FB coeff、aux scaling）。

Gate-2 判定的诚实结论：原判定标准（49k 不复现 -2.19）以 raw reward 计**未通过**，但该指标本身测的是 D 的 raw logit（无界 by construction）；以"训练用 reward 是否稳定有界"计**通过**。行为改善的验收要移到更长时程。

**【1M 结果（`results/bfm-dynamics-pool-clip1-1m`）——首次出现持续行为改善】**

四类指标（新增 `diag/reward_{p05,p25,p50,p75,p95,std,frac_at_lower_clip}`，开关 `BDX_DIAG_REWARD_STATS`）：

1. **reward 分布**：p50 自 49k 起贴 -1，贴下 clip 79~95%——确有有界饱和成分；但 std 0.27~0.55、p95 波动于 -0.67~+0.91：**梯度未死，只在尾部 ~10% 样本存活**。若要更宽的梯度带，试 reward_clip=2~3。
2. **行为代理**：245k 后 aux 惩罚单调大幅改善（贴限位 0.61→0.18、action_rate 63→10、力矩 5954→1910）；raw D reward 从 -4.93 回升至 ≈-4.06 并稳定。此前所有 run 这些指标只恶化。
3. **actor 平衡**：FB:D 量级比 35:1 → **1.2:1**（收敛），scale_reg 的自适应系数有效平衡了三项；Aux 相对份额下降（7.6→0.42）。
4. **FB 尺度**：|Q_FB| 10.7→423 仍无界增长（与旧轨迹同），但行为在改善——"300+ 且行为不改善才 rebalance"只满足一半。Q1 有界 ≈-17。

**结论：domain 修复 + reward 有界化 = reward 病理链切断，学习已启动**。下一阶段问题：梯度带宽（clip 松紧）、FB 尺度无界、以及 raw D reward 能否随行为改善持续上行至 expert 邻域。

**3M 延伸跑**（`results/bfm-dynamics-pool-clip1-3m`，nohup pid 见 `results/poolclip3m_console.log`，约 45 分钟）：关注 1M 后 aux 代理是否继续改善、reward 分布是否展宽、|Q_FB| 斜率。注意 `checkpoint_every_steps=2_000_000` 硬编码——1M run 无 checkpoint；3M run 在 2M 处有 checkpoint。若需断点续跑，先加 launcher 开关。

**【3M 判读 + clip 1/2/3 A/B 定案（2026-08-20 深夜）】**

3M（clip=1）三问结果：

1. **行为**：aux 代理继续"改善"到荒谬极值（贴限位 0.007、act_rate 1.7、力矩 35）——**这是瘫软/冻结不是平滑动作**；raw D reward 1M 回升到 -4.12 后再度恶化至 -6.09。机制：reward 饱和区无梯度 → aux（奖励不动）接管 policy → 冻结 → D 更确信 fake → 饱和质量更大——有界饱和锁死的恶性循环。
2. **reward 分布**：不但未展宽反而锁死——贴 clip 91%→97.3%，带宽 p95-p05 从 2.0（409k 峰值）→0.0（2M 起），unclip 仅 2.7%。
3. **|Q_FB|**：400→峰值 ~500（1.6M）→回落 425，**增速转负=尺度稳定非失控**；actor 平衡 1.01:1:0.62。

clip 1/2/3 各 1M A/B（`bfm-dynamics-pool-clip{1,2,3}-1m`，同 seed 同配置单变量）：

| @999k | unclip% | 带宽 p95-p05 | raw D | lim | actR | Q1 |
|---|---:|---:|---:|---:|---:|---:|
| clip=1 | 9.4% | 0.82 | -4.06 | 0.181 | 10.1 | -17.0 |
| clip=2 | 16.3% | 1.72 | -4.06 | 0.144 | 10.5 | -33.0 |
| clip=3 | 25.8% | 2.60 | -4.17 | 0.123 | 10.6 | -47.0 |

**结论：放宽 clip 成比例增加排序带宽，但不改变学习轨迹；且任何固定 clip 的带宽都随时间收窄**（三者均在 409k 峰值后单调收窄）——BCE margin 持续增长使低于 -c 的样本比例单调上升。固定 clip 是延迟战术。

**下一刀（设计已定，待实施）：batch-standardized reward**——`r = (logit - batch_mean)/batch_std`（update_critic 内 no_grad 逐 batch 计算，可选再接软 clip）。性质：保留排序、margin 增长在归一化中抵消、D 与 aux 的尺度关系自动稳定（D 归一到单位方差 vs aux≈-0.8）。实施点：`fb_cpr/agent.py update_critic` 与 `reward_clip` 同位置加 `reward_norm` 开关。判定：unclip/带宽不再随时间收窄 + aux 代理不再走向冻结极值 + raw D 不再二次恶化。若标准化后仍冻结，则嫌疑转向 aux 设计本身（slippage/action_rate 对静止的全奖励）。

**【batch-standardized reward 结果（1M + 3M，`bfm-dynamics-pool-rnorm-{1m,3m}`，开关 `BDX_REWARD_NORM=1`+安全阀 `BDX_REWARD_CLIP=5`）——四判据定案】**

| 判据 | 结果 |
|---|---|
| 1 带宽稳定 | **完美**：std=1.000、p95-p05 3.1~3.3 全程稳定（clip1 同期 2.0→0.0）、unclip 100% 至 3M；Q1 零中心 ±4（相对 reward 语义，勿与历史 Q 横比） |
| 2 D 置信与 RL 信号解耦 | **成立**：raw D 降至 -5.5 而 standardized 分布不动 |
| 3 aux 不推向静止 | **失败——机器人仍冻结**：3M 时 lim 0.003 / actR 1.9 / 力矩 30，与 clip1 的冻结极值几乎相同（rnorm 甚至略更冻） |
| 4 actor 平衡 | FB:D 从 6.4 漂回 11.8:1（Q_D 小尺度化后 FB 相对更大；次要问题） |

**正式结论：在 domain 统一 + 排序保留 + 带宽稳定三条件全部成立下，policy 仍收敛到静止解 → 根因升级为 aux objective 的静止解/reward hacking。** aux 六项（slippage -2.0、limits_dof_pos -10.0、ankle_roll -4.0、undesired_contact -1.0、feet_ori -0.4、action_rate -0.1）全部只罚运动、无一罚静止；mean_aux 3M 时 ≈-0.75 接近静止最优。D 的 standardized 信号是零均值相对排序——在 policy 尚未找到更高排名的运动状态时，唯一稳定梯度方向就是 aux 的静止最优。

**下一阶段（顺序）**：
1. **aux 全局缩放单变量**：`reg_coeff_aux` 0.02→0.002（或 aux_rewards_scaling 全乘 0.1），pool+rnorm 不变，1M。判据：aux 代理不再趋向冻结极值、raw D 走势、是否出现新的运动崩坏（aux 太弱时可能撞限位/打滑——那就要逐项审）。
2. 若 1 无效或矫枉过正：**逐项 aux 审计**（静止解敏感度排序：slippage/action_rate 对 a≈const 的梯度 vs 对运动状态的惩罚），或加 alive-motion 项（罚关节速度长期为零）。
3. 中期：FB:D 漂移（11.8:1）在 aux 问题解决后再评估。

**【aux=0.002 1M 结果（`bfm-dynamics-pool-rnorm-aux002-1m`，开关 `BDX_REG_COEFF_AUX`；新增 motion 诊断 `BDX_DIAG_MOTION`→`diag/mean_abs_action`/`diag/mean_abs_dof_vel_norm`）】**

1. motion activity：|a| 0.86→0.30、|dq| 0.83→0.61 仍单调下滑但明显变慢（actR@1M=53 vs aux0.02 的 10.7≈其 600k 水平）。**偏情况 B：更慢地走向冻结，而非开始运动**；1M 时程未完全排除"延迟冻结"。
2. imitation：rawD 略差（-4.95 vs -4.36）——减弱 aux 未换来模仿改善；带宽 3.29 保持。
3. **关键发现：六项 aux 中四项全程恒零**（slippage/feet_ori/ankle_roll/undesired_contact 两 run 均 0.000）——**aux objective 实际只有 limits_dof_pos + action_rate 在起作用**，静止盆地由这两项+aux normalizer 构成。逐项审计对象缩到两项。
4. actor：Aux 占比 0.4-0.45，FB:D 7.4:1。

**更新后的下一阶段（重要：出现新假设）**：aux002 显示即使把唯一的运动惩罚项（action_rate）削弱 10 倍，activity 仍下滑且 imitation 不升——静止解可能不只是 aux 造成：**数据集含大量近静态 clip（v1_stand/stand_sweep 等），冻结/站立状态在 D 的相对排序中可能本身就排中上游**，即 standardized D 信号对"静止"不敏感。检验（便宜）：用 3M checkpoint 的 D 对 (a) 静止状态 obs（零速注入站立）与 (b) 运动中的 expert pool obs 打分对比；若静止分数不低，则 reward 设计需要 anti-stillness 成分（或数据重加权），而不是继续调 aux 系数。次选：aux002 延伸 3M 确认"延迟冻结 vs 不冻结"。

### Step 1（次优先，稳定性）：reward 有界化

注意：clip logit 只是把 Q 的 -200 吸引子变成 -40 量级常数，是稳定性 ablation 而非根因修复。仅在 Step 0 之后仍有尺度问题时做。

### Step 2：若 Step 0 后 D 仍过快饱和

同管线下的 behavior 判别仍可能很快（policy 最初确实差）。可再考虑：expert-sim 混入比例、D lr/更新频率、GP。这些现在才是合法的调参对象，因为任务已可满足。

### Step 3：结题最小验收（不变）

修复方向确认后 1–3M 重训：50k 不复现 collapse、2M reward 不单调进入 -5~-10、occupancy/MPJPE 不恶化、Q 不收敛到 -10/(1-gamma) 常数、同 seed/数据/步数单变量 A/B。

### 已废弃的方向（不要再跑）

- relabel ablation（已完成，排除）
- FB actor term ablation（已完成，排除）
- scale_reg ablation（Run B 已完成，排除）
- 逐通道修 watermark（归因显示分布式，修不完；管线统一一步覆盖全部）
- clip=1 Run C：actor 更新与崩塌无关
- 任何"调 actor 目标/系数"的方案

## 8. 不要做的事

- 不要回滚已确认的 angular smoothing 与 base frame correctness 修复。
- 不要再声称 z-only shortcut；已被反事实 probe 排除。
- 不要再跑 actor 侧 ablation（relabel/FB term/scale_reg/actor 冻结均已完成并排除）。
- 不要把"D 太强"当框架去调 D lr/GP/batch：domain ablation 证明任务是判来源而非行为，调强度只改饱和速度。
- 不要逐通道修 watermark（换掉单一通道 AUC 仍≥0.999；ang_vel 之外 pos/rot/vel 都携带信号）——管线统一一步覆盖。
- 不要把 reward clip 当根因修复：只改 Q 常数的数值，不恢复梯度。
- 不要把 `used-z reward > rollout-z reward` 解读成 relabel 无害；它只说明 relabel 不直接降低当前 D reward。
- 不要把 `corr(Q_FB,D reward)>0` 解读为 FB objective 正确；它是当前 replay batch 的局部关系，不等价于更新后的真实 trajectory improvement。
- 不要伪造 expert action distance：motion 数据没有与 policy actuator command 语义等价的 expert action。
- 不要在现有 dirty worktree 做 reset/checkout；有大量用户/历史改动混在一起。

## 9. 工程状态与风险

- worktree 很脏，很多改动并非本轮独占，必须逐文件保留。本轮新增文件均为 additive：`humanoidverse/scripts/analyze_bfm_dynamics_runs.py`（只读分析）、`humanoidverse/scripts/train_discriminator_domain_ab.py`（domain ablation）、launcher/agent 的三个默认值不变的开关。
- `git status` 仍报：`fatal: unable to read effe47d289896ca1f6bd69292caca5ec2fb40493`；`git fsck` 显示多个 tree→blob 链接损坏（不止一个 object）。当前策略：完全不碰 git（不 add/commit/checkout/gc），工作区文件是唯一事实。修复需用户决策。
- 进程托管：2026-08-20 的 5 个 200k run 前两个用 ZCode 后台任务托管，后三个用 `nohup ... &` 均正常存活至完成（旧警告未复现，但仍不保证）。200k run 全程约 3 分钟（~1100 FPS）。
- domain ablation 脚本注意：repo 的 mujoco env 在 `num_envs>1` 时有 buffer shape 初始化 bug（gravity_vec [N,3] vs base_quat [1,4]），必须 `build(1)`；obs normalizer 接收 obs dict 而非逐 tensor（tree_map 用法见脚本）；单 env 注入 4096 状态约 4 分钟。
- 两个旧 viewer 进程（tracking_inference，各约 0.5 GiB）仍在 GPU 上，不影响训练。
- Run B 第一次启动曾因 `scale_reg=False` 返回 Python float metric 而失败；已修。

## 10. 当前最可信事故模型（2026-08-20 晚间最终版，经实验确认）

```text
expert 正样本 = 运动学管线 obs（含 sigma=2 线速度平滑、FK/插值残差）
fake 负样本   = 仿真管线 obs（瞬时速度、mj 状态读数）
        |
        v
两类样本携带稳定 domain 水印（分布式：pos/rot/vel 均有；
BatchNorm running stats 把 raw~5e-4 的残差放大到 norm p95≈0.8）
        |
        v
D 的最优解 = 永远赢：判"来源"而非"行为"
（matched-state 注入实验：AUC=1.0 即使物理状态完全相同）
        |
        v
D 训练 → logit 饱和：r_D(rollout) → ~-10/step
（与 actor 行为/更新无关 —— actor 冻结逐点复现；D 冻结完全不发生）
        |
        v
critic 忠实拟合：Q → -10/(1-0.95) = -200（旧训练 Q≈-185 吸引子）
        |
        v
FB 表征尺度同步增长 → scale_reg 放大 actor 系数（后期放大器）
        |
        v
actor 优化一个结构性"全盘皆输"的 reward，无有效梯度方向
```

修复主轴：让 D 两类样本共享观测生成管线（expert 经 sim 注入），使任务回到可满足的 behavior 判别。

次要独立发现（与崩塌无因果但值得知晓）：

- z 条件分布随 B encoder 训练漂移（z_buffer 每 100 步刷新），冻结权重的 actor 行为也会因此改变（actrate 230→161）；固定 D 下该漂移 reward 中性。
- Adam 尺度不变性使不同 actor objective 的参数移动速率几乎相同（≈0.1%/update），解释所有 objective ablation 崩溃逐点重合。


## 11. P0 离线审计结果（2026-08-21，外部评审后执行；脚本 `audit_pool_prior.py` / `audit_dz_conditioning.py`）

**主线正式转向 z-conditioning。**

P0.4 pool prior 审计（`results/bfmzero-bdx-full/pool_prior_audit.json`）：
- **coverage 严重不足：pool 仅覆盖 300/1462 条运动（20.5%）**——D 正类只见过 1/5 的运动种类
- stand 帧占比 22.2%（数据集）→31.2%（pool），activity 均值 3.50→3.31（静态过采样温和）
- 短 clip 排除效应小（50 条 <266 帧，0.8% 帧份额）

P0.5 aux 接线检查：
- **ankle_roll(-4.0) 是死代码**：BDX `ankle_dof_names` 单元素，reward 取 `[1:2]` 空切片（G1 遗留索引伪影）
- slippage/feet_ori 接触门控（冻结合法为零，正常步态也近零——给静止免费通行证）；undesired_contact 静止合法为零
- 有效 aux = limits_dof_pos + action_rate 两项，确认

P0.1+2 D×z 条件辨识（rnorm-3M checkpoint 的 D，64 窗分层 stand/dynamic；`dz_conditioning_audit.json`）：
- **z 条件判别接近失效**：diag−offdiag=0.21，top-1 retrieval 9.4%（随机 1.6%）；D 对 z 有响应（col/row var≈5）但无法配对
- **D 系统性偏好静态观测**：stand obs（任何 z）-0.52~-0.57 vs dynamic obs -1.19~-1.33（差 0.6-0.8 logit）
- **条件静止测试**：冻结站立在 dynamic-z 下也打 -0.76，与真运动帧配对 z（-0.67）几乎持平——**"该 z 下该不该动"的判别不存在**

**结论：静止解是当前 conditional D 打分下的近似最优行为（不是 aux 的意外产物，aux 是放大器）。**

下一阶段（按外部评审 P2 框架）：
- (B) pool 重建至全 1462 覆盖 + clip/activity-balanced 采样（共同前置，CPU 生成器已支持，扩 chunks 数即可；按 clip 均衡需改采样器）
- (C) mismatched (stand obs, dynamic z) hard negatives 修 D 条件判别——D 训练改动，需在 `update_discriminator` 加负对
- (D) 弱 conditional anti-stillness 仅作诊断臂
- batch-standardized reward 降级为诊断臂不升 canonical；aux002 延伸取消；ankle_roll 修复列为 hygiene（但它本身也是运动惩罚项，修复会加重静止压力，需与 (C) 同期评估）

## 12. P0.55 + 形成机制回放（2026-08-21 下午，双会话协同；冲突纪要见文末）

**分钟级前置三件套（全部完成）：**
1. **z8/tracking-z 精确等价**（`check_z8_tracking_equivalence.py`）：同 250 帧序列同窗口双路径，`allclose=True`（max diff ~2e-6）、cosine=1.0 全程、尾部截断 t=243-249 也仅 2e-6——**嫌疑人 #4（z 契约错位）正式关闭**，两条路径逐位等价（250 是调度长度不是聚合窗口）。
2. **D-batch 配对账本**（代码级）：D 的 BCE 正类 = 同一 expert_batch 的 (obs, z) 同 8 帧窗口配对；负类 = replay 中 rollout 时的真实 z（行为配对）；mixed-z/relabel 只进 critic/actor/FB（发生在 D 训练之后）。**"positive obs + random z → positive"不存在**——条件失效归因于"捷径主导"（obs 边际可分时 BCE 无压力学配对），非"从未见过配对"。
3. **coverage 数学验证**：340 次有放回抽样的期望独立 clip 数 = 1462×(1−(1−1/1462)^340) ≈ **303.5 vs 实测 300**——coverage 20.5% 纯属抽样预算（生日悖论），与 ≥251 约束无关（仅 50 条 clip 不合格）。全量重建必须用 per-episode 显式分配（生成器 `--sampling per-episode` 已加）。

**形成机制回放**（`replay_d_checkpoints.py`，收割 run `bfm-dynamics-rnorm-harvest300k`：300k 步、ckpt 5k~300k×10 快照、配置=rnorm canonical）：

| step | M_t(dyn−stand) | frozen_stand | policy_t_self | policy \|a\| |
|---:|---:|---:|---:|---:|
| 5k | **+0.44** | 1.76 | +0.56 | 0.88 |
| 25k | **+0.55**(峰) | 2.91 | +0.85 | 0.89 |
| 75k | +0.32 | 3.67 | — | — |
| 100k | **−0.02**(翻转) | 3.45 | −0.80 | 0.61 |
| 150k | +0.04 | **3.90**(反超) | — | — |
| 300k | +0.22 | 3.46 | −1.72 | 0.61 |

**"早期乱动锻造反动态捷径"从假设升级为回放确认**：M_t 与 policy activity 同步由正转负（75k→100k），五段式签名齐备。注意与 aux002/rnorm run 的 activity 下滑时段（~100k 起）吻合——闭环时间线一致。

**并行会话冲突纪要**：双会话各启动了一个 Pool-Full 生成（同输出文件），side-session 已杀掉自己的重复进程保留主会话的（2050475）；主会话日志曾被 side 重定向截断（进程句柄继续写，tail 自愈）。教训：跨会话共享 artifacts 需显式所有权声明。

**当前在飞**：Pool-Full 全覆盖生成（主会话，per-episode 1462 clips，~68fps）；L_cond 探针（主会话已完成 200k：disc_cond_loss 1.32→0.62 持续下降=配对可学首个直接证据；|a| 0.58→0.87 回升）。下一步：L_cond + Pool-Full 的正式 A/B（canonical 判定=冻结解除+raw D 改善+配对审计复验 top-1）。

## 12. Probe A 闸门 + conditional loss 修复链（2026-08-21 下午）

**Probe A（`probe_z_activity.py` → `z_activity_probe.json`）：z 信息充足——情况 A。**
- unseen-clip（按 clip 切分，无帧泄漏）stand/dynamic AUROC：linear **1.0** / MLP 0.999 / kNN 1.0
- activity 回归（含未来 0.5s）Spearman≈0.86（线性即可读）
- train≈test：泛化行为语义非 clip 记忆。闸门通过 → D 不肯用 z，不是 z 没信息。

**L_cond 实现**（`BDX_COND_LOSS`→`cond_loss_coeff`，独立 softplus margin 排序损失，batch 内高/低活动窗口交叉配对，不混入 BCE fake 类；`disc_cond_loss` 指标）。

**Pool-Full**（`bdx_expert_sim_pool_full.pt`）：1462/1462 全覆盖，385,968 帧，零污染（per-episode 采样模式 `--sampling per-episode`）。

**ankle_roll 正确性修复**：空切片显式返回 0 + 注释（单 DOF 脚踝 not applicable），launcher scaling 改 0.0 并注明——按外部裁决不重映射语义。

**fullpool+condloss 2M（`bfm-dynamics-fullpool-condloss-2m`，2M checkpoint 已存）**：
- cond_loss 1.10→0.094（学成）；带宽 3.12 稳定；Q1 零中心
- **行为冻结轨迹与 rnorm 参照完全相同**（actR 2.5 vs 2.5 @1.96M，力矩 43）

**dz 复测（cond-trained D，`dz_audit_cond.json.log`）**：
- diag−off 0.21→**0.39**，top1 9.4%→14.1%
- **条件静止测试反转**：冻结站立配 dyn_z -0.87→**-1.53**（vs stand_z -1.21）——D 已正确判"dynamic-z 下静止=假"
- 边际偏好仍在（stand_obs 块 -1.18/-1.45 > dyn_obs 块 -1.62/-1.78）

**结论与下一刀**：D 的条件指挥已修复但 actor 冻结照旧 → 瓶颈正式下移到 **actor-z 响应与 rollout z 采样**（外部评审 Probe 4/5）：
1. Probe 4：固定 obs 扫 z，测 actor action 方差/雅可比；rollout activity 按 z 分层
2. Probe 5：z_buffer/tracking-z/relabel 的 activity 分布——rollout 时 z 落在 stand 模式的比例（若大多数 z 是 stand-z，冻结就是 actor 对 z 的正确响应！）
3. 之后才考虑 actor 侧训练改动（z-conditioning 注入方式/容量）

## 13. 探针 4 定案 + D 条件预训练设计（2026-08-21 晚，外部裁决后）

**探针 4（cond margin 形成速度，`fullpool-condloss-2m` train_log）**：隐含 conditional margin（由 softplus 反解）**前 172k 全为负**（8k: -0.98 → 100k: -0.55 → 172k: -0.01 → 180k 转正）——**#2 反动态捷径形成窗口（~100k，收割回放 M_t 翻转点）内条件保护完全缺位**。这直接支持外部裁决的"D 条件预训练"方案：在 RL 开始前把 cond margin 训到 ≥+0.5，使 policy 从第一步就面对正确的条件 D。时间线咬合：actR 主下滑段（100k-2M，85→2.5）恰逢条件保护缺位/初成期。

**300k 实验矩阵（外部裁决，未跑）**：

| Arm | D 初始化 | Policy 状态初始化 | 目的 |
|---|---|---|---|
| A | 在线 L_cond | 默认 | 现有失败基线（已有 2M 数据可截） |
| B | **条件预训练 D** | 默认 | L_cond 学得太晚？ |
| C | 在线 L_cond | expert-state curriculum | 动态探索不足？ |
| D | 条件预训练 D | expert-state curriculum | 完整组合 |

诊断点 0/10k/25k/50k/100k/150k/200k/300k（`BDX_CHECKPOINT_EVERY` 开关已由并行会话加入）。成功指标：dyn−stand margin 不翻负、dyn-z rollout activity 不单调坍缩、stand-z 仍稳定、actor 对 dyn/stand z 动作显著不同、r̂ 与 tracking 质量正相关。

**D 条件预训练实施**：离线脚本用 Pool-Full 的 matched/mismatched expert pairs 只训 D（BCE expert-vs-expert-sim 无意义——正类 pool 即 sim 域；预训练目标 = L_cond 排序 + GP 正则，或加 policy 先验负类……注意：无 policy 数据时 BCE 无负类，预训练只能以 L_cond 为主 + 现有 relabel 机制的 z 配对结构）。验收：cond margin ≥ +0.5 且冻结站立配 dyn-z 显著低于配对（dz audit 复测），然后启动 RL。

**探针 1+2（rollout z activity 分布 / actor-z 响应）**：脚本 `probe_z_rollout_and_actor.py` 已就绪但被用户自己的 GPU 任务（AniMoFormer 12GB）阻塞，OOM 退出；待 GPU 空闲重跑。注意 actor 输入 filter 需 `history_actor` 键（探针中补零）。

**项目管理（外部要求）**：共享 artifact 需写 owner/output/start/finish/write-mode；pool 与 checkpoint 用临时文件生成后原子 rename。

**探针 1+2 结果（2026-08-21 深夜，`z_rollout_actor_probe.json`，probe holdout AUROC 0.999）**：
- **探针 1**：rollout 混合 z（60/20/20）**≈74% dynamic**（expert-z 71.7% / goal-z 77.5% / random-z 77.5%）——排除"冻结=正确执行采样先验"；random-z 也偏 dynamic（B 空间 dynamic 半空间占主导，probe 校准注记）。
- **探针 2**：2M actor 换 dyn/stand z 的 Δa≈0.22（cosine 0.84）vs **随机 init actor Δa≈1.22**——训练把 actor 的 z 响应压缩 5 倍（"学会忽略 z"，与 #2 锁定机制一致）。
- 四种子裁决：#1 L_cond 学太慢=✅（探针4 margin 前 172k 负）；#3 z 采样偏静态=❌（探针1）；#4 actor 不用 z=✅（探针2，响应被压）；#2 standardization 削弱=❓（探针3 未做，留待）。
- **强化结论**：D 条件预训练（arm B）是第一刀；探针 2 的 z 响应压缩提示 arm C/D（expert-state curriculum）可能也必要——预训练 D 只保证指挥正确，不阻止 actor 再次学会忽略 z。arm B 跑完立即用 probe2 的 Δa 复测 actor 响应是否保持。

## 14. 探针 3 定案（2026-08-22 凌晨，`probe3_signal_chain.json`）——信号链断裂点定位，Arm B 暂缓

**设置**：2M actor 在 CPU env 真实 rollout 2048 步（混合 pool z）→ 反事实对嵌入真实 batch composition → 逐层追踪。

| 层 | 数值 | 判读 |
|---|---|---|
| L1 raw logit | dyn 对 +0.48 vs stand 对 -0.39（gap +0.87） | **cond-trained D 的条件指挥正确** |
| L2 norm+clip5 | 两者均 +5.0，**gap=0** | **断裂点**：policy batch logit μ=-5.49 σ=0.72，expert 级状态在 +6~8σ → 全部撞上限 |
| L3 percentile | 双双 100% | batch 内排序仍在 |
| L4 TD/Q | gap 1.28/0.40 | 残余但源头已平 |
| L5 actor grad | stand 157 > dyn 119 | 不指向多动 |

**附加发现（比断裂点更深）**：batch 内 Spearman(r̂, activity) = **-0.337**——D 对 policy 状态的排序系统性反动态（“动=更假”）。L_cond 只修了 expert 域条件性，policy 域的 #2 捷径仍在。另一修正：2M actor rollout dof_vel≈4.5——“冻结”实为恒定小动作（|a|0.12）下的低控制漂移，非完全静止。

**决策表命中第一行：raw D 正确 + normalized reward gap≈0 → 先修 reward normalization，Arm B（D 预训练）暂缓。** 但 batch 内负相关同时保留 fresh-D 预训练的独立理由（预训练 D 未见 policy 乱动状态，不会形成 policy-域捷径）——两条修复在下一轮 300k 中可组合。

**下一刀（单变量优先序，300k 内）**：
1. reward 传递修复：去掉 hard clip 只留 batch-norm（或固定参考 μ/σ——老师 arm C 校准方案；或 winsorize 软限）。判据：反事实 L2 gap > 0 且 batch 内 Spearman 转正。
2. 若 1 后 Spearman 仍负 → fresh-D 条件预训练（原 Arm B 方案不变）上马，因 policy-域捷径需从源头防。
3. actor-z 响应压缩与 curriculum（arm C/D）在其后。

**硬截止提醒：2026-08-23 Go/No-Go。** 当前 Go/No-Go 倾向：路线未死（D 指挥正确、信号链断点明确可修、z 信息充足），但需 300k reward 修复臂的结果支撑。

## 15. Probe 3B + 组合 300k + Go/No-Go（2026-08-22 深夜，截止日内完成）

**Probe 3B（`probe3b_stratified.json`，2M ckpt 基线）**：
- dyn-z 子集 Spearman(r̂, activity) = **-0.326**——混合 z 伪影排除，policy 域反动态实锤（stand-z 内 -0.10 合理）
- alignment 相关 ≈0.05（D 排序既非 activity 亦非匹配度）；tracking proxy -0.34 被 activity 混杂污染不可单独解读（诚实标注）
- OOD 通过（goal-z 0.83/random-z 1.03 vs 参考 0.79）——74% dynamic 可信
- 真实 history 使 L1 gap 增大到 **+2.0**（零 history 版为 0.87）——D 指挥比初测更强
- **映射对比定案**：noclip gap +2.61 ✓；clip5 = 0；**tanh2 = 0.003（9σ 处同样饱和，判死）**→ 选 noclip
- **新发现：critic 方向性 z 盲**——真实 policy obs 换 dyn/stand z 的 actor 梯度 cosine 0.98（终态测量，非结构上限）

**fresh-D 条件预训练**（`pretrain_cond_d.py`→`bdx_pretrained_cond_d.pt`，原子 rename）：held-out 反事实 margin **+2.68** 达标（600 步早停）；`BDX_PRETRAINED_D` 挂载（只载 D，actor/critic/F/B fresh）。

**组合 300k（`bfm-dynamics-combo300k`：预训 D + noclip rnorm + cond 0.3 + D lr 2e-6 + 25k 间隔 ckpt）**：
- rawD **-0.17→-2.46**（历史所有 run 同期 ≈-4.9，好一个量级）；actR 无 100k 坍缩点（139k 仍 94），295k 缓降 49.6 未冻结；r̂ 零中心带宽 3.30；Q1=3.0 有界
- checkpoint 复测：**L1 gap +1.88 存活**（anchor 有效）；反事实对落入 batch 分布内（+2.0~+4.7，不再撞顶——可达集内正分状态出现）；**stand-z Spearman 转正 +0.37**
- **但 dyn-z 内 Spearman -0.446（比 2M 的 -0.326 更负）**——三件套未阻止 policy-域反动态捷径重新形成

**Go/No-Go 判定（按分级标准）：条件 Go。**
- 已过闸门：条件指挥存活 ✓、reward 传递（noclip gap 保留）✓、分布稳定 ✓、activity 无坍缩点 ✓、可达集正分状态 ✓
- 未过：dyn-z 内 policy 排序仍反动态（-0.45）——**#2 根源（早期乱动→"policy 的动=假"焊接）正式实锤为三件套外的问题**
- 下一刀唯一候选：**reference-state exploration curriculum**（arm C/D：episode 从 expert 动态状态附近初始化、渐退）——让"运动中的 policy 状态"从第一步就在正分区，切断 #2 的形成路径。之后若 dyn-z 排序仍负，才动 D 输入结构。
- 路线本身 Go：z 信息充足、D 可教、reward 链可修，全部有因果证据；剩余问题有明确机制与对应干预。

**新增工程**：`BDX_PRETRAINED_D`、`pretrain_cond_d.py`、probe3b 脚本（含 combo ckpt 复测路径）。双会话 artifact 规范（owner+原子 rename）已应用于预训练产物。

## 16. 最终定案（2026-08-21）：reset state 与 rollout z 条件错配

训练环境原本已经在 episode reset 时把机器人放到随机专家 motion state；所以“增加专家 reset”并不完整。真正遗漏的是：

```text
physical reset state <- motion A
rollout context z    <- independently sampled motion/latent B
```

因此越频繁 reset，越频繁制造 `(state_from_A, z_from_B)`。这解释了：8→500 horizon 的首版 curriculum 把 dyn-z Spearman 从 -0.446 拉到约 0，但 8→100 的强版反而回落到 -0.413。

### 正式修复

新增 `BFM_REF_Z_ALIGNED=1`：从 `bdx_expert_sim_pool_full.pt` 选择动态窗口，把 simulator reset 到该窗口对应 motion/time，并用**同一 8-frame simulator-domain window**经当前 backward map 编码 rollout z。实现还显式处理了 full-pool global motion id 到训练时 256-motion 子集 local id 的映射。

- `humanoidverse/train.py`：共享已加载 pool、同窗口 z 编码与 reset-row context override；
- `humanoidverse/envs/legged_robot_motions/legged_robot_motions.py`：动态窗口选择、global→local id 映射、reset provenance；
- `humanoidverse/scripts/probe3b_stratified.py`：支持 `DZ_CKPT` / `DZ_OUTPUT`。

### 300k 定案实验

结果：`results/bfm-dynamics-refzalign300k`（最终 checkpoint 275,968）；探针：`results/bfmzero-bdx-full/probe3b_refzalign300k.json`。

| 指标 | 修复前 | z-aligned |
|---|---:|---:|
| dyn-z 内 Spearman(reward, activity) | -0.446 | **+0.100** |
| overall Spearman | 负 | **+0.092** |
| stand-z 内 Spearman(reward, activity) | +0.37（旧误设目标） | **-0.135**（stand 下乱动应扣分） |
| tracking proxy Spearman | +0.299（首版 curriculum） | **+0.529** |
| dyn/stand observation-gradient cosine | ~0.92 | **0.248** |
| rollout activity mean | 6.90（首版） | **6.19** |
| real-history L1 dyn-stand gap | 1.70（首版） | **1.80** |

主闸门的符号已经翻转，且条件梯度由近乎同向变为明显分化，因此不进入 bilinear/FiLM D 重构。最终固定组合为：full expert-sim pool + pretrained conditional D + batch-standardized noclip reward + cond loss 0.3 + D lr 2e-6 + z-aligned reference curriculum（horizon 8→500、动态比例 1.0→0.2 over 300k）。下一步只需做 1–3M 性能验收，检查正排序保持与 MPJPE 越过历史平台；这不再属于根因定位。

## 17. 1M 验收否决 300k 乐观结论（2026-08-21）

§16 的 300k 结论只在短时成立，不能视为最终解决。`results/bfm-dynamics-refzalign1m` 的 900,864 checkpoint：dyn-z Spearman +0.021（近零）、tracking proxy -0.047、dyn/stand gradient cosine 0.964。concat D 长期重新退化成近乎条件无关。

随后完成两项定案 ablation：

1. **strict bilinear D**（无 observation-only 输出路径）：预训练 held-out margin +1.33；300k alignment +0.309，但 dyn-z activity -0.371、tracking -0.074，说明强制交互本身不等于正确 imitation ranking，正式否决。
2. **episode-level z lock**：发现 reset 对齐 z 原先会在 `update_z_every_step=100` 时被独立替换。修复为所有 reset state/z 同窗口且 episode 内保持到下一 reset。300k tracking +0.357、stand activity -0.493、dyn activity ≈0；但 1M 的 900,864 checkpoint 再次退化：tracking -0.021、alignment -0.214、gradient cosine 0.986，正式否决其为充分修复。

产物：

- `results/bfm-dynamics-refzalign1m`
- `results/bfm-dynamics-refzalign-bilinear300k`
- `results/bfm-dynamics-refzlock300k`
- `results/bfm-dynamics-refzlock1m`
- `results/bfmzero-bdx-full/probe3b_refzalign1m.json`
- `results/bfmzero-bdx-full/probe3b_refzalign_bilinear300k.json`
- `results/bfmzero-bdx-full/probe3b_refzlock{300k,1m}.json`

**当前准确结论**：state/z 对齐属于 correctness fix，能改善早期训练；但当前 `L_cond` 只在 expert 域做 paired-vs-mismatched ranking，无法阻止 policy 域长期条件退化。不能启动 3M/100M。下一设计必须直接约束 policy 域的 conditional reward geometry（而不是继续调 curriculum、clip、lr 或单纯换 bilinear），并需要一个不会把坏 policy pairing当正样本的目标。

## 18. Gate 0：测量契约错误修复 + 污染数值作废（2026-08-24，外部裁决执行）

**发现的契约错误**：BDX state 布局经 wrapper 代码确认为 `[dof_pos(0:14), dof_vel(14:28), projected_gravity(28:31), base_ang_vel(31:34)]`。此前全部 activity 计算使用 `state[:, 20:34]` 是**错误的**（实为 dof_vel 后 6 通道 + gravity + base_ang_vel 的混合）。probe3b 的 build_history 四块切片全部错位（"真实 history"实为乱序拼接）。

**修复**：新增唯一 schema API `humanoidverse/utils/bdx_state.py`（`split_bdx_state`/`bdx_motion_activity`，带 assert）+ 人工构造单元测试 `tests/test_bdx_state_schema.py`（逐块扰动响应，已通过）。全部 12 处污染点已替换并编译通过：`fb_cpr/agent.py`（L_cond 的 win_act + policy_cond 三处）、`fb_cpr_aux/agent.py`（motion diag）、`pretrain_cond_d.py`、`probe_z_activity.py`、`probe_z_rollout_and_actor.py`、`audit_pool_prior.py`（两处）、`probe3_signal_chain.py`、`probe3b_stratified.py`（含 history 四块修复）。

**数值作废声明（引用时必读）**：
- Probe A 的 Spearman≈0.86 **作废**（AUROC=1.0 有定性参考价值待重跑确认）；
- probe3/probe3b 全部 dyn-z/overall/stand-z Spearman 与 alignment 数值**作废**（含 §14/§15/§16/§17 表格中的该类数字；重跑前连符号也不应引用）；
- 所有 run 日志中的 `diag/mean_abs_dof_vel_norm` 列**作废**（混合通道口径）；`diag/mean_abs_action` 不受影响；
- L_cond 的 batch 内高/低活动配对此前基于混合通道——方向大体保留（dof_vel 后 6 通道占主要方差）但排序受污染，历史 cond loss 数值仅供趋势参考；
- policy_cond 300k run（若存在于任何证据表）：**invalidated by gating/index contract violation**，不进入证据表。

**第五层正式改名**：`concat-D 长时程条件性侵蚀` → **长时程条件闭环退化：D 条件使用、B/z 表示稳定性与 policy 条件响应三者尚未解耦**（H1 z 年龄错配 / H2 B 坐标系漂移 / H3 B-expert OOD / H4 D 条件捷径，四假设待判）。

**待执行（Gate 1 设计，Codex 首优先级）**：固定面板（disjoint expert dyn/stand + 按时间分桶 policy 窗口）× 25k checkpoint 的 z 流形审计总表：expert-z effective rank / activity AUROC(Spearman) / 跨 ckpt raw cosine + Procrustes 残差 + CKA + kNN 保持率 / 旧 probe zero-shot 迁移 / stored_z-to-current_B 距离（shrinkage Mahalanobis + kNN） / D matched-vs-permuted margin / actor 多 z action 方差 / 行为 tracking。判因果=首个异常时间（Gate 2 时间表逻辑见外部裁决原文）。**z 账本**（z_source/z_age_steps/encoder_version + tracking_z sampled/committed/overwritten 计数）需加到 GPT 的 zlock/refzalign 代码路径。首轮在线臂仅两个候选：short-age replay / 统一 target encoder——由 Gate 1 结果决定。

**Gate 0 重跑结果（正确 dof_vel 口径，2026-08-24，颠覆性修正）**：

| checkpoint | dyn-z Spearman（旧口径→正确口径） | 方向梯度 cosine（旧→新） | L1 gap |
|---|---|---|---|
| combo300k | -0.446 → **+0.222** | 0.98 → 0.82 | 1.88 |
| refzalign1m(900k) | +0.021 → **+0.033** | 0.964 → **0.399** | 2.24 |
| refzlock1m(900k) | → **+0.030** | 0.986 → 0.981 | 1.58 |

1. **旧口径的"反动态排序"是测量伪影**：三个 checkpoint 的 dyn-z 排序全部 ≥0——§15 的"三件套未阻止 #2 重演（-0.446）"结论**撤回**，combo300k 在正确口径下 policy 域排序已转正。
2. **1M"退化"重写为"正排序衰减至中性"**（+0.22→+0.03），非负；且 refzalign1m 的方向性 z 响应实为 **0.399（健康！）**——旧 0.964 也是伪影。refzlock1m 的方向响应确实死（0.981）——两个 1M run 呈现**不同**的退化签名（refzalign 排序弱但方向好；zlock 方向死但 stand-z 排序正），无一致侵蚀模式。
3. **第五层（改名后）的实证基础被重置**：H1-H4 四假设的先验全部重排，Gate 1 固定面板审计（§18 上文设计）现在是唯一正确的前进方式——z 账本 + 25k checkpoint 总表判"谁先坏"。

**Gate 0.1 完成（2026-08-24 终审裁决执行）**：

1. **history 契约修复**：确认 `HistoryHandler.add` 为 newest-first（`query[:,:4]=[t,t-1,t-2,t-3]`）；统一构建器 `build_bdx_actor_history` 入 `bdx_state.py`（t≥3 全前缀要求，拒绝 padding 伪装），与 handler 逐位对照契约测试通过。probe3b 已切换。
2. **provenance 修复**：被覆盖的 `probe3b_stratified.json` 迁移为 `probe3b__combo300k__state-v2__history-v1__OVERWROTE_2M_BASELINE.json` + `MANIFEST_provenance.json`（原始 2M 旧契约基线仅存于 §14/15 文本，不回填伪造）。probe3b 输出新增 `_provenance` 字段、拒绝覆盖、原子写。**2M 基线 corrected-contract 已重建**：`probe3b__condloss2m__state-v2__history-v2.json`。
3. **指标依赖标注**：D 不看 history（输入 filter=[state,privileged]）→ L1 gap/所有 Spearman/cosine 与 history 无关、站得住；`real_history_pair` 的 L4（critic Q）在旧 history 下 provisional，现已修复。
4. **policy_cond 事实勘误 + 判读**：run 确实存在（`bfm-policy-negcond300k/-smoke8k{,-v2}`）。离线判读=裁决情况 1：gate 覆盖率 0.168→**0.006**（300k 时几乎不激活）——该 run 实际接近 policy_cond-off，不能证明 policy_cond 有效亦非失败；margin 1.67 仅在稀疏 gated 样本上学到。恢复实验须用正确契约 + 重标定 `policy_max_abs_dof_vel=10.0`（绝对阈值，corrected 尺度下需复核；env 侧 `BFM_REF_ACTIVITY_MIN=2.0` 本用正确 `14:28` 无需调整）。
5. **修正后四 run 对比（corrected contract）**：condloss-2M dyn-z **-0.477（真负——反动态捷径在 2M 配置真实存在）** / combo300k +0.222 / refzalign1m +0.033 / refzlock1m +0.030；cosine 0.90/0.82/0.399/0.981。叙事修正：**反动态是真实的（2M 配置），combo 修复了它（300k 正），1M 衰减至中性**——"伪影"仅解释 combo300k 的旧读数（activity 口径对 2M 的 overall 几乎无影响：旧 -0.337 vs 新 -0.341）。
6. **假设排序（终审采纳）**：第一梯队 H1（stored-z 与当前坐标错配）/H2（B 坐标系漂移）；第二梯队 H3（B-expert OOD）；第三梯队 H4（D 主动忽略条件）；**新增 H5（zlock 的条件激励不足/共线性——非表示陈旧）**。zlock×cosine 强指纹不构成因果定案（单 run、多变量混杂）。
7. **Gate 1 前进方式（Codex 执行）**：三类数据不可混淆——A 固定面板表示审计（raw+Procrustes 分离旋转 vs 几何漂移）、B 各 checkpoint 原生 rollout、C stored-z 审计（需 z 账本：source_motion_id/window/z_collect_step/z_source/encoder_version——老 checkpoint 无此数据，须新短 run 带账本重采）。首轮在线臂：Arm 1 = replay age cap（最轻）；target encoder 缓行。2×2 冻结分解（old/new dyn subset × old/new activity）待做以升级"合理的物理解释"为"已验证的伪影机制"。

**Gate 0.2（最终复核执行，2026-08-24）**：

1. **契约测试升级**：`build_bdx_actor_history` 与**真实 `HistoryHandler`**（生产配置嵌套）+ 生产 `_get_obs_long_history` 的 sorted-keys flatten 顺序**端到端逐位一致**（哨兵序列测试通过——sorted(keys) 顺序经实证确认）。D 不看 history 的结论限定为"当前 launcher 的 discriminator filter 配置下"。
2. **provenance 字段级 manifest**：`MANIFEST_provenance.json` 逐 artifact 登记有效/无效指标（两个 1M `_v2` 文件的 L4 无效因 history-v1；D 指标全部有效；`z_activity_probe.json` 同样被覆盖、旧值仅存 §12）。
3. **policy_cond 累计分析**（`policy_cond_cumulative_audit.json`）：update-weighted 覆盖 **3.77%**（50k 时 10.4%、100k 时 7.4%）、每 batch 非零、活跃时损失量级为 cond loss 的 **56%**——**介于 off 与早期瞬态之间，不等价为 policy_cond-off**（终审措辞采纳，撤回 Gate 0.1 的"情况 1"判定）。
4. **within-run 数据限制**：两个 1M run 仅存最终 900,864 checkpoint（滚动覆盖）——**同轨迹 300k→1M corrected 曲线无法离线补出**。已用同配置独立 300k run 作近似早期点（caveat：跨 run，warp 非确定性下不保证同轨迹）：
   - refzalign300k(276k)：dyn-z **-0.024**、cosine 0.954
   - refzlock300k：dyn-z **-0.325**、cosine 0.837
   近似读法（非因果）：refzalign 族 300k→1M 排序 -0.02→+0.03（中性稳定）、cosine 0.95→0.40（方向改善）；zlock 族 -0.32→+0.03、cosine 0.84→0.98——两族签名相反，无一致衰减模式。
5. **叙事三分离（终审采纳）**：condloss2M→combo300k=跨配置差异（非时间修复）；combo300k→refzalign1M=跨 run 族（非长期衰减证据）；refzalign→zlock=复合 z 供给协议差异（非纯 z-age）。"-0.477 捷径"降格为"真实反动态排序，与静止捷径机制一致"；"combo 修复"降格为"combo 复合配置在 300k 消除负排序"（无单变量消融不归因单项）。Spearman（全局排序偏置）与 cosine（局部条件敏感）为两个独立故障轴，加 tracking 共三轴，不合并评分。
6. **age-cap 定性修正**：Arm 1 是筛查实验（恢复→replay 非平稳相关，≠stale-z）；更干净的 H1 对照 = **D-only 双胞胎离线实验**（同 checkpoint 克隆两个 fresh D、完全相同 batch、stored-z vs fresh 重编码 z、比 held-out margin/cosine）+ instrumented 短 run 记账本（source_motion_id/window/z_encoder_step/z_collect_step）。
7. **正式前进路径（Codex）**：同族 instrumented 1M 重跑（`BDX_CHECKPOINT_EVERY=25000` 保留快照）取得真正的 within-run 三轴时间曲线 → Gate 1 固定面板 B 漂移（raw+Procrustes）→ z 账本 → D-only H1 对照 → 才决定 age cap/fresh re-encode/target encoder。

## 19. Gate 0 关闭 + Gate 1.0 启动（2026-08-24 终审）

**两处表述修正（终审采纳）**：
1. `0.248→0.954` 是**测量定义漂移（construct shift）**——activity 契约+z 分类器+配对集同时改变，不是 corrected 仪器的随机误差棒；不得用它推"新契约下差异不可解释"。
2. z-aligned curriculum 定性为**"疗效未被独立证明"**（§16 旧证据失效 + 行为轴未做 matched 对照），不是"价值归零"。

**Gate 1.0 离线前置（部分完成，余项 Codex）**：
- ✅ `BFM_SNAPSHOT_EVERY` 非滚动快照保留已加入 `train.py save()`（ckpt_<time>/，模型+状态不含 buffer）——300k 筛选臂将保留 25k 间隔全部快照
- ⬜ 固定源窗口面板（固定 stand/dynamic source windows + 每期当期 B 重编码 + bootstrap CI）——probe 框架待 Codex 构建；冻结窗口身份与配对，冻结的是身份不是旧 z 数值（避免把 B 漂移混进 D probe）
- ⬜ 同数据 old/new 2×2 契约分解、manifest 伞形 invalid 条目
- D-only 四臂 H1 对照（stored-z / fresh-B(source) / permuted / **Procrustes-transported stored-z**——第四臂分离坐标旋转 vs 几何变形）refzalign 与 zlock 各一套，比较 Δ(fresh−stored)

**300k 匹配筛选（两臂 + 可选第三臂）**：A=combo（无 curriculum）、B=combo+z-aligned、C 可选=+zlock；同 seed、25k 快照、固定面板评估、z 账本从第一步；行为轴+排序轴+方向轴三重验收后仅延长有信息量的臂到 1M。两个 300k 近似点（refzalign300k -0.024/0.954、zlock300k -0.325/0.837）只作**假设生成**（跨轨迹横截面，"独立 run 呈相反协议签名"），不得写成时间演化。

**policy_cond 终态**：稀疏/前置/路径依赖干预（3.77% 加权覆盖、损失比 56%），缺梯度贡献比，不等价 off 亦不证明有效——留作已污染辅助诊断 run，不重训。

**工程清理（不阻塞）**：生产 flatten 提取为 `flatten_history_by_config(handler, config)` 纯函数供 env/probe/test 共用（消灭第二份实现）。

**Gate 1.0 首批数据（2026-08-24 晚）：300k 匹配筛选 A vs B 完成**

- 工程前置就位：`BFM_SNAPSHOT_EVERY`（修复步数网格偏移 25088 的索引判定 bug）——两臂各保留 **11 个 25k 间隔快照**（`ckpt_25088..274432`），within-run 三轴曲线首次可测
- Arm A（combo 无 curriculum）与 Arm B（combo+z-aligned）：同 seed/初始化/数据顺序，唯一差异 `BFM_REF_Z_ALIGNED`
- **初步读数（294,912 终点）**：A rawD -2.30/|a| 0.578/actR 43.9/力矩 3413；B rawD -2.60/|a| 0.650/actR 48.9/力矩 3960——B 的 activity 略高、rawD 略低、cond loss 持平（0.61/0.62），轨迹前 106k 几乎重合、163k 起 B 的 |a| 与力矩更高。**行为轴差异小且方向不明**——curriculum 的 matched 疗效未现显著优势（与终审"疗效未被独立证明"的预判一致）
- **下一步（Codex）**：用两臂的 11 个快照 × 固定源窗口面板（当期 B 重编码）算三轴 within-run 曲线 + bootstrap CI；z 账本；D-only 四臂 H1 对照（含 Procrustes-transported 第四臂）。若 A/B 在条件轴与行为轴均无差异 → curriculum 降级，注意力转向 H1/H2 的 B 漂移审计。

**Gate 1.0 执行（2026-08-24 深夜二，终审放行项）**：

1. **snapshot 修复**：索引恒推进（resume 安全）+ tmp→READY→原子 rename（probe 只读完成目录）。
2. **manifest 伞形条目 + 跨 run 比较规则**：rawD 仅 run 内趋势、train cond loss 跨 run 不可比、快照非独立 seed。
3. **同缓存 old/new 2×2（双 checkpoint，`probe_contract_2x2{,_combo300k}.json`）——construct shift 闭环定案**：
   - 2M：全四格负（-0.26/-0.38/-0.26/-0.35）；combo300k：全四格正/中性（+0.20/+0.05/+0.17/+0.03）
   - **符号由配置决定，口径只调幅度（±0.2）**；旧→新的大翻转（-0.45→+0.22）主要来自口径外成分
   - **新教训：单次 2048 步 rollout 的 Spearman 自身有 0.2-0.5 抽样变异**（2M 复测 -0.477/-0.35、combo +0.22/+0.03）——固定面板+bootstrap CI 是硬必需，此前所有单 rollout 点估计（含 v2 系列）都应视为±0.25 精度
   - 通道分解确认旧口径缺陷机制：dof_vel 前 6 通道与 new activity 相关 0.73 vs old 仅 0.19；base_ang_vel 混入 0.35
4. **A/B 筛选初读（按终审规则）**：rawD 跨 run 不可比（撤回 §19 表内直接并列的用法）；B 活动量略高≠疗效。行为轴 B 力矩/actR 略差。**tracker 选择初判：若固定面板 tracking 相近 → 冻结 Arm A**（终审规则）；固定面板三轴曲线为下一步（22 快照 × native-current panel），判读按"首异常时间"且不得把快照当独立 seed。

**Gate 1.0 双面板裁决（2026-08-24 夜，`paired_panel_arm{A,B}.json`，`probe_paired_panel.py`）**：

统计修正全部采纳（撤回 ±0.25/伪重复/0.5 判决线；manifest 新增 replication_status；cosine 标 valid-but-unreplicated；2×2 关闭——"契约改幅度、符号随配置、跨 rollout 重采贡献更大但两次重复无法分离分量"）。

**双面板结果（22 快照）**：
- Expert 面板：两臂 margin 0.08→~0.5、L1 0.73→~2.2 同步增长（300k 内无侵蚀）；A 后期略优（0.53 vs 0.45）
- Policy 配对面板（固定 z 源+CRN seeds+episode 级配对 bootstrap）：**四轴全部 CI 跨零**——tracking Δ=+0.55[-0.78,+1.90]、activity Δ=-0.16[-2.78,+2.09]、action-rate Δ=+0.003[-0.004,+0.010]、dyn-z Sp Δ=-0.06[-0.34,+0.21]
- 注记：episode reset 用 env 自身运动 reset（非 pool 窗口精确注入——z 源与初始姿态不同源），tracking 绝对值有偏但两臂配对一致，A/B 差值有效

**裁决：curriculum（z-aligned）降级——四轴无显著优势 + 训练期力矩/action-rate 更差 → 冻结 Arm A（combo）为临时 tracker 基础。**

**新观察（进入下一阶段问题清单）**：
1. 两臂 activity 仍同步下滑（19.7→8.1 vs 数据集 3.5）——300k 尺度内未止跌
2. within-run margin 曲线健康增长——"长时程侵蚀"在 300k 内未现，1M 侵蚀问题仍开放（需 1M 快照曲线）
3. 下一步不变：z 账本 → D-only 四臂（stored/fresh/permuted/Procrustes-transported）→ 若 A 追加 1M 须带 25k 快照

**Gate 1.0 终局：arm×对齐 2×2（2026-08-24 夜三，`probe_aligned_2x2.json`，`probe_aligned_2x2.py`）**：

终审评测混杂修正全部落实：正式 aligned reset（motion lib 原始物理状态注入 + env reset 语义零 history + 配对 DR 种子 + deterministic mean action）、扩大面板（12 episodes）、标准化 margin（尺度无关）、条件 activity gap、source-relative activity、全 provenance 落盘。前次"curriculum 降级/冻结 A"表述按终审改写。

**结果**：
- tracking 四格全无显著差异（aligned Δ+0.22[-0.57,+1.12]、unaligned Δ+0.52[-0.42,+1.53]）——curriculum 对 tracking 无已证疗效（含 aligned 格）
- expert 标准化 margin A 1.10 vs B 1.07——两臂条件配对强度相当（撤回"A 略优"，raw margin 跨模型不可比）
- **关键新发现：条件活动响应分化**——A-unaligned 的 dyn−stand activity gap **-0.70（条件区分丢失）** vs B-unaligned **+3.19（保持）**；B 的 source-relative activity 在 unaligned 下显著更高（Δ+2.18[+0.64,+3.98]）
- 判读：按终审表最接近"tracking 无差异但条件响应有差异"——curriculum 不改善 tracking 但**保留未对齐部署下的 z 条件活动响应**。若部署=任意状态+外部 z，此差异有实际意义；若仅关心 tracking 精度则无差别

**当前定案**：Arm A 为未对齐部署协议下默认 tracker 候选（简单+贴近训练分布）；curriculum 未被判死——其条件响应保留作用是否值得引入，取决于部署协议需求（用户决策项）。1M 延长前清单（终审八项）中已完成的：正式 aligned reset ✓、2×2 ✓、扩大面板 ✓、标准化 margin+种子化 permutation ✓（部分）、条件/source-relative activity ✓、provenance 落盘 ✓；待做：reset 元数据 checksum 校验、z 账本、D-only 四臂。

**Gate 1.0 最终定案（2026-08-24 夜四，`probe_aligned_2x2_DiD.json` + `probe_command_response.json`）**：

1. **DiD 交互 CI**（stratified bootstrap）：gap 四格 CI——A-unaligned -0.70[-2.28,+0.43]（跨零）、B-unaligned +3.19[+1.30,+5.37]（显著正）、DiD +4.88[+0.05,+10.99]（擦线，探索性）。漏报的 aligned reward-activity Spearman Δ=-0.210[-0.451,-0.001] 已入表（探索性、无多重校正、混合 arm-specific D 与 occupancy）。source-rel +2.18 修正解读：B 更活跃非更准确（目标 0，A 更近）。
2. **命令响应面板**（同 reset state × dyn1/dyn2/stand z 交叉，8 states，checksum 落盘，deterministic action）：
   - g(dyn−stand)：A +1.68 vs B +4.67，**Δ=+3.00[+0.78,+5.25] 显著**（同状态干净配对下 B 活动区分确实更大）
   - **z 特异 tracking（matched vs permuted，命令特异性主终点）：A +0.90 vs B +0.59，Δ=-0.31[-1.85,+1.30] 不显著——两臂都有正特异性，B 不优于 A**
   - 校准误差：B dyn 侧更差（5.85 vs 3.68，过动）
3. **终局裁决（按外部终审资格标准）**："B 只提升粗 activity gap、不提升命令特异 tracking"→ **curriculum 正式降级，不投入 1M**。**Arm A 获正面证据**（z 特异 tracking +0.90、同状态条件区分 +1.68——2×2 的 -0.70 确认为混杂/低功效产物）→ **Arm A 定位升级：临时默认 standalone tracker（未对齐+aligned 双协议均可用），产品冻结仍待新 seed 复现**。
4. 预注册纪律采纳：本轮全部 300k 发现标记 exploratory；下一批实验（新 seed/1M）须预注册主终点（matched-vs-permuted tracking advantage）、次终点（g）、安全/机制终点——写入 manifest 后执行。

**下一步队列（按优先级）**：新 seed 复现 Arm A 的 z 特异 tracking（预注册）→ z 账本 → D-only 四臂 H1 → 复现成功后 1M（带 25k 快照）。

**对称改判（2026-08-24 夜五，外部指令一）**：per-arm vs-zero CI 正式写入 JSON——A z-specific tracking +0.899 [-0.859,+2.502]（**CI 跨零，撤回"A 获正面证据"**）、B +0.592 [+0.056,+1.013]（擦线排除零）、Δ(B−A) 不显著（互不优越）；A gap +1.68 [+0.102,+3.293]、B +4.67 [+3.068,+6.458]（皆正，B 更强）；B dynamic calibration 更差（过动）。**撤回三处表述**（A 唯一证据持有者/curriculum 正式无效/B 更懂 z）。curriculum 定位改写："放大同状态活动响应，命令特异 tracking 收益未证；无净收益支持单独 1M，但未被判死。" n=8 全部 exploratory。长期目标正式定义：任意状态+外部 z → 稳定产生该 z 对应的**具体动作**（"动得多"不算）。

**扩展命令响应面板 v2（2026-08-24 夜六，`probe_command_response_v2.json`，24 states × dyn1/dyn2/dyn3/stand 四路 z 交叉，预注册方向：matched-vs-permuted tracking advantage）**：

| | z 特异 tracking [95% CI] | retrieval@1（随机 0.25） | g |
|---|---|---|---|
| A | **+1.63 [+0.99,+2.21] 显著** | **0.50 [0.29,0.71] 显著超随机** | -0.07 |
| B | -0.20 [-0.90,+0.53] | **0.17 [0.04,0.33]≈低于随机** | +5.61 |
| Δ(B−A) | **-1.83 [-2.83,-0.80] 显著负** | — | +5.68** 显著 |

**判读**：B = "所有 dynamic z 触发同一种运动"模式（指令第二节列出的失效模式之一）；**A 是唯一同时具备具体动作 tracking 特异性 + retrieval 能力证据的 arm**（CI 齐备、n=24）。A 的粗 activity gap 在不同 z 组合下不稳定（+1.68→-0.07）——gap 不可作主终点的又一实证；**retrieval@1 成为最有信息量的新指标**。n=8→24 的结论反转再次证明小样本擦线不可裁决。

**当前状态（更新夜五的改判）**：A 的具体动作语义证据在 n=24 下成立（夜五的撤回基于 n=8，本轮以更大面板重建）；curriculum 维持"活动响应放大器、无命令语义收益"定位；长期目标（具体动作区分执行）目前只有 Arm A 有正向证据。**下一步按指令顺序：z 账本 → D-only 四臂 → 预注册新 seed（验证 A 的 retrieval 复现性）→ 1M 门槛六条件。**

**正式产物修正 + 裁决措辞采纳（2026-08-24 夜七）**：
- n=8 措辞按裁决全文采纳（A 点高方差跨零/B 弱一致/臂间不可判/B 放大 activity 且 dyn 校准差/curriculum 无已证净收益不单独投 1M/未判死）；撤回四句（含"curriculum 问题已科学定案"）。
- **`probe_command_response_v2_formal.json`**：20+ contrast 全表（per-arm vs-zero + B−A，含此前缺失的 retrieval Δ 与全部 z 变体 calibration）、state-cluster bootstrap（seed/replicates/n 落盘）、全部标 exploratory+无多重校正。核心（n=24）：A z-特异 +1.63*、retrieval 0.50*；B -0.20 n.s./0.17；**Δ(B−A) z-特异 -1.83* 与 retrieval -0.333[-0.583,-0.083]* 双双显著**——单 seed、未校正，**暂定读法**：A 具备具体动作语义、B 符合"同一运动"失效模式，待新 seed 复现。A 的方差来源：worst-6 states（z-特异 -2.09~+1.29）已列出供 phase/动作分析。
- Arm 定位（裁决版）：A=canonical research baseline（校准较好、点估计高但状态依赖）；B=curriculum mechanism reference（响应强一致、dyn 过动、未证具体 tracking 优）。**Demo 用哪个 checkpoint 由 Demo agent bakeoff 决定，tracker 线不替 Demo 冻结 A。**
- 队列不变：z 账本 → D-only 四臂 → 预注册新 seed（A retrieval 复现为主终点之一）→ 1M 六条件门槛。

**retrieval vs-chance 正式检验 + z 账本 Q1（2026-08-24 夜八）**：
- **Retrieval**（精确二项 + McNemar，`probe_retrieval_vs_chance.json`）：A 12/24=0.50 **p=0.008 显著超随机**；B 4/24=0.17 p=0.48 **与随机不可分辨**（不得称显著低于）；配对 A-only 胜 10 vs 2 **p=0.039**。裁决读法采纳：A 高于 chance 成立但边界薄（CI 下限 0.29）。
- **z 账本 Q1**（encoder 侧，`z_ledger_Q1_encoder.json`，单位 bug 已修正）：两臂 z embedding 健康——effective rank A 24.0/B 22.7，dyn/stand 聚类温和但方向正确（A：dyn-stand 14.3 > dyn-dyn 11.5；B：13.1 > 9.6），具体 dyn 动作在 z 空间两臂都可分。**Q1 清除：B 的失效不在 encoder——分叉指向 Q2（actor conditioning，H6）**。
- H6 定性采纳："100% state-z 对齐可能导致细粒度 z 冗余；v2 结果支持假说，机制归因未完成"。下一分叉：Q2（actor first-action swap/轨迹分离/source-policy 几何一致性）→ Q3（控制校准）→ Q4（D-only 四臂）。48 states≠2 seeds×24（层级区分采纳）；Holm/FDR 非复现终点硬前置（采纳：v2 保持 exploratory，新 seed 预注册主终点=**A/B matched-vs-permuted tracking advantage 差值**，retrieval 为 gate）。

**Q1 held-out + Q1.5 D 条件信号时间曲线（2026-08-24 夜九，最新裁决执行）**：
产物：`probe_q1_heldout_and_q15_dcurves.json`（脚本 `probe_q15_d_curves.py`；修复两处运行 bug：mids/pred 跨设备、score 返回 tensor 被强转 float）。
- **Q1 held-out z 检索**（40 motions × 4 远距非重叠窗口、LOO 去自匹配 centroid）：**A 3/160 = B 3/160 = 0.019 ≈ chance 0.025，两臂均在随机水平**。夜八"encoder 健康"的 12/24 检索由 self-match 驱动。**措辞修正（夜十裁决采纳）：LOO 测的是跨 phase motion-identity 不变性；该性质并非短窗口编码目标明确要求，"两臂均未形成跨 phase motion-identity 表征"不构成 encoder 故障证据，也不直接支持 H6。撤回夜九的"与 H6 一致"与"细粒度 z 身份信息不可恢复=encoder 细粒度冗余"两句。** 正式读法：两臂仍保留 dyn/stand 粗结构（夜八聚类结论不变）；局部 motion-segment 信息是否足以驱动 actor 由 Q2 判断。长期若需 motion-level（而非 local-segment-level）命令语义，需未来加 motion-level contrastive/invariant objective，当前不阻塞。
- **三层 retrieval 永久区分（裁决固定用语）**：

| 层级 | 测量对象 | 当前结果 |
|---|---|---|
| Encoder LOO retrieval | 跨 phase motion identity | A/B 均≈随机（非故障证据，见上） |
| D retrieval | D 是否区分 matched 条件 | 两臂粗条件结构强（retrieval@1 0.66-0.94） |
| Behavioral retrieval | actor rollout 是否对应正确动作 | A 0.50（p=0.008 超随机）、B 0.17（p=0.48 不可分辨）；条件于 state-level independence，待 lineage 确认 |

  禁止使用无层级定语的"retrieval 12/24""encoder retrieval"。
- **Q1.5 D 条件信号时间曲线**（A/B 各 11 快照 × 32 固定窗口面板）：粗 margin（dyn-vs-stand）两臂同步增长 1.3→~4.5-4.9，D retrieval@1 达 0.66-0.94（chance 0.25）——两臂 conditional D 均有强条件结构。细 margin（同 activity 邻窗）两臂中段出现较高点值（A 峰值 1.23@150k、B 1.07@225k），后段点值相对降低（A 0.31、B 0.43@275k）。**该时间趋势目前为描述性观察（descriptive）——peak 为事后选择、末点单次评估，不得写成已确认的"D 后期衰减"；正式检验待 state-cluster bootstrap + 预定义 early/middle/late 对比。**
- **路由判决（按裁决四分叉）**：A/B 都有中段细粒度 D margin 且粗结构强 → **进入 Q2（actor conditioning 审计）**；非"D 条件失效主嫌疑"（两臂 D 均未失效）、非 Q4 升级。actor 侧是否使用 z 的细粒度分量成为核心未测环节。
- 下一步（Q2 正式设计，裁决版）：Q2.1 first-action valid-z swap（全 11 checkpoint；粗响应 ||a_dyn−a_stand||、细响应 ||a_dyn1−a_dyn2||、valid-z secant S=||Δa||/||Δz||、语义方向一致性 vs source action/target-state difference——大 action 差不等于具体动作语义）；Q2.3 同状态多步轨迹分离（50k/150k/225k/275k，1/10/25/50/100 步；核心=policy 轨迹差异是否复现 source 轨迹差异结构：corr(D_policy,D_source)、matched-vs-wrong tracking advantage、trajectory retrieval、fall/torque/action-rate）；Q2.4 actor fine sensitivity 与 D fine margin 时间对齐（双假设：同步变化→D 直接控制 actor 细粒度能力，优先修复 D 长期条件保真；actor 保留→历史遗产，来源归因留训练消融，不得直接命名 FB/relabel 遗产）。actor/B/normalizer/D 必须同 checkpoint；不用全维 Jacobian norm 作唯一结论（z 可能在受限流形上，主指标用 valid-z secant）。不开 1M。

**Lineage + D 曲线 bootstrap + Q2 actor conditioning 审计（2026-08-24 夜十一，夜十裁决执行）**：

*（1）Behavioral panel lineage*（`probe_panel_lineage.py` → `probe_panel_lineage.json`）：
- 24 个 state 是 env reset 的 motion-sampled 状态（torch multinomial over 全 motion 集 + rand phase + init noise，seed 5000+s_i）。**qpos sha 24/24 精确重建匹配**（qvel sha 因 mj warm-start 残留不可复现，已从验证标准剔除并记录）。lineage 落盘：每 state 的 motion_id/name/phase。
- 结构：**23 个唯一 motion，唯一碰撞 = stand_sweep_w001e26s0（states 20/22，均为 stand 家族）**。motion-cluster bootstrap：A retrieval@1 0.50 CI[0.29,0.71]、z-特异 tracking +1.63 CI[+0.97,+2.20]、B 均不变；**motion-cluster permutation（B−A retrieval）p=0.019** 维持。结论：state-level independence 近似成立（1/24 碰撞），夜八二项/McNemar 结论不需要改写，但保留"条件于 independence、exploratory"标注。后续面板必须 reset 时直接落盘 motion_id。

*（2）D 曲线 state-cluster bootstrap 正式化*（`probe_q15_bootstrap.py` → `probe_q15_cluster_bootstrap.json`；per-window 重算，窗口为 cluster，预定义 early 25-75k / middle 125-225k / late 250-300k，10000 reps）：
- **A fine margin 确认非单调**：middle−early +0.122 [+0.074,+0.173]（升）、late−middle **−0.086 [−0.142,−0.028]**（降，CI 不跨零）。
- **B fine margin 升后平**：middle−early +0.070 [+0.015,+0.124]、late−middle −0.006 [−0.121,+0.102]（跨零，无确认衰减）。
- 两臂 coarse margin 全程增长（4 个 contrast CI 全正）。D retrieval late 下降两臂均不显著。A/B 配对差各期均小（|Δ|≤0.16）。
- 措辞升级：descriptive→"A fine D margin 统计确认 mid 峰后衰减；B plateau；两臂 coarse 持续强化"。

*（3）Q2.1 first-action valid-z swap（全 11 ckpt × A/B，rng-606 32 窗面板，deterministic）*（`probe_q2_actor_conditioning.py` → `probe_q2_actor_conditioning.json`）：
- **两臂 actor 均未忽略细粒度 z**：fine response ||a(s,z_i)−a(s,z_j)|| 全程 1.4-3.4；粗响应 1.6-5.1。
- **语义方向一致性超 null（两臂、全部 ckpt）**：fine cos A +0.31~+0.53 / null +0.15~+0.24；B +0.29~+0.43 / null +0.13~+0.21；coarse cos 更高（0.35-0.77）。null = 随机重配 source 对。（注：pool `last_action` 全零——注入式 pool 无 expert 动作——语义参考改用 next-frame dof_pos 差，面板内 z-score。）
- actor fine sensitivity（secant 与 ||Δa||）**两臂从 25k 起单调下降**（secant A 0.60→0.12、B 0.57→0.17）。

*（4）Q2.3 同状态多步轨迹分离（50k/150k/225k/275k × 6 resets × 8 z[6 dyn+2 stand] × 100 步；全部 rollout 跑满 100 步——frac_terminated 原 1.00 是哨兵值 bug，已按双存档纪律补正为 0.0，轨迹指标不受影响）*：
- **matched-vs-wrong tracking advantage 两臂全部为负**（A −0.37~−0.61、B −0.39~−0.79；8 候选下 matched z 不比平均 wrong z 更贴自己的 source）。
- **corr(D_policy, D_source) 两臂正且随训练上升**（A 0.42→0.70、B 0.37→0.86@225k）——policy 轨迹差异复现 source 轨迹差异的**粗结构**。
- trajectory retrieval 0.15-0.23（chance 0.125），弱高于随机。

*（5）Q2 路由判读（裁决结果 A-E 对照，exploratory）*：
- 排除结果 A（actor 完全忽略细粒度 z：两臂 fine 响应+语义方向一致均超 null）。
- Q2.1 层面也不满足结果 B（只响应 activity）：first-action 的细响应带语义方向。**但 B 臂与 A 臂在 Q2.1 无分化**——A/B 行为差异（v2 面板）不在第一步动作层，在多步 rollout 层才出现。
- Q2.3 呈现**结果 C 模式**：actor 读取了 z（first-action 语义响应存在）但多步 matched tracking 失败（advantage 负、retrieval 弱）→ **转入 Q3（控制校准/动态可实现性：activity overshoot、phase drift、contact mismatch、horizon error growth）**。
- 时间对齐（双假设）：actor fine sensitivity 两臂自 25k 单调下降，与 D fine margin（A mid 峰后降、B plateau）**不同步**——B 的 actor 降而 D fine 不降，说明 actor 细粒度敏感度衰减不单由 D 条件信号驱动，支持 Q3 控制侧主嫌疑（非结果 E）。
- **矛盾待解**：v2 面板 A z-特异 +1.63 [CI 正] vs 本面板 A advantage 全负。差异点：v2 4 路 z（dyn1/dyn2/dyn3/stand 固定 w0∈[20,60]）vs 本面板 8 路 z（6 dyn+2 stand，panel 窗口任意相位）；v2 只对 src1 测、本面板对全部 8 个 query 平均（含 2 个 stand query）。下一轮需对齐两套面板设置定位分歧来源（z 相位分布 or 候选数 or 求平均方式），再定 Q3 具体探针。
- 队列：Q3（控制校准）为主分叉；Q4（D-only 四臂）保留；预注册新 seed 不变。不开 1M。

**数据消费层改组（2026-08-24 夜十二，分工裁决执行）**：canonical `final-build2`（RELEASE.json，13,619 clips，只读冻结）由数据集 agent 交付，tracker 侧此前仅消费 1,462 clips（10.7%，VERSION_CAPS 粗糙分层）。裁决：不回头改 canonical，训练侧重建消费池。
- **冻结**：`humanoidverse/data/bdx_14dof_train.pkl`（1,462 clips）标记为 **`legacy_tracker_pool_v0`**；`bdx_expert_sim_pool_full.pt`、`bdx_pretrained_cond_d.pt`、gate1 A/B 筛选 run、以及夜八至夜十一全部 Q1/Q1.5/Q2 产物**均标记 legacy-1462-pool**（结论在新池上需复验，特别是 v2-vs-Q2 面板矛盾与 Q3 路由）。
- 训练侧任务队列：①全量 13,619 NPZ 只读 dry-run（schema/长度/FK/lineage）→ ②tracker eligibility manifest（episode≥251 帧等约束）→ ③废弃 VERSION_CAPS，按 semantic family（caption/tags）+ activity + lineage 定采样权重 → ④`tracker_dataset_v1` 重建 train/eval pkl → ⑤重建 expert-sim pool + coverage/activity audit + D 预训练 + gate → ⑥新池 canonical Arm A 300k（无 curriculum）。不开 1M。

**v2-vs-Q2 面板矛盾定位（2026-08-24 夜十二补充，LEGACY-1462-POOL 产物 `probe_panel_contradiction.json`）**：
- 6 resets（seed 5000+i，v2 parity）× 4 路 z（v2 固定 episodes/windows）× 150 步 × A/B 末 checkpoint；z 相位两档（early [20,60] vs late T-300）。
- **裁决性结果：矛盾由 F1（query/平均定义）造成，不是 F3（z 相位）**。v2 单对定义（dyn2−dyn1→src1）在**两臂、两相位全为正**（A early +1.89 / late +1.02；B early +1.60 / late +2.67）；全 query 聚合定义两臂全负（−0.5~−0.65）；dyn-only 候选两臂 ≈0（−0.07~+0.01）。
- **判读（exploratory，n=6）**：v2 的"A z-特异 +1.63"来自单对 query 定义——该定义方差大且不区分臂（B 在此定义下同样为正）；聚合 matched-vs-wrong 定义下两臂均无 matched 特异性。err 矩阵行间（query z 间）差异远大于列间（source 间）——policy 轨迹到各 source 的距离主要由 rollout z 决定，matched 特异性弱。**behavioral retrieval 层的"A 有具体动作语义"结论降级为定义依赖的弱证据**，最终裁决留给 v1 池预注册新 seed 面板（用聚合定义 + lineage 记录）。

**tracker_dataset_v1 产物链完成 + Arm A 开训（2026-08-25 凌晨，夜十二续）**：
- **v1 expert-sim pool**：6,419/6,419 episodes 全覆盖、0 bad frames、0 dropped、1,694,616 帧（94.1h 采样窗），`bdx_expert_sim_pool_v1.pt`（2.07GB）。
- **pretrain cond D（v1）**：held-out dyn−stand margin **3.33**（legacy 2.68），`bdx_pretrained_cond_d_v1.pt`。
- **coverage/activity audit**（`v1_pool_coverage_activity_audit.json`）：coverage **1.0**（per-episode 1:1）；stand 家族占比 **10.5%**（legacy pool 31%）；pool 与 kinematic buffer 的 activity 四分位一致（[2.90,5.18,7.04] vs [2.69,4.78,6.92]）——无采样偏置。版本混合：intervene 2995 / perturb 800 / walk 700 / patch 400 / backward 341 等，符合配额。
- **v1 Arm A 300k 已开训**（`results/v1-armA-300k`，seed 4728、cond 0.3、reward_norm、pool+预训 D 挂载确认、无 curriculum）。插曲：audit 首跑与训练抢 GPU OOM——已改 CPU 设备重跑成功（audit 不占训练 GPU 的规矩记入）；训练本身未受影响。
- 下一步：Arm A 跑完后按夜十二标准 gate 检查（不开 1M）。

**v1 Arm A 300k 完成 + 训练 gate 通过（2026-08-25 晨，夜十二终）**：
- **训练完成**：`results/v1-armA-300k`（300k 步，~22 分钟 @ ~1000 FPS，11 个 25k 快照齐全，seed 4728 canonical 复刻：cond 0.3 / reward_norm / v1 pool + v1 预训 D 挂载确认、无 curriculum）。
- **训练健康**：reward 轨迹与 legacy-A 同型（batch-std 制下 expert_z 至 -1.71 vs legacy -1.58；p50 全程 -0.2~0；reward_std=1.0；Q1 末期 +4.95 无 Q 崩溃；disc_cond_loss 1.24→0.80）。50k 判定点无崩溃签名。
- **Gate-1（v1 pool，fresh-D，`expert_sim_pool_gate1_v1.json`，对照通过的 legacy gate-v4）**：purity local_body_pos raw p50 5.7e-4 / frac>0.1 = 0（legacy 1.5e-4/0）✓；control regen1-vs-regen2 holdout AUC 0.516（legacy 0.55，≈0.5 通过）✓；domain_B_warp AUC 1.0（与 legacy 同，pool 在 sim 域）✓；self-consistency train/heldout AUC 1.0/0.99（legacy 0.965/0.791——数值残留可分性，非纯度问题，gate 判据为 purity+control+domain）。
- **v1 重建链全部完成**：dry-run→eligibility→pkl→pool（coverage 1.0）→audit（stand 10.5% vs legacy 31%）→pretrain D（margin 3.33）→Arm A 300k→Gate-1。下一步待裁决：v1 面板上跑预注册 behavioral 面板（聚合定义 + lineage 记录）判读 A/B 语义问题；不开 1M。
- 工程教训：①audit/gate 类离线检验一律 CPU 设备跑（首跑与训练抢 GPU OOM）；②gate 脚本必须带 `BFM_POOL_DATA` 指向 v1 pkl，否则 legacy buffer 索引 v1 motion_id CUDA 越界崩溃。

**v1 Arm A behavioral 面板（2026-08-25 晨，`probe_v1_behavioral_panel_armA.json`，tracker_dataset_v1 产物）**：
- 首跑 bug（双存档：`probe_v1_behavioral_panel_armA_bug_nomotionload.json` + `correction_manifest_v1_panel.json`）：探针漏 kbuf 预载，CPU build(1) 的 runtime motion_lib 只持 1 条 motion，24 resets 全落 motion 0。**训练不受影响**——train.py 在 256-motion runtime 批之后全量加载 kbuf（v1_armA.log 01:39:40 "Loaded 6419 motions"），训练采样覆盖全部 6,419 条。修正后 lineage 24/24 唯一。
- **结果（v1-A，单臂 exploratory，聚合定义为主）**：aggregate matched-vs-wrong（4 路）**-0.002 [-0.27,+0.29] = 0**；dyn-only 候选 +0.010 [-0.26,+0.29] = 0；trajectory retrieval 0.271 vs chance 0.25（CI 触 chance）；**v2 单对参考 -0.27 [-0.70,+0.08]——legacy A 的 +1.63 单对优势在 v1 数据上不复现**（与 F1 教训一致：单对定义不稳定）。
- **判读**：v1 数据（static 6.5%、全家族覆盖、coverage 1.0）上 Arm A 的 behavioral 层 z-特异 tracking 仍为零——数据重构本身没有带来具体动作语义；与 Q2 结论（actor 首步读 z、多步 matched 失败 → Q3 控制侧）一致。B 臂尚未在 v1 训练；A/B 对比面板待 B 臂或裁决。

## Night-13-bis (2026-08-25 上午): GPT 复核四项修正 + H3 z 时序契约审计 + Q3 schedule/扰动实验

### 裁定接收
旧正面结论正式撤回（"+1.63" 为单对定义伪迹）；v1 数据重建正确措辞 = "数据覆盖不足不是充分解释"；Q3 前插入 H3（z 时序契约错位）。

### 四项修正（全部完成）
1. **contradiction JSON 修复**：原件保留，`probe_panel_contradiction_repaired.json`（raw_decode 拆分两个拼接对象 + repair_note）
2. **sem_cos_fine CI**（probe_q21_semcos_ci_v1a.py，运行中自动落盘）：bootstrap CI + permutation null，11 ckpt 全曲线。前 4 ckpt 已显示 sem_cos_fine 随训练**衰减**（0.49→0.22），perm p 0.014→0.063（失去显著）——单步细粒度响应是早期暂态
3. **v1-A D 曲线**（probe_q15_d_curves_v1a.json）：fine margin 单调增 early 0.43[0.31,0.55] → mid 0.71[0.55,0.87] → late 1.24[1.02,1.45]，CI 不重叠；coarse 同趋势；D 层条件信号存在且增强
4. **cond_d_v1 provenance**（provenance_check_cond_d_v1.json）：文件内部无 manifest（缺口已记录）；过程级（v1_chain.sh env + mtime 链）+ 内容级（v1-only motion ids 1462-6418 上 matched-vs-mismatched margin=0.205，legacy-only 训练的 D 不可能配对未见 motion）→ **CONFIRMED v1**

### z 生命周期审计（z_lifecycle_audit_v1a.json）——决定性先验
- 训练 z 双轨：50% envs = expert tracking rollouts，**z 逐步移动**（z_t = project_z(mean B(source[t:t+8]))，前向 8 帧窗）；50% envs = z_buffer 静态 100 步但**与 episode motion 无关**
- v1-A 未启用 BFM_REF_Z_ALIGNED（log: ref_curriculum=False）→ **"matched static z 长时保持"在训练中从未出现**；matched z 永远是移动的
- z 是局部相位描述子（前向 8 帧窗），encoder LOO 无跨 phase identity 与此一致
- **结论：H3 有强先验，behavioral 面板失败不能先归因控制侧**

### Q3 schedule 四臂（probe_q3_z_schedule_arms.json）——H3 被否定
12 resets × 9 条件 × 150 步。err@100: S=13.68, O=14.21, C4=14.21, C8=14.43, C16=14.58, P_delay=14.31, P_shift=14.16, P_other=14.11, P_perm=14.56。
- **O vs P_other（配对）= -0.09 [-1.31, +1.14]**：oracle 逐步移动 z（完全复刻训练契约）相对喂错误 motion 的 z **零收益**
- 所有臂同一模式：h8≈6 → h100≈14（z 无关的发散）；phase err O=14 vs S=18（微弱）；fall=0
- **判读（按裁定解释树）：即使给正确时序的 z 也跟不住 → H3（接口契约错位）作为充分解释被否定；无扰动也跟不住 → H1/H2 路由**

### Q3 扰动-回拉（probe_q3_perturb_recovery.json）
8 resets × {no_pert, q_kick, root_kick} × {correct_z, wrong_z}，oracle schedule，t=50 踢。
- z-specific recovery（wrong−correct d55_80）：no_pert -0.60 [-2.27,+1.18]，q_kick +0.08 [-1.43,+1.44]，root_kick +0.68 [-2.15,+3.05]——全部跨零，**细粒度 z 无纠偏方向**
- wrong_z 的 e50 系统性低于 correct_z ~1.5-2.5（与面板负 advantage 一致）
- 注意：kick 幅度相对已巨大的基线误差（16-18）偏小，recovery 测试饱和；但基线本身就跟不住是主导事实

### 当前路由
H3 否定 → H1（可实现性）/ H2（闭环 sustain）。下一步 = 特权可实现性基线（PD/reference controller 用完整 q/dq 跟踪同一 source）：特权控制器也跟不住 → H1；能跟 → H2/条件训练目标问题。暂不训练 B，不开 1M。
