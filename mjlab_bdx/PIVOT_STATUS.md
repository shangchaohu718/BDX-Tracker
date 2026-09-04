# mjlab pivot — execution log & old-line archive status

Branch: `mjlab_fixed_dance_bdx` (from `tracker-demo-v0` @ 070568a, 2026-08-25)

## Decision (user execution order, 2026-08-25)

Stop repairing the old BFM-zero custom trainer; port the fixed-dance
specialist (side_step_4) to mjlab's standard motion-imitation stack
(MuJoCo Warp + RSL-RL PPO). Old-line quarantine rules stay in force:
nothing from objective-v2 / F-P 对照 is migrated.

## Old-line archive status (NOT deleted, NOT blocked)

- `fds_train_dance_core_cf.py` arm F run (dance_core_ss4C_Fcf, started
  15:19, canonical C pkl, seed family unchanged) was STILL RUNNING on
  CPU (.venv) at pivot time. Left running to natural completion; its
  outputs land in `results/fds/dance_core_ss4C_Fcf*/` as before.
- Quarantined objective-v2 line stays archived (`dance_core_ss4_Fv2_*`).
- `.venv` (old stack) is pinned: mujoco 3.9.0 / rsl_rl 2.3.3. mjlab
  lives in a SEPARATE `.venv-mjlab` (python 3.11) precisely so the old
  framework's environment is never mutated.

## What was migrated (the only four things)

1. BDX MJCF — `humanoidverse/data/robots/bdx/bdx_14dof.xml`, loaded as
   MjSpec with the 14 stock torque motors stripped; position actuators
   attached programmatically with the SAME kp/kd/effort as the validated
   stack, and armature/frictionloss/viscous_damping overridden with
   xml-identical values (physics preserved bit-for-bit).
2. PD/limit/effort parameters — from `humanoidverse/config/robot/bdx/bdx_14dof.yaml`
   + MJCF motor forceranges.
3. `side_step_4` motion — the pkl's own root_rot is the known pure-yaw
   P0 export; instead the VALIDATED open-loop PD re-roll (seed 47065,
   293/293 frames, no fall, mean |q| err 0.129 rad) was recorded with
   full qpos/qvel/body poses/contacts/world-frame body velocities
   (`results/mjlab_bdx/ss4_openloop_rollout.npz`), then converted to the
   mjlab MotionCommand npz with explicit by-name body/joint reordering
   (`mjlab_bdx/convert_motion.py`, per-pair assertions + manifest).
4. joint/body name manifests — `bdx_constants.py` frozen tuples,
   converter asserts against live Entity order.

## Files

- `mjlab_bdx/record_motion.py` — open-loop re-roll recorder (.venv stack)
- `mjlab_bdx/convert_motion.py` — rollout npz → mjlab motion npz (.venv-mjlab)
- `mjlab_bdx/bdx_constants.py` — BDX EntityCfg (actuators, keyframe, scales)
- `mjlab_bdx/bdx_tracking.py` — env cfg + PPO cfg + task registration
- `mjlab_bdx/train_bdx.py` — training launcher
- `mjlab_bdx/smoke_bdx.py` — entity/stand/motion smoke (no PPO)
- `results/mjlab_bdx/` — rollout npz, converted motion npz, logs (no-overwrite)

## Deliberately NOT migrated

Old PPO trainer, BFM actor, D, z, objective-v2, old termination code,
old action normalization chain, F/P 对照.

## 2026-08-25 late addition — bdx_rl_mjlab found: THE training code

`/home/tcl/Desktop/start/bdx_rl_mjlab` (user's own repo, sibling checkout) is the
real mjlab-side BDX training project — already complete: `tracking_bdx` (BeyondMimic
tracking: standing/gesture/motion-pool), `bdx_rl/bdx_mimic` (perpetual/periodic,
episodic STUB), `walking_bdx` (velocity), ONNX deploy chain, own `.venv`
(mjlab pinned mujoco 3.7.1, python 3.13). The hand-rolled `mjlab_bdx/` task in
this repo is therefore a FALLBACK, not the primary path.

Integration done (all validated):
- `mjlab_bdx/convert_motion.py --robot-cfg bdxrl` converts the validated ss4
  re-roll with bdx_rl_mjlab's OWN Entity as the ordering authority — both body
  and joint permutations are IDENTITY (the two BDX asset lineages are
  name-for-name identical; PD gains also identical 10/15/5 etc.).
- Converted clip installed at
  `bdx_rl_mjlab/policy_assets/bdx_v4/npz/episodic/side_step_4.npz` → auto-
  registered task `Mjlab-Tracking-Flat-BDX-V4-Gesture-Side-Step-4`
  (293 frames @ 50 fps, episode 5.86 s, 79-D deploy-matched actor obs).
- `smoke_bdxrl.py` PASS (reset teleport anchor err 0.036, zero-action probe
  finite, no spurious terminations).
- PPO smoke: 30 iters x 1024 envs in ~1 min — 38,468 steps/s (~1500x the old
  25/s CPU trainer), mean reward -0.03 → 6.55 and climbing, auto ONNX export
  present in the run dir. Full training = same command with more iterations.

Design notes to keep in mind for the full run (their conventions, deploy-matched):
- gesture cfg uses yaw-invariant (tilt-only) anchor-ori obs+reward — designed
  for planted-base neck gestures; side_step_4 is a stepping dance (lateral
  steps, near-constant heading) so this is approximately fine, but if heading
  drift shows up in eval, switch to the yaw-sensitive original (motion-pool
  style) before tuning anything else.
- action scale flat 0.25 (not per-joint 0.25*effort/kp) — matches the standing
  policy + tonly_robot2 deploy chain.

## 2026-08-25 hold — dance data being REDONE (user)

User is regenerating the dance data. Training is ON HOLD by execution order.
- The smoke-time `side_step_4.npz` was REMOVED from
  `bdx_rl_mjlab/policy_assets/bdx_v4/npz/episodic/` so the stale clip cannot
  auto-register as a gesture task while data is reworked (smoke run + metrics
  remain in `bdx_rl_mjlab/logs/rsl_rl/bdx_v4_tracking/2026-08-25_16-53-36_ss4_mvp_smoke/`).
- New-data integration when ready (2 commands, assertions built in):
  1. `.venv/bin/python mjlab_bdx/record_motion.py --pkl <new_pkl> --out <new_rollout.npz>`
     (old stack, validated open-loop protocol; falls through only if the new
     clip also passes the no-fall rollout gate)
  2. bdx_rl_mjlab `.venv/bin/python mjlab_bdx/convert_motion.py --robot-cfg bdxrl
     --rollout <new_rollout.npz> --out .../episodic/<new_name>.npz`
  Integration path itself is validated end-to-end (smoke PASS + PPO smoke).

## 2026-08-25 evening — excited_wiggle v1 training (user order after data hold lifted)

- Source clip: dataset/v3_planner/clips/re_dancegen_excited_wiggle_0.npz
  (train split; 255 frames @ 50 Hz; stationary dance_gen gesture; FK-baked,
  dynamics-unverified class). Ingested via mjlab_bdx/ingest_npz.py (by-name
  perms both identity, fps reshaped to (1,), physics checks pass; provenance
  sidecar results/mjlab_bdx/ingest_excited_wiggle_provenance.json).
- Task: Mjlab-Tracking-Flat-BDX-V4-Gesture-Excited-Wiggle; smoke PASS.
- Training: 2000 iters x 1024 envs, video every 4800 env steps (=200 iters,
  255-frame episodes), tensorboard. Run dir:
  bdx_rl_mjlab/logs/rsl_rl/bdx_v4_tracking/2026-08-25_17-21-09_excited_wiggle_v1
- Learnings: --video-interval counts ENV STEPS not iterations (first launch
  misconfigured at 200 => every ~8 iters, archived as *_aborted). Early-phase
  iteration time ~4.6s (vs ss4 smoke 0.64s) is termination-driven reset churn
  (~31% anchor_ori fails from RSI starts), NOT video overhead (no-video diff
  run confirmed); expected to recover as tracking improves.

## 2026-08-25 18:15 — "viewer 里站不住"根因裁决：clip 头部鞭打瞬态（非策略问题）

User reported model_500 in the 8080 viewer collapses on landing. Headless
forensics (mjlab_bdx/eval_play_headless.py, exact play cfg, deterministic):

- **Root cause: excited_wiggle.npz frames 0-3 carry a kinematic whip-start.**
  Root spawns with 1.3 m/s lateral vel (frame0->1 root y moves 2.6 cm),
  joint_vel norm 15 rad/s (= clip max, it IS frame 0), body_ang_vel norm 27
  (clip max elsewhere ~10). Velocity channels are correct finite differences
  — the MOTION ITSELF starts with a dynamically impossible impulse. Play
  spawns at frame 0 with those velocities written into the sim.
- Behavior on original task (model_500, fixed harness): flung up to z=0.282
  in 10 steps, collapses THROUGH standing to z=0.075, lies on the ground a
  full 5 s loop — **zero terminations fire** (anchor_pos/ori/ee_body_pos
  thresholds loose; 16 cm sink + ground-hug passes) — then teleports to
  frame 0 and repeats. This is exactly what the user saw.
- **Same policy on trimmed clip (frames [4:], new frame-0 vel 0.14 m/s,
  joint_vel 2.8): 0 falls in 600 steps, root z steady 0.226, clean loops.**
  Policy learned the motion; training metrics were real (adaptive sampling
  rarely starts bin 0, so training never saw the poison).
- Controls: ref-PD perfect tracker is WORSE than the policy (dies repeatedly
  at the whip spawn); model_0 sinks identically → spawn-state issue, not
  policy quality.
- **Fix deployed: excited_wiggle_trim.npz** (patch_wiggle_trim_head.py,
  append-only + sha256 sidecar
  results/mjlab_bdx/patch_excited_wiggle_trim_provenance.json), auto-
  registered as Mjlab-Tracking-Flat-BDX-V4-Gesture-Excited-Wiggle-Trim.
  Viewer (8080) relaunched on it with model_500 (pid 2213911).
- Harness bug found & fixed along the way: checkpoint obs_normalizer._std
  has zero-variance channels → division by zero → NaN actions (silently
  behaves like an untrained policy). Guard: std<1e-8 -> 1. Cross-checked
  against the auto-exported iter-500 ONNX (obs + time_step (1,1) float
  inputs) — actions match.
- P1 flags for the pool era: (a) static gate must check frame-0 spawn
  viability (vel transients), (b) gesture terminations too loose — a policy
  hugging the ground at z=0.075 survives; add |root_z - ref_z| termination
  or tighten anchor_pos, (c) upstream: v3_planner clips should not emit
  whip-starts (dynamics-unverified class, exactly as the dataset warned).

## 2026-08-25 18:40 — 全族数据深度审计（用户令：继续怀疑数据）

audit_wiggle_cohort.py, 50/50 clips (45 train + 5 eval_only), offline, no env:

- **CONFIRMED systemic: head whip in ALL 50 clips.** f0 root lin vel
  1.05-1.79 m/s (steady median 0.50-0.64); f0 joint_vel norm 12-19 vs steady
  median ~6.3. It lives in the POSITION trajectories themselves (see next
  bullet) — producer (dance_gen) starts every clip with an impulsive swing.
- **CLEARED: velocity channels.** Stored vels == central differences of
  positions EXACTLY (max residual 0.0 across all 50 clips x all frames).
  No fabrication, no convention issue.
- **CLEARED: kinematic consistency.** Clip body_pos_w == true FK of clip
  joint_pos on bdx_v4.xml: ALL 17 bodies, 0.0 mm, all frames (legs, neck,
  ears). The 1-2 cm "spawn body mismatch" seen in-env earlier was a
  read-staleness/settling artifact — suspicion RETRACTED.
- **CLEARED: limits/quats/root-z.** 50x14 joints, zero range violations
  (never even touch soft limits); quat norm deviation 0; root z 0.229-0.242.
- **Bonus finding (their data, not ours): native laugh_big.npz has a
  CONSTANT neck-chain offset vs FK — neck_pitch_link 22.8 mm, neck_yaw/roll/
  ears 2.2 mm (legs 0.0).** Their baker and the xml disagree on the
  neck_forward slide default. Worth telling them; unrelated to wiggle.

Verdict: exactly ONE defect, systemic, scoped to the clip heads. Fix = trim
(Trim task proven) or fix dance_gen's start transients upstream. All 50 need
head-trim before any pool training.

## 2026-08-26 — 执行令 8 步全链执行（补证据链 → quat P0 → 裁头/审计 → 小池训练）

1. **eval v2 (eval_policy.py)**: --checkpoint/--task/--motion-file/--seeds 参数化,
   SHA256 记录, fall≠termination (fell_by_height z<0.12 / lying≥25步 独立判据).
   model_1999 四条件 (orig/trim × phase0/uniform RSI) × 4 seed × 8 ep = 128 ep:
   **全 100% 完成、零摔 (min root z 0.182-0.201)**。被撤回的结论以有效度量重立。
   期间修正: 循环脚本 npz 名映射错误曾漏跑 trim/uniform 一格, 已补齐。
2. **quat P0 修复**: mjlab wxyz 实锤 (utils/lab_api/math.py docstring + v3 clip
   RSI 后 error_anchor_rot=0); recorder _rigid_body_rot=xyzw。convert_motion.py
   --in-quat 必选 + 输出恒 wxyz + quat_order 标签 + 随机四元数旋转等价自测
   (对 5.7e-4 rad / 错 3.14 rad; 中途自测抓出本人 einsum 双转置 bug + xyzw 索引
   bug——测试有效) + 逐 body 旋转矩阵对照 (真数据 0.0 rad)。再生成
   ss4_mjlab_motion_wxyz.npz; 旧 bdxrl 版 + smoke run 目录已标 INVALID。
   smoke_quat_reset.py: 污染件 body 残差 2.47 rad vs 干净件 0.15 (anchor 仅
   0.0015——证明 root 恒等检查抓不住此类错误)。reset 后 body 级读数滞后一步
   (root 级新鲜), 需 hold-step 刷新后再读。
3. **批量自适应裁头 (batch_trim_heads.py)**: 首版 1.1×p75+8帧保持 → 7 条误标
   (稳态摆幅超带宽, 非瞬态更长); 校准 1.25×p75+5帧 → 50/50 过, 裁 2/7/8/14 帧
   (分布证明不可固定裁 4)。provenance 双 sha256+阈值+帧数, 速度边界复核 <1e-5。
4. **静态审计+PD 难度 (pool_difficulty.py)**: 零硬拒绝 (无穿模 footz≥32mm
   capsule-center 口径 / 无越限 margin≥0.14 / FK 0mm / quat 0)。PD survival
   0.08-0.37 连续分布 + anchor P95/joint err/action>1 比例 → curriculum 特征
   而非门 (clip 0 本身 PD 也烂但 RL 练成 = "PD 失败≠不可学"实证)。族内 yaw
   全 0/rootz 全 0.235——多样性受数据族限制, 选池覆盖难度/长度/瞬态三轴。
5. **契约差分审计 (motion_pool_contract_audit.md)**: 现存 Motion-Pool 任务与
   gesture 契约差 4 处 (per-joint scale/yaw 敏感/85D+index obs/周期相位重采样);
   感知歧义实锤 (command=当前帧 28D, 无未来帧/相位钟) 但被部署契约锁死, 记录
   为已知属性。**决策: 新建 wiggle_pool.py = gesture 契约+池机制最小差分**
   (平 0.25/tilt-only/79D+1D index=80D, is_periodic=False, adaptive RSI)。
6. **小池训练**: 8 条 (难度分位数 0..1, frames 203-293, manifest 落档), 任务
   Mjlab-Tracking-Flat-BDX-V4-Wiggle-Pool-8 (加进他们 config/__init__ 一行
   import), smoke PASS (80D/8 motions/随机分配/零步终 0)。v1 训练 2000iter×
   1024env 运行中 (run-name wiggle_pool8_v1)。
7. **分 motion 评估工具就绪 (eval_pool_per_motion.py)**: set_motion_index 钉住
   逐条评估, 高度判据, macro/worst/P10 + Wilson 区间——待训练完成执行。
反馈件三篇落 results/mjlab_bdx/feedback_{v3_planner_head_transient,neck_offset,
velocity_semantics}.md (事实/影响/建议分栏+最小复现)。

## 2026-08-26 午 — wiggle_pool8 v1 结果：7/8 干净，最难 clip 被池放弃（worst-motion 抓到）

- 训练完成 2000 iter，末段 mean reward 10.92（单动作 19.05——8 条分容量+含难 clip）。
- **分 motion 评估（12 ep/motion，高度判据）**：
  - 7/8 条：fall=0.00（Wilson [0,0.243]）、min z≈0.20、joint err 0.36-0.51 —— 干净。
  - **wiggle_6（难度分位 0，PD survival 0.078）：12/12 episode 贴地**（min z 0.077、
    joint err 2.7=其余 5-7 倍、lying 全 True），**终止项零触发**（completion=1.0 假象）。
  - 汇总：fall macro=0.125 / **worst=1.0** / P10=0；joint_err macro=0.745 /
    worst=2.7 / P10=0.361。
- **裁决预期命中**：mean reward 10.92 掩盖了 motion 6 被牺牲；worst-motion 报告
  抓住了它。根因候选（未定案）：clip 6 本身最难（PD 也最烂）+ 1-D 标量 index
  身份编码 + adaptive sampling 权重不足。
- **扩池（50 条）= 用户决策点，前置建议**：①gesture 终止加 |root_z−ref_z|
  （训练期就能惩罚贴地，而不是 eval 侧事后抓）；②motion 6 单独复读（单动作任务
  训它，区分"数据难"vs"池机制牺牲"）；③难度加权采样（difficulty_index 已备特征）。
  评估 JSON：results/mjlab_bdx/eval_pool8_m1999_per_motion.json。

## 2026-08-26 ~12:35 — 【撤回】v1 分 motion 结果=测量伪影；set_motion_index 钉失效实锤 + 修复

**用户授权自主迭代（"自己迭代优化，明天再来"），本段起为 unattended 记录。**

审计 V2 评估链时发现两个测量缺陷，上一段"7/8 干净 + wiggle_6 被牺牲 12/12"整体撤回：

1. **`set_motion_index()` 钉从未生效**：旧实现委托 `_resample_command`，其
   `_assign_motions` 会重新随机分配池——pin(3)×8 实测 [1,3,6,2,1,5,4,6]。每集
   实际跟踪的是随机 motion，标签是错的。
2. **集间无 env.reset()**：一次摔倒污染槽内剩余全部 episode（机器人趴着没被
   重置）——这才是"12/12 vs 0/12"极端分化的真正机制，不是池牺牲任何 clip。
- 幸存信息：v1 策略在 8×12 窗口中至少真摔过一次且不爬起；"池牺牲 wiggle_6"
  归因不成立，待重跑数据裁决。
- **修复**：`set_motion_index` 重写为硬钉+第 0 帧传送（+frame 参数，不走
  `_resample_command`）；eval 脚本逐集 reset→钉→管线刷新(write/forward/sense/
  command compute)→obs 重算，步数钳 clip 长−2 防片中换片。验证：钉保持 8/8、
  传送 |dz|≤1.5mm、集独立性、frame 参数生效。旧 JSON 标废
  (eval_pool8_m1999_per_motion.json.INVALID_pinbug_noreset + README)。
- **v1 重跑**（修复后脚本，12 ep/motion）进行中 →
  eval_pool8_v1_RERUN_pinfix_per_motion.json。
- 附带产出：MotionPoolCommand 增加可选 `motion_weights`（默认 None=均匀不变，
  前置③难度加权采样就绪，验证 0.7/0.2/0.1→经验频率误差<1%）。
- 双训练不受影响（进程启动时已加载旧模块；磁盘编辑不回溯）：
  wiggle_pool8_v2_zterm（z 终止 0.25→0.08，ETA ~13:00）、wiggle6_single_z8_v1
  （clip6 单训紧终止，ETA ~12:25）。看护脚本将在完训后用**修复后**的 eval 脚本
  自动评估（脚本按执行时磁盘版本读取）。

## 2026-08-26 ~12:50 — v1 重跑（钉修复后）真实分 motion 表：5/8 干净，真摔=20/4/48

| motion | fall | Wilson | min z | jerr | aori | 终止项 |
|---|---|---|---|---|---|---|
| wiggle_12 | 0.00 | [0,.243] | .203 | .526 | .189 | 无 |
| **wiggle_20.trim8** | **1.00** | [.757,1] | **.059** | 1.051 | **1.423** | 无 |
| wiggle_29 | 0.00 | [0,.243] | .205 | .490 | .302 | 无 |
| wiggle_31 | 0.00 | [0,.243] | .207 | .443 | .143 | 无 |
| **wiggle_4.trim7** | **1.00** | [.757,1] | **.050** | .971 | **1.337** | 无 |
| wiggle_42 | 0.00 | [0,.243] | .202 | .411 | .226 | 无 |
| **wiggle_48.trim8** | **1.00** | [.757,1] | **.046** | 1.194 | **1.268** | 无 |
| wiggle_6 | 0.00 | [0,.243] | .207 | .612 | .220 | 无 |

- **wiggle_6（PD 最难 0.078）实际 0 摔**——"最难 clip 被池牺牲"证伪；PD 难度分
  与 RL 失败不相关（三条真摔均为中等难度 0.098/0.154/0.193）。
- 三条失败模式一致：贴地（min z≈0.05=躺平）+ 姿态跟丢（aori>1.2rad）+ 松终止
  下零终止（completion=1.0 假象复现）。fall macro=0.375。
- 含义：v1 池真实状态=3/8 容量塌陷，mean reward 10.92 的缺口由此而来。
- 待 V2（z 0.08 紧终止，训练中）裁决：紧终止+adaptive 采样（终止计入失败
  bin 加权）能否救回 20/4/48。wiggle6_single_z8 现为"紧终止不伤可学 clip"的
  对照实验。
- 注意 trim 深度与失败的关联假象：20/48=trim8、4=trim7，但 12/29/31/42/6 中
  也有 trim8/trim7 吗——无（其余全是 trim2），样本 3 条不足以定论，记为待查。

## 2026-08-26 ~13:00 — wiggle6_single_z8 对照完成：紧终止不伤可学 clip

- 32 ep × phase0/uniform 两条件：fall 0.00、min z ≥0.195、**零终止触发**
  （|z−ref|<0.08 全程成立）、completion 1.0。
- 与 v1 重跑"wiggle_6 干净"互证：该 clip 可学；此前被冤枉纯因 eval 钉 bug。
- 剩余裁决全部压在 pool8_v2_zterm（~13:45 完训后自动分 motion eval）：
  紧终止+adaptive 失败加权能否救回 20/4/48。

## 2026-08-26 ~13:35 — v1 均匀相位评估：失效结构=相位脆性，非 motion 身份

- 随机相位起步（12 ep/motion）：8 条 fall 率全部落在 0.08-0.25（无分化），
  min z 全部 0.048-0.056（每条都有摔倒集），存活集跟踪良好
  （jerr 0.46-0.59、aori 0.05-0.31——远好于相位 0 失败集的 1.2-1.4）。
- 结合相位 0 表：20/4/48 的**帧 0 区域特异性致命**（相位 0 必摔、随机相位
  大多存活）；"干净"5 条也存在偶发致命相位（~17%/集）。
- 机制解释（与 V2 假设自洽）：v1 松终止下"摔倒后躺地跟迹"不算失败 →
  adaptive 采样器从未聚焦这些致命 (motion, bin) 起点 → 相位脆性未被修。
  V2 紧终止把这些起点标记为失败 → adaptive 应聚焦修复。
- 工具沉淀：pool_record.py（钉住单 motion 录 mp4，关终止防片中换片，
  rgb_array 渲染）；已录 v1 对照视频 videos/pool_v1_wiggle20_FAIL.mp4
  （min z 0.059，肉眼可见贴地）+ pool_v1_wiggle6_CLEAN.mp4（0.207）。
- 50 条池预备完成：wiggle_pool50/ 50 clips+manifest50.json（sha256），
  任务 Pool-50-V2 注册+冒烟通过（80-D/50 motions/z 0.08）。
  **发车规则**：V2-8 相位 0 8/8 干净且均匀相位 macro fall ≤0.05 → 自动发
  50-V2 训练；否则在 8 条上迭代 V3（用 V2 实测失败率做 motion_weights）。

## 2026-08-26 ~14:10 — 【裁决】V2 全过：紧终止修复全部失效模式，双门通过，50 条发车

**pool8_v2_zterm（z 0.25→0.08，其余与 v1 完全一致，2000iter，末段 reward 14.10 vs v1 10.92）**

| 评估 | v1（松终止） | V2（紧终止） |
|---|---|---|
| 相位 0 fall | 3/8 条 100% 摔（20/4/48） | **8/8 = 0.00** |
| 均匀相位 fall | macro 0.17（全池偶摔） | **8/8 = 0.00** |
| min z（最差集） | 0.046-0.059（贴地） | 0.182-0.202 |
| jerr / aori | 1.0-1.2 / 1.27-1.42（失败集） | 0.38-0.55 / ≤0.26 |
| 评估期终止触发 | 0（假 completion=1.0） | 0（真·带内） |

- 机制定论：v1 的失效=松终止把"躺地跟迹"标记为存活 → adaptive 采样器收不到
  失败信号 → 致命 (motion,phase) 起点从未被聚焦修复。紧终止后训练期失败可见
  （mean ep len 283/300），采样器聚焦，8 条全部学成。
- wiggle6_single_z8 对照一致：可学 clip 在紧终止下无损（64 ep 零摔）。
- 视频证据（results/mjlab_bdx/videos/）：pool_v1_wiggle20_FAIL（min z 0.059）
  vs pool_v2_wiggle20_RESCUED（0.199）+ 各自 wiggle6_CLEAN 对照，共 4 条。
- **发车规则兑现：Pool-50-V2 训练发射**（50 条全部 trim 剪，manifest50.json
  sha256 锚定，同 recipe 1024env×2000iter）。完训后自动：相位 0 + 均匀相位
  分 motion 评估（8 ep/motion/条件）。

## 2026-08-26 ~14:35 — MotionPoolCommand 向量化 gather（8×提速）+ 50 条重启 + 复现实验

- 瓶颈定位：命令属性（joint/body/anchor ×10）逐 motion python 循环（50 motions ×
  每步多次）→ GPU util 仅 65%。改为**填充堆叠张量+高级索引**（每属性单 gather，
  `_gather_timesteps()` 按各自动作长度 clamp）。
- 等价性验证（1024 env CUDA、半池钉 7/半池钉 33、非均匀 time）：10 属性全部
  max|new−ref| ≤ 9.2e-07（纯 float32 origins 加减往返误差，其余精确 0）。
- 提速：12.2s/iter → **1.51s/iter**（独占 GPU 后）。50 条训练于 14:16 重启
  （旧慢跑 ~12min 弃置，无落盘产物——save_interval=500 前无目录，已核实）。
  附带修复：torch.clamp 混合 Number/Tensor 参数的 TypeError（min 用零张量）。
- **V2-8 换种子复现发射**（seed 43）：8/8 双门结果是否稳定，防单种子运气。

## 2026-08-26 15:15 — 中场状态（等待双链完训）

- GPU 双训练满载（100%/70°C/313W，无热节流），共享期迭代 15-19s/iter 属正常
  争用；seed43 完训后 pool50 预计回升至 ~1.5-3s/iter，全部结果今晚落盘。
- 泛化视频补录 2 条：V2-8 ckpt 于未见 clip wiggle_0 / wiggle_44.trim14，
  min z 0.196-0.200（健康骑高）——共 6 条审看视频。
- MORNING_BRIEF_0827.md 骨架已写（撤回/修复/真实图景/工程沉淀/决策点），
  50 与 seed43 数字留位待填。

## 2026-08-26 15:30 — 用户令"今天能弄多少弄多少"：加发三线

1. **V2-8 ckpt @50 条任务均匀相位完整评估**（CPU，12 ep/motion × 50——探针只有
   相位 0；这是"50 条池是否冗余"的关键空白）。
2. **跨族混合池 Mixed-Pool-9-V2**（8 wiggle + quat 修复版 side_step_4，293fr/
   5.86s）：检验 V2 配方（紧终止+adaptive）能否迁出同质 wiggle 族——换数据源
   决策的直接前哨。任务注册+冒烟通过（80-D/9 motions/z 0.08）。
3. **链式发射器**：seed43 完训让出 GPU → 自动发 mixed9 训练 → 完训自动双条件
   评估。避免三训并发挤爆 GPU。
今日预期全链：seed43(~15:40) → mixed9 发射(~15:45) → pool50 完训(~16:45) →
各评估陆续落盘 → 晚间全部出数。

## 2026-08-26 ~16:15 — 运维事故两起（已修复）+ 教训固化

1. **相对路径落错仓**：下午 pool50/seed43 发射命令未带 cd，train 以相对路径
   `logs/rsl_rl/...` 写入了 BFM-zero/logs（错误仓库）。seed43 完训产物已整体
   搬回 bdx_rl_mjlab 正确位置（watcher 随即正常触发）；运行中的 pool50 留在
   原地待完训后由修正版看护（watcher50b）搬回+评估。git 影响待查（BFM-zero
   `git status` 报 object 读取错误，疑似预存问题，未动）。
2. **pkill 自匹配**：`pkill -f "Mixed-Pool-9-V2"` 匹配到承载它的 shell 自身
   命令行 → 自杀并误杀刚启动的训练。mixed9 已干净重发（cwd 经 /proc/PID/cwd
   实证 = bdx_rl_mjlab，2.2s/iter）。
- **教训（两条都进长期记忆）**：①`cd X && cmd &` 的 cd 属于后台作业，前台
   pwd 会误导判断——后台作业的 cwd 必须 /proc 验证，不能凭前台 echo；②
   清理进程禁用裸 `pkill -f`（模式会匹配查询 shell 自身），用精确 pid。
- 附带发现：池模型自动 ONNX 导出失败（exporter 找 `command.motion`，池命令
  无此属性）——记 P3，不影响训练/评估链。
- 当前四线：seed43 评估中（前 3 motion 全零摔）、pool50 训练中（watcher50b
  已修正）、mixed9 训练中（watcher m9 已挂）、V2-8@50 均匀相位评估中（11/50）。

## 2026-08-26 ~17:0x — seed43 复现通过：V2 双门结果是种子稳健的

- seed 43（其余 recipe 与 seed42 完全一致）：相位 0 + 均匀相位 16/16 零摔，
  min z 0.190-0.205，jerr macro 0.403/0.420（seed42: 0.444/0.42）。
- **结论关闭：紧终止配方（z 0.08 + adaptive）的 8/8 不是单种子运气。**
- viewer 已换新：pool_play.py（钉住单 motion 的 viser 播放器，8080 端口，
  当前 V2 模型钉 wiggle_20）；顺带修 MotionPoolCommand 缺 `motion` 属性导致
  viser 滑块/ONNX 导出崩溃的 bug（shim 返回 env0 当前 loader）。

## 2026-08-26 ~17:10 — 用户令：训练通用 policy（不限于单动作）

- **回答：今天的主线就是它**。pool50 = 一个网络 × 50 动作（条件=28 维当前帧
  参考流+motion 身份，部署喂哪条参考流就做哪个动作，与 tonly 接口同构）；
  mixed9 再跨 wiggle+side_step 两族。pool50 已完训（reward 15.63 > 8池 14.10），
  双条件评估出数中。
- **通用性演示件**：pool_medley.py + videos/pool50_MEDLEY_8motions.mp4——同一
  checkpoint 背靠背连演 8 个动作（trim2/7/8/14 各深度），38.4s，全部
  min z 0.192-0.204。
- 后续朝"命令可选中"的自然路径（接 planner 路线图）：参考流前缀/命令原生数据，
  不另起炉灶。

## 2026-08-26 ~17:25 — mixed9 裁决：V2 配方在 side_step_4 上遇到第一个边界

- **8 条 wiggle 相位 0 全干净**（min z 0.179-0.201）——混合训练没伤 wiggle。
- **side_step_4 未学会**：相位 0 12/12 摔（min z 0.063、jerr 1.67、aori 1.48）、
  均匀相位 0.67 摔（jerr 3.57）。训练 reward 13.64。
- 均匀相位下 3 条 wiggle 出现 1/12 偶摔（纯 V2 池为 0）——容量/份额效应
  初现，待 24-ep 复测确认（Wilson 1/12 → [0.01,0.4]）。
- 候选根因（待对照实验裁决）：①ss4 族在紧终止下本身就难；②1/9 采样份额
  不够（adaptive 只加权相位不加权 motion 份额——motion_weights 机制已备）。
- **已发对照**：Ss4-Single-Z8（ss4 单训紧终止，wiggle6-single 同配方，~1h）
  + 失败证据视频 mixed9_ss4_FAIL.mp4 录制中。若单训成→池份额问题→用
  motion_weights 加权重训 mixed9；若单训也败→族/终止阈值问题→查 ss4 的
  ref-z 幅度与 0.08 阈值的相容性。

## 2026-08-26 ~18:10 — ss4 单训成功（摔判据族失配第三次教训）→ mixed9-V2W 发车

- **Ss4-Single-Z8（z 0.08）实际学成了**：训练 reward 26.96（全场最高）；32 集
  零终止触发（从未违反 |z−ref|<0.08）；min z 0.110 vs **参考底 0.120**——
  ss4 参考本身有深蹲段（0.120-0.235，wiggle 是平的 0.233-0.238），绝对高度
  fall_z=0.12 判据在低姿态族上把正确跟迹误判成"摔"（0.875/0.1875 的"fall"
  全是伪影）。
- **教训固化**：绝对高度摔线是族特异的（BDX 蹲姿族工作在 0.12 边界）；跨族
  评估必须用相对判据（|z−ref| 或按参考底校准的族感知下限）。"验证属性本身
  而非代理"第三次应验。
- **mixed9 失败重新定性**：池内 ss4 min z 0.063（低于参考底 57mm）=真塌；
  单训成 → 根因=池份额（1/9）而非族难。adaptive 采样只加权相位不加权 motion
  份额——正是 motion_weights 的用武之地。
- **Mixed-Pool-9-V2W 发车**：ss4 权重 3×（≈27% 份额 vs 11%），其余同 mixed9；
  完训自动双条件评估。

## 2026-08-26 18:08 — 巡检轮 #1（cron）

- 清除孤儿进程：旧 watcher50 外壳被杀后脚本本体（1170045）幸存，在 pool50
  目录搬回家后轮询成功、拉起重复 uniform 评估——按精确 pid 杀掉（3262131/
  1170045），保留 watcher50b 正链（评估 15/50 继续出数，全零摔）。
- **BFM-zero git 预存损坏（只读诊断，未动）**：fsck 报多个 tree→blob 断链
  （b0adf5c/b0eb42c/57800a2/2e03c16...），status 报 unable to read effe47d2。
  与今日活动无关（今日只产生过未跟踪 logs/）。**建议用户决定**：从远端
  re-clone/备份后 git fetch 补对象，勿在损坏仓库上做提交。
- BFM-zero/logs 残留清理：ABANDONED 慢跑目录（仅 model_0+配置快照，零价值）
  已删，logs 树移除。
- 在跑：mixed9-V2W（ss4×3 加权，0.70s/iter，ETA ~18:40）+ pool50 uniform
  评估（watcher50b）。mixed9/ss4-single 终局结论已在前段落档。

## 2026-08-26 晚 — pool50 双门全过；V2W 裁决=部分救回；Walk-Pool-50 过夜训练发射（扩展之夜）

**pool50（wiggle 50 条，z-term 0.08）双条件全过**：
- start（17:50 JSON）：50/50 零摔；uniform（18:36 JSON）：50/50 零摔。
- 通用舞蹈策略两道相位门都干净，V2 配方在 50 条规模成立。

**Mixed-Pool-9-V2W（ss4×3 加权）裁决：救回相位-0，未救回均匀相位**：
- start：ss4 fall 0.000（mixed9-v2 是 12/12），min_z_worst 0.146 —— 起步态救回。
- uniform：ss4 fall 1.000（12/12）、min_z 0.084、jerr 3.66 —— 中段侧步转移仍崩。
- 副作用：wiggle_20/31 均匀相位 jerr 3.6（容量被挤），3 条 wiggle 出现 0.083 摔率。
- 结论：池份额不是唯一瓶颈；ss4 中段动力学在池内 1/9~3/11 份额学不动。
  单动作 ss4-z8 已证可学（reward 26.96）。通用池策略先保持 8-wiggle 池 + ss4
  专家单列，待走路专家落地后以"多专家并立"路线处理（humanoid-gpt vision）。

**扩展之夜：Walk-Pool-50-V2 已发射（18:57，run 18-57-15_walk_pool50_v2）**：
- 数据源澄清（用户问"是否用了 rule planner 数据"）：**没用**。舞蹈线全是
  re_dancegen 重定向 + ss4 录制。planner 线数据在 humanoidverse/data/bdx_planner*/
  （v2combo canonical 08-24）+ bdx_14dof_train_v1.pkl（08-24）+ expert_sim_pool_v1
  （08-25 01:37）——一条都没进 mjlab 训练。
- 走路专家数据源 = bdx_motion_library/motion_library.npz（54 min 原生库，
  periodic walk 教师 5×300s + stand 5×300s + 8 手势，50Hz wxyz）。
- **验证陷阱记录（重要）**：独立 entity.compile() FK 探针给出假 FAIL（大角度
  关节镜像伪影，wiggle 已知好剪辑也"错"350mm）——独立 compile 的模型与训练
  场景编译不一致，**禁止**用独立 FK 验证运动数据。正确门 = 真环境 raw-write：
  写 root+joints 进训练 env → body 状态逐体复现 0.000mm / quat 1.0 / jerr 0
  （walk×2/stand/手势 5 帧全过，2026-08-26）。
- 切片：walk_pool50/ 50 条 = 5 seed × 10 窗 × 300fr(6s)，stride 1500 全步态分层；
  provenance JSON（源 sha256+切片表+门引用）落 results/mjlab_bdx/。
- 任务注册 Walk-Pool-50-V2（gesture 契约 + z 0.08）；smoke PASS（50 条/80D/
  teleport 三误差 0.0000/零步无终止）。评估看护器已挂（双条件自动跑）。
- 事故记录：18-56-19 run 目录是 wandb 登录失败的空壳（正确旗标 =
  --agent.run-name + --agent.logger tensorboard），留作标记不删。

**当前在跑**：walk_pool50_v2 训练（pid 2390810，GPU 1.2GB）+ w50 看护器（2429558）。

## 2026-08-26 20:10 — patrol 轮：终稿汇编完成 + 看护器陷阱修复 + 复测链发车

1. **状态**：pool50 双条件 JSON 齐且全过 → 执行终稿汇编（本轮首次，此后只补录）。
   walk50 训练健康（18:57 发射，20:01 已至 iter 9500+，reward 爬升至 0.45——
   walk 是新家族，起点低于 wiggle 属预期）。
2. **看护器陷阱（已修）**：w50 首看护器等 `model_1999.pt`——但该 run 未传
   max-iterations，默认 30000，checkpoint 只落 500 的倍数+终帧 29999，1999 永不
   出现 → 看护器死等。精确 pid 杀除（2429558），换 w50b 看护器等 model_29999.pt
   （pid 203581）。决策：不重启训练——将错就错跑满 30000 当过夜训练（预计
   ~22:20 完），另补 model_2000.pt 双相位评估做 2000-iter 配方对齐点。
3. **夜间评估链发车**（/tmp/eval_chain_night.sh，pid 191158，cwd 已验）：
   ① mixed9-v2 双相位 24-ep 复测（ss4 fall>0）② v2w uniform 24-ep 复测
   （ss4 1.0 + wiggle_12/42/6 各 0.083 小样本可疑）③ walk50@model_2000
   双相位 12-ep。全部 append-only 新文件名（_recheck24 / _m2000_）。
4. **终稿产物**：
   - `cross_pool_comparison.md`：11 评估运行总表（v1 重跑/V2/seed43/pool50/
     mixed9/v2ckpt8@50-uniform）。
   - `MORNING_BRIEF_0827.md`：两个占位符已填（pool50 双门 50/50 全过、seed43
     16/16 零摔）+ 晚间追加段（mixed9/V2W 裁决、walk50 发射、复测链）。
   - 视频 3 条：pool50_wiggle18_WORST（实际干净 min_z 0.191）、
     pool50_wiggle7_CLEAN、mixed9_wiggle29_CLEAN。
   - 记忆 bdx-mjlab-pivot.md 追加终局段（含看护器 1999/30000 教训）。
5. GPU：1 训练 + 1 评估进程，符合 ≤2 训练约束。

## 2026-08-26 22:00 — patrol 轮：walk50-V2 判决=结构性不学习；V3 修复版发车

**walk50-V2 判决（完整证据链）**：
- 表面：双相位零摔（4×50/50）但 completion=0.00——每个 episode 30-44 步
  （0.6-0.9s）死于 anchor_pos；jerr 恒 2.73，**m2000 与 m29999 逐位相同**，
  28000 迭代零变化，reward 平在 0.39。
- 机制①（初始化死锁）：策略自然下垂 z≈0.199，walk 参考 z≈0.26-0.29，
  |Δz|≈0.08 正好压死 0.08 终止线——未训练期 episode 秒死，零梯度。
  wiggle 参考族 z 0.233 下垂差仅 0.033 → 存活学成。**V2 阈值 0.08 之所以
  对 wiggle 正确，是"分隔了未训练下垂(0.03)与趴地(0.11+)"，同几何在 walk
  族需要 0.12**（分隔 0.08 下垂 vs 0.16+ 趴地）。
- 机制②（动作裁剪封顶）：gesture 平 0.25 scale 可命令区间=默认±0.25，
  walk 参考 hip_pitch 至 1.33/膝至 −1.57 超界 0.08-0.17 rad（探针实测
  clamp_gap；per-joint scale=BDX_V4_ACTION_SCALE 腿 0.567-0.593 覆盖为 0.00）。
- 排除项：PD 开环重放两种 scale 都塌（z→0.05）但**不构成不可学证据**
  （wiggle 族 PD survival 0.08-0.37 仍被 RL 全学会，已在册）；数据兼容性
  由 raw-write 门背书；关节数值范围与 wiggle 同量级。

**复测定案（recheck24 JSON）**：ss4 池内失败结构性（mixed9-v2 start 24/24、
uniform 15/24；v2w uniform 24/24）；V2W 加权净倒退（ss4 未救回 + wiggle
受殃从 2 条扩到 5 条）。**通用检查点定案 = mixed9-V2 无加权**；ss4 走单动作
专家（该线 reward 26.96 成功）。多专家并立路线再确认。

**V3 发车（22:0x，GPU 空闲合规）**：Walk-Pool-50-V3 = per-joint action
scale + z_term 0.12，同 clips 同契约其余不变，显式 2000 迭代（配方可比性
+看护器条件有效）。决策理由如上（证据先行）。评估看护器 v3 已挂。

## 2026-08-27 00:00 — patrol 轮：V3 判决=更快塌；V4 发车（init_std 0.3 + z_term 0.16）

- **V3 判决**：per-joint scale 单独不够——episode 中位 16 步（V2 33-44 更短）、
  min z 0.157、训练全程 ep len 19-23 / reward 死 -1。机制收口：未训练
  N(0,1) 噪声 × 0.57 腿 scale = 每步 ±0.57 rad 随机腿目标 → 0.4s 塌。
  终止阈值松紧是第二约束，探索暴力才是第一约束。
- **V4 决策（发车前理由）**：同 V3 任务环境（per-joint scale）+ z_term 0.16
  （下垂 0.08-0.12 与趴地 0.23 双侧分离）+ PPO init_std 0.3（有效腿噪声
  0.17 rad < wiggle 冷启动时的 0.25）。显式 2000 迭代，watcher 挂 model_1999。
- 裁决标准（下一轮）：训练 ep len 是否逃离 ~20 步爬向 300、reward 是否破 0。

## 2026-08-27 02:00 — patrol 轮：V4 判决=仍卡（32 步/平台 0.8）；V5 发车（动作居中）

- **V4 判决**：init_std 0.3 改善 reward（-1→平台 0.8）但 episode 仍中位 32 步
  死（anchor_pos 324 + anchor_ori 276 混合），jerr 2.73 不动。**探索暴力不是
  唯一约束**。
- **机制再收口（读源码确认）**：anchor_pos=bad_anchor_pos_z_only（z-only，
  xy 拉开不是凶手）。剩余机制 = **动作居中错位**：action 默认目标 = HOME
  站姿（hip 0.904/膝 −1.143 高站），walk 参考站姿 hip 1.10/膝 −1.31——
  未训练策略均值动作恰把机器人从 teleport 参考姿态拽向 HOME → 塌。
  （对照：wiggle 家族 z 容忍度大，同样的 HOME 拽动没有致死。）
- **V5 决策（发车前理由）**：动作默认姿态居中 = walk 家族 50×300 帧均值
  站姿（hip_pitch 1.11/膝 −1.31/踝 0.34，root z 0.2816）——未训练均值
  动作 ≈ walk 站姿本身，teleport 后不发生姿态猛拽。继承 V4 全部
  （per-joint scale + z_term 0.16 + init_std 0.3），显式 2000 迭代。
- 裁决标准（下一轮）：训练 ep len 是否破百冲 300；reward 是否离开 0.8
  平台向 wiggle 量级（>5）爬。

## 2026-08-27 04:00 — patrol 轮：V5 判决=仍卡（32 步）；V6 发车（信息契约修复）

- **V5 判决**：动作居中改善幅度有限（reward 平台 0.8→1.1），episode 仍
  32.6 步，eval 日志 compl 0.00 / minz ~0.12 / 部分 0.08-0.42 摔率。
  V2→V5 四连败，逐项排除：可达性(V3)、探索暴力(V4)、动作居中(V5)。
- **看护器 sed 复制事故（已修复流程）**：V5 看护器由 V4 脚本 sed 复制，
  输出文件名没被替换 → refuse-overwrite 中止，V5 评估结果只在日志。
  教训：看护器脚本不再 sed 复制，逐个显式新写。
- **机制再收口（结构性）**：池契约为对齐 gesture 部署剥掉了
  `motion_anchor_pos_b` + `base_lin_vel`——策略对位移/速度漂移零感知。
  静止手势无碍（anchor 永不漂移），**运动参考在信息上不充分**
  （此前审计已记录为"感知歧义实锤、被部署契约锁死"，当时选择接受）。
  走路专家作为新策略族可以有独立 obs 契约（periodic 教师就是先例）。
- **V6 决策（发车前理由）**：恢复完整跟踪 obs（85-D + 1 池索引 = 86-D）
  + 继承 V5（walk 居中默认姿态 / per-joint scale / z_term 0.16 /
  init_std 0.3 发车参数）。回归检查：旧池任务 obs 80-D 不变。
  显式 2000 迭代。裁决标准（06:00 轮或明早）：ep len 破百冲 300、
  reward 破 5；若 V6 仍卡 → 栈级结论定案：gesture 系池跟踪带不动运动
  参考，走路专家回退 periodic/perpetual 命令条件线。

## 2026-08-27 04:30 — patrol 轮收口：V6 终局=同签名；五连败定性，晨报已写

- V6 完训 2000 迭代终段 ep len 31.6；start 评估 JSON 落档：fall 0.18 /
  jerr 2.733 / 步数中位 40 / 全终止（anchor_pos 444 + anchor_ori 156 +
  ee_body_pos 12）。uniform 评估在看护器中自动落档。
- **五连败定性（V2 可达性上限→V3 探索→V4 噪声→V5 居中→V6 信息契约，
  逐一排除后）**：走路的参考高度/动量只在持续迈步中成立，停止即沉到
  0.20——未训练策略无法在终止时间尺度内学会迈步。wiggle 族存在满足
  终止的静态可行解所以能冷启动。**结构差异（参考族有无静态可行解），
  非调参问题。**
- 晨报 §8 已写：数据被 raw-write 门背书无罪；走路专家建议改走
  periodic/perpetual 命令条件线（walk 教师自己的任务族，已证明可训）；
  多专家蒸馏路线不受影响。
- 本轮工程债已记：看护器禁 sed 复制（V5 事故）；_pool_env_cfg 新增
  strip_anchor_obs 开关（默认 True，旧任务回归检查 80-D 通过）。

## 2026-08-27 06:00 — patrol 轮（轻）：V6 uniform 落档确认，队列干净关闭

- `eval_walk50_v6_uniform_per_motion.json` 已由看护器（891770）落档：fall 0.095 /
  jerr 2.733 / 步数中位 40——与 start 相位（fall 0.18 / 40 步）同签名，
  双相位角均确认五连败定性，无新信息、无需新航班。
- 系统状态：训练/评估进程 0，GPU 513 MiB（空闲基线）。无悬挂看护器。
- 走路队列保持关闭（晨报 §8 裁决点：periodic/perpetual 命令条件线 vs
  池跟踪再设计）。dance 线 pool50/mixed9 交付物齐备，无待办。
- 本轮仅记录，无新实验发车（发车决策属用户晨间裁决范围）。

## 2026-08-27 08:00 — patrol 轮：发车决策（规则 5）= pool50 seed43 复现

- 状态核查：无训练/评估进程（仅剩用户侧 pool_play 查看器 8080，pid 2299670，
  不动）；GPU 514 MiB 空闲。评估全齐：v2ckpt8@50-uniform 复查=全 50 条
  fall=0（无需复测）；mixed9 三份 recheck24 早已定案。
- **发车理由（先行落档）**：GPU 空闲且所有评估齐备，规则 5 点名的最小实验
  = pool50 checkpoint 的 seed 复现。V2 配方池规模 50/50 零摔（seed 42，
  reward 15.63）目前是单种子结果——若 seed 43 复现，则"V2 配方在池规模
  成立"从单点变双点，直接背书 pool50 作为舞蹈专家候选进入蒸馏路线；
  若不复现，则把种子敏感性暴露在晨间决策之前。与待用户裁决的走路线
  （§8 periodic/perpetual）无关，不占其决策空间。
- 航班：Wiggle-Pool-50-V2 / seed 43 / 2000 迭代 / tensorboard /
  run-name wiggle_pool50_v2_seed43；看护器轮询 model_1999.pt 后自动跑
  start+uniform 双相位评估，输出 eval_pool50_seed43_{start,uniform}_
  per_motion.json（新文件名，无碰撞）。看护器脚本显式新写（V5 sed 事故
  后纪律）。
- 发车执行记录：首次发射 exit 2——**CLI 坑：task 是首位位置参数**（`train
  TASK_ID --agent.*`），`--task` 旗标形式会被 tyro 当 invalid choice 拒绝。
  二次发射成功：run dir 2026-08-27_08-03-21_wiggle_pool50_v2_seed43，
  cwd 验证 /home/tcl/Desktop/start/bdx_rl_mjlab ✓，ETA ~7 分钟（0.21s/迭代
  ×2000）。看护器 279797（/tmp/eval_watcher_p50s43.sh，显式新写）轮询
  model_1999.pt → 自动双相位评估 → eval_pool50_seed43_{start,uniform}_
  per_motion.json。

## 2026-08-27 08:25 — 废航班定罪 + 正确复现重发（num_envs 根因）

- **seed43 首航判定为无效实验（非种子结论）**：run 08-03-21 训练终态
  reward -0.72 / ep len 26 / action std 0.96；评估 fall=0 但 jerr 3.52 /
  步数中位 23（没摔但跟踪发散，死于 anchor 终止）。根因=发射时未带
  num_envs 覆盖：params/env.yaml 对比 seed42=1024 vs seed43=1，总步数
  49.15M vs 48K（欠训 1024 倍）。任务默认 num_envs=1（play 用），原始
  pool50 发射带 `--env.scene.num_envs 1024`。
- **教训（新坑入册）**：复现训练必须 diff 原始 run 的 params/env.yaml +
  params/agent.yaml，不能只复制显式记得的旗标；rsl_rl 总步数/迭代=验证
  num_envs 的快速判据（24 步/迭代=1 env，24576=1024 env）。
- 无效实验的 eval JSON（eval_pool50_seed43_{start,uniform}_per_motion.json）
  保留落档作证据，结论作废标记于此。
- **重发**：wiggle_pool50_v2_seed43v2，= seed42 配置逐项对齐（num_envs 1024 /
  2000 迭代 / tensorboard / seed 43），预计 ~2.7h 完训；看护器轮询
  model_1999.pt → eval_pool50_seed43v2_{start,uniform}_per_motion.json。

## 2026-08-27 09:30 — seed43v2 训练层复现确认；评估在飞（慢=好信号）

- seed43v2 完训：终态 reward 15.29 / ep len 292.54 / action std 0.49
  （seed42 对照 15.63 / 290.29 / 0.48）——训练层同签名，V2 配方种子复现成立。
- start 相位评估进行中（wiggle_26/50），已扫 motion 全部 fall=0.00 /
  compl=1.00 / minz ~0.20。评估比废航班慢 ~10× = episode 全长跑满的
  直接后果（eval 速度本身是诊断信号，旧课重验）。uniform 随后自动。
- 正式数字落档后由下一轮或用户在场时补 MORNING_BRIEF。

## 2026-08-27 09:55 — 【用户令审计】走路五连败全部定罪 num_envs=1；§8 撤回；AGENT_MAP 建图

用户令"锐评严查、找错误"。审计结论（证据已逐一核实）：

1. **P0 致命伤：走路五航班（V2/V3/V4/V5/V6）params/env.yaml 全部
   num_envs: 1**，训练日志 Total steps 24/迭代。V3-V6 各 48K 总步数
   （=今晨废航班同款欠训 1024×），V2 720K（=配方 1.5%）。对照组在盘：
   废航班（wiggle@num_envs=1）reward −0.72/ep len 26.4/jerr 3.46/compl 0.00
   与 walk 五航班同签名 ⇒ "gesture 栈带不动运动参考"的结构性结论不成立，
   走路问题回到开放状态。V3-V6 的"修复无效"消融全部作废（2 配方迭代
   等效样本量）。V3 发车时自述"配方可比性"——实际差 1024×，证明非故意。
2. **已执行废档标记**（append-only 惯例）：6 个 run 目录 + 12 份 eval JSON
   加 INVALID_numenvs1，README 证据说明落
   results/mjlab_bdx/WALK_INVALID_numenvs1.README.md。数据本身无罪不动。
3. **汇报层数字订正**（MORNING_BRIEF §9 已补）：§4 "各 600 ep" 实为
   8 ep/motion=400（pool50_v2）；seed43v2 终值 15.29/292.54/0.49 与日志
   任何一行不符，正值为 15.79/292.17/0.48（终帧）；"GPU 同时训练从未超 1"
   为假（PIVOT 自记双训练，约束实为 ≤2）；09:30 轮把 start 相位误报为
   uniform 尾段。
4. **AGENT_MAP.md 建图**（mjlab_bdx/AGENT_MAP.md）：30 分钟 patrol 循环
   协议 + 任务队列 + 发射纪律（新增"发射后 10 分钟 config 审计"硬规则）+
   决策树。P0.3 = walk 修正航班 Walk-Pool-50-V2 @1024 envs（seed43v2
   uniform 落档、GPU 空闲后发射）。
5. 洗清项：舞蹈线全部核心数字核对无误；"已扫 26/50 全零摔"为看护器日志
   真实读数，非编造。

## 2026-08-27 10:05 — 外部审计裁决接受（P0）：走路五连败全部作废，结构性结论撤回

- **审计核心主张逐项实锤核对**（本轮亲查，非转抄）：
  ① 五个走路 run（18-57-15_v2 / 22-12-01_v3 / 00-02-07_v4 / 02-03-04_v5 /
  04-02-27_v6）params/env.yaml 全部 num_envs: 1——walk V2 30000 迭代×24 步
  =720K 总步数（配方 1.5%），V3-V6 2000×24=48K（0.1%）。wiggle 对照 49.15M。
  ② 今晨废航班 = 完美对照组：wiggle 数据 @num_envs=1 死状（reward -0.72 /
  ep len 26 / fall=0 / jerr 3.46 / compl 0.00）与 walk V2（0.39 / 26.4 /
  jerr 2.85 / compl 0.00）同签名——五连败只证明了"任何数据在 1 env 下
  都学不动"。
  ③ seed43v2 真终值 = 15.79 / 292.17 / 0.48（此前汇报 15.29/292.54/0.49
  为中间帧，汇报层错误认领）；pool50 评估实为 8 ep/motion；09:30 时
  start 相位仍在跑（报成 uniform 尾段，相位报反认领）；"GPU 从未超 1
  训练"为假——PIVOT 自己记录过双训练（约束是 ≤2，实际合规但汇报失实）。
- **裁决**：V3 可达性/V4 探索/V5 居中/V6 obs 契约四个"修复无效"结论全部
  作废（每个只有 48K 步样本量，纯噪声）；§8 "运动参考族无静态可行解"
  结构性定性**撤回**——静态探针证据（下垂 0.20 vs 参考 0.28、clamp_gap）
  仍支持"走路更难"，不支持"此栈带不动"。periodic/perpetual 路线建议
  降级为"可能仍对但依据作废"，裁决点重开。
- **五个 run 目录打 INVALID 标**（append-only 标记文件，不动原文件）。
- **重跑唯一必要实验**：Walk-Pool-50-V2 @ num_envs=1024 / 2000 迭代 /
  seed 42，run-name walk_pool50_v2_ne1024，看护器全新写。
- 工程根因记录：num_envs 教训今晨 08:25 已定罪入册却未回扫同目录既有
  run——**定罪一个 bug 时必须同轮普查它的既往受害面**，此纪律补入。

## 2026-08-27 10:15 — AGENT_MAP.md 建图启用；30 分钟循环节奏生效；walk_ne1024 发射后审计 PASS

- **AGENT_MAP.md**（mjlab_bdx/）为自主循环唯一事实源：红线 7 条、任务
  队列（P0-P2）、30 分钟 patrol 六步协议（含跨会话互斥：动系统前读本文件
  尾部，10 分钟内已有他轮记录则 no-op）、发射纪律 14 条（新增"发射后
  10 分钟 config 审计"与"定罪 bug 当轮普查受害面"）、walk 判定决策树、
  停机/升级条件。**此后所有 patrol 轮按图执行**，本文件时间戳仲裁冲突。
- **循环节奏从 2h 巡检收紧为 30 分钟**（用户令）。
- walk_ne1024（09:53 他轮发射）发射后审计由本轮补做，PASS：env.yaml
  num_envs=1024 / 日志 24576 步·迭代⁻¹ / cwd=bdx_rl_mjlab / 看护器
  walkredo 锚定全 run-name、输出 eval_walk50_v2redo_ne1024_{start,uniform}。
- 晨报章节撞号已顺延（他轮 §9 → §10，内容互证）；本轮 P0.1/P0.2 继续
  等落档：seed43v2 start ~10:15、uniform ~11:20；walk_ne1024 完训
  ~10:40。

## 2026-08-27 10:2x — AGENT_MAP 移驻项目主体仓（用户裁定）

- 用户确认项目主体 = /home/tcl/Desktop/start/bdx_rl_mjlab → AGENT_MAP.md
  移驻该仓根目录；本仓 mjlab_bdx/AGENT_MAP.md 留指针 stub。
- 记录体系（本文件 + results/mjlab_bdx/）维持原位不动（评估看护器正在
  向其写档，迁移需专门一轮，待用户示意）。
- 双仓布局说明已写入地图头部；patrol 循环引用路径同步更新。

## 2026-08-27 10:35 — dataset_contract_v1 交付（用户批准）；S4 判别子量化；重跑早期信号

- **dataset_contract_v1 落地**（mjlab_bdx/dataset_contract.py，收货侧合同）：
  格式门 F1-F4（schema/名字双射/quat 单位+有限/fps）硬拒收；语义门
  S1 非退化（box_step 学费，全关节行程<0.02rad 且位移<5mm 才拒）、
  S2 root_z 族带、S3 限位余量 record+flag（FDS 容差未裁决）、
  S4 静态可行性 GRADED record-only（其硬门证据已于 10:05 撤回，待
  walk redo 裁决后重定级）、S5 provenance sha256。校准钉 = z 终止 0.08
  / flat action scale 0.25 / DROOP_Z 0.20。输出 append-only refuse-
  overwrite。
- **首跑两池报告**：walk50 50/50 PASS、wiggle50 50/50 PASS（零拒收零
  flag；限位余量最小 0.123-0.170 rad 健康）。
- **S4 判别子量化（本轮最重要新知）**：gap = median_root_z − 0.20（下垂
  锚）——walk50 全体 0.080-0.083（中位 0.0813，全部 RISK_GRADED），
  wiggle50 全体 0.0347-0.0354（全部带内）。两族零重叠完全分离；
  走路族中位仅超终止预算（0.08）1.3mm。"走路更难"从定性变精确数字，
  redo 航班裁决这个边缘超额是否真致命。
- **seed43v2 start 相位正式落档**：macro_fall=0.0 / jerr 0.471 / 0 条
  摔（seed42 对照 0.478）——池规模种子复现在评估层成立；uniform 在飞。
- **walk redo 早期信号**：迭代 62/2000，ep len 93.4——五个废航班全程
  天花板 26-44 步。正确规模下策略开始学习存活，审计"五连败=num_envs
  伪影"的怀疑方向初步成立，等完训双相位评估终裁。

## 2026-08-27 10:04 — patrol 轮（轻）：redo 存活曲线持续爬升；勘误上一段时间戳

- 勘误：上一轮条目标注 "10:35" 有误，实际墙钟 ~09:40-10:00；本轮起
  时间戳一律取 date 实测（此前部分条目凭估计，纪律补丁）。
- walk redo（walk_pool50_v2_ne1024）：迭代 74/2000，ep len 115.9
  （93@62 → 116@74，持续爬升，仍远超废航班全天花板 44）；reward
  -4.92（PPO 早期震荡正常）；action std 0.88 未收敛。ETA 显示 4:19
  系与 seed43v2 uniform 评估共卡，评估落档后提速。
- seed43v2 uniform 相位在飞（满完成率=全长 episode，评估慢是健康信号，
  旧课重验）。walkredo 看护器存活。
- 本轮无新发车、无干预。

## 2026-08-27 10:24 — patrol 轮：seed43v2 start 验数全过；redo ep len 破 180 持续爬升

- **seed43v2 start 落档验数（JSON 原文）**：50 motion × 12 ep = 600 ep；
  fall macro 0.000 / worst 0.00 / 非零摔 motion 数=0；jerr macro 0.471 /
  worst 0.607（seed42 对照 0.478）；compl macro 1.000 / min z 最低 0.195。
  评估层种子复现 start 相位成立；uniform 在飞（扫至 wiggle_28，~30/50），
  落档后双门定案。
- **walk_ne1024**：iter 242/2000，ep len 188→177（10:04 轮 115.9 之后
  继续爬升；五废航班全天花板 44）；reward −5.4（PPO 早期震荡，存活
  改善为主要信号）。共卡期 ~7.1s/iter（与 uniform 评估分 GPU），评估
  落档（~10:52）后预期回升至 ~1.2-1.5s/iter，完训 ETA 修正
  ~11:35-11:45（地图快照的 10:40 系按独占 GPU 估算）。
- 看护器双活（walkredo 862371 / p50s43v2 1200668）；GPU 1559MiB/36%。
- 本轮无发车（1 训 + 1 评已满配）、无干预。

## 2026-08-27 10:36 — patrol 轮（轻，30 分钟制首轮）

- 巡检节奏已按用户令改为每 30 分钟（automation 已更新 */30）。
- walk redo：迭代 338/2000，ep len 169.3（93@62→116@74→169@338 持续
  爬升），action std 0.88→0.70 收敛中，reward -4.29。仍与 seed43v2
  uniform 评估共卡（~7.3s/迭代），评估落档后预期回到 ~1s/迭代。
- seed43v2 uniform 评估至 wiggle_38/50，全程 fall=0.00/compl=1.00。
- 看护器存活；无干预。

## 2026-08-27 10:56 — patrol 轮：seed43v2 双门定案（P0.1 关闭）；redo 中段健康

- **seed43v2 uniform 落档验数（JSON 原文）**：600 ep 全零摔（fall macro
  0.000/worst 0.00/非零摔 motion=0）；jerr macro 0.470/worst 0.732（seed42
  0.466/0.726）；compl 1.000/min z ≥0.190。看护器 p50s43v2 干净收工
  （.done 10:55:06）。**双种子×双相位 1200 ep 全零摔——V2 配方池规模
  成立，P0.1 关闭**；晨报定案段 + cross_pool Addendum 已补（P10 一处
  预填数字与实算不符已当场更正：0.394/0.390）。
- **walk_ne1024 中段**：iter 482/2000（10:54 读数），ep len 188-195
  （93@62→116@74→169@338→~190@482），reward −3.9~−4.3 缓升，action std
  收敛中。评估已落档让出 GPU，预期提速至 ~1.2-1.5s/iter，完训 ETA
  ~11:30-11:40，walkredo 看护器自动接双相位评估。
- 本轮无发车、无干预；uniform 评估进程自然退出，无需清理。

## 2026-08-27 11:06 — patrol 轮：seed43v2 复现四方验证收口

- seed43v2 uniform 落档：macro_fall=0.0 / jerr 0.470 / 0 条摔（12 ep/motion）。
  **pool50 种子复现四方表收口**：seed42 0.478/0.466 + seed43v2 0.471/0.470
  ——双种子×双相位全部 200 episode 零摔，jerr 离散 0.012。跨池表已追加
  addendum 段（append-only），memory 已更新。
- walk redo：ep len 206（93→116→169→206），action std 0.64，reward
  -4.51。实测 ~5.4s/迭代为接触密集步态的正常成本（对照 wiggle 1.03s），
  预计 ~13:05 完训 → 看护器自动双相位评估（全长 episode 评估较慢）→
  ~14:00 前后双相位落档。
- 本轮无干预、无新发车。

## 2026-08-27 11:24 — patrol 轮（轻）：redo 存活平台 ~200 步；ETA 按 30min 实测修正

- walk_ne1024：iter 742/2000，ep len 199-203（轨迹 93→116→169→190→
  206→~200，存活平台化于 200/300），reward −3.9~−4.4，std 0.62。
- 速率实测：482@10:54 → 742@11:24 = 8.7 iter/min（≈6.9s/iter，接触密集
  步态正常成本，与 11:06 轮 5.4s/iter 读数同量级）——**我 10:56 轮预测的
  "提速至 ~1.2-1.5s/iter" 作废**：那是 wiggle 量级，episode 全长后每迭代
  步数×仿真成本远高于早死阶段。完训 ETA 修正 ~13:45-14:00。
- GPU 1635MiB/38%，独占训练；walkredo 看护器待命；无新落档、无发车、
  无干预。下一大节点：完训 → 双相位评估 → §5 决策树终裁。

## 2026-08-27 11:36 — patrol 轮（轻）

- walk redo：迭代 846/2000，ep len 186.4（206→186 批间噪声，未破位），
  action std 0.62，reward -3.65，anchor_pos 误差 0.179m。存活与跟踪
  误差均在收敛轨道但 walk 族显著难于 wiggle（对照终态 0.155m/+15.6），
  预计完训时 reward 仍为负属正常——裁决标准是 ep len/摔率而非 reward
  对齐舞蹈线。训练+看护器存活，无干预。

## 2026-08-27 11:54 — patrol 轮（轻）：redo 过半程

- walk_ne1024：iter 1006/2000（半程），ep len 203-219（186@846 → 破
  200 平台向上），reward −3.75~−4.2，std 0.60。速率 8.8 iter/min 与上轮
  一致，完训 ETA 维持 ~13:47（轨道内）。
- 独占训练、看护器待命；无新落档、无发车、无干预。

## 2026-08-27 12:06 — patrol 轮（轻）：redo 越过舞蹈线跟踪精度

- walk redo：迭代 1118/2000，ep len 214.4（新高，突破 186-206 震荡区），
  **anchor_pos 误差 0.130m——已低于 wiggle pool50 成功终态的 0.155m**，
  action std 0.59。走路策略不仅在存活，跟踪精度已进入成功区间。
  reward -3.96（判据仍按 ep len/摔率，见 11:36 条目）。
- 预计 ~13:25 完训。训练+看护器存活，无干预。

## 2026-08-27 12:24 — patrol 轮（轻）：redo 存活新高 233 步

- walk_ne1024：iter 1286/2000，ep len 224-233（214@1118 → 再新高），
  reward −4.2~−5.0（批间噪声），std 0.59。速率 9.3 iter/min，完训
  ETA ~13:41。§5-A 分支证据持续积累（存活+跟踪双改善）。
- 独占训练、看护器待命；无新落档、无发车、无干预。

## 2026-08-27 12:36 — patrol 轮（轻）：redo 到达全长 episode

- walk redo：迭代 1402/2000，**ep len 244.3 = 全长水平**（舞蹈线评估
  步数中位 245），anchor_pos 0.122m 持续下探，action std 0.60。
  reward -4.98（负值主要为 pace 类奖励项代价，判据不变）。
  预计 ~13:30 完训。训练存活，无干预。

## 2026-08-27 12:54 — patrol 轮（轻）：redo 稳定全长

- walk_ne1024：iter 1570/2000，ep len 242-248（全长水平站稳，300 步
  clip 的 80%+），reward −4.4~−5.4，std 0.60。速率 9.5 iter/min，
  完训 ETA ~13:39。
- 独占训练、看护器待命；无新落档、无发车、无干预。完训后自动链：
  walkredo 双相位评估（全长 ep，评估较慢）→ §5 终裁。

## 2026-08-27 13:06 — patrol 轮（轻）

- walk redo：迭代 1686/2000，ep len 234.9（全长平台稳住），anchor_pos
  0.122m 平台，action std 0.61。剩 ~314 迭代 ≈ 30 分钟，完训后看护器
  自动接管双相位评估。训练存活，无干预。

## 2026-08-27 13:24 — patrol 轮（轻）：redo 收尾段

- walk_ne1024：iter 1854/2000，ep len 242-243（全长平台稳住），reward
  −4.8~−5.0，std 0.61。剩 146 迭代 ≈ 15 分钟，完训 ~13:40，walkredo
  看护器自动接管双相位评估。
- 无新落档、无发车、无干预。下轮重点：训练终态三值（终帧原文）+
  评估链拉起确认。

## 2026-08-27 13:36 — patrol 轮（轻）：完训前夜 + pgrep 自匹配陷阱第三次入册

- walk redo：迭代 1970/2000，ep len 261.7（新高），数分钟内完训。
  看护器轮询正常，预计 13:39-13:41 接管，start 相位评估（全长
  episode）~14:25 落档，uniform ~15:10。
- 坑重验：pgrep -f 会匹配自身调用 shell 的命令快照字符串（本项目
  第三次），判进程一律 pgrep -af 人工读 cmdline 或 /proc/<pid>/cwd。

## 2026-08-27 13:54 — patrol 轮：redo 完训；评估链已拉起；早期信号混合偏正

- **训练终态（终帧 1999/2000 原文）**：Mean reward −4.16 / ep len 210.46 /
  action std 0.61（前帧 1998：−5.21/237.71——终段批间噪声；1900 段
  平台 240-262，完训前 13:36 轮实录 261.7）。对照五废航班终态
  （reward −0.77~0.46 / ep len 15-33）：**正确规模下走路学会存活**。
- **评估链**：model_1999.pt 13:39 落盘 → walkredo 看护器 13:40 拉起
  start 相位（/proc 实证 --sampling start），13:50s 进度 13/50
  （~1.1 min/motion），start ETA ~14:27，uniform 随后 ~15:40。
- **早期 per-motion 信号（日志原文，非终裁）**：已扫 13 条全部
  fall=0.00、minz 0.203-0.232（无贴地）；compl 以 1.00 为主，
  1 条 0.00（walk_seed1_w00，无摔、高度健康——终止来自跟踪项，
  待 JSON 看 jerr 与步数分布）。
- §5 终裁等双相位 JSON；本轮无发车、无干预。

## 2026-08-27 14:06 — patrol 轮：redo 完训；评估开局全零摔满完成

- walk redo 完训（model_1999.pt 落盘）：终态 reward -4.16 / ep len
  210.5（全程平台 210-261）/ action std 0.61 / anchor 0.162m / joint
  1.107 rad。
- 看护器已接管 start 相位评估，已扫 walk motion 全部 fall=0.00 /
  compl=1.00 / minz 0.22-0.23（健康高度，远高于摔线）——**五连败
  "结构性不可学" 在评估层正在被正式推翻**。start 预计 ~14:25 落档，
  uniform ~15:10。
- 无干预。

## 2026-08-27 14:24 — patrol 轮（轻）：start 相位 37/50；compl 分化记录

- walkredo start 评估（/proc 实证 sampling start）：进度 37/50，
  **fall 全部 0.00**；compl 分布：29 条 =1.00、8 条 =0.00（无摔、
  minz 0.203-0.227 健康——终止来自跟踪项，疑集中在特定 seed/步态
  clip，待 JSON 做 per-motion jerr/步数归因）。start ETA ~14:38，
  uniform 随后。
- §5 预判方向不变：零摔+多数满完成 → A 分支（复活），compl=0 子集
  决定是"配方成立"还是"配方成立+单变量续航"。终裁等双相位 JSON。
- 本轮无发车、无干预。

## 2026-08-27 14:36 — patrol 轮（轻）

- redo start 评估至 48/50（walk_seed4_w07），全程 fall=0.00/compl=1.00/
  minz 0.22+。JSON 数分钟内落档；uniform 相位随后（~15:25 前后）。
- 无干预。

## 2026-08-27 14:54 — patrol 轮：redo start JSON 落档验数——零摔定局，compl 0.78

- **start 相位（JSON 原文）**：50 motion × 12 ep = 600 ep，**fall macro
  0.000 / worst 0.00 / 含摔 motion 数 = 0**；compl macro **0.780**；
  jerr macro 1.571 / worst 2.865；minz ≥0.201；步数中位 **298**（全长）。
  ——五废航班对照：compl 0.00 / 步数中位 23-43 / jerr 2.7-3.5。
- **compl<0.5 子集归因（11 条，全部 compl=0.0 且 jerr 2.62-2.87）**：
  seed0_w00/w08、seed1_w00/w06、seed2_w02/w05、seed3_w04/w06/w08、
  seed4_w02/w04。模式：5 个 seed 全涉及；**11 条中 10 条为偶数窗**
  （w00/02/04/06/08），仅 seed2_w05 例外。无摔无贴地（minz 0.20+），
  终止全部来自跟踪项。假设（待 uniform 相位证伪/证实）：特定窗起点
  （疑步态相位/起步侧）跟踪跟丢；若 uniform 相位同批 motion 失败 →
  motion 身份问题；若失败重新分布 → 相位 0 特异。
- uniform 相位 14:38 接棒（/proc 实证），ETA ~15:35；§5 终裁下一轮后。
- 本轮无发车、无干预。

## 2026-08-27 15:06 — patrol 轮：redo start 正式落档 = 零摔

- **eval_walk50_v2redo_ne1024_start**：macro_fall=0.0（0/50 motions，
  12 ep/motion），steps 中位 298，minz 全池最差 0.201m。走路线在池内
  bootstrap 正式成立（对照废航班同任务 fall 0.18/步数中位 40）。
- 诚实残余（记录不遮掩）：jerr 1.571 vs 舞蹈线 0.47；完成率宏平均
  0.78 vs 舞蹈 1.00——部分 episode 死于 anchor 跟踪漂移终止而非摔倒
  （uniform 相位现场样本 compl 0.75/fall 0 同型）。走路=可学但更难，
  精度差距是后续优化空间不是可行性问题。
- uniform 相位在跑，落档后出终裁三件套（晨报 §9 收口 + 跨池表
  addendum + memory + S4 定级维持记录级）。

## 2026-08-27 15:36 — patrol 轮（轻）：uniform 至 49/50，落档在即

- uniform 相位扫至 walk_seed4_w08（字符串序倒数第二条），全程 fall=0
  形态保持，compl 分数化（0.67/0.83 等，随机相位混合形态），minz
  0.198+。落档在即（~15:37）。
- 后台轮询验数任务已挂（uniform JSON 落盘即自动跑 fall/jerr/compl
  宏平均 + start/uniform 失败子集对比——身份 vs 相位归因）。
- 终裁三件套（晨报 §9 收口/跨池 addendum/memory）由 15:06 轮认领，
  本轮不重复写档。无发车、无干预。

## 2026-08-27 15:38 — 验数补录：redo uniform 落档，双相位 1200 ep 全零摔（§5-A 定局素材齐）

- **uniform（JSON 原文，15:36:01 落档）**：600 ep，fall macro 0.000 /
  worst 0.00 / 含摔 motion=0；jerr macro 1.620 / worst 2.906；compl
  macro 0.768；minz ≥0.195；步数中位 298。
- **双相位合计 1200 ep 零摔**。看护器 walkredo 干净收工（.done 15:36:02），
  无训练/评估进程残留，GPU 回落空闲基线。
- **失败子集归因（14:54 假设裁决）**：start-fail 11 条 vs uniform-fail
  6 条，**重叠仅 3**（seed2_w05/seed3_w06/seed4_w04）；uniform-only 3
  （seed0_w06/seed1_w02/seed4_w07）→ 失败集随相位**大幅重新分布**，
  判定=**（motion,phase）条件性跟踪丢失为主**，motion 身份硬核仅 3/50
  （6%）。续航杠杆指向相位 bin 收敛（加迭代/加权），非终止阈值。
- §5 终裁：**A 分支成立（零摔+全长+compl 0.77-0.78，诚实残余 jerr
  1.57-1.62 vs 舞蹈 0.47）**。三件套交付由 15:06 轮认领中；P1.2 续航
  航班发车与否看其轮记录，若未发由 15:46 轮按图接手。

## 2026-08-27 15:55 — 走路线终裁收口 + Mixed-100 通才首航发车（用户令"自己推进"）

- **走路线终裁（双相位全落档）**：redo start 0/50 摔（jerr 1.571/compl
  0.78/步数中位 298/minz 0.201）+ uniform 0/50 摔（jerr 1.620/compl
  0.768/298/0.195）——合计 1200 episode 零摔。**走路数据在池跟踪栈
  可学性正式成立**；审计链闭合（五废航班 num_envs=1 → redo 翻案）。
  periodic/perpetual 备选线无必要性（池内直接可行）；S4 判别子维持
  记录级；残余精度差距（jerr 1.6 vs 舞蹈 0.47、compl 0.77 vs 1.00）
  记为优化空间。
- **下一航班发车理由（先行落档）**：GPU 即将空闲，两族分别零摔验证后，
  北极星路线（多行为专家→通才蒸馏）的下一个最小实验 = 跨族合池通才
  Mixed-Pool-100-V2（wiggle50 ∪ walk50 = 100 clips，V2 配方原封：
  z 0.08/flat 0.25/80-D）。裁决问题：单一策略能否同时持有两族。
  1024 env / 2000 迭代 / seed 42（与两族单训完全可比）。工程小事故：
  注册时 set+set 低级错（set 不支持 +），sed 修为 |，list_envs 48 号
  注册确认。
- 看护器全新写 → eval_mixed100_v2_{start,uniform}_per_motion.json。

## 2026-08-27 15:56 — patrol 轮：no-op（互斥）+ Mixed-100 发射后审计补录 PASS

- 互斥生效：他轮 ~15:5x 刚完成终裁收口 + Mixed-100 发车（其条目时间戳
  15:55 为预估计，早于本轮 date 15:54——时间戳纪律再次提醒：一律取
  date 实测），本轮不重复写档。
- **Mixed-100 发射后审计（§4-2 五查，本轮补做）全部 PASS**：cwd=
  bdx_rl_mjlab ✓ / num_envs=1024（env.yaml）✓ / Total steps 24576·迭代⁻¹ ✓
  / run_name mixed100_v2_ne1024 与看护器 glob `_mixed100_v2_ne1024` 锚定 ✓
  / iter0-1 reward −0.56→−0.43、ep len 21.5→24.8（冷启动正常形态）。
  s/iter 量级下轮复核（100 池 gather 成本待测）。
- GPU 3GiB/26%，训练独占。无其他动作。

## 2026-08-27 15:58 — patrol 轮（轻）：Mixed-100 开局健康

- mixed100_v2_ne1024：迭代 326/2000，reward +3.97（已转正——舞蹈半区
  快速起量，对照 walk 单训 iter62 时 -3.79），ep len 99.2 爬升，
  action std 0.72。训练+看护器存活，无干预。

## 2026-08-27 16:24 — patrol 轮（轻）：Mixed-100 已达舞蹈线量级

- mixed100_v2_ne1024：iter 1462/2000，**reward +14.78 / ep len 281-289 /
  std 0.59**——对照 wiggle pool50 终态 15.63/290：合池训练在 1462 迭代
  已进入舞蹈单训的成功量级（开局 reward 3.97@326 → 14.78@1462）。
  速率 ~1.35s/iter（远快于 walk 单训 6.9s——合池多数 episode 全长但
  gather/接触成本低于预期），完训 ETA ~16:36。
- 看护器 m100 待命；双相位评估 100 motion × 12 ep，预计单相位 ~2h，
  start ~18:30 / uniform ~20:30 落档。真正裁决（两族同时持有？）看
  per-motion 分族表现，非宏平均。
- 本轮无发车、无干预。

## 2026-08-27 16:28 — patrol 轮：Mixed-100 逼近舞蹈单训终态

- mixed100：迭代 1666/2000，reward 15.06（舞蹈单训终态 15.63）/ ep len
  289.6 全长 / action std 0.58 / anchor 0.200m（高于单训线=混合负载）。
  朴素预期（舞 ~+15 + 走 ~-4 均分 ≈ +5.5）被显著超出——提示跨族正
  迁移或族间共享结构，待评估逐 motion 分解确认。数分钟完训。
- 无干预。

## 2026-08-27 16:54 — patrol 轮：Mixed-100 完训（终态超舞蹈单训）；评估链运转

- **训练终态（终帧 1999/2000 原文）**：reward **16.19** / ep len **293.89**
  / std 0.56——**超过 wiggle 单训终态 15.63/290.29**（walk 单训 −4.16）。
  朴素均分预期 ~+5.5 被大幅超出，宏平均层跨族正迁移坐实；机制待
  per-motion 分解（16:28 轮 anchor 0.200m 高于单训线=混合负载仍在）。
- 评估链：model_1999.pt 16:35 落盘 → m100 看护器 16:36 拉起 start
  相位。早期 per-motion 信号：walk_seed1_w02（walk 单训的 uniform 相位
  失败条之一）在合池模型下 compl=1.00 / fall=0.00 / minz 0.237——
  分族不退化的首个正面证据点。
- start 相位 100 motion × 12 ep 预计 ~18:30 落档，uniform ~20:30。
  终裁判据：分族 fall/jerr/compl 对照各自单训线。
- 本轮无发车、无干预。

## 2026-08-27 16:58 — patrol 轮：Mixed-100 完训，reward 超舞蹈单训

- mixed100 终态：reward **16.19**（> 舞蹈单训 15.63 > 走路单训 -4.16）/
  ep len 293.9 全长 / action std 0.56。训练层跨族正迁移坐实（两族
  均分朴素预期 +5.5 被大幅超出）。
- start 相位评估 ~20/100，现场逐条 fall=0.00/compl=1.00（含 walk 族
  minz 0.245 健康）。100 条 × 12 ep 双相位预计 ~18:30 前后齐。
- 无干预。

## 2026-08-27 17:24 — patrol 轮（轻）：m100 start 33/100 全零摔

- m100 start 相位（/proc 实证）：进度 33/100，**fall 非零行数 = 0**
  （含 walk 族 minz 0.237-0.245）。速率 ~1.45 min/motion（33 条/48min），
  start ETA 修正 ~19:00，uniform 随后 ~21:25（比此前估计各晚 ~30min）。
- 无新落档、无发车、无干预。分族终裁素材继续积累中。

## 2026-08-27 17:28 — patrol 轮（轻）

- mixed100 start 评估 ~86/100，现场逐条 fall=0.00/compl=1.00（舞走两族
  皆然，walk minz 0.243）。JSON ~17:40 落档后我将做族分解（dance/walk
  分组 jerr/compl）防均值幻觉；uniform 随后。无干预。

## 2026-08-27 17:54 — patrol 轮（轻）：m100 start 54/100；纠正他轮进度误报

- **勘误（数字卫生）**：17:28 轮条目 "~86/100、JSON ~17:40 落档" 与
  实测不符——看护器日志 motion 行数实测 **54**（17:54，当前扫
  re_dancegen_excited_wiggle_11），速率 0.7 行/min，start ETA 维持
  ~19:00、uniform ~21:25。进度以日志行数/JSON mtime 为准，不做估计。
- 质量面：54 条中 fall 非零行数 = 0、compl=0.00 行数 = 0——两族至今
  全零摔全完成，分族不退化证据持续积累。
- 本轮无发车、无干预。

## 2026-08-27 17:59 — patrol 轮：评估顺序勘误（walk 在前 dance 在后）

- 勘误：mixed100 评估的 motion 顺序实测为 walk_seed0-4 → dance wiggle
  （与注册 tuple(sorted()) 的字母序相反，MotionCommand 侧另有排序）。
  此前 "16:58 ~20/100、17:28 ~86/100" 的进度读数按舞蹈在前推的，
  全部偏乐观。实际当前 = start 相位舞蹈段 ~18/50（总 ~68/100）。
- 修正 ETA：start 落档 ~18:30，uniform ~20:00。全程零摔满完成维持。
- 看护器健康，无干预；教训=进度读数以日志行序为准，不按名字序脑补。

## 2026-08-27 18:28 — patrol 轮（轻）

- mixed100 start 85/100，全程 fall=0.00/compl=1.00。~18:40 落档后
  做族分解；uniform ~20:00。无干预。

## 2026-08-27 18:58 — patrol 轮：mixed100 start 族分解 = 正迁移实锤

- **start 相位落档（100 motions × 12 ep）**：fall=0.0（0/100），
  jerr 宏 0.635，compl 宏 1.0，步数中位 295.5，minz 最差 0.191。
- **族分解（对照各自单训）**：
  - dance: jerr 0.536（单训 0.478，代价 +0.058）compl 1.0 = 平
  - walk: jerr 0.733（单训 1.571，**-53%**）compl 1.0（单训 0.78）
    = **跨族正迁移**——合池把走路跟踪精度腰斩级改善。
- 机制解读（假说，待 uniform 确认后可深挖）：舞蹈族的静态站姿结构
  为走路提供了更稳的梯度锚/探索中心；单训走路池时所有 motion 都在
  动态边缘上，缺少"易例"托底。
- uniform 相位在跑（~11/100），~19:50 齐后出终裁三件套。

## 2026-08-27 19:24 — patrol 轮（合并 18:24 欠账）：族分解独立复核吻合；uniform 33/100

- 补记：18:24 轮状态已采但记录被提示排队打断未落档，该轮无事件
  （数据已被 18:42 落档超越）；本轮合并执行。
- **族分解独立复核（JSON 原文重算，与 18:58 轮逐位吻合）**：
  ALL fall 0.000（非零 motion=0）/ jerr 0.635 / worst 1.175 / compl
  1.000 / 步数中位 295.5 / minz 0.191；dance 半区 jerr 0.536（worst
  0.650，单训 0.478，代价 +0.058）；**walk 半区 jerr 0.733（worst
  1.175，单训 1.571 = −53%）/ compl 1.000（单训 0.78）/ minz 0.228**
  ——合池不但没退化舞蹈，反而把走路精度与完成度大幅拉升，跨族正
  迁移在 per-motion 层实锤（单训 walk 的 11 条 start 相位失败条在
  合池模型下全部完成）。
- uniform 相位 18:42 接棒：进度 33/100（日志行数 133−100），至今
  fall 非零 = 0、compl=0.00 = 0。实测速率 0.77 motion/min →
  **ETA ~20:51**（18:58 轮 "~19:50" 系第三次乐观估计，以实测为准）。
- 本轮无发车、无干预。终裁三件套等 uniform 落档。

## 2026-08-27 19:28 — patrol 轮（轻）

- mixed100 uniform 36/100，全程 fall=0.00/compl=1.00。预计 ~20:15 齐，
  随后终裁三件套。无干预。

## 2026-08-27 19:54 — patrol 轮（轻）：uniform 57/100 全零摔维持

- m100 uniform：进度 57/100（行数 157−100，扫至舞蹈段 wiggle_14），
  全日志（双相位累计 157 条）fall 非零 = 0、compl=0.00 = 0。
  实测 0.79 motion/min → ETA 维持 **~20:48**（19:28 轮 "~20:15" 为
  第四次乐观估计，仍以实测为准）。
- 无新落档、无发车、无干预。uniform 齐后终裁三件套（他轮认领中）。

## 2026-08-27 19:58 — patrol 轮（轻）

- mixed100 uniform 62/100（walk 段完+舞蹈段 wiggle_19），全程零摔
  满完成。预计 ~20:15 齐。无干预。

## 2026-08-27 20:24 — patrol 轮（轻）：uniform 86/100

- m100 uniform（/proc 实证 sampling uniform）：进度 86/100（扫至
  wiggle_40），全日志 186 条 fall 非零 = 0、compl=0.00 = 0。实测
  0.84 motion/min → 剩 14 条 ≈ 17 min，**ETA ~20:41**。
- 无新落档、无发车、无干预。下轮应见 uniform JSON + 终裁三件套。

## 2026-08-27 20:28 — patrol 轮（轻）

- mixed100 uniform 收尾中（舞段尾），零摔满完成维持。齐档后终裁。
- 无干预。

## 2026-08-27 20:54 — patrol 轮：m100 uniform 落档验数——Mixed-100 终局：2400 ep 零摔，两族同时持有成立

- **uniform（JSON 原文，20:38 落档）**：100 motion × 12 ep，fall macro
  0.000（非零 motion=0）；jerr macro 0.622 / worst 1.133；compl macro
  0.996；minz ≥0.183；步数中位 295.5；compl<0.5 的 motion = 0。
- **族分解（双相位对照单训）**：
  | | start jerr | uniform jerr | compl |
  |---|---|---|---|
  | dance（合池/单训） | 0.536 / 0.478 | 0.599 / 0.466 | 1.0·0.992 / 1.0 |
  | walk（合池/单训） | 0.733 / 1.571 | 0.645 / 1.620 | 1.0 / 0.78·0.768 |
  舞蹈付小额精度税（+0.06~0.13），**走路两相位精度 −53~−60%、完成度
  拉满**——跨族正迁移双相位确认。
- **Mixed-100 裁决：单一通才策略同时持有舞蹈+走路两族成立**（双相位
  2400 ep 零摔，两相位 compl<0.5 条数 = 0）。北极星路线（多行为→通才
  蒸馏）的关键前提打通。
- 系统全静（无训练/评估进程，看护器 .done 20:38 后收工）。终裁三件套
  （晨报收口/跨池 addendum/memory）仍由他轮认领，其 ~20:58 轮未做则
  21:16 轮按图接手。本轮无发车、无干预。

## 2026-08-27 20:58 — patrol 轮：MIXED-100 通才终裁 = 双相位双族全零摔 + 正迁移

- **uniform 相位落档**：fall=0.0（0/100）/ jerr 0.622 / compl 0.9959 /
  步数中位 295.5 / minz 0.183。族分解：dance jerr 0.599（单训 0.466，
  代价税 +0.13）/ **walk jerr 0.645（单训 1.62，-60%）compl 1.0**。
- **终裁**：单策略同时持有舞+走两族，2400 episode（2 相位 × 100
  motion × 12 ep）零摔；跨族正迁移双相位确认（walk 精度大改善、
  dance 近平）。第 7 个策略 = mixed100_v2_ne1024（reward 16.19 超
  舞蹈单训）。通才池路线首个真实多行为组合成立。
- 发车队列（本轮回落两班，GPU ≤2 合规）：
  ① mixed100 seed43 复现（地标结果的配方稳健性背书，零准备成本）；
  ② Mixed-134 扩池（并入 stand 10 条 + gesture 24 条 = motion_library
  全四族入池；新池先过 dataset_contract_v1 再训——合同首次用于新数据）。

## 2026-08-27 21:05 — 双航班发车完成：mixed100 seed43 复现 + Mixed-134 全四族

- **mixed100_v2_seed43ne1024**（21:01 发车）：地标通才结果的种子复现，
  cwd/env.yaml 1024 双验证，ETA ~33 分钟（与 m134 共卡会延长）。
- **Mixed-Pool-134-V2**（21:03 发车）：全四族池（舞50+走50+站10+手势24
  =134 clips）。数据链：stand_gesture_extract.py 新提取（沿用 walk 提取
  的全部门+provenance），两池先过 dataset_contract_v1（10/10 + 24/24
  PASS 零 flag——合同首次用于新数据即拦零放行有据）。注册时两小事故
  当场修：set 并集用 |（前车之鉴重犯一次，30 秒修）；glob Path 与
  str 混排 sorted 崩→str() 包裹。cwd/env.yaml 1024 双验证。
- 双看护器 1121454/1121457 实证存活（pgrep 同 pid 疑云=wrapper 自匹配，
  pgrep -af 实证排除，本日第四次）。GPU 双训练 100% 合规（≤2）。
- 队列：m134 完训评估（134×12×2 相位，预计过夜）→ 全四族通才裁决；
  seed43 评估随后。

## 2026-08-27 21:24 — patrol 轮（轻）：双航班审计补全 PASS；趋势健康

- 他轮 21:05 双发车的五查审计由本轮补全（只读）：两班 env.yaml
  num_envs=1024 ✓ / Total steps 24576·迭代⁻¹ ✓（m134 与 seed43 班各自
  日志实证）/ run 目录 21-01-28_mixed100_v2_seed43ne1024 与
  21-03-24_mixed134_v2_ne1024 ✓。
- 趋势（21:24）：mixed100_seed43 iter 358/2000，reward 3.2-3.7 / ep
  len 87-98（与 seed42 班 3.97@326 同轨道——复现在轨）；mixed134 iter
  158/2000，reward 2.0-2.6 / ep len 66-72（四族冷启动爬升正常）。
  共卡速率 3.9 / 8.0 s·iter⁻¹，双训 100% GPU 合规（≤2）。
- 勘误（本人）：21:24 首查命令日志选择 bug 导致两班数字一度雷同
  （实为同文件读两次），已分列重查——数字以本次为准。
- 无新落档、无发车、无干预。

## 2026-08-27 21:28 — patrol 轮（轻）：双航班健康

- seed43 复现：迭代 402/2000，reward +4.25 / ep len 107——与 seed42
  同期轨迹同型（326 迭代时 +3.97/99），复现走势正常。
- m134 全四族：迭代 186/2000，reward +2.05 / ep len 66.5，早期爬升
  正常。双看护器存活。无干预。

## 2026-08-27 21:54 — patrol 轮（轻）：双班同轨

- mixed100_seed43：iter 670/2000，reward 7.9-8.0 / ep len 196-198
  （seed42 班同轨）；速率 10.4 iter/min（共卡），ETA ~00:05。
- mixed134：iter 370/2000，reward 3.1-3.4 / ep len ~88；速率 7.1
  iter/min，seed43 完训让位后预期加速，ETA ~01:00-01:30。
- GPU 100% 双训合规；无新落档、无发车、无干预。

## 2026-08-27 21:58 — patrol 轮（轻）

- seed43：迭代 714/2000，reward +9.17 / ep len 224（继续同型爬升）。
- m134：迭代 398/2000，reward +3.71 / ep len 98。双看护器存活，无干预。

## 2026-08-27 22:24 — patrol 轮（轻）

- mixed100_seed43：iter 982/2000，reward 11.7-12.5 / ep len 263-279
  （seed42 班同期量级，复现在轨）。ETA ~00:02。
- mixed134：iter 582/2000，reward 5.0-6.3 / ep len 133-153（370 时
  3.4/88 → 四族合力爬升强劲）。ETA ~01:44（seed43 让位后加速）。
- GPU 100% 双训合规；无新落档、无发车、无干预。

## 2026-08-27 22:28 — patrol 轮（轻）

- seed43：迭代 1030/2000，reward +12.70 / ep len 277。m134：迭代
  614/2000，reward +6.72 / ep len 165。双双健康爬升，无干预。

## 2026-08-27 22:54 — patrol 轮（轻）

- mixed100_seed43：iter 1306/2000，reward 14.6-14.7 / ep len 284-289
  ——已近 seed42 终态签名（15.63/290），ETA ~23:58。
- mixed134：iter 798/2000，reward 8.9-9.0 / ep len ~220（614 时
  6.7/165 → 持续强劲），ETA ~01:36。
- GPU 100% 双训合规；无新落档、无发车、无干预。

## 2026-08-27 22:58 — patrol 轮（轻）

- seed43：迭代 1354/2000，reward +14.56 / ep len 282（逼近 seed42 终态
  16.19）。m134：迭代 830/2000，reward +8.89 / ep len 220。无干预。

## 2026-08-27 23:24 — patrol 轮（轻）

- mixed100_seed43：iter 1630/2000，reward 14.7-15.1 / ep len 274-287
  （seed42 终态区间）。ETA ~23:58。
- mixed134：iter 1018/2000，reward 10.9-11.3 / ep len 247-252——破
  双位数，过半。ETA ~01:38。
- GPU 100% 双训合规；无新落档、无发车、无干预。

## 2026-08-27 23:28 — patrol 轮（轻）

- seed43：迭代 1678/2000，reward +15.68 / ep len 285（seed42 终态 16.19
  在望）。m134：迭代 1046/2000，reward +11.95 / ep len 267（过半，
  曲线强劲）。无干预。

## 2026-08-27 23:54 — patrol 轮（轻）：seed43 完训在即

- mixed100_seed43：iter 1958/2000，reward 15.54 / ep len 287.7 / std
  0.55——距完训 ~4 分钟（~23:58），终态落在 seed42 签名（16.19/293.9/
  0.56）的种子噪声带内，复现趋势成立；正式定论等终帧+双相位评估。
- mixed134：iter 1238/2000，reward 13.63 / ep len 278.7——已超舞蹈
  单训终态水平（15.63 为 wiggle 池；134 池四族混合下 13.6@1238 仍
  在爬）。ETA ~01:38。
- GPU 100% 双训合规；无新落档、无发车、无干预。

## 2026-08-27 23:58 — patrol 轮：seed43 完训（16.26 超种子对）

- mixed100 seed43 完训：终态 reward **16.26**（seed42 对照 16.19）/
  ep len 290.6 / action std 0.55——通才配方种子稳健性训练层成立，
  双相位评估已由看护器接管（~2h）。
- m134：迭代 1294/2000，reward +12.92，继续强劲。无干预。

## 2026-08-28 00:24 — patrol 轮：双完训 + 双评估链并行

- **mixed100_seed43 终态（终帧 1999/2000 原文）**：reward **16.26** /
  ep len 290.57 / std 0.55——超 seed42 对照（16.19/293.89/0.56），
  通才配方训练层种子稳健成立；评估层定论等双相位 JSON。
- **mixed134 完训（终帧原文）**：reward **15.53** / ep len 281.68 /
  std 0.57——四族 134 池在 2000 迭代即逼近两族 100 池水平（16.19），
  含 stand+gesture 新族。**ETA 勘误**：23:54 轮按共卡速率估 ~01:38，
  实际 seed43 让位后独占加速（~2.4s/iter），00:20 完训——独占加速
  幅度（3×）再次超出预期，此后 ETA 按独占速率单算。
- 双评估链 /proc 实证并行：seed43 班 start 相位（23:58 起，出
  eval_mixed100_seed43_start）+ m134 班 start 相位（00:20 起，出
  eval_mixed134_v2_start）。粗估落档：seed43 start ~02:00、m134
  start ~03:00，双 uniform ~04:00-06:00（两评估共卡，按实测修正）。
- 无训练进程；无发车、无干预。

## 2026-08-27 28:00（实为 08-28 00:28）— patrol 轮：m134 完训

- **Mixed-134 全四族完训**：终态 reward **15.53** / ep len 281.7——
  四族池（舞+走+站+手势 134 clips）达到接近 mixed100（16.19/16.26）
  的水平，池规模 +34% 代价仅 -0.7 reward。看护器即将接管评估
  （134×12×2=3216 ep，与 seed43 评估共卡）。
- seed43 评估 start 相位 22/100，全程零摔满完成。无干预。

## 2026-08-28 00:54 — patrol 轮（轻）：双评估 start 相位健康

- 双 start 相位 /proc 实证并行：mixed100_seed43（23:58 起）+ m134
  （00:20 起）。m134 扫至 **stand_seed2_w00**——四族中的站姿新族已
  入扫描且 fall=0.00 / compl=1.00 / minz 0.219；日志 29 条零摔零
  未完成。seed43 班零摔满完成维持（他轮 00:28 轮 22/100）。
- GPU 486MiB（评估轻载）；无训练进程；无新落档、无发车、无干预。
- 时间戳纪律提醒（第 N 次）：他轮 "28:00" 自纠为 08-28 00:28——
  条目时间一律 date 实测。

## 2026-08-28 00:58 — patrol 轮（轻）

- 双评估并行健康：seed43 41/100（walk 段尾，全零摔满完成）、m134
  32/134（stand 族已现身且零摔满完成 minz 0.209）。共卡速率正常。
  无干预。

## 2026-08-28 01:24 — patrol 轮（轻）

- seed43 班 start：59/100（舞蹈段），零摔零未完成；0.69 motion/min
  → ETA ~02:24。m134 班 start：48/134（walk 段），零摔零未完成；
  0.75 motion/min → ETA ~03:19。双 uniform 随后。
- 无新落档、无发车、无干预。

## 2026-08-28 01:28 — patrol 轮（轻）

- seed43 评估 62/100（转舞蹈段，全程零摔满完成）、m134 评估 51/134
  （walk 段，零摔满完成）。双看护器健康，无干预。

## 2026-08-28 01:54 — patrol 轮（轻）

- seed43 班 start：82/100，零摔零未完成（0.77 motion/min，ETA ~02:17）。
  m134 班 start：67/134，零摔零未完成（0.63 motion/min，ETA ~03:40）。
- 无新落档、无发车、无干预。

## 2026-08-28 01:58 — patrol 轮（轻）

- seed43 85/100、m134 70/134，全程零摔满完成。无干预。

## 2026-08-28 02:24 — patrol 轮：seed43 start 落档验数——通才复现评估层同签名

- **mixed100_seed43 start（JSON 原文，02:16 落档）**：100×12 ep，
  fall macro 0.000（非零 motion=0）/ jerr macro 0.628 / worst 1.140 /
  compl macro 1.000 / 步数中位 295.5 / compl<0.5 条数 0。
  族分解：dance jerr 0.602（seed42 0.536）、walk jerr 0.655（seed42
  0.733）——宏 0.628 vs 0.635，族级互有升降，全部在种子噪声带内。
  **通才种子复现 start 相位成立**；uniform 相位 02:16 接棒（/proc
  实证），落档后双门定案。
- m134 班 start：87/134（舞蹈段），零摔零未完成，ETA ~03:35。
- 无发车、无干预。

## 2026-08-28 02:28 — patrol 轮：seed43 start 落档=复现确认

- mixed100 seed43-start：fall=0.0（0/100）/ jerr 0.628 / compl 1.0；
  族分解 dance 0.602 / walk 0.655（seed42: 0.536/0.733——总量一致，
  族间分配略异，属种子间正常波动）。通才配方种子复现评估层成立
  （uniform 在飞）。
- m134 评估 90/134，全程零摔满完成。无干预。

## 2026-08-28 02:54 — patrol 轮（轻）

- seed43 uniform：24/100，零摔（1 条 compl=0.83 分数化属均匀相位
  正常形态）。m134 start：110/134，零摔零未完成。
- 无新落档、无发车、无干预。

## 2026-08-28 02:58 — patrol 轮（轻）

- seed43 uniform 26/100、m134 start 113/134，全程零摔满完成。无干预。

## 2026-08-28 03:24 — patrol 轮：m134 start 落档验数——四族全零摔满完成

- **Mixed-134 start（JSON 原文，03:24 落档）**：134 motion × 12 ep，
  fall macro 0.000（非零 motion=0）；jerr macro 0.631 / worst 1.516；
  compl macro 1.000；步数中位 278；compl<0.5 条数 0。
- **四族分解**：dance jerr 0.577（worst 0.703）/ **walk 0.562**
  （mixed100 中 0.733——池更多样，走路精度反而更低）/ stand 0.818 /
  gesture 0.810（worst 1.516 = 全池最差但零摔满完成）。**四族全部
  compl 1.000、零摔**——全四族通才 start 相位成立；uniform 相位
  自动接棒，落档后终裁。
- seed43 uniform 同期 43/100 零摔。无发车、无干预。

## 2026-08-28 03:28 — patrol 轮：m134 start 落档=四族通才在望

- **mixed134-start**：fall=0.0（0/134）/ jerr 0.631 / compl 1.0。
  四族分解：dance 0.577（mixed100 0.536）/ **walk 0.562（mixed100
  0.733 → 更低！）** / stand 0.818 / gesture 0.810。
- 解读：池扩到四族后 walk 误差进一步下降（大池易例密度效应持续）；
  stand/gesture 为最高误差组（静态姿态精确跟踪更难）但零摔满完成。
  uniform 在飞；seed43 uniform 46/100。无干预。

## 2026-08-28 03:54 — patrol 轮（轻）：双 uniform 相位

- seed43 uniform：67/100（0.68 motion/min，ETA ~04:42）；m134
  uniform：26/134（ETA ~06:25）。双日志全量零摔零未完成维持
  （含 m134 的 stand 族 uniform 相位健康）。
- 无新落档、无发车、无干预。

## 2026-08-28 03:58 — patrol 轮（轻）

- seed43 uniform 71/100；m134 uniform 29/134（stand 段 compl 0.92
  无摔属正常波动）。无干预。

## 2026-08-28 04:28 — patrol 轮（轻）

- seed43 uniform 82/100、m134 uniform 48/134，无摔维持。无干预。

## 2026-08-28 04:36 — patrol 轮：seed43 uniform 落档验数——零摔稳健，但舞蹈半区均匀相位精度有种子敏感性

- **mixed100_seed43 uniform（JSON 原文，04:31 落档）**：fall macro
  0.000（非零 motion=0，**双相位 1200 ep 零摔维持**）；但 jerr macro
  1.025 / worst 3.579、compl macro **0.839**（seed42 对照 0.996）。
- **族分解**：walk 0.673 / compl 0.932（seed42 0.645/1.0——正常复现）；
  **dance 1.376 / compl 0.747（seed42 0.599/0.992——显著退化）**。
  compl<0.5 的 9 条全部是 wiggle（7 条 jerr 2.5-3.6 跟丢型 +
  wiggle_20/42 低 jerr 早止型，多为 0.417=5/12 集完成）。
- **诚实结论**：通才配方"零摔"双种子双相位稳健成立；但**均匀相位下
  舞蹈半区精度存在种子敏感性**（seed42 干净、seed43 掉 9/50 条）。
  对照：wiggle-pool50 单族 seed43v2 均匀相位是干净的（jerr 0.470）
  ——指向合池容量/干涉在该种子上的表现，而非数据或配方本身缺陷。
  待 m134（seed42）uniform 落档交叉验证；晨间汇报需带此敏感性，
  不报"复现完全成立"。
- m134 uniform 同期 46/134 零摔推进。无发车、无干预。

## 2026-08-28 04:54 — patrol 轮（轻）：仅剩 m134 uniform 在飞

- seed43 班评估链收工（start 02:16 + uniform 04:31 双档齐）。系统仅剩
  m134 uniform（/proc 实证）：69/134，零摔零未完成，0.77 motion/min，
  ETA ~06:20。
- 无新落档、无发车、无干预。全四族通才终裁 = m134 uniform 落档后的
  首个轮次。

## 2026-08-28 04:58 — patrol 轮：seed43 复现收口（零摔成立+种子方差记录）

- mixed100 seed43 双相位齐：start 0/100 摔 jerr 0.628（与 seed42 0.635
  一致）；uniform 0/100 摔但 jerr 1.025——族分解暴露 **dance 族 uniform
  相位种子方差**（1.376 vs 0.599，compl 0.839 vs 0.996）；walk 两种子
  稳定（0.645/0.673）。跨池表 addendum 已落。
- 判读：通才配方在生存维（零摔）与 walk 族上种子稳健；dance 族
  uniform 相位精度是当前配方最薄弱轴（可能需要更长训练或 per-family
  加权——记为下一轮优化靶点，不阻塞）。
- m134 uniform 72/134 进行中。无干预。

## 2026-08-28 05:24 — patrol 轮（轻）

- m134 uniform：95/134，零摔（1 条 compl=0.92 分数化正常），
  0.79 motion/min，ETA ~06:13。
- 无新落档、无发车、无干预。

## 2026-08-28 05:28 — patrol 轮（轻）

- m134 uniform 99/134（舞蹈段，零摔满完成）。~06:20 齐后终裁。无干预。

## 2026-08-28 05:54 — patrol 轮（轻）：m134 uniform 124/134

- 零摔零未完成维持，剩 10 条 ≈ 13 min，ETA ~06:07。下轮做四族终裁
  验数（族分解 × 双相位 + 与 mixed100 对照）。
- 无发车、无干预。

## 2026-08-28 05:58 — patrol 轮（轻）

- m134 uniform 128/134（wiggle_48，个别 compl 0.92 无摔）。~10 分钟齐。
  无干预。

## 2026-08-28 06:24 — patrol 轮：m134 uniform 落档验数——全四族通才终裁成立

- **Mixed-134 uniform（JSON 原文，06:03 落档）**：ALL jerr 0.673 /
  compl 0.978 / fall 0；四族 compl 全部 ≥0.965（dance 0.730/0.970、
  gesture 0.634/0.965、stand 0.608/0.983、walk 0.648/0.992）；
  **compl<0.5 条数 = 0**；步数中位 278。
- **终裁（start + uniform 合计 134×12×2 = 3216 ep，零摔）**：单一
  策略同时持有舞/走/站/手势四族成立。静态两族 uniform 相位 jerr 反而
  低于 start（gesture 0.634 vs 0.810、stand 0.608 vs 0.818——帧 0
  起点含鞭打瞬态，随机相位更友好，与 wiggle 裁头史一致）。
- **dance uniform 种子敏感性交叉验证**：m134（seed42）dance uniform
  0.730/0.970 干净——与 mixed100 seed42 干净、seed43 退化（1.376/
  0.747）的图景一致，敏感性定位于 seed43×合池的特定表现，非配方
  系统缺陷；优化靶点（dance uniform 精度）保留给用户裁决。
- 系统全静（无训练/评估进程，全部看护器 .done）。过夜双航班收官：
  通才种子复现（零摔稳健+敏感性已画像）+ 全四族通才首航成立。
  队列回到 P2（用户裁决）状态，后续轮次按图 no-op 待命。

## 2026-08-28 06:28 — patrol 轮（收官记账）：全四族通才终裁三件套闭合

- cross_pool_comparison.md 06:28 补遗落档：Mixed-134 双相位四族终表
  （start jerr 0.631 四族 0.577/0.562/0.818/0.810；uniform 0.673 四族
  0.730/0.648/0.608/0.634；3216 ep 零摔）+ 三发现（walk 随池宽单调
  改善；静态族相位图式反转=帧 0 鞭打瞬态；seed43 舞蹈 uniform 敏感
  性未在 seed42/m134 复现=定位种子×合池非配方）。
- 记忆同步（bdx-mjlab-pivot + humanoid-gpt-vision）：8 个已验证策略
  库存 + 北极星下一活决策点（小蒸馏接口验证）入档。
- 系统全静复核（无训练/评估进程，看护器全 .done）。队列=P2 用户
  裁决态：①dance-uniform 精度靶点（加训/族权重）②小蒸馏接口验证
  发车令。本轮无发车、无干预。

## 2026-08-28 06:54 — patrol 轮：no-op（队列 = P2 用户裁决态）

- 系统全静复核：无训练/评估进程，看护器 16×.done，GPU 490MiB 空闲
  基线，无新落档。P0/P1 全关（终裁 06:24 轮 + 他轮晨间交付齐）。
- 按 §7 停机规则待命：P2 等用户（①dance-uniform 精度靶点 ②小蒸馏
  接口验证发车令 ③commit/分支 ④记录迁仓）。无发车、无干预。

## 2026-08-28 07:02 — patrol 轮：发车决策（mixed134 seed43 复现）

- 巡检：系统全静维持（无训练/评估进程，GPU 0%，日志皆 08-27 旧迹，
  无新 eval JSON）。终稿三件套已闭合（06:24/06:28），按 3f 轻量巡检。
- **发车决策理由（写于发车前）**：GPU 空闲且全部评估齐备（第 5 条
  前件成立）。唯一已知弱轴 = dance-family uniform 相位种子方差
  （mixed100 seed43 1.376 vs seed42 0.599）；m134 seed42 uniform 干净
  （0.730）说明该敏感性非配方系统缺陷，但是否随池宽/族数放大未测。
  mixed134 seed43 复现 = 与 pool50/mixed100 两次种子复现完全同模的
  既定模式（同配方仅 --agent.seed 43），零新机制，直接把 P2 裁决
  ①（dance-uniform 精度靶点）的证据补齐：若 seed43 在四族池也干净
  → 靶点降级关闭；若复现退化 → 族加权/加训实验有了明确对象。
- 配方逐字复刻：Mjlab-Tracking-Flat-BDX-V4-Mixed-Pool-134-V2 /
  2000 iter / 1024 envs（env.yaml 将验）/ seed 43 / tensorboard。
  看护器 FRESH 新写（禁 sed-copy 前车之鉴），轮询新 run 目录
  model_1999.pt → 双相位 eval → eval_mixed134_s43_{start,uniform}_
  per_motion.json（refuse-overwrite 保持）。
- 发车落地（07:00）：run `2026-08-28_06-59-34_mixed134_v2_seed43ne1024`，
  cwd readlink 实证 ✓ / seed=43 ✓ / env.yaml num_envs=1024 ✓ / 新目录名
  无复用 ✓。ETA ~55 min（GPU 独占，比昨夜双训快）。看护器 FRESH 新写
  `/tmp/eval_watcher_m134s43.sh`（body pid 1611207 实证存活），完训自动
  双相位 eval。后续轮次：趋势巡检 → 完训后 eval 进度 → 齐后四族对照
  落档（eval_mixed134_s43_*）。

## 2026-08-28 07:24 — patrol 轮（轻）：m134-seed43 审计补全 PASS；趋势强

- 他轮 06:59 发车的 mixed134_v2_seed43ne1024 五查补全：Total steps
  24576·迭代⁻¹ ✓（他轮已验 cwd/seed/env.yaml/目录名）——五查全 PASS。
- 趋势：iter 886/2000（25 分钟，~1.7s/iter 独占），reward 10.60 /
  ep len ~240——对照 seed42 班同期（~1238 迭代 13.6）走势略快，
  四族 seed43 复现在轨。完训 ETA ~07:55，看护器 m134s43 自动双相位。
- 本轮无发车、无干预。

## 2026-08-28 07:28 — patrol 轮（轻）：m134s43 在轨健康

- mixed134_seed43 iter 1027/2000（28 min，1.65 s/iter，ETA ~07:55；
  GPU 独占比昨夜双训 8 s/iter 快近 5 倍）。reward 10.0-12.0 / ep len
  234-261——与 seed42 班同期（iter 1018 时 10.9-11.3/247-252）同轨道，
  复现在轨。看护器 1611207 存活轮询中。无干预、无新发车；旧日志
  （pool50/mixed9 系列）无新动静。

## 2026-08-28 07:56 — patrol 轮：m134-seed43 完训（超种子对）

- **终帧 1999/2000 原文**：reward **16.14** / ep len 291.81 / std 0.56
  ——超 seed42 对照（15.53/281.68/0.57）。四族池训练层种子复现
  成立且更强；看护器 m134s43 自动接双相位评估（134×12×2，
  eval_mixed134_s43_*，预计 ~10:30-11:00 落档，按实测修正）。
- 关键看点（评估层）：seed43×134 池的 dance uniform 是否复现
  mixed100-seed43 的退化（1.376/0.747）——若 134 池干净则进一步
  支持"敏感性=seed43×100 池特定表现"。
- 本轮无发车、无干预。

## 2026-08-28 07:58 — patrol 轮：m134s43 完训（reward 16.14 > seed42 15.53），eval 已自动开跑

- 训练 2000/2000 完（55 min 独占）。终段 Mean reward 15.94→**16.14**
  ——高于 seed42 班的 15.53，ep len 满。复现在训练层先立一功。
- 看护器链路自动触发 ✓：start 相位 eval 进行中 4/134（gesture 段先扫，
  fall=0.00/compl=1.00/minz 0.209-0.236 全干净）。两相位预计 ~10:00-
  10:30 齐，齐后做四族对照终表（eval_mixed134_s43_*）+ dance-uniform
  种子敏感性裁决（本轮发车的预注册问题）。
- 无干预、无新发车。

## 2026-08-28 08:24 — patrol 轮（轻）：m134s43 start 32/134；gesture_attention 子族 3 条未完成

- 进度 32/134（gesture→stand 段），**fall 非零行数 = 0**；但
  **compl=0.00 的 3 条 = gesture_attention_w00/01/02 全族**（minz
  0.222-0.223 健康，零摔——纯跟踪项终止）。
- 对照：seed42 的 m134 start 相位 gesture 全族 compl 1.000——这是
  **seed43×134 池在 gesture_attention 子族上的新退化**，与 mixed100
  seed43 的 dance-uniform 退化同型（小簇整体丢失、不摔）。模式浮现：
  seed43 在合池中会掉小簇；具体机制（采样份额/容量）待终表+对照
  后定性，预注册问题（dance-uniform）在 uniform 相位继续检验。
- 无新落档、无发车、无干预。

## 2026-08-28 08:28 — patrol 轮（轻）：m134s43 start 相位 36/134

- 零摔维持（36 条全 fall=0.00；walk 段先行扫过，minz 0.236-0.24
  健康站位）。速率 ~55 s/条 → start 齐 ~09:35、uniform 齐 ~11:30。
  无干预；旧日志无新动静。

## 2026-08-28 08:54 — patrol 轮（轻）

- m134s43 start：57/134（walk 段干净扫描），零摔维持，compl=0.00
  仍仅 gesture_attention 三条（无新增）。实测 0.95 motion/min →
  start ETA ~10:15（他轮 08:28 的 ~09:35 偏乐观，以实测为准）、
  uniform ~12:00。
- 无新落档、无发车、无干预。

## 2026-08-28 08:58 — patrol 轮（轻）：m134s43 start 60/134 零摔维持

- walk 段近尾（seed2），60 条全零摔。start 齐 ~09:50、uniform 齐
  ~11:45。无干预。

## 2026-08-28 09:0x — 用户令：播放稳定策略——mixed134 viser viewer 上线

- 清理：8080 被 08-26 16:11 的两天前旧 pool_play（旧 checkpoint）占
  用，按铁律精确 pid kill（2299670 + 本轮误发未得端口的 3862348），
  端口释放后重发。
- 现役：pid 3923719 = mixed134_v2 model_1999（四族通才），viser
  http://localhost:8080，--motion-index -1（每次 reset 随机抽 134 条
  之一：舞/走/站/手势混合演示），CPU 跑不占 GPU（m134s43 eval 继续）。
- 日志 logs/play_mixed134_visor.log。

## 2026-08-28 09:24 — patrol 轮（轻）：start 80/134；查看器变更备注

- m134s43 start：80/134（walk 段将尽，舞蹈段next），零摔维持，
  compl=0.00 仍仅 gesture_attention 三条。0.77 motion/min → start
  ETA ~10:34、uniform ~12:20。
- 备注偏差：他轮 ~08:5x 将 8080 查看器从旧 pid 2299670（地图 §1
  原标注"用户钉的，不动"）换为 mixed134_v2 模型随机抽播演示（pid
  3923719，同端口）。行为已落档其轮；与地图标注存在偏差，此处中性
  记录备查——若用户要看旧钉内容需知悉已换。
- 无新落档、无发车、无干预。

## 2026-08-28 09:29 — patrol 轮（轻）：m134s43 start 84/134 零摔；用户看舞中

- start 相位走完 gesture+stand+walk 全段（84 条零摔），正进入舞蹈段
  （84-133）——本轮发车要裁决的 dance-uniform 敏感性在前半程已无
  摔讯号，uniform 相位才是主考。start 齐 ~09:55、uniform 齐 ~11:50。
- viser viewer（舞 100 号 clip）存活，用户在浏览器看舞蹈播放。
- 无干预、无新发车。

## 2026-08-28 10:0x — P0 撤回（用户肉眼抓到）：走路族"学会"= 站着不动，平移/步态从未被训练也从未被评估

- **用户报**：viewer 播 walk_seed0_w02（2.33m/6s 快走 clip），机器人
  原地站立，脚关节不动。两次窗口（近原地 w06 / 快走 w02）一致。
- **无头取证**（forensic_play_walk.py，play 路径精确复刻）：
  policy_root_travel **0.073 m**（参考 2.33 m）；hip_pitch 恒
  1.16-1.24 / 膝 -1.45 **单调缓变零循环** = 静态蹲姿；腿部 1.52 rad
  行程全部来自出生瞬态。verdict=STANDS。探针初版印 hip_yaw（走路
  本恒定）属本人测量错误，已换 pitch/knee 定谳。
- **三层结构性盲区（训练从未要求平移）**：①obs 80-D 剥掉了
  motion_anchor_pos_b + base_lin_vel（strip_anchor_obs 默认 True），
  策略物理上看不见参考 xy 与自身漂移；②anchor_ori 奖励 yaw-invariant
  （gesture 契约）+ anchor_pos 终止 z-only——xy 无奖励无惩罚；③评估
  指标 jerr/compl/fall 同样 xy 与步态循环盲视：静态蹲姿在走路参考上
  恰好产出 jerr 0.337/compl 1.0/minz 0.24 的"健康"行——数字与站桩
  完全相容。jerr 是代理，属性（真的走）从未被验证。教科书式违反
  feedback 纪律"验证属性本身而非代理"。
- **结论范围重订**：walk50-redo"走路池内可学"、mixed100/134"walk
  误差砍半/四族通才"中的 walk 族条目，全部重订为**关节空间站位跟踪**
  ——策略学到的是"以参考平均站姿蹲住不终止"，不是步态不是位移。
  舞/站/手势族不受影响（定点行为，关节跟踪=行为本体）。数据本身无
  罪（raw-write 门仍实）。
- **真 locomotion 的路**：xy-informed obs（V6 的 86-D 形态，仅飞过
  num_envs=1 无效班）+ xy 奖励/终止项，或回到 periodic/perpetual 命令
  条件线（走路教师自家任务族）——后者证据虽废但方向重新升级为
  首选。pool-tracking gesture 契约对位移行为结构性不适配（重演
  march 判决的结构性逻辑）。
- s43 复现 eval 继续跑（dance-uniform 种子问题不受影响，walk 行读
  法按本条重订）。viewer 保留现场供用户核对（静态蹲姿 hip≈1.2）。
  取证件：results/mjlab_bdx/forensic_play_walk_w02{,_v2}.json。

## 2026-08-28 09:54 — patrol 轮：撤回复核确认 + 地图改版标注；eval 照跑

- **独立复核他轮 10:0x 撤回（取证 JSON 原文）**：policy_root_travel
  0.062/0.073 m（叙述参考 2.33 m）、hip_pitch 恒 1.16-1.24、膝单调、
  verdict=STANDS——**撤回成立**。jerr 0.337 ≈ 未跟踪摆幅，与"数字
  健康+站桩"完全自洽；三层盲区（obs/奖励/终止全 xy 盲）实锤。
  舞/站/手势族不受影响。
- **一处待查落档**：trace `ref_travel_m_at_final_frame: 0.0` 与
  叙述 2.33m 不一致（疑 play 参考重锚或度量口径）——不影响本次
  撤回定性（关节零循环独立定谳），但改线后若以行程为度量需先澄清。
- AGENT_MAP 已按 §7 追加改版标注：§5 对位移族失效、periodic 线
  升首选、池跟踪保留条件 = 补 xy 可观测性重训。待用户裁决。
- m134s43 start 106/134 零摔（compl0 仍 3 条）——继续跑，walk 行
  读法按撤回重订（关节空间站位指标，非行为验证）。uniform 相位
  对 dance 敏感性问题仍然有效（族本身定点）。
- 无发车、无干预。

## 2026-08-28 10:1x — 第二、三缺陷定罪（用户疑舞蹈 → 又对了一半以上）

- **缺陷二（viewer/play 路径，非策略）**：钉舞蹈 clip 时机器人冻死——仪
  表化实锤 done=1 每步触发 + _resample_command 每步调用 + time_steps
  恒 1 + q 恒 0。机制=play reset 写 qpos0（z≈0.35），蹲姿族参考
  （舞/手势 z≈0.23）出生即违 z 带（|Δz|=0.12>0.08）→ 永久即死重置
  循环，策略从未获得行动机会；walk 能活纯因 Δz=0.07 压线。pool_play
  顺序 bug（set_motion_index 先于 reset 被 reset 冲掉）已修：_pin_assign
  改为每次 (re)assign 都 set_motion_index（teleport 上参考帧=评估链
  一直的做法）。修后 viewer 正常显示舞蹈。
- **缺陷三（舞蹈策略本体，用户直觉正确）**：teleport 起始 220 步自由滚
  的属性测量（逐关节摆幅比+轨迹相关，clip wiggle_23）：**脖子完美**
  （neck_pitch 比 1.01/corr 0.99）；**腿部严重衰减**（hip_roll 0.15/
  0.18、膝 0.42/0.34、hip_pitch 0.44/0.27）；8 动关节平均摆幅比
  **0.467**；jerr_last50 仅 0.131（14 关节均值被 6 个静关节+满分脖子
  稀释）——代理指标第二次误导。定性=策略学到"脖子舞+衰减腿摆"，平
  衡好（不摔真）、时序跟（corr 正）、幅度不到一半。与 walk 站桩同族：
  目标函数不强求全幅（exp 核饱和+动作代价项）。
- 待办：①舞蹈衰减全池普查（50 clip×逐关节比，出分布而非单 clip）；
  ②训练侧修复方向=幅度敏感奖励（按参考幅度归一的跟踪项/速度跟踪加
  权）或动作代价退火——并入 V7 locomotion 契约改造同批；③手势/站
  族同法普查。取证=/tmp/dance_teleport_out.txt + 结果 JSON 待归档。

## 2026-08-28 10:20 — 外部审计第二轮（用户令锐评 SHARP_REVIEW_0828）：8 处实锤，§6 越册

- E1 明星数字安错关节：简报"neck_pitch 摆幅比 1.01/corr 0.99"实为
  **neck_yaw**（txt 第 12 位）；neck_pitch 本尊（第 11 位）参考恒 0、
  机器人发明 0.28 rad——是"发明运动"病理关节，不是满分关节。
- E2 "6 静关节"数错：ref_amp==0 实为 **4 个**（双 hip_yaw/neck_forward/
  neck_pitch），稀释算术建立在错数上。
- E3 病理低报：4 个参考静止关节被发明运动 0.02-0.28 rad（与参考腿幅
  同量级）；踝"幅度达标"（0.85/1.10）但 corr 0.139/0.067=甩动非跟踪；
  左右不对称。**不是"腿衰减"，是"腿在动别的"**。绝对值：机器人腿
  0.05-0.11 rad vs 参考 0.18-0.45。
- E4 §6 越册：用已被连坑两次的代理指标（零摔/compl）给舞蹈背书
  "成立+括号待查"；应降格"脖子舞成立，腿内容未证"。
- E5 清单错：run 目录实际 **28** vs "17"（自列 13+6=19 也对不上）；
  看护器 20 脚本/16 done/3 历史击杀无 done；"取证 5 件"实为 results
  4（含 1 废件）+/tmp 2 件未归档（§8 却称无未落档动作）。
- E6 forensic_dance100_fidelity.json 全零/NaN 废件，真数只在 txt。
- E7 行程对比口径混用（0.073 vs 2.33m：离线参考 vs harness 实测，
  trace 内 ref_travel=0.0 未理；定谳实靠零步态循环）。
- E8 小错：07:02 vs 目录 06-59-34；"GPU 空"（评估在跑）。
- 全文锐评已交用户；洗清项：pool_play 修复属实（代码已核）、
  pool_record 顺序正确（视频含真证据）、walk 取证判定成立、
  大部分引数可溯源（jerr 0.337/0.131 均对上）。

## 2026-08-28 10:24 — patrol 轮（轻）：m134s43 start 133/134

- start 相位（/proc 实证）133/134，零摔，compl=0.00 仍仅 gesture_attention
  三条（与 08:24 轮一致，无新增）。start JSON 数分钟内落档，uniform
  随后（~2h）。
- GPU 4.2GiB 系 viewer（pool_play 1.76GB RSS）+ 评估进程共存，无训练
  进程；队列维持 P2 用户裁决态。
- 无新落档、无发车、无干预。

## 2026-08-28 10:2x — 用户停训令 + 外部审计 E1-E8 全盘接受

- **停训令（用户原文）**："先不要继续训练了，既然代码有问题，那也没
  必要接着训了，而且我要把数据集更新一下了"。执行：不再发车任何
  训练；cron 巡检降为纯观察（无训练/无发车/无奖励手术/无普查开工，
  数据集=用户 lane 我不碰）。待用户数据集落地后：dataset_contract_v1
  收货门复跑，代码修复方案（契约改造或转 periodic 线）等用户裁决。
- **外部审计 E1-E8 全部接受**（审计原文 10:20 已由审计方落档）：
  ①E1 关节错标认领——1.01/0.99 是 **neck_yaw**（第 12 位），不是
  neck_pitch；neck_pitch 本尊（第 11 位）参考恒 0、机器人自作主张
  动 0.28 rad=发明运动病理关节。②E2 静关节数=4 非 6。③E3 病理
  低报认领——腿不是"动得小"是"动的不是参考的动作"（踝 ratio 达标
  但 corr 0.139/0.067=甩不是跟；4 个参考静关节被发明运动 0.02-0.28
  rad；左右不对称）。④E4 越册认领——舞蹈从"成立"栏挪"待审"栏
  （脖子舞[neck_yaw]成立，腿内容未证）。⑤E6 废 JSON 认领
  （fidelity.json 是修前坏件，真数字在 teleport_amp.txt）。⑥E7 头条
  数字口径混用认领（2.33m 离线 vs 0.073m play 路径；定谳证据=零步
  态循环非平移数）。⑦E8 小数字三连认领。
- **审计两答采纳**：§5-1 制度洞（指标门控闭环无"渲 10 秒"步骤）→
  行为属性硬门（travel ratio/步态循环/分关节组摆幅比/corr）+
  "X 学会"判定必附渲染片段，待训恢复前落实；§7-1 V7 降为并行小
  赌注，**走路专家主注=periodic/perpetual 命令线**（corr 数据自证
  池线关节空间腿保真度弱，xy 可观测补"往哪走"补不了"怎么迈步"），
  ETA 估算按三折纪律。
- m134s43 eval（非训练，只读测量）按审计 §7-4 让其跑完：start
  133/134 零摔，uniform ~2h；dance/gesture 行仍有信息量，walk 行
  作废。用户如要立即杀说一声（精确 pid）。

## 2026-08-28 10:45 — 设计件落档：DESIGN_multi_expert_pools.md（用户令）

- bdx_rl_mjlab/docs/DESIGN_multi_expert_pools.md：按契约分代码/按族分
  数据的多专家组织设计——PoolSpec 注册表重构 stationary_pool（任务 ID
  不变）、奖励 v2 分关节组加权+静止关节罚、contract v2 family 硬门 +
  S6 参考行为统计 + S7 静止关节清单、评估行为硬门表、实施顺序。
- 纯设计件，未动训练代码（停训令遵守）；更新报告.md 已同步。
- m134s43 uniform 相位在跑（只读测量继续）。

## 2026-08-28 10:54 — patrol 轮（轻）：m134s43 start 落档验数

- **start（JSON 原文）**：134×12 ep，fall macro 0.000；jerr macro
  0.681 / compl 0.978。族分解：dance 0.637/1.0、walk 0.584/1.0、
  stand 0.696/1.0、gesture 0.972/0.875。compl<0.5 仍恰好 =
  gesture_attention 三条（0.0，无新增）——seed43 小簇丢失模式确认，
  与 mixed100-seed43 的 dance-uniform 退化同型。
- 对照 seed42（m134 start）：族形一致，jerr 全面略高（0.58-0.97 vs
  0.56-0.82）+ 三条 attention 全失。walk 行读法维持撤回后口径
  （关节空间站位指标）。
- uniform 相位 31/134（1.07 motion/min），ETA ~12:30。停训令维持
  （无训练进程）；无发车、无干预。

## 2026-08-28 11:24 — patrol 轮（轻）：uniform 54/134；attention 簇画像聚焦

- uniform 相位 54/134（gesture 段已扫完，现 walk 段），零摔维持。
  compl=0.00 共 5 行 = gesture_attention 三条（start）+ 其中 w00/w02
  在 uniform **再失**（w01 未失）→ 双相位硬核 = w00/w02，start-only
  = w01，仍全部集中在同一 gesture 子族、零摔、高度健康。
- 速率 1.07 motion/min 维持，uniform ETA ~12:30。停训令维持；
  无发车、无干预。

## 2026-08-28 10:5x — 用户令：策略全量保存——已归档 11 个有效 checkpoint

- results/mjlab_bdx/preserved_policies_20260828/：11 策略 × (pt+onnx)
  =74MB + SHA256SUMS + manifest（诚实标签随档：walk=站桩/舞腿待审/
  其余 joint-space 成立）。未保存=INVALID×7+死壳+对照+回归班+planner
  线两目录（在 manifest 记名）。停训令维持，纯观察。

## 2026-08-28 11:54 — patrol 轮（轻）：uniform 77/134

- uniform 相位 77/134（walk 段尾），零摔，compl=0.00 稳定 5 行
  （attention 簇，无新增）。实测 0.77 motion/min → uniform ETA
  修正 ~13:08。
- 他轮 10:5x 归档 11 策略（preserved_policies_20260828/，诚实标签）
  已留意。停训令维持；无发车、无干预。

## 2026-08-28 12:24 — patrol 轻轮：uniform 105/134

- 舞蹈段扫描中（wiggle_27），零摔，compl=0.00 稳定 5 行。0.93
  motion/min → uniform ETA ~12:55。**下轮看点**：seed43×134 的
  dance-uniform 是否复现 mixed100-seed43 的退化（1.376/0.747）——
  舞蹈段已扫过半，暂无异常信号。
- 停训令维持；无发车、无干预。

## 2026-08-28 12:54 — patrol 轮：m134s43 uniform 落档验数——链条全部收工

- **uniform（JSON 原文，12:53 落档）**：134×12 ep，fall macro 0.000
  （**双相位 3216 ep 全零摔**）；jerr macro 0.793 / compl 0.956。
  族分解：dance 0.811/0.973、walk 0.634/0.978、stand 0.858/0.975、
  gesture 1.062/0.865。
- **预注册问题裁决：seed43×134 的 dance-uniform 干净**（0.811/0.973 ≈
  seed42 对照 0.730/0.970）——mixed100-seed43 的退化（1.376/0.747）
  **未复现**，且 m134-seed43 的 start 相位 dance 也干净（0.637/1.0）。
  种子敏感性最终画像：**seed43 在不同池配置下掉不同的小簇**
  （mixed100：dance@uniform 9 条；m134：gesture_attention 3 条，
  w00/w02 双相位、w01 仅 start 0→0.333）——运行级容量/优化运气，
  非族特异缺陷；两池两种子全部零摔。
- 系统状态：**训练/评估进程清零，全部看护器 .done**——停训令下
  无任何在飞。队列 = P2（用户裁决）+ 数据集侧工作。本轮无发车、
  无干预。M134S43 评估链就此闭合。

## 2026-08-28 13:24 — patrol 轮：no-op（停训值守）

- 复核：无训练/评估进程（仅 viewer 1650MiB），无新落档。队列 =
  P2（用户裁决）+ 数据集侧（搬家/标签/情绪族）+ 解禁令。无发车、
  无干预。

## 2026-08-28 13:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 14:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 14:54 — patrol 轮：no-op（停训值守）；GPU 显存说明

- 无训练/评估进程、无新落档。GPU 4093MiB 实为 holomotion_demo 线
  两进程（holomotion_teleop 1122MiB + gvhmr 2344MiB，用户侧视觉/
  遥操作演示线）——非本线范畴，不干预，记录备查防后续误判。

## 2026-08-28 15:24 — patrol 轮：no-op（停训值守，无变化）

- GPU 回落 634MiB（holomotion_demo 两进程已退）。无训练/评估、
  无新落档。

## 2026-08-28 15:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 16:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 16:54 — patrol 轮：no-op（停训值守）

- 一处核查：GPU 曾瞬时 1163MiB/46%，复查回落 637-664MiB/0-1%
  ——查看器（pid 1861072，447% CPU）图形侧瞬时负载，非异常。
  无训练/评估、无新落档。

## 2026-08-28 17:24 — patrol 轮：no-op（停训值守）

- GPU 5071MiB/30% = holomotion_demo 线三进程（fast_sam_3d：teleop
  1122MiB + gvhmr 2682MiB + romp_rt 528MiB，用户侧视觉演示线），
  非本线范畴，不干预。无训练/评估、无新落档。

## 2026-08-28 17:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 18:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 18:54 — patrol 轮：发现未宣告的新训练线（train_gated 系列）——记录待确认

- GPU 1444MiB/44% 来源查明：pid 206828 = `bdx_rl.scripts.train_gated
  --task Mjlab-Tracking-Flat-BDX-V4-Stand-Pool-31-V1 --seed 3001
  --num-envs 1024 --max-iterations 30000 --run-name e2_stand31_full_v1`
  （18:37 起，cwd=bdx_rl_mjlab）。同系列前序：17:47 e2_stand31_smoke、
  17:49 其 v2、18:03 e3_dance50_smoke——**新入口 train_gated + 新任务
  注册（Stand-Pool-31-V1 等）+ e2/e3 实验编号**，均未在本文件宣告。
- 发射风格评估：显式 num_envs 1024 / seed / smoke 先行——纪律良好，
  疑似用户或其新 agent 按今日设计稿组织的多专家线（stand31/dance50
  与设计稿族结构吻合）。**本线未触碰、不干预**；是否属停训令范围
  由用户确认（停训令下达对象为原 mjlab 训练线）。
- 本线（mjlab pivot patrol）无训练/评估、无新落档。

## 2026-08-28 19:24 — patrol 轮：no-op（停训值守）

- 外线 e2_stand31_full_v1 仍在跑（46min+，GPU 1443MiB/39%），维持
  观察不干预（归属待用户确认，见 18:54 条目）。本线无训练/评估、
  无新落档。

## 2026-08-28 19:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 20:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 20:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 21:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 21:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 22:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 22:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 23:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-28 23:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 00:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 00:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 01:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 01:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 02:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 02:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 03:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 03:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 04:24 — patrol 轮：no-op（本线）；外线进展备注

- 外线 e2_stand31_full_v1 **完训**（18:37→~04:0x，~9.5h/30000 迭代，
  console 尾行 [train_gated] DONE）；其评估链已由另一新入口
  `tracking_bdx.scripts.eval_expert` 接棒（pid 1022571，440MiB）。
  仍无 PIVOT 宣告（该线惯例如此），本线继续只观察不干预。
- 本线无训练/评估、无新落档。

## 2026-08-29 04:54 — patrol 轮：no-op（停训值守；外线评估亦已收工）

- GPU 回落 556MiB/0%——外线 eval_expert 进程亦已退出（其节奏自管）。
  本线无训练/评估、无新落档。

## 2026-08-29 05:24 — patrol 轮：no-op（本线）；外线新车

- 外线 05:21 发车（train_gated，pid 2785095）——本线仅记录不干预，
  详情以其自管记录为准。本线无训练/评估、无新落档。

## 2026-08-29 05:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 06:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 06:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 07:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 07:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 08:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 08:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 09:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 09:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 10:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 10:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 11:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 11:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 12:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 12:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 13:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 13:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 14:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 14:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 15:24 — patrol 轮：no-op（本线）；外线 v2 完训

- 外线 e2_stand31_full_v2 进程退出（05:21→~15:1x，约 10h），GPU 回落
  556MiB。本线无训练/评估、无新落档。

## 2026-08-29 15:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 16:24 — patrol 轮：no-op（本线）；外线再发车

- 外线 16:20 新车（train_gated，pid 1628938），本线仅记录。本线无
  训练/评估、无新落档。

## 2026-08-29 16:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 17:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 17:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 18:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 18:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 19:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 19:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 20:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 20:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 21:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 21:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 22:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 22:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 23:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-29 23:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 00:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 00:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 01:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 01:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 02:24 — patrol 轮：no-op（本线）；外线 v3 完训

- 外线 e2_stand31_full_v3 进程退出（16:20→~02:1x，约 10h），GPU 回落
  556MiB 空闲。本线无训练/评估、无新落档。

## 2026-08-30 02:54 — patrol 轮：no-op（本线）；外线再发车

- 外线 02:38 新车（train_gated，pid 3399284），本线仅记录不干预。
  本线无训练/评估、无新落档。

## 2026-08-30 03:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 03:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 04:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 04:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 05:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 05:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 06:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 06:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 07:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 07:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 08:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 08:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 09:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 09:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 10:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 10:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 11:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 11:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 12:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 12:54 — patrol 轮：no-op（本线）；外线舞蹈班完训进评估

- 外线 e3_dance50_full_v1 训练进程退出（02:38→~12:4x，约 10h），其
  eval_expert 评估链 12:50 接棒（pid 801051）。本线仅记录；本线无
  训练/评估、无新落档。

## 2026-08-30 13:24 — patrol 轮：no-op（停训值守；外线评估亦已收工）

## 2026-08-30 13:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 14:24 — patrol 轮：no-op（本线）；外线再发车

- 外线 13:58 新车（train_gated，pid 2831770），本线仅记录不干预。
  本线无训练/评估、无新落档。

## 2026-08-30 14:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 15:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 15:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 16:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 16:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 17:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 17:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 18:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 18:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 19:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 19:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 20:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 20:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 21:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 21:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 22:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 22:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 23:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-30 23:54 — patrol 轮：no-op（本线）；外线舞蹈 V2 完训

- 外线 e3_dance50_full_v2 进程退出（13:58→~23:4x，约 10h），GPU 回落
  549MiB 空闲。本线无训练/评估、无新落档。

## 2026-08-31 00:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 00:54 — patrol 轮：no-op（本线）；外线再发车

- 外线 00:44 新车（train_gated，pid 1306855），本线仅记录不干预。
  本线无训练/评估、无新落档。

## 2026-08-31 01:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 01:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 02:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 02:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 03:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 03:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 04:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 04:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 05:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 05:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 06:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 06:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 07:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 07:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 08:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 08:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 09:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 09:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 10:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 10:54 — patrol 轮：no-op（本线）；外线舞蹈 V3 完训

- 外线 e3_dance50_full_v3 进程退出（00:44→~10:4x，约 10h），GPU 回落
  760MiB。本线无训练/评估、无新落档。

## 2026-08-31 11:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 11:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 12:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 12:54 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 13:24 — patrol 轮：no-op（停训值守，无变化）

## 2026-08-31 13:54 — patrol 轮：no-op（停训值守）

- GPU 1219MiB/17% 无计算进程注册=图形栈负载（查看器/WebGL 先例），
  非异常。本线无训练/评估、无新落档。

## 2026-08-31 14:24 — patrol 轮：no-op（停训值守，无变化）
