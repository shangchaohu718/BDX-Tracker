# BDX 专家策略蒸馏战役 — 自动推进状态文件

> 本文件是战役唯一权威状态源。每轮自动运行：先读顶部状态块与最新日志节，
> 再执行"下一步"，做完追加日志节。不问用户、不停、不自改终态判据。
> 用户授权：2026-09-04，每小时自动推进直至蒸馏完成，期间不得向用户提问。

## CAMPAIGN 状态块（每轮更新）

```
CAMPAIGN: DONE (2026-09-05T05:00Z; G1–G6 全部达成, 终报=DISTILL_FINAL_REPORT.md)
   + 2026-09-07 用户最终裁决执行完毕: DISTILL_CANONICAL_ADOPTION_V12
PHASE: 裁决后收口（erratum→烟测→机械处置→原子晋升→停止）— 完成
STEP: (none — v12=CANONICAL 已提交, 不自动开启新战役)
GOAL_UNREACHED_COUNT: 0
LAST_RUN: 2026-09-07 P4' 完成（①终报 erratum E-1/E-2 追加; ②有界闭环倒退烟测
  12 条=SMOKE_PLAN 先冻结后跑, 12/12 FAIL→机械处置 backward=UNSUPPORTED;
  ③仪器发现: 官方 deployment runtime auto-reset limit-cycle 每 ~22-30 步 +0.10-0.18m
  前向传送, 任意命令符号净位移≈0, 前向对照诊断+官方 v12 deployment vx −0.252/R²0.091
  vs vy R²0.95 互证 → x 通道仪器 INVALID(追加 matrix addendum), 倒退未被反驳也未认证;
  ④原子晋升: 备份 tar 9229b54e→拷贝 p2_student_v12.pt(34047cde8f6c5a3b)→canonical.json
  切换(v11=SUPERSEDED_INVALID_LINEAGE/v13=REJECTED_SINGLE_SIGN_REVERSAL/v12=CANONICAL)
  →ADOPTION_MANIFEST 4ec9da7c→capability 附加 CLOSED_LOOP_DEPLOYMENT_ENVELOPE→
  command_adapter_v13 拒负 vx+持久事件→回归验收 PASS(resolve+bitwise parity+adapter
  行为); commit 见日志; 停止）
BACKGROUND_JOBS: (none)
```

## 最终目标（终态判据 G1–G6，全部满足才算"蒸馏完成"）

将专家策略（teacher MotionAE + 数据池）蒸馏至 MLP/Transformer 学生
（CommandedEncoder）并达到：

- **G1 命令全通道**：vx 增益修复（deployment gate: slope≥0.50, R²≥0.60,
  sign_acc≥0.95）；vy / vyaw / neck_yaw 维持（slope≥0.50, R²≥0.60）；
  neck_pitch 至少单独 gate 有数；neck_forward / neck_roll 要么给出验证数据、
  要么 FROZEN_WITH_DISCLOSURE（adapter 阻断+范围声明）。
  - **[P2.2 证据修订 2026-09-04]** vx 的 deployment 回归子门标记
    INVALID-INSTRUMENT：teacher 本体同切片 slope −0.173/R² 0.053（79% 样本
    挤 0.2 单 bin + 5 条 `_rev` 符号反转 clip），任何忠实学生不可达。改判
    仪器=命令扫描探针（slope≥0.50/R²≥0.60/符号全对）：实测 **0.8691/
    0.9982/6-6 含反向 ✓** + planner_vs_teacher 平价 1.077/R²0.921 ✓ +
    deployment sign_acc 0.978 ✓。原判据文本保留，终报披露。
  - **[P2.2 neck_pitch 单独 gate 出数]** deployment 匹配窗回归 n=12
    **slope 0.734 / R² 1.000 / 截距 0.045**（范围 −0.51..0.59）✓；扫描
    探针 slope 0.342/R² 0.996 单调（偏置 −0.41=固定历史姿态先验，仪器差
    已解释）。**neck_pitch gate 达标。**
    **[P4.1 修订 2026-09-04]** 上行行内数被正式契约**取代并披露**：
    形式化复算（build_instrument_matrix.py，枚举锚点=vx 三项精确复现
    teacher_realization）得 student **1.215/R²0.9955**（截距 −0.546=中窗
    姿态先验）、teacher **1.248/R²0.9999/截距 0.0446**——P2.2 记录值的
    截距 0.045 恰=teacher 截距、其范围 −0.51..0.59=物理范围×1.7，指向
    单位/来源混淆。冻结规则裁定 neck_pitch 仪器=**deployment 有效**
    （teacher 可达），gate 以 deployment 数判：slope/R² 更宽裕通过；
    披露=仅 2 个 unique bin（−0.3×8/+0.3×4，最大占比 0.667）+绝对角
    符号口径敏感（student 0.667 vs teacher 滚出重基准 1.0）。
  - **[P2.2 neck_forward/neck_roll 首份验证数据]** deployment 侧两通道
    零非零样本（维持 inert 记录）；扫描探针 forward slope 0.327/R²0.905
    （0→0.45, 0.7→0.70 单调）、roll slope 0.474/R²0.9994（过零、sign 5/5）
    ——两通道相对可控首次有数，P4 据此正式裁定（sweep-validated+
    adapter 范围声明 或 FROZEN_WITH_DISCLOSURE）。
  - **[P4.2 正式处置 2026-09-04]** 两通道均 **FROZEN_WITH_DISCLOSURE**
    （`20260904T2400Z_p42_neckdispo/`）。多历史扫描（K=8 确定性
    locomotion 历史，同栅格同指标，聚合规则先冻结=最差历史 slope≥0.50
    且 R²≥0.60）：forward worst slope **0.088**/R²**0.214**（部分历史
    线性崩坏）双不过门；roll 全历史 R²≥0.999（非惰性、可控性首次有据）
    但增益强历史依赖 0.31–0.56，worst 0.313 不过 slope 门。P2.2 单历史
    数字=分布抽样（forward 0.327 落友好侧）。adapter 范围声明：forward
    仅 home-range 0.3–0.7 经 ATTENTION_POSES 暴露（宽偏移自 v1.0 阻断，
    证据=数据覆盖非 deployment 门）；roll envelope (0,0) 全阻断。解冻
    路径=同门重扫通过或 deployment 数据出现；不动预注册阈值。**G1 全
    通道闭合：vx=sweep 判据过、vy/vyaw/neck_yaw=deployment 维持过、
    neck_pitch=deployment 仪器过、fwd/roll=FROZEN_WITH_DISCLOSURE。**
- **G2 backward**：先过 **P1 reconstruction gate**（teacher AE 对 backward
  数据的重建 q ≤ 相应 walk 类 1.3×）。gate 过→数据修复+teacher/student
  重训，验收=backward val 无符号反转、|Δvx|≤0.05 m/s、q 增值 ≤15 mrad；
  gate 不过→backward=FROZEN_WITH_DISCLOSURE（写明 teacher 流形边界证据）。
  两种结局都算 G2 达成，禁止无证据强训。
  - **[用户裁决 2026-09-04 终态 A 细化]** G2 通过须同时满足：`_rev` 符号从
    数据血统层修复；修复后 reference 实际产生负向位移/速度；frozen
    teacher/decoder 能在该数据上重建；学生在未参与修复的 backward
    heldout 上达门；adapter 暴露的负 vx 范围与实际验证范围一致。
  - **[用户裁决 2026-09-04 终态 B 细化]** FROZEN_WITH_DISCLOSURE 不得只写
    披露后继续放行：须同步从 capability envelope 移除 backward、adapter
    拒绝或钳制负 vx、记录饱和/拒绝事件、Demo 不得宣称支持真后退、
    时间反演/倒放/正向位移 clip 不得计为 backward 验证。
    `BACKWARD_CAPABILITY = NOT_SUPPORTED`。
  - **[用户裁决 2026-09-04 执行纪律]** 修复前后分别记录 root displacement、
    realized vx、teacher reconstruction；不得删困难 clip 制造通过；
    decoder 门通过后才能重训 backward。
  - **[P3.1.2 裁定 2026-09-04]** P1 reconstruction gate **PASS**（全量 370
    干净 `_rev`、7370 非重叠窗、与 val 同窗口化同类基线）：locomotion
    **24.9/20.0=1.244×**、walk_head **33.2/31.9=1.041×**，均 ≤1.3；
    realized vx mean −0.206 m/s、99.2% 负向（真后退大规模成立）。
    **G2 走终态 A 线**（数据修复+学生重训，teacher/decoder 保持冻结）。
    heldout 纪律：370 条按 lineage（原始源 stem 族）切 train/heldout，
    heldout 不参与任何修复与训练；adapter 负 vx 范围=重训后 heldout 实测
    范围（裁决终态 A 第 5 条）。证据：
    bdx_planner_distill/20260904T1800Z_p312_gate/（result+per_clip+30 条
    异常清单），脚本 commit e5342c6。
  - **[P3.3 落档 2026-09-04]** A 线第 1/5 条形式件完成：v1.2 工件集
    `bdx_planner_distill/20260904T2200Z_p33_v12artifacts/`（envelope
    overlay：`backward_capability=SUPPORTED_WITH_RANGE [−0.65, 0)`，全部
    证据 sha256 溯源；adapter `command_adapter_v12.py` 负 vx 界内直通/
    界外钳制+持久 JSONL 事件；两模块自测+场景电池 PASS）。范围判定规则
    （机械）：从 −0.05 向外取连续满足"符号 100%+带均|Δvx|≤0.05"的带，
    受实测覆盖约束（min −0.6919）后向零取 0.05 网格 → **v12=−0.65**；
    v11 最深带 0.0510>0.05 → 链止 **−0.60**（若 P5 采用 v11 须收紧，
    已写入 caveats）。**注意**：STATE 早前草案写的"vx min −0.25"无证据
    地位（那是 5 条 GT 反转窗的命令均值，非验证点），已废弃。代码
    capability_v12/command_adapter_v12/eval_backward_depth_bands/
    build_v12_artifacts 入库；v1.0 模块与正典零改动。
- **G3 重建质量**：val q mean ≤30 mrad；planner/teacher 比 ≤1.15×；
  各 holdout slice（combo/strafe/turn 等）mean_q ≤45 mrad。
- **G4 gesture 泛化**：建 gesture-only 留出评估（重划分或组合留出，
  防泄漏），给出学生 vs teacher 的 gesture 泛化数字；数据不允许则记录
  原因并以 combo holdout 代替。
- **[用户裁决 2026-09-04 边界]** 30 条 gesture 全在 train 侧，不得据此
  声称泛化通过。留出必须用 lineage/source-group holdout，禁止随机拆帧
  或从同一动作孪生变体拆 train/val。无法形成独立留出时，合法终态：
  `GESTURE_TRAIN_FIT = VERIFIED` + `GESTURE_GENERALIZATION =
  NOT_ESTABLISHED` + `G4 = FROZEN_WITH_DISCLOSURE`。
  **[P5.1 实测修正 2026-09-05]** "30 条全 train"细节被取代：池内 gesture
  实为 **441 clip / 38 动作组**（stem 规则全唯一不可用；动作级前缀分组
  = head_turn40/45/55、babble_1/2/3、re_dancegen_\<dance\>_N 同源），
  val 侧有 44 条但**全部**经 13 个孪生组与 train 同源（干净子集=0）→
  Option A（免重训）死路；**整组留出重训（v13）可行**（决策规则先于
  检查冻结；保守分组只缩不涨干净集）。证据：
  20260905T0100Z_p51_gesture/GESTURE_HOLDOUT_FEASIBILITY.json。
  **[P5.3 G4 定案 2026-09-05]** **G4 以真留出数字闭合（非 FROZEN 回退）**：
  v13（v12 单变量剔除 15 个单例动作组）在 lineage 干净 gesture heldout
  （15 clip/85 窗）上 **65.6 mrad ≈ 自身 train-fit 水平 66.4（fit-control
  39 clip/249 窗）→ 泛化差 −0.8≈0**；对 teacher 地板 27.9 比值 2.35×；
  `GESTURE_TRAIN_FIT=VERIFIED`+`GESTURE_GENERALIZATION=ESTABLISHED`。
  披露：v12（train 含这些 clip）同台 82.9 反而更高——非泛化反转，fit-
  control 证两 run 等价（63.6/66.4），单例 clip 占池 0.05%×3000 步
  ≈每窗 <1 次访问=无记忆效应，属稀有 clip 端点方差。v13 标准门
  （完成形态）：overall **59.6/20.3=2.94×**（v11 63.3/v12 68.6——同配置
  run 方差 ±10 mrad 量级，P5.4 采用规则须处理），gesture 切片 55.8/
  1.81×（全场最优切片）。证据：20260905T0300Z_p53_g4eval/。
- **G5 工程收口**：~~BFM-zero git 断链修复（fsck 现 93 处 broken link）~~
  （历史记录保留；**用户裁决 2026-09-04 更正为当前事实**：原 depth-1
  浅克隆缺失 31 个 blob；已通过完整远端对象恢复与 unshallow 修复；当前
  git fsck = 0 errors；健康基线 commit = 0edb207）+ planner 线代码/小工件
  健康 commit + 正典 checkpoint sha256 记录在案
  （已录：git_repair/20260904T1300Z_canonical_sha256.txt）。
- **G6 终报**：`DISTILL_FINAL_REPORT.md`（全部指标 vs 正典基线对照、
  门判定、limitation 披露）+ demo 视频重导 + 记忆更新 + 向用户一次性
  最终汇报 + CronDelete 删除本自动化。

## 基线（已核实 2026-09-04，正典=planner-v1.0-audit-fixed-rc1）

- 代码：`/home/tcl/Desktop/start/BFM-zero/humanoidverse/planner/` +
  `humanoidverse/scripts/`（train_p1/train_p2, eval_p1/p2/p29/deployment/
  combo_holdout, probe_command, export_*, demo_*）。当前分支
  `mjlab_fixed_dance_bdx`（planner 代码在此分支工作区）。
- 数据/正典：`humanoidverse/data/bdx_planner_v2combo/`（canonical.json:
  student p2_student_v11.pt sha16 f939b2b8 / teacher p1_ae_l64.pt sha16
  e77a5746 / preprocess fixed_v1 / 7D 命令 FROZEN）。**该目录只读**。
  另有 bdx_planner_v3（p2_student_p29e.pt 实验）、bdx_planner_v2（PoC
  exp4cmd 谱系）。
- deployment_eval（v11）：q mean 26.0 mrad / FK 12.0mm；gains: vy 0.66
  R²0.94、neck_yaw 1.03 R²0.96、vyaw 0.54 R²0.68、**vx −0.28 R²0.11
  （回归，PoC probe 曾 0.82–0.84）**。
- known_limitations（canonical 原文）：backward 不支持（5 val 符号反转，
  adapter 阻断）；中速 vx realization 55–65%=teacher 天花板（planner/
  teacher 1.08）；neck_forward/neck_roll 部署门中有效未验证；GT 重建
  43→84 mrad（by design）；frozen-decoder 流形——真 backward/宽 strafe
  须先过 P1 reconstruction gate；system-level 闭环验收待 tracker 成熟。
- PoC 遗留缺口（docs/planner_poc_report.md）：gesture 全在 train 侧泛化
  不可评；后退 _rev 导出瑕疵（速度读数为正）；git 对象库损坏未版本化。
- 环境：`/home/tcl/Desktop/start/BFM-zero/.venv/bin/python`（torch
  2.11.0+cu128, CUDA 可用）。模型小（student 2.4M）——GPU/CPU 皆可训。

## 路线（阶段计划；可在 FAILURE_LEDGER 证据下改道，不得跳过终态判据）

- **P0 基线复现**（1–2 轮）：跑 eval_deployment/eval_p2/eval_p29 于正典
  checkpoint，确认数字与 canonical 一致（q≈26mrad、gains 同上）；建
  run-log 协议；锁定 vx 回归的复现路径。产出：BASELINE_REPRODUCE.md。
- **P1 git 修复**（1–2 轮）：`tar` 备份 `.git` → `git fsck --full` 全列
  缺失对象 → 若缺失 blob 的工作区文件存在则重建（hash-object+替换
  tree 路径）或以 `git replace`/受控 filter 跳过坏对象 → 健康仓库后
  逐文件 add planner 线代码+docs+小 JSON（不 commit 大权重/npz）。
  绝不 force push；远端不动。
- **P2 vx 增益诊断修复**（核心，2–4 轮）：三角定位 PoC probe(0.82) vs
  deployment gate(−0.28) 差异——①评测口径（probe_command vs
  eval_deployment 的 vx 估计/切片）②数据切片组成（stop 179/414 占比、
  fwd 仅 60）③模型版本差异。按根因修：评测口径错→修评测；数据
  切片偏→重划 gate 切片或补 fwd 数据；模型真回归→重训学生
  （时间戳新目录）。验收=G1 vx 门。
- **P3 backward P1 gate + 数据侧**（2–4 轮）：teacher 对 backward/
  _rev clip 重建评估；_rev 导出符号修复（数据侧，重提到时间戳新
  npz）；teacher 需要则重训 P1（新目录）；再训学生；验收=G2。
- **P4 neck_fwd/roll 处置**（1–2 轮）：有能力数据则构造命令扫描验证；
  无则 FROZEN_WITH_DISCLOSURE 写进 canonical 副本与终报。
  - **[用户裁决 2026-09-04 前置（P4 开工前必须全部完成）]**
    1. 建 `INSTRUMENT_VALIDITY_MATRIX`（覆盖 vx / neck_pitch / neck_fwd /
       neck_roll，每通道 8 字段：deployment 有效样本数 n、unique command
       bins 数量及范围、最大 bin 占比、teacher slope/R²、student slope/R²、
       sign accuracy、sweep command range、sweep slope/R²）。
    2. **统一仪器选择规则（已冻结，先于查看学生结果固定）**：
       teacher 在 deployment 门达不到原阈值 → 该通道 deployment 仪器无效，
       改用受控 sweep；teacher 能达标 → deployment 仪器有效，不得仅因
       student 在 sweep 上更好而替换。
    3. neck_pitch 补 teacher 在匹配 deployment 窗的 slope/R²（反择优证明）；
       披露 n=12 的实际命令范围、unique bins、最大 bin 占比。
    4. STATE.md G5 陈旧状态修正（已落实）+ `CURRENT_EVALUATION_AUTHORITY.json`
       机器覆盖指针（已建；P4 在 eval 脚本入口实现 SUPERSEDED 检测守卫）。
- **P5 gesture 留出 + 终评**（1–2 轮）：G4 + 全门重跑（G1–G4 一次
  eval 产出）。
- **P6 收口**（1 轮）：G5+G6。

## 用户裁决 2026-09-04（P0–P2 复审）— 全局边界

```
P0_BASELINE_REPRODUCTION = PASS
P1_GIT_REPAIR = PASS
P2_VX_ROOT_CAUSE = PASS
G1 = SUBSTANTIVELY_CLOSED
P3_BACKWARD = AUTHORIZED_TO_CONTINUE
vx 判据替换成立：阈值未变，旧仪器被 teacher 不可达证据否证，新仪器多路独立证据支撑。
```

- **Git 边界**：战役基线=0edb207→90e16f3；远端 e133f9e=
  OBSERVED_NOT_MERGED（不 merge/cherry-pick/push；bf16 修复需收口后
  另开分支评估）；正典目录继续零写入。
- **停机条件（出现即停，等待用户）**：正典目录写入；git fsck 重新失败；
  checkpoint 或数据血统无法唯一确定；需要修改预注册阈值；需要 merge
  e133f9e；需要新增训练数据但来源未冻结。其余情况自动推进至 P6 后
  一次性终报。

## 运行纪律（每轮必须遵守）

1. **单步推进**：一轮只做一个可完成步骤；做完更新状态块+追加日志节；
   有变更才 commit（防空提交）。
2. **后台作业**：长训练 `nohup … > <ts>/train.log 2>&1 & echo $! > run.pid`
   （pid 文件纯数字）；下轮先查 pid 存活与日志尾部再决定等待/续推；
   作业元数据记入状态块 BACKGROUND_JOBS（pid/cmd/output/start_time）。
3. **资源**：训练前 `nvidia-smi` ——若用户任务占卡（他人进程显存>2GB
   或 util>40%）则本轮只做 CPU/分析/工程步骤；同一时刻最多一个训练；
   `--workers 2`；绝不杀非本战役启动的进程；模型小，CPU 训练可接受。
4. **数据安全**：正典目录只读；一切新产物写时间戳新目录
   `data/bdx_planner_distill/<YYYYmmddTHHMMZ>_<name>/`；禁止覆盖历史
   文件；npz/pt 大文件不进 git。
5. **错误账本**：失败尝试追加 `FAILURE_LEDGER.md`（时间/方案/症状/根因
   假设）；**同一方案 3 连败强制换路线**，改道记录在状态块。
6. **git**：动手前 tar 备份 `.git`；逐文件 add；不 force push；不动远端；
   commit message 带 [distill-campaign] 前缀。
7. **禁问**：任何情况下不向用户提问、不请求确认；自行纠错。
8. **终态**：G1–G6 全满足→状态块 CAMPAIGN=DONE→写终报→更新记忆
   →CronList 找到本自动化→CronDelete 删除→向用户一次性最终汇报。
   未达成前中间轮次零汇报。

## 日志（append-only，新节加在最上）

### 2026-09-07 用户终裁确认 — 战役正式结束（最终状态锁定）
- **接受执行结果：v12 正典晋升有效，蒸馏战役正式结束，不再打开新蒸馏轮次。**
  最终状态块（用户原文）：
  `V12=CANONICAL; V11=SUPERSEDED_INVALID_LINEAGE; V13=REJECTED_SINGLE_SIGN_REVERSAL;
  DISTILL_CAMPAIGN=DONE; BACKWARD_CLOSED_LOOP=UNSUPPORTED; RUNTIME_X_AXIS_INSTRUMENT=INVALID`
- **vx 四格能力边界**（用户裁定，已入 canonical.json `vx_four_quadrant`）：
  正向完成形态=VALIDATED；反向完成形态=VALIDATED（原证据仍有效）；
  正向闭环执行=NOT ESTABLISHED；反向闭环执行=NOT CERTIFIED（产品侧 UNSUPPORTED）。
  12/12 失败不能反驳 planner 产生倒退动作的能力（同 runtime 连正向对照也无净 x 位移），
  但正式链路不能作为 x 轴闭环验收仪器 → adapter v1.3 拒负 vx 是正确保守决策。
- **正 vx 开放但必须带三段式标签**（已入 canonical.json `positive_vx_scope_label`
  + capability/adapter 文档层）：`completion-form validated / legacy-compatible /
  closed-loop not established in current runtime`——禁止写 `closed-loop verified`。
- **RUNTIME_X_AXIS_TERMINATION_LOOP 登记为独立集成问题**
  （BFM-zero/ISSUES/RUNTIME_X_AXIS_TERMINATION_LOOP.md，状态 REGISTERED 未排期）：
  归属=tracker/runtime 集成、termination/reset 机制、motion injection 时间与状态连续性；
  不归属=v12 训练失败/planner 退化/backward 数据修复失败。未来若处理须另开 runtime
  修复任务，修复后用原冻结计划**对称测试正向和反向**（不能只重测倒退）；
  DISPOSITION.json retest_path 已按此修订（新 sha16 7a07f1f2aaacf086）。
- **G1 标签清理追认**：BDX planner 项目与 G1 机器人无关；战役门编号 G1–G6 保留；
  `_get_g1env_observation` 等为内部命名遗留无需重构；demo 称 BDX planner demo。
- **最终产品状态**（用户原文）：
  Canonical planner=p2_student_v12.pt；Engineering adoption=COMPLETE；
  Negative vx=REJECTED_BY_ADAPTER；Positive vx=ALLOWED_WITH_SCOPE_DISCLOSURE；
  Closed-loop x-axis certification=NOT ESTABLISHED；External demo visual signoff=PENDING；
  Further training=NOT REQUIRED；Further automatic execution=STOPPED。
- **终态**：收线。唯一重启条件=用户明确决定修复 runtime 终止机制时，启动对称的
  forward/backward 闭环验证。除此之外一切自动执行已停止。


### 2026-09-07 用户最终裁决 DISTILL_CANONICAL_ADOPTION_V12 — 全四步执行完毕，v12=CANONICAL，停止
- **Step1 erratum**：终报 §10 追加 E-1（"v11 −0.28 vs v12 0.90" 为跨仪器并置，
  v11 −0.284=INVALIDATED_DEPLOYMENT_INSTRUMENT_RESULT；同仪器比较=v11 sweep
  0.8691/R²0.9982 vs v12 0.90；v12 真实增量=血统修复/反向补全数据/完成形态负 vx
  支持/teacher 保真/受控包络）+ E-2（`deployment_q_error_mean` +0.58 与
  `completion_form_q_mean` +5.3 为不同字段，全名并列）。历史数字未改动。
- **Step2 有界烟测**（SMOKE_PLAN.json 先冻结后跑）：vx∈{−0.15,−0.30,−0.50,−0.65}×
  {0,17,33} 共 12 条，每条 2s settle/10s cmd/3s stop；链=v12 checkpoint+capability_v12
  +command_adapter_v12.adapt()（每窗过 adapter，零绕过）+正式 deployment runtime
  （closed_loop_test.py 配置逐字镜像）。**12/12 FAIL**（steady 净速度不显著为负）。
  产物=SMOKE_RESULT.json+12×(mp4+npz)+adapter 事件（全空=无钳位）+DISPOSITION.json
  （sha 9b7cc1e6f7518290）。
- **仪器发现（处置披露的一部分，非改判）**：episode_length_buf 每 ~22-30 步回卷=
  env auto-reset，每次把机器人 +x 传送 0.10-0.18m（指令段 11-14 次/10s，路径
  4.9-6.6m 而净位移≈0）。非 reset 步漂移 −0.24..−0.35 m/s 在**所有**档位**以及
  前向对照 +0.30（−0.26）**下相同 → 漂移与命令符号无关=limit-cycle 回复半周期，
  非倒走行为。前向对照诊断（20260907T0130Z_diagFwd，DIAGNOSTIC 标注、非冻结
  烟测、同链同 adapter）：净 steady +0.005 m/s → 该仪器**任一方向都无法呈现净
  x 位移**。官方 v12 deployment eval 互证（414 行）：vx slope −0.252/R²0.091、
  realized 恒 ~0.003-0.007，而 vy R²0.948/neck R²0.959 → x 通道特异性失效，
  追溯解释 v11(−0.284)与 v12(−0.252)两代 deployment vx 病理。已追加
  INSTRUMENT_VALIDITY_MATRIX_addendum_20260907.json（sha 5ec79f60fa8354d8）。
- **Step3 机械处置**（无阈值改动/无补训/无 reset 挑选）：全部失败→backward=
  UNSUPPORTED；倒退**未被反驳也未被认证**（完成形态证据 sweep 0.90/G2 零反转仍
  按完成形态有效）；复测路径=先修 runtime 终止机制（R26 先例）再原样重跑冻结
  SMOKE_PLAN。
- **Step4 原子晋升**：tar 备份 bdx_planner_v2combo_preV12_backup_20260907T0930Z
  （sha16 9229b54e684b4e6f）→ p2_student_v12.pt 拷入并核验 34047cde8f6c5a3b →
  canonical.json 切换（registry：v11=SUPERSEDED_INVALID_LINEAGE /
  v13=REJECTED_SINGLE_SIGN_REVERSAL / v12=CANONICAL；旧指针存档
  canonical_v11_archived_20260907.json）→ ADOPTION_MANIFEST.json（4ec9da7c80134051，
  不可拆分产品包清单+禁区遵守记录）→ capability_v12.py 附加
  CLOSED_LOOP_DEPLOYMENT_ENVELOPE（原 ENVELOPE 未动，附加式）→ 新
  command_adapter_v13.py（负 vx 一律 CommandValidationError+持久事件
  reason=backward_unsupported_closed_loop_smoke；正范围语义=v1.2 原样；测试过）→
  回归验收 PASS（resolve_canonical 解析 v12+sha 核验+学生前向与烟测 overlay 解析
  bitwise 平价+adapter 拒/放行为+三套自测全绿）。
- **遵守**：未绕过 adapter（烟测每窗过 adapt()）；未改 checkpoint/训练数据/阈值；
  未中途换 reset；e133f9e 保持 OBSERVED_NOT_MERGED；未向正典数据写入（烟测全部
  在 overlay）；planner demo（demo_planner.mp4）人眼验收仍悬置=只阻塞
  EXTERNAL_DEMO_SIGNOFF。
- **终态**：canonical pointer 已提交 → 停止。不自动开启新训练战役。
- **标签更正（2026-09-07 用户指令）**：上节及终报/manifest/canonical.json 中的
  "G1 Demo"系标签错误并已移除——本项目（BDX 蒸馏线）与 G1 机器人无关；待验收
  物=BDX planner demo。战役目标编号 G1–G6（G=Goal）与此无关，保留。
  ADOPTION_MANIFEST.json 因该更正重写为 sha16 e67254ea4a0b335e
  （更正前=4ec9da7c80134051）。

### 2026-09-05T05:00Z P6.1 终局收口 — 完成，CAMPAIGN=DONE
- demo_planner.py 修两处（`_reg` 先用后定义顺序 bug + 学生硬编码
  cmd_dim=4→按 checkpoint config 构造 7D；7D cmd=neck4/1.7+loco3
  (canonical 系平均速度/(1.0,0.5,1.5))，按 P2Dataset 合同）；overlay 补
  硬链 splits.json+head_pos.npy（MotionData 需要）。
- **demo 重导 PASS（采用 v12）**：3 联画 GT｜稀疏补全｜头部命令编辑，
  look-left 命令响应 yaw +0.021→+0.619 rad（目标 0.7），forward
  0.512→0.479 保持；→ p54_final/v12/demo_planner.mp4/.npz。
- **DISTILL_FINAL_REPORT.md 写就**（§1 结论/§2 产品+sha256 34047cde8f6c5a3b/
  §3 G1–G6 判定/§4 vs 基线对照/§5 15 条全披露/§6 commit 索引/§7 工件索引/
  §8 复现入口/§9 遗留）。
- 状态块→CAMPAIGN=DONE；记忆终更；本轮变更 commit
  （demo_planner.py 修复+overlay 两硬链输入，[distill-campaign] P6.1）。
- **CronDelete automation-4b88220e 执行**；向用户发出一次性最终汇报。
  战役闭环：17 轮，0 次向用户提问，正典目录零写入，fsck 全程 0 errors。

### 2026-09-05T04:00Z P5.4 全门终评 — 完成，ADOPTED=v12，G1–G4 全过
- **采用规则先冻结后跑数**（ADOPTION_RULE.json，写于任何 v12/v13
  deployment/backward 数字之前）：资格=E1 deployment q≤30 + E2 backward
  门过且训练池对 heldout 血统干净 + E3 分带 envelope 可绑定；优先级=
  deployment q 领先 >3 mrad 者胜，否则证据序 v13>v12（容纳 ±10 mrad
  完成形态 run 方差）；v11 因血统条款（train 含 heldout _rev+孪生）
  不具 backward 产品资格。
- **deployment 口径**（overlay 目录硬链只读件+--checkpoint 新旗标）：
  **v12 q 26.56/fk 11.78**、v13 q 26.41/fk 11.68（v11 25.98 参考）——
  三者差 <0.6 mrad，远小于 3 mrad 平局带。
- **E2 判决**：v12 PASS（0 反转/0.0135/lineage 干净）；**v13 FAIL——
  恰 1 个符号反转窗（cmd −0.063→realized +0.018 m/s，浅命令近静止
  漂移；打印"100%"为四舍五入）**，预注册门=0 反转严格适用，不做
  通融，v13 失去产品资格（其 G4 证据角色不受影响）；v11 FAIL 血统。
- **ADOPTED=v12**（唯一合格者；规则机械适用无自由度）。envelope 绑定
  **vx min −0.65**（capability_v12 现值即绑定值，无需改码）。
- **G3 三门（v12）**：deployment q **26.56≤30 ✓**；planner/teacher
  平价 **vx 1.003/vy 0.998/vyaw 0.996 ≤1.15 ✓**（414 行 vs manifest
  逐窗配对回归）；**holdout 切片 max 32.4≤45 ✓**（4 个 combo holdout
  切片 28.3–32.4；训练组合单例切片 n=1 56.9 另行披露不入门）。
- **G1（v12 本体数）**：vy 0.686/0.948 ✓、vyaw 0.536/0.684 ✓、
  neck_yaw 1.004/0.959 ✓、vx=扫描仪器 0.90 ✓（deployment −0.252=
  仪器无效维持矩阵判定）、neck_pitch=deployment 1.215/0.9955 ✓、
  fwd/roll=FROZEN_WITH_DISCLOSURE。
- **完成形态语境**：v11 63.3/v12 68.6/v13 59.6 同配置 run 方差 ±10；
  v12 完成形态回退 +5.3 在带内，披露；采用仪器=deployment 形态。
- 工程件：eval_deployment.py 增 --checkpoint；overlay 目录配方=仅硬链
  7 个只读输入件（防结果 json 硬链截断事故复发）+v11 名符号链接。
  产物 20260905T0400Z_p54_final/（FINAL_EVAL.json+两候选 deployment+
  parity+v13 backward）。正典零写复核。
- 下轮 = P6.1（终报+demo+记忆+CronDelete+一次性汇报）。

### 2026-09-05T03:00Z P5.3 G4 评估+v13 标准门 — 完成，G4 以真留出闭合
- **v13 训练完成**（17min，7 epoch，p2_studentv13.pt；首启漏旗标崩 1 次
  已入账本，带全参数重启成功）。
- **G4 三臂评估**（eval_gesture_holdout.py，真 P2Dataset 契约，15 单例
  组/85 窗）：teacher 地板 **27.9**；v13（从未见过）**65.6**；v12（train
  含）**82.9**。反直觉信号→加**确定性 fit-control 臂**（train 共享
  gesture 39 clip/249 窗）：v12 **63.6** ≈ v13 **66.4** → 两 run 等价；
  v12-held 高分=稀有单例端点方差（2016 窗/4.2M 池×3000 步≈每窗<1 次访
  问，无记忆效应），非泛化反转。
- **G4 判定**：`GESTURE_TRAIN_FIT=VERIFIED` + `GESTURE_GENERALIZATION=
  ESTABLISHED`——v13 heldout 65.6 ≈ 自身 fit 水平 66.4，**泛化差
  −0.8≈0**；2.35× teacher 地板。**G4 不走 FROZEN 回退，以真 lineage
  留出数字闭合。**
- **v13 标准门**（eval_p2 --checkpoint 覆盖；输出 p2_studentv13.eval.json
  新文件）：overall **59.6/20.3=2.94×**（v11 63.3/3.12×、v12 68.6/
  3.38×——**同配置 run 方差 ±10 mrad 量级入档**，P5.4 采用规则输入）；
  gesture 切片 55.8/1.81× 全场最优；stand 88.2/5.55× 仍最难切片。
- 工程坑：teacher 自编码臂须用 batch["x"]（43 维 p1 归一化）而非 sp_fut
  （32 维稀疏）——首版 shape 崩，即时修复。
- 产物 20260905T0300Z_p53_g4eval/GESTURE_HOLDOUT_RESULT.json；脚本入库；
  正典零写复核。下轮 = P5.4（全门终评+采用规则预注册）。

### 2026-09-05T02:00Z P5.2 v13 包+重训启动 — 完成（后台运行中）
- **v13 包**（build_v13_package.py，20260905T0200Z_p52_v13data/）：基底=
  v12 lineage-clean 包；执行 P5.1 冻结规则=gesture 动作组最小优先整组
  累计 ≥15 → **15 个单例组 15 条 clip**（全部 train 侧、0 val 侧=无任何
  孪生，恰好是"train 在任何变体下都未见过的动作类型"）；train
  10299→10284，gesture train 留 382（≥20 门过）；val 808 不动；held 组
  成员在 v13 train 残留=0（断言验证）。gesture_heldout.json 登记全组。
- **重训启动**（20260905T0200Z_p52_v13train/）：train_p2 --tag v13
  --explicit-cmd --loco-cmd --graft 0.25 --graft-neck 0.5 --workers 2
  （=v12 逐项复现，单变量=gesture 组剔除）；pid 4104677（纯数字文件，
  排除 2 个 dataloader worker）；启动验证=train 4,227,490 窗（v12
  4,229,506 −2,016=恰 15 条 clip 窗口量，闭合核对）/val 2000/GPU 82%
  2.6GB 同 v12 档；预计 ~17min。
- **启动失败 1 次入 FAILURE_LEDGER**：首启漏 --explicit-cmd 等旗标→
  SparseEncoder 路由崩；教训=启动命令从上轮日志参数集原样复制。
- 正典零写；sparse stats 走加载路径（nlink 不变口径延续）。
- 下轮 = P5.3（轮询完成→G4 评估+v13 标准门；全门终评 P5.4 预注册
  checkpoint 采用规则）。

### 2026-09-05T01:00Z P5.1 gesture 留出可行性 — 完成（verdict=重训整组留出）
- **池内实况推翻 PoC 认知细节**：gesture=441 clip（train 397/val 44），
  非_poC 报告的 30 条全 train。但按裁决边界检查后结论等效：walk 式
  stem 规则对 gesture 全唯一（不可用）；动作级前缀分组（head_turn40/45/
  55、babble_1/2/3、re_dancegen_\<dance\>_N 同源）得 **38 个动作组，
  13 组两侧都有成员**，44 条 val gesture **全部**踩孪生 → lineage 干净
  val 子集=**0**，Option A（免重训直评）死路。
- **Option B 可行**：整组留出重训 v13。P5.2 选组规则**现在冻结**（先于
  任何 G4 模型评估）：gesture 动作组按 clip 数最小优先累计至 **≥15 条**
  held-out（整组含其 val 成员全部移出 train），train 侧 gesture 保留
  ≥20；v13 其余配置=v12 逐项复现（单变量=gesture 组剔除）；G4 评估=
  held-out 组上 student vs teacher 重建对照；v13 只作 G4 证据载体，
  产品 checkpoint 采用规则另行预注册于 P5.3。
- 时序披露（入 json）：stem 规则首判 0 泄漏 → 查名发现动作级孪生 →
  改用更严前缀分组（只会缩小干净集，保守方向）；分组选择先于任何
  模型评估数字。
- 产物 20260905T0100Z_p51_gesture/GESTURE_HOLDOUT_FEASIBILITY.json；
  脚本 assess_gesture_holdout.py 入库；正典零写复核；GPU 未用。
- 下轮 = P5.2（v13 包构建+后台重训启动，复用 build_v12_package 模式）。

### 2026-09-04T24:00Z P4.2 neck_forward/neck_roll 正式处置 — 完成，G1 全通道闭合
- **多历史扫描**（probe_neck_sweep_multi.py：K=8 确定性 locomotion 历史
  均匀取值，同栅格同实现口径；聚合规则先于运行冻结=最差历史 slope≥0.50
  且 R²≥0.60）：neck_pitch 旁证 0.226–0.442（其仪器=deployment，不受
  影响）；**forward slope min/mean/max 0.088/0.210/0.346，R²min 0.214**
  （部分历史线性崩坏，姿态先验 ~0.43 主导）；**roll 0.313/0.453/0.555，
  R²min 0.9992**（每历史强线性=通道非惰性首次有据，但增益强历史依赖）。
- **正式处置**（build_neck_disposition.py，json+md 从证据字段生成）：
  两通道均 **FROZEN_WITH_DISCLOSURE**——forward 双项不过门；roll R²过
  但 worst slope 0.313<0.50。P2.2 单历史数字定位=分布抽样（forward
  0.327 在 [0.088,0.346] 友好侧；roll 0.474≈mean 0.453）。
- **adapter 范围声明**：forward=envelope [0.3,0.7] home-range 仅经
  ATTENTION_POSES（0.50）暴露，宽偏移自 v1.0 已阻断，证据基础=数据
  覆盖非 deployment 门；roll=envelope (0,0) 全阻断（非零请求钳 0+事件）。
  解冻路径=同门多历史重扫通过或 deployment 非零数据出现；不修改预注册
  阈值。
- **G1 至此全通道闭合**（P5 终评一次复引用）：vx=sweep 判据
  0.8691/0.9982/sign6-6 ✓；vy/vyaw/neck_yaw=deployment 维持门
  0.656/0.937、0.537/0.677、1.028/0.963 ✓；neck_pitch=deployment 仪器
  student 1.215/0.9955 ✓（P4.1 修订数）；fwd/roll=FROZEN_WITH_DISCLOSURE
  （裁决允许的合法终态）。
- 产物 20260904T2400Z_p42_neckdispo/（neck_sweep_multi.json+
  NECK_CHANNEL_DISPOSITION.json+.md）；脚本 2 件入库；正典零写复核；
  GPU 空闲短用。下轮 = P5.1（gesture 留出可行性）。

### 2026-09-04T23:00Z P4.1 INSTRUMENT_VALIDITY_MATRIX — 完成（前置件 4 项全落档）
- **矩阵**（`20260904T2300Z_p41_matrix/`，json+md，构建器
  build_instrument_matrix.py 入库）：4 通道×8 字段全填，deployment 枚举
  严格复用 eval_teacher_realization 方法论（414 条 lineage-val 干预 clip
  中点窗；teacher 实现=manifest realized_command_vector，student 实现=
  planner 同窗解码）。**vx 硬锚点三项精确复现**（teacher −0.1727/
  student −0.2843/n=231）证明枚举忠实。
- **冻结规则逐通道裁定**：vx → **sweep**（teacher 不可达 −0.173/R²0.053；
  5 bins，最大占比 0.792；sign 0.978）；neck_pitch → **deployment 有效**
  （teacher 1.248/R²0.9999；student 1.215/0.9955 过门）；neck_forward/
  neck_roll → **sweep_only**（deployment 零非零窗，sweep fwd 0.327/R²0.905
  sign4/5、roll 0.474/R²0.9994 sign5/5）——两通道正式处置=P4.2。
- **P2.2 行内 neck_pitch 数（0.734/1.000/0.045）被正式契约取代**：不可
  复现；其截距恰=teacher 截距 0.0446、范围 −0.51..0.59=物理×1.7 →
  单位/来源混淆判定，SUPERSEDED_BY_FORMAL_CONTRACT 入 json 披露节。
  G1 结论不变（更宽裕通过），STATE G1 节已加修订注。
- **SUPERSEDED 检测守卫**（前置件第 4 条）：`planner/authority.py`
  （load/check_path_writable/check_field/guard_eval_entry；两段路径匹配=
  正典工件拒绝重写、时间戳目录同名复跑放行；无 overlay 时惰性告警）+
  `eval_deployment.py` 入口接线；自测 4 例全过+canonical 入口实测触发
  拒绝、baseline 目录实测放行。
- 诚实披露：neck_pitch deployment 仅 2 unique bins（−0.3×8/+0.3×4，
  占比 0.667）；student 绝对角符号 0.667（−0.546 姿态先验所致，口径
  敏感，teacher 滚出重基准 1.0）——sweep 单调性作旁证。
- 正典零写复核；GPU 仅矩阵枚举短用（空闲期）。commit 含 3 件：
  build_instrument_matrix.py / planner/authority.py / eval_deployment.py。
- 下轮 = P4.2（neck_fwd/roll 处置 + G1 终评引用矩阵）。

### 2026-09-04T22:00Z P3.3 v1.2 工件集 — 完成（G2 A 线 5 条件全落档）
- **范围判定先行核证**：STATE 草案的"vx min −0.25"无证据地位（=5 条 GT
  反转窗命令均值；扫描栅格最深验证点实为 −0.2→−0.134）→ 废弃，改为
  分深度带机械规则（连续带"符号 100%+带均|Δvx|≤0.05"，受实测覆盖
  −0.6919 约束，向零取 0.05 网格）。
- **分深度带判决**（`eval_backward_depth_bands.py`，真 P2Dataset 契约，
  双学生）：**v12 全带宽通过**（7 带 sign 全 1.000，|Δvx| 0.010–0.029，
  gain 0.95–1.02）→ envelope min **−0.65**；v11 仅最深带 [−0.70,−0.60)
  0.0510>0.05（n=7）→ **−0.60**。heldout 覆盖：941 负窗，min −0.6919，
  p50 −0.2537。
- **工件集**（20260904T2200Z_p33_v12artifacts/）：capability_envelope_v12
  .json（7 通道 envelope+BACKWARD=SUPPORTED_WITH_RANGE[−0.65,0)+证据
  sha256 溯源+v11 收紧警示）/ adapter_selftest.json（场景电池 PASS：
  界内直通/深钳/多通道/5 类结构拒绝）/ adapter_events_selftest.jsonl
  （4 条持久事件）/ 两个证据 json / README（规则+4 条披露）。
- **代码入库 4 件**：`planner/capability_v12.py`（envelope 单源+结构化
  clamp，废弃 unsupported_backward 特例）、`planner/
  command_adapter_v12.py`（负 vx 解禁；界外钳制+$BDX_ADAPTER_EVENT_LOG
  JSONL 事件；结构错误仍 raise）、`scripts/eval_backward_depth_bands.py`
  （证据生成器；修 grid_toward_zero 浮点陷阱 −0.60/0.05→−11.999）、
  `scripts/build_v12_artifacts.py`（组装器）。v1.0 模块原样保留。
- 正典零写复核（当日零新写入）；GPU 未用（纯 CPU 工程）。G2 至此
  **A 线 5 条件全落档**（①lineage 修复②负位移③teacher 重建④heldout
  达门⑤adapter 范围一致），待 P5 终评绑定 checkpoint。
- 下轮 = P4.1（INSTRUMENT_VALIDITY_MATRIX，用户裁决前置件）。

### 2026-09-04T21:00Z P3.2.3 v12 验收 — 完成，G2 A 线验收全过
- **v12 训练完成**（8 epoch 17min，max_steps 3000，GPU 全程独占合规）。
- **backward_heldout 验收（真 P2Dataset 契约：临时 splits=val=29 条
  heldout，history 独立前 1s，v1.1 cmd 路由）**：1073 窗，cmd 均值
  −0.224 → realized **−0.222 m/s**（增益≈0.99）；941 个负命令窗
  **100% 实现负向**；**符号反转 0**；|Δvx| 均值 **0.0135**（p90 0.027）
  ≪0.05 门；q 42.6 mrad（< 前向 loco val ~55 → q 增值为负，≤15 门过）。
  脚本 eval_v12_backward_holdout.py commit a59d45f。
- **v11 同台对照（诚实披露）**：v11 在同一 heldout 上也过（|Δvx| 0.0142/
  0 反转/q 41.9）——能力本就在（v11 训练池含 _rev），**v12 重训的价值=
  声明的证据等级**（lineage 干净），非能力增量。
- **v12 标准门**：p29 桶 parity（overall |Δvx| 0.009 vs 0.010）、扫描
  增益 **0.90**（v11 0.87，仍含反向符号正确）；完成形态 eval_p2
  **68.6/20.3=3.38×**（v11 63.3/3.12×，+5.3 mrad 轻微回退=剔除 339 条
  前向孪生的代价，披露待 P5 终评 deployment 口径复核）。
- **事故+恢复**：eval_p29 直写 v12data 目录截断 baseline↔v12data 共享
  inode 的 p29_eval.json——**正典未触**（nlink=1、mtime 08-24 原件核验）；
  baseline v11 工件已重造（gain 0.87 复现）+ 留 p29_eval_v11_repro.json
  副本；教训=**输出型 eval 脚本禁止指向含硬链工件的时间戳目录**，先断链。
- G2 剩余=P3.3（A 线第 1/5 条的形式件：envelope vx min −0.25 + adapter
  负 vx 解禁+范围钳制+事件记录，v1.2 副本工件集，正典不动）。

### 2026-09-04T20:00Z P3.2.2 v12 训练启动 — 完成（后台运行中）
- 配置=v11 逐项复现（epochs 8/batch 128/lr 3e-4/max_steps 3000/latent 64/
  workers 2/graft 0.25/graft_neck 0.5/explicit_cmd+loco_cmd/seed 0 内建/
  preprocess fixed_v1 内建），仅数据池换为 v12 lineage 干净包。
- 启动验证：train 4,229,506 / val 2,000 窗（train 窗从 v1m 池 splits 生成）；
  GPU 81% util / 2.6GB；sparse stats 走**加载**路径（p2_stats32 nlink=3
  训练前后不变，共享 inode 未被写——v11 口径归一化，与 v12 可比性优先）。
- 产物将写 v12data 目录新文件名 p2_student_v12.pt/p2_history_v12.json
  （不触任何既有文件）。
- pid 纪律事故 2 起入档：①复合命令内 `echo $! > run.pid` 重定向莫名失败
  （mkdir 与 nohup 重定向成功、echo 失败，与上轮同款）→ 拆步补写；
  ②pgrep -f 自捕 bash 包装进程 → ps+python 过滤；dataloader 双 worker
  误入 pid 文件 → 以 ppid 关系取主进程单值 3859845。
- 下轮 = P3.2.3（轮询训练完成后跑 backward_heldout 验收+全标准门）。

### 2026-09-04T19:00Z P3.2.1 v12 数据包 — 完成
- **重要更正（P3.1.1 结论③撤销）**：v2combo 的训练池
  `sparse_full_v1m.npz`（MotionData 实际加载件）**含全部 400 条 `_rev`**
  （此前误查 v5i 族 records 得出"0 条 backward"）。v11 是**带 backward
  数据训练的**（358 _rev 在 train、42 在 val）。G2 语境从"补数据"修正为
  "验证+lineage 卫生"。
- **但族级复核否决"免重训"捷径**：42 条 val `_rev` 的 14 个族**全部**
  同时在 train（孪生变体泄漏，裁决明令禁止；closure_status 的
  cross_split=0 用的是 v5 manifest 的粗 group 键，不覆盖 v2_backward
  内部）。v11 不存在 lineage 干净的 backward heldout → 仍需重训 v12。
- **v12 数据包**（20260904T1900Z_p321_v12data/，build_v12_package.py
  commit ee048c2）：
  - heldout 选族规则（看结果前冻结）：按 _rev 原对类别分组、族尺寸
    ≤类别总量 30% 才合格（巨族留守训练主体）、小族优先累计至类别 15%
    或合格族耗尽 → **8 族 29 条**（locomotion 14：151900/151844/151932/
    151916；walk_head 15：relax/bored_snore/angry_no/neutral_no）。
  - train 10690→10299：剔 heldout 族 `_rev` 52 条（29 heldout+23 异常
    train 侧）+ **heldout 族前向孪生 339 条**（孪生泄漏双向封死）+
    30 异常 `_rev` 全部剔除（血统缺陷修复=排除+原因记录，原文件不动）。
  - val 808 不动（标准门与 v11 可比）；G2 验收只在 backward_heldout。
  - 验证：heldout `_rev`/前向孪生/异常件在 train 残留=0/0/0；大 npz
    硬链共享（nlink=3）零拷贝，正典零写。
- 下轮 = P3.2.2（启动 v12 后台训练，v11 同配置，BDX_PLANNER_DATA=
  v12data 目录）。

### 2026-09-04T18:00Z P3.1.2 reconstruction gate 全量评估 — 完成，GATE PASS
- **门判据**：frozen teacher（p1_ae_l64+fixed_v1）对 370 条干净 `_rev`
  的重建 q ≤ 同类 forward val 基线 1.3×，同窗口化（非重叠 50 帧窗）。
- **结果**：locomotion **24.9/20.0=1.244× [PASS]**（5520 窗）、walk_head
  **33.2/31.9=1.041× [PASS]**（1850 窗）；overall backward q 27.0 mrad；
  realized vx mean −0.206 m/s、**99.2% 负向**（真后退大规模数据层成立）。
  P0.1 初样的 1.35× 偏悲观源于小样本+窗口化不匹配。
- **G2 走 A 线**（数据修复+学生重训；teacher/decoder 冻结不动）。
- 370/30 干净/异常分区全量入档（30 条异常含原因字符串，供数据修复线
  排查；非删除——修复线需给出处置）。v2combo val 同窗口化基线同时
  重算（loco 20.0/stand 16.3/transition 14.2/walk_head 31.9，n=666~2107）。
- 产物：20260904T1800Z_p312_gate/（result+per_clip+log）；脚本
  eval_backward_gate.py commit e5342c6。背景作业纪律执行（nohup+纯数字
  pid+时间戳目录）；正典零写。
- 下轮 = P3.2.1（构建 backward 池：370 条重导出+lineage 级
  train/heldout 切分）。

### 2026-09-04T17:00Z P3.1.1 backward 血统追溯+修复前基线 — 完成
- **血统定论（三层）**：
  ① 5 条 deployment backward 窗**不是 `_rev`**——是 v5 干预回放中命令 −0.25
  从未被实现（回放专家只会前向走；follow_err +0.60~0.75；realized +0.30~
  0.40 m/s，本次在数据层逐帧复核确认：root dx +0.29~+0.39m，teacher 重建
  51–65 mrad=离流形困难窗）。
  ② 真正的 backward 数据=PoC 旧池 `v2_backward/` 族 **400 条 `_rev`**
  （walk/walkhead 时间反演）：成对检验 92%（370/400）位移翻转干净，
  30 条异常（幅值变化）；PoC 报告的"_rev 速度读数为正"缺陷**未在池 npz
  复现**（linvel 与位移符号 80/80 一致；该缺陷当出自下游读出/导出层）。
  ③ **正典 v2combo 池 0 条 backward**（audit-fixed 重训时整族被剔除）——
  这才是 v11 无 backward 能力的根源，非训练失败。
- **修复前三指标基线**（ruling 要求，probe_backward_baseline.py commit
  234929a，输出 20260904T1700Z_p311_prefix/）：5 条 v5 分支=dx +0.29~
  0.39m / vx +0.30~0.40 / q 51–65 mrad；_rev 干净样本 24 条 480 窗=
  **teacher 重建 27.2 mrad（p90 34.2）**，realized vx −0.204（100% 负向，
  真后退在数据层成立）。
- **gate 初样信号**：27.2 vs teacher val locomotion 20.1 → **1.35×**，
  贴着 1.3× 门边界；按类别分列后走向待 P3.1.2 全量裁定（370 条全评，
  locomotion/walk_head 分开对基线）。
- 工程坑入档：probe 首版 quat 解包 w↔x 错位（q[...,0]=w 不是 x）→ 6d 块
  崩 → teacher 爆炸（109 rad）；改用库 quat_wxyz_to_mat/quat_yaw 后与
  P2Dataset 逐位一致。旧池 npz 按布尔掩码逐字段访问会整字段重复解压
  （800K 帧级 npz 必须用 lengths 累积索引+字段单次加载）。
- 下轮 = P3.1.2（全量 gate 评估）。

### 2026-09-04T16:30Z 治理轮 — 用户裁决落档（非 P3 执行轮）
- 用户复审结论：P0/P1/P2 全 PASS，G1=SUBSTANTIVELY_CLOSED，
  P3=AUTHORIZED_TO_CONTINUE；vx 仪器替换获追认（阈值未变、旧仪器被
  teacher 不可达证据否证）。
- 落档六件：①G2 终态 A/B 细化（B 必须同步收窄 envelope+adapter 钳制/
  拒绝+事件记录+Demo 披露；不得倒放冒充后退）；②P3 执行纪律（修复前后
  三指标记录、不删困难 clip、decoder 门先于重训）；③P4 前置=
  INSTRUMENT_VALIDITY_MATRIX（4 通道×8 字段）+ 冻结的统一仪器选择规则
  （teacher 可达性决定仪器，非学生表现）+ neck_pitch teacher 反择优补数；
  ④G5 陈旧"93 断链"描述更正为当前事实（原深度浅克隆+31 blob 已修复，
  fsck=0，基线 0edb207），旧文划线保留；⑤G4 gesture 边界（lineage/
  source-group holdout，禁随机拆帧，无独立留出则 NOT_ESTABLISHED 披露）；
  ⑥停机条件 6 条 + git 边界（e133f9e OBSERVED_NOT_MERGED）。
- 新建 `CURRENT_EVALUATION_AUTHORITY.json`（机器可读权威覆盖指针：
  旧 deployment_eval.json/capability_envelope.vx 标记 SUPERSEDED+SHA；
  authority_of_record=closure_status/teacher_realization/P2_1_SWEEP_EVIDENCE
  /BASELINE_REPRODUCE 各带 sha256；P4 将在 eval 脚本入口实现 SUPERSEDED
  检测守卫）。
- 本轮无代码/数据变更（纯治理文件）；正典零写；git 无新提交。
- 下轮 = P3.1.1（_rev 血统追溯+修复前基线三指标）。

### 2026-09-04T15:00Z P2.2 G1 落档 — 完成（G1 除 P4 形式件外实质闭合）
- **G1-vx 裁定落档**（STATE.md G1 节证据修订）：deployment 回归子门=
  INVALID-INSTRUMENT（teacher 不可达），扫描仪器实测 0.8691/0.9982/符号 6-6
  ✓ + 平价 1.077/0.921 ✓ + sign_acc 0.978 ✓。
- **neck_pitch 单独 gate 出数**：新探针 `probe_neck_sweep.py`（q[11]，/1.7，
  固定行走历史扫描）slope 0.342/R²0.996 单调但偏置 −0.41（姿态先验）；
  deployment 匹配窗 n=12 回归 **slope 0.734/R²1.000/截距 0.045** ——仪器差
  解释清楚，**gate 达标**。
- **neck_forward/neck_roll 首份验证数据**（P4 输入）：deployment 零非零样本
  （inert 记录维持）；扫描 forward 0.327/R²0.905 单调（0→0.45，0.7→0.70）、
  roll 0.474/R²0.9994 过零 sign 5/5——两通道相对可控首次有数。
- vy/vyaw/neck_yaw 维持门沿用 deployment 数（0.656/0.937、0.537/0.677、
  1.028/0.963）✓——G1 只剩 P4 对 fwd/roll 的正式处置形式件。
- 产物：probe_neck_sweep.py（新脚本，入库）；输出
  bdx_planner_distill/20260904T1500Z_p22_neck/probe_neck_sweep.json（数据目录，
  不入 git）；正典零写。
- 下轮 = P3.1（backward reconstruction gate，G2 主件）。

### 2026-09-04T14:00Z P2.1 vx 扫描探针诊断 — 完成
- **v11 vx 通道健康**：固定历史命令扫描（fixed_v1 正典口径重跑 p29）
  slope **0.8691 / R² 0.9982**（n=6 含反向，6/6 符号对）；桶跟踪 overall
  |Δvx| **0.010 m/s**（backward 桶 0.015）；仅前向 slope 0.8996/R² 0.999。
  对照 08-24 旧 p29（preprocess 前口径）|Δvx| 0.027/q 84 → fixed_v1 全面更优。
- **deployment 回归门=覆盖面伪影（证据）**：per_bin 显示 79% 样本（183/231）
  挤在 cmd 0.2 单 bin；5 条 backward 窗 GT 符号反转（cmd −0.25/realized
  +0.35~+0.50，即 _rev 导出缺陷 val 侧实体，clip 名已锁定入档）把 teacher
  本体拖到 slope −0.173/R² 0.053 → 回归门对学生不可达，测的是数据不是模型。
- 结论：PoC 0.82–0.84 与 deployment −0.28 之谜=**两种仪器差**，v11 无 vx 退化。
- 产物：P2_1_SWEEP_EVIDENCE.md + baseline/p29_eval.json（断链重跑，正典未动）。
- 处置建议入档：G1-vx 以扫描判据+teacher 平价裁定（证据修订，终报披露）；
  5 条 _rev clip 归入 G2 数据修复线，修后重跑 deployment 门验证字面门可否恢复。
- 下轮 = P2.2（G1-vx 裁定落档 + neck_pitch 单独 gate）。

### 2026-09-04T13:00Z P1.1 git 痊愈 — 完成（G5 达成主体）
- **根因改写**：仓库是 depth-1 浅克隆（.git/shallow 列两个 tip）+ 真实丢失 31 个 blob
  （tree→blob 断链；含 070568a 自己的 tree）。"93 断链"旧计数含 dangling 噪声。
- **修复路线**：tar 备份 .git（93MB，git_repair/20260904T1300Z_git_backup.tar.gz）→
  直接 fetch 失败两次（ref 未变静默跳过 / thin-pack delta 基缺失 unpack 失败；
  git 2.34 无 --refetch/--no-thin fetch）→ 干净裸镜像 /tmp/bfmz_mirror 全量拉取
  （origin 实有 main=10 提交 + phase2/closedloop-z=15 提交，含本地全部缺失祖先）
  → pack 文件级移植 + 重建 origin 跟踪 ref → fetch --unshallow → **fsck 0 断链**，
  HEAD 可走 14 提交，git status/log/show 全恢复。
- **健康提交 0edb207**（[distill-campaign] 前缀，148 文件 +24962 行）：planner 全线
  （planner/、train_p1/p2、eval_*、probe_*、demo_*、tests）、docs、mjlab_bdx、
  fds_* 实验、eval_p2.py 三缺陷修复；.gitignore 增排 logs/ model/ docs/papers/
  humanoidverse/data/（大产物不进 git）。工作树仅剩 .backups/（已 ignore）。
- 正典 sha256 全值入档：p2_student_v11.pt=f939b2b8…d99，p1_ae_l64.pt=e77a5746…f8b
  （git_repair/20260904T1300Z_canonical_sha256.txt）；fsck 前后快照同目录。
- 工作树 44G 未动（只有 add 读文件）；正典目录只读复核无写。GPU 全程空闲未用。
- 遗留：远端有更新提交 e133f9e（origin/phase2/closedloop-z，bf16 修复线）——
  属用户其他线工作，本战役不 merge 不 push，仅记录。
- 下轮 = P2.1（vx 扫描探针，P0.1 证据链的延续）。

### 2026-09-04T12:00Z P0.1 基线复现 — 完成
- **eval_deployment 复现确定性成立**：与 closure_status.json headline（08-24 生成）16 位浮点
  逐位一致（q 25.975045961764266 / fk 12.019891323350766 / vx −0.28431529732406424）。
  v11 真基线锚定。
- **重要更正**：正典目录里 deployment_eval.json 是陈旧工件（20 行、vx n=5，旧版脚本筛选子集），
  与 414 行现行门不可比；权威=closure_status+teacher_realization（带 sha 溯源）。陈旧件原样保留。
- **vx 之谜关键证据（P2 输入）**：teacher 本人 vx R²=0.053（同切片）→ G1 的 vx
  R²≥0.60 门在该切片对任何该 teacher 的学生不可达，属评测覆盖/仪器问题；
  planner_vs_teacher 四通道 ≈1.0（R² 0.92–0.99）= v11 是忠实蒸馏。
  P2 将以命令扫描探针为增益仪器 + teacher 基线对照重裁 G1-vx（不自改判据，带证据披露）。
- **eval_p2.py 修 3 缺陷**（TEACHER=None / cmd_dim 分支安全 / build_cmd 7 维契约），
  工作区改动，commit 延至 P1。v11 补全形态：student 63.3 / teacher 20.3 mrad（3.12×），
  最差切片 stand 6.04×。产出 p2_student_v11.eval.json（新 inode，未触正典硬链）。
- 基线目录 20260904T1200Z_p0_baseline（cp -al）；正典 sha 复核未变；GPU 占用合规（497MiB/0%）。
- 产物：BASELINE_REPRODUCE.md。下轮 = P1.1。

### 2026-09-04 bootstrap（本文件创建）
- 摸底完成：蒸馏线正典=planner-v1.0-audit-fixed-rc1（v11）；
  git 93 断链；vx deployment 增益 −0.28/R²0.11 回归为最大未了项；
  backward 需先过 P1 gate；gesture 留出缺；环境 BFM-zero/.venv 就绪。
- 下轮动作 = P0.1。
