# DISTILL_FINAL_REPORT — BDX 专家策略蒸馏战役终报

- 生成：2026-09-05T05:00Z（P6.1，战役最后一轮）
- 权威状态：`/home/tcl/Desktop/start/DISTILL_CAMPAIGN/STATE.md`（状态块 CAMPAIGN=DONE）
- 战役窗口：2026-09-04 12:00Z 启动（自动化 automation-4b88220e，每小时一轮）→ 2026-09-05 05:00Z 收口，共 17 轮
- 最终目标（用户 2026-09-04 授权原意）：将专家策略（teacher MotionAE）蒸馏至 MLP/Transformer 学生（CommandedEncoder），达成 STATE.md G1–G6 全部终态判据

## 1. 结论

**G1–G6 全部达成，战役完成。采用产品 = v12 学生（CommandedEncoder, latent 64, 7D 显式命令, ~2.15M 参数）**，按预注册采用规则（数字出炉前冻结）机械适用后唯一合格候选。相对正典基线 v11：部署形态质量持平（26.56 vs 25.98 mrad，同阈 ≤30 门内、同 CI 带内），**倒退能力从"不支持"变为受控支持（vx 包络 [−0.65, 0)）**，这是本战役的主要能力增量。

## 2. 最终产品

| 项 | 值 |
|---|---|
| 学生 checkpoint | `p2_student_v12.pt`，sha256 first16 **34047cde8f6c5a3b** |
| 训练配置 | `--explicit-cmd --loco-cmd --graft 0.25 --graft-neck 0.5`，7D cmd=[neck4/1.7, loco3 (1.0/0.5/1.5)]，train 10299 窗池 |
| teacher | `p1_ae_l64.pt`（sha16 e77a5746，未动） |
| 正典基线参照 | v11 `p2_student_v11.pt`（sha16 f939b2b8，未动） |
| 能力包络 | `capability_v12.py` + `capability_envelope_v12.json`：vx ChannelEnvelope(−0.65, 0.5) |
| 命令适配器 | `command_adapter_v12.py`：越界钳位 + 饱和事件 JSONL（$BDX_ADAPTER_EVENT_LOG）+ 结构错误 raise |
| 终评工件 | `humanoidverse/data/bdx_planner_distill/20260905T0400Z_p54_final/`（FINAL_EVAL.json + v12/v13 overlay 双候选 + parity/backward 证据） |
| demo | `…/p54_final/v12/demo_planner.mp4`（3 联画 GT｜稀疏补全｜头部命令编辑；look-left 命令响应 yaw +0.021→+0.619 rad vs 目标 0.7，forward 保持 0.512→0.479） |

注：正典目录 `bdx_planner_v2combo/` 全程只读零写入（每轮核验）；产品 v12 未写入正典注册表 —— 是否晋升正典由用户裁决（候选+证据链已就绪）。

## 3. 终态判据 G1–G6 判定

| 门 | 判定 | 依据（数字） |
|---|---|---|
| **G1 逐通道命令可控** | PASS | vx：扫描仪器 slope 0.90 含反向（部署仪器对 vx 无效—teacher 不可达，冻结教师优先规则改判）；vy 0.686/R²0.948；vyaw 0.536/R²0.684；neck_yaw 1.004/R²0.959；neck_pitch 1.215/R²0.9955（teacher 1.248/0.9999 可达性成立）；neck_forward/roll **FROZEN_WITH_DISCLOSURE**（见披露 6） |
| **G2 倒退能力** | PASS（终态 A，5/5 条件） | P1 重建门 locomotion 1.244×/walk_head 1.041× ≤1.3；血统干净 heldout 941 负命令窗 **0 符号反转**、mean\|Δvx\| 0.0135≤0.05、q 42.6 mrad；包络 vx [−0.65, 0) + 适配器钳位+事件；逐深度带验证 −0.05→−0.65 |
| **G3 部署形态质量** | PASS | val-lineage 414 clips：q 26.56≤30 mrad；FK 11.78 mm；planner/teacher 平价 slope vx 1.003/vy 0.998/vyaw 0.996（R²≥0.999）≤1.15；combo holdout 切片 max 32.4≤45 |
| **G4 泛化（血统 holdout）** | ESTABLISHED | v13 单变量实验（v12 配置逐字复制，仅剔除 15 个手势 motion group）：unseen 65.63 vs fit-control 66.36，**gap −0.73 mrad ≈ 0**；teacher floor 27.87 |
| **G5 工程收口** | PASS | git fsck = 0 errors（原 93 断链 = depth-1 浅克隆缺 31 blob，已 unshallow+完整对象恢复，用户 2026-09-04 裁决更正）；健康基线 0edb207；正典 checkpoint sha256 记录在案（git_repair/20260904T1300Z_canonical_sha256.txt） |
| **G6 终报+收口** | PASS（本文件） | 终报 + demo 重导（v12）+ 记忆更新 + CronDelete automation-4b88220e + 本一次性汇报 |

## 4. 关键指标 vs 正典基线

| 指标 | v11（基线） | v12（采用） | 说明 |
|---|---|---|---|
| 部署形态 q (mrad, 414 val clips) | 25.98 | 26.56 | 同阈 ≤30、同 CI 带内持平 |
| 部署形态 FK (mm) | 12.02 | 11.78 | 略优 |
| vy / vyaw 增益 | 0.66 / 0.54 | 0.686 / 0.536 | 同级 |
| neck_yaw 增益 | 1.03 | 1.004 | 同级 |
| **vx 增益** | **−0.28（R²0.11 回归）** | **0.90（扫描仪器，含反向）** | **主要修复** |
| 倒退支持 | 不支持（5 val 符号反转） | [−0.65, 0) 受控支持 | **主要能力增量** |
| 完成形态 q (mrad) | 63.3 | 68.6 | +5.3，在 ±10 同配置 run 方差带内（见披露 3） |

## 5. 全部披露（limitation 清单，全部如实入档）

1. **vx 仪器修订**：部署仪器（被动、覆盖饥饿）对 vx 无效——teacher 在部署 clip 集上 vx 不可达（teacher −0.1727 vs student −0.2843, n=231 硬锚点），冻结"teacher 可达性决定仪器非学生表现"规则改用受控扫描仪器（slope 0.90）；INSTRUMENT_VALIDITY_MATRIX（4ch×8field）在案（20260904T2300Z_p41_matrix/）。
2. **P2.2 内联 neck_pitch 数字（0.734/R²1.000）被正式合同数字取代**（1.215/0.9955）：内联值按正式合同不可复现（其截距=teacher 截距、量程=物理量程×1.7），已挂 SUPERSEDED 守卫（planner/authority.py，eval 入口接线），旧值仅作历史记录。
3. **完成形态 run 方差 ±10 mrad**：v12 完成形态 68.6 vs v11 63.3（+5.3）在带内；采用仪器=部署形态（run 方差 <0.6 mrad），完成形态仅披露不入判。
4. **v13 失格于恰好 1 个符号反转窗**（shallow 命令 −0.063 → realized +0.018 m/s，边界浅命令）：预注册门 100% 严格机械适用，未做任何阈值通融。
5. **v11 血统失格**：v11 训练集含 heldout `_rev` 及其前向孪生（其"0 反转"数字为训练污染产物），E2 血统条款不合格——数字好也不采用。
6. **neck_forward / neck_roll = FROZEN_WITH_DISCLOSURE**：K=8 多历史 worst-case 门（跑前冻结）下 fwd worst slope 0.088/R²0.214、roll worst 0.313（R²≥0.999 全历史）；两通道冻结不宣称可控，适配器语句已发布。
7. **neck_pitch 部署仪器仅 2 bin 覆盖**（max share 0.667），且对符号约定敏感——数字有效但覆盖稀疏。
8. **深带 q 抬升 ~80 mrad**：深度带越深质量越降；可暴露深度由"链式合格+实测覆盖"双界共同限定（−0.65 为证据界，非物理界）。
9. **包络仅完成形态验证**：部署闭环倒退未运行（闭环 backward 不在本战役授权范围）；包络+适配器是部署侧安全钳位，不是闭环性能声明。
10. **G4 证据归属训练配置线（v13），非产品 v12 本身**：v12 训练含全部手势 clip（441/441），产品层面无法构造手势 holdout；泛化结论来自 v13 单变量受控实验（差异仅 15 个 motion group）。
11. **训练 combo 切片 max 56.9 mrad（n=1 单例切片）**：G3 门文本只覆盖 4 个 combo holdout 切片（max 32.4≤45 过门）；训练侧最差切片如实披露不入门。
12. **G3 判定集 = val-lineage 414 clips**（train-side 部署 clip 剔除，group 血统规则）。
13. **30 个异常 `_rev`（dx 未翻转/无配对）作为谱系缺陷修复从 v12 训练集剔除**；原始正向 clip 未动。
14. **e133f9e = OBSERVED_NOT_MERGED** 全程遵守；战役基线 0edb207→90e16f3 + 本战役 12 个 commit（见 §6），无 force push，正典零写入。
15. demo 重导出用的窗口 = val locomotion clip `v0/1242_-0.149_-0.0_0.0`（GT 命令派生，非人工摆拍命令序列）。

## 6. commit 索引（战役全部提交，`[distill-campaign]` 前缀）

```
90e16f3 (战役前基线顶) →
234929a P3.1.1 倒退基线探针
e5342c6 P3.1.2 重建门 PASS → G2=A 线
ee048c2 P3.2.1 v12 血统干净数据包
a59d45f P3.2.3 v12 heldout 验收 PASS
57243bb P3.3 v1.2 包络+适配器+深度带
15c6f7d P4.1 仪器矩阵+SUPERSEDED 守卫
799b524 P4.2 颈部两通道 FROZEN_WITH_DISCLOSURE
9ec0700 P5.1 手势 holdout 可行性（Option A 死→重训线）
2d3ba0f P5.2 v13 包+重训（单变量）
02efebe P5.3 G4 关闭（gap≈0）
e5f52d5 P5.4 终评 ADOPTED=v12
547d3ee P6.1 终报+demo+收口
```

## 7. 工件目录索引（`humanoidverse/data/bdx_planner_distill/`）

| 目录 | 内容 |
|---|---|
| 20260904T1200Z_p0_baseline | P0 基线复测（部署/完成形态双轨） |
| 20260904T1500Z_p22_neck | 颈部扫描探针（内联版，已被 P4.1 取代） |
| 20260904T1700Z_p311_prefix / T1800Z_p312_gate | 倒退三指标基线 + P1 重建门 |
| 20260904T1900Z_p321_v12data / T2000Z_p322_v12train | v12 数据包 + 训练产物 |
| 20260904T2100Z_p323_accept | v12 heldout 验收（941 窗 0 反转） |
| 20260904T2200Z_p33_v12artifacts | 包络 v1.2 + 适配器自检 + 深度带 + 覆盖 |
| 20260904T2300Z_p41_matrix | INSTRUMENT_VALIDITY_MATRIX（.json/.md） |
| 20260904T2400Z_p42_neckdispo | NECK_CHANNEL_DISPOSITION |
| 20260905T0100Z_p51_gesture / T0200Z_p52_v13data / T0200Z_p52_v13train / T0300Z_p53_g4eval | 手势血统分析、v13 包+训练、G4 三臂评测 |
| **20260905T0400Z_p54_final** | **终评：FINAL_EVAL.json、ADOPTION_RULE.json、v12/v13 overlay 双候选、parity、backward、demo** |

## 8. 复现入口

```bash
cd /home/tcl/Desktop/start/BFM-zero
P=humanoidverse/data/bdx_planner_distill/20260905T0400Z_p54_final
# 终评重放（只读 overlay，写入需换时间戳目录）
BDX_PLANNER_DATA=$PWD/$P/v12 .venv/bin/python humanoidverse/scripts/eval_deployment.py --checkpoint $P/v12/p2_student_v12.pt
# demo
BDX_PLANNER_DATA=$PWD/$P/v12 .venv/bin/python humanoidverse/scripts/demo_planner.py
# 包络/适配器（含自检）
.venv/bin/python -m humanoidverse.planner.capability_v12
.venv/bin/python -m humanoidverse.planner.command_adapter_v12
```

## 9. 遗留（非本战役范围，供后续路线参考）

- 部署闭环倒退验证（G2 包络只有完成形态证据）；v13 的 1 个边界反转窗是否为可训练掉的浅命令端点效应；
- neck_forward/roll 解冻条件（需教师可达性或表示层改造）；vx 部署仪器覆盖扩容（需含倒退的部署谱数据）；
- v12 是否晋升正典注册表（canonical.json 仍指向 v11——晋升属用户裁决，本战役未动正典）。

## 10. Erratum（2026-09-07 追加，用户终裁后 append-only 修正，§1–9 原文未动）

**E-1 §4 表 vx 行仪器口径错误。** 表中将
`v11 −0.28（R²0.11，回归）vs v12 0.90` 并排呈现，读作"同仪器模型增益修复"。
这是两个不同仪器的数字，该叙事已被 P2.1/P4.1 推翻。正确口径：

```text
v11 deployment 仪器 slope −0.2843 = INVALIDATED_DEPLOYMENT_INSTRUMENT_RESULT
  （无效原因：覆盖饥饿——79.2% 样本挤在 0.2 单 bin、5 条 _rev 导出符号反转
   污染；teacher 同切片 R²=0.053 不可达，仪器对学生无效）
v11 有效扫描（sweep）仪器 slope = 0.8691 / R² 0.9982
v12 扫描（sweep）仪器 slope = 0.90（P3.2.3，含反向）
```

v11 **没有发生 forward-vx 模型退化**。同仪器比较为 0.869 vs 0.90。
v12 的真实增量为：训练/留出血统干净、真正 backward 数据修复（异常 `_rev`
剔除+孪生防漏）、负 vx 完成形态支持（[−0.65,0) 包络）、teacher 保真
（parity 1.003）、受控包络+适配器建立。不是"把 forward vx 从 −0.28
训练到 0.90"。历史 −0.2843 保留但只作
INVALIDATED_DEPLOYMENT_INSTRUMENT_RESULT 引用。
（§4 表 vy/vyaw/neck_yaw 行为同仪器 deployment 数字，P4.1 判定该仪器
对这些通道有效，不受本条影响。）

**E-2 0.58 mrad 与 5.3 mrad 为不同字段，非笔误。** 全字段名：

```text
deployment_q_error_mean（部署形态，414 val-lineage clips）：
  v11 25.975 → v12 26.56 = +0.58 mrad
completion_form_q_mean（标准门完成形态，val 全集）：
  v11 63.3 → v12 68.6 = +5.3 mrad（同配置 run 方差 ±10 带内）
```

两指标仪器与样本集均不同，引用时必须带全字段名，禁止裸写"q 差 X mrad"。

---

## §11 Canonical Adoption Execution Record (2026-09-07, user ruling DISTILL_CANONICAL_ADOPTION_V12)

### 11.1 Bounded closed-loop backward smoke (Step 2)

Frozen pre-run plan (`bdx_planner_distill/20260907T0100Z_v12smoke/SMOKE_PLAN.json`):
vx ∈ {−0.15, −0.30, −0.50, −0.65} × 3 deterministic seed-frame resets (0/17/33) = 12
trials; each 2 s settle + 10 s command + 3 s stop; all other 7-D commands zero/neutral.
Chain: v12 checkpoint + capability_v12 + `command_adapter_v12.adapt()` on EVERY window
(zero bypass; adapter event logs all empty = no unexpected clamping) + the official
deployment runtime (closed_loop_test.py config mirrored verbatim).

**Result: 12/12 FAIL → mechanical envelope = UNSUPPORTED** (per frozen mapping:
全部失败 → backward=UNSUPPORTED, adapter rejects negative vx). No falls
(min_z 0.179–0.197), no sustained non-foot contact, zero G2 sign-reversal windows.

### 11.2 Instrument finding (disclosure; the frozen verdict is NOT reinterpreted)

- `episode_length_buf` wraps every ~22–30 steps in EVERY trial: the env auto-reset
  teleports the robot +0.10–0.18 m along +x (11–14 times per 10 s command segment;
  per-trial path 4.9–6.6 m vs net ≈ 0).
- Non-reset drift is −0.24…−0.35 m/s at ALL four backward levels AND under the
  forward control (+0.30 → −0.26 m/s): the drift is command-sign-independent — the
  recovery half of a teleport limit cycle, NOT policy backward walking.
- Forward-control diagnostic (`20260907T0130Z_diagFwd`, labeled DIAGNOSTIC, identical
  harness/adapter): net steady **+0.005 m/s** → this runtime cannot demonstrate net
  x-locomotion in ANY direction.
- Corroboration in the campaign's own official v12 deployment eval (414 rows):
  vx slope −0.252 / R² 0.091 with realized vx ≈ 0.003–0.007 regardless of command,
  while vy R² 0.948 and neck R² 0.959 stay healthy → x-channel-specific pinning.
  This retroactively explains BOTH generations of deployment-vx pathology
  (v11 −0.284/R² 0.053 and v12 −0.252/R² 0.091) as instrument-dominated.
- Recorded as `INSTRUMENT_VALIDITY_MATRIX_addendum_20260907.json` (x-displacement
  channel INVALID; vy/vyaw/neck/q-error/FK and all completion-form instruments
  unaffected).

**Capability statement:** backward is NOT refuted and NOT certified. Completion-form
evidence (sweep 0.90, G2 zero reversals) remains valid as completion-form. Re-test
path: fix the runtime termination regime (R26 precedent), then re-run the frozen
SMOKE_PLAN unchanged.

### 11.3 Atomic promotion (Step 4)

- Backup: `bdx_planner_v2combo_preV12_backup_20260907T0930Z.tar.gz` (sha16 9229b54e684b4e6f);
  superseded pointer archived as `canonical_v11_archived_20260907.json`.
- `p2_student_v12.pt` copied into `bdx_planner_v2combo/`, sha256-16 verified
  `34047cde8f6c5a3b`; `canonical.json` switched (registry: v11=SUPERSEDED_INVALID_LINEAGE,
  v13=REJECTED_SINGLE_SIGN_REVERSAL, v12=CANONICAL).
- `ADOPTION_MANIFEST.json` (sha16 4ec9da7c80134051): indivisible package inventory +
  forbidden-list compliance record.
- `capability_v12.py`: additive `CLOSED_LOOP_DEPLOYMENT_ENVELOPE` (validated open-loop
  ENVELOPE untouched).
- `command_adapter_v13.py` (canonical binding): every negative `velocity.forward` →
  `CommandValidationError` + durable JSONL event (`reason=backward_unsupported_closed_loop_smoke`);
  positive-range semantics identical to v1.2; no bare-checkpoint path around the envelope.
- Regression acceptance PASS: resolve_canonical → v12 sha verified; student forward pass
  bitwise-identical to the smoke-overlay resolution; adapter v1.3 reject/pass behavior
  verified; v1.2/capability self-tests unchanged-green.

Forbidden list respected: no adapter bypass, no checkpoint/training-data/threshold
edits, no mid-run reset re-picking, e133f9e remains OBSERVED_NOT_MERGED, no writes to
canonical data during the smoke (overlay only). Planner demo (demo_planner.mp4) human
signoff remains pending and blocks EXTERNAL_DEMO_SIGNOFF only.
[2026-09-07 correction, user order: the earlier "G1 demo" label was a mislabel and is
removed from all living records — this project (BDX distill line) has no G1-robot
involvement; the campaign goal numbering G1–G6 (G = Goal) is unrelated and retained.]

**Canonical pointer committed → STOP. No new training campaign opened.**

---

## §12 Terminal Confirmation (2026-09-07, user final ruling — campaign closed)

Execution accepted. **v12 canonical promotion is valid; the distill campaign is
formally over; no further distill rounds.**

```text
V12 = CANONICAL
V11 = SUPERSEDED_INVALID_LINEAGE
V13 = REJECTED_SINGLE_SIGN_REVERSAL

DISTILL_CAMPAIGN = DONE
BACKWARD_CLOSED_LOOP = UNSUPPORTED
RUNTIME_X_AXIS_INSTRUMENT = INVALID
```

### 12.1 vx four-quadrant capability boundary (recorded in canonical.json)

| Capability | Status |
|---|---|
| Forward completion-form generation | VALIDATED |
| Backward completion-form generation | VALIDATED (original evidence remains valid) |
| Forward closed-loop execution | NOT ESTABLISHED |
| Backward closed-loop execution | NOT CERTIFIED (product side UNSUPPORTED) |

The 12/12 smoke failure cannot refute the planner's ability to generate backward
motion — the same runtime cannot produce net x-displacement even for the forward
control, and resets constantly. It does establish that the official
planner → motion injection → BFM tracker → MuJoCo chain is not a valid
acceptance instrument for x-axis closed-loop capability. `command_adapter_v13`
rejecting negative vx is the correct conservative product decision. Positive vx
stays open for legacy compatibility but must be labeled
`completion-form validated / legacy-compatible / closed-loop not established in
current runtime` — never `closed-loop verified`.

### 12.2 Runtime issue ownership — RUNTIME_X_AXIS_TERMINATION_LOOP

Registered as an independent integration issue
(`BFM-zero/ISSUES/RUNTIME_X_AXIS_TERMINATION_LOOP.md`, status REGISTERED, not
scheduled). Belongs to: tracker/runtime integration; termination/reset
mechanism; motion injection timing and state continuity. Does NOT belong to:
v12 distillation training failure; planner model degradation; backward
data-repair failure. Any future fix is a separate runtime task, and afterwards
the original frozen plan must be re-run with **symmetric forward and backward
testing** — backward-only retests are not acceptable.

### 12.3 Final product state

```text
Canonical planner:                   p2_student_v12.pt
Engineering adoption:                COMPLETE
Negative vx:                         REJECTED_BY_ADAPTER
Positive vx:                         ALLOWED_WITH_SCOPE_DISCLOSURE
Closed-loop x-axis certification:    NOT ESTABLISHED
External demo visual signoff:        PENDING
Further training:                    NOT REQUIRED
Further automatic execution:         STOPPED
```

G1 label cleanup ratified: the BDX planner project is unrelated to the G1 robot;
campaign gate numbering G1–G6 is retained; legacy internal names
(`_get_g1env_observation` etc.) need no refactoring; `demo_planner.mp4` is the
BDX planner demo. Its human signoff affects the external demo only, not v12's
standing as engineering canonical.

**Line closed. The only restart condition is an explicit decision to fix the
runtime termination mechanism, which would trigger symmetric forward/backward
closed-loop verification.**
