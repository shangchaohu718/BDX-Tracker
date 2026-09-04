# BDX Sparse-to-Full Motion Planner — PoC 结项报告

> 2026-08-17 ~ 08-19。分支 `sonic_planner_p1`。数据/产物在 `humanoidverse/data/bdx_planner_v2/`（v2 为最终版；`bdx_planner/` 保留 v1 谱系）。

## 一句话结论

**sparse 状态（1s 历史）+ 7 维命令 [vx, vy, vyaw, neck×4] → 64 维 latent → 完整 14-DoF 未来 1s 运动 → 现有 BFM tracker 执行**，全链在官方防泄漏留出集上闭环：tracker MPJPE 仅为 GT 参考的 1.04×（自回归部署形态），命令速度跟踪误差 ≤3.5 cm/s。

## 管线与代码地图

```
命令 [vx,vy,vyaw,neck×4] ──┐
                           ├→ CommandedEncoder(2.4M) → latent(64) → 冻结 TeacherAE 解码
1s sparse 历史(32维/帧) ───┘                                    │
                                                                ▼
                                              完整 14-DoF 参考 → tracking_inference → BDX
```

| 模块 | 文件 |
|---|---|
| 数据抽取（全量池→npz） | `scripts/extract_sparse_bdx.py --full-pool` |
| 可微 FK（对源数据 0.001mm 验证） | `planner/fk.py` |
| 窗口数据集 + 命令标签 | `planner/dataset.py`（P2Dataset） |
| 模型（MotionAE / CommandedEncoder） | `planner/model.py` |
| teacher 训练 / 评估 | `scripts/train_p1.py` / `eval_p1.py` |
| 学生训练（--explicit-cmd / --loco-cmd / --graft） | `scripts/train_p2.py` |
| 学生评估 / 命令探针 / P2.9a 评估 | `eval_p2.py` / `probe_command.py` / `eval_p29.py` |
| tracker 参考导出（teacher 重建 / planner tf/ar） | `scripts/export_ae_refs.py` / `export_planner_refs.py` |
| 演示视频 | `demo_planner.py`（三联）/ `demo_planner_v29.py`（命令时间线） |
| contact 回归测试 | `tests/test_contact_detection.py` |

环境切换：`BDX_PLANNER_DATA=<dir>`（默认 v1 目录）；训练均 `--workers 2`（mem_watchdog）。

## 阶段结论与数字（val = 官方防泄漏留出集）

| 阶段 | 结论 | 关键数字 |
|---|---|---|
| P0 数据 | 4820 clip / 4.11M 帧 / 22.8h / 50Hz | 量够、缺覆盖；发现并修复 contact 检测全零 bug（阈值 0.07→0.08m，站立踝高实测 0.0736） |
| P1 teacher | 64 维 latent 连续可解码（容量消融：64 饱和） | q 20.6 mrad / foot 6.6mm / 插值 SMOOTH |
| P2/P2.8 学生 | 显式命令通道使头部可控（观测空间头部表示对 neck_yaw 信息量为零：corr −0.005，这是 4 次训练干预全失败的根因） | 补全形态 1.77× teacher；yaw 命令增益 1.41 |
| P2.9a planner | 命令取代未来轨迹（真 planner 形态） | 全速度档 |Δvx| ≤0.035 m/s；vx 扫描增益 0.82–0.84 含反向 |
| P3 端到端 | planner 输出喂 tracker（自回归历史由 FK 重建） | **GT 1.00 / teacher重建 1.18 / planner-tf 1.06 / planner-ar 1.04**；AR 漂移有界（92–141 mrad / 25s） |
| 演示 | 命令时间线（走→左看→转→停）自动服从 | `demo_planner_v29.mp4` |
| VLA 接口 | Level-1 JSON→7D 命令的规则 adapter（含校验，测试通过） | `planner/command_adapter.py` |
| 长时程 | 2 分钟 AR 持续行走 21.4m；但参考的 base 高度随分钟级下沉（≤25s 有界）——部署模式每秒从真实状态重规划，自漂移不累积 | `longhorizon_traj.npy` |

方法学要点：tracker 评测判据用 `mpjpe_mean`（completion/fell_frame 在该 harness 中不可信）；评估一律 clip 级防泄漏；归一化统计只用 train 侧。

## 已知缺口（全部有量化记录）

1. walk_head 是 teacher 最难类别（30.8 mrad），学生已贴天花板，planner 形态 tracker 比值已到 1.04–1.08。
2. （已修正）vyaw 曾误报 0.227 rad/s——实为探针读错输出通道（angvel 场恒零但姿态轨迹正确）；用 base_R 差分重测为 **0.013 rad/s**，正常。真实验留：模型的 angvel 输出场与旋转轨迹不一致（下游 pkl 导出不消费该场，无影响）。
3. 后退走：planner 命令形态正确（|Δvx| 0.014），但补全形态重建差（240 mrad）且时间反演导出有瑕疵（_rev clip 速度读数为正）。
4. gesture 全部 30 条都在官方切分的 train 侧，泛化不可评估。
5. git 对象库损坏（`git show HEAD:*` 失败），全部实验尚未版本化——修复后应尽快健康 commit。

## 数据侧反馈清单

manifest yaw 回绕伪影修复（±156 rad/s，影响数据标签质量）；后退走反演导出修复；补 gesture-only 留出 clip；M2 程序化数据引擎（产 OOD 成对数据）的投入决策——覆盖度证据链已完整（"量够、缺覆盖"）。

## 复现（关键命令）

```bash
BDX_PLANNER_DATA=$PWD/humanoidverse/data/bdx_planner_v2 \
  .venv/bin/python humanoidverse/scripts/eval_p29.py            # planner 速度跟踪
BDX_PLANNER_DATA=$PWD/humanoidverse/data/bdx_planner_v2 \
  .venv/bin/python humanoidverse/scripts/probe_command.py       # 命令可控性
.venv/bin/python -m humanoidverse.tracking_inference \
  --model-folder results/bfmzero-bdx-full \
  --data-path humanoidverse/data/bdx_planner_v2/tracker_planner_ar_refs.pkl \
  --simulator mujoco --motion-list 0 1 2 3 4 5 6                # P3 tracker 终测
```

最终权重：`p1_ae_l64.pt`（teacher）、`p2_student_exp4cmd.pt`（补全）、`p2_student_p29b.pt`（planner，canonical）。

## 后续路线（按优先级）

1. BFM tracker 训完 → 最终 P3 复测（一行命令，见上）。
2. M2 数据引擎（OOD 覆盖）→ 重训 planner 扩分布。
3. VLA 接口：语义命令 → 本层 7 维命令的 JSON 协议（模型侧接口已就绪）。
4. M4：tracker 重训时混入 planner 生成参考 + 修复后的 contact obs。

---

## Errata — 2026-08-24

以下修正追加于原始报告之后，历史正文保留不改：

1. **head_rotvec 表示缺陷（legacy）**：`canonical_sparse` 曾将头部轴角当三维向量旋转（而非先左乘 R(−ψ) 再取 log），使该通道残留绝对世界 yaw——训练 stats 中 z 维标准差 1.529 为该问题的证据。已修复为 `fixed_v1` 并版本化；**当前 canonical checkpoint 绑定 `legacy_v0`（训练时语义）**，parity 显示两版本高度接近（latent cos 0.9985，q +0.51 mrad），`fixed_v1` 留待 v1.1 重算 stats 并重训时切换。
2. **P2.6/P2.7 归因降级**：当时测得的 corr(head 朝向 z, neck_yaw) ≈ −0.005 是在 legacy 表示（含上述污染）下取得，**不构成"所有正确头部观测均无信息"的信息论证明**。当时条件下四种 conditioning 方法未获可控响应属实；显式命令通道的架构决策由干预实验与控制任务定义独立支持，不受影响。
3. **通道依赖的准确措辞**：当前 checkpoint 对 head_rotvec 的**测得依赖较弱**——置训练均值的消融仅造成约 +1.35 mrad 的 q 变化。"通道完全死亡"不成立。
4. **评估 SCALE bug（8-21）**：combo holdout 曾将 manifest 物理命令再乘归一化系数（neck ×1.7 / vyaw ×1.5 / vy ×0.5），导致"同向 neck 崩溃 0.70 rad"的假象；修正后真实 Δneck 0.235–0.35。受影响的第五阶段结论已全部重测。
5. **export 旋转逆（8-21）**：世界系恢复误用 Rz 而非 Rz.T（2ψ 误差，174° yaw 窗口实测 12.2°）；已修复并有生产级合同测试守护。
6. **"planner-ar 1.04×" 的限定**：该数字测于修复前的导出参考、7 条精选 clip、tracker 中途 checkpoint；应表述为"修复前参考在当前 tracker 上的相对可追踪性"，不作为端到端闭环结论。重测待 tracker 可用。
