# P0.1 基线复现报告（planner-v1.0-audit-fixed-rc1 / p2_student_v11）

> 2026-09-04T12:00Z run。基线目录：`humanoidverse/data/bdx_planner_distill/20260904T1200Z_p0_baseline/`
> （`cp -al` 硬链副本，不触正典；正典 `bdx_planner_v2combo/` 只读未动，全仓 sha 复核无变化）。

## 1. eval_deployment 复现（GPU，确定性成立）

- 复现命令：`BDX_PLANNER_DATA=.../20260904T1200Z_p0_baseline python humanoidverse/scripts/eval_deployment.py`（默认 max-clips 600）
- 复现结果与 **closure_status.json 的 deployment_eval_headline（2026-08-24T15:11:18 生成）16 位浮点逐位一致**：
  - q mean **25.975045961764266** mrad，CI95 [24.492346137002702, 27.475973353976985]，max 89.335
  - fk mean **12.019891323350766** mm
  - gains（414 行 val 窗）：vx n=231 slope **−0.28431529732406424** R² 0.1125901731143194 sign 0.978；
    vy 0.656/0.9366；vyaw 0.537/0.677；neck_yaw 1.028/0.963（与 teacher_realization.json planner_vs_requested 逐位一致，其 provenance 记 checkpoint sha16=f939b2b8=v11）
- **结论：v11 的 deployment gate 真基线 = 上述数字（q 26.0 mrad / fk 12.0 mm / vx −0.284）。eval 确定性成立（跨 11 天、跨进程逐位复现）。**

## 2. 正典目录内 deployment_eval.json 是陈旧工件（重要更正）

- `bdx_planner_v2combo/deployment_eval.json`（mtime 08-24T15:11）与复现不一致（q 22.5 / fk 8.4 / vx slope **0.8049** R² 0.7866 sign 1.0）。
- rows 对比：陈旧件仅 **20 行**（vx n=**5**），复现=**414 行**（vx n=231）——陈旧件来自旧版脚本的小筛选子集（canonical.json 里 "vx validated true / realization 0.57-0.65" 的叙述即基于该 n=5 证据）。
- 权威裁定：以 closure_status.json headline + teacher_realization.json（带 provenance sha）为准；陈旧 json 保留原样不动（历史文件禁改），在本报告中标记 **SUPERSEDED**。

## 3. vx 之谜的初步诊断证据（P2 输入，非结论）

teacher_realization.json 三层表（v11，n 同上）：

| 通道 | teacher_vs_requested | planner_vs_requested | planner_vs_teacher |
|---|---|---|---|
| vx | **slope −0.173 / R² 0.053** | −0.284 / 0.113 | **1.077 / R² 0.921** |
| vy | 0.648 / 0.985 | 0.656 / 0.937 | 1.014 / 0.561 |
| vyaw | 0.539 / 0.684 | 0.537 / 0.677 | 0.996 / 0.988 |
| neck_yaw | 0.996 / 1.000 | 1.028 / 0.963 | 1.032 / 0.963 |

- **teacher 本人在该 val 切片上 vx R²=0.05**——任何蒸馏该 teacher 的学生都不可能在同一切片过 "R²≥0.60" 门。vx 失败主要是**该切片的命令覆盖/评测仪器问题**，不是学生退化。
- planner_vs_teacher 四通道全部 ≈1.0（R² 0.92–0.99）：**v11 是忠实蒸馏**。
- PoC 时代的 0.82–0.84 出自**命令扫描探针**（probe/sweep 工具），与 deployment-val 被动回归是两种仪器。P2 需以扫描探针（或覆盖门控回归）为增益仪器重测 v11，并检查 p29_eval（v11 自身，backward |Δvx|=0.038）。

## 4. eval_p2.py 修复（工作区改动，commit 延至 P1）

三处上游缺陷（v11 之前该脚本对任何 checkpoint 都跑不通）：
1. `TEACHER=None` 陈旧变量直接喂 `torch.load` → 改为 `resolve_canonical(OUT_DIR)["teacher"]`（registry 重构未尽项）。
2. `CommandedEncoder(latent_dim=…)` 缺 `cmd_dim` → 按 config 分支：`loco_cmd=True → cmd_dim=7`，否则 4；SparseEncoder 分支不带该参（其构造器无 cmd_dim，防崩）。
3. cmd 张量未按 `train_p2.build_cmd` 契约拼 7 维 → 补 `cat([cmd(B,W,4), loco(B,3)])`（顺序=[neck×4, vx, vy, vyaw]，与冻结接口一致）。

修复后 v11 补全形态（P2Dataset val，4000 窗）：
```
overall joint MAE student 63.3 / teacher 20.3 mrad (ratio 3.12)
gesture 53.0/30.8(1.72)  locomotion 55.6/20.1(2.77)  stand 95.9/15.9(6.04)
transition 49.2/15.0(3.27)  walk_head 67.5/31.4(2.15)  |  latent R² 0.805 (64/64>0.5)
```
（注意：这是补全形态口径，与 deployment 口径 q 26.0 不同源；stand 6.04× 为最差切片，P5 前观察项。）

## 5. G3 基线锚点

- deployment 口径 q 25.98 mrad（目标 ≤30 ✓ 基线已达标）；planner/teacher 增益比 ≈1.0（目标 ≤1.15 ✓）。
- 但 G3 的 "val q" 判据用哪个口径（P2Dataset 补全 63.3 vs deployment 26.0）需在 P5 终评时明确写清——两口径并列披露。
