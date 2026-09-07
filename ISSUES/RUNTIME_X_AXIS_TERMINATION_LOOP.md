# RUNTIME_X_AXIS_TERMINATION_LOOP

```text
ID:       RUNTIME_X_AXIS_TERMINATION_LOOP
Status:   REGISTERED (open; NOT scheduled — no fix task auto-opened)
Opened:   2026-09-07 (user terminal confirmation of DISTILL_CANONICAL_ADOPTION_V12)
Severity: blocks ALL closed-loop x-axis capability certification
```

## Summary

The official deployment runtime (`closed_loop_test.py` config family — the only
config changes are `max_episode_length_s=10000 / headless / simulator=mujoco`)
enters an auto-reset / teleport limit cycle: `episode_length_buf` wraps every
~22–30 steps and each wrap teleports the robot +0.10–0.18 m along +x. Net
x-displacement is pinned to ~0 **regardless of command sign** — a forward
control trial (vx=+0.30, identical harness/adapter) also nets ≈ 0.

Consequence: the chain planner → motion injection → BFM tracker → MuJoCo
**cannot serve as a valid acceptance instrument for x-axis closed-loop
capability** in its current configuration. `RUNTIME_X_AXIS_INSTRUMENT = INVALID`.

## Evidence

- `humanoidverse/data/bdx_planner_distill/20260907T0100Z_v12smoke/` — 12 frozen
  backward trials: 11–14 teleports per 10 s command segment, path 4.9–6.6 m vs
  net ≈ 0; steady decomposition in DISPOSITION.json.
- `humanoidverse/data/bdx_planner_distill/20260907T0130Z_diagFwd/` — labeled
  forward-control diagnostic (vx=+0.30): net steady +0.005 m/s.
- `humanoidverse/data/bdx_planner_distill/20260905T0400Z_p54_final/v12/deployment_eval.json`
  — official eval: vx slope −0.252 / R² 0.091 (realized ≈ 0.003–0.007 regardless
  of command) while vy R² 0.948 / neck R² 0.959 stay healthy → x-specific.
- `humanoidverse/data/bdx_planner_distill/20260904T2300Z_p41_matrix/
  INSTRUMENT_VALIDITY_MATRIX_addendum_20260907.json` — formal matrix entry.

## Ownership (user ruling 2026-09-07, verbatim boundary)

Belongs to:
- tracker/runtime integration;
- termination/reset mechanism;
- motion injection timing and state continuity.

Does NOT belong to:
- v12 distillation training failure;
- planner model degradation;
- backward data-repair failure.

## Disposition

Independent runtime-fix task (to be opened ONLY by explicit user decision).
After a fix, re-run the ORIGINAL frozen smoke plan with **symmetric forward AND
backward trials** — backward-only re-testing is not acceptable (user ruling).
Until then: adapter v1.3 rejects negative vx (conservative product decision);
positive vx stays open as legacy-compatible with scope disclosure
("completion-form validated / legacy-compatible / closed-loop not established
in current runtime") and must never be represented as "closed-loop verified".
