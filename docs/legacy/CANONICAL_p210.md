# Canonical Planner: P2.10 (frozen 2026-08-21)

## Checkpoints
- planner: p2_student_p210.pt (sha256 see provenance.json)
- teacher: p1_ae_l64.pt (A-group frozen decoder, also used by B — clean control)

## Training recipe
- graft-neck 0.5 / full-graft 0.25 (decoupled — the vx-disentanglement lever)
- explicit-cmd + loco-cmd (command replaces future trajectory)
- 8 epochs, seed 0, max-steps 3000, workers 2
- train pool: planner_v1(7531) + v5 combos(2695) = 10226 clips

## Evaluation snapshot (SCALE bug fixed 2026-08-21)
- strict-unseen combos: vx degradation ×1.37 / Δneck same-dir 0.251 /
  Δvyaw 0.018 / q unchanged — compositional generalization largely solved
- 7/7 channels in-distribution gains 0.96-0.98
- command interface STABLE for VLA: 7D [neck×4, vx, vy, vyaw] unchanged

## NOT canonical
- p210B (patch variant): neck 0.235 marginal gain, keep as ablation record
- p29c/p29e/v1a: earlier generations

## Next retrain trigger (support-set change, NOT date/count)
Any 2-3 of: true backward / substantial strafe / teacher full neck execution /
transition matrix / recovery data / gesture+skill interface
