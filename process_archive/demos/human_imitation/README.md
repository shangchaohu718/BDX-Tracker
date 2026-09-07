# Human-imitation closed-loop wiring demo (2026-09-07)

User request: wire existing components only (no new wheels) — human sparse
points → upstream planner (student v12) → z → tracker (bfmzero-bdx-full) →
MuJoCo; qualitatively check whether BDX imitates the human's leg motions.

## Chain (all existing parts)
LAFAN dance clip → G1-MJCF FK → 5 sparse points (pelvis / head / both feet,
world-frame, BDX-scaled s=0.383) → canonical_sparse 32-d → **student v12
(p2_student_v12.pt, canonical 34047cde8f6c5a3b)** → teacher.decode →
absolute-frame motion injection → get_backward_observation → backward_map →
project_z → **tracker bfmzero-bdx-full** → MuJoCo 50 Hz.

Script: `humanoidverse/scripts/human_imitation_closedloop.py`
Clip: `dance1_subject3_clip1` (300 f @ 30 fps → 500 f @ 50 fps).

## Deliverables / SHA256
- `human_imitation.mp4`  2acacc7e028669665354bacdf76fdc01fee4de5e19f7f06dc62a9c09fecb35de
  3-panel: human stick (LAFAN→G1 FK) | planner-planned BDX | tracker-executed.
  1440x360 @ 25 fps, every 2nd sim frame. **Primary evidence.**
- `HUMAN_IMITATION_RESULT.json`  c61ec3e55f191b54e9227113c2039bcdbdbe5b58dfde35e065e17ca983aeaa38
- `adapter_events.jsonl`  dda0081a2efe133eb2303a9028a8072495987e4b31f490c9181fa62058b130a5
  (appended across launches; final-run events are the last 6 lines)
- `runlog/run.log` full console log.

## Honest scope labels (per campaign terminal ruling 2026-09-07)
- Leg imitation = QUALITATIVE only. Probe-frame review: same-direction knee
  bending, robot crouches deeper than the human; tracker never trained on
  dance data; planner sees ONLY 5 sparse points.
- adapter v1.3 stayed in the loop: windows whose human velocity had a small
  negative forward component were rejected → cmd=0 fallback, logged
  (`backward_unsupported_closed_loop_smoke`).
- `RUNTIME_X_AXIS_TERMINATION_LOOP` (ISSUES/) applies to any closed-loop run;
  here only 1 step >5 cm displacement jump in 500 (mostly-stationary clip).
- This is a wiring/diagnostic demo, NOT a certification of closed-loop
  tracking quality. Positive vx stays `ALLOWED_WITH_SCOPE_DISCLOSURE`
  (completion-form validated / legacy-compatible / closed-loop not
  established in current runtime).

## Walk-segment rerun (user request: walking, not single-leg dance)

User ruling 2026-09-07: dance1 contains single-leg balances BDX cannot
replicate; use a plain walking clip (with side-steps). Findings:

1. **LAFAN walk families are heavily mocap-corrupted in this pkl**: most
   walk1-4 clips contain dropout segments (feet fly to 1.3-1.5 m, torso below
   pelvis). Only 14/283 walk clips are physically sane over their full
   length, and none of those travels forward. Full-library scan (862 clips,
   window-level sanity: both feet <0.15 scaled for >90% of window frames):
   114 sane runs, exactly ONE forward-clean segment under the frozen rule
   (vx>0.12, zero negative-vx windows, >=5 windows):
   **fightAndSports1_subject1_clip19 windows 0-5** (5 s, vx +0.26 scaled,
   lateral |vy| 0.043 L / 0.094 R). Vision-check of the human stick frames:
   genuine alternating gait, stance feet near z~0, contact 0.66.
2. Script changes: `END` trim arg (argv[2]); window-MEAN velocity for the
   command (single-frame gradient was noisy → spurious adapter rejections);
   per-clip output filenames.
3. Walk run (250 frames): `imitation_fightAndSports1_subject1_clip19.mp4`
   + `result_fightAndSports1_subject1_clip19.json`.
   - **adapter_events_total = 0** (all commands valid forward).
   - Probe frames: human mid-stride → planned mid-stride → executed
     mid-stride, robot upright — qualitatively the closest match of the two
     runs. Deeper knee bend than the human remains (morphology + planner
     prior).
   - **teleport_steps_gt5cm = 16 in 250 steps**: walking triggers the known
     RUNTIME_X_AXIS_TERMINATION_LOOP much more than the stationary dance
     (env auto-reset teleports the robot; visible as snapping in the video;
     NOT policy failure — fix requires the separate runtime task per ISSUES).
   - Lateral component of the chosen segment is modest; strong left-side
     stepping simply does not exist in the sane-forward pool of this pkl
     (the corruption ate it).
