# Crafting a Torch-Free Deployment Runtime for BFM-Zero — Key Points

Notes from building `humanoidverse/scripts/onnx_runtime_deploy.py` (numpy +
onnxruntime + mujoco only, no PyTorch), validated bit-exact against the torch
training-env path on the G1 750M checkpoint (299-step rollout, root height
matching to ~1e-3) and reused for BDX 14-DOF.

The runtime replaces three things the training env does implicitly:
**observation assembly**, **the ONNX actor + z conditioning**, and **the
physics/torque loop**. The actor is the easy part — every real bug was in the
other two.

---

## 1. Methodology: diff against a ground-truth dump, not against theory

The single most valuable tool was an instrumented *hybrid* runner
(`onnx_inference.py`) that uses the env's torch machinery for obs/physics but
the ONNX actor for actions, with an `OBS_DUMP=<npz>` env var recording per
step: `qpos, qvel, state, last_action, history, action, torque`.

Then debug by levels, cheapest first:

1. **Actor replay** — feed the dump's recorded obs to ONNX, compare to the
   recorded action. (Matched to 6e-6 immediately → actor export is fine,
   suspect everything else.) Also perturb the obs by ±0.01 to measure
   sensitivity; if the policy is not noise-sensitive, small obs errors are
   survivable and the bug is elsewhere.
2. **State→obs replay** — recompute `state` from the dump's own `qpos/qvel`
   and compare to the dump's `state`. This validates obs semantics *in
   isolation*, before physics enters the picture.
3. **One-step physics replay** — start from `qpos[i]`, apply the dump's
   `action[i]`, integrate one control step, compare to `qpos[i+1]`.
4. **Free-running replay** — same, but don't re-anchor each step; watch error
   growth. Linear slow growth = float chaos; immediate growth = systematic.

Two gotchas that cost real time:
- **Dumps record at a specific point in the loop.** Our dump's `torque[i]`
  was the *stale ctrl from the previous step's last substep* — comparing
  torques directly against a freshly computed law was meaningless and sent
  the investigation down a wrong path. Verify what each recorded field
  actually corresponds to before diffing it.
- **Align frames explicitly.** The dump's frame-0 obs is a warm-up transient
  that does not even match its own `qpos[0]` (its ang-vel segment differed by
  0.44 — one substep of drift). Start replay at frame 1, prime history and
  `last_action` from the dump's frame-1 values verbatim, and offset the z
  schedule by the same amount. "First divergence at frame 0" almost always
  means a framing/alignment bug, not a physics bug.

## 2. Observation semantics (env-specific, must be derived empirically)

- **Layout**: `actor_obs = [state (2d+6), last_action (d), history]`, history
  = per-key rolling buffers, **newest first**, concatenated in **sorted-key
  order** (`actions, base_ang_vel, dof_pos, dof_vel, projected_gravity`), 4
  slots each, per-key dims differ (d, 3, d, d, 3) — don't reshape uniformly.
- **Quaternion convention**: MuJoCo `qpos[3:7]` is **wxyz**. Convert to xyzw
  (or index accordingly) before building the rotation for
  `projected_gravity = Rᵀ[0,0,−1]`. A wrong convention shows up as a large
  pg-z error and *exact* matches in every other segment — check segments
  separately (dof_pos / dof_vel / pg / ang_vel) to localize.
- **`qvel[3:6]` is already body-local** (mujoco free-joint convention). The
  env's world→body round-trip cancels; do NOT rotate it yourself.
- **Obs scales**: `base_ang_vel` × 0.25; everything else raw. The ONNX graph
  has the normalizer baked in, so feed raw (scaled) values.
- **`last_action` and history actions are stored AFTER env rescale**
  (`normalize_action` → ×`to/from`, e.g. ×5 for G1), not the raw policy
  output; and history lags differ: obs slots lag 1 step, action slots lag 2.
- **History push ordering**: action is added to history BEFORE the physics
  step; the observation when read. Mirror the env's wrapper ordering exactly.

## 3. Physics loop — where all the real bugs were

The torque law as written in the config docs is **incomplete**. The actual
chain (`legged_robot_base.py`):

```
a      = clip(raw_action * normalize_to/from, ±clip_value)   # _pre_physics_step
asc    = a * action_scale                                     # 0.25
asc   *= tau_lim / kp          # ONLY if control.action_rescale: True  ← easy to miss
tau    = clip(kp*(asc + default_dof_pos - q) - kd*v, ±tau_lim)
```

computed **every substep** on the evolving state (not zero-order-hold), with
`decimation` (4) substeps per control step.

Three separate seams, each individually enough to make the policy fall:

1. **Timestep override.** The env sets `model.opt.timestep = 1/sim_fps`
   (0.005 s) *after* loading, ignoring the XML's 0.002. With decimation 4,
   loading the XML verbatim gives 0.008 s per control step instead of
   0.02 s — a 2.5× speed error. **Never trust the XML timestep; replicate
   whatever the env does to `opt`.**
2. **`action_rescale`** (`asc *= tau_lim/kp`). A config flag, not part of the
   documented P-law. G1 has it on. Grep the *code path*, not the docs.
3. **Effort limits come from the raw yaml list.** `dof_effort_limit_scale`
   exists in the yaml but is **dead** on the mujoco path — the env reads the
   list verbatim. Applying the "obvious" 0.8× scale silently saturates
   torques 11% early and destabilizes the gait with no visible error
   anywhere.

Also: `kp/kd` expand by **substring match over joint-name fragments with
last-match-wins** — replicate the same expansion (and same failure modes:
bare `hip:` fragments, `ankle_pitch` vs `ankle`, etc.) rather than writing
per-joint gains by hand.

**Actuator layout**: G1 has 6 virtual base actuators before the joint motors
(`ctrl_offset = nu − action_dim` = 6); BDX has none (offset 0). Derive from
`model.nu`, never hardcode.

## 4. Domain randomization and noise

Inference-time defaults matter:
- The hybrid/inference path may leave **domain randomization on** by default
  — random mass/friction/gain scales per run make the "ground truth"
  nondeterministic and produce plausible-but-unexplainable ~0.1 rad
  mismatches. **Always regenerate dumps with DR off** (`--disable_dr`) before
  diffing; also disable obs noise.
- The runtime itself should apply none of this.

## 5. Init / warm-up transient

The env's inference path does a zero-action warm-up step before the policy
loop, and the first recorded obs doesn't cleanly correspond to any state you
can reconstruct. Two options:
- Prime from a dump (adopt its history/last_action at frame 1 verbatim, skip
  your own warm-up — replaying a warm-up on top double-steps the state), or
- Replicate the warm-up exactly (4 substeps of zero-action PD from the init
  pose) when no dump is available — good enough for standalone deployment,
  but expect frame-0 obs differences when diffing.

## 6. General principles

- **The env code is the spec.** Every constant in the runtime was taken from
  the env's source or the robot yaml, never from memory or documentation.
  When docs and code disagree (timestep, torque limits, action scaling), the
  code wins — verify by replay.
- **Validate in layers with exact tolerances** (we got 4e-7 on state replay
  and ~1e-3 on free-running physics). "Roughly matches" at any layer means
  the bug is at that layer.
- **Segment your diffs.** A single max-error over a 465-dim obs hides which
  component is wrong; split by obs segment and by joint to localize in one
  step.
- **Isolate variables when comparing rollouts**: same init, same z schedule,
  same frame alignment, DR off, noise off. Any free variable left different
  will masquerade as the bug.
- Chaos is real: after ~100 frames, actions drift from float-level
  divergence in a contact-rich rollout even when everything is correct. Judge
  correctness by whether both trajectories remain behaviorally equivalent
  (standing vs falling), not by exact frame-level equality at long horizons.
