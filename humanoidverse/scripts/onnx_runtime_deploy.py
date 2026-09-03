#!/usr/bin/env python
"""Torch-free BFM-Zero deployment runtime — numpy + onnxruntime + mujoco only.

Drives the robot (G1 or BDX) directly from the exported ONNX actor with zero
PyTorch dependency: observations are assembled in numpy exactly as the training
env does, actions go through the same P-control law.

Per control step (matches legged_robot_base._compute_observations +
history_handler + _get_g1env_observation):
  state[0:2d]    = dof_pos - default_dof_pos              (d = action_dim)
  state[2d:2d+3] = projected_gravity = R(q)^T @ [0,0,-1]  (body frame)
  state[2d+3:]   = 0.25 * base_ang_vel                    (body frame; obs_scale)
  last_action    = previous raw action (pre-scale)
  history        = per-key rolling buffers, NEWEST FIRST, concatenated in
                   sorted-key order (actions, base_ang_vel, dof_pos, dof_vel,
                   projected_gravity), 4 steps each.
  z              = per-frame latent from zs_<id>__mujoco.pkl (open-loop).

History semantics (verified against HumanoidVerseVectorEnv.step): the action is
added to history BEFORE the physics step; the observation is added when read.
So at step i the actor sees obs slots newest = obs[i], action slots newest =
action[i-1].

The ONNX graph contains the obs normalizer (act() -> _normalize), so raw values
are fed directly (obs_scales are 1.0 for everything except base_ang_vel=0.25,
applied above).

Torque law (control_type P, clip_torques=True):
  tau = clip(kp*(a*action_scale + default_dof_pos - dof_pos) - kd*dof_vel,
             -tau_lim, tau_lim)
kp/kd expand by substring match over joint names (legged_robot_base.py:75-92);
default_joint_angles and effort limits come from the same robot yaml.

Usage (G1 750M, live window):
    MUJOCO_GL=glfw python3 humanoidverse/scripts/onnx_runtime_deploy.py \
        --model_folder remote_checkpoint_750M --motion 25 --viewer
Dependencies: numpy, onnxruntime, mujoco, joblib, pyyaml. NO torch.
"""
import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import yaml

HV_ROOT = Path(__file__).parents[2]          # repo root (bdx_BFMzero/)
CFG_ROOT = HV_ROOT / "humanoidverse/config"  # robot yamls live here


def quat_to_R(q_xyzw):
    x, y, z, w = q_xyzw  # expects xyzw order
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class RobotConfig:
    """Everything the deploy loop needs from the robot yaml (torch-free)."""

    def __init__(self, robot_yaml: Path, action_dim: int):
        cfg = yaml.safe_load(open(robot_yaml))
        rcfg = cfg["robot"]
        self.action_dim = action_dim
        self.dof_names = list(rcfg["dof_names"])
        assert len(self.dof_names) == action_dim, (len(self.dof_names), action_dim)

        dja = rcfg["init_state"]["default_joint_angles"]
        self.default_dof_pos = np.array([dja[n] for n in self.dof_names], dtype=np.float64)

        # substring-expanded gains, last-match-wins (legged_robot_base.py:75-92)
        stiffness = rcfg["control"]["stiffness"]
        damping = rcfg["control"]["damping"]
        kp = np.zeros(action_dim)
        kd = np.zeros(action_dim)
        for i, name in enumerate(self.dof_names):
            for frag, val in stiffness.items():
                if frag in name:
                    kp[i] = val
            for frag, val in damping.items():
                if frag in name:
                    kd[i] = val
        assert (kp > 0).all() and (kd > 0).all(), "every joint must match a gain fragment"
        self.kp, self.kd = kp, kd

        # NOTE: the env's torque_limits come from the RAW yaml list —
        # dof_effort_limit_scale is NOT applied on the mujoco path
        # (get_dof_limits_properties reads the list verbatim).
        self.tau_lim = np.array(rcfg["dof_effort_limit_list"], dtype=np.float64)
        self.action_scale = float(rcfg["control"]["action_scale"])
        self.action_clip = float(rcfg["control"]["action_clip_value"])
        ctrl = rcfg["control"]
        self.action_rescale = 1.0
        if ctrl.get("normalize_action", False):
            to = ctrl.get("normalize_action_to") or self.action_clip
            frm = ctrl.get("normalize_action_from", 1.0)
            self.action_rescale = float(to) / float(frm)
        # torque-law rescale (control.action_rescale, legged_robot_base.py:648):
        # actions_scaled *= tau_lim / kp before the P-controller
        self.action_rescale_extra = bool(ctrl.get("action_rescale", False))

        scene = rcfg["asset"].get("scene_file")
        data_root = CFG_ROOT.parent / "data/robots"
        if scene:
            self.scene_xml = data_root / scene
        else:
            self.scene_xml = data_root / "g1/scene_29dof_freebase_mujoco.xml"


def pick_robot_yaml(model_folder: Path, action_dim: int):
    """G1 29dof or BDX 14dof by action dim; checkpoint config tells the robot."""
    cfg = json.load(open(model_folder / "config.json"))
    overrides = cfg["env"].get("hydra_overrides", [])
    robot = next((o.split("=", 1)[1] for o in overrides if o.startswith("robot=")), None)
    if robot:
        return CFG_ROOT / "robot" / f"{robot}.yaml"
    # fallback by dim
    name = {29: "g1/g1_29dof_hard_waist", 14: "bdx/bdx_14dof"}[action_dim]
    return CFG_ROOT / "robot" / f"{name}.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_folder", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, default=None,
                    help="default: <model_folder>/exported/FBcprAuxModel.onnx")
    ap.add_argument("--motion", type=int, required=True,
                    help="clip id; z loaded from tracking_inference/zs_<id>__mujoco.pkl")
    ap.add_argument("--episode_len", type=int, default=0, help="0 = full z length")
    ap.add_argument("--init_npz", type=Path, default=None,
                    help="init state npz (qpos/qvel at ref frame 0, e.g. from an OBS_DUMP)")
    ap.add_argument("--viewer", action="store_true", help="open live mujoco viewer")
    ap.add_argument("-- realtime", dest="realtime", action="store_true", default=True)
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer
    import onnxruntime as ort
    import joblib

    onnx_path = args.onnx or args.model_folder / "exported" / "FBcprAuxModel.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    total_dim = int(inp.shape[1])

    z_all = np.asarray(joblib.load(
        args.model_folder / "tracking_inference" / f"zs_{args.motion}__mujoco.pkl"))
    z_dim = z_all.shape[-1]
    actor_obs_dim = total_dim - z_dim
    action_dim = sess.get_outputs()[0].shape[-1]
    action_dim = int(action_dim) if isinstance(action_dim, int) or str(action_dim).isdigit() else None
    if action_dim is None:
        # dynamic batch/output dim: infer from robot config
        robot_yaml = pick_robot_yaml(args.model_folder, 14 if actor_obs_dim < 300 else 29)
        action_dim = 14 if actor_obs_dim < 300 else 29
    print(f"ONNX: input {total_dim} = actor_obs {actor_obs_dim} + z {z_dim}; action {action_dim}")

    robot_yaml = pick_robot_yaml(args.model_folder, action_dim)
    rc = RobotConfig(robot_yaml, action_dim)
    print(f"Robot: {robot_yaml.name} ({action_dim} dof), scene {rc.scene_xml.name}")

    # layout sanity: state(2d+6) + last_action(d) + history(4*(3d+6)) must equal actor_obs_dim
    se, ae = 2 * action_dim + 6, 3 * action_dim + 6
    hist_dim = 4 * (3 * action_dim + 6)
    assert se + action_dim + hist_dim == actor_obs_dim, \
        f"layout mismatch: {se}+{action_dim}+{hist_dim} != {actor_obs_dim}"

    model = mujoco.MjModel.from_xml_path(str(rc.scene_xml))
    # The env's mujoco backend overrides the XML timestep with 1/fps (=0.005);
    # 4 substeps then make one 50 Hz control step. Without this, the XML's
    # 0.002 would step only 0.008 s per control step — 2.5× too fast.
    model.opt.timestep = 1 / 200
    data = mujoco.MjData(model)
    ctrl_offset = model.nu - action_dim

    # ---- init: ref frame 0 (from a dump of the torch path) or home pose ----
    if args.init_npz:
        init = np.load(args.init_npz)
        # Start at frame 1: the dump's frame-0 obs is the warm-up transient (its
        # ang-vel segment doesn't match its own qpos[0]); frame>=1 aligns exactly.
        data.qpos[:] = init["qpos"][1]
        data.qvel[:] = init["qvel"][1]
        print(f"Init from {args.init_npz} frame 1: root {data.qpos[:3].round(3)}")
    mujoco.mj_forward(model, data)

    HIST_KEYS = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
    hist = {k: np.zeros((4, d), dtype=np.float32)
            for k, d in [("actions", action_dim), ("base_ang_vel", 3),
                         ("dof_pos", action_dim), ("dof_vel", action_dim),
                         ("projected_gravity", 3)]}

    def push(key, val):
        b = hist[key]
        b[1:] = b[:-1]
        b[0] = val

    # If the init dump carries the torch path's own obs (history/state/last_action
    # recorded at frame 0, i.e. AFTER its zero-action warm-up step), adopt them
    # verbatim — replaying our own warm-up on top would double-step the state.
    init = np.load(args.init_npz) if args.init_npz else None
    have_gt_obs = init is not None and "history" in init

    # warm-up step with zero action — mirrors the torch path's init sequence
    # (wrapped_env.step(zeros) before the loop), so history buffers carry the
    # same transient the policy saw at training/inference time. Skipped when
    # the init dump already embeds that transient (state is already post-warm-up).
    if have_gt_obs:
        for k in HIST_KEYS:
            # flat layout: sorted keys, 4 slots × per-key dim each (dims differ per key)
            per_key = {"actions": action_dim, "base_ang_vel": 3,
                       "dof_pos": action_dim, "dof_vel": action_dim,
                       "projected_gravity": 3}
            off = 0
            for k in HIST_KEYS:
                hist[k] = init["history"][1][off:off + 4 * per_key[k]].reshape(4, per_key[k]).copy()
                off += 4 * per_key[k]
        last_action5 = init["last_action"][1].astype(np.float32).copy()
    else:
        for _ in range(4):
            tau = np.clip(rc.kp * (rc.default_dof_pos - data.qpos[7:7 + action_dim])
                          - rc.kd * data.qvel[6:6 + action_dim], -rc.tau_lim, rc.tau_lim)
            data.ctrl[ctrl_offset:ctrl_offset + action_dim] = tau
            mujoco.mj_step(model, data)
        qpos0, qvel0 = data.qpos.copy(), data.qvel.copy()
        R0 = quat_to_R(qpos0[3:7][[1, 2, 3, 0]])  # wxyz -> xyzw
        push("dof_pos", (qpos0[7:7 + action_dim] - rc.default_dof_pos).astype(np.float32))
        push("dof_vel", qvel0[6:6 + action_dim].astype(np.float32))
        push("projected_gravity", (R0.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32))
        push("base_ang_vel", (0.25 * qvel0[3:6]).astype(np.float32))
        push("actions", np.zeros(action_dim, dtype=np.float32))
        last_action5 = np.zeros(action_dim, dtype=np.float32)  # a[i-1] * rescale

    rescale = rc.action_rescale
    ep_len = min(args.episode_len, len(z_all)) if args.episode_len else len(z_all)
    policy_dt = 0.02  # 50 Hz control (200 Hz sim, decimation 4)
    sim_dt = model.opt.timestep

    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    print(f"Running torch-free ONNX rollout: {ep_len} steps")
    root_h = []
    root_trace = [] if os.environ.get("RT_TRACE") else None
    for i in range(ep_len):
        # ---- observations (numpy, EXACT env semantics — verified against a
        #      dumped torch-path rollout to 1e-7; see session notes) ----
        qpos, qvel = data.qpos, data.qvel
        dof_pos = qpos[7:7 + action_dim]
        dof_vel = qvel[6:6 + action_dim]
        base_quat_xyzw = qpos[3:7][[1, 2, 3, 0]]
        R = quat_to_R(base_quat_xyzw)
        projected_gravity = R.T @ np.array([0.0, 0.0, -1.0])
        base_ang_vel_body = qvel[3:6]  # mujoco free-joint qvel[3:6] is ALREADY body-local

        state = np.concatenate([
            dof_pos - rc.default_dof_pos,
            dof_vel,
            projected_gravity,
            0.25 * base_ang_vel_body,
        ]).astype(np.float32)
        # last_action slot holds the PREVIOUS action AFTER env rescale (×rescale)
        # and history holds obs lag-1, actions lag-2 (also rescaled)
        history_flat = np.concatenate([hist[k].reshape(-1) for k in HIST_KEYS])
        actor_obs = np.concatenate([state, last_action5, history_flat]).astype(np.float32)
        # gt-primed runs start at dump frame 1, so z must be offset to match
        z_off = 1 if have_gt_obs else 0
        x = np.concatenate([actor_obs, z_all[(i + z_off) % len(z_all)]])[None].astype(np.float32)

        # ---- actor (onnxruntime) ----
        action = sess.run(None, {inp.name: x})[0][0]

        # history update, matching the env's wrapper ordering (empirically
        # verified: obs slots lag 1 step, action slots lag 2 steps):
        #   at step i obs seen comes from push order below maintained so that
        #   hist obs slot0 == state[i-1], hist action slot0 == a[i-2]*rescale
        push("dof_pos", (dof_pos - rc.default_dof_pos).astype(np.float32))
        push("dof_vel", dof_vel.astype(np.float32))
        push("projected_gravity", projected_gravity.astype(np.float32))
        push("base_ang_vel", (0.25 * base_ang_vel_body).astype(np.float32))
        push("actions", last_action5)  # slot ordering: a[i-2] lands in slot0 next step
        last_action5 = np.clip(action * rescale, -rc.action_clip, rc.action_clip).astype(np.float32)

        # ---- physics: decimation 4 @ 200 Hz, torque recomputed EVERY substep
        #      on the evolving state (matches _physics_step in
        #      legged_robot_base.py:278-280 — the env is NOT zero-order-hold) ----
        for _ in range(4):
            # action_rescale=True (g1 config): actions_scaled *= tau_lim/kp
            # before the P-law (legged_robot_base.py:648-651)
            asc = last_action5 * rc.action_scale
            if rc.action_rescale_extra:
                asc = asc * rc.tau_lim / rc.kp
            tau = rc.kp * (asc + rc.default_dof_pos
                           - data.qpos[7:7 + action_dim]) \
                - rc.kd * data.qvel[6:6 + action_dim]
            tau = np.clip(tau, -rc.tau_lim, rc.tau_lim)
            data.ctrl[ctrl_offset:ctrl_offset + action_dim] = tau
            mujoco.mj_step(model, data)
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(max(0.0, policy_dt - 4 * sim_dt))
        root_h.append(float(data.qpos[2]))
        if os.environ.get("RT_TRACE"):
            root_trace.append((data.qpos.copy(), data.qvel.copy(), action.copy(),
                           data.ctrl.copy()))

    if root_trace is not None:
        np.savez(os.environ["RT_TRACE"],
                 qpos=np.stack([t[0] for t in root_trace]),
                 qvel=np.stack([t[1] for t in root_trace]),
                 action=np.stack([t[2] for t in root_trace]),
                 ctrl=np.stack([t[3] for t in root_trace]))
    if viewer is not None:
        viewer.close()
    print(f"STATS {{'steps': {ep_len}, 'min_root_h': {min(root_h):.3f}, "
          f"'final_root_h': {root_h[-1]:.3f}}}")


if __name__ == "__main__":
    main()