"""[BFM-DIAG-NAN] Differential sim-IO driver — run the G1 under one simulator
backend from a fixed state + action tape, recording env-facing IO after *every
physics substep*, then dump to npz for offline comparison against the other
backend.

Usage (run ONCE PER BACKEND in separate processes; the env is a process singleton
and the two GPU sims can't coexist in one process):

    uv run python -m humanoidverse.scripts.diff_sim_io --backend mujoco_warp --out /tmp/io_mjw.npz
    uv run python -m humanoidverse.scripts.diff_sim_io --backend isaacsim     --out /tmp/io_isaac.npz

Design: rather than calling the env's control `step()` (which advances
control_decimation substeps at once and only exposes IO at the 0.02s control
boundary), we replicate `LeggedRobotBase._physics_step` ourselves —
    for _ in range(control_decimation):
        base_env._apply_force_in_physics_step()   # backend-correct actuation
        simulator.simulate_at_each_physics_step() # ONE physics substep
— but record the IO between substeps. Both backends share fps=200 /
control_decimation=4 / substeps=1, so dt=0.005s is aligned and the two advance
substep-for-substep. CRITICAL: _apply_force_in_physics_step dispatches on the
backend name — mujoco_warp uses explicit PD torques (Euler), isaacsim uses
implicit PD (PhysX drive). That actuation difference IS the leading NaN suspect,
so we deliberately let each backend use its OWN real path (no forcing a common
torque), making the diff faithful to production.
"""

import argparse
import os
import sys

import numpy as np
import torch


def build_env(backend: str, num_envs: int):
    """Build a single env for the requested backend, mirroring train.py's
    HumanoidVerseIsaacConfig construction but with DR + obs-noise OFF and a fixed
    upright init (no lie_down penetration) so the two backends start identical."""
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

    # Anchor the motion file ABSOLUTELY: build() rewrites asset_root/assetRoot to
    # absolute (replace "humanoidverse" with HUMANOIDVERSE_DIR) but leaves motion_file
    # relative (lafan_tail_path passed through verbatim). A relative path then breaks
    # if anything shifts CWD before the motion lib globs it. Resolve it here instead.
    import humanoidverse as _hv

    hv_dir = os.path.dirname(_hv.__file__)
    lafan_abs = os.path.join(hv_dir, "data", "lafan_29dof_10s-clipped.pkl")

    # headless + upright init: identical across both backends.
    hydra_overrides = [
        f"simulator={backend}",
        "robot=g1/g1_29dof_hard_waist",
        "robot.control.action_scale=0.25",
        "robot.control.action_clip_value=5.0",
        "robot.control.normalize_action_to=5.0",
        "env.config.lie_down_init=False",  # upright only — no penetrating init state
        "env.config.headless=True",
    ]
    cfg = HumanoidVerseIsaacConfig(
        name="humanoidverse_isaac",
        device="cuda:0",
        lafan_tail_path=lafan_abs,
        enable_cameras=False,
        camera_render_save_dir="isaac_videos",
        max_episode_length_s=None,
        disable_obs_noise=True,  # deterministic reads — strip obs noise
        disable_domain_randomization=True,  # identical dynamics across backends
        relative_config_path="exp/bfm_zero/bfm_zero",
        include_last_action=True,
        hydra_overrides=hydra_overrides,
        context_length=None,
        include_dr_info=False,
        included_dr_obs_names=None,
        include_history_actor=True,
        include_history_noaction=False,
        make_config_g1env_compatible=False,
        root_height_obs=True,
    )
    env, _ = cfg.build(num_envs=num_envs)
    return env


def _snapshot(sim, num_envs: int) -> dict:
    """Capture env-facing IO from the backend. All tensors detached→cpu→numpy.
    Mirrors the reads used by legged_robot_base / the observation builder."""
    with torch.no_grad():
        rec = {}
        root = sim.robot_root_states.detach()  # [N,13] pos(3) quat_xyzw(4) linvel(3) angvel(3)
        rec["robot_root_states"] = root.cpu().numpy()
        rec["base_pos"] = root[:, 0:3].cpu().numpy()
        rec["base_quat_xyzw"] = root[:, 3:7].cpu().numpy()
        rec["base_lin_vel"] = root[:, 7:10].cpu().numpy()
        rec["base_ang_vel"] = root[:, 10:13].cpu().numpy()  # mjw: BODY frame; isaacsim: WORLD frame
        rec["dof_pos"] = sim.dof_pos.detach().cpu().numpy()  # [N,29]
        rec["dof_vel"] = sim.dof_vel.detach().cpu().numpy()  # [N,29]
        rec["contact_forces"] = sim.contact_forces.detach().cpu().numpy()  # [N,B,3]
        rb_pos = sim._rigid_body_pos.detach().cpu().numpy()  # [N,B,3]
        rec["rigid_body_pos"] = rb_pos[:, :6, :]  # first 6 bodies (base + legs) to keep it small
        # finiteness across everything we recorded
        all_vals = np.concatenate([v.reshape(num_envs, -1) for v in rec.values()], axis=1)
        rec["_finite"] = np.isfinite(all_vals).all(axis=1)
    return rec


def _run_replay(env, base_env, sim, args, num_dof):
    """Replay a captured divergence-onset state (from the probe's
    BFM_ZERO_NAN_REPLAY_OUT npz) on this backend: set the exact (qpos, qvel) on
    all envs via the backend's REAL setters, then apply the captured 29-joint
    torque as EFFORT each substep and step. This is the cleanest solver-robustness
    test — identical state + identical applied torque on both backends. If
    mujoco_warp NaNs here while isaacsim survives -> warp solver gap."""
    d = dict(np.load(args.replay, allow_pickle=True))
    eid_src = int(d["env_id"]); src_step = int(d["step_count"])
    qpos = d["pre_qpos"]  # [nq=36] wxyz base quat
    qvel = d["pre_qvel"]  # [nv=35]
    ctrl = d["pre_ctrl"]  # [nu=35]
    src_backend = "mjw"
    print(f"[{args.backend}] REPLAY divergence-onset state (captured env={eid_src} "
          f"step={src_step}, source={src_backend}): nq={qpos.shape} nv={qvel.shape} nu={ctrl.shape}")
    print(f"[{args.backend}]   base_pos={qpos[:3]}  base_quat(wxyz)={qpos[3:7]}")
    print(f"[{args.backend}]   max|dof_pos|={np.abs(qpos[7:]).max():.4f}  max|qvel|={np.abs(qvel).max():.4f}  "
          f"max|ctrl(joints)|={np.abs(ctrl[6:]).max():.4f}")

    N = args.num_envs
    device = base_env.device
    # Build root_states [N,13] = pos(3) quat_xyzw(4) linvel(3) angvel_world(3) from qpos/qvel.
    quat_wxyz = qpos[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    root_row = np.concatenate([qpos[0:3], quat_xyzw, qvel[0:6]])  # [13] (ang vel is world frame in qvel)
    root_states = np.tile(root_row, (N, 1)).astype(np.float32)
    root_t = torch.from_numpy(root_states).to(device)
    # dof_state [N, num_dof, 2] = (pos, vel)
    dof_pos = np.tile(qpos[7:7 + num_dof], (N, 1)).astype(np.float32)
    dof_vel = np.tile(qvel[6:6 + num_dof], (N, 1)).astype(np.float32)
    dof_state = np.stack([dof_pos, dof_vel], axis=-1)
    dof_t = torch.from_numpy(dof_state).to(device)
    # captured joint torques [N, num_dof] (the ctrl[6:] = 29 joints)
    joint_torque = np.tile(ctrl[6:6 + num_dof], (N, 1)).astype(np.float32)
    torque_t = torch.from_numpy(joint_torque).to(device)
    all_ids = torch.arange(N, device=device)

    # Set the exact captured state via the backend's REAL setters (so we exercise the real write path).
    sim.set_actor_root_state_tensor(all_ids, root_t.clone())
    sim.set_dof_state_tensor(all_ids, dof_t.clone())

    # Record the pre-step (just-set) state, then step replay_steps substeps applying the torque.
    records = {"dof_pos": [], "dof_vel": [], "robot_root_states": [], "_finite": []}
    rec0 = _snapshot(sim, N)
    for k, v in rec0.items():
        records[k] = [v]
    print(f"[{args.backend}] replay substep 0 (state-set, finite={rec0['_finite'].all()}): "
          f"max|dof_pos|={np.abs(rec0['dof_pos']).max():.4f} max|dof_vel|={np.abs(rec0['dof_vel']).max():.4f}")

    for s in range(args.replay_steps):
        # Apply the captured joint torque as EFFORT (identical on both backends — the solver test).
        sim.apply_torques_at_dof(torque_t.clone())
        sim.simulate_at_each_physics_step()  # ONE physics substep
        rec = _snapshot(sim, N)
        for k in records:
            records[k].append(rec.get(k, records[k][-1]))
        finite = rec["_finite"].all()
        print(f"[{args.backend}] replay substep {s + 1}: finite={finite} "
              f"max|dof_pos|={np.abs(rec['dof_pos']).max():.4f} "
              f"max|dof_vel|={np.abs(rec['dof_vel']).max():.4f}")
        if not finite:
            bad = np.where(~rec["_finite"])[0]
            print(f"[{args.backend}] !!! {args.backend.upper()} NON-FINITE at replay substep {s + 1} "
                  f"in env(s) {bad.tolist()}")

    out = {k: np.stack(v, axis=0) for k, v in records.items()}
    out["_meta"] = np.array(
        [N, num_dof, int(base_env.config.simulator.config.sim.control_decimation), args.replay_steps, base_env.sim_dt],
        dtype=np.float64,
    )
    np.savez(args.out, **out)
    all_finite = np.stack(records["_finite"], axis=0).all()
    print(f"[{args.backend}] replay wrote {args.out}: finite={bool(all_finite)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["mujoco_warp", "isaacsim"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_control_steps", type=int, default=20, help="control periods to run")
    parser.add_argument("--limit_slam", action="store_true", help="add a joint-limit-slamming action period")
    parser.add_argument(
        "--replay", default=None,
        help="npz from the probe (BFM_ZERO_NAN_REPLAY_OUT): set the captured divergence-onset "
             "(qpos,qvel) on all envs, apply the captured 29-joint torque as effort, step. "
             "Tests whether this backend NaNs on that exact state+torque.",
    )
    parser.add_argument("--replay_steps", type=int, default=8, help="substeps to run in replay")
    args = parser.parse_args()

    # Isaac Sim's teardown can discard buffered stdout, so force a reconfigure to
    # line-buffered output + flush on every print. Lets us watch per-substep NaN.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    torch.manual_seed(0)
    np.random.seed(0)

    env = build_env(args.backend, args.num_envs)
    base_env = env.base_env  # unwrapped LeggedRobotMotions
    sim = base_env.simulator
    decimation = int(base_env.config.simulator.config.sim.control_decimation)
    num_dof = base_env.num_dof

    if args.replay:
        _run_replay(env, base_env, sim, args, num_dof)
        return

    # ---- identical, deterministic reset to the default standing pose ----
    obs, _ = env.reset()  # reset_all with target_states=None -> default init pose (DR off)

    # ---- record substep 0 (post-reset, pre-action) ----
    records = {
        k: [] for k in (
            "robot_root_states", "base_pos", "base_quat_xyzw", "base_lin_vel",
            "base_ang_vel", "dof_pos", "dof_vel", "contact_forces", "rigid_body_pos",
            "_finite", "action", "torques",
        )
    }
    rec0 = _snapshot(sim, args.num_envs)
    for k, v in rec0.items():
        records[k].append(v)
    # no action applied yet at substep 0
    records["action"].append(np.zeros((args.num_envs, num_dof), dtype=np.float32))
    records["torques"].append(np.zeros((args.num_envs, num_dof), dtype=np.float32))
    print(f"[{args.backend}] substep 0 (post-reset): finite={rec0['_finite'].all()}  "
          f"dof_pos[0,:3]={rec0['dof_pos'][0,:3]}")

    # ---- action tape (identical across backends): zeros, then held torque, then optional limit slam ----
    # action_space is [-5,5]^num_dof (normalize_action_to=5). We drive joint-space targets
    # via the env's real PD path, so the SAME action vector reaches both backends identically.
    tape = []
    for s in range(args.num_control_steps):
        if s < 5:
            a = torch.zeros(args.num_envs, num_dof, device=base_env.device)
        elif s < 10:
            # a held, mild offset so the policy-free scenario still exercises PD dynamics
            a = torch.full((args.num_envs, num_dof), 1.0, device=base_env.device)
        else:
            if args.limit_slam:
                # full-scale action: drives every joint toward its limit — the suspected NaN trigger
                a = torch.full((args.num_envs, num_dof), 5.0, device=base_env.device)
            else:
                a = torch.full((args.num_envs, num_dof), -1.0, device=base_env.device)
        tape.append(a)

    substep = 0
    for ctrl in range(args.num_control_steps):
        action = tape[ctrl]
        # Mirror LeggedRobotBase: _pre_physics_step sets self.actions + actions_after_delay
        # (clip, normalize, ctrl-delay) exactly as in production.
        base_env._pre_physics_step(action.clone())
        last_action = base_env.actions_after_delay.detach().cpu().numpy()
        # --- replicate _physics_step's per-substep loop, recording each substep ---
        for _ in range(decimation):
            base_env._apply_force_in_physics_step()  # backend-correct actuation (torque/implicit PD)
            last_torques = base_env.torques.detach().cpu().numpy() if base_env.torques is not None else \
                np.zeros((args.num_envs, num_dof), dtype=np.float32)
            sim.simulate_at_each_physics_step()  # ONE physics substep (mjw.step / sim.step)
            substep += 1
            rec = _snapshot(sim, args.num_envs)
            for k, v in rec.items():
                records[k].append(v)
            records["action"].append(last_action)
            records["torques"].append(last_torques)
            finite = rec["_finite"].all()
            print(f"[{args.backend}] ctrl={ctrl} substep={substep} finite={finite} "
                  f"max|dof_pos|={np.abs(rec['dof_pos']).max():.4f} "
                  f"max|dof_vel|={np.abs(rec['dof_vel']).max():.4f} "
                  f"max|ang_vel|={np.abs(rec['base_ang_vel']).max():.4f}")
            if not finite:
                bad = np.where(~rec["_finite"])[0]
                print(f"[{args.backend}] !!! NON-FINITE at substep={substep} in env(s) {bad.tolist()} — "
                      f"continuing to capture the rest of the trajectory")

    # ---- stack + save ----
    out = {}
    for k, v in records.items():
        arr = np.stack(v, axis=0)  # [S, N, ...]
        out[k] = arr
    out["_meta"] = np.array(
        [args.num_envs, num_dof, decimation, substep + 1, base_env.sim_dt],
        dtype=np.float64,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f"[{args.backend}] wrote {args.out}: {substep + 1} substeps, "
          f"{args.num_envs} envs, dt={base_env.sim_dt:.5f}, dof={num_dof}, decimation={decimation}")
    # overall finiteness summary
    all_finite = np.stack(records["_finite"], axis=0).all()
    print(f"[{args.backend}] ALL SUBSTEPS FINITE: {bool(all_finite)}")


if __name__ == "__main__":
    main()