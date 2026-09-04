"""Record a physically-consistent full-body motion by re-rolling the validated
open-loop PD protocol on the ORIGINAL BFM-zero stack (.venv), for use as the
mjlab motion-imitation reference.

Why a re-roll instead of the pkl's own root tracks:
  The pkl exports carry the known root_rot P0 defect (pure yaw, roll/pitch
  stripped). The open-loop PD rollout of `re_dancegen_side_step_4` was already
  validated 292/292 frames without falling (fds_ss4_phase_scan report, variant
  A, seed 47065), so re-running it and recording the SIMULATOR's own qpos/qvel/
  body poses yields a self-consistent motion: a perfect tracker reproduces it
  exactly.

Protocol (VERBATIM from fds_ss4_phase_scan.py openloop_pd, variant A):
  inject frame 0 (settle_reps=2), then for t in 0..T-1:
      action = clamp(kp*(q_ref[t]-default)/(1.25*effort), -1, 1)
      wrapped.step(action)
  Recording: motion frame 0 = state right after injection (the reference
  frame-0 state); motion frame t+1 = state after the step commanded with
  q_ref[t] (the old gate compared that state against q_ref[t+1]).

Outputs (never overwrite): results/mjlab_bdx/ss4_openloop_rollout.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BFM_ZERO_HARD_LIMIT_RESET", "0")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

ENV_CFG = ROOT / "results/bfm-dynamics-baseline/config.json"
FPS = 50.0

# Reproduces the validated variant-A run (phase_scan: seed_base_single 40000
# + 7000 + ord('A') = 47065).
SEED = 47065


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(ROOT / "humanoidverse/data/fds_re_dancegen_side_step_4.pkl"))
    ap.add_argument("--out", default=str(ROOT / "results/mjlab_bdx/ss4_openloop_rollout.npz"))
    args = ap.parse_args()
    pkl_path = Path(args.pkl).resolve()
    out_path = Path(args.out).resolve()
    assert not out_path.exists(), f"refusing to overwrite {out_path}"

    os.environ["BFM_POOL_DATA"] = str(pkl_path)
    from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env, inject_read_exact_lean

    cfg = json.loads(ENV_CFG.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped = build_cpu_env(cfg, device)
    inner = wrapped._env
    dev = inner.device

    data = joblib.load(pkl_path)
    assert len(data) == 1, list(data.keys())
    clip_name, entry = next(iter(data.items()))
    q_ref = torch.as_tensor(np.asarray(entry["dof"]), dtype=torch.float32, device=dev)
    T = q_ref.shape[0]
    assert abs(float(entry["fps"]) - FPS) < 1e-6

    rcfg = yaml.safe_load(open(ROOT / "humanoidverse/config/robot/bdx/bdx_14dof.yaml"))["robot"]
    joint_names = list(rcfg["dof_names"])
    body_names = list(inner.body_names)
    effort = torch.tensor(rcfg["dof_effort_limit_list"], device=dev, dtype=torch.float32)
    kp = inner.p_gains.flatten().float().to(dev)
    kd = inner.d_gains.flatten().float().to(dev)
    default_flat = inner.default_dof_pos.flatten().float().to(dev)
    off = getattr(inner, "default_dof_pos_offset", None)
    if off is not None and torch.is_tensor(off):
        default_flat = default_flat + off.flatten().float().to(dev)

    inner.is_evaluating = True  # pushes OFF — same as the validated run

    def snapshot(t_label: int):
        sim = inner.simulator
        qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()      # (21,) freejoint7+dof14
        qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()      # (20,)
        root = sim.robot_root_states[0].detach().cpu().numpy().copy()  # (13,) pos+quat(xyzw)+vels
        bpos = sim._rigid_body_pos[0].detach().cpu().numpy().copy()    # (17,3) body_names order
        bquat = sim._rigid_body_rot[0].detach().cpu().numpy().copy()   # (17,4) xyzw (repo convention)
        cforce = sim.contact_forces[0].detach().cpu().numpy().copy()   # (17,3)
        dof_pos = sim.dof_pos[0].detach().cpu().numpy().copy()
        dof_vel = sim.dof_vel[0].detach().cpu().numpy().copy()
        # World-frame 6D velocity at each body origin via the unambiguous
        # mj_objectVelocity primitive (cvel semantics avoided deliberately).
        nb = len(body_names)
        bvel6 = np.zeros((nb, 6), dtype=np.float64)
        res = np.zeros(6, dtype=np.float64)
        for i in range(nb):
            mujoco.mj_objectVelocity(
                sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY,
                int(sim.body_id[i]), res, 0)
            bvel6[i] = res  # [ang(3), lin(3)] world axes at body origin
        return dict(t=t_label, qpos=qpos, qvel=qvel, root=root, bpos=bpos, bquat=bquat,
                    cforce=cforce, dof_pos=dof_pos, dof_vel=dof_vel,
                    body_ang_vel=bvel6[:, :3].copy(), body_lin_vel=bvel6[:, 3:].copy())

    seed_all(SEED)
    torch.manual_seed(SEED)
    ids = torch.zeros(1, dtype=torch.long, device=dev)
    times = torch.full((1,), 0.0, device=dev)
    inject_read_exact_lean(wrapped, ids, times, settle_reps=2)

    frames = [snapshot(0)]
    terminated = False
    per_err = []
    for t in range(T):
        raw = (kp * (q_ref[t] - default_flat) / (1.25 * effort))
        action = raw.clamp(-1.0, 1.0)
        obs, _, term, trunc, _ = wrapped.step(action.unsqueeze(0), to_numpy=False)
        nxt = min(t + 1, T - 1)
        err = (inner.simulator.dof_pos[0].float() - q_ref[nxt]).abs()
        per_err.append(float(err.mean()))
        frames.append(snapshot(t + 1))
        if bool(term.any() or trunc.any()):
            terminated = True
            break

    Tn = len(frames)
    pack = {
        "clip_name": clip_name,
        "source_pkl": str(pkl_path),
        "seed": SEED,
        "protocol": ("fds_ss4_phase_scan.openloop_pd variant A: inject frame0, "
                     "action=clamp(kp*(q_ref[t]-default)/(1.25*effort),+-1)"),
        "joint_names": np.array(joint_names),
        "body_names": np.array(body_names),
        "kp": kp.cpu().numpy(), "kd": kd.cpu().numpy(),
        "effort": effort.cpu().numpy(),
        "default_dof_pos": default_flat.cpu().numpy(),
        "q_ref": q_ref.cpu().numpy(),
        "fps": np.float64(FPS),
        "terminated_early": np.bool_(terminated),
        "frames_played": np.int64(Tn - 1),
        "mean_abs_err": np.float64(np.mean(per_err)),
        "per_frame_mean_err": np.asarray(per_err),
        "qpos": np.stack([f["qpos"] for f in frames]),          # (Tn,21)
        "qvel": np.stack([f["qvel"] for f in frames]),          # (Tn,20)
        "root_states": np.stack([f["root"] for f in frames]),   # (Tn,13)
        "body_pos": np.stack([f["bpos"] for f in frames]),      # (Tn,17,3)
        "body_quat": np.stack([f["bquat"] for f in frames]),    # (Tn,17,4)
        "body_lin_vel": np.stack([f["body_lin_vel"] for f in frames]),  # (Tn,17,3) world
        "body_ang_vel": np.stack([f["body_ang_vel"] for f in frames]),  # (Tn,17,3) world
        "contact_forces": np.stack([f["cforce"] for f in frames]),  # (Tn,17,3)
        "dof_pos": np.stack([f["dof_pos"] for f in frames]),    # (Tn,14)
        "dof_vel": np.stack([f["dof_vel"] for f in frames]),    # (Tn,14)
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **pack)

    summary = {
        "out": str(out_path),
        "clip": clip_name,
        "frames_recorded": Tn,
        "expected_frames": T + 1,
        "terminated_early": terminated,
        "openloop_mean_abs_err": float(np.mean(per_err)),
        "steady_last32_median_err": float(np.median(per_err[-32:])),
        "root_z_min": float(np.min([f["qpos"][2] for f in frames])),
        "root_z_range": [float(np.min([f["qpos"][2] for f in frames])),
                         float(np.max([f["qpos"][2] for f in frames]))],
        "seed": SEED,
        "recorded_at": pack["recorded_at"],
    }
    print(json.dumps(summary, indent=2))
    if terminated or Tn != T + 1:
        raise SystemExit("VALIDATION FAILED: rollout did not complete T frames")


if __name__ == "__main__":
    main()
