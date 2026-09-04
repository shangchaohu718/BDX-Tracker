"""Closed-loop integration test (MPC-style deployment loop) — the real system:
every second, plan from the SIM's true state, inject the reference into the
motion lib at runtime (zero repo-file modifications), re-extract z, and let
the tracker execute. Validates: no base-height sink (vs open-loop AR), 2-min
stability, command realization, tracking error vs generated reference.

Mechanics:
  - dummy 6000-frame motion pkl preloaded (content overwritten window by window)
  - per window w: sim qpos history -> sparse (FK) -> P2.9c planner -> world
    frame -> write gts/grs/gvs/gavs/dvs/dof_pos (+_t, grvs/gravs) slices at
    [w*50, (w+1)*50) -> get_backward_observation (pure fn) sliced to window
    -> backward_map -> act 50 steps, collect sim state.

Run: BDX_PLANNER_DATA=.../bdx_planner_v2 .venv/bin/python \
  humanoidverse/scripts/closed_loop_test.py [--windows 120]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO.parents[0]))

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, WINDOW,
                                           canonical_sparse, load_stats,
                                           sixd_to_mat)
from humanoidverse.planner.fk import BDXFK, rotmat_to_rotvec
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    REPO / "data" / "bdx_planner"))
MODEL_FOLDER = REPO.parent / "results" / "bfmzero-bdx-full"
N_WINDOWS_DEF = 120

# command program: (n_windows, vx, vyaw, neck_yaw)
PROGRAM = [(30, 0.30, 0.0, 0.0), (30, 0.30, 0.3, 0.0),
           (30, 0.30, 0.0, 0.7), (30, 0.15, 0.0, 0.0)]


def build_dummy_pkl(path, frames=6000):
    """Loop a GT walk clip to fill a long dummy motion (overwritten at runtime)."""
    md = MotionData()
    name = next(n for n in md.split_names["train"]
                if n.startswith("v1_walk/")
                and int(md.d["lengths"][md.name2clip[n]]) >= 1000)
    T0 = 250   # skip the clip's startup transient — seed from steady walking
    f = md.frames(md.name2clip[name], T0,
                  int(md.d["lengths"][md.name2clip[name]]) - T0)
    reps = int(np.ceil(frames / len(f["q"])))
    print(f"dummy motion: {name} x{reps} -> {frames} frames")
    def loop(x):
        return np.concatenate([x] * reps, 0)[:frames]
    pose_aa = np.zeros((frames, 17, 3), np.float32)
    fk = BDXFK()
    # dummy pose_aa from q (verified formula); root from stored rotvec
    pose_aa[:, 0] = sRot.from_quat(
        np.roll(loop(f["base_quat"]), -1, axis=-1)).as_rotvec()
    for b in range(1, 17):
        if fk.qadr[b] >= 0:
            pose_aa[:, b] = fk.axis[b] * (
                loop(f["q"])[:, fk.qadr[b]] - fk.ref[b])[:, None]
    joblib.dump({"closedloop_dummy": {
        "root_trans_offset": loop(f["base_pos"]).astype(np.float32),
        "pose_aa": pose_aa, "dof": loop(f["q"]).astype(np.float32),
        "root_rot": loop(f["base_quat"]).astype(np.float32), "fps": 50}},
        path, compress=3)
    return path


def quat_wxyz_from_R(R):
    return np.roll(sRot.from_matrix(R).as_quat(), 1, axis=-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=N_WINDOWS_DEF)
    ap.add_argument("--checkpoint", default=None,
                    help="default: canonical registry")
    ap.add_argument("--gt-mode", action="store_true",
                    help="skip planner+injection: track the preloaded GT walk "
                         "(discriminates tracker weakness vs planner chain)")
    args = ap.parse_args()
    device = "cuda"

    # ---- planner stack ----
    from humanoidverse.planner.registry import resolve_canonical
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(resolve_canonical(OUT_DIR)["teacher"], map_location=device,
                                       weights_only=False)["model"])
    planner = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    from humanoidverse.planner.registry import resolve_canonical
    if args.checkpoint is None:
        args.checkpoint = str(resolve_canonical(OUT_DIR)["planner"])
    print(f"canonical: {args.checkpoint}")
    planner.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                       weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v).to(device) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    fk = BDXFK().to(device)
    spm, sps = sp["mean"], sp["std"]
    mean, std = stats["mean"].to(device), stats["std"].to(device)

    @torch.no_grad()
    def plan(f_hist, psi, cmd7):
        sh = canonical_sparse(f_hist, psi)
        c = planner(((sh.to(device) - spm) / sps).unsqueeze(0),
                    torch.zeros(1, WINDOW, sh.shape[-1], device=device),
                    cmd7.unsqueeze(0).to(device))
        return teacher.decode(c)[0][0] * std + mean

    def fk_all(base_pos, base_R, q):
        """(W,3),(W,3,3),(W,14) -> body pos/rot/linvel/angvel tensors on device"""
        pos, R = fk.forward(base_pos, base_R, q, return_rot=True)  # (W,17,3/3,3)
        return pos, R

    def vel_from(pos, R):
        lv = torch.zeros_like(pos)
        lv[1:] = (pos[1:] - pos[:-1]) * 50
        av = torch.zeros_like(pos)
        rv = rotmat_to_rotvec(R[1:] @ R[:-1].transpose(-1, -2)) * 50
        av[1:] = rv
        return lv, av

    # ---- env / tracker (mirror tracking_inference setup) ----
    dummy = build_dummy_pkl(OUT_DIR / "closedloop_dummy.pkl",
                            frames=max(6000, args.windows * 50 + 200))
    model = load_model_from_checkpoint_dir(MODEL_FOLDER / "checkpoint", device=device)
    model.to(device).eval()
    config = json.load(open(MODEL_FOLDER / "config.json"))
    use_root_height_obs = config["env"].get("root_height_obs", False)
    config["env"]["lafan_tail_path"] = str(dummy.resolve())
    # the parallel session's config refactor removed some keys referenced by
    # the checkpoint's saved overrides — "+" appends them back instead of failing
    import re as _re
    fixed = []
    for o in config["env"]["hydra_overrides"]:
        fixed.append("+" + o if o.startswith("simulator.config.sim.") and
                     not o.startswith("+") else o)
    config["env"]["hydra_overrides"] = fixed
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append("env.config.headless=True")
    config["env"]["hydra_overrides"].append("simulator=mujoco")
    config["env"]["disable_domain_randomization"] = True
    config["env"]["disable_obs_noise"] = True
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    env.set_is_evaluating(0)
    ml = env._motion_lib
    tensors = {k: getattr(ml, k) for k in
               ("gts", "grs", "gvs", "gavs", "dvs", "dof_pos", "grvs", "gravs")
               if hasattr(ml, k)}
    tensors_t = {k: getattr(ml, k) for k in ("gts_t", "grs_t", "gvs_t", "gavs_t")
                 if hasattr(ml, k)}
    print(f"motion lib tensors: {list(tensors)} + {list(tensors_t)}")

    obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)
    z_full = model.backward_map(obs)
    observation, info = wrapped_env.reset(to_numpy=False)
    # start robot at the reference frame 0 pose
    ref_root = obs_dict["ref_body_pos"][0, 0]
    rr = obs_dict["ref_body_rots"][0, 0].clone()
    dof0 = obs_dict["dof_pos"][0]
    root_state = torch.cat([ref_root, rr, obs_dict["ref_body_vels"][0, 0],
                            obs_dict["ref_body_angular_vels"][0, 0]])
    dst = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0])
    dst[..., 0] = dof0
    observation, _ = env.reset_envs_idx(
        torch.arange(1), target_states={
            "dof_states": dst[None],
            "root_states": root_state[None]})
    # convert to the model-facing obs format (as tracking_inference does)
    observation, _, _, _, _ = wrapped_env.step(
        torch.zeros((1, wrapped_env.action_space.shape[-1]), dtype=torch.float32),
        to_numpy=False)
    observation = wrapped_env._get_g1env_observation(to_numpy=False)

    # sim state history buffer: world (W,21) [pos3, quat4(wxyz), q14]
    def read_sim_qpos():
        qp = env.simulator.data.qpos.copy()          # mujoco sim
        return np.asarray(qp, np.float32)

    hist = []   # list of qpos(21,)
    ref_end_pos = None    # reference's own endpoint (smooth anchor state)
    ref_end_quat = None
    log = {"height": [], "track_err": [], "seg": []}
    import time as _t
    t0 = _t.time()
    for w in range(args.windows):
        # --- command from program
        acc = 0
        for seg_n, vx, vyaw, neck in PROGRAM:
            if acc <= w < acc + seg_n:
                break
            acc += seg_n
        # --- history from TRUE sim state (seed: reference frames 0..50)
        if len(hist) < WINDOW:
            # seed history from the preloaded reference frames 0..50
            # motion-lib quats are xyzw; our buffer is wxyz
            q_xyzw = ml.grs[0:WINDOW, 0].cpu().numpy()
            hist = list(np.concatenate([
                ml.gts[0:WINDOW, 0].cpu().numpy(),
                np.roll(q_xyzw, 1, axis=-1),
                ml.dof_pos[0:WINDOW].cpu().numpy()], -1))
        h = np.stack(hist[-WINDOW:])                  # (50,21) world
        # --- build sparse history from sim state
        bq = torch.from_numpy(h[:, 3:7].astype(np.float32))
        psi = float(torch.atan2(2 * (bq[-1, 0] * bq[-1, 3] + bq[-1, 1] * bq[-1, 2]),
                                1 - 2 * (bq[-1, 2] ** 2 + bq[-1, 3] ** 2)))
        f_hist = {"base_pos": h[:, 0:3], "base_quat": h[:, 3:7],
                  "q": h[:, 7:], "qdot": np.gradient(h[:, 7:], axis=0) * 50}
        pos_h, R_h = fk_all(torch.from_numpy(h[:, 0:3]).to(device),
                            _qwm(bq).to(device), torch.from_numpy(h[:, 7:]).to(device))
        lv_h, av_h = vel_from(pos_h, R_h)
        yaw_f = np.arctan2(R_h[:, [5, 10]].cpu().numpy()[..., 1, 0],
                           R_h[:, [5, 10]].cpu().numpy()[..., 0, 0])
        f_hist.update({
            "base_linvel": lv_h[:, 0].cpu().numpy(),
            "base_angvel": av_h[:, 0].cpu().numpy(),
            "head_rotvec": rotmat_to_rotvec(R_h[:, 12]).cpu().numpy(),
            "head_angvel": av_h[:, 12].cpu().numpy(),
            "foot_pos": pos_h[:, [5, 10]].cpu().numpy().astype(np.float32),
            "foot_yaw": yaw_f.astype(np.float32),
            "contact": np.zeros((WINDOW, 2), np.float32),
            "head_pos": pos_h[:, 12].cpu().numpy()})
        # --- plan (anchored at CURRENT sim base)
        cmd = torch.zeros(WINDOW, 7)
        cmd[:, 0] = torch.from_numpy(h[:, 7:][:, 10].astype(np.float32)) / 1.7
        cmd[:, 1] = torch.from_numpy(h[:, 7:][:, 11].astype(np.float32)) / 1.7
        cmd[:, 2] = neck
        cmd[:, 3] = torch.from_numpy(h[:, 7:][:, 13].astype(np.float32)) / 1.7
        cmd[:, 4] = vx
        cmd[:, 6] = vyaw / 1.5
        f0, f1 = w * WINDOW, w * WINDOW + WINDOW
        if args.gt_mode:
            # GT window straight from the preloaded (steady-walk) dummy
            gts = tensors["gts"][f0:f1].cpu().numpy()
            grs_q = ml.grs[f0:f1].cpu().numpy()      # xyzw
            pos_w = gts[:, :, :1].squeeze(-1) if False else gts[:, 0].copy()
            quat_w = np.roll(grs_q[:, 0], 1, axis=-1)
            q_w = ml.dof_pos[f0:f1].cpu().numpy().copy()
            p = None
        else:
            p = plan(f_hist, psi, cmd)
        c_, s_ = np.cos(psi), np.sin(psi)
        Rz = np.array([[c_, s_, 0.], [-s_, c_, 0.], [0., 0., 1.]], np.float32)
        if p is not None:
            pos_w = (p[:, 28:31].cpu().numpy() @ Rz + h[-1, 0:3]).astype(np.float32)
            R_w = Rz.T[None] @ sixd_to_mat(p[:, 31:37]).cpu().numpy()
            quat_w = quat_wxyz_from_R(R_w).astype(np.float32)
            for i in range(1, len(quat_w)):
                if np.dot(quat_w[i], quat_w[i - 1]) < 0:
                    quat_w[i] = -quat_w[i]
            q_w = p[:, 0:14].cpu().numpy().astype(np.float32)

        if not args.gt_mode:
            # --- inject into motion lib at frames [w*50, w*50+50)
            f0, f1 = w * WINDOW, w * WINDOW + WINDOW
            bp = torch.from_numpy(pos_w[:, 0:3].copy()).to(device)
            bR = torch.from_numpy(R_w).to(device)
            qd = torch.from_numpy(q_w).to(device)
            gts_w, grs_R = fk_all(bp, bR, qd)
            lv, av = vel_from(gts_w, grs_R)
            quat_xyzw = np.roll(quat_w, -1, axis=-1)          # wxyz -> xyzw for lib
            quat17 = torch.from_numpy(
                np.tile(quat_xyzw[:, None, :], (1, 17, 1))).to(device)
            f0, f1 = w * WINDOW, w * WINDOW + WINDOW
            tensors["gts"][f0:f1] = gts_w
            tensors["grs"][f0:f1] = quat17
            tensors["gvs"][f0:f1] = lv
            tensors["gavs"][f0:f1] = av
            tensors["dof_pos"][f0:f1] = qd
            tensors["dvs"][f0:f1] = torch.from_numpy(
                np.gradient(q_w, axis=0) * 50).to(device)
            tensors["grvs"][f0:f1] = lv[:, 0]
            tensors["gravs"][f0:f1] = av[:, 0]
            for k, tk in (("gts_t", "gts"), ("grs_t", "grs"), ("gvs_t", "gvs"),
                          ("gavs_t", "gavs")):
                if k in tensors_t:
                    tensors_t[k][f0:f1] = tensors[tk][f0:f1]

        # --- z for this window (reference-driven, sliced)
        obs_w, _ = get_backward_observation(
            env, 0, use_root_height_obs=use_root_height_obs)
        z_w = model.backward_map({k: v[f0:f1] for k, v in obs_w.items()})
        z_w = model.project_z(z_w)

        # --- execute 50 steps
        seg_d0 = h[-1, 0:2].copy()
        for i in range(WINDOW):
            action = model.act(observation, z_w[i][None], mean=True)
            wrapped_env.step(action, to_numpy=False)
            observation = wrapped_env._get_g1env_observation(to_numpy=False)
            qp = read_sim_qpos()
            hist.append(qp)
            log["height"].append(float(env.simulator._rigid_body_pos[0, 0, 2]))
            log["track_err"].append(float(torch.norm(
                env.dif_global_body_pos[0], dim=-1).mean()))
        hist = hist[-WINDOW:]
        seg_disp = float(np.linalg.norm(qp[0:2] - seg_d0))
        ref_disp = float(np.linalg.norm(pos_w[-1, :2] - pos_w[0, :2]))
        log["seg"].append({"w": w, "cmd_vx": vx, "cmd_vyaw": vyaw,
                           "neck": neck, "disp_m": seg_disp, "ref_disp": ref_disp,
                           "dyaw_deg": float(np.degrees(
                               np.arctan2(2*(qp[3]*qp[6]+qp[4]*qp[5]),
                                          1-2*(qp[5]**2+qp[6]**2))
                               - np.arctan2(2*(h[0,3]*h[0,6]+h[0,4]*h[0,5]),
                                            1-2*(h[0,5]**2+h[0,6]**2)))),
                           "h_mean": float(np.mean(log["height"][-50:]))})
        print(f"w{w:3d} cmd(vx{vx:.2f},wy{vyaw:.1f},nk{neck:.1f}) "
              f"disp {seg_disp:.2f}m ref {ref_disp:.2f}m "
              f"dyaw {log['seg'][-1]['dyaw_deg']:+.0f}° "
              f"h {log['seg'][-1]['h_mean']:.3f} "
              f"trk {np.mean(log['track_err'][-50:])*1000:.0f}mm", flush=True)

    zt = np.array(log["height"]); te = np.array(log["track_err"])
    summary = {"windows": args.windows,
               "height_min/mean/max": [float(zt.min()), float(zt.mean()), float(zt.max())],
               "track_err_mean_mm": float(te.mean() * 1000),
               "segments": log["seg"],
               "elapsed_s": _t.time() - t0}
    (OUT_DIR / "closed_loop_stats.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", json.dumps({k: v for k, v in summary.items()
                                 if k != "segments"}))


def _qwm(bq):
    from humanoidverse.planner.dataset import quat_wxyz_to_mat
    return quat_wxyz_to_mat(bq)


if __name__ == "__main__":
    main()
