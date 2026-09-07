"""[human-imitation] Wire the EXISTING stack into a human-imitation closed loop.

User request (2026-09-07): reuse the pieces we already have, no new wheels:
  LAFAN human mocap -> 5 sparse points (pelvis / head / both feet, BDX-scaled)
  -> canonical sparse history -> student v12 planner (canonical) -> teacher
  decode -> motion-lib injection -> backward_map -> z -> tracker
  (bfmzero-bdx-full) -> MuJoCo physics.

This answers: can BDX roughly imitate a person's LEG motion through the
trained sparse-commanded stack? Output = 3-panel side-by-side video
(human stick | planner-planned BDX | tracker-executed robot) + metrics JSON.

Honest labels: DIAGNOSTIC demo, not a gate. The official runtime's
RUNTIME_X_AXIS_TERMINATION_LOOP still applies (teleports disclosed in
metrics). Commands go through command_adapter_v13; negative forward is
REJECTED by the canonical adapter -> falls back to 0 and is logged.

Run: BDX_PLANNER_DATA=<v2combo> .venv/bin/python \
  humanoidverse/scripts/human_imitation_closedloop.py [lafan_clip_name]
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import joblib
import mediapy as media
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO.parents[0]))

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.planner.dataset import (MODEL_DIM, WINDOW,
                                           canonical_sparse, load_stats,
                                           sixd_to_mat)
from humanoidverse.planner.fk import BDXFK, rotmat_to_rotvec
from humanoidverse.planner.model import CommandedEncoder, MotionAE
from humanoidverse.planner.command_adapter_v13 import adapt, CommandValidationError
from humanoidverse.planner.registry import resolve_canonical
from humanoidverse.scripts.smoke_backward_closedloop import build_dummy_pkl

DATA = REPO / "data"
LAFAN = DATA / "lafan_29dof_10s-clipped.pkl"
OUT_DIR = REPO / "data" / "bdx_planner_distill" / "20260907T_humanImitation"
MODEL_FOLDER = REPO.parent / "results" / "bfmzero-bdx-full"
CLIP = sys.argv[1] if len(sys.argv) > 1 else "dance1_subject3_clip1"
# optional end-frame trim: most LAFAN clips contain mocap-dropout garbage
# (feet flying to 1.4m) outside the physically-sane segment chosen by scan
END = int(sys.argv[2]) if len(sys.argv) > 2 else None
FPS = 50
SCALE_TARGET_BASE_Z = 0.30      # BDX walking base height (dataset convention)

# G1-29dof body names (LAFAN retarget skeleton; smpl_joints field is dead)
KEY_BODIES = {
    "pelvis": "pelvis", "torso": "torso_link",
    "lhip": "left_hip_pitch_link", "lknee": "left_knee_link",
    "lank": "left_ankle_roll_link", "rhip": "right_hip_pitch_link",
    "rknee": "right_knee_link", "rank": "right_ankle_roll_link",
    "lsh": "left_shoulder_pitch_link", "lel": "left_elbow_link",
    "lwr": "left_wrist_yaw_link", "rsh": "right_shoulder_pitch_link",
    "rel": "right_elbow_link", "rwr": "right_wrist_yaw_link",
}


def load_human(name):
    """LAFAN-29dof clip -> 50 Hz BDX-scaled sparse fields via G1-MJCF FK."""
    d = joblib.load(LAFAN)
    if name not in d:
        name = [k for k in d if k.startswith(name)][0]
    c = d[name]
    rt = c["root_trans_offset"].astype(np.float64)
    rq = c["root_rot"].astype(np.float64)
    dof = c["dof"].astype(np.float64)
    fps, n = int(c["fps"]), len(rt)

    m = int(round(n * FPS / fps))
    t_src, t_dst = np.arange(n) / fps, np.arange(m) / FPS
    q = rq.copy()
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    rt2 = np.stack([np.interp(t_dst, t_src, rt[:, k]) for k in range(3)], 1)
    q2 = np.stack([np.interp(t_dst, t_src, q[:, k]) for k in range(4)], 1)
    q2 /= np.linalg.norm(q2, axis=1, keepdims=True)
    dof2 = np.stack([np.interp(t_dst, t_src, dof[:, k])
                     for k in range(dof.shape[1])], 1)

    g1 = mujoco.MjModel.from_xml_path(
        str(DATA / "robots" / "g1" / "g1_29dof_mujoco.xml"))
    dat = mujoco.MjData(g1)
    bid = {k: mujoco.mj_name2id(g1, mujoco.mjtObj.mjOBJ_BODY, v)
           for k, v in KEY_BODIES.items()}
    P = {k: np.zeros((m, 3)) for k in KEY_BODIES}
    quat_torso = np.zeros((m, 4))
    quat_feet = np.zeros((m, 2, 4))
    for i in range(m):
        dat.qpos[:3] = rt2[i]
        dat.qpos[3:7] = q2[i]                     # wxyz (verified)
        dat.qpos[7:] = dof2[i]
        mujoco.mj_forward(g1, dat)
        for k, b in bid.items():
            P[k][i] = dat.xpos[b]
        quat_torso[i] = dat.xquat[bid["torso"]]
        quat_feet[i, 0] = dat.xquat[bid["lank"]]
        quat_feet[i, 1] = dat.xquat[bid["rank"]]

    yaw = np.arctan2(2 * (q2[:, 0] * q2[:, 3] + q2[:, 1] * q2[:, 2]),
                     1 - 2 * (q2[:, 2] ** 2 + q2[:, 3] ** 2))
    vel = np.gradient(P["pelvis"], axis=0) * FPS
    msk = np.linalg.norm(vel[:, :2], axis=1) > 0.3
    if msk.sum() > 20:
        err = np.abs(np.angle(np.exp(
            1j * (yaw[msk] - np.arctan2(vel[msk, 1], vel[msk, 0])))))
        print(f"[human] yaw-vs-motion check: median |err| "
              f"{np.median(err):.2f} rad (n={msk.sum()})")

    s = SCALE_TARGET_BASE_Z / max(np.mean(P["pelvis"][:, 2]), 1e-6)
    print(f"[human] {name}: {n}f@{fps} -> {m}f@50, scale s={s:.3f}")
    bp = P["pelvis"] * s
    bp[:, :2] -= bp[0, :2]
    # foot point = ankle + forward (from foot quat yaw) * 0.12 (G1 foot length)
    fyaw = np.arctan2(2 * (quat_feet[..., 0] * quat_feet[..., 3]
                           + quat_feet[..., 1] * quat_feet[..., 2]),
                      1 - 2 * (quat_feet[..., 2] ** 2 + quat_feet[..., 3] ** 2))
    ank = np.stack([P["lank"], P["rank"]], 1)                    # (m,2,3)
    fp = ank.copy()
    fp[..., 0] += 0.12 * np.cos(fyaw)
    fp[..., 1] += 0.12 * np.sin(fyaw)
    fp = fp * s
    fp[..., :2] -= (P["pelvis"] * s)[0, :2]
    hp = (P["torso"] + np.array([0., 0., 0.12])) * s
    hp[:, :2] -= (P["pelvis"] * s)[0, :2]
    blv = np.gradient(bp, axis=0) * FPS
    dyaw = np.angle(np.exp(1j * np.gradient(yaw))) * FPS
    vz = np.gradient(fp[..., 2], axis=0) * FPS
    contact = ((fp[..., 2] < 0.05) & (np.abs(vz) < 0.5)).astype(np.float32)
    P2 = {k: (v - P["pelvis"]) * s for k, v in P.items()}        # pelvis-rel
    return {"name": name, "T": m, "s": s, "J": P2, "yaw": yaw,
            "base_pos": bp.astype(np.float32), "base_quat": q2.astype(np.float32),
            "base_linvel": blv.astype(np.float32),
            "base_angvel": np.zeros((m, 3), np.float32),
            "foot_pos": fp.astype(np.float32), "foot_yaw": fyaw.astype(np.float32),
            "contact": contact, "head_pos": hp.astype(np.float32),
            "head_rotvec": sRot.from_quat(np.roll(
                quat_torso, -1, axis=-1)).as_rotvec().astype(np.float32),
            "head_angvel": (np.gradient(sRot.from_quat(np.roll(
                quat_torso, -1, axis=-1)).as_rotvec(), axis=0)
                * FPS).astype(np.float32)}


def stick_panel(ax, pts_by_line, view_yaw, title):
    c, sn = np.cos(view_yaw), np.sin(view_yaw)
    ax.clear()
    ax.set_xlim(-0.28, 0.28); ax.set_ylim(-0.04, 0.60)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    for chain, col in pts_by_line:
        P = np.asarray(chain)
        u = P[:, 0] * c + P[:, 1] * sn
        ax.plot(u, P[:, 2], color=col, lw=2)


def main():
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["BDX_ADAPTER_EVENT_LOG"] = str(OUT_DIR / "adapter_events.jsonl")
    hu = load_human(CLIP)
    if END is not None:
        hu = {k: (v[:END] if (isinstance(v, np.ndarray)
                              and v.shape[:1] == (hu["T"],)) else v)
              for k, v in hu.items()}
        hu["T"] = min(END, hu["T"])
        print(f"trimmed to first {hu['T']} frames (sane segment)")
    T = hu["T"]

    reg = resolve_canonical(DATA / "bdx_planner_v2combo")
    print(f"canonical planner: {reg['planner'].name}")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    planner = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    planner.load_state_dict(torch.load(reg["planner"], map_location=device,
                                       weights_only=False)["model"])
    stats = load_stats(DATA / "bdx_planner_v2combo" / "p1_stats.json")
    sp = {k: torch.tensor(v).to(device) for k, v in __import__("json").loads(
        (DATA / "bdx_planner_v2combo" / "p2_stats32.json").read_text()).items()}
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
        return fk.forward(base_pos, base_R, q, return_rot=True)

    def vel_from(pos, R):
        lv = torch.zeros_like(pos); lv[1:] = (pos[1:] - pos[:-1]) * 50
        av = torch.zeros_like(pos)
        av[1:] = rotmat_to_rotvec(R[1:] @ R[:-1].transpose(-1, -2)) * 50
        return lv, av

    # ---- env (same family as closed_loop_test / smoke) ----
    n_frames = T + 400
    dummy = build_dummy_pkl(OUT_DIR / "human_dummy.pkl", n_frames)
    model = load_model_from_checkpoint_dir(MODEL_FOLDER / "checkpoint", device=device)
    model.to(device).eval()
    config = __import__("json").load(open(MODEL_FOLDER / "config.json"))
    use_root_height_obs = config["env"].get("root_height_obs", False)
    config["env"]["lafan_tail_path"] = str(dummy.resolve())
    fixed = []
    for o in config["env"]["hydra_overrides"]:
        fixed.append("+" + o if o.startswith("simulator.config.sim.")
                     and not o.startswith("+") else o)
    config["env"]["hydra_overrides"] = fixed
    config["env"]["hydra_overrides"] += [
        "env.config.max_episode_length_s=10000", "env.config.headless=True",
        "simulator=mujoco"]
    config["env"]["disable_domain_randomization"] = True
    config["env"]["disable_obs_noise"] = True
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    env.set_is_evaluating(0)
    tensors = {k: getattr(env._motion_lib, k) for k in
               ("gts", "grs", "gvs", "gavs", "dvs", "dof_pos", "grvs", "gravs")}
    tensors_t = {k: getattr(env._motion_lib, k) for k in
                 ("gts_t", "grs_t", "gvs_t", "gavs_t") if hasattr(env._motion_lib, k)}

    renderer = None
    try:
        renderer = mujoco.Renderer(env.simulator.model, height=360, width=480)
    except Exception as e:
        print(f"WARN offscreen render unavailable: {e}")
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 1.4, -10, -90

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14.4, 3.6), dpi=100)

    obs, _ = wrapped_env.reset(to_numpy=False)
    root_state = torch.cat([tensors["gts"][0, 0], tensors["grs"][0, 0],
                            tensors["grvs"][0], tensors["gravs"][0]])
    dst = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0])
    dst[..., 0] = tensors["dof_pos"][0]
    obs, _ = env.reset_envs_idx(torch.arange(1), target_states={
        "dof_states": dst[None], "root_states": root_state[None]})
    obs, _, _, _, _ = wrapped_env.step(
        torch.zeros((1, wrapped_env.action_space.shape[-1]), dtype=torch.float32),
        to_numpy=False)
    obs = wrapped_env._get_g1env_observation(to_numpy=False)

    gts_plan = np.zeros((T, 17, 3), np.float32)     # planned stick source
    robot_qpos, teleports = [], 0
    frames = []
    n_win = T // WINDOW
    for wi in range(n_win):
        f0, f1 = wi * WINDOW, (wi + 1) * WINDOW
        hu_w = {k: v[f0:f1] for k, v in hu.items()
                if isinstance(v, np.ndarray) and v.shape[:1] == (hu["T"],)}
        psi = float(hu["yaw"][f0])
        # command from human pelvis velocity, window-mean (less noisy than
        # single-frame gradient), yaw frame at window start, BDX-scaled
        v = hu["base_linvel"][f0:f1].mean(0)
        c_, s_ = np.cos(psi), np.sin(psi)
        vx, vy = v[0] * c_ + v[1] * s_, -v[0] * s_ + v[1] * c_
        vyaw = float(np.angle(np.exp(
            1j * (hu["yaw"][min(f1, T - 1)] - psi))) / (WINDOW / 50))
        try:
            out = adapt({"skill": "locomotion",
                         "velocity": {"forward": float(vx), "lateral": float(vy),
                                      "yaw_rate": float(vyaw)},
                         "attention": {"target": "forward"}, "duration": 1.0})
            cmd7v = out["cmd7"]
        except CommandValidationError:
            cmd7v = [0.0] * 7                            # negative vx rejected
        cmd = torch.tensor(cmd7v, dtype=torch.float32)[None] \
            .expand(WINDOW, 7).contiguous()
        p = plan(hu_w, psi, cmd)
        pos_w = (p[:, 28:31].cpu().numpy() @ np.array(
            [[c_, s_, 0.], [-s_, c_, 0.], [0., 0., 1.]], np.float32)
            + hu["base_pos"][f0]).astype(np.float32)
        R_w = np.array([[c_, s_, 0.], [-s_, c_, 0.], [0., 0., 1.]]).T[None] \
            @ sixd_to_mat(p[:, 31:37]).cpu().numpy()
        quat_w = sRot.from_matrix(R_w).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
        for i in range(1, len(quat_w)):
            if np.dot(quat_w[i], quat_w[i - 1]) < 0:
                quat_w[i] = -quat_w[i]
        q_w = p[:, 0:14].cpu().numpy().astype(np.float32)

        bp = torch.from_numpy(
            pos_w[:, 0:3].copy().astype(np.float32)).to(device)
        bR = torch.from_numpy(R_w.astype(np.float32)).to(device)
        qd = torch.from_numpy(q_w.astype(np.float32)).to(device)
        gts_w, grs_R = fk_all(bp, bR, qd)
        lv, av = vel_from(gts_w, grs_R)
        quat_xyzw = np.roll(quat_w, -1, axis=-1)
        tensors["gts"][f0:f1] = gts_w
        tensors["grs"][f0:f1] = torch.from_numpy(
            np.tile(quat_xyzw[:, None, :], (1, 17, 1))).to(device)
        tensors["gvs"][f0:f1] = lv
        tensors["gavs"][f0:f1] = av
        tensors["dof_pos"][f0:f1] = qd
        tensors["dvs"][f0:f1] = torch.from_numpy(
            np.gradient(q_w, axis=0) * 50).to(device)
        tensors["grvs"][f0:f1] = lv[:, 0]
        tensors["gravs"][f0:f1] = av[:, 0]
        for k, tk in (("gts_t", "gts"), ("grs_t", "grs"),
                      ("gvs_t", "gvs"), ("gavs_t", "gavs")):
            if k in tensors_t:
                tensors_t[k][f0:f1] = tensors[tk][f0:f1]
        gts_plan[f0:f1] = gts_w.cpu().numpy()

        obs_w, _ = get_backward_observation(
            env, 0, use_root_height_obs=use_root_height_obs)
        z_w = model.backward_map({k: v[f0:f1] for k, v in obs_w.items()})
        z_w = model.project_z(z_w)

        for i in range(WINDOW):
            action = model.act(obs, z_w[i][None], mean=True)
            wrapped_env.step(action, to_numpy=False)
            obs = wrapped_env._get_g1env_observation(to_numpy=False)
            qp = np.asarray(env.simulator.data.qpos.copy(), np.float32)
            robot_qpos.append(qp)
            if renderer is not None and (f0 + i) % 2 == 0:
                view = hu["yaw"][f0 + i]
                O = hu["base_pos"][f0 + i].copy()
                J = {k: v[f0 + i] + O for k, v in hu["J"].items()}
                if not fig.axes:
                    axH = fig.add_subplot(131); axP = fig.add_subplot(132)
                    axR = fig.add_subplot(133)
                else:
                    axH, axP, axR = fig.axes
                O = np.zeros(3)
                stick_panel(axH, [
                    ([O, J["torso"]], "k"),
                    ([O, J["lhip"], J["lknee"], J["lank"]], "tab:blue"),
                    ([O, J["rhip"], J["rknee"], J["rank"]], "tab:red"),
                    ([J["torso"], J["lsh"], J["lel"], J["lwr"]], "tab:gray"),
                    ([J["torso"], J["rsh"], J["rel"], J["rwr"]], "tab:gray"),
                ], view, "human (LAFAN->G1 FK)")
                gp = gts_plan[f0 + i]
                rel = gp - gp[0]
                rel[:, 2] += gp[0, 2]  # ground-anchor: feet near 0, pelvis ~0.3
                stick_panel(axP, [
                    ([rel[0], rel[14]], "k"),
                    ([rel[0], rel[1], rel[3], rel[4], rel[5]], "tab:blue"),
                    ([rel[0], rel[6], rel[8], rel[9], rel[10]], "tab:red"),
                ], view, "planner-planned BDX")
                cam.lookat[:] = [qp[0], qp[1], 0.15]
                renderer.update_scene(env.simulator.data, camera=cam)
                imgR = renderer.render()
                axR.clear(); axR.imshow(imgR); axR.set_xticks([])
                axR.set_yticks([]); axR.set_title("tracker-executed", fontsize=8)
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        print(f"win {wi + 1}/{n_win} done", flush=True)

    qp_all = np.stack(robot_qpos)
    # teleport census (runtime limitation disclosure)
    jumps = np.linalg.norm(np.diff(qp_all[:, :3], axis=0), axis=1)
    teleports = int((jumps > 0.05).sum())
    from humanoidverse.planner.command_adapter_v13 import EVENTS
    metrics = {
        "clip": hu["name"], "frames": int(T), "scale": float(hu["s"]),
        "teleport_steps_gt5cm": teleports,
        "adapter_events_total": len(EVENTS),
        "runtime_issue": "RUNTIME_X_AXIS_TERMINATION_LOOP (disclosed)",
        "note": "leg imitation = qualitative (3-panel video is primary "
                "evidence); planner input = human 5 sparse points only "
                "(pelvis/head/feet, BDX-scaled); tracker = bfmzero-bdx-full",
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / f"result_{hu['name']}.json").write_text(
        __import__("json").dumps(metrics, indent=1, ensure_ascii=False))
    if frames:
        vp = OUT_DIR / f"imitation_{hu['name']}.mp4"
        media.write_video(str(vp), frames, fps=25)
        print("video:", vp)
    print("DONE", metrics)


if __name__ == "__main__":
    main()
