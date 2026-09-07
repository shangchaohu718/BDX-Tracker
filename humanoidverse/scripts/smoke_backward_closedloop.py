"""[distill-adoption-v12] Bounded closed-loop backward smoke (user ruling 2026-09-07).

SMOKE_PLAN.json in OUT_DIR is the frozen governing plan (written BEFORE this
run). 12 trials = 4 vx levels (-0.15/-0.30/-0.50/-0.65) x 3 deterministic
seed-frame resets; each 2s settle + 10s command + 3s stop.

Runtime chain verbatim from closed_loop_test.py (the registry-documented
"real system"): sim true state -> FK -> sparse hist -> planner -> teacher
decode -> runtime motion-lib injection -> get_backward_observation slice ->
backward_map -> project_z -> model.act (MuJoCo 50 Hz, tracker =
results/bfmzero-bdx-full). EVERY window command goes through
planner.command_adapter_v12.adapt() — no direct student calls.

Multi-trial reuse: injection writes ABSOLUTE frames [750*trial, 750*trial+750);
seed frames are read at [750*trial+off, +50) which no trial has written, so
they are pristine walk content without any restore step.

Run: BDX_PLANNER_DATA=<smoke overlay> .venv/bin/python \
  humanoidverse/scripts/smoke_backward_closedloop.py
"""
import hashlib
import json
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
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, WINDOW,
                                           canonical_sparse, load_stats,
                                           sixd_to_mat)
from humanoidverse.planner.fk import BDXFK, rotmat_to_rotvec
from humanoidverse.planner.model import CommandedEncoder, MotionAE
from humanoidverse.planner.command_adapter_v12 import adapt, EVENTS
from humanoidverse.planner.registry import resolve_canonical

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA"))
MODEL_FOLDER = REPO.parent / "results" / "bfmzero-bdx-full"
LEVELS = [-0.15, -0.30, -0.50, -0.65]
SEEDS = [0, 17, 33]                      # frames offset inside gait cycle
SETTLE_W, CMD_W, STOP_W = 2, 10, 3       # 1 s windows
TW = SETTLE_W + CMD_W + STOP_W           # 15 windows / trial
TOTAL_STEPS = TW * WINDOW                # 750
CMD_F0, CMD_F1 = SETTLE_W * WINDOW, (SETTLE_W + CMD_W) * WINDOW
STEADY_F0 = CMD_F0 + 4 * WINDOW          # steady = t in [4,10) s of command
STOP_LAST1S = slice(TOTAL_STEPS - WINDOW, TOTAL_STEPS)


def build_dummy_pkl(path, frames):
    md = MotionData()
    name = next(n for n in md.split_names["train"]
                if n.startswith("v1_walk/")
                and int(md.d["lengths"][md.name2clip[n]]) >= 1000)
    T0 = 250
    f = md.frames(md.name2clip[name], T0,
                  int(md.d["lengths"][md.name2clip[name]]) - T0)
    reps = int(np.ceil(frames / len(f["q"])))
    print(f"dummy motion: {name} x{reps} -> {frames} frames")

    def loop(x):
        return np.concatenate([x] * reps, 0)[:frames]
    pose_aa = np.zeros((frames, 17, 3), np.float32)
    fk = BDXFK()
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


def _qwm(bq):
    from humanoidverse.planner.dataset import quat_wxyz_to_mat
    return quat_wxyz_to_mat(bq)


def yaw_of(q):   # qpos wxyz
    return float(np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]),
                            1 - 2 * (q[5] ** 2 + q[6] ** 2)))


def max_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


@torch.no_grad()
def main():
    assert (OUT_DIR / "SMOKE_PLAN.json").exists(), "frozen plan missing"
    device = "cuda"

    # ---- planner stack (overlay registry mirrors post-promotion state) ----
    _reg = resolve_canonical(OUT_DIR)
    print(f"canonical: {_reg['planner'].name} + {_reg['teacher'].name}")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    planner = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    planner.load_state_dict(torch.load(_reg["planner"], map_location=device,
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
        return fk.forward(base_pos, base_R, q, return_rot=True)

    def vel_from(pos, R):
        lv = torch.zeros_like(pos)
        lv[1:] = (pos[1:] - pos[:-1]) * 50
        av = torch.zeros_like(pos)
        av[1:] = rotmat_to_rotvec(R[1:] @ R[:-1].transpose(-1, -2)) * 50
        return lv, av

    # ---- env / tracker (mirror closed_loop_test setup) ----
    n_frames = len(LEVELS) * len(SEEDS) * TOTAL_STEPS + 400
    dummy = build_dummy_pkl(OUT_DIR / "closedloop_dummy_smoke.pkl", n_frames)
    model = load_model_from_checkpoint_dir(MODEL_FOLDER / "checkpoint", device=device)
    model.to(device).eval()
    config = json.load(open(MODEL_FOLDER / "config.json"))
    use_root_height_obs = config["env"].get("root_height_obs", False)
    config["env"]["lafan_tail_path"] = str(dummy.resolve())
    fixed = []
    for o in config["env"]["hydra_overrides"]:
        fixed.append("+" + o if o.startswith("simulator.config.sim.")
                     and not o.startswith("+") else o)
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
               ("gts", "grs", "gvs", "gavs", "dvs", "dof_pos", "grvs", "gravs")}
    tensors_t = {k: getattr(ml, k) for k in ("gts_t", "grs_t", "gvs_t", "gavs_t")
                 if hasattr(ml, k)}
    runtime_meta = {
        "use_root_height_obs": bool(use_root_height_obs),
        "use_contact_in_obs_max": bool(getattr(env, "use_contact_in_obs_max", None)),
        "add_time_aware_observation": bool(getattr(
            wrapped_env, "add_time_aware_observation", None)),
        "tracker": str(MODEL_FOLDER),
    }
    print("runtime:", runtime_meta)

    # non-foot geom classification (disclosed in result)
    nonfoot, contact_ok = set(), True
    try:
        m = env.simulator.model
        for gid in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if "foot" not in nm and nm != "floor" and "plane" not in nm:
                nonfoot.add(gid)
        print(f"non-foot robot geoms: {sorted(nonfoot)} "
              f"({m.ngeom} geoms total)")
    except Exception as e:                                  # pragma: no cover
        contact_ok = False
        print(f"WARN contact introspection failed: {e}")

    renderer = None
    try:
        renderer = mujoco.Renderer(env.simulator.model, height=240, width=320)
    except Exception as e:
        print(f"WARN offscreen render unavailable: {e}")
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 1.6, -12, -90

    obs, obs_dict = get_backward_observation(
        env, 0, use_root_height_obs=use_root_height_obs)
    z_full = model.backward_map(obs)     # warm start (mirrors original)

    def read_sim_qpos():
        return np.asarray(env.simulator.data.qpos.copy(), np.float32)

    trials = []
    t_glob = 0                             # absolute window counter
    for li, level in enumerate(LEVELS):
        for si, seed in enumerate(SEEDS):
            base = t_glob * WINDOW         # absolute frame base of this trial
            tidx = li * len(SEEDS) + si
            ev_path = OUT_DIR / f"adapter_events_trial{tidx:02d}.jsonl"
            if ev_path.exists():
                ev_path.unlink()
            os.environ["BDX_ADAPTER_EVENT_LOG"] = str(ev_path)
            EVENTS.clear()

            # ---- deterministic reset (robot state from pristine walk at
            #      frames [base+seed, +50); never written by any injection)
            sf = base + seed
            observation, _ = wrapped_env.reset(to_numpy=False)
            root_state = torch.cat([tensors["gts"][sf, 0],
                                    tensors["grs"][sf, 0],       # xyzw
                                    tensors["grvs"][sf],
                                    tensors["gravs"][sf]])
            dst = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0])
            dst[..., 0] = tensors["dof_pos"][sf]
            observation, _ = env.reset_envs_idx(
                torch.arange(1), target_states={
                    "dof_states": dst[None], "root_states": root_state[None]})
            observation, _, _, _, _ = wrapped_env.step(
                torch.zeros((1, wrapped_env.action_space.shape[-1]),
                            dtype=torch.float32), to_numpy=False)
            observation = wrapped_env._get_g1env_observation(to_numpy=False)
            hist = list(np.concatenate([
                tensors["gts"][sf:sf + WINDOW, 0].cpu().numpy(),
                np.roll(tensors["grs"][sf:sf + WINDOW, 0].cpu().numpy(), 1, -1),
                tensors["dof_pos"][sf:sf + WINDOW].cpu().numpy()], -1))

            qps, zs, eps, cts, frames = [], [], [], [], []
            ep_len = None
            clamped_any = False
            for wi in range(TW):
                seg = ("settle" if wi < SETTLE_W else
                       "cmd" if wi < SETTLE_W + CMD_W else "stop")
                vx = level if seg == "cmd" else 0.0
                l1 = {"skill": "locomotion",
                      "velocity": {"forward": vx, "lateral": 0.0,
                                   "yaw_rate": 0.0},
                      "attention": {"target": "forward"}, "duration": 1.0}
                out = adapt(l1)            # <- adapter in the chain
                if any(out["saturation_mask"]):
                    clamped_any = True
                cmd = torch.tensor(out["cmd7"], dtype=torch.float32)[None] \
                    .expand(WINDOW, 7).contiguous()

                h = np.stack(hist[-WINDOW:])
                bq = torch.from_numpy(h[:, 3:7].astype(np.float32))
                psi = float(torch.atan2(
                    2 * (bq[-1, 0] * bq[-1, 3] + bq[-1, 1] * bq[-1, 2]),
                    1 - 2 * (bq[-1, 2] ** 2 + bq[-1, 3] ** 2)))
                f_hist = {"base_pos": h[:, 0:3], "base_quat": h[:, 3:7],
                          "q": h[:, 7:],
                          "qdot": np.gradient(h[:, 7:], axis=0) * 50}
                pos_h, R_h = fk_all(torch.from_numpy(h[:, 0:3]).to(device),
                                    _qwm(bq).to(device),
                                    torch.from_numpy(h[:, 7:]).to(device))
                lv_h, av_h = vel_from(pos_h, R_h)
                yaw_f = np.arctan2(
                    R_h[:, [5, 10]].cpu().numpy()[..., 1, 0],
                    R_h[:, [5, 10]].cpu().numpy()[..., 0, 0])
                f_hist.update({
                    "base_linvel": lv_h[:, 0].cpu().numpy(),
                    "base_angvel": av_h[:, 0].cpu().numpy(),
                    "head_rotvec": rotmat_to_rotvec(R_h[:, 12]).cpu().numpy(),
                    "head_angvel": av_h[:, 12].cpu().numpy(),
                    "foot_pos": pos_h[:, [5, 10]].cpu().numpy()
                    .astype(np.float32),
                    "foot_yaw": yaw_f.astype(np.float32),
                    "contact": np.zeros((WINDOW, 2), np.float32),
                    "head_pos": pos_h[:, 12].cpu().numpy()})
                p = plan(f_hist, psi, cmd)
                c_, s_ = np.cos(psi), np.sin(psi)
                Rz = np.array([[c_, s_, 0.], [-s_, c_, 0.], [0., 0., 1.]],
                              np.float32)
                pos_w = (p[:, 28:31].cpu().numpy() @ Rz + h[-1, 0:3]) \
                    .astype(np.float32)
                R_w = Rz.T[None] @ sixd_to_mat(p[:, 31:37]).cpu().numpy()
                quat_w = quat_wxyz_from_R(R_w).astype(np.float32)
                for i in range(1, len(quat_w)):
                    if np.dot(quat_w[i], quat_w[i - 1]) < 0:
                        quat_w[i] = -quat_w[i]
                q_w = p[:, 0:14].cpu().numpy().astype(np.float32)

                f0, f1 = base + wi * WINDOW, base + (wi + 1) * WINDOW
                bp = torch.from_numpy(pos_w[:, 0:3].copy()).to(device)
                bR = torch.from_numpy(R_w).to(device)
                qd = torch.from_numpy(q_w).to(device)
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

                obs_w, _ = get_backward_observation(
                    env, 0, use_root_height_obs=use_root_height_obs)
                z_w = model.backward_map({k: v[f0:f1] for k, v in obs_w.items()})
                z_w = model.project_z(z_w)

                for i in range(WINDOW):
                    action = model.act(observation, z_w[i][None], mean=True)
                    wrapped_env.step(action, to_numpy=False)
                    observation = wrapped_env._get_g1env_observation(
                        to_numpy=False)
                    qp = read_sim_qpos()
                    qps.append(qp)
                    zs.append(float(env.simulator._rigid_body_pos[0, 0, 2]))
                    try:
                        e = int(env.episode_length_buf[0])
                        ep_len = e if ep_len is None else ep_len
                        eps.append(e)
                    except Exception:
                        eps.append(-1)
                    nc = False
                    if contact_ok:
                        d = env.simulator.data
                        for ci in range(d.ncon):
                            ct = d.contact[ci]
                            if int(ct.geom1) in nonfoot or \
                                    int(ct.geom2) in nonfoot:
                                nc = True
                                break
                    cts.append(nc)
                    if renderer is not None and len(qps) % 2 == 0:
                        cam.lookat[:] = [qp[0] + 0.15, qp[1], 0.22]
                        renderer.update_scene(env.simulator.data, camera=cam)
                        frames.append(renderer.render())
                hist = hist[-WINDOW:]
            t_glob += TW

            # ---- per-trial frozen checks ----
            qp = np.stack(qps)
            z = np.array(zs)
            ep = np.array(eps)
            nominal = float(np.median(z[:CMD_F0]))
            falls = z < 0.5 * nominal
            fall_run = max_run(falls)
            teleport = float(np.abs(np.diff(qp[:, 0:2], axis=0))
                             .max()) if len(qp) > 1 else 0.0
            ep_reset = bool(np.any(np.diff(ep) < 0)) if ep[0] >= 0 else None
            ct_run = max_run(cts)
            yaw0 = yaw_of(qp[CMD_F0])
            exy = np.array([np.cos(yaw0), np.sin(yaw0)])
            pv = qp[:, 0:2]
            sub = np.array([(pv[f + WINDOW] - pv[f]) @ exy
                            for f in range(CMD_F0, CMD_F1, WINDOW)])
            steady = sub[4:]
            stop_disp = (pv[TOTAL_STEPS - 1] - pv[TOTAL_STEPS - WINDOW - 1]) @ exy
            checks = {
                "no_fall": bool(fall_run < 25),
                "no_termination": bool(teleport <= 0.1 and
                                       (ep_reset in (None, False))),
                "no_sustained_nonfoot_contact": bool(ct_run < 25),
                "no_sign_reversal_G2": int((sub > 0).sum()) == 0,
                "steady_negative": bool(steady.mean() < 0),
                "stop_residual_ok": bool(abs(stop_disp) <= 0.10
                                         and stop_disp >= -0.05),
                "adapter_no_clamp": bool(not clamped_any and len(EVENTS) == 0),
            }
            trial = {
                "trial": tidx, "level": level, "seed_frame": seed,
                "checks": checks,
                "pass": all(checks.values()),
                "nominal_z": nominal, "min_z": float(z.min()),
                "fall_max_run_steps": int(fall_run),
                "teleport_max_step_m": teleport, "episode_reset": ep_reset,
                "nonfoot_contact_max_run_steps": int(ct_run),
                "realized_vx_subwindows": sub.tolist(),
                "steady_mean_vx": float(steady.mean()),
                "steady_abs_mean_vx": float(np.abs(steady).mean()),
                "stop_last1s_vx_m": float(stop_disp),
                "adapter_events": len(EVENTS),
                "adapter_clamped": bool(clamped_any),
            }
            trials.append(trial)
            np.savez(OUT_DIR / f"trial{tidx:02d}.npz", qpos=qp, base_z=z,
                     ep_len=ep, nonfoot_contact=np.array(cts, bool),
                     sub_vx=sub, level=level, seed=seed)
            if renderer is not None and frames:
                media.write_video(str(OUT_DIR / f"trial{tidx:02d}.mp4"),
                                  frames, fps=25)
            print(f"T{tidx:02d} vx{level:+.2f} seed{seed:02d} "
                  f"{'PASS' if trial['pass'] else 'FAIL'} "
                  f"steady {steady.mean():+.3f} m/s min_z {z.min():.3f} "
                  f"tele {teleport:.3f} ct_run {ct_run} "
                  f"stop {stop_disp:+.3f} ev {len(EVENTS)}", flush=True)

    # ---- mechanical envelope (frozen mapping) ----
    levels_sorted = sorted(LEVELS, key=abs)
    lvl_pass = {lv: all(t["pass"] for t in trials if t["level"] == lv)
                for lv in LEVELS}
    deepest = 0.0
    for lv in levels_sorted:
        if lvl_pass[lv]:
            deepest = lv
        else:
            break
    if deepest == 0.0:
        envelope = "UNSUPPORTED"
    else:
        envelope = [deepest, 0.0]
    means = {lv: float(np.mean([t["steady_abs_mean_vx"] for t in trials
                                if t["level"] == lv])) for lv in LEVELS}
    seq = [means[lv] for lv in levels_sorted]
    monotone = all(seq[i + 1] >= seq[i] - 0.02 for i in range(len(seq) - 1)) \
        and seq[-1] > seq[0]

    result = {
        "plan": "SMOKE_PLAN.json (frozen 2026-09-07T01:00Z)",
        "generated": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_meta,
        "contact_introspection_ok": contact_ok,
        "trials": trials,
        "level_pass": lvl_pass,
        "steady_abs_mean_vx_by_level": means,
        "monotone_with_0.02_tol": bool(monotone),
        "envelope_decision": envelope,
        "all_12_pass": all(t["pass"] for t in trials),
    }
    (OUT_DIR / "SMOKE_RESULT.json").write_text(json.dumps(result, indent=1))
    for p in sorted(OUT_DIR.glob("trial*")) + [OUT_DIR / "SMOKE_RESULT.json"]:
        h = hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
        print(f"sha256[{p.name}] {h}")
    print("ENVELOPE:", envelope, "| monotone:", monotone,
          "| all12:", result["all_12_pass"])


if __name__ == "__main__":
    main()
