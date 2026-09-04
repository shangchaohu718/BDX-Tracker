"""Compare expert-FK and MuJoCo observation builders at identical states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch
from easydict import EasyDict

from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from humanoidverse.utils.torch_utils import quat_rotate_inverse


BODY_NAMES = [
    "base_link", "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link", "left_knee_link",
    "left_ankle_link", "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
    "right_knee_link", "right_ankle_link", "neck_forward_link", "neck_pitch_link", "neck_yaw_link",
    "neck_roll_link", "left_ear_link", "right_ear_link",
]


def quat_matrix_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def error_stats(a, b):
    e = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1)
    return {
        "mean_abs": float(e.mean()),
        "p50_abs": float(np.quantile(e, 0.50)),
        "p95_abs": float(np.quantile(e, 0.95)),
        "p99_abs": float(np.quantile(e, 0.99)),
        "max_abs": float(e.max()),
        "rmse": float(np.sqrt(np.mean(e * e))),
    }


def rotation_error(a_xyzw, b_xyzw):
    dot = np.abs(np.sum(a_xyzw * b_xyzw, axis=-1))
    return 2 * np.arccos(np.clip(dot, -1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", type=int, default=32)
    ap.add_argument("--frames-per-motion", type=int, default=64)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--output", default="results/bfmzero-bdx-full/matched_state_observation_audit.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    motion_data = joblib.load("humanoidverse/data/bdx_14dof_train.pkl")
    keys = np.array(sorted(motion_data.keys()))
    usable = np.array([k for k in keys if len(motion_data[k]["pose_aa"]) > args.frames_per_motion + 12])
    selected = rng.choice(usable, size=min(args.motions, len(usable)), replace=False)

    cfg = EasyDict({
        "asset": {"assetRoot": "humanoidverse/data/robots/bdx/", "assetFileName": "bdx_v4.xml"},
        "extend_config": [],
        "smooth_angular_velocity": False,
    })
    hb = Humanoid_Batch(cfg, torch.device("cpu"))
    model = mujoco.MjModel.from_xml_path("humanoidverse/data/robots/bdx/scene_bdx_freebase_mujoco.xml")
    data = mujoco.MjData(model)
    body_ids = np.arange(1, model.nbody)
    sim_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in body_ids]
    if hb.body_names != sim_names or sim_names != BODY_NAMES:
        raise RuntimeError("body order mismatch")
    default_dof = model.qpos0[7:].copy()

    rows = {k: [] for k in (
        "expert_dof_pos", "sim_dof_pos", "expert_dof_vel", "sim_dof_vel",
        "expert_gravity", "sim_gravity", "expert_base_ang_world", "sim_base_ang_body",
        "expert_base_ang_body", "expert_body_pos", "sim_body_pos", "expert_body_rot", "sim_body_rot",
        "expert_body_vel", "sim_body_vel", "expert_body_ang", "sim_body_ang",
    )}
    clip_info = []
    for key in selected:
        clip = motion_data[str(key)]
        dt = 1.0 / float(clip["fps"])
        pose = torch.from_numpy(clip["pose_aa"]).float()[None]
        trans = torch.from_numpy(clip["root_trans_offset"]).float()[None]
        fk = hb.fk_batch(pose, trans, return_full=True, dt=dt)
        rot = fk.global_rotation[0].numpy()
        pos = fk.global_translation[0].numpy()
        av = fk.global_angular_velocity[0].numpy()
        lv = fk.global_velocity[0].numpy()  # legacy sigma=2 linear velocity remains unchanged
        q = fk.dof_pos[0].numpy()
        qd = fk.dof_vels[0].numpy()
        frames = np.linspace(5, len(q) - 6, args.frames_per_motion).astype(int)
        clip_info.append({"key": str(key), "frames": len(frames), "fps": float(clip["fps"])})
        for t in frames:
            root_xyzw = rot[t, 0]
            root_wxyz = root_xyzw[[3, 0, 1, 2]]
            root_R = quat_matrix_wxyz(root_wxyz)
            data.qpos[:] = model.qpos0
            data.qpos[:3] = trans[0, t].numpy()
            data.qpos[3:7] = root_wxyz
            data.qpos[7:] = q[t]
            data.qvel[:3] = lv[t, 0]
            data.qvel[3:6] = root_R.T @ av[t, 0]
            data.qvel[6:] = qd[t]
            mujoco.mj_forward(model, data)

            sim_pos = data.xpos[body_ids].copy()
            sim_rot = data.xquat[body_ids][:, [1, 2, 3, 0]].copy()
            sim_av = data.cvel[body_ids, :3].copy()
            sim_lv = np.zeros((17, 3))
            for j, bid in enumerate(body_ids):
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(bid), vel, 0)
                sim_lv[j] = vel[3:6]

            gravity = torch.tensor([[0.0, 0.0, -1.0]])
            root_q = torch.from_numpy(root_xyzw[None]).float()
            projected = quat_rotate_inverse(root_q, gravity, w_last=True).numpy()[0]
            rows["expert_dof_pos"].append(q[t] - default_dof)
            rows["sim_dof_pos"].append(data.qpos[7:] - default_dof)
            rows["expert_dof_vel"].append(qd[t]); rows["sim_dof_vel"].append(data.qvel[6:].copy())
            rows["expert_gravity"].append(projected); rows["sim_gravity"].append(projected.copy())
            rows["expert_base_ang_world"].append(av[t, 0])
            rows["sim_base_ang_body"].append(root_R.T @ sim_av[0])
            rows["expert_base_ang_body"].append(root_R.T @ av[t, 0])
            rows["expert_body_pos"].append(pos[t]); rows["sim_body_pos"].append(sim_pos)
            rows["expert_body_rot"].append(rot[t]); rows["sim_body_rot"].append(sim_rot)
            rows["expert_body_vel"].append(lv[t]); rows["sim_body_vel"].append(sim_lv)
            rows["expert_body_ang"].append(av[t]); rows["sim_body_ang"].append(sim_av)

    rows = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}
    # Reproduce max-local observation blocks, including heading transforms.
    exp_priv = compute_humanoid_observations_max(
        torch.from_numpy(rows["expert_body_pos"]), torch.from_numpy(rows["expert_body_rot"]),
        torch.from_numpy(rows["expert_body_vel"]), torch.from_numpy(rows["expert_body_ang"]), True, True,
    )
    sim_priv = compute_humanoid_observations_max(
        torch.from_numpy(rows["sim_body_pos"]), torch.from_numpy(rows["sim_body_rot"]),
        torch.from_numpy(rows["sim_body_vel"]), torch.from_numpy(rows["sim_body_ang"]), True, True,
    )
    result = {
        "samples": int(rows["expert_dof_pos"].shape[0]),
        "motions": clip_info,
        "body_order_equal": True,
        "discriminator_inputs": ["state", "privileged_state"],
        "discriminator_uses_history": False,
        "blocks": {
            "dof_pos": error_stats(rows["expert_dof_pos"], rows["sim_dof_pos"]),
            "dof_vel": error_stats(rows["expert_dof_vel"], rows["sim_dof_vel"]),
            "projected_gravity": error_stats(rows["expert_gravity"], rows["sim_gravity"]),
            "base_ang_vel_legacy_world_frame": error_stats(rows["expert_base_ang_world"], rows["sim_base_ang_body"]),
            # Current fixed builders convert expert root angular velocity to
            # the same body frame used by policy proprioception.
            "base_ang_vel_current_builders": error_stats(rows["expert_base_ang_body"], rows["sim_base_ang_body"]),
            "root_height": error_stats(exp_priv["root_height"].numpy(), sim_priv["root_height"].numpy()),
            "local_body_pos": error_stats(exp_priv["local_body_pos"].numpy(), sim_priv["local_body_pos"].numpy()),
            "local_body_rot": error_stats(exp_priv["local_body_rot"].numpy(), sim_priv["local_body_rot"].numpy()),
            "local_body_vel": error_stats(exp_priv["local_body_vel"].numpy(), sim_priv["local_body_vel"].numpy()),
            "local_body_ang_vel": error_stats(exp_priv["local_body_ang_vel"].numpy(), sim_priv["local_body_ang_vel"].numpy()),
        },
        "rotation_geodesic_rad": error_stats(
            np.zeros_like(rotation_error(rows["expert_body_rot"], rows["sim_body_rot"])),
            rotation_error(rows["expert_body_rot"], rows["sim_body_rot"]),
        ),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
