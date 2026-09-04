"""Experiments A+B on the intro endpoint FAIL (ruling 2026-08-25 evening).

A endpoint hold: quintic intro (1.0 s) -> HOLD final q_ref 1.0 s. Classify
  post-endpoint error: growing / plateau / converging. Per-joint endpoint err.
B gravity balance offset: during hold, q_des = q_ref + clip(qfrc_bias/kp).
  If endpoint L2 gap drops materially -> supports "residual policy mainly
  learns static dynamics compensation".

Runs on both box_step_15 and side_step_4 (same env machinery as the gates).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

import mujoco

FPS = 50.0
DT = 1 / FPS
HOLD = 50  # 1.0 s


def main():
    rcfg = yaml.safe_load(open("humanoidverse/config/robot/bdx/bdx_14dof.yaml"))["robot"]
    from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())

    out = {"_provenance": {"created_at": datetime.now().isoformat(),
                           "script": "fds_intro_hold_gravity_probe.py",
                           "design": "A: quintic intro 1.0s -> hold q_ref 1.0s; B: hold with q_des=q_ref+qfrc_bias/kp",
                           "preregistered_readout": "endpoint gap classification + gravity-offset delta + per-joint pattern (knee/hip/ankle same-direction sag?)"}}

    import sys
    clip = sys.argv[1] if len(sys.argv) > 1 else "re_dancegen_box_step_15"
    for clip in [clip]:
        pkl = "fds_box_step_15.pkl" if "box" in clip else f"fds_{clip}.pkl"
        os.environ["BFM_POOL_DATA"] = str(Path(f"humanoidverse/data/{pkl}").resolve())
        npz = dict(np.load(f"/home/tcl/Desktop/start/dataset/v3_planner/clips/{clip}.npz", allow_pickle=True))
        jn = [str(x) for x in npz["joint_names"]]
        jorder = [jn.index(n) for n in rcfg["dof_names"]]
        jp = npz["joint_pos"][:, jorder].astype(np.float64)
        jv = npz["joint_vel"][:, jorder].astype(np.float64)
        home = np.array([float(rcfg["init_state"]["default_joint_angles"][n]) for n in rcfg["dof_names"]])

        gate = json.loads(Path(f"results/fds/{clip}_dynamic_gate.json" if "side" in clip else "results/fds/boxstep15_dynamic_gate.json").read_text())
        anchor = gate["2_transition_anchor"]["anchor_frame"]

        wrapped = build_cpu_env(cfg, torch.device("cuda"))
        inner = wrapped._env
        dev = inner.device
        default_dof_pos = inner.default_dof_pos.flatten()
        effort = torch.tensor(rcfg["dof_effort_limit_list"], device=dev)
        kp_j = inner.p_gains.flatten().cpu().numpy()

        def quintic(t, T, p0, v0, a0, p1, v1, a1):
            A = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],
                          [1,T,T**2,T**3,T**4,T**5],[0,1,2*T,3*T**2,4*T**3,5*T**4],
                          [0,0,2,6*T,12*T**2,20*T**3]])
            c = np.linalg.solve(A, np.stack([p0,v0,np.zeros(14),p1,v1,a1]))
            tp = np.stack([np.ones_like(t), t, t**2, t**3, t**4, t**5], -1)
            return tp @ c

        D = 1.0; n = int(D * FPS)
        ts = np.arange(n + 1) * DT
        q_path = quintic(ts, D, home, np.zeros(14), np.zeros(14), jp[anchor], jv[anchor], np.gradient(jv, DT, axis=0)[anchor])

        def run(mode):
            data = inner.simulator.data
            root_a = npz["body_pos_w"][anchor, 0].astype(np.float64)
            quat = npz["body_quat_w"][anchor, 0].astype(np.float64)
            data.qpos[0:3] = [root_a[0], root_a[1], 0.35]
            data.qpos[3:7] = [quat[1], quat[2], quat[3], quat[0]]
            data.qpos[7:21] = home
            data.qvel[:] = 0.0
            mujoco.mj_forward(inner.simulator.model, data)
            errs = []
            for t in range(n + HOLD):
                q_t = q_path[min(t, n)]
                if t >= n:  # hold
                    q_t = jp[anchor].copy()
                    if mode == "B":
                        q_t = q_t + np.clip(data.qfrc_bias[6:20] / kp_j, -0.5, 0.5)
                raw = (kp_j * (q_t - default_dof_pos.cpu().numpy())) / (1.25 * effort.cpu().numpy())
                a = np.clip(raw, -1.0, 1.0)
                wrapped.step(torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(0), to_numpy=False)
                errs.append(np.abs(data.qpos[7:21] - jp[anchor]).mean() if t >= n
                            else np.abs(data.qpos[7:21] - q_path[min(t + 1, n)]).mean())
            hold_errs = errs[n:]
            end_j = data.qpos[7:21] - jp[anchor]
            return {
                "endpoint_gap_l2": float(np.linalg.norm(end_j)),
                "hold_err_start_mid_end": [float(hold_errs[0]), float(hold_errs[HOLD // 2]), float(hold_errs[-1])],
                "classification": ("converging" if hold_errs[-1] < 0.8 * hold_errs[0] else
                                   "growing" if hold_errs[-1] > 1.2 * hold_errs[0] else "plateau"),
                "per_joint_endpoint_err": [round(float(x), 4) for x in end_j],
            }

        out[clip] = {"A_hold": run("A"), "B_gravity_offset": run("B")}
        delta = out[clip]["A_hold"]["endpoint_gap_l2"] - out[clip]["B_gravity_offset"]["endpoint_gap_l2"]
        out[clip]["B_minus_A_gap_reduction"] = float(delta)
        print(clip, json.dumps(out[clip], indent=1))

    p = Path(f"results/fds/intro_hold_gravity_probe_{clip}.json")
    assert not p.exists()
    p.write_text(json.dumps(out, indent=2) + "\n")
    print("saved", p)


if __name__ == "__main__":
    main()
