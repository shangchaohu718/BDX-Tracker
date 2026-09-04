"""Q3 perturbation-recovery probe under oracle moving schedule (ruling step 6).

v1-armA-300k final ckpt. 8 resets (seeds 6000+i, same lineage as schedule
probe). Oracle z_t from the reset's own source pool episode (training
contract). At t=50 inject a state perturbation directly into the MuJoCo sim:
  q_kick    dof_pos += 0.15*randn (14 dof), qvel += 2.0*randn
  root_kick base lin/ang vel += 1.5*randn / 3*randn
Conditions: {no_pert, q_kick, root_kick} x {correct_z, wrong_z(other motion)}.
Metric: e_t - e_50 (error growth after perturbation), recovery over
t in {55..150} vs the same rollout's pre-pert trend. Fall count.
H2 test: does correct z accelerate error decay vs wrong z?
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

SEQ = 8
DEV = torch.device("cuda")
H = 150
T_KICK = 50
N_RESETS = 8
CKPT = Path("results/v1-armA-300k/ckpt_275968")
POOL = "humanoidverse/data/bdx_expert_sim_pool_v1.pt"


def main():
    pool = torch.load(POOL, map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]
    by_motion = {m["motion_id"]: i for i, m in enumerate(meta)}

    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, DEV)
    inner = wrapped._env
    load_expert_trajectories_from_motion_lib(inner, FBcprAuxAgentConfig(**cfg["agent"]),
                                             device="cpu", base_ang_vel_frame="body")
    model = load_model_from_checkpoint_dir(CKPT, device="cuda")
    model.eval(); model._obs_normalizer.eval()
    rng_kick = np.random.default_rng(31337)

    conds = ["no_pert", "q_kick", "root_kick"]
    zconds = ["correct_z", "wrong_z"]
    results = {}
    lineage = []

    with torch.no_grad():
        def encode_windows(ep_i, w0):
            ep = eps[ep_i]
            T = ep["observation"]["state"].shape[0]
            zs = []
            for t in range(H):
                a, b = w0 + t, min(w0 + t + SEQ, T)
                if a >= T:
                    a, b = T - 1, T
                nxt = {k: ep["observation"][k][a:b].to(DEV) for k in ("state", "privileged_state")}
                bm = model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)
                zs.append(model.project_z(bm).flatten())
            src = model._obs_normalizer(
                {"privileged_state": ep["observation"]["privileged_state"][w0:w0 + H].to(DEV)})["privileged_state"]
            return torch.stack(zs), src

        def kick(kind):
            data = inner.simulator.data  # MuJoCo MjData
            if kind == "q_kick":
                data.qpos[7:21] += 0.15 * rng_kick.standard_normal(14)
                data.qvel[6:20] += 2.0 * rng_kick.standard_normal(14)
            elif kind == "root_kick":
                data.qvel[0:3] += 1.5 * rng_kick.standard_normal(3)
                data.qvel[3:6] += 3.0 * rng_kick.standard_normal(3)

        for r in range(N_RESETS):
            torch.manual_seed(6000 + r); np.random.seed(6000 + r)
            wrapped.reset(to_numpy=False)
            mid = int(inner.motion_ids[0].item())
            mtime = float(inner.motion_start_times[0].item())
            ep_i = by_motion[mid]
            T = eps[ep_i]["observation"]["state"].shape[0]
            w0 = int(np.clip(round(mtime / 0.02), 0, T - H - SEQ - 1))
            z_or, src = encode_windows(ep_i, w0)
            other_i = by_motion[meta[(ep_i + 7) % len(meta)]["motion_id"]]
            To = eps[other_i]["observation"]["state"].shape[0]
            z_wg, _ = encode_windows(other_i, int(np.clip(round(mtime / 0.02), 0, To - H - SEQ - 1)))
            src_np = src.cpu().numpy()
            lineage.append({"reset": r, "motion_id": mid, "motion_time": mtime})

            for cond in conds:
                for zc in zconds:
                    zs = z_or if zc == "correct_z" else z_wg
                    torch.manual_seed(6000 + r); np.random.seed(6000 + r)
                    obs, _ = wrapped.reset(to_numpy=False)
                    o = tree_map(lambda x: x.to(DEV), obs)
                    priv = np.zeros((H, src_np.shape[1]), dtype=np.float32)
                    term = H
                    for t in range(H):
                        a = model.act(o, zs[t].unsqueeze(0).expand(o["state"].shape[0], -1), mean=True)
                        o_n = model._obs_normalizer(o)
                        priv[t] = o_n["privileged_state"][0].cpu().numpy()
                        o2, _, tr, _, _ = wrapped.step(a, to_numpy=False)
                        o = tree_map(lambda x: x.to(DEV), o2)
                        if tr.any():
                            term = t + 1
                            priv[t + 1:] = priv[t:t + 1]
                            break
                        if t == T_KICK - 1 and cond != "no_pert":
                            kick(cond)
                    err = np.linalg.norm(priv - src_np, axis=-1)
                    e50 = err[T_KICK - 1]
                    # recovery slope: mean(e_{55..80}) - e50 and mean(e_{120..term}) - e50
                    rec_early = float(np.mean(err[55:min(80, term)]) - e50)
                    rec_late = float(np.mean(err[120:term]) - e50) if term > 120 else None
                    results.setdefault(f"{cond}|{zc}", []).append({
                        "reset": r, "term": term, "e50": float(e50),
                        "delta_55_80": rec_early, "delta_120_end": rec_late,
                        "fall": term < H,
                    })
                    print(f"r{r} {cond}|{zc}: e50={e50:.1f} d55_80={rec_early:+.1f} d120={rec_late if rec_late is None else round(rec_late,1)} term={term}", flush=True)

    agg = {}
    for k, rows in results.items():
        agg[k] = {
            "e50_mean": float(np.mean([r["e50"] for r in rows])),
            "delta_55_80_mean": float(np.mean([r["delta_55_80"] for r in rows])),
            "delta_120_end_mean": float(np.mean([r["delta_120_end"] for r in rows if r["delta_120_end"] is not None])),
            "fall_rate": float(np.mean([r["fall"] for r in rows])),
        }
    # z-specific recovery: (wrong_z - correct_z) delta, paired per reset
    zdiff = {}
    for cond in conds:
        ds = []
        rc = {r["reset"]: r for r in results[f"{cond}|correct_z"]}
        rw = {r["reset"]: r for r in results[f"{cond}|wrong_z"]}
        for k in rc:
            if k in rw:
                ds.append(rw[k]["delta_55_80"] - rc[k]["delta_55_80"])
        boot = [float(np.mean(np.array(ds)[np.random.default_rng(700 + i).integers(0, len(ds), len(ds))])) for i in range(2000)]
        zdiff[cond] = {"wrong_minus_correct_d55_80": float(np.mean(ds)),
                       "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}

    out = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_q3_perturb_recovery.py",
            "artifact_class": "tracker_dataset_v1",
            "model": str(CKPT),
            "setup": f"{N_RESETS} resets (6000+i, schedule-probe lineage), oracle moving z, kick at t={T_KICK}",
            "status": "exploratory",
        },
        "aggregate": agg,
        "z_specific_recovery_paired": zdiff,
        "reset_lineage": lineage,
        "per_reset": results,
    }
    p = Path("results/bfmzero-bdx-full/probe_q3_perturb_recovery.json")
    assert not p.exists()
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(agg, indent=1)); print(json.dumps(zdiff, indent=1)); print("saved")


if __name__ == "__main__":
    main()
