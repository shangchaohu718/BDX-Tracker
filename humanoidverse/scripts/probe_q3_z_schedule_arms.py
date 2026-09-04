"""Q3 z-schedule 4-arm probe (ruling 2026-08-24: H3 temporal contract first).

Model: v1-armA-300k final ckpt. 12 resets (seeds 5000+i), lineage recorded
(motion_id, motion_time). Oracle z sequence per reset: encode the v1-pool
episode of that motion_id at 8-frame FORWARD windows (training contract:
z_t = project_z(mean B(source[t:t+8]))), 150 steps.

Arms (z fed to policy during rollout):
  S        static z_0 (matched start window, held — current-eval contract)
  O        oracle z_t each step (training tracking-env contract)
  C4/C8/C16 oracle refreshed every 4/8/16 steps
  P_delay  z_{t-20} (delayed)
  P_shift  z_{t+20} (phase-shifted)
  P_other  oracle of a DIFFERENT motion (fixed pairing)
  P_perm   oracle frames of the SAME motion, temporally permuted (blocks of 8)

Metrics per arm, per horizon h in {1,8,16,25,50,100,150}:
  tracking error (normalized privileged_state L2 to source, mean over t<=h)
  phase error (argmin over source frames of ||priv_t - src_f||, |f - t| median)
  root drift (xy displacement of policy vs source root)
  fall (terminated), action rate
Aggregate matched-vs-wrong advantage: O (and S) vs P_other as wrong.

Full kbuf preload (motionload lesson). Deterministic actions. Exploratory.
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
N_RESETS = 12
CKPT = Path("results/v1-armA-300k/ckpt_275968")
POOL = "humanoidverse/data/bdx_expert_sim_pool_v1.pt"
HORIZONS = [1, 8, 16, 25, 50, 100, 150]
ARMS = ["S", "O", "C4", "C8", "C16", "P_delay", "P_shift", "P_other", "P_perm"]


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

    # preload all oracle z sequences + source trajectories lazily per reset
    results = {}
    lineage = []
    rng = np.random.default_rng(4242)

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
            root = ep["observation"]["state"][w0:w0 + H, 24:27].to(DEV)  # placeholder idx fixed below
            return torch.stack(zs), src

        for r in range(N_RESETS):
            # seed MUST match the per-arm rollout resets below so lineage
            # (motion_id, motion_time) describes the actual rollout reset state
            torch.manual_seed(6000 + r); np.random.seed(6000 + r)
            obs, _ = wrapped.reset(to_numpy=False)
            mid = int(inner.motion_ids[0].item())
            mtime = float(inner.motion_start_times[0].item())
            lineage.append({"reset": r, "motion_id": mid, "motion_time": mtime})
            ep_i = by_motion.get(mid)
            if ep_i is None:
                print(f"reset {r}: motion {mid} not in pool, skip", flush=True)
                continue
            T = eps[ep_i]["observation"]["state"].shape[0]
            w0 = int(np.clip(round(mtime / 0.02), 0, T - H - SEQ - 1))
            z_or, src = encode_windows(ep_i, w0)
            # other-motion oracle (fixed partner: next pool episode with different motion)
            other_i = by_motion[meta[(ep_i + 7) % len(meta)]["motion_id"]]
            To = eps[other_i]["observation"]["state"].shape[0]
            w0o = int(np.clip(round(mtime / 0.02), 0, To - H - SEQ - 1))
            z_ot, _ = encode_windows(other_i, w0o)
            # permuted blocks: shuffle 8-step blocks of the oracle sequence
            # (H may not divide by 8 — index via per-t block map, last block short)
            n_blocks = (H + 7) // 8
            perm = rng.permutation(n_blocks)
            z_pm = torch.stack([z_or[min(perm[t // 8] * 8 + (t % 8), H - 1)] for t in range(H)])
            src_np = src.cpu().numpy()  # [H, 253]

            sched = {
                "S": torch.stack([z_or[0]] * H),
                "O": z_or,
                "C4": torch.stack([z_or[(t // 4) * 4] for t in range(H)]),
                "C8": torch.stack([z_or[(t // 8) * 8] for t in range(H)]),
                "C16": torch.stack([z_or[(t // 16) * 16] for t in range(H)]),
                "P_delay": torch.stack([z_or[max(0, t - 20)] for t in range(H)]),
                "P_shift": torch.stack([z_or[min(H - 1, t + 20)] for t in range(H)]),
                "P_other": z_ot,
                "P_perm": z_pm,
            }

            for arm in ARMS:
                torch.manual_seed(6000 + r); np.random.seed(6000 + r)  # same stream as lineage reset
                obs, _ = wrapped.reset(to_numpy=False)
                o = tree_map(lambda x: x.to(DEV), obs)
                priv = np.zeros((H, src_np.shape[1]), dtype=np.float32)
                rates = []
                prev = None
                term = H
                for t in range(H):
                    a = model.act(o, sched[arm][t].unsqueeze(0).expand(o["state"].shape[0], -1), mean=True)
                    if prev is not None:
                        rates.append(float((a - prev).norm()))
                    prev = a
                    o_n = model._obs_normalizer(o)
                    priv[t] = o_n["privileged_state"][0].cpu().numpy()
                    o2, _, tr, _, _ = wrapped.step(a, to_numpy=False)
                    o = tree_map(lambda x: x.to(DEV), o2)
                    if tr.any():
                        term = t + 1
                        priv[t + 1:] = priv[t:t + 1]
                        break
                # metrics
                err_t = np.linalg.norm(priv[:term] - src_np[:term], axis=-1)  # [term]
                cum = {h: float(err_t[:min(h, term)].mean()) for h in HORIZONS}
                # phase error: best-matching source frame per t (search +-40)
                ph = []
                for t in range(0, term, 5):
                    lo, hi = max(0, t - 40), min(H, t + 41)
                    d = np.linalg.norm(src_np[lo:hi] - priv[t], axis=-1)
                    ph.append(abs(lo + int(d.argmin()) - t))
                results.setdefault(arm, []).append({
                    "reset": r, "term": term,
                    "cum_err_by_horizon": cum,
                    "phase_err_median": float(np.median(ph)) if ph else None,
                    "action_rate": float(np.mean(rates)) if rates else 0.0,
                    "fall": term < H,
                })
                print(f"r{r} {arm}: err100={cum[100]:.2f} term={term}", flush=True)

    # aggregate
    agg = {}
    for arm in ARMS:
        rows = results[arm]
        agg[arm] = {
            "n": len(rows),
            "cum_err_by_horizon_mean": {h: float(np.mean([r["cum_err_by_horizon"][h] for r in rows])) for h in HORIZONS},
            "phase_err_median_of_medians": float(np.median([r["phase_err_median"] for r in rows if r["phase_err_median"] is not None])),
            "fall_rate": float(np.mean([r["fall"] for r in rows])),
            "action_rate_mean": float(np.mean([r["action_rate"] for r in rows])),
        }
    # matched-vs-wrong (per-reset paired): O vs P_other, S vs P_other
    pair = {}
    for good in ("O", "S", "C8"):
        diffs = []
        for i in range(len(results[good])):
            if results[good][i]["reset"] == results["P_other"][i]["reset"]:
                diffs.append(results["P_other"][i]["cum_err_by_horizon"][100] - results[good][i]["cum_err_by_horizon"][100])
        boot = [float(np.mean(np.array(diffs)[np.random.default_rng(1000 + k).integers(0, len(diffs), len(diffs))])) for k in range(2000)]
        pair[f"{good}_vs_P_other_err100"] = {
            "mean_diff": float(np.mean(diffs)),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }

    out = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_q3_z_schedule_arms.py",
            "artifact_class": "tracker_dataset_v1",
            "model": str(CKPT),
            "setup": f"{N_RESETS} resets (seeds 5000+i), {H} steps, deterministic; oracle z = project_z(mean B(source[t:t+8])) per training tracking-env contract",
            "motivation": "z_lifecycle_audit_v1a.json H3: training matched-z always moved per-step; static z never matched",
            "status": "exploratory",
        },
        "arms": agg,
        "matched_vs_wrong_paired": pair,
        "reset_lineage": lineage,
        "per_reset": results,
    }
    p = Path("results/bfmzero-bdx-full/probe_q3_z_schedule_arms.json")
    assert not p.exists()
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v["cum_err_by_horizon_mean"] for k, v in agg.items()}, indent=1))
    print("saved")


if __name__ == "__main__":
    main()
