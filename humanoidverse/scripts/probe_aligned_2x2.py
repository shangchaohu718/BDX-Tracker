"""Arm x eval-alignment 2x2 (final ruling 2026-08-24): does z-aligned
curriculum help UNDER ITS OWN (aligned) evaluation condition?

Aligned eval: reset state = source window's PHYSICAL state (motion-lib
get_motion_state at its exact time, via reset_envs_idx target_states — the
env's own reset path: qpos/qvel/root restored, history zero-filled per env
reset semantics, DR redrawn under the paired seed), z = same window encoded
by the checkpoint's own B. Unaligned eval: env random motion reset + fixed
source-window z (previous panel semantics).

Metrics per episode: tracking (privileged dist to window frame h),
activity, dyn-stand conditional activity gap, source-relative activity,
action-rate, falls, reward-activity Spearman. Expert panel: standardized
matched-vs-permuted margin (seeded permutations) + sign accuracy.
Paired episode bootstrap for all 2x2 contrasts; full provenance on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env
from humanoidverse.utils.bdx_state import bdx_motion_activity

SEQ = 8
N_DYN, N_STAND, HORIZON = 8, 4, 150
DT = 0.02
STAND = ("stand", "pose", "idle")


def to_dev(d, dev):
    return tree_map(lambda x: x.to(dev), d)


def run_cell(model, wrapped, eps, meta, m2e, names, ep_specs, dev, aligned):
    inner = wrapped._env
    out = []
    for ep_i, (e_idx, w0, kind) in enumerate(ep_specs):
        torch.manual_seed(7000 + ep_i)
        np.random.seed(7000 + ep_i)
        wrapped.reset(to_numpy=False)
        if aligned:
            t_abs = meta[e_idx]["t0"] + w0 * DT
            res = inner._motion_lib.get_motion_state(
                torch.tensor([int(meta[e_idx]["motion_id"])], dtype=torch.long, device=inner.device),
                torch.tensor([t_abs], dtype=torch.float32, device=inner.device))
            n = 1
            dof_states = torch.zeros(n, inner.num_dof, 2, device=inner.device)
            root_states = torch.zeros(n, 13, device=inner.device)
            dof_states[..., 0] = res["dof_pos"]; dof_states[..., 1] = res["dof_vel"]
            root_states[:, :3] = res["root_pos"]; root_states[:, 3:7] = res["root_rot"]
            root_states[:, 7:10] = res["root_vel"]; root_states[:, 10:13] = res["root_ang_vel"]
            env_ids = torch.arange(n, device=inner.device)
            inner.reset_envs_idx(env_ids, target_states={"dof_states": dof_states, "root_states": root_states})
            ids = inner.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
            inner.simulator.set_actor_root_state_tensor(ids, inner.target_robot_root_states)
            inner.simulator.set_dof_state_tensor(ids, inner.target_robot_dof_state)
            inner.need_to_refresh_envs[ids] = False
            import mujoco
            mujoco.mj_forward(inner.simulator.model, inner.simulator.data)
            inner._pre_compute_observations_callback()
            inner._compute_observations()
            obs = wrapped._get_g1env_observation(to_numpy=False)
            obs = to_dev(obs, dev)
        else:
            obs, _ = wrapped.reset(to_numpy=False)
            obs = to_dev(obs, dev)
        # z from this window via the checkpoint's own B
        with torch.no_grad():
            nxt = {k: eps[e_idx]["observation"][k][w0 + 1:w0 + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
            z_ep = model.project_z(model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True))
        src = eps[e_idx]["observation"]["privileged_state"][w0:min(w0 + HORIZON, eps[e_idx]["observation"]["privileged_state"].shape[0])].to(dev)
        src_act = float(bdx_motion_activity(eps[e_idx]["observation"]["state"][w0:w0 + SEQ]).mean())
        rews, dvs, trk, acts = [], [], [], []
        with torch.no_grad():
            o = obs
            for h in range(HORIZON):
                a = model.act(o, z_ep.expand(o["state"].shape[0], -1), mean=True)  # deterministic main eval
                r = float(model._discriminator.compute_reward(o, z_ep.expand(o["state"].shape[0], -1)).flatten()[0])
                acts.append(float(a.abs().mean()))
                rews.append(r)
                dvs.append(float(bdx_motion_activity(o["state"])[0]))
                o_n = model._obs_normalizer(o)
                trk.append(float((o_n["privileged_state"][0] - src[min(h, src.shape[0] - 1)]).norm()))
                o2, _, term, trunc, _ = wrapped.step(a, to_numpy=False)
                o = to_dev(o2, dev)
                if term.any():
                    break
        r_arr, dv_arr = np.array(rews), np.array(dvs)
        out.append({
            "kind": kind, "steps": len(rews), "source_activity": src_act,
            "activity_mean": float(dv_arr.mean()),
            "source_relative_activity": float(dv_arr.mean() - src_act),
            "action_abs_mean": float(np.mean(acts)),
            "action_rate": float(np.abs(np.diff(np.array(acts))).mean()),
            "tracking_mean": float(np.mean(trk)),
            "fall": bool(term.any()),
            "reward_activity_spearman": float(spearmanr(r_arr, dv_arr).statistic) if len(set(dv_arr.round(3))) > 3 else None,
        })
    return out


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_full.pt", map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]
    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    m2e = {}
    for i in range(len(names)):
        m2e.setdefault(int(ep_motion[i]), i)
    fam = lambda i: 1 if any(s in names[m2e[int(meta[i]["motion_id"])]].lower() for s in STAND) else 0
    rng = np.random.default_rng(303)
    dyn_pool = [i for i in range(len(eps)) if fam(i) == 0 and eps[i]["observation"]["state"].shape[0] > SEQ + HORIZON + 40]
    stand_pool = [i for i in range(len(eps)) if fam(i) == 1 and eps[i]["observation"]["state"].shape[0] > SEQ + HORIZON + 40]
    ep_specs = [(int(i), int(rng.integers(20, 60)), "dyn") for i in rng.choice(dyn_pool, N_DYN, replace=False)] + \
               [(int(i), int(rng.integers(20, 60)), "stand") for i in rng.choice(stand_pool, N_STAND, replace=False)]

    arms = {"A": "results/gate1-screenA-combo-300k/checkpoint", "B": "results/gate1-screenB-refzalign-300k/checkpoint"}
    dev = torch.device("cuda")
    results = {}
    expert = {}
    for arm, ck in arms.items():
        model = load_model_from_checkpoint_dir(Path(ck), device="cuda")
        model.eval(); model._obs_normalizer.eval()
        results[arm] = {
            "aligned": run_cell(model, wrapped, eps, meta, m2e, names, ep_specs, dev, aligned=True),
            "unaligned": run_cell(model, wrapped, eps, meta, m2e, names, ep_specs, dev, aligned=False),
        }
        # standardized matched-vs-permuted margin with seeded permutations
        with torch.no_grad():
            zd = []
            od = []
            for i in rng.choice(dyn_pool, 32, replace=False):
                w = int(rng.integers(0, 40))
                nxt = {k: eps[i]["observation"][k][w + 1:w + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
                zd.append(model.project_z(model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)))
                od.append(model._obs_normalizer({k: eps[i]["observation"][k][w + SEQ - 1:w + SEQ].to(dev) for k in ("state", "privileged_state")}))
            zd = torch.cat(zd); od = tree_map(lambda *xs: torch.cat(xs), *od)
            dm = (model._discriminator.compute_logits(od, zd) - model._discriminator.compute_logits(od, zd[torch.randperm(32, device=dev)])).flatten()
        expert[arm] = {
            "std_margin": float(dm.mean() / (dm.std() + 1e-8)),
            "sign_acc": float((dm > 0).float().mean()),
        }
        del model
        torch.cuda.empty_cache()
        print(f"arm {arm} done", flush=True)

    # paired bootstrap contrasts
    def paired(cellA, cellB, key, kind=None):
        d = []
        for a, b in zip(cellA, cellB):
            if kind and a["kind"] != kind: continue
            if a.get(key) is None or b.get(key) is None: continue
            d.append(b[key] - a[key])
        if len(d) < 3: return None
        d = np.array(d)
        boots = [d[np.random.randint(0, len(d), len(d))].mean() for _ in range(10000)]
        return {"n": len(d), "delta": float(d.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}

    contrasts = {}
    for cond in ("aligned", "unaligned"):
        for key in ("tracking_mean", "activity_mean", "action_rate", "source_relative_activity", "reward_activity_spearman"):
            contrasts[f"B_minus_A__{cond}__{key}"] = paired(results["A"][cond], results["B"][cond], key)
        # conditional activity gap within arm
        for arm in ("A", "B"):
            dy = np.mean([e["activity_mean"] for e in results[arm][cond] if e["kind"] == "dyn"])
            st = np.mean([e["activity_mean"] for e in results[arm][cond] if e["kind"] == "stand"])
            contrasts[f"{arm}_{cond}_dyn_minus_stand_activity"] = float(dy - st)

    out = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "probe_aligned_2x2.py",
            "checkpoints": arms,
            "episodes": N_DYN + N_STAND, "horizon": HORIZON,
            "deterministic_action": True,
            "history_at_reset": "zero-filled (env reset semantics)",
            "bootstrap": "paired_episode, 10000 reps, seed=global-np",
            "state_contract": "bdx-state-v2", "history_contract": "newest-first-v2",
        },
        "expert_std_margin": expert,
        "cells": results,
        "contrasts": contrasts,
    }
    p = Path("results/bfmzero-bdx-full/probe_aligned_2x2.json")
    assert not p.exists()
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"expert": expert, "contrasts": {k: v for k, v in contrasts.items() if v and ("tracking" in k or "gap" in k or "dyn_minus" in k)}}, indent=2))


if __name__ == "__main__":
    main()
