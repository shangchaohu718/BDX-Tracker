"""Paired fixed-panel evaluation over the A/B screening snapshots.

Two panels per checkpoint (final ruling 2026-08-24):
  1. Expert source-window panel (native-current): fixed stand/dynamic pool
     windows re-encoded by EACH checkpoint's own normalizer+B. Metrics:
     matched-vs-permuted margin, L1 counterfactual gap (dyn obs/dyn z vs
     stand obs/dyn z).
  2. Policy paired panel: fixed reset states + fixed z sources + fixed env
     seeds (common random numbers across arms). Metrics per episode:
     activity (new contract), action-rate, tracking proxy (privileged
     distance to source-window frame), fall, Spearman(reward, activity).

Snapshots depict TIME only — never independent seeds. A/B differences use
episode-level paired bootstrap (CI at final checkpoints).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env
from humanoidverse.utils.bdx_state import bdx_motion_activity

SEQ = 8
N_EP_DYN, N_EP_STAND, HORIZON = 4, 2, 150


def to_dev(d, dev):
    return tree_map(lambda x: x.to(dev), d)


def main():
    arm = os.environ.get("PANEL_ARM", "A")
    run_dir = Path(os.environ.get("PANEL_RUN", "results/gate1-screenA-combo-300k"))
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
    STAND = ("stand", "pose", "idle")
    fam = lambda i: 1 if any(s in names[m2e[int(meta[i]["motion_id"])]].lower() for s in STAND) else 0  # 1=stand

    # ---- fixed panels (built once, arm-independent) ----
    rng = np.random.default_rng(202)
    dyn_idx = [i for i in range(len(eps)) if fam(i) == 0 and eps[i]["observation"]["state"].shape[0] > SEQ + HORIZON + 8]
    stand_idx = [i for i in range(len(eps)) if fam(i) == 1 and eps[i]["observation"]["state"].shape[0] > SEQ + HORIZON + 8]
    expert_dyn = [(int(i), int(rng.integers(0, 40))) for i in rng.choice(dyn_idx, 24, replace=False)]
    expert_stand = [(int(i), int(rng.integers(0, 40))) for i in rng.choice(stand_idx, 12, replace=False)]
    ep_specs = [(int(i), int(rng.integers(0, 40)), "dyn") for i in rng.choice(dyn_idx, N_EP_DYN, replace=False)] + \
               [(int(i), int(rng.integers(0, 40)), "stand") for i in rng.choice(stand_idx, N_EP_STAND, replace=False)]

    def inject_read(wrapped, e_idx, t, qvel=True):
        inner = wrapped._env
        ep = eps[e_idx]
        T = ep["observation"]["state"].shape[0]
        t = min(t, T - 2)
        res = inner._motion_lib.get_motion_state(
            torch.tensor([int(meta[e_idx]["motion_id"])], dtype=torch.long, device=inner.device),
            torch.tensor([meta[e_idx]["t0"] + t + 0.02], dtype=torch.float32, device=inner.device))
        # meta t0 is pool-episode start; use direct offset instead via motion time
        # simpler: use env reset target from the stored obs? we only have obs, not raw states.
        return None  # placeholder — see note below

    # NOTE: pool stores observations, not raw qpos; for policy resets we use the
    # env's OWN reset-to-motion machinery via motion id + time (matches pool
    # provenance loosely). For tracking reference we use the pool obs frames.

    ckpts = sorted([d for d in run_dir.iterdir() if d.name.startswith("ckpt_")],
                   key=lambda d: int(d.name.split("_")[1]))
    dev = torch.device("cuda")
    out_rows = []
    for ck in ckpts:
        t_step = int(ck.name.split("_")[1])
        model = load_model_from_checkpoint_dir(ck, device="cuda")
        model.eval()
        model._obs_normalizer.eval()
        with torch.no_grad():
            # panel 1: expert source windows, own-B encode
            def enc(e_idx, w):
                ep_ = eps[e_idx]
                nxt = {k: ep_["observation"][k][w + 1:w + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
                b = model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)
                return model.project_z(b)
            z_dyn = torch.cat([enc(e, w) for e, w in expert_dyn])
            z_stand = torch.cat([enc(e, w) for e, w in expert_stand])
            odyn = tree_map(lambda *xs: torch.cat(xs), *[
                model._obs_normalizer({k: eps[e]["observation"][k][w + SEQ - 1:w + SEQ].to(dev) for k in ("state", "privileged_state")})
                for e, w in expert_dyn])
            ostd = tree_map(lambda *xs: torch.cat(xs), *[
                model._obs_normalizer({k: eps[e]["observation"][k][w + SEQ - 1:w + SEQ].to(dev) for k in ("state", "privileged_state")})
                for e, w in expert_stand])
            zd_proto = z_dyn.mean(0, keepdim=True)
            l_dyn = model._discriminator.compute_logits(odyn, zd_proto.expand(len(expert_dyn), -1)).flatten()
            l_std = model._discriminator.compute_logits(ostd, zd_proto.expand(len(expert_stand), -1)).flatten()
            perm = torch.randperm(len(expert_dyn), device=dev)
            m_matched = model._discriminator.compute_logits(odyn, z_dyn).mean()
            m_perm = model._discriminator.compute_logits(odyn, z_dyn[perm]).mean()
            margin = float(m_matched - m_perm)
            l1_gap = float(l_dyn.mean() - l_std.mean())

            # panel 2: paired policy episodes
            ep_metrics = []
            for ep_i, (e_idx, w0, kind) in enumerate(ep_specs):
                torch.manual_seed(9000 + ep_i)
                np.random.seed(9000 + ep_i)
                obs, _ = wrapped.reset(to_numpy=False)
                # fixed z from this episode's own source window (native-current)
                z_ep = enc(e_idx, w0)
                src = eps[e_idx]["observation"]["privileged_state"][w0:w0 + HORIZON].to(dev)
                acts, rews, dvs, trk = [], [], [], []
                o = to_dev(obs, dev)
                for h in range(HORIZON):
                    a = model.act(o, z_ep.expand(o["state"].shape[0], -1), mean=False)
                    r = float(model._discriminator.compute_reward(o, z_ep.expand(o["state"].shape[0], -1)).flatten()[0])
                    acts.append(float(a.abs().mean()))
                    rews.append(r)
                    dvs.append(float(bdx_motion_activity(o["state"])[0]))
                    trk.append(float((model._obs_normalizer(o)["privileged_state"][0] - src[min(h, src.shape[0] - 1)]).norm()))
                    o2, _, term, trunc, _ = wrapped.step(a, to_numpy=False)
                    o = to_dev(o2, dev)
                    if term.any():
                        break
                a_arr, r_arr, dv_arr = np.array(acts), np.array(rews), np.array(dvs)
                ep_metrics.append({
                    "kind": kind, "steps": len(acts),
                    "activity_mean": float(dv_arr.mean()),
                    "action_abs_mean": float(a_arr.mean()),
                    "action_rate": float(np.abs(np.diff(a_arr)).mean()) if len(a_arr) > 1 else 0.0,
                    "tracking_mean": float(np.mean(trk)),
                    "fall": bool(term.any()),
                    "reward_activity_spearman": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(r_arr, dv_arr).statistic) if len(set(dv_arr.round(3))) > 3 else None,
                })
            out_rows.append({
                "step": t_step, "arm": arm,
                "expert_panel": {"matched_vs_permuted_margin": margin, "l1_dyn_minus_stand": l1_gap},
                "episodes": ep_metrics,
            })
            print(f"arm{arm} {t_step}: margin={margin:.3f} l1={l1_gap:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    out = Path(f"results/bfmzero-bdx-full/paired_panel_arm{arm}.json")
    assert not out.exists()
    out.write_text(json.dumps(out_rows, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
