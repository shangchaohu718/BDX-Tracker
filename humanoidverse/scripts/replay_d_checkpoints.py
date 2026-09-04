"""Formation-mechanism replay: score FIXED probes under harvested D
checkpoints (5k..300k) to test whether the anti-dynamic direction was forged
by early chaotic policy phases.

Probes (fixed across checkpoints):
  expert-stand windows (held-out pool, low activity)
  expert-dynamic windows (held-out pool, top activity deciles)
  frozen injected stand (zero-velocity standing pose)
  policy snapshots: rollout obs from each checkpoint's own actor under a
  fixed dynamic z (chaotic 5k/25k/100k -> frozen 300k)

Key series:
  M_t = D_t(expert_dyn) - D_t(expert_stand)      (sign flip = anti-dynamic forging)
  C_t = D_t(expert_dyn) - D_t(policy_t)          (D separating expert-dyn from chaos)
Each checkpoint's own normalizer is used (contemporaneous, per ruling).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import (
    build_cpu_env, inject_read_exact_lean, OBS_KEYS,
)

SEQ = 8
STAND_MARKERS = ("stand", "pose", "idle")
ROLL_STEPS = 384


def family(name):
    n = name.lower()
    return "stand" if any(m in n for m in STAND_MARKERS) else "dynamic"


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_cpu.pt", map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]
    dev = torch.device("cuda")

    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    motion_to_ep = {}
    for i in range(len(names)):
        motion_to_ep.setdefault(int(ep_motion[i]), i)

    rng = np.random.default_rng(23)
    # fixed probes from pool
    stand_idx = [i for i in range(len(eps))
                 if family(names[motion_to_ep[int(meta[i]["motion_id"])]]) == "stand" and meta[i]["length"] > SEQ + 2]
    dyn_idx = [i for i in range(len(eps))
               if family(names[motion_to_ep[int(meta[i]["motion_id"])]]) == "dynamic" and meta[i]["length"] > SEQ + 2]
    def grab(idxs, n):
        take = rng.choice(idxs, size=min(n, len(idxs)), replace=False)
        out = []
        for i in take:
            T = eps[i]["observation"]["state"].shape[0]
            w = int(rng.integers(0, T - SEQ - 1))
            out.append({k: eps[i]["observation"][k][w:w + SEQ] for k in OBS_KEYS})
        return {k: torch.cat([o[k] for o in out]) for k in OBS_KEYS}
    probe_stand = grab(stand_idx, 16)
    probe_dyn = grab(dyn_idx, 32)
    # window-mean obs per window for scoring
    def win_mean(o):
        return {k: v.view(-1, SEQ, v.shape[-1]).mean(dim=1) for k, v in o.items()}
    probe_stand_m, probe_dyn_m = win_mean(probe_stand), win_mean(probe_dyn)
    # a fixed dynamic z (window-mean of a dynamic episode's encodings via final ckpt later)

    # frozen injected stand probe (generate once)
    s_ep = stand_idx[0]
    ids_s = torch.tensor([meta[s_ep]["motion_id"]], dtype=torch.long, device=inner.device)
    t_s = torch.tensor([meta[s_ep]["t0"] + 4.0], dtype=torch.float32, device=inner.device)
    wrapped.reset(to_numpy=False)
    frozen = inject_read_exact_lean(wrapped, ids_s, t_s, settle_reps=2)

    ckpts = [5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 250000, 300000]
    roll_ckpt_policy = [5000, 25000, 100000, 300000]
    results = []
    fixed_z = None
    policy_obs_cache = {}
    for t in ckpts:
        ck_dir = Path(f"results/bfm-dynamics-rnorm-harvest300k/ckpt_{t}")
        if not (ck_dir / "model").exists() and t == 300000:
            ck_dir = Path("results/bfm-dynamics-rnorm-harvest300k/checkpoint")
        if not (ck_dir / "model").exists() and not (ck_dir.exists() and (ck_dir / "config.json").exists()):
            results.append({"step": t, "missing": True})
            continue
        model = load_model_from_checkpoint_dir(ck_dir, device="cuda")
        model.eval()
        model._obs_normalizer.eval()
        with torch.no_grad():
            norm = lambda d: model._obs_normalizer(tree_map(lambda x: x.to(dev), d))
            # fixed z from a dynamic window via THIS ckpt's B (z drifts with B; use final-model z for policy rollouts instead)
            if fixed_z is None and t == ckpts[-1]:
                pass
            ps, pd_ = norm(probe_stand_m), norm(probe_dyn_m)
            # z for scoring probes: use each window's own encoding (self-consistent conditional scoring)
            def z_of(o_m):
                # window obs mean -> approximate B on the 8-frame window via win obs directly:
                # use per-window next-obs mean is unavailable here; approximate z with B(obs window mean)
                return model.project_z(model._backward_map(o_m))
            zs, zd = z_of(ps), z_of(pd_)
            def sc(o, z):
                return model._discriminator.compute_logits(o, z).flatten().mean().item()
            row = {
                "step": t,
                "expert_stand": sc(ps, zs),
                "expert_dyn": sc(pd_, zd),
                "frozen_stand": sc(norm(frozen), model.project_z(model._backward_map(norm(frozen)))),
            }
            row["M_t_dyn_minus_stand"] = row["expert_dyn"] - row["expert_stand"]
            # policy rollout obs from this ckpt's actor (only selected steps)
            if t in roll_ckpt_policy:
                # rollout under a fixed dynamic z from the FINAL checkpoint model
                obs, _ = wrapped.reset(to_numpy=False)
                obs = tree_map(lambda x: x.to(dev), obs)
                rows_obs = []
                zfix = model.project_z(model._backward_map(pd_))[:1]
                for _ in range(ROLL_STEPS):
                    with torch.no_grad():
                        a = model.act(tree_map(lambda x: x, obs), zfix.expand(obs["state"].shape[0], -1), mean=True)
                    obs, _, _, _, _ = wrapped.step(a, to_numpy=False)
                    obs = tree_map(lambda x: x.to(dev), obs)
                    rows_obs.append(tree_map(lambda x: x.detach().clone(), obs))
                proll = tree_map(lambda *xs: torch.cat([x for x in xs], 0), *rows_obs)
                pm = {k: proll[k].view(-1, SEQ, proll[k].shape[-1])[:, -1].mean(dim=1)[: ROLL_STEPS // SEQ * SEQ // SEQ][:64] for k in ("state", "privileged_state")} if False else None
                # score each rollout frame under its own z(zfix)
                pr = {k: proll[k] for k in ("state", "privileged_state")}
                row["policy_t_self"] = float(model._discriminator.compute_logits(norm(pr), zfix.expand(pr["state"].shape[0], -1)).flatten().mean())
                row["policy_t_activity_abs_action"] = float(model.act(tree_map(lambda x: x, obs), zfix.expand(obs["state"].shape[0], -1), mean=True).abs().mean())
            results.append(row)
        del model
        torch.cuda.empty_cache()
        print(f"ck {t}: {json.dumps({k: round(v, 3) for k, v in row.items() if isinstance(v, (int, float))})}", flush=True)

    out = Path("results/bfmzero-bdx-full/d_formation_replay.json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("saved", out)


if __name__ == "__main__":
    main()
