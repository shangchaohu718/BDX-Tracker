"""Probe 3: signal-chain audit under REAL training batch composition.

Rollout the 2M actor in the CPU env (mixed z from pool, mimicking z_buffer),
embed the counterfactual pair
    (true_dynamic_obs + dynamic_z)  vs  (frozen_stand_obs + dynamic_z)
into that policy batch, then propagate through the exact training pipeline
layers and report gaps:

  L1 raw D logit
  L2 reward after reward_norm (+clip5)   [two calib variants: stats from
      policy-batch only vs pair included]
  L3 percentile/rank within batch
  L4 TD target (r + gamma * V(next)) and critic Q(o, a_policy, z)
  L5 actor loss-term gradient norm at those states

Also: Spearman of normalized reward vs rollout activity within the policy
batch (is the ranking behavior-relevant?).

Caveat (accepted by review): "frozen stand" = a real stand-clip window from
the pool (slow/standing expert states), not a synthetically zero-vel frozen
state; sufficient for chain-level questions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity
from scipy.stats import spearmanr
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

STAND_MARKERS = ("stand", "pose", "idle")
SEQ = 8
N_ROLLOUT = 2048
DISCOUNT = 0.95


def family(name):
    n = name.lower()
    return "stand" if any(m in n for m in STAND_MARKERS) else "dynamic"


def to_dev(d, dev):
    return tree_map(lambda x: x.to(dev), d)


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_full.pt", map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]

    model = load_model_from_checkpoint_dir(Path("results/bfm-dynamics-fullpool-condloss-2m/checkpoint"), device="cuda")
    model.eval()
    model._obs_normalizer.eval()
    dev = torch.device("cuda")

    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    m2e = {}
    for i in range(len(names)):
        m2e.setdefault(int(ep_motion[i]), i)

    # ---- z pool from expert windows (mimics z_buffer's 60% expert share) ----
    rng = np.random.default_rng(41)
    z_pool, z_dyn_idx = [], []
    with torch.no_grad():
        for e_idx in rng.choice(len(eps), 200, replace=False):
            ep = eps[e_idx]
            T = ep["observation"]["state"].shape[0]
            if T < 2 * SEQ + 2:
                continue
            w = int(rng.integers(0, T - 2 * SEQ))
            nxt = {k: ep["observation"][k][w + 1:w + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
            b = model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)
            z_pool.append(model.project_z(b))
            z_dyn_idx.append(family(names[m2e[int(meta[e_idx]["motion_id"])]]) == "dynamic")
    z_pool = torch.cat(z_pool)
    z_dyn_idx = np.array(z_dyn_idx)

    # ---- rollout the 2M actor with mixed z ----
    obs, _ = wrapped.reset(to_numpy=False)
    rows, acts, zrows, dvs = [], [], [], []
    cur_z = z_pool[0:1]
    import time
    t0 = time.time()
    with torch.no_grad():
        for t in range(N_ROLLOUT):
            if t % 100 == 0:
                cur_z = z_pool[rng.integers(0, len(z_pool))][None]
            o = to_dev(obs, dev)
            a = model.act(o, cur_z.expand(o["state"].shape[0], -1), mean=False)
            rows.append(tree_map(lambda x: x.detach().clone(), o))
            acts.append(a.detach().clone())
            zrows.append(cur_z.expand(o["state"].shape[0], -1).detach().clone())
            dvs.append(bdx_motion_activity(o["state"]).detach().cpu())
            obs, _, _, _, _ = wrapped.step(a, to_numpy=False)
    pol_obs = tree_map(lambda *xs: torch.cat([x for x in xs]), *rows)
    pol_act = torch.cat(acts)
    pol_z = torch.cat(zrows)
    pol_dv = torch.cat(dvs).numpy()
    print(f"rollout done in {time.time()-t0:.0f}s, activity mean={pol_dv.mean():.2f}", flush=True)

    # ---- counterfactual pair from pool windows ----
    dy_e = next(i for i in range(len(eps)) if family(names[m2e[int(meta[i]['motion_id'])]]) == "dynamic"
                and eps[i]["observation"]["state"].shape[0] > 2 * SEQ + 2)
    st_e = next(i for i in range(len(eps)) if family(names[m2e[int(meta[i]['motion_id'])]]) == "stand"
                and eps[i]["observation"]["state"].shape[0] > 2 * SEQ + 2)
    def grab(e_idx):
        ep = eps[e_idx]
        T = ep["observation"]["state"].shape[0]
        w = T // 3
        cur = tree_map(lambda x: x[w:w + 1].to(dev), {k: ep["observation"][k] for k in ("state", "privileged_state", "last_action")})
        nxt = tree_map(lambda x: x[w + 1:w + 2].to(dev), {k: ep["observation"][k] for k in ("state", "privileged_state", "last_action")})
        cur = model._obs_normalizer(cur); nxt = model._obs_normalizer(nxt)
        for d in (cur, nxt):
            d["history_actor"] = torch.zeros(1, 192, device=dev)
        return cur, nxt
    dy_o, dy_next = grab(dy_e)
    st_o, st_next = grab(st_e)
    dyn_z = z_pool[z_dyn_idx][rng.integers(0, z_dyn_idx.sum())][None]  # a dynamic z

    with torch.no_grad():
        # L1 raw logits on batch + pair
        r_batch = model._discriminator.compute_reward(pol_obs, pol_z).flatten().float()
        r_dy = model._discriminator.compute_reward({k: dy_o[k] for k in ("state", "privileged_state")}, dyn_z).flatten().float()
        r_st = model._discriminator.compute_reward({k: st_o[k] for k in ("state", "privileged_state")}, dyn_z).flatten().float()
        L1 = (r_dy.item(), r_st.item(), r_dy.item() - r_st.item())
        # L2 reward_norm (policy-batch stats) + clip 5
        mu, sd = r_batch.mean(), r_batch.std()
        def rn(r):
            x = (r - mu) / (sd + 1e-5)
            return torch.clamp(x, -5, 5)
        L2 = (rn(r_dy).item(), rn(r_st).item(), rn(r_dy).item() - rn(r_st).item())
        # L3 percentile of pair inside batch distribution
        L3 = (float((r_batch < r_dy).float().mean()), float((r_batch < r_st).float().mean()))
        # L4 TD target + Q at the pair (actor-sampled actions)
        def td_and_q(o, o_next):
            dist = model._actor(o, dyn_z, model.cfg.actor_std)
            a_pair = dist.sample(clip=0.3)
            nq = model._target_critic(o_next, dyn_z, model._actor(o_next, dyn_z, model.cfg.actor_std).sample(clip=0.3))
            nV = nq.mean(0).flatten()
            target = rn(model._discriminator.compute_reward({k: o[k] for k in ("state","privileged_state")}, dyn_z)).flatten() + DISCOUNT * nV
            q = model._critic(o, dyn_z, a_pair).mean(0).flatten()
            return target.item(), q.mean().item()
        t_dy, q_dy = td_and_q(dy_o, dy_next)
        t_st, q_st = td_and_q(st_o, st_next)
        L4 = {"td_target_gap": t_dy - t_st, "q_gap": q_dy - q_st,
              "td_dy": t_dy, "td_st": t_st, "q_dy": q_dy, "q_st": q_st}
        L5 = {"note": "computed after no_grad block below"}
        # ranking relevance: normalized reward vs activity within policy batch
        rn_batch = rn(r_batch).cpu().numpy()
        sp = float(spearmanr(pol_dv, rn_batch).statistic)
        sp_raw = float(spearmanr(pol_dv, r_batch.cpu().numpy()).statistic)

    out = {
        "batch": {"n": N_ROLLOUT, "activity_mean": float(pol_dv.mean()),
                  "raw_logit_mean": float(r_batch.mean()), "raw_logit_std": float(r_batch.std())},
        "L1_raw_logit": {"dyn_pair": L1[0], "stand_pair": L1[1], "gap": L1[2]},
        "L2_reward_norm_clip5": {"dyn_pair": L2[0], "stand_pair": L2[1], "gap": L2[2],
                                  "batch_mu": float(mu), "batch_sd": float(sd)},
        "L3_percentile_in_batch": {"dyn_pair": L3[0], "stand_pair": L3[1]},
        "L4_td_q": L4,
        "L5_actor_grad": L5,
        "ranking_spearman_rhat_vs_activity": sp,
        "ranking_spearman_raw_vs_activity": sp_raw,
    }
    # L5 actor gradient norms at the pair states (needs autograd)
    def actor_grad(o):
        dist = model._actor(o, dyn_z, model.cfg.actor_std)
        a = dist.sample(clip=0.3)
        q = model._critic(o, dyn_z, a).mean()
        g = torch.autograd.grad(q, list(model._actor.parameters()), retain_graph=False)
        return float(torch.sqrt(sum((x.detach() ** 2).sum() for x in g)))
    for prm in model._actor.parameters():
        prm.requires_grad_(True)
    L5 = {"grad_dy": actor_grad(dy_o), "grad_st": actor_grad(st_o)}
    for prm in model._actor.parameters():
        prm.requires_grad_(False)
    out["L5_actor_grad"] = L5
    p = Path("results/bfmzero-bdx-full/probe3_signal_chain.json")
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
