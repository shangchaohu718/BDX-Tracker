"""Probe 3B: stratified re-test of the policy-domain ranking + real-history
re-check + reward-map offline comparison + z OOD audit (external ruling).

Fixes over probe3:
  - Spearman(r, activity) stratified by z's predicted dynamic-ness
  - alignment metric: r vs -|a_obs - a_target(z)| (ridge z->activity)
  - tracking proxy: r vs -distance(rollout obs, z's source expert window)
  - Mahalanobis OOD of goal-z / random-z vs expert-z distribution
  - counterfactual pair with REAL history_actor built from pool frames
  - offline reward-map comparison: noclip / clip5 / tanh(2)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env
from humanoidverse.utils.bdx_state import bdx_motion_activity, build_bdx_actor_history

STAND_MARKERS = ("stand", "pose", "idle")
SEQ = 8
N_ROLLOUT = 2048
HIST_KEYS_DIMS = {"actions": 14, "base_ang_vel": 3, "dof_pos": 14, "dof_vel": 14, "projected_gravity": 3}


def family(name):
    n = name.lower()
    return "stand" if any(m in n for m in STAND_MARKERS) else "dynamic"


def to_dev(d, dev):
    return tree_map(lambda x: x.to(dev), d)


def build_history(ep, t, dev):
    """history_actor layout: sorted keys x last-4 frames (see _get_obs_long_history)."""
    blocks = []
    for key in sorted(HIST_KEYS_DIMS):  # actions, base_ang_vel, dof_pos, dof_vel, projected_gravity
        for k in range(4, 0, -1):  # most recent last, matching handler semantics approximately
            i = max(0, t - k)
            if key == "actions":
                blocks.append(ep["observation"]["last_action"][i].to(dev))
            elif key == "base_ang_vel":
                blocks.append(ep["observation"]["state"][i][31:34].to(dev))
            elif key == "dof_pos":
                blocks.append(ep["observation"]["state"][i][0:14].to(dev))
            elif key == "dof_vel":
                blocks.append(ep["observation"]["state"][i][14:28].to(dev))
            else:
                blocks.append(ep["observation"]["state"][i][28:31].to(dev))
    return torch.cat(blocks).unsqueeze(0)  # [1, 192]


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_full.pt", map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]

    checkpoint = Path(os.environ.get("DZ_CKPT", "results/bfm-dynamics-combo300k/checkpoint"))
    model = load_model_from_checkpoint_dir(checkpoint, device="cuda")
    model.eval()
    model._obs_normalizer.eval()
    dev = torch.device("cuda")

    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    m2e = {}
    for i in range(len(names)):
        m2e.setdefault(int(ep_motion[i]), i)

    # ---- z pool with per-z provenance + probes (z->p_dyn, z->activity) ----
    rng = np.random.default_rng(53)
    n_train = 160
    Z, z_act, z_fam, z_ep = [], [], [], []
    with torch.no_grad():
        for e_idx in rng.permutation(len(eps)):
            ep = eps[e_idx]
            T = ep["observation"]["state"].shape[0]
            if T < 3 * SEQ + 2:
                continue
            w = int(rng.integers(0, T - 2 * SEQ))
            nxt = {k: ep["observation"][k][w + 1:w + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
            b = model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)
            Z.append(model.project_z(b).flatten().cpu().numpy())
            dv = bdx_motion_activity(ep["observation"]["state"])
            z_act.append(float(dv[w:w + SEQ].norm(dim=-1).mean()))
            z_fam.append(family(names[m2e[int(meta[e_idx]["motion_id"])]]))
            z_ep.append((e_idx, w))
            if len(Z) >= 400:
                break
    Z = np.array(Z); z_act = np.array(z_act); z_fam = np.array(z_fam); z_ep = np.array(z_ep, dtype=object)
    tr, te = slice(0, n_train), slice(n_train, None)
    sc = StandardScaler().fit(Z[tr])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Z[tr]), (z_fam[tr] == "dynamic").astype(int))
    ridge = Ridge().fit(sc.transform(Z[tr]), z_act[tr])
    a_target_all = ridge.predict(sc.transform(Z))
    sp_reg = float(spearmanr(z_act[te], ridge.predict(sc.transform(Z[te]))).statistic)
    print(f"z probes: ridge holdout spearman={sp_reg:.3f}", flush=True)

    # ---- rollout with mixed z, record provenance ----
    z_t = torch.tensor(Z, dtype=torch.float32, device=dev)
    obs, _ = wrapped.reset(to_numpy=False)
    rows, acts, zrows, dvs, zprov = [], [], [], [], []
    with torch.no_grad():
        for t in range(N_ROLLOUT):
            if t % 100 == 0 or t == 0:
                zi = int(rng.integers(0, len(Z)))
                cur_z = z_t[zi:zi + 1]
                cur_prov = zi
            o = to_dev(obs, dev)
            a = model.act(o, cur_z.expand(o["state"].shape[0], -1), mean=False)
            rows.append(tree_map(lambda x: x.detach().clone(), o))
            acts.append(a.detach().clone())
            zrows.append(cur_z.detach().clone())
            dvs.append(bdx_motion_activity(o["state"]).detach().cpu())
            zprov.append(cur_prov)
            obs, _, _, _, _ = wrapped.step(a, to_numpy=False)
    pol_obs = tree_map(lambda *xs: torch.cat([x for x in xs]), *rows)
    pol_act = torch.cat(acts)
    pol_z = torch.cat(zrows)
    pol_dv = torch.cat(dvs).numpy().flatten()
    zprov = np.array(zprov)
    print(f"rollout done, activity mean={pol_dv.mean():.2f}", flush=True)

    with torch.no_grad():
        r_raw = model._discriminator.compute_reward(pol_obs, pol_z).flatten().float().cpu().numpy()
        mu, sd = float(r_raw.mean()), float(r_raw.std())
        r_hat = (r_raw - mu) / (sd + 1e-5)

    # ---- stratified Spearman + alignment + tracking proxy ----
    p_dyn_z = clf.predict_proba(sc.transform(Z[zprov]))[:, 1]
    a_tgt = a_target_all[zprov]
    dyn_mask = p_dyn_z > 0.5
    align = -np.abs(pol_dv - a_tgt)
    # tracking proxy: distance from rollout privileged obs to z's source window obs
    with torch.no_grad():
        src_obs = []
        for zi in zprov:
            e_idx, w = z_ep[zi]
            src_obs.append(eps[e_idx]["observation"]["privileged_state"][w + SEQ - 1].to(dev))
        src_obs = torch.stack(src_obs)
        priv = model._obs_normalizer(pol_obs)["privileged_state"]
        dist = (priv - src_obs).norm(dim=-1).cpu().numpy()
    res = {
        "z_probe_ridge_holdout_spearman": sp_reg,
        "rollout_activity_mean": float(pol_dv.mean()),
        "frac_steps_under_dyn_z": float(dyn_mask.mean()),
        "spearman_overall": float(spearmanr(r_hat, pol_dv).statistic),
        "spearman_within_dyn_z": float(spearmanr(r_hat[dyn_mask], pol_dv[dyn_mask]).statistic) if dyn_mask.sum() > 30 else None,
        "spearman_within_stand_z": float(spearmanr(r_hat[~dyn_mask], pol_dv[~dyn_mask]).statistic) if (~dyn_mask).sum() > 30 else None,
        "spearman_align_overall": float(spearmanr(r_hat, align).statistic),
        "spearman_align_within_dyn_z": float(spearmanr(r_hat[dyn_mask], align[dyn_mask]).statistic) if dyn_mask.sum() > 30 else None,
        "spearman_tracking_proxy_overall": float(spearmanr(r_hat, -dist).statistic),
    }

    # ---- OOD: goal-z / random-z vs expert-z distribution ----
    with torch.no_grad():
        Gz = []
        for e_idx in rng.choice(len(eps), 60, replace=False):
            ep = eps[e_idx]
            i = int(rng.integers(SEQ, ep["observation"]["state"].shape[0] - 1))
            o = {k: ep["observation"][k][i:i + 1].to(dev) for k in ("state", "privileged_state")}
            Gz.append(model.project_z(model._backward_map(model._obs_normalizer(o))).flatten().cpu().numpy())
        Gz = np.array(Gz)
        Rz = np.array([model.sample_z(1, device=dev).flatten().cpu().numpy() for _ in range(200)])
    mu_z, sd_z = Z[tr].mean(0), Z[tr].std(0) + 1e-6
    def mah(X):
        return float(np.median(np.sqrt(((X - mu_z) / sd_z) ** 2).mean(-1)))
    res["ood_mahalanobis_median"] = {"expert_z_ref": mah(Z[te]), "goal_z": mah(Gz), "random_z": mah(Rz)}

    # ---- real-history counterfactual pair + reward maps ----
    dy_e = next(i for i in range(len(eps)) if family(names[m2e[int(meta[i]["motion_id"])]]) == "dynamic"
                and eps[i]["observation"]["state"].shape[0] > 3 * SEQ + 4)
    st_e = next(i for i in range(len(eps)) if family(names[m2e[int(meta[i]["motion_id"])]]) == "stand"
                and eps[i]["observation"]["state"].shape[0] > 3 * SEQ + 4)
    dyn_z1 = torch.tensor(Z[[i for i, f in enumerate(z_fam) if f == "dynamic"]].mean(0), dtype=torch.float32, device=dev).unsqueeze(0)
    def grab_hist(e_idx):
        ep = eps[e_idx]
        w = ep["observation"]["state"].shape[0] // 3
        assert w >= 3, "history needs a full 4-frame prefix"
        cur = tree_map(lambda x: x[w:w + 1].to(dev), {k: ep["observation"][k] for k in ("state", "privileged_state", "last_action")})
        cur = model._obs_normalizer(cur)
        cur["history_actor"] = build_bdx_actor_history(
            ep["observation"]["state"], ep["observation"]["last_action"], w
        ).to(dev).unsqueeze(0)
        return cur
    dy_o, st_o = grab_hist(dy_e), grab_hist(st_e)
    with torch.no_grad():
        ld = model._discriminator.compute_logits({k: dy_o[k] for k in ("state", "privileged_state")}, dyn_z1).flatten().item()
        ls = model._discriminator.compute_logits({k: st_o[k] for k in ("state", "privileged_state")}, dyn_z1).flatten().item()
        q_dy = model._critic(dy_o, dyn_z1, model._actor(dy_o, dyn_z1, model.cfg.actor_std).sample(clip=0.3)).mean().item()
        q_st = model._critic(st_o, dyn_z1, model._actor(st_o, dyn_z1, model.cfg.actor_std).sample(clip=0.3)).mean().item()
    def maps(r):
        return {"noclip": (r - mu) / (sd + 1e-5),
                "clip5": float(np.clip((r - mu) / (sd + 1e-5), -5, 5)),
                "tanh2": float(np.tanh(((r - mu) / (sd + 1e-5)) / 2))}
    res["real_history_pair"] = {
        "L1_gap": ld - ls, "L1_dyn": ld, "L1_stand": ls,
        "L4_q_gap": q_dy - q_st,
        "maps_dyn": maps(ld), "maps_stand": maps(ls),
        "map_gaps": {k: maps(ld)[k] - maps(ls)[k] for k in ("noclip", "clip5", "tanh2")},
    }

    # ---- directional gradient at a REAL policy obs: swap z ----
    i_real = int(np.argmax(pol_dv))  # a high-activity real rollout state (real history inside)
    o_real = tree_map(lambda x: x[i_real:i_real + 1].clone(), pol_obs)
    st_z1 = torch.tensor(Z[[i for i, f in enumerate(z_fam) if f == "stand"]].mean(0), dtype=torch.float32, device=dev).unsqueeze(0)
    for prm in model._actor.parameters():
        prm.requires_grad_(True)
    def grad_q(o, z):
        a = model._actor(o, z, model.cfg.actor_std).sample(clip=0.3)
        q = model._critic(o, z, a).mean()
        g = torch.autograd.grad(q, list(model._actor.parameters()))
        return torch.cat([x.flatten() for x in g])
    g_dy = grad_q(o_real, dyn_z1)
    g_st = grad_q(o_real, st_z1)
    for prm in model._actor.parameters():
        prm.requires_grad_(False)
    res["directional_grad_real_obs"] = {
        "cosine_dyn_vs_stand": float(torch.nn.functional.cosine_similarity(g_dy, g_st, dim=0)),
        "norm_dyn": float(g_dy.norm()), "norm_stand": float(g_st.norm()),
    }

    import datetime
    res["_provenance"] = {
        "checkpoint": str(os.environ.get("DZ_CKPT", "results/bfm-dynamics-combo300k/checkpoint")),
        "state_contract": "bdx-state-v2 (dof_vel 14:28)",
        "history_contract": "newest-first-v2 (t,t-1,t-2,t-3)",
        "created_at": datetime.datetime.now().isoformat(),
    }
    out = Path(os.environ.get("DZ_OUTPUT", "results/bfmzero-bdx-full/probe3b_stratified.json"))
    assert not out.exists(), f"refusing to overwrite existing artifact: {out}"
    tmp = out.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(res, indent=2) + "\n")
    tmp.rename(out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
