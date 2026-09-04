"""P0 probes 1+2 (external review 2026-08-21 evening).

Probe 1 — rollout z activity distribution: linear probe (trained on pool
windows, held-out validated) predicts the commanded activity of each z source
that feeds rollouts: expert-z (== tracking-z, proven equivalent), goal-z
(B(next obs)), random-z. If most rollout z are low-activity, freezing is
partly correct execution of the sampling prior.

Probe 2 — actor-z response: fixed stand / dynamic observations, swap
stand-z vs dynamic-z, measure action delta for the 2M actor and a freshly
initialized actor. No response => conditioning dead; response-but-frozen =>
critic/exploration problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

STAND_MARKERS = ("stand", "pose", "idle")
SEQ = 8
N_EP = 240


def family(name):
    n = name.lower()
    return "stand" if any(m in n for m in STAND_MARKERS) else "dynamic"


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

    rng = np.random.default_rng(31)
    ep_sel = rng.choice(len(eps), size=min(N_EP, len(eps)), replace=False)
    Z, act, fam = [], [], []
    with torch.no_grad():
        for e_idx in ep_sel:
            ep = eps[e_idx]
            T = ep["observation"]["state"].shape[0]
            if T < 2 * SEQ + 2:
                continue
            w = int(rng.integers(0, T - 2 * SEQ))
            nxt = {k: ep["observation"][k][w + 1:w + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
            b = model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)
            Z.append(model.project_z(b).flatten().cpu().numpy())
            dv = bdx_motion_activity(ep["observation"]["state"])
            act.append(float(dv[w:w + SEQ].norm(dim=-1).mean()))
            fam.append(family(names[m2e[int(meta[e_idx]["motion_id"])]]))
    Z = np.array(Z); act = np.array(act); fam = np.array(fam)
    y = (fam == "dynamic").astype(int)

    # probe 1: activity of each z source
    half = len(Z) // 2
    sc = StandardScaler().fit(Z[:half])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Z[:half]), y[:half])
    auroc = roc_auc_score(y[half:], clf.predict_proba(sc.transform(Z[half:]))[:, 1])

    def act_dist(zs, tag):
        p_dyn = clf.predict_proba(sc.transform(zs))[:, 1]
        return {"source": tag, "n": len(zs),
                "frac_dynamic(>0.5)": float((p_dyn > 0.5).mean()),
                "p10": float(np.quantile(p_dyn, .1)), "p50": float(np.quantile(p_dyn, .5)),
                "p90": float(np.quantile(p_dyn, .9))}

    with torch.no_grad():
        # goal-z: B(next obs) per single frame, projected
        g_rows = rng.choice(len(eps), 40, replace=False)
        Gz = []
        for e_idx in g_rows:
            ep = eps[e_idx]
            idx = int(rng.integers(SEQ, ep["observation"]["state"].shape[0] - 1))
            o = {k: ep["observation"][k][idx:idx + 1].to(dev) for k in ("state", "privileged_state")}
            Gz.append(model.project_z(model._backward_map(model._obs_normalizer(o))).flatten().cpu().numpy())
        Gz = np.array(Gz)
        Rz = np.array([model.sample_z(1, device=dev).flatten().cpu().numpy() for _ in range(200)])
    probe1 = {
        "probe_holdout_auroc": auroc,
        "expert_z(tracking-z)": act_dist(Z, "expert_z"),
        "goal_z": act_dist(Gz, "goal_z"),
        "random_z": act_dist(Rz, "random_z"),
        "rollout_mix_60_20_20_frac_dynamic": float(0.6 * (clf.predict_proba(sc.transform(Z))[:, 1] > 0.5).mean()
                                                   + 0.2 * (clf.predict_proba(sc.transform(Gz))[:, 1] > 0.5).mean()
                                                   + 0.2 * (clf.predict_proba(sc.transform(Rz))[:, 1] > 0.5).mean()),
    }

    # probe 2: actor-z response (2M actor vs fresh init actor)
    fresh_cfg = agent_cfg
    fresh = fresh_cfg.build(obs_space=None, action_dim=14) if False else None
    # build a fresh model via config (random init) matching archi
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig
    fresh_model = cfg["agent"]["model"]  # dict
    # simpler: perturb nothing — compare 2M actor against itself is useless; use
    # an untrained copy built from the checkpoint config
    import json as _json
    import gymnasium
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig
    with open(Path("results/bfm-dynamics-fullpool-condloss-2m/checkpoint/model/init_kwargs.json")) as f:
        init_kw = _json.load(f)
    _sp = {}
    for k, v in init_kw["obs_space"]["spaces"].items():
        shp = tuple(v["shape"]) if "shape" in v else (len(v["low"]),)
        _sp[k] = gymnasium.spaces.Box(-np.inf, np.inf, shp)
    model_cfg = FBcprAuxModelConfig(**{k: v for k, v in cfg["agent"]["model"].items() if k not in ("device",)} | {"device": "cuda"})
    fresh_m = model_cfg.build(obs_space=gymnasium.spaces.Dict(_sp), action_dim=init_kw["action_dim"])
    fresh_m.to(dev); fresh_m.eval()

    # pick a stand and a dynamic window from pool
    stand_ep = next(i for i in ep_sel if fam[list(ep_sel).index(i)] == "stand") if (fam == "stand").any() else ep_sel[0]
    with torch.no_grad():
        def window_obs(e_idx):
            ep = eps[e_idx]
            T = ep["observation"]["state"].shape[0]
            w = T // 2
            o = {k: ep["observation"][k][w:w + SEQ].to(dev) for k in ("state", "privileged_state", "last_action")}
            o = tree_map(lambda x: x.mean(0, keepdim=True), model._obs_normalizer(o))
            o["history_actor"] = torch.zeros(1, 192, device=dev)  # actor filter requires the key; content unused by probes
            return o
        st_i = next(i for i, f in enumerate(fam) if f == "stand")
        dy_i = next(i for i, f in enumerate(fam) if f == "dynamic")
        obs_st, obs_dy = window_obs(ep_sel[st_i]), window_obs(ep_sel[dy_i])
        z_st = torch.tensor(sc.inverse_transform(clf.coef_)[0] * 0 + 0, device=dev)  # placeholder
        # z prototypes: mean of stand / dynamic window z, projected norms already ~sqrt(128)
        z_stand = torch.tensor(Z[[i for i, f in enumerate(fam) if f == "stand"]].mean(0), dtype=torch.float32, device=dev).unsqueeze(0)
        z_dyn = torch.tensor(Z[[i for i, f in enumerate(fam) if f == "dynamic"]].mean(0), dtype=torch.float32, device=dev).unsqueeze(0)
        for tag, mdl in (("actor_2M", model._actor), ("actor_fresh", fresh_m._actor)):
            for obs_tag, o in (("stand_obs", obs_st), ("dyn_obs", obs_dy)):
                a_st = mdl(o, z_stand, 0.05).mean
                a_dy = mdl(o, z_dyn, 0.05).mean
                probe1[f"probe2_{tag}_{obs_tag}_delta_a"] = float((a_st - a_dy).norm())
                probe1[f"probe2_{tag}_{obs_tag}_cosine"] = float(
                    torch.nn.functional.cosine_similarity(a_st.flatten(), a_dy.flatten(), dim=0))
    out = Path("results/bfmzero-bdx-full/z_rollout_actor_probe.json")
    out.write_text(json.dumps(probe1, indent=2, default=float) + "\n")
    print(json.dumps(probe1, indent=2, default=float))


if __name__ == "__main__":
    main()
