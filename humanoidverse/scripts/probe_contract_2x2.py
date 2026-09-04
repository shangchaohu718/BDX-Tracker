"""Same-cache old/new contract 2x2 decomposition (final ruling item 6.1).

ONE rollout of the 2M actor under mixed pool z; ONE set of D logits; then
compute Spearman(r, activity) in the 2x2 grid {old,new dyn-subset} x
{old,new activity}, plus subset Jaccard, old/new activity correlation, and
contributions of the channels the old slice dropped/mixed.

old activity = ||state[:, 20:34]|| (dof_vel[6:14]+gravity+ang_vel mix)
new activity = ||state[:, 14:28]|| (true dof_vel)
old dyn-subset = classifier trained on old-activity labels; new likewise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

STAND_MARKERS = ("stand", "pose", "idle")
SEQ = 8
N_ROLLOUT = 2048


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

    model = load_model_from_checkpoint_dir(Path(__import__("os").environ.get("X22_CKPT", "results/bfm-dynamics-fullpool-condloss-2m/checkpoint")), device="cuda")
    model.eval()
    model._obs_normalizer.eval()
    dev = torch.device("cuda")

    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    m2e = {}
    for i in range(len(names)):
        m2e.setdefault(int(ep_motion[i]), i)

    # z pool with OLD and NEW activity labels per window
    rng = np.random.default_rng(71)
    Z, a_old, a_new, fam = [], [], [], []
    with torch.no_grad():
        for e_idx in rng.permutation(len(eps)):
            ep = eps[e_idx]
            T = ep["observation"]["state"].shape[0]
            if T < 2 * SEQ + 2:
                continue
            w = int(rng.integers(0, T - 2 * SEQ))
            st = ep["observation"]["state"][w:w + SEQ]
            nxt = {k: ep["observation"][k][w + 1:w + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
            b = model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)
            Z.append(model.project_z(b).flatten().cpu().numpy())
            a_old.append(float(st[:, 20:34].norm(dim=-1).mean()))
            a_new.append(float(st[:, 14:28].norm(dim=-1).mean()))
            fam.append(family(names[m2e[int(meta[e_idx]["motion_id"])]]) == "dynamic")
            if len(Z) >= 400:
                break
    Z = np.array(Z); a_old = np.array(a_old); a_new = np.array(a_new)
    fam = np.array(fam)
    tr, te = slice(0, 320), slice(320, None)
    sc = StandardScaler().fit(Z[tr])
    clf_old = LogisticRegression(max_iter=2000).fit(sc.transform(Z[tr]), (a_old[tr] > np.median(a_old[tr])).astype(int))
    clf_new = LogisticRegression(max_iter=2000).fit(sc.transform(Z[tr]), (a_new[tr] > np.median(a_new[tr])).astype(int))

    z_t = torch.tensor(Z, dtype=torch.float32, device=dev)
    obs, _ = wrapped.reset(to_numpy=False)
    r_rows, act_old_rows, act_new_rows, dv_front_rows, ang_rows, zsel = [], [], [], [], [], []
    with torch.no_grad():
        for t in range(N_ROLLOUT):
            if t % 100 == 0:
                zi = int(rng.integers(0, len(Z)))
            o = tree_map(lambda x: x.to(dev), obs)
            a = model.act(o, z_t[zi:zi + 1].expand(o["state"].shape[0], -1), mean=False)
            s = o["state"]
            r_rows.append(float(model._discriminator.compute_reward(o, z_t[zi:zi + 1].expand(o["state"].shape[0], -1)).flatten()[0]))
            act_old_rows.append(float(s[:, 20:34].norm(dim=-1)[0]))
            act_new_rows.append(float(s[:, 14:28].norm(dim=-1)[0]))
            dv_front_rows.append(float(s[:, 14:20].norm(dim=-1)[0]))
            ang_rows.append(float(s[:, 31:34].norm(dim=-1)[0]))
            zsel.append(zi)
            obs, _, _, _, _ = wrapped.step(a, to_numpy=False)
    r = np.array(r_rows); ao = np.array(act_old_rows); an = np.array(act_new_rows)
    vf = np.array(dv_front_rows); av = np.array(ang_rows); zs = np.array(zsel)

    p_old = clf_old.predict_proba(sc.transform(Z[zs]))[:, 1] > 0.5
    p_new = clf_new.predict_proba(sc.transform(Z[zs]))[:, 1] > 0.5

    def sp(m, act):
        return float(spearmanr(r[m], act[m]).statistic) if m.sum() > 30 else None

    res = {
        "grid": {
            "old_subset_x_old_activity": sp(p_old, ao),
            "old_subset_x_new_activity": sp(p_old, an),
            "new_subset_x_old_activity": sp(p_new, ao),
            "new_subset_x_new_activity": sp(p_new, an),
        },
        "subset_jaccard_old_new": float((p_old & p_new).sum() / max(1, (p_old | p_new).sum())),
        "old_new_activity_spearman": float(spearmanr(ao, an).statistic),
        "channel_contributions_vs_old_activity": {
            "dof_vel_front6": float(spearmanr(vf, ao).statistic),
            "base_ang_vel": float(spearmanr(av, ao).statistic),
        },
        "channel_contributions_vs_new_activity": {
            "dof_vel_front6": float(spearmanr(vf, an).statistic),
            "base_ang_vel": float(spearmanr(av, an).statistic),
        },
        "frac_old_subset": float(p_old.mean()), "frac_new_subset": float(p_new.mean()),
        "holdout_clf_agreement_with_fam": {
            "old": float((clf_old.predict(sc.transform(Z[te])) == fam[te]).mean()),
            "new": float((clf_new.predict(sc.transform(Z[te])) == fam[te]).mean()),
        },
    }
    import os as _os
    out = Path("results/bfmzero-bdx-full") / _os.environ.get("X22_OUT", "probe_contract_2x2.json")
    assert not out.exists()
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
