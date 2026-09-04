"""Probe A (gate for the conditional-D repair): does z carry predictive
information about the behavior it is supposed to command?

Setup matches training semantics: z_i = mean of B(next_obs) over an 8-frame
expert window (same as encode_expert / tracking-z construction), B from the
rnorm-3M checkpoint.

Labels (from raw pool obs, NOT from z):
  - binary: clip family stand vs dynamic (clip name)
  - regression: window activity (mean dof_vel L2) and FUTURE activity
    (next K frames after the window), honoring z's "occupancy summary"
    contract.

Probes: linear logistic, 2-layer MLP, kNN — trained on windows from 80% of
clips, evaluated on held-out CLIPS (no adjacent-frame leakage).

Gate (per external review): unseen-clip stand/dynamic AUROC > 0.8 with both
linear and nonlinear probes => z informative, D fails to use it; linear low /
MLP high => conditioning architecture issue; all ~random => fix B/z first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    load_expert_trajectories_from_motion_lib,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

STAND_MARKERS = ("stand", "pose", "idle")
SEQ = 8
WINDOWS_PER_EP = 8
FUTURE_K = 25  # frames after the window (0.5 s at 50 Hz)


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

    model = load_model_from_checkpoint_dir(Path("results/bfm-dynamics-pool-rnorm-3m/checkpoint"), device="cuda")
    model.eval()
    model._obs_normalizer.eval()
    dev = torch.device("cuda")

    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    motion_to_ep = {}
    for i in range(len(names)):
        motion_to_ep.setdefault(int(ep_motion[i]), i)

    rng = np.random.default_rng(23)
    Z, fam_labels, act_win, act_future, ep_ids = [], [], [], [], []
    with torch.no_grad():
        for e_idx in range(len(eps)):
            ep = eps[e_idx]
            T = ep["observation"]["state"].shape[0]
            if T < SEQ + FUTURE_K + 2:
                continue
            fam = family(names[motion_to_ep[int(meta[e_idx]["motion_id"])]])
            w_pos = rng.integers(0, T - SEQ - FUTURE_K, size=WINDOWS_PER_EP)
            for w in w_pos:
                w = int(w)
                nxt = {k: ep["observation"][k][w + 1:w + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
                nxt_n = model._obs_normalizer(nxt)
                b = model._backward_map(nxt_n).mean(dim=0, keepdim=True)
                z = model.project_z(b).flatten().cpu().numpy()
                dv = bdx_motion_activity(ep["observation"]["state"])
                Z.append(z)
                fam_labels.append(fam)
                act_win.append(float(dv[w:w + SEQ].norm(dim=-1).mean()))
                act_future.append(float(dv[w + SEQ:w + SEQ + FUTURE_K].norm(dim=-1).mean()))
                ep_ids.append(e_idx)
    Z = np.array(Z); fam = np.array(fam_labels)
    act_w = np.array(act_win); act_f = np.array(act_future); ep = np.array(ep_ids)
    n_eps = len(np.unique(ep))
    print(f"windows={len(Z)} episodes={n_eps} stand_share={float((fam=='stand').mean()):.3f}", flush=True)

    # held-out CLIP split
    tr_eps, te_eps = train_test_split(np.unique(ep), test_size=0.2, random_state=7)
    tr = np.isin(ep, tr_eps); te = ~tr
    y = (fam == "dynamic").astype(int)
    if len(np.unique(y[te])) < 2:
        raise RuntimeError("held-out split has a single family; widen STAND_MARKERS")

    sc = StandardScaler().fit(Z[tr])
    Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])

    out = {"n_windows": int(len(Z)), "n_episodes": int(n_eps),
           "stand_share": float((fam == "stand").mean())}

    lin = LogisticRegression(max_iter=2000).fit(Ztr, y[tr])
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=800, random_state=0).fit(Ztr, y[tr])
    knn = KNeighborsClassifier(15).fit(Ztr, y[tr])
    for name, clf in (("linear", lin), ("mlp", mlp), ("knn", knn)):
        pred = clf.predict_proba(Zte)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Zte)
        out[f"auroc_dynamic_{name}"] = float(roc_auc_score(y[te], pred))
        # train-set AUROC too (memorization check)
        ptr = clf.predict_proba(Ztr)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Ztr)
        out[f"auroc_dynamic_{name}_train"] = float(roc_auc_score(y[tr], ptr))

    for tag, target in (("win", act_w), ("future", act_f)):
        ridge = Ridge().fit(Ztr, target[tr])
        pr = ridge.predict(Zte)
        out[f"activity_{tag}_spearman_linear"] = float(spearmanr(target[te], pr).statistic)
        # kNN regression proxy
        knr = KNeighborsClassifier(15).fit(Ztr, np.digitize(target[tr], np.quantile(target[tr], [0.33, 0.66])))
        pk = knr.predict(Zte)
        out[f"activity_{tag}_spearman_knn"] = float(spearmanr(target[te], pk).statistic)
        out[f"activity_{tag}_train_mean"] = float(target[tr].mean())

    out_path = Path("results/bfmzero-bdx-full/z_activity_probe.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
