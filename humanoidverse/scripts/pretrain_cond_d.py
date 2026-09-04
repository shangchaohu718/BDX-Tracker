"""Fresh-D conditional pretraining on Pool-Full matched/mismatched pairs.

Per approved plan (2026-08-22): train a FRESH discriminator (B/normalizer/z
come from the 2M checkpoint — Probe A proved z carries full information)
with the conditional ranking loss + a light logit-magnitude regularizer
(prevents absolute drift while only relative order is supervised). Early
stop on held-out clips once the counterfactual margin target is met.

Output: torch.save {"discriminator": state_dict, "stats": {...}}.
Acceptance (offline, then RL gate): held-out (dyn_obs,dyn_z)-(stand_obs,dyn_z)
margin >= 1.0 and cond margin positive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from humanoidverse.utils.bdx_state import bdx_motion_activity
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env

STAND_MARKERS = ("stand", "pose", "idle")
SEQ = 8
LOGIT_REG = 0.01
MARGIN = 1.0
TARGET_MARGIN = 1.0
MAX_STEPS = 4000
LR = 1e-5


def family(name):
    n = name.lower()
    return "stand" if any(m in n for m in STAND_MARKERS) else "dynamic"


def main():
    cfg = json.loads(Path("results/bfm-dynamics-baseline/config.json").read_text())
    wrapped = build_cpu_env(cfg, torch.device("cuda"))
    inner = wrapped._env
    agent_cfg = FBcprAuxAgentConfig(**cfg["agent"])
    kbuf = load_expert_trajectories_from_motion_lib(inner, agent_cfg, device="cpu", base_ang_vel_frame="body")
    pool = torch.load(os.environ.get("BFM_COND_POOL", "humanoidverse/data/bdx_expert_sim_pool_full.pt"), map_location="cpu")
    eps, meta = pool["episodes"], pool["meta"]

    model = load_model_from_checkpoint_dir(Path("results/bfm-dynamics-fullpool-condloss-2m/checkpoint"), device="cuda")
    model.eval()
    model._obs_normalizer.eval()
    dev = torch.device("cuda")

    # fresh discriminator with the checkpoint's architecture
    import gymnasium
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig
    init_kw = json.load(open(Path("results/bfm-dynamics-fullpool-condloss-2m/checkpoint/model/init_kwargs.json")))
    _sp = {k: gymnasium.spaces.Box(-np.inf, np.inf, tuple(v["shape"])) for k, v in init_kw["obs_space"]["spaces"].items()}
    model_cfg = {k: v for k, v in cfg["agent"]["model"].items() if k != "device"} | {"device": "cuda"}
    model_cfg["archi"]["discriminator"]["conditioning"] = os.environ.get("BDX_D_CONDITIONING", "concat")
    mc = FBcprAuxModelConfig(**model_cfg)
    fresh = mc.build(obs_space=gymnasium.spaces.Dict(_sp), action_dim=init_kw["action_dim"]).to(dev)
    disc = fresh._discriminator
    for p in disc.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(disc.parameters(), lr=LR)

    names = [str(x) for x in kbuf.file_names]
    starts = kbuf.start_idx[:, 0].long()
    ep_motion = kbuf.storage["motion_id"][starts].long()
    m2e = {}
    for i in range(len(names)):
        m2e.setdefault(int(ep_motion[i]), i)
    fams = np.array([family(names[m2e[int(m["motion_id"])]]) for m in meta])

    rng = np.random.default_rng(67)
    idx_all = np.arange(len(eps))
    tr_eps = set(rng.choice(idx_all[fams == "dynamic"], 420, replace=False).tolist() + rng.choice(idx_all[fams == "stand"], 180, replace=False).tolist())
    # fixed held-out counterfactual probes
    def probe_state(e_idx):
        ep = eps[e_idx]
        w = ep["observation"]["state"].shape[0] // 3
        o = tree_map(lambda x: x[w:w + 1].to(dev), {k: ep["observation"][k] for k in ("state", "privileged_state")})
        return model._obs_normalizer(o)
    dy_e = int(rng.choice([i for i in idx_all if fams[i] == "dynamic" and i not in tr_eps]))
    st_e = int(rng.choice([i for i in idx_all if fams[i] == "stand" and i not in tr_eps]))
    pr_dy, pr_st = probe_state(dy_e), probe_state(st_e)

    def encode(ep, w):
        with torch.no_grad():
            nxt = {k: ep["observation"][k][w + 1:w + 1 + SEQ].to(dev) for k in ("state", "privileged_state")}
            b = model._backward_map(model._obs_normalizer(nxt)).mean(0, keepdim=True)
            return model.project_z(b)

    # precompute z + activity per training window (one window per episode, 2 for long)
    data = []
    for e_idx in tr_eps:
        ep = eps[e_idx]
        T = ep["observation"]["state"].shape[0]
        n_w = 2 if T > 3 * SEQ + 20 else 1
        for _ in range(n_w):
            w = int(rng.integers(0, T - 2 * SEQ))
            dv = float(bdx_motion_activity(ep["observation"]["state"][w:w + SEQ]).mean())
            data.append((e_idx, w, dv))
    print(f"train windows={len(data)}", flush=True)
    z_cache = {}
    def get_z(e_idx, w):
        key = (e_idx, w)
        if key not in z_cache:
            z_cache[key] = encode(eps[e_idx], w)
        return z_cache[key]

    hist = []
    for step in range(MAX_STEPS + 1):
        if step % 200 == 0:
            with torch.no_grad():
                zd = torch.cat([get_z(e, w) for e, w in [(dy_e, eps[dy_e]["observation"]["state"].shape[0] // 3)]])
                ld = disc.compute_logits(pr_dy, zd).item()
                ls = disc.compute_logits(pr_st, zd).item()
            hist.append({"step": step, "heldout_margin_dyn_minus_stand": ld - ls, "ld": ld, "ls": ls})
            print(hist[-1], flush=True)
            if step > 500 and (ld - ls) >= TARGET_MARGIN:
                break
        if step == MAX_STEPS:
            break
        picks = [data[i] for i in rng.integers(0, len(data), 96)]
        acts = np.array([p[2] for p in picks])
        med = np.median(acts)
        hi = [p for p, a in zip(picks, acts) if a > med]
        lo = [p for p, a in zip(picks, acts) <= med] if False else [p for p, a in zip(picks, acts) if a <= med]
        if len(hi) < 2 or len(lo) < 2:
            continue
        # Counterfactual pairs must be one-to-one.  The previous code let the
        # median split produce (47,49), then caught and skipped the resulting
        # shape error, silently discarding many pretraining batches.
        n_pair = min(len(hi), len(lo))
        hi = [hi[i] for i in rng.choice(len(hi), n_pair, replace=False)]
        lo = [lo[i] for i in rng.choice(len(lo), n_pair, replace=False)]
        def obs_of(p):
            e_idx, w, _ = p
            o = tree_map(lambda x: x[w + SEQ - 1:w + SEQ].to(dev), {k: eps[e_idx]["observation"][k] for k in ("state", "privileged_state")})
            return model._obs_normalizer(o)
        def zs(ps):
            return torch.cat([get_z(e, w) for e, w, _ in ps])
        try:
            o_hi = tree_map(lambda *xs: torch.cat(xs), *[obs_of(p) for p in hi])
            o_lo = tree_map(lambda *xs: torch.cat(xs), *[obs_of(p) for p in lo])
            z_hi, z_lo = zs(hi), zs(lo)
            pos_h = disc.compute_logits(o_hi, z_hi).flatten()
            neg_h = disc.compute_logits(o_hi, z_lo).flatten()
            pos_l = disc.compute_logits(o_lo, z_lo).flatten()
            neg_l = disc.compute_logits(o_lo, z_hi).flatten()
            cond = 0.5 * (torch.nn.functional.softplus(MARGIN - pos_h + neg_h).mean()
                          + torch.nn.functional.softplus(MARGIN - pos_l + neg_l).mean())
            all_logits = torch.cat([pos_h, pos_l])
            reg = (all_logits ** 2).mean()
            loss = cond + LOGIT_REG * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        except Exception as ex:
            print("skip batch:", ex, flush=True)

    out = Path(os.environ.get("BDX_PRETRAIN_OUTPUT", os.environ.get("BFM_COND_D_OUT", "humanoidverse/data/bdx_pretrained_cond_d.pt")))
    tmp = out.with_suffix(".tmp.pt")
    torch.save({"discriminator": disc.state_dict(),
                "stats": {"steps": step, "history": hist[-5:], "final_margin": ld - ls}}, tmp)
    tmp.rename(out)  # atomic
    print(json.dumps({"final_margin": ld - ls, "saved": str(out)}))


if __name__ == "__main__":
    main()
