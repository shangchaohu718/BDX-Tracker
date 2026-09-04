"""P0.55-adjacent: exact equivalence of expert-z (encode_expert) and tracking-z
(_sample_tracking_z) on the same 250-frame sequence, same next_obs windows.

Per code both average B over an 8-frame forward window then project_z; the
250 in tracking is the schedule length, not the averaging window. Expected:
near-exact equality for t<=242, truncation drift only for t=243..249.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

SEQ = 8
SCHED = 250


def main():
    pool = torch.load("humanoidverse/data/bdx_expert_sim_pool_cpu.pt", map_location="cpu")
    eps = pool["episodes"]
    model = load_model_from_checkpoint_dir(Path("results/bfm-dynamics-pool-rnorm-3m/checkpoint"), device="cuda")
    model.eval()
    model._obs_normalizer.eval()
    dev = torch.device("cuda")

    results = []
    for ep_idx in (0, 7, 42):
        obs = eps[ep_idx]["observation"]  # keys state/last_action/privileged_state, [264, ...]
        with torch.no_grad():
            # full-sequence B encodings (what _sample_tracking_z computes on)
            seq_next = {k: v[1:SCHED + 1] for k, v in obs.items()}  # next_obs rows for frames 0..249
            B_all = model._backward_map(model._obs_normalizer(tree_map(lambda x: x.to(dev), seq_next)))  # [250, dz]
            # tracking path: per-step mean over [t, min(t+8,250)) then project
            z_track = torch.empty(SCHED, B_all.shape[-1], device=dev)
            for t in range(SCHED):
                end = min(t + SEQ, SCHED)
                z_track[t] = B_all[t:end].mean(dim=0)
            z_track = model.project_z(z_track)
            # encode_expert path per t: 8-frame window starting at row t
            z8 = torch.empty(SCHED, B_all.shape[-1], device=dev)
            for t in range(SCHED):
                end = min(t + SEQ, SCHED)
                win = {k: v[t:end] for k, v in obs.items()}  # obs rows t..t+7; next rows t+1..t+8
                win_next = {k: v[t + 1:end + 1] for k, v in obs.items()}
                b = model._backward_map(model._obs_normalizer(tree_map(lambda x: x.to(dev), win_next)))
                z8[t] = model.project_z(b.mean(dim=0))
        cos = torch.nn.functional.cosine_similarity(z8, z_track, dim=-1)
        diff = (z8 - z_track).abs().max(dim=-1).values
        results.append({
            "episode": ep_idx,
            "norm_z8_mean": float(z8.norm(dim=-1).mean()),
            "norm_ztrack_mean": float(z_track.norm(dim=-1).mean()),
            "cos_t0_242_mean": float(cos[:SCHED - 7].mean()),
            "cos_t0_242_min": float(cos[:SCHED - 7].min()),
            "maxdiff_t0_242": float(diff[:SCHED - 7].max()),
            "allclose_t0_242": bool(torch.allclose(z8[:SCHED - 7], z_track[:SCHED - 7], atol=1e-5)),
            "cos_t243_249_mean": float(cos[SCHED - 7:].mean()),
            "maxdiff_t243_249": float(diff[SCHED - 7:].max()),
        })
    out = Path("results/bfmzero-bdx-full/z8_tracking_equivalence.json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
