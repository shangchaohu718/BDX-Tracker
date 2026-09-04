"""Build the Tracker Demo v0 action library (spec §6-7).

For each demo action and each candidate checkpoint, encode one z from the
expert-sim pool window [w0, w0+seq) using the checkpoint's own normalizer +
B encoder (the B-arm aligned-training protocol; z is held for the whole
action — see train.py [BFM-REF-Z-ALIGN] "hold z until the next reset").

Includes an integrity check that pool motion ids match the train pkl's key
order (pool dof_pos == pkl dof at the pool window's frames).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

from tracker_demo_common import (
    ARMS, POOL_PATH, REPO, TRAIN_PKL, load_demo_model, z_checksum, sha256_file,
)

DEFAULT_ACTIONS = {
    # action_id -> (motion name, window offset, horizon, family)
    "stand":       ("v1_stand/stand_sweep_w000e00s0", 16, 250, "stand"),
    "sway_soft":   ("v1_stand/stand_sweep_w000e01s0", 16, 250, "stand"),
    "sway_strong": ("v1_stand/stand_sweep_w000e02s1", 16, 250, "stand"),
    "head_turn":   ("v0/pose_head_yaw_sweep_000", 16, 250, "gesture"),
    "relax":       ("v0/relax", 16, 250, "gesture"),
    "squat":       ("v0/pose_crouch_007", 8, 250, "crouch"),
    "walk":        ("v0/forward_slow", 16, 250, "locomotion"),
    "attention":   ("v0/attention", 4, 130, "gesture"),
}

DEFAULT_ARMS = {
    "A": str(ARMS["A"]),
    "B": str(ARMS["B"]),
    "B_ckpt25088": str(ARMS["B"]).replace("checkpoint", "ckpt_25088"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/tracker_demo_v0/action_library")
    args = ap.parse_args()

    names = list(joblib.load(TRAIN_PKL).keys())
    pool = torch.load(POOL_PATH, map_location="cpu", weights_only=False)
    episodes, meta = pool["episodes"], pool["meta"]
    row_by_motion = {int(m["motion_id"]): i for i, m in enumerate(meta)}
    pkl_data = joblib.load(TRAIN_PKL)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "git_commit": (REPO / "results/tracker_demo_v0/git_commit.txt").read_text().strip(),
            "z_encoding": "project_z(backward_map(pool_window[w0:w0+seq])).mean(seq) — checkpoint's own normalizer+B",
            "z_hold_protocol": "single z held for the whole action (B-arm aligned training)",
            "seq_length": 8,
            "expert_source": str(POOL_PATH),
            "expert_pool_stats": pool["stats"],
            "state_contract": "bdx-state-v2",
            "history_contract": "newest-first (z windows use state/last_action/privileged_state only)",
            "note": "in-distribution scripted demo (training-set motions)",
        },
        "actions": {},
    }

    # integrity check once: pool state[:, :14] is dof_pos relative to the
    # default pose; verify a CONSTANT offset vs the raw pkl dof over frames
    # (a wrong motion-id mapping would show a drifting, non-constant offset).
    checks = []
    for mid in (30, 215, 16, 18):  # squat, stand, forward_fast, attention ids
        if mid not in row_by_motion:
            continue
        row = row_by_motion[mid]
        t0 = float(meta[row]["t0"])
        pd = np.asarray(pkl_data[names[mid]]["dof"])
        pool_dof = episodes[row]["observation"]["state"][:, :14].numpy()
        f0 = int(round(t0 * 50.0))
        n = min(200, pool_dof.shape[0], pd.shape[0] - f0)
        diffs = np.stack([pool_dof[i] - pd[f0 + i] for i in range(n)])
        drift = float(diffs.std(0).max())
        checks.append({"motion_id": mid, "name": names[mid], "t0": t0,
                       "offset_std_max": drift,
                       "default_pose_offset": np.round(diffs.mean(0), 4).tolist()})
        assert drift < 0.05, f"motion id mapping mismatch for {names[mid]}: offset drift {drift}"
    manifest["_provenance"]["motion_id_mapping_checks"] = checks

    for tag, ck in DEFAULT_ARMS.items():
        ck_path = Path(ck)
        model = load_demo_model(ck_path)
        cfg = json.loads((ck_path / "config.json").read_text())
        step = json.loads((ck_path / "train_status.json").read_text())["time"]
        model_sha = sha256_file(ck_path / "model" / "model.safetensors")
        for act, (name, w0, h, fam) in DEFAULT_ACTIONS.items():
            mid = names.index(name)
            row = row_by_motion[mid]
            ep_obs = episodes[row]["observation"]
            win = {k: v[w0 : w0 + 8].to(model.cfg.device) for k, v in ep_obs.items()}
            with torch.no_grad():
                z = model.project_z(model.backward_map(win).view(1, 8, -1).mean(dim=1))
            act_dir = out_dir / tag / act
            act_dir.mkdir(parents=True, exist_ok=True)
            np.save(act_dir / "z.npy", z.detach().cpu().numpy())
            meta_out = {
                "action_id": act,
                "family": fam,
                "policy_arm": tag,
                "checkpoint_path": str(ck_path),
                "checkpoint_step": step,
                "checkpoint_sha256": model_sha,
                "source_motion_id": mid,
                "source_motion_name": name,
                "source_window_start": w0,
                "source_window_length": 8,
                "source_window_t0_s": float(meta[row]["t0"]) + w0 * 0.02,
                "duration_steps": h,
                "z_refresh_steps": None,
                "z_hold": True,
                "initialization_mode": "aligned",
                "return_action": "stand",
                "z_shape": list(z.shape),
                "z_checksum": z_checksum(z),
                "state_contract": "bdx-state-v2",
                "history_contract": "newest-first",
                "expert_pool_row": row,
                "created_at": datetime.now().isoformat(),
            }
            (act_dir / "metadata.json").write_text(json.dumps(meta_out, indent=2))
            manifest["actions"].setdefault(act, {})[tag] = {
                "z_checksum": meta_out["z_checksum"], "checkpoint_step": step}
        print(f"[{tag}] encoded {len(DEFAULT_ACTIONS)} actions (ckpt step {step})", flush=True)
        del model
        torch.cuda.empty_cache()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"WROTE {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
