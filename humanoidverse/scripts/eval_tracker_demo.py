"""Tracker Demo v0 evaluation (spec §10): per-action fixed-seed rollouts with
matched / wrong / stand z, explicit fall criterion, full episode logs.

The env itself never terminates on falls (legged_motions.yaml disables all
termination checks), so falls are detected here: base height < 0.15 m or
projected-gravity tilt > 0.7.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from tracker_demo_common import (
    ARMS, REPO, DemoEnv, aligned_reset, encode_z, load_demo_model,
    rollout_action, source_activity,
)

ACTIONS = {
    # action_id: (motion name, window offset, horizon)
    "stand":      ("v1_stand/stand_sweep_w000e00s0", 16, 250),
    "sway_soft":  ("v1_stand/stand_sweep_w000e01s0", 16, 250),
    "sway_strong":("v1_stand/stand_sweep_w000e02s1", 16, 250),
    "head_turn":  ("v0/pose_head_yaw_sweep_000", 16, 250),
    "relax":      ("v0/relax", 16, 250),
    "squat":      ("v0/pose_crouch_007", 8, 250),
    "walk":       ("v0/forward_slow", 16, 250),
    "attention":  ("v0/attention", 4, 130),
}

DEFAULT_CKPTS = {
    "A_final": str(ARMS["A"]),
    "B_final": str(ARMS["B"]),
    "B_ckpt25088": str(ARMS["B"]).replace("checkpoint", "ckpt_25088"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/tracker_demo_v0/eval_results.json")
    ap.add_argument("--episode-logs-dir", default="results/tracker_demo_v0/episode_logs")
    ap.add_argument("--wrong-z-actions", default="squat,walk,attention")
    args = ap.parse_args()

    env = DemoEnv()
    out = {}
    logs_dir = REPO / args.episode_logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    mids = {k: env.motion_id_of(v[0]) for k, v in ACTIONS.items()}

    for tag, ck in DEFAULT_CKPTS.items():
        t0 = time.time()
        model = load_demo_model(ck)
        zs = {k: encode_z(model, env, mids[k], ACTIONS[k][1]) for k in ACTIONS}
        out[tag] = {"checkpoint": ck, "actions": {}}
        for act, (name, w0, h) in ACTIONS.items():
            mid = mids[act]
            priv = env.episodes[env.episode_row(mid)]["observation"]["privileged_state"][w0 : w0 + h].to(model.cfg.device)
            src_priv = model._obs_normalizer({"privileged_state": priv})["privileged_state"]
            sa = source_activity(env, mid, w0, h)
            rows = []
            for seed in range(args.seeds):
                obs = aligned_reset(env, mid, w0, seed)
                log = rollout_action(env, model, zs[act], obs, h, src_priv=src_priv)
                rows.append(log.summary(sa))
            out[tag]["actions"][act] = {
                "source": name, "window_start": w0, "horizon": h,
                "source_activity": sa,
                "per_seed": rows,
                "aggregate": {
                    "fall_rate": float(np.mean([r["fall"] for r in rows])),
                    "completion_mean": float(np.mean([r["completion"] for r in rows])),
                    "completion_min": float(np.min([r["completion"] for r in rows])),
                    "tracking_mean": float(np.mean([r["tracking_mean"] for r in rows])),
                    "activity_mean": float(np.mean([r["activity_mean"] for r in rows])),
                    "action_rate": float(np.mean([r["action_rate"] for r in rows])),
                    "saturation": float(np.mean([r["action_saturation_frac"] for r in rows])),
                    "torque_abs_mean": float(np.mean([r["torque_abs_mean"] for r in rows])),
                },
            }
            agg = out[tag]["actions"][act]["aggregate"]
            print(f"[{tag}] {act:12s} fall={agg['fall_rate']:.2f} compl={agg['completion_mean']:.2f} "
                  f"trk={agg['tracking_mean']:.1f} act={agg['activity_mean']:.2f} sat={agg['saturation']:.3f}", flush=True)

        # command-specificity: matched vs stand-z vs walk-z on key actions
        out[tag]["wrong_z"] = {}
        for act in args.wrong_z_actions.split(","):
            if act not in ACTIONS:
                continue
            name, w0, h = ACTIONS[act]
            mid = mids[act]
            priv = env.episodes[env.episode_row(mid)]["observation"]["privileged_state"][w0 : w0 + h].to(model.cfg.device)
            src_priv = model._obs_normalizer({"privileged_state": priv})["privileged_state"]
            entry = {}
            for zname in ("matched", "stand_z", "walk_z"):
                z = zs["stand"] if zname == "stand_z" else zs["walk"] if zname == "walk_z" else zs[act]
                rows = []
                for seed in range(args.seeds):
                    obs = aligned_reset(env, mid, w0, seed)
                    log = rollout_action(env, model, z, obs, h, src_priv=src_priv)
                    rows.append(log.summary())
                entry[zname] = {
                    "tracking_mean": float(np.mean([r["tracking_mean"] for r in rows])),
                    "fall_rate": float(np.mean([r["fall"] for r in rows])),
                }
            entry["matched_vs_stand_advantage"] = entry["stand_z"]["tracking_mean"] - entry["matched"]["tracking_mean"]
            out[tag]["wrong_z"][act] = entry
            print(f"[{tag}] {act:12s} wrong-z: matched={entry['matched']['tracking_mean']:.1f} "
                  f"stand_z={entry['stand_z']['tracking_mean']:.1f} walk_z={entry['walk_z']['tracking_mean']:.1f} "
                  f"adv={entry['matched_vs_stand_advantage']:+.1f}", flush=True)
        out[tag]["elapsed_s"] = time.time() - t0
        del model
        torch.cuda.empty_cache()

    payload = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "git_commit": (REPO / "results/tracker_demo_v0/git_commit.txt").read_text().strip(),
            "seeds": args.seeds,
            "deterministic_action": True,
            "reset_mode": "aligned",
            "fall_criterion": "base_height<0.15 or ||g_xy||>0.7 (env has no fall termination)",
            "z_protocol": "hold single z (B-arm aligned training protocol), window [w0,w0+8)",
            "state_contract": "bdx-state-v2", "history_contract": "newest-first",
            "note": "in-distribution scripted demo evaluation",
        },
        "results": out,
    }
    p = REPO / args.out
    p.write_text(json.dumps(payload, indent=2))
    print(f"WROTE {p}")


if __name__ == "__main__":
    main()
