"""Tracker Demo v0 runner (spec §3, §8): action_id -> z library -> FSM -> frozen tracker.

FSM: READY -> PREPARE -> EXECUTE -> RECOVER -> READY, ABORT on safety events.
Aligned mode resets the env to the source window's physical state through the
env's own reset path (disclosed per spec §2.3: one initial-state injection per
action, episodes between actions are environment resets).

Falls are detected by the runner (the env never terminates on falls):
sustained (>=25 steps) base height < 0.15 m or gravity tilt > 0.7.

Usage:
  python -m humanoidverse.scripts.run_tracker_demo --action squat \
      --policy B_ckpt25088 --mode aligned --seed 0 [--record-video] [--all-actions]

Policy tags follow the action-library arm tags (A, B, B_ckpt25088, or a
checkpoint dir path). z comes from the library built by
build_tracker_demo_library.py (checkpoint's own encoder/normalizer).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from tracker_demo_common import (
    ARMS, REPO, DemoEnv, aligned_reset, encode_z, load_demo_model,
    rollout_action, source_activity, fall_criterion, FALL_SUSTAIN_STEPS,
)

LIB_DIR = REPO / "results/tracker_demo_v0/action_library"


def load_action(lib_dir: Path, tag: str, action_id: str):
    meta = json.loads((lib_dir / tag / action_id / "metadata.json").read_text())
    z = torch.from_numpy(np.load(lib_dir / tag / action_id / "z.npy"))
    return meta, z


class DemoFSM:
    """READY/PREPARE/EXECUTE/RECOVER/ABORT per spec §8."""

    def __init__(self, env: DemoEnv, model, tag: str, lib_dir=LIB_DIR, record=False):
        self.env, self.model, self.tag, self.lib_dir = env, model, tag, Path(lib_dir)
        self.record = record
        self.frames = []
        self.events = []

    def _renderer(self):
        if not self.record:
            return None
        import mujoco

        if not hasattr(self, "_rend"):
            self._rend = mujoco.Renderer(self.env.inner.simulator.model, height=360, width=480)
        return self._rend

    def _snap(self, label: str):
        r = self._renderer()
        if r is None:
            return
        import mujoco

        mujoco.mj_forward(self.env.inner.simulator.model, self.env.inner.simulator.data)
        r.update_scene(self.env.inner.simulator.data)
        frame = np.asarray(r.render())[..., :3].copy()
        self.frames.append((label, frame))

    def run_action(self, action_id: str, seed: int = 0, recover_steps: int = 50):
        """One full FSM cycle for one action. Returns a report dict."""
        report = {"action_id": action_id, "seed": seed, "states": []}
        meta, z = load_action(self.lib_dir, self.tag, action_id)
        z = z.to(self.model.cfg.device)
        mid = int(meta["source_motion_id"])
        w0 = int(meta["source_window_start"])
        horizon = int(meta["duration_steps"])
        report["metadata"] = meta

        # READY: stand z if available, else the action z
        self.events.append(f"READY({action_id})")

        # PREPARE: aligned reset to the source window's physical state
        obs = aligned_reset(self.env, mid, w0, seed)
        self._snap(f"PREPARE {action_id} (aligned reset)")
        report["states"].append("PREPARE")

        # EXECUTE: policy rollout with the held z
        row = self.env.episode_row(mid)
        priv = self.env.episodes[row]["observation"]["privileged_state"][w0 : w0 + horizon].to(self.model.cfg.device)
        src_priv = self.model._obs_normalizer({"privileged_state": priv})["privileged_state"]
        sa = source_activity(self.env, mid, w0, horizon)
        log = rollout_action(self.env, self.model, z, obs, horizon, src_priv=src_priv)
        if self.record:
            for h in range(0, log.steps, 5):
                self._snap(f"EXECUTE {action_id} t={h}")
        report["execute"] = log.summary(sa)
        fell = report["execute"]["fall"]

        # ABORT checks
        if fell or report["execute"]["termination_reason"] in ("nonfinite_obs", "timeout"):
            self.events.append(f"ABORT({action_id}: {report['execute']['termination_reason']})")
            report["states"].append("ABORT")
            return report

        # RECOVER: switch back to stand z (if this wasn't stand)
        if action_id != "stand" and (self.lib_dir / self.tag / "stand").exists():
            _, z_stand = load_action(self.lib_dir, self.tag, "stand")
            z_stand = z_stand.to(self.model.cfg.device)
            obs = self.env.wrapped._get_g1env_observation(to_numpy=False)
            rlog = rollout_action(self.env, self.model, z_stand, obs, recover_steps, src_priv=None)
            self._snap(f"RECOVER stand-z t={recover_steps}")
            report["recover_fall"] = rlog.summary()["fall"]
            report["states"].append("RECOVER")
        report["states"].append("READY")
        self.events.append(f"READY(after {action_id})")
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="stand", help="action_id or 'all'")
    ap.add_argument("--mode", default="aligned", choices=["aligned"])
    ap.add_argument("--policy", default="B_ckpt25088", help="library arm tag or checkpoint dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--out-dir", default="results/tracker_demo_v0/demo_runs")
    args = ap.parse_args()

    env = DemoEnv()
    tag = args.policy if (LIB_DIR / args.policy).exists() else None
    if tag:
        ckpt = json.loads((LIB_DIR / tag / "stand" / "metadata.json").read_text())["checkpoint_path"]
    else:
        ckpt = args.policy
    model = load_demo_model(ckpt)
    fsm = DemoFSM(env, model, tag=tag or "adhoc", record=args.record_video)

    actions = list(json.loads((LIB_DIR / "manifest.json").read_text())["actions"].keys()) if args.action == "all" else [args.action]
    reports = []
    for act in actions:
        r = fsm.run_action(act, seed=args.seed)
        reports.append(r)
        s = r["execute"]
        print(f"{act:12s} steps={s['steps']} fall={s['fall']} (step {s['fall_step']}) "
              f"completion={s['completion']:.2f} reason={s['termination_reason']}", flush=True)

    out = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "policy": args.policy, "mode": args.mode, "seed": args.seed,
            "reset_disclosure": "one aligned initial-state injection per action via env reset path; "
                                "environment reset between actions (spec §2.3)",
            "fsm_events": fsm.events,
        },
        "runs": reports,
    }
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"run_{args.policy}_seed{args.seed}_{args.action}_{datetime.now().strftime('%H%M%S')}.json"
    (out_dir / name).write_text(json.dumps(out, indent=2))
    print(f"WROTE {out_dir/name}")

    if args.record_video and fsm.frames:
        try:
            import imageio.v2 as imageio

            vid = out_dir / (name.replace(".json", ".mp4"))
            labels, fr = zip(*fsm.frames)
            imageio.mimwrite(vid, list(fr), fps=10, quality=8)
            print(f"WROTE {vid} ({len(fr)} frames)")
        except Exception as e:  # video is best-effort evidence
            print(f"video write failed: {e}")


if __name__ == "__main__":
    main()
