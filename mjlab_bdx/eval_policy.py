"""Parameterized headless policy eval (v2) for mjlab tracking tasks.

Supersedes eval_play_headless.py (archived; its `falls` counted terminations,
which the gesture terminations do not reliably produce for ground-hugging).

Contract per execution order 2026-08-26:
  - checkpoint / task / motion-file are CLI args, never hardcoded;
  - JSON records checkpoint SHA256 + task + motion file + sampling + seeds;
  - `fall` is NOT terminated-or-truncated. Independent criteria:
      fell_by_height   : min root z < --fall-z at any pre-step sample
      lying_on_ground  : >= --lying-steps consecutive pre-step samples z < --fall-z
      fall             : fell_by_height OR lying_on_ground
    terminated / truncated / anchor_ori_violation are recorded separately.
  - sampling: "start" = fixed phase 0, "uniform" = random RSI starts.
  - deterministic policy: distribution mean (obs normalizer zero-var guard).

Run from bdx_rl_mjlab repo root with ITS venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(p: str) -> str:
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def load_actor(path: str, device: str):
  ck = torch.load(path, map_location="cpu", weights_only=False)
  sd = ck["actor_state_dict"]
  mean = sd["obs_normalizer._mean"].float().to(device)
  std = sd["obs_normalizer._std"].float().to(device)
  std = torch.where(std > 1e-8, std, torch.ones_like(std))  # zero-var guard
  layers = [(sd[f"mlp.{i}.weight"].float().to(device),
             sd[f"mlp.{i}.bias"].float().to(device)) for i in (0, 2, 4, 6)]

  @torch.no_grad()
  def act(obs: torch.Tensor) -> torch.Tensor:
    x = (obs - mean) / std
    for w, b in layers[:-1]:
      x = torch.nn.functional.elu(x @ w.T + b)
    w, b = layers[-1]
    return x @ w.T + b

  return act, int(ck["iter"])


def run_episodes(env, motion, robot, act, n_episodes, max_steps, fall_z,
                 lying_min_steps):
  episodes = []
  obs, _ = env.reset()
  for ep in range(n_episodes):
    start_frame = int(motion.time_steps[0].item())
    zs, terms_fired, terminated, truncated = [], [], False, False
    done = False
    steps = 0
    while steps < max_steps and not done:
      a = act(obs["actor"])
      z_pre = float(robot.data.root_link_pose_w[0, 2])
      zs.append(z_pre)
      obs, _, term, trunc, _ = env.step(a)
      steps += 1
      terminated, truncated = bool(term[0]), bool(trunc[0])
      if terminated or truncated:
        tm = env.termination_manager
        terms_fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
        done = True
    zs_t = torch.tensor(zs)
    below = zs_t < fall_z
    # longest run of consecutive below-threshold samples
    run = best = 0
    for b in below.tolist():
      run = run + 1 if b else 0
      best = max(best, run)
    episodes.append({
      "start_frame": start_frame,
      "steps": steps,
      "completed": steps >= max_steps and not (terminated or truncated),
      "terminated": terminated,
      "truncated": truncated,
      "terms_at_done": terms_fired,
      "anchor_ori_violation": "anchor_ori" in terms_fired,
      "min_root_z": round(min(zs), 4),
      "final_root_z": round(zs[-1], 4),
      "fell_by_height": bool(below.any()),
      "lying_steps_max": int(best),
      "lying_on_ground": bool(best >= lying_min_steps),
      "fall": bool(below.any() or best >= lying_min_steps),
    })
  return episodes


def summarize(episodes):
  n = len(episodes)
  fall_h = sum(e["fell_by_height"] for e in episodes)
  lying = sum(e["lying_on_ground"] for e in episodes)
  term = sum(e["terminated"] for e in episodes)
  aori = sum(e["anchor_ori_violation"] for e in episodes)
  fall = sum(e["fall"] for e in episodes)
  return {
    "episodes": n,
    "fall_rate": round(fall / n, 4),
    "fell_by_height_rate": round(fall_h / n, 4),
    "lying_on_ground_rate": round(lying / n, 4),
    "terminated_rate": round(term / n, 4),
    "anchor_ori_violation_rate": round(aori / n, 4),
    "completed_rate": round(sum(e["completed"] for e in episodes) / n, 4),
    "mean_steps": round(sum(e["steps"] for e in episodes) / n, 1),
    "min_root_z_overall": round(min(e["min_root_z"] for e in episodes), 4),
    "min_root_z_p10": round(sorted(e["min_root_z"] for e in episodes)[n // 10], 4),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--motion-file", required=True)
  ap.add_argument("--sampling", choices=["start", "uniform"], default="start")
  ap.add_argument("--seeds", default="0,1,2,3")
  ap.add_argument("--episodes-per-seed", type=int, default=8)
  ap.add_argument("--max-steps", type=int, default=260)
  ap.add_argument("--fall-z", type=float, default=0.12)
  ap.add_argument("--lying-min-steps", type=int, default=25)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  device = "cpu"
  import tracking_bdx.config.bdx_v4  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  act, it = load_actor(args.checkpoint, device)
  all_eps = []
  for seed in [int(s) for s in args.seeds.split(",")]:
    torch.manual_seed(seed)
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.commands["motion"].sampling_mode = args.sampling
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    motion = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    eps = run_episodes(env, motion, robot, act, args.episodes_per_seed,
                       args.max_steps, args.fall_z, args.lying_min_steps)
    for e in eps:
      e["seed"] = seed
    all_eps.extend(eps)
    env.close()

  report = {
    "task": args.task,
    "motion_file": str(Path(args.motion_file).resolve()),
    "checkpoint": str(Path(args.checkpoint).resolve()),
    "checkpoint_sha256": sha256(args.checkpoint),
    "checkpoint_iter": it,
    "sampling": args.sampling,
    "seeds": args.seeds,
    "thresholds": {"fall_z": args.fall_z,
                   "lying_min_steps": args.lying_min_steps,
                   "max_steps": args.max_steps},
    "summary": summarize(all_eps),
    "episodes": all_eps,
  }
  out = Path(args.out)
  if out.exists():
    raise SystemExit(f"refusing overwrite: {out}")
  out.write_text(json.dumps(report, indent=2))
  print(json.dumps(report["summary"], indent=2))
  print("saved", out)


if __name__ == "__main__":
  main()
