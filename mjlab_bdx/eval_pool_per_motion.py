"""Per-motion eval for pool-trained checkpoints (execution order 1-7).

For each motion in the pool: pin ALL envs to that motion (set_motion_index),
run deterministic-policy episodes, judge falls by HEIGHT criteria (never by
termination — gesture terminations are known-loose vs ground-hugging), and
report per-motion metrics + macro / worst-motion / P10 aggregates.

Wilson intervals on per-motion fall and completion rates.

Run from bdx_rl_mjlab repo root with ITS venv.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def wilson(p_hat: float, n: int, z: float = 1.96) -> list[float]:
  if n == 0:
    return [0.0, 1.0]
  den = 1 + z * z / n
  c = (p_hat + z * z / (2 * n)) / den
  hw = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / den
  return [round(max(0.0, c - hw), 3), round(min(1.0, c + hw), 3)]


def load_actor(path: str, device: str = "cpu"):
  ck = torch.load(path, map_location="cpu", weights_only=False)
  sd = ck["actor_state_dict"]
  mean = sd["obs_normalizer._mean"].float().to(device)
  std = torch.where(sd["obs_normalizer._std"].float().to(device) > 1e-8,
                    sd["obs_normalizer._std"].float().to(device),
                    torch.ones_like(sd["obs_normalizer._std"].float()))
  layers = [(sd[f"mlp.{i}.weight"].float().to(device),
             sd[f"mlp.{i}.bias"].float().to(device)) for i in (0, 2, 4, 6)]

  @torch.no_grad()
  def act(obs):
    x = (obs - mean) / std
    for w, b in layers[:-1]:
      x = torch.nn.functional.elu(x @ w.T + b)
    w, b = layers[-1]
    return x @ w.T + b

  return act, int(ck["iter"])


def episodes_for_motion(env, mc, robot, act, motion_i, n_eps, max_steps,
                        fall_z, lying_min, sampling="start"):
  """Per-motion episodes with a HARD pin (fixed set_motion_index).

  Each episode: env.reset() -> set_motion_index(i, frame) -> refresh pipeline
  (write/forward/sense/command-compute, mirroring env.reset internals) ->
  recompute obs. steps capped at clip length - 2 so the end-of-clip
  _resample_command (random reassignment) can never fire mid-episode.
  """
  T = int(mc.motion_pool[motion_i].time_step_total)
  steps_cap = max(10, min(max_steps, T - 2))
  eps = []
  for _ in range(n_eps):
    env.reset()
    frame = 0 if sampling == "start" else int(torch.randint(0, T - 2, (1,)))
    mc.set_motion_index(motion_i, frame=frame)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    env.command_manager.compute(dt=0.0)
    obs = env.observation_manager.compute(update_history=True)
    zs, terms, done, steps = [], [], False, 0
    while steps < steps_cap and not done:
      a = act(obs["actor"])
      zs.append(float(robot.data.root_link_pose_w[0, 2]))
      obs, _, term, trunc, _ = env.step(a)
      steps += 1
      if bool(term[0]) or bool(trunc[0]):
        tm = env.termination_manager
        terms = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
        done = True
    below = [z < fall_z for z in zs]
    run = best = 0
    for b in below:
      run = run + 1 if b else 0
      best = max(best, run)
    eps.append({
      "steps": steps, "steps_cap": steps_cap,
      "completed": steps >= steps_cap and not done,
      "terminated": done, "terms": terms,
      "min_z": round(min(zs), 4),
      "fell_by_height": any(below), "lying": best >= lying_min,
      "fall": any(below) or best >= lying_min,
      "joint_err": round(float(mc.metrics["error_joint_pos"][0]), 3),
      "anchor_pos_err": round(float(mc.metrics["error_anchor_pos"][0]), 3),
      "anchor_ori_err": round(float(mc.metrics["error_anchor_rot"][0]), 3),
    })
  return eps


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Mjlab-Tracking-Flat-BDX-V4-Wiggle-Pool-8")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--episodes-per-motion", type=int, default=12)
  ap.add_argument("--sampling", choices=["start", "uniform"], default="start")
  ap.add_argument("--max-steps", type=int, default=300)
  ap.add_argument("--fall-z", type=float, default=0.12)
  ap.add_argument("--lying-min-steps", type=int, default=25)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import tracking_bdx.config.bdx_v4  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  torch.manual_seed(0)
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  cfg.commands["motion"].sampling_mode = args.sampling
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  mc = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  act, it = load_actor(args.checkpoint)

  per_motion = []
  for i in range(len(mc.motion_pool)):
    f = Path(str(mc.cfg.motion_files[i]))
    eps = episodes_for_motion(env, mc, robot, act, i,
                              args.episodes_per_motion, args.max_steps,
                              args.fall_z, args.lying_min_steps,
                              sampling=args.sampling)
    n = len(eps)
    fr = sum(e["fall"] for e in eps) / n
    cr = sum(e["completed"] for e in eps) / n
    per_motion.append({
      "motion": f.name, "index": i,
      "episodes": n,
      "fall_rate": round(fr, 3), "fall_rate_wilson": wilson(fr, n),
      "completion_rate": round(cr, 3),
      "completion_rate_wilson": wilson(cr, n),
      "min_z_worst": round(min(e["min_z"] for e in eps), 3),
      "joint_err_last": eps[-1]["joint_err"],
      "anchor_pos_err_last": eps[-1]["anchor_pos_err"],
      "anchor_ori_err_last": eps[-1]["anchor_ori_err"],
      "episodes_detail": eps,
    })
    print(f"[{f.name}] fall={fr:.2f} compl={cr:.2f} minz={per_motion[-1]['min_z_worst']}",
          flush=True)

  def agg(key):
    vals = [m[key] for m in per_motion]
    s = sorted(vals)
    return {"macro_avg": round(sum(vals) / len(vals), 3),
            "worst_motion": round(max(vals), 3),
            "p10_motion": round(s[max(0, len(s) // 10 - 1)], 3)}

  report = {
    "task": args.task, "checkpoint": str(Path(args.checkpoint).resolve()),
    "checkpoint_iter": it,
    "thresholds": {"fall_z": args.fall_z,
                   "lying_min_steps": args.lying_min_steps,
                   "max_steps": args.max_steps,
                   "sampling": args.sampling},
    "aggregates": {k: agg(k) for k in
                   ["fall_rate", "completion_rate", "min_z_worst",
                    "joint_err_last", "anchor_ori_err_last"]},
    "per_motion": per_motion,
  }
  out = Path(args.out)
  assert not out.exists(), f"refusing overwrite {out}"
  out.write_text(json.dumps(report, indent=2))
  print(json.dumps(report["aggregates"], indent=2))
  print("saved", out)


if __name__ == "__main__":
  main()
