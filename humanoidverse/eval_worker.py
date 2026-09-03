# [EVAL-SUBPROC] Tracking-evaluation subprocess entry.
#
# Runs the full HumanoidVerseIsaac tracking evaluation in an isolated process so the
# transient host-RAM staging of the motion lib (chunked loading, see
# evaluations/humanoidverse_isaac.py) is guaranteed to be returned to the OS when this
# process exits — the trainer's RSS stays flat across eval ticks (64 GiB cgroup budget).
#
# Usage (spawned by Workspace.eval via _run_tracking_eval_subprocess):
#   python -m humanoidverse.eval_worker \
#       --checkpoint <dir with config.json/init_kwargs.json/optimizers.pth/model/> \
#       --env-config <json: HumanoidVerseIsaacConfig.model_dump()> \
#       --eval-config <json: HumanoidVerseIsaacTrackingEvaluationConfig.model_dump()> \
#       --timestep <int> --results-json <out path> [--device cuda:0]
#
# Writes {"evaluation_metrics": ..., "wandb_dict": ...} to --results-json and exits 0.
# On any failure writes {"error": ...} and exits 1 — the parent skips this eval tick.

import argparse
import json
import sys
import traceback


class _NullLogger:
    """The parent owns the CSV/wandb logging rights; the worker only returns data."""

    def log(self, *args, **kwargs):
        pass


def _to_jsonable(x):
    import numbers

    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, numbers.Number):
        return x
    if isinstance(x, (str, bool)):
        return x
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    results_path = args.results_json
    try:
        from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
        from humanoidverse.agents.evaluations.humanoidverse_isaac import (
            HumanoidVerseIsaacTrackingEvaluation,
            HumanoidVerseIsaacTrackingEvaluationConfig,
        )
        from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent

        with open(args.env_config) as f:
            env_cfg_json = json.load(f)
        with open(args.eval_config) as f:
            eval_cfg_json = json.load(f)

        num_envs = int(eval_cfg_json.get("num_envs") or 1024)

        agent = FBcprAuxAgent.load(args.checkpoint, device=args.device)

        env, _ = HumanoidVerseIsaacConfig(**env_cfg_json).build(num_envs=num_envs)
        evaluation = HumanoidVerseIsaacTrackingEvaluationConfig(**eval_cfg_json).build()
        evaluation_metrics, wandb_dict = evaluation.run(
            timestep=args.timestep,
            agent_or_model=agent,
            logger=_NullLogger(),
            env=env,
        )

        payload = {
            "evaluation_metrics": _to_jsonable(evaluation_metrics),
            "wandb_dict": _to_jsonable(wandb_dict),
        }
        with open(results_path, "w") as f:
            json.dump(payload, f)
        env.close()
        return 0
    except Exception:
        with open(results_path, "w") as f:
            json.dump({"error": traceback.format_exc()}, f)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
