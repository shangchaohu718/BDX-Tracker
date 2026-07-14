#!/usr/bin/env python3
# export_onnx.py — load a checkpoint and dump the actor (policy) to ONNX for deployment.
#
# Uses the codebase's own export_meta_policy_as_onnx (helpers.py:240), which wraps the actor
# in a PPOWrapper that splits [state | last_action | history...] | z and calls actor.act.
# Output: <model_folder>/exported/<ModelName>.onnx
#
# Usage:
#   uv run python -m humanoidverse.scripts.export_onnx --model_folder remote_checkpoint_301M
import argparse
from pathlib import Path
import torch
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_folder", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None, help="override output filename (default <ModelName>.onnx)")
    args = ap.parse_args()

    model = load_model_from_checkpoint_dir(args.model_folder / "checkpoint", device=args.device)
    model.eval()

    # Export in fp32 (not the training bf16). The model's act()/forward()/project_z() wrap
    # every op in autocast(dtype=bfloat16, enabled=self.cfg.amp) — see fb/model.py:106,137.
    # torch.onnx traces those ops at bf16, and stock ONNXRuntime rejects Mish on bf16.
    # The cfg is a FROZEN Pydantic model, so we replace model.cfg with a new frozen copy that
    # has amp=False (model_copy carries through the deepcopy done inside export_meta_policy_...).
    model.cfg = model.cfg.model_copy(update={"amp": False})
    model.float()

    z_dim = model.cfg.archi.z_dim
    actor_obs_dim = model._actor.input_filter.output_space.shape[0] + z_dim
    history = "history_actor" in model.cfg.archi.actor.input_filter.key
    out_name = args.name or f"{model.__class__.__name__}.onnx"

    print(f"model: {model.__class__.__name__}")
    print(f"actor_obs_dim (incl z): {actor_obs_dim}  z_dim: {z_dim}  history: {history}")

    export_meta_policy_as_onnx(
        model,
        path=str(args.model_folder / "exported"),
        exported_policy_name=out_name,
        example_obs_dict={"actor_obs": torch.randn(1, actor_obs_dim)},
        z_dim=z_dim,
        history=history,
        use_29dof=True,
    )
    out = args.model_folder / "exported" / out_name
    print(f"ONNX written: {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()