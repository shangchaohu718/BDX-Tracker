#!/usr/bin/env python3
# validate_onnx.py — load the exported ONNX, check graph validity, and compare its output
# against the PyTorch PPOWrapper forward on the same input (numerical parity for fp32 export).
import sys
from pathlib import Path
import numpy as np
import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx  # noqa: F401  (ensures import side-effects)
import onnx
import onnxruntime as ort


def main():
    model_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("remote_checkpoint_301M")
    onnx_path = model_folder / "exported" / "FBcprAuxModel.onnx"
    device = "cuda"

    # --- 1. graph validity ---
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    ir = m.opset_import[0].version
    # report dtypes of every value_info/initializer to confirm fp32 (no bf16 leak)
    dtypes = set()
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        if vi.type.tensor_type.elem_type:
            dtypes.add(vi.type.tensor_type.elem_type)
    for init in m.graph.initializer:
        dtypes.add(init.data_type)
    # onnx elem_type 1=float32, 16=bfloat16
    print(f"opset: {ir}  onnx IR: {m.ir_version}  dtypes(elem_type): {sorted(dtypes)} "
          f"({'fp32-only OK' if dtypes == {1} else 'NON-fp32 PRESENT'})")

    # --- 2. load PyTorch actor in fp32, amp OFF (same state as export) ---
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.eval()
    model.cfg = model.cfg.model_copy(update={"amp": False})
    model.float()

    z_dim = model.cfg.archi.z_dim
    history = "history_actor" in model.cfg.archi.actor.input_filter.key
    use_29dof = True
    state_end = 64 if use_29dof else 52
    action_end = state_end + (29 if use_29dof else 23)

    actor_obs_dim = model._actor.input_filter.output_space.shape[0] + z_dim
    rng = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(8, actor_obs_dim, generator=rng, device=device, dtype=torch.float32)

    # replicate PPOWrapper.forward slicing (helpers.py:260-276), then model.act (the real
    # entry the wrapper calls) — NOT the inner _actor network directly.
    def pytorch_forward(actor_obs):
        ao, ctx = actor_obs[:, :-z_dim], actor_obs[:, -z_dim:]
        d = {"state": ao[:, :state_end], "last_action": ao[:, state_end:action_end]}
        if history:
            d["history_actor"] = ao[:, action_end:]
        return model.act(d, ctx)

    with torch.no_grad():
        pt_out = pytorch_forward(x).cpu().numpy()

    # --- 3. ONNXRuntime inference on same input ---
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {in_name: x.cpu().numpy()})[0]

    diff = np.abs(pt_out - ort_out)
    print(f"PyTorch out shape: {pt_out.shape}  ONNX out shape: {ort_out.shape}")
    print(f"max abs diff: {diff.max():.3e}   mean abs diff: {diff.mean():.3e}")
    ok = diff.max() < 1e-3
    print("PARITY OK" if ok else "PARITY MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()