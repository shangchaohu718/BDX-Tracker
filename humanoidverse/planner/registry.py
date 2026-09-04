"""Single source of truth for the canonical planner checkpoint.

All entry scripts should resolve defaults through resolve_canonical()
instead of hardcoding checkpoint filenames — prevents silently loading
a legacy generation (audit 2026-08-24 finding).
"""
import json
from pathlib import Path

_DEFAULT_DIR = Path(__file__).parents[1] / "data" / "bdx_planner_v2combo"


def resolve_canonical(data_dir=None):
    """Returns dict with checkpoint/teacher paths; raises if missing.
    Also activates the checkpoint's TRAINED preprocess version so runtime
    input semantics match training (audit 2026-08-24 parity requirement)."""
    d = Path(data_dir) if data_dir else _DEFAULT_DIR
    reg = json.loads((d / "canonical.json").read_text())
    ckpt, teacher = d / reg["checkpoint"], d / reg["teacher"]
    if not ckpt.exists():
        raise FileNotFoundError(f"canonical checkpoint missing: {ckpt}")
    from humanoidverse.planner.dataset import set_preprocess_version
    set_preprocess_version(reg.get("preprocess_version", "fixed_v1"))
    return {"planner": ckpt, "teacher": teacher, "registry": reg}
