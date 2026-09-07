"""v1.3 Command Adapter — CANONICAL binding (adoption 2026-09-07).

Sole delta vs command_adapter_v12 (whose level-1 schema, attention poses,
scales and event transport are inherited verbatim): velocity.forward < 0
is REJECTED, not clamped. Basis: the bounded closed-loop backward smoke
(2026-09-07, frozen SMOKE_PLAN, 4 levels x 3 deterministic resets) failed
12/12 under the official deployment runtime — mechanical disposition per
user ruling: backward = UNSUPPORTED, adapter rejects negative vx. The
rejection is also written to the durable event log (requested/applied/
reason), so no negative command can silently disappear.

Everything else behaves exactly as v1.2: positive-range clamping against
capability_v12.ENVELOPE, structural validation raising
CommandValidationError, per-saturation JSONL events via
$BDX_ADAPTER_EVENT_LOG.

There is deliberately NO path that bypasses this envelope to a bare
checkpoint — callers go through adapt()/adapt_json() or they are not
using the canonical package.

Run tests: .venv/bin/python -m humanoidverse.planner.command_adapter_v13
"""
import datetime
import json
import os

from humanoidverse.planner.command_adapter_v12 import (
    ADAPTER_VERSION as _V12_VERSION, ATTENTION_POSES, CommandValidationError,
    EVENTS,
)

ADAPTER_VERSION = "v1.3"
assert _V12_VERSION == "v1.2"          # inheritance contract

REJECTION_REASON = "backward_unsupported_closed_loop_smoke"


def _log_rejection(requested):
    # v1.2's _log_event casts applied to float; a rejection has no applied
    # value, so the event is written here under the same JSONL contract.
    ev = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "adapter": ADAPTER_VERSION, "field": "velocity.forward",
          "requested": float(requested), "applied": None,
          "reason": REJECTION_REASON}
    EVENTS.append(ev)
    path = os.environ.get("BDX_ADAPTER_EVENT_LOG", "adapter_events.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(ev) + "\n")


def adapt(level1: dict):
    """Validate a Level-1 command; translate to the planner's 7-D vector.

    Returns the same structure as v1.2: {"cmd7", "requested_cmd7",
    "saturation_mask", "warnings", "n_windows"}. Raises
    CommandValidationError when velocity.forward < 0 (UNSUPPORTED in
    closed loop — durable event written first).
    """
    if not isinstance(level1, dict):
        raise CommandValidationError("level-1 command must be a JSON object")
    vel = level1.get("velocity", {})
    vx = float(vel.get("forward", 0.0))
    if vx < 0.0:
        _log_rejection(vx)
        raise CommandValidationError(
            f"velocity.forward={vx} rejected: backward walking is "
            f"UNSUPPORTED in closed loop (smoke 2026-09-07, 12/12 FAIL; "
            f"see bdx_planner_distill/20260907T0100Z_v12smoke/DISPOSITION.json)")

    # v1.2 handles everything else (schema, attention, duration, clamping)
    import humanoidverse.planner.command_adapter_v12 as v12
    return v12.adapt(level1)


def adapt_json(text: str):
    return adapt(json.loads(text))


# ---------------------------------------------------------------- tests
def _test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.environ["BDX_ADAPTER_EVENT_LOG"] = os.path.join(td, "ev.jsonl")
        EVENTS.clear()

        # 1. zero and positive forward behave exactly as v1.2
        ok = adapt({"velocity": {"forward": 0.0}})
        assert ok["cmd7"][4] == 0.0 and not ok["saturation_mask"][4]
        ok = adapt({"velocity": {"forward": 0.3, "lateral": 0.05}})
        assert ok["cmd7"][4] == 0.3

        # 2. positive over-range still clamps + logs (v1.2 semantics)
        ok = adapt({"velocity": {"forward": 9.0}})
        assert ok["cmd7"][4] == 0.5 and ok["saturation_mask"][4]
        assert EVENTS[-1]["reason"] == "above_validated_range"

        # 3. EVERY negative value is rejected, boundary 0.0 passes
        for v in (-0.0001, -0.15, -0.3, -0.65, -0.9):
            try:
                adapt({"velocity": {"forward": v}})
                raise AssertionError(f"should have rejected {v}")
            except CommandValidationError:
                pass
        assert adapt({"velocity": {"forward": 0.0}})["cmd7"][4] == 0.0

        # 4. rejections are durable events with requested/applied/reason
        rejs = [e for e in EVENTS if e["reason"] == REJECTION_REASON]
        assert len(rejs) == 5 and rejs[0]["requested"] == -0.0001 \
            and rejs[0]["applied"] is None \
            and rejs[0]["field"] == "velocity.forward"
        lines = open(os.environ["BDX_ADAPTER_EVENT_LOG"]).read().splitlines()
        assert sum(json.loads(l)["reason"] == REJECTION_REASON
                   for l in lines) == 5

        # 5. structural errors unchanged
        for bad in ({"style": "happy"}, {"skill": "dance"},
                    {"attention": {"target": "back"}}, {"duration": 60}):
            try:
                adapt(bad)
                raise AssertionError(f"should have rejected {bad}")
            except CommandValidationError:
                pass
        print("command_adapter_v13: all tests passed")


if __name__ == "__main__":
    _test()
