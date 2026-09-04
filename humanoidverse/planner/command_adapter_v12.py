"""v1.2 Command Adapter — P3.3 artifact (distill campaign 2026-09-04).

Delta vs planner.command_adapter (v1.0): negative forward velocity is
UNBLOCKED within the validated range [-0.65, 0) (G2 terminal-A condition:
the adapter must expose exactly the validated range). Beyond it, the
command is CLAMPED (never a silent pass) and the saturation is recorded as
a durable JSONL event — in addition to the structured warnings v1.0 already
returned. Structural errors (unknown fields/skill/target, bad duration)
still raise CommandValidationError.

Level-1 schema unchanged (frozen 2026-08-19, deliberately thin):
  {"skill": "locomotion",
   "velocity": {"forward": -0.3, "lateral": 0.0, "yaw_rate": 0.2},
   "attention": {"target": "left"},
   "duration": 2.0}

Event log: appends one JSON line per saturation to the file at
$BDX_ADAPTER_EVENT_LOG (default ./adapter_events.jsonl). Fields:
ts / adapter / field / requested / applied / reason. Rejections (structural
raises) are not logged — they never reach the planner.

Run tests: .venv/bin/python -m humanoidverse.planner.command_adapter_v12
"""
import datetime
import json
import os

NECK_SCALE = 1.7
LOCO_SCALES = (1.0, 0.5, 1.5)
ADAPTER_VERSION = "v1.2"

ATTENTION_POSES = {
    "forward": {"neck_forward": 0.50, "neck_pitch": -0.50, "neck_yaw": 0.0},
    "left":    {"neck_forward": 0.50, "neck_pitch": -0.50, "neck_yaw": 0.60},
    "right":   {"neck_forward": 0.50, "neck_pitch": -0.50, "neck_yaw": -0.60},
    # "up" REMOVED (v1.0 ruling kept): neck_forward=1.2 outside envelope
}
# ALL limits from the v1.2 validated envelope — this module invents nothing
from humanoidverse.planner.capability_v12 import ENVELOPE, clamp_command
VELOCITY_LIMITS = {"forward": (ENVELOPE["vx"].min, ENVELOPE["vx"].max),
                   "lateral": (ENVELOPE["vy"].min, ENVELOPE["vy"].max),
                   "yaw_rate": (ENVELOPE["vyaw"].min, ENVELOPE["vyaw"].max)}


class CommandValidationError(ValueError):
    pass


EVENTS = []   # in-memory transport (mirrors the durable JSONL)


def _log_event(field, requested, applied, reason):
    ev = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "adapter": ADAPTER_VERSION, "field": field,
          "requested": float(requested), "applied": float(applied),
          "reason": reason}
    EVENTS.append(ev)
    path = os.environ.get("BDX_ADAPTER_EVENT_LOG", "adapter_events.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(ev) + "\n")


def _record_warnings(warns):
    """v1.0 warned in-structure only; v1.2 also persists every saturation."""
    for w in warns:
        field = {"vx": "velocity.forward", "vy": "velocity.lateral",
                 "vyaw": "velocity.yaw_rate"}.get(
                     w["channel"], "neck." + w["channel"])
        _log_event(field, w["requested"], w["applied"], w["reason"])


def adapt(level1: dict):
    """Validate a Level-1 command and translate to the planner's 7-D vector.

    Returns {"cmd7": [...], "requested_cmd7": [...], "saturation_mask": [...],
    "warnings": [...], "n_windows": int}. cmd7 is per-frame constant.
    """
    if not isinstance(level1, dict):
        raise CommandValidationError("level-1 command must be a JSON object")
    unknown = set(level1) - {"skill", "velocity", "attention", "duration"}
    if unknown:
        raise CommandValidationError(f"unknown fields: {sorted(unknown)}")

    skill = level1.get("skill", "locomotion")
    if skill != "locomotion":
        raise CommandValidationError(
            f"skill '{skill}' not supported yet (only 'locomotion')")

    vel = level1.get("velocity", {})
    unknown_v = set(vel) - set(VELOCITY_LIMITS)
    if unknown_v:
        raise CommandValidationError(f"unknown velocity fields: {sorted(unknown_v)}")
    vx = float(vel.get("forward", 0.0))     # negatives now legal to -0.65
    vy = float(vel.get("lateral", 0.0))
    vyaw = float(vel.get("yaw_rate", 0.0))

    att = level1.get("attention", {})
    unknown_a = set(att) - {"target"}
    if unknown_a:
        raise CommandValidationError(f"unknown attention fields: {sorted(unknown_a)}")
    target = att.get("target", "forward")
    if target not in ATTENTION_POSES:
        raise CommandValidationError(
            f"attention.target '{target}' not in {sorted(ATTENTION_POSES)}")
    pose = ATTENTION_POSES[target]

    duration = level1.get("duration", 1.0)
    if not 0.5 <= duration <= 10.0:
        raise CommandValidationError("duration must be in [0.5, 10] seconds")

    import numpy as np
    cmd_phys = np.array([pose["neck_forward"], pose["neck_pitch"],
                         pose["neck_yaw"], 0.0, vx, vy, vyaw], dtype=float)
    applied_phys, mask, warns = clamp_command(cmd_phys)
    _record_warnings(warns)
    scales = np.array([1.7, 1.7, 1.7, 1.7, 1.0, 0.5, 1.5])
    return {"cmd7": (applied_phys / scales).tolist(),
            "requested_cmd7": (cmd_phys / scales).tolist(),
            "saturation_mask": mask,
            "warnings": warns,
            "n_windows": max(1, round(duration))}


def adapt_json(text: str):
    return adapt(json.loads(text))


# ---------------------------------------------------------------- tests
def _test():
    import os as _os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _os.environ["BDX_ADAPTER_EVENT_LOG"] = _os.path.join(td, "ev.jsonl")
        EVENTS.clear()

        # 1. in-range backward passes untouched, no event
        ok = adapt({"skill": "locomotion",
                    "velocity": {"forward": -0.3, "lateral": 0.0,
                                 "yaw_rate": 0.2},
                    "attention": {"target": "left"}, "duration": 2.0})
        assert ok["cmd7"][4] == -0.3 and not ok["saturation_mask"][4]
        assert ok["warnings"] == [] and EVENTS == []

        # 2. deeper than validated: clamps to -0.65 + durable event
        ok = adapt({"velocity": {"forward": -0.9}})
        assert ok["cmd7"][4] == -0.65 and ok["saturation_mask"][4]
        assert ok["requested_cmd7"][4] == -0.9
        assert EVENTS[-1]["field"] == "velocity.forward"
        assert EVENTS[-1]["reason"] == "below_validated_range"
        assert EVENTS[-1]["applied"] == -0.65
        lines = open(_os.environ["BDX_ADAPTER_EVENT_LOG"]).read().splitlines()
        assert len(lines) == 1 and json.loads(lines[0]) == EVENTS[-1]

        # 3. boundary values pass: -0.65 and +0.5 both in range
        for v in (-0.65, 0.5):
            ok = adapt({"velocity": {"forward": v}})
            assert ok["cmd7"][4] == v and not ok["saturation_mask"][4]

        # 4. multi-channel saturation: all clamped + all logged
        #    (cmd7 is normalized: vy/0.5, vyaw/1.5)
        ok = adapt({"velocity": {"forward": 9.0, "lateral": 5.0,
                                 "yaw_rate": -5.0}})
        assert ok["cmd7"][4:] == [0.5, 0.2 / 0.5, -0.5 / 1.5]
        assert ok["warnings"][0]["applied"] in (0.5, 0.2, -0.5)
        assert len([e for e in EVENTS if e["field"].startswith("velocity.")]) == 4

        # 5. structural errors still raise
        for bad in ({"style": "happy"}, {"skill": "dance"},
                    {"velocity": {"sideways": 1}},
                    {"attention": {"target": "back"}}, {"duration": 60}):
            try:
                adapt(bad)
                raise AssertionError(f"should have rejected {bad}")
            except CommandValidationError:
                pass
        print("command_adapter_v12: all tests passed")


if __name__ == "__main__":
    _test()
