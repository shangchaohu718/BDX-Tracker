"""VLA Level-1 → planner Level-2 Command Adapter (rule-based first version).

Level-1 schema (frozen 2026-08-19 per external review — deliberately THIN):
  {"skill": "locomotion",
   "velocity": {"forward": 0.3, "lateral": 0.0, "yaw_rate": 0.2},
   "attention": {"target": "left"},       # forward|left|right|up
   "duration": 2.0}                        # seconds

Translates to the 7-D Level-2 command [neck_forward, neck_pitch, neck_yaw,
neck_roll, vx, vy, vyaw] (normalized by the planner's fixed scales:
neck /1.7 rad, vx /1.0, vy /0.5, vyaw /1.5). Rules now, learned later — the
interface is what matters: small surface, easy to tell VLA errors from
planner errors.

Run tests: .venv/bin/python -m humanoidverse.planner.command_adapter
"""

import json

NECK_SCALE = 1.7
LOCO_SCALES = (1.0, 0.5, 1.5)

# attention targets CONSTRAINED to the validated envelope; "up" disabled
# (neck_forward unvalidated in deployment data) — audit final round
ATTENTION_POSES = {
    "forward": {"neck_forward": 0.50, "neck_pitch": -0.50, "neck_yaw": 0.0},
    "left":    {"neck_forward": 0.50, "neck_pitch": -0.50, "neck_yaw": 0.60},
    "right":   {"neck_forward": 0.50, "neck_pitch": -0.50, "neck_yaw": -0.60},
    # "up" REMOVED: neck_forward=1.2 outside validated envelope
}
# ALL limits come from planner.capability (the validated envelope) — this
# module must not invent its own ranges (audit 2026-08-24 final round)
from humanoidverse.planner.capability import ENVELOPE, clamp_command
VELOCITY_LIMITS = {"forward": (ENVELOPE["vx"].min, ENVELOPE["vx"].max),
                   "lateral": (ENVELOPE["vy"].min, ENVELOPE["vy"].max),
                   "yaw_rate": (ENVELOPE["vyaw"].min, ENVELOPE["vyaw"].max)}


class CommandValidationError(ValueError):
    pass


CLIP_LOG = []   # transport for saturation events (adapter-level)


def _clamp(name, v, lo, hi):
    if not lo <= v <= hi:
        if name == "velocity.forward" and v < lo:
            # unsupported backward: safe-saturate + record (GPT ruling 2026-08-24)
            CLIP_LOG.append({"field": name, "requested": v, "applied": lo,
                             "reason": "unsupported_backward"})
            return lo
        raise CommandValidationError(
            f"{name}={v} outside feasible range [{lo}, {hi}]")
    return v


def adapt(level1: dict):
    """Validate a Level-1 command and translate to the planner's 7-D vector.

    Returns {"cmd7": [7 floats], "n_windows": int} — cmd7 is per-frame
    constant; the planner caller broadcasts it over each 1s window.
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
    # envelope handles saturation; only structural errors raise here
    vx = float(vel.get("forward", 0.0))
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

    cmd7 = [pose["neck_forward"] / NECK_SCALE,
            pose["neck_pitch"] / NECK_SCALE,
            pose["neck_yaw"] / NECK_SCALE,
            0.0,
            vx / LOCO_SCALES[0],
            vy / LOCO_SCALES[1],
            vyaw / LOCO_SCALES[2]]
    # structured saturation — the ONLY return path for applied-command info
    import numpy as _np
    applied, mask, warns = clamp_command(_np.array(cmd7) * _np.array(
        [ENVELOPE[n].max if ENVELOPE[n].max != 0 else 1.0 for n in
         ("neck_forward","neck_pitch","neck_yaw","neck_roll","vx","vy","vyaw")])
        * _np.array([1.7,1.7,1.7,1.7,1.0,0.5,1.5]))  # de-normalize to physical
    # simpler: clamp on the physical command before normalize — redo cleanly
    cmd_phys = _np.array([pose["neck_forward"], pose["neck_pitch"],
                          pose["neck_yaw"], 0.0, vx, vy, vyaw], dtype=float)
    applied_phys, mask, warns = clamp_command(cmd_phys)
    scales = _np.array([1.7, 1.7, 1.7, 1.7, 1.0, 0.5, 1.5])
    return {"cmd7": (applied_phys / scales).tolist(),
            "requested_cmd7": (cmd_phys / scales).tolist(),
            "saturation_mask": mask,
            "warnings": warns,
            "n_windows": max(1, round(duration))}


def adapt_json(text: str):
    return adapt(json.loads(text))


# ---------------------------------------------------------------- tests
def _test():
    ok = adapt({"skill": "locomotion",
                "velocity": {"forward": 0.3, "lateral": 0.0, "yaw_rate": 0.2},
                "attention": {"target": "left"}, "duration": 2.0})
    assert ok["cmd7"][2] == 0.70 / 1.7 and ok["cmd7"][4] == 0.3
    assert ok["n_windows"] == 2

    for bad, why in [
        ({"style": "happy"}, "unknown field"),
        ({"skill": "dance"}, "unsupported skill"),
        ({"velocity": {"forward": 5.0}}, "out of range"),
        ({"attention": {"target": "back"}}, "unknown target"),
        ({"duration": 60}, "duration out of range"),
    ]:
        try:
            adapt(bad)
            raise AssertionError(f"should have rejected {bad} ({why})")
        except CommandValidationError:
            pass
    print("command_adapter: all tests passed")


if __name__ == "__main__":
    _test()
