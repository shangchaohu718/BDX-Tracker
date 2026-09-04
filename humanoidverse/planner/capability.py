"""Single source of truth for the planner's VALIDATED capability envelope.

Priority (GPT ruling 2026-08-24): gate-validated range > data coverage >
adapter intention. All consumers (adapter/VLA/demo/closed-loop/eval/docs)
read from here — no component invents its own limits.
"""
from dataclasses import dataclass, asdict


@dataclass
class ChannelEnvelope:
    min: float
    max: float
    validated: bool
    note: str = ""


ENVELOPE = {
    # gate-validated (deployment gate on 414 lineage-val clips)
    "neck_forward": ChannelEnvelope(0.3, 0.7, True,
        "home range 0.4-0.6 ubiquitous in all data; wide excursions "
        "(e.g. 'up' pose 1.2) UNVALIDATED in deployment gate"),
    "neck_pitch":   ChannelEnvelope(-0.8, 0.0, False,
        "partial coverage via look_d pose only"),
    "neck_yaw":     ChannelEnvelope(-0.6, 0.6, True,
        "slope 1.03 R².96 on val"),
    "neck_roll":    ChannelEnvelope(0.0, 0.0, False,
        "intervention data carries neck_roll≡0 — unvalidated"),
    "vx":           ChannelEnvelope(0.0, 0.5, True,
        "no backward (saturate+record); realization is teacher's ~0.57-0.65"),
    "vy":           ChannelEnvelope(-0.2, 0.2, True,
        "realization ~0.65 of requested (teacher ceiling)"),
    "vyaw":         ChannelEnvelope(-0.5, 0.5, True,
        "realization ~0.54 of requested (teacher ceiling)"),
}


def clamp_command(cmd7):
    """Clamp a physical 7D command to the envelope. Returns
    (applied_cmd, saturation_mask, warnings) — structured, no globals."""
    import numpy as np
    applied = np.array(cmd7, dtype=float).copy()
    mask = [False] * 7
    warns = []
    names = list(ENVELOPE.keys())
    for i, name in enumerate(names):
        env = ENVELOPE[name]
        v = applied[i]
        if v < env.min:
            applied[i] = env.min
            mask[i] = True
            warns.append({"channel": name, "requested": float(v),
                          "applied": env.min,
                          "reason": "unsupported_backward" if (name == "vx" and not env.validated) or (name == "vx" and v < 0) else "below_validated_range"})
        elif v > env.max:
            applied[i] = env.max
            mask[i] = True
            warns.append({"channel": name, "requested": float(v),
                          "applied": env.max, "reason": "above_validated_range"})
    return applied, mask, warns


def envelope_dict():
    return {k: asdict(v) for k, v in ENVELOPE.items()}
