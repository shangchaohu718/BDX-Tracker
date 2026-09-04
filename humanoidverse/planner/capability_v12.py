"""v1.2 capability envelope — P3.3 artifact (distill campaign 2026-09-04).

Delta vs planner.capability (v1.0): BACKWARD UNBLOCKED. vx.min extends to
-0.65, backed by per-depth-band validation on the lineage-clean backward
heldout (29 `_rev` clips, 8 families, never in v12 training in any variant):

  bands [<-0.05 .. -0.69]: sign-correct 941/941 windows, band-mean |dvx|
  0.010-0.029 (gate 0.05), gain 0.95-1.02 (v12 student); deepest contiguous
  valid chain = -0.70, observed coverage -0.6919 -> envelope min -0.65
  (toward-zero 0.05 grid; never exposes beyond observed coverage).

  v11 caveat (disclosed): same heldout, chain valid only to -0.60 (deepest
  band |dvx| 0.0510 > 0.05, n=7). IF the finally adopted checkpoint is v11,
  vx.min must tighten to -0.60 (P5 final-eval decides adoption).

All other channels are carried VERBATIM from v1.0 — their formal disposition
(esp. neck_forward/neck_roll) belongs to P4 INSTRUMENT_VALIDITY_MATRIX, not
to this artifact. Consumers read ranges from here only (single source of
truth preserved); evidence provenance lives in the artifact-set JSON, not in
code notes.

Run tests: .venv/bin/python -m humanoidverse.planner.capability_v12
"""
from dataclasses import dataclass, asdict


@dataclass
class ChannelEnvelope:
    min: float
    max: float
    validated: bool
    note: str = ""


ENVELOPE = {
    # carried from v1.0 (P4 disposes formally)
    "neck_forward": ChannelEnvelope(0.3, 0.7, True,
        "carried v1.0: home range 0.4-0.6 ubiquitous; wide excursions "
        "UNVALIDATED in deployment gate"),
    "neck_pitch":   ChannelEnvelope(-0.8, 0.0, False,
        "carried v1.0: partial coverage via look_d pose only"),
    "neck_yaw":     ChannelEnvelope(-0.6, 0.6, True,
        "carried v1.0: slope 1.03 R².96 on val"),
    "neck_roll":    ChannelEnvelope(0.0, 0.0, False,
        "carried v1.0: intervention data carries neck_roll≡0 — unvalidated"),
    # THE v1.2 CHANGE: backward unblocked, range == actually-validated range
    "vx":           ChannelEnvelope(-0.65, 0.5, True,
        "v1.2: backward validated on lineage-clean heldout (941/941 "
        "sign-correct, band |dvx|<=0.029, gain 0.95-1.02); if v11 adopted "
        "at P5, tighten min to -0.60"),
    "vy":           ChannelEnvelope(-0.2, 0.2, True,
        "carried v1.0: realization ~0.65 of requested (teacher ceiling)"),
    "vyaw":         ChannelEnvelope(-0.5, 0.5, True,
        "carried v1.0: realization ~0.54 of requested (teacher ceiling)"),
}


def clamp_command(cmd7):
    """Clamp a physical 7D command to the v1.2 envelope. Returns
    (applied_cmd, saturation_mask, warnings) — structured, no globals.
    Backward is NO LONGER special-cased: below -0.65 saturates to -0.65 with
    reason below_validated_range (same taxonomy as every other channel)."""
    import numpy as np
    applied = np.array(cmd7, dtype=float).copy()
    mask = [False] * 7
    warns = []
    for i, (name, env) in enumerate(ENVELOPE.items()):
        v = applied[i]
        if v < env.min:
            applied[i] = env.min
            mask[i] = True
            warns.append({"channel": name, "requested": float(v),
                          "applied": env.min,
                          "reason": "below_validated_range"})
        elif v > env.max:
            applied[i] = env.max
            mask[i] = True
            warns.append({"channel": name, "requested": float(v),
                          "applied": env.max,
                          "reason": "above_validated_range"})
    return applied, mask, warns


def envelope_dict():
    return {k: asdict(v) for k, v in ENVELOPE.items()}


def _test():
    import numpy as np
    a, m, w = clamp_command([0.5, -0.5, 0.0, 0.0, -0.5, 0.0, 0.0])
    assert a[4] == -0.5 and not m[4], "in-range backward must pass untouched"
    a, m, w = clamp_command([0.5, -0.5, 0.0, 0.0, -0.9, 0.0, 0.0])
    assert a[4] == -0.65 and m[4] and w[0]["reason"] == "below_validated_range"
    a, m, w = clamp_command([0.5, -0.5, 0.0, 0.0, 0.8, 3.0, 0.0])
    assert a[4] == 0.5 and a[5] == 0.2 and len(w) == 2
    d = envelope_dict()
    assert d["vx"]["min"] == -0.65 and d["vx"]["validated"] is True
    print("capability_v12: all tests passed")


if __name__ == "__main__":
    _test()
