"""P3.3 builder: v1.2 artifact set (capability envelope + adapter self-test).

Deterministically assembles, into ART_DIR:
  capability_envelope_v12.json   full 7-channel envelope + backward capability
                                 + evidence provenance (sha256 of every cited
                                 artifact) + v11-adoption caveat
  adapter_selftest.json          scenario battery through the REAL
                                 command_adapter_v12.adapt() incl. rejections
  adapter_events_selftest.jsonl  the durable event log written during battery

Run: ART_DIR=<artifact dir> \
     .venv/bin/python humanoidverse/scripts/build_v12_artifacts.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner import capability_v12, command_adapter_v12

REPO = Path("/home/tcl/Desktop/start/BFM-zero")
ART = Path(os.environ["ART_DIR"])
V12DATA = REPO / "humanoidverse/data/bdx_planner_distill/20260904T1900Z_p321_v12data"
ACCEPT = REPO / "humanoidverse/data/bdx_planner_distill/20260904T2100Z_p323_accept"
CANON = REPO / "humanoidverse/data/bdx_planner_v2combo/canonical.json"


def sha256(p, n=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def main():
    # ---- 1. capability envelope v1.2 --------------------------------------
    bands = json.loads((ART / "backward_depth_bands.json").read_text())
    hold12 = json.loads((ACCEPT / "v12_holdout.json").read_text())
    hold11 = json.loads((ACCEPT / "v11_holdout_same.json").read_text())
    env = {
        "artifact": "capability_envelope_v12",
        "version": "1.2",
        "generated": "2026-09-04T22:00Z (P3.3, distill campaign)",
        "backward_capability": "SUPPORTED_WITH_RANGE",
        "backward_validated_range": [-0.65, 0.0],
        "envelope": capability_v12.envelope_dict(),
        "delta_vs_v1.0": {
            "vx": {"v1.0": [0.0, 0.5],
                   "v1.2": [-0.65, 0.5],
                   "reason": "G2 terminal-A: lineage-clean heldout validation "
                             "(v12 student), per-depth-band gate"},
            "other_channels": "carried verbatim; formal disposition deferred "
                              "to P4 INSTRUMENT_VALIDITY_MATRIX",
        },
        "evidence": {
            "heldout_v12": hold12, "heldout_v11_same": hold11,
            "depth_bands": {t: {"bands": v["bands"],
                                "deepest_contiguous_valid":
                                    v["deepest_contiguous_valid"],
                                "observed_cmd_min": v["observed_cmd_min"],
                                "envelope_min_grid_toward_zero":
                                    v["envelope_min_grid_toward_zero"]}
                            for t, v in bands.items()
                            if t not in ("band_rule", "grid_edges")},
            "band_rule": bands["band_rule"],
            "sweep_instrument": "P2_1_SWEEP_EVIDENCE.md: grid -0.2..+0.8, "
                                "slope 0.869-0.90, R²>=0.998, sign 6/6",
        },
        "caveats": [
            "IF the finally adopted checkpoint is v11 (P5 final eval decides), "
            "vx.min must tighten to -0.60 (v11 deepest band |dvx| 0.0510>0.05).",
            "Deep-band q is elevated for both students (~80 mrad at "
            "[−0.70,−0.60), n=7) — reconstruction quality at extreme depth is "
            "disclosed, tracking gate unaffected.",
            "Deployment-form (closed-loop) backward acceptance not yet run; "
            "this envelope is completion-form validated.",
        ],
        "provenance": {
            "supersedes": {
                "file": str(CANON),
                "field": "capability_envelope.vx",
                "sha256_16": sha256(CANON, 16),
                "old_value": [0.0, 0.5],
                "note": "canonical dir READ-ONLY, untouched; this overlay is "
                        "the authority for vx until P5 binds a checkpoint"},
            "evidence_sha256": {
                "v12_holdout.json": sha256(ACCEPT / "v12_holdout.json"),
                "v11_holdout_same.json": sha256(ACCEPT / "v11_holdout_same.json"),
                "backward_depth_bands.json": sha256(ART / "backward_depth_bands.json"),
                "cmd_coverage_holdout.json": sha256(ART / "cmd_coverage_holdout.json"),
                "p2_student_v12.pt": sha256(V12DATA / "p2_student_v12.pt"),
                "p2_student_v11.pt": sha256(
                    REPO / "humanoidverse/data/bdx_planner_v2combo/p2_student_v11.pt"),
            },
            "generators": ["humanoidverse/planner/capability_v12.py",
                           "humanoidverse/planner/command_adapter_v12.py",
                           "humanoidverse/scripts/eval_backward_depth_bands.py",
                           "humanoidverse/scripts/build_v12_artifacts.py"],
        },
    }
    (ART / "capability_envelope_v12.json").write_text(json.dumps(env, indent=1))

    # ---- 2. adapter scenario battery --------------------------------------
    os.environ["BDX_ADAPTER_EVENT_LOG"] = str(ART / "adapter_events_selftest.jsonl")
    evlog = ART / "adapter_events_selftest.jsonl"
    if evlog.exists():
        evlog.unlink()          # fresh log for a reproducible battery
    command_adapter_v12.EVENTS.clear()

    scenarios = [
        ("forward_normal", {"velocity": {"forward": 0.3, "lateral": 0.0,
                                         "yaw_rate": 0.2},
                            "attention": {"target": "left"}, "duration": 2.0}),
        ("backward_in_range", {"velocity": {"forward": -0.3}}),
        ("backward_boundary", {"velocity": {"forward": -0.65}}),
        ("backward_deep_clamp", {"velocity": {"forward": -0.9}}),
        ("multi_channel_clamp", {"velocity": {"forward": 9.0, "lateral": 5.0,
                                              "yaw_rate": -5.0}}),
    ]
    results = []
    for name, cmd in scenarios:
        r = command_adapter_v12.adapt(cmd)
        results.append({"scenario": name, "input": cmd,
                        "cmd7": r["cmd7"], "requested_cmd7": r["requested_cmd7"],
                        "saturation_mask": r["saturation_mask"],
                        "warnings": r["warnings"]})
    rejections = []
    for name, bad in [("unknown_field", {"style": "happy"}),
                      ("bad_skill", {"skill": "dance"}),
                      ("unknown_velocity", {"velocity": {"sideways": 1}}),
                      ("bad_target", {"attention": {"target": "back"}}),
                      ("bad_duration", {"duration": 60})]:
        try:
            command_adapter_v12.adapt(bad)
            rejections.append({"scenario": name, "rejected": False,
                               "error": None})
        except command_adapter_v12.CommandValidationError as e:
            rejections.append({"scenario": name, "rejected": True,
                               "error": str(e)})

    (ART / "adapter_selftest.json").write_text(json.dumps({
        "artifact": "adapter_selftest_v12",
        "module": "humanoidverse/planner/command_adapter_v12.py",
        "adapter_version": command_adapter_v12.ADAPTER_VERSION,
        "invariants": [
            "backward in [-0.65, 0) passes untouched (no warning, no event)",
            "forward < -0.65 clamps to -0.65 + below_validated_range event",
            "every saturation appended to JSONL event log (durable)",
            "structural errors raise CommandValidationError (unlogged)",
        ],
        "scenarios": results, "rejections": rejections,
        "events_written": len(evlog.read_text().splitlines()),
        "event_log": evlog.name,
    }, indent=1))
    print(f"-> {ART}/capability_envelope_v12.json")
    print(f"-> {ART}/adapter_selftest.json (+{len(evlog.read_text().splitlines())} events)")
    ok = (all(r["rejected"] for r in rejections)
          and results[1]["cmd7"][4] == -0.3 and not results[1]["saturation_mask"][4]
          and results[3]["cmd7"][4] == -0.65 and results[3]["saturation_mask"][4])
    print("battery:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
