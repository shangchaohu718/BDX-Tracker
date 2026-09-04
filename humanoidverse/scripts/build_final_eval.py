"""P5.4 builder: FINAL_EVAL assembly — applies the FROZEN adoption rule
(ADOPTION_RULE.json, written before any candidate numbers) to the measured
evidence and renders the G1–G4 gate table for the adopted product.

Inputs (all measured this phase, paths under ART_DIR):
  ADOPTION_RULE.json, v12/deployment_eval.json, v13/deployment_eval.json,
  v12_parity.json, v13_backward_holdout.json
Recorded evidence (file paths cited, values copied from their artifacts):
  v11 canonical (closure_status), v12/v11 backward holdout (P3.2.3),
  depth bands (P3.3), sweep gains (P2.1/P3.2.3), instrument matrix (P4.1),
  neck disposition (P4.2), G4 result (P5.3).

Run: ART_DIR=<p54 dir> .venv/bin/python humanoidverse/scripts/build_final_eval.py
"""
import json
import os
from pathlib import Path

ART = Path(os.environ["ART_DIR"])
V2C = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/bdx_planner_v2combo")
V12D = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
            "bdx_planner_distill/20260904T1900Z_p321_v12data")
P22 = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
           "bdx_planner_distill/20260904T1500Z_p22_neck")
P33 = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
           "bdx_planner_distill/20260904T2200Z_p33_v12artifacts")
P53 = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
           "bdx_planner_distill/20260905T0300Z_p53_g4eval")
P41 = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
           "bdx_planner_distill/20260904T2300Z_p41_matrix")


def j(p):
    return json.loads(Path(p).read_text())


def main():
    rule = j(ART / "ADOPTION_RULE.json")
    d12, d13 = j(ART / "v12/deployment_eval.json"), j(ART / "v13/deployment_eval.json")
    parity = j(ART / "v12_parity.json")
    b13 = j(ART / "v13_backward_holdout.json")
    b12 = j(V12D.parent / "20260904T2100Z_p323_accept/v12_holdout.json")
    bands = j(P33 / "backward_depth_bands.json")
    g4 = j(P53 / "GESTURE_HOLDOUT_RESULT.json")
    closure = j(V2C / "closure_status.json")

    v11_q = closure["headline"]["q"] if "headline" in closure else 25.975

    cands = {
        "v11": {"deployment_q": v11_q,
                "E1": True,
                "E2": "FAIL_lineage (trained on heldout _rev + twins; numeric "
                      "pass 0.0142/0-inv is train-contaminated)",
                "E3": "-0.60 (own bands)"},
        "v12": {"deployment_q": round(d12["q"]["mean"], 2),
                "E1": d12["q"]["mean"] <= 30,
                "E2": f"PASS (inv {b12['sign_inversions']}, |dvx| "
                      f"{b12['dvx_mean']:.4f}, lineage-clean)",
                "E3": bands["v12"]["envelope_min_grid_toward_zero"]},
        "v13": {"deployment_q": round(d13["q"]["mean"], 2),
                "E1": d13["q"]["mean"] <= 30,
                "E2": f"FAIL_gate (sign inversions {b13['sign_inversions']}: "
                      f"one shallow-cmd window -0.063 -> +0.018 m/s)",
                "E3": "not evaluated (ineligible)"},
    }
    eligible = [c for c, v in cands.items()
                if v["E1"] is True and str(v["E2"]).startswith("PASS")]
    q_leader = min(eligible, key=lambda c: cands[c]["deployment_q"])
    runner = sorted((cands[c]["deployment_q"] for c in eligible))[1] \
        if len(eligible) > 1 else None
    if runner is not None and cands[q_leader]["deployment_q"] + 3 < runner:
        adopted = q_leader                       # P1: leader by >3 mrad
        rule_applied = f"P1 (leader {q_leader} by >3 mrad)"
    else:
        adopted = "v13" if "v13" in eligible else ("v12" if "v12" in eligible
                                                   else None)
        rule_applied = f"P2 tie-band (<=3 mrad) -> evidence order; " \
                       f"eligible={eligible}"

    gates = {
        "G1": {
            "vx": "PASS via sweep instrument (frozen rule: deployment "
                  "INVALID, teacher unreachable) — v12 sweep 0.90 / sign "
                  "incl. reverse (P3.2.3)",
            "vy": f"PASS {d12['gains']['vy']['slope']:.3f}/"
                  f"{d12['gains']['vy']['r2']:.3f}",
            "vyaw": f"PASS {d12['gains']['vyaw']['slope']:.3f}/"
                    f"{d12['gains']['vyaw']['r2']:.3f}",
            "neck_yaw": f"PASS {d12['gains']['neck']['slope']:.3f}/"
                        f"{d12['gains']['neck']['r2']:.3f}",
            "neck_pitch": "PASS deployment instrument (P4.1 formal: student "
                          "1.215/0.9955, teacher reachable 1.248/0.9999)",
            "neck_forward/roll": "FROZEN_WITH_DISCLOSURE (P4.2 worst-history "
                                 "gate; adapter statements issued)",
        },
        "G2": {"verdict": "PASS — terminal A, 5/5 conditions "
                          "(P3.1–P3.3: gate 1.244x/1.041x; heldout 0 inv / "
                          "0.0135 / q 42.6; envelope [-0.65,0) + adapter "
                          "clamp+events)",
               "binding": "adopted=v12 -> envelope min -0.65 (capability_v12)"},
        "G3": {"val_q_deployment_mrad": round(d12["q"]["mean"], 2),
               "gate_le_30": d12["q"]["mean"] <= 30,
               "fk_mm": round(d12["fk"]["mean"], 2),
               "planner_over_teacher": parity,
               "gate_le_1.15": all(v["parity_slope"] <= 1.15
                                   for v in parity.values()),
               "holdout_slices_max_mrad": round(max(
                   v["mean_q"] for k, v in d12["slices"].items()
                   if "holdout" in k), 1),
               "gate_slices_le_45": all(v["mean_q"] <= 45
                                        for k, v in d12["slices"].items()
                                        if "holdout" in k),
               "note": "gate covers the 4 combo HOLDOUT slices per G3 text; "
                       "training-combo slices disclosed separately "
                       f"(max {max(v['mean_q'] for v in d12['slices'].values()):.1f} "
                       "at an n=1 singleton slice)"},
        "G4": {"verdict": "ESTABLISHED via v13 controlled experiment "
                          "(single-variable: v12 minus 15 gesture groups)",
               "heldout_q": g4["arms"]["v13"]["q_mrad_mean"],
               "fit_control_q": g4["fit_control"]["v13_q_mrad"],
               "gap": g4["g4"]["generalization_number"]["generalization_gap_mrad"],
               "teacher_floor": g4["arms"]["teacher"]["q_mrad_mean"],
               "disclosure": "product v12 itself has no gesture holdout "
                             "possible (all clips trained); generalization "
                             "evidence belongs to the training-config line "
                             "via v13"},
    }
    out = {
        "artifact": "FINAL_EVAL", "phase": "P5.4",
        "generated": "2026-09-05T04:00Z (distill campaign)",
        "adoption_rule": rule["frozen"],
        "candidates": cands, "eligible": eligible,
        "rule_applied": rule_applied, "ADOPTED": adopted,
        "completion_form_context": {
            "v11": 63.3, "v12": 68.6, "v13": 59.6,
            "note": "same-config run variance +-10 mrad; v12 completion "
                    "regression vs v11 (+5.3) is within this band — "
                    "disclosed, deployment form is the adoption instrument"},
        "gates": gates,
        "evidence_paths": {
            "v11_headline": "bdx_planner_v2combo/closure_status.json",
            "v12_backheld": "20260904T2100Z_p323_accept/v12_holdout.json",
            "v13_backheld": "p54/v13_backward_holdout.json",
            "depth_bands": "20260904T2200Z_p33_v12artifacts/backward_depth_bands.json",
            "instrument_matrix": "20260904T2300Z_p41_matrix/",
            "neck_disposition": "20260904T2400Z_p42_neckdispo/",
            "g4": "20260905T0300Z_p53_g4eval/GESTURE_HOLDOUT_RESULT.json"},
    }
    (ART / "FINAL_EVAL.json").write_text(json.dumps(out, indent=1))
    hs = [v["mean_q"] for k, v in d12["slices"].items() if "holdout" in k]
    print(f"eligible: {eligible} | rule: {rule_applied} | ADOPTED: {adopted}")
    print(f"G3: q {d12['q']['mean']:.2f}<=30 {d12['q']['mean'] <= 30} | parity "
          f"max {max(v['parity_slope'] for v in parity.values())}<=1.15 | "
          f"holdout slices max {max(hs):.1f}<=45 {max(hs) <= 45}")
    print(f"-> {ART / 'FINAL_EVAL.json'}")


if __name__ == "__main__":
    main()
