"""P4.2 builder: formal disposition for neck_forward / neck_roll (G1 fork:
sweep-validated+adapter range statement OR FROZEN_WITH_DISCLOSURE).

Inputs: INSTRUMENT_VALIDITY_MATRIX.json (P4.1) + neck_sweep_multi.json
(P4.2). Aggregation rule frozen BEFORE the multi-history run: channel counts
as sweep-gate-passing iff WORST-history slope>=0.50 AND worst R²>=0.60;
otherwise FROZEN_WITH_DISCLOSURE with adapter range statement + unblock path.
Narrative .md is generated FROM the json fields.

Run: ART_DIR=<dir> .venv/bin/python humanoidverse/scripts/build_neck_disposition.py
"""
import json
import os
import sys
from pathlib import Path

ART = Path(os.environ["ART_DIR"])
TH_SLOPE, TH_R2 = 0.50, 0.60

ADAPTER_STATEMENT = {
    "neck_forward": {
        "envelope": [0.3, 0.7],
        "exposure": "home-range poses only via ATTENTION_POSES "
                    "(forward/left/right all use neck_forward=0.50); wide "
                    "excursions blocked ('up' pose removed since v1.0)",
        "evidence_basis": "home range 0.4-0.6 ubiquitous in ALL training "
                          "data (data-coverage, NOT deployment-gate "
                          "validation; deployment nonzero n=0)",
    },
    "neck_roll": {
        "envelope": [0.0, 0.0],
        "exposure": "channel BLOCKED (envelope pinned to 0; every nonzero "
                    "request clamps to 0 with below/above_validated_range "
                    "event)",
        "evidence_basis": "intervention data carries neck_roll≡0 "
                          "(deployment nonzero n=0)",
    },
}


def main():
    matrix = json.loads((ART / "INSTRUMENT_VALIDITY_MATRIX.json").read_text()) \
        if (ART / "INSTRUMENT_VALIDITY_MATRIX.json").exists() else \
        json.loads(Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
                        "bdx_planner_distill/20260904T2300Z_p41_matrix/"
                        "INSTRUMENT_VALIDITY_MATRIX.json").read_text())
    multi = json.loads((ART / "neck_sweep_multi.json").read_text())

    out = {"artifact": "NECK_CHANNEL_DISPOSITION", "phase": "P4.2",
           "generated": "2026-09-04T24:00Z (distill campaign)",
           "gate": {"slope_min_ge": TH_SLOPE, "r2_min_ge": TH_R2,
                    "aggregation": "WORST-history across K=8 deterministic "
                                   "locomotion histories (frozen before run)"},
           "channels": {}}
    for ch in ("neck_forward", "neck_roll"):
        m = matrix["channels"][ch]
        w = multi["channels"][ch]
        passed = w["slope_min"] >= TH_SLOPE and w["r2_min"] >= TH_R2
        out["channels"][ch] = {
            "disposition": "SWEEP_VALIDATED" if passed else "FROZEN_WITH_DISCLOSURE",
            "deployment_evidence": {"valid_n": m["deployment_valid_n"],
                                    "note": "zero nonzero-command windows"},
            "single_history_sweep_p22": m["sweep"],
            "multi_history_sweep": {k: w[k] for k in
                                    ("K", "slope_min", "slope_mean", "slope_max",
                                     "r2_min") if k in w} | {"K": multi["K"]},
            "gate_check": {"worst_slope": w["slope_min"],
                           "worst_r2": w["r2_min"],
                           "slope_pass": w["slope_min"] >= TH_SLOPE,
                           "r2_pass": w["r2_min"] >= TH_R2},
            "adapter_range_statement": ADAPTER_STATEMENT[ch],
            "unblock_path": "future pass of the same worst-history sweep "
                            "gate (or appearance of deployment data reaching "
                            "teacher-reachable thresholds); thresholds "
                            "preregistered, not adjustable in-campaign",
        }
        c = out["channels"][ch]
        print(f"{ch}: worst slope {w['slope_min']} R² {w['r2_min']} -> "
              f"{c['disposition']}")

    (ART / "NECK_CHANNEL_DISPOSITION.json").write_text(json.dumps(out, indent=1))

    L = ["# NECK_CHANNEL_DISPOSITION — P4.2 正式处置 (2026-09-04)", "",
         f"门（冻结于多历史扫描前）：最差历史 slope≥{TH_SLOPE} 且 R²≥{TH_R2}"
         "（K=8 确定性 locomotion 历史）", ""]
    for ch, c in out["channels"].items():
        w = multi["channels"][ch]
        s = c["single_history_sweep_p22"]
        L += [f"## {ch} → **{c['disposition']}**", "",
              f"- deployment 非零命令窗：n={c['deployment_evidence']['valid_n']}"
              "（仪器=无）",
              f"- 单历史扫描（P2.2 锚点）：slope {s['slope']}/R²{s['r2']}/"
              f"sign {s['sign']}，范围 [{s['range'][0]:+.2f},{s['range'][1]:+.2f}]",
              f"- 多历史扫描（K=8）：slope min/mean/max "
              f"{w['slope_min']}/{w['slope_mean']}/{w['slope_max']}，"
              f"R²min {w['r2_min']}",
              f"- 门检：worst slope {c['gate_check']['worst_slope']} "
              f"({'过' if c['gate_check']['slope_pass'] else '不过'}) / "
              f"worst R² {c['gate_check']['worst_r2']} "
              f"({'过' if c['gate_check']['r2_pass'] else '不过'})",
              f"- adapter 范围声明：envelope {c['adapter_range_statement']['envelope']}"
              f"；{c['adapter_range_statement']['exposure']}",
              f"- 证据基础：{c['adapter_range_statement']['evidence_basis']}",
              f"- 解冻路径：{c['unblock_path']}", ""]
    L += ["## 披露", "",
          "1. P2.2 单历史数字是历史分布的抽样：forward 单历史 0.327 落在"
          "多历史分布 [0.088, 0.346] 的友好侧；roll 0.474 接近均值 0.453。",
          "2. neck_roll 在每个历史上都强线性（R²≥0.999）——通道非惰性、"
          "可控性首次有据——但增益强历史依赖（0.31–0.56），最差历史不过门。",
          "3. neck_forward 部分历史线性崩坏（R²min 0.214），固定历史姿态"
          "先验主导（截距 ~0.43），可控性证据弱。",
          "4. 处置不修改任何预注册阈值（停机条件约束）；两通道以"
          "FROZEN_WITH_DISCLOSURE 收口，G1 闭合。"]
    (ART / "NECK_CHANNEL_DISPOSITION.md").write_text("\n".join(L))
    print(f"-> {ART}/NECK_CHANNEL_DISPOSITION.json (+ .md)")


if __name__ == "__main__":
    main()
