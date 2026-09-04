"""Merge per-motion eval JSONs into one cross-pool comparison table (markdown).

Usage: python3 compare_pools.py out.md json1.json json2.json ...
Each JSON's short label = its filename (editable below in the report header).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: str):
  d = json.load(open(p))
  label = Path(p).name.replace("eval_", "").replace("_per_motion.json", "")
  rows = {m["motion"].replace("re_dancegen_excited_", "").replace(".npz", ""):
          m for m in d["per_motion"]}
  agg = d["aggregates"]
  return label, rows, agg


def main():
  out_path, jsons = sys.argv[1], sys.argv[2:]
  pools = [load(p) for p in jsons]
  all_motions = sorted({m for _, rows, _ in pools for m in rows})

  lines = [f"# Cross-pool per-motion comparison ({len(jsons)} runs)",
           "", "fall rate / min z worst / joint err last", "",
           "| motion | " + " | ".join(lbl for lbl, _, _ in pools) + " |",
           "|---" * (len(pools) + 1) + "|"]
  for m in all_motions:
    cells = []
    for _, rows, _ in pools:
      r = rows.get(m)
      cells.append("—" if r is None else
                   f"{r['fall_rate']:.2f} / {r['min_z_worst']:.3f} / {r['joint_err_last']}")
    lines.append(f"| {m} | " + " | ".join(cells) + " |")

  lines += ["", "## Aggregates (fall)", "",
            "| pool | macro | worst | P10 |", "|---|---|---|---|"]
  for lbl, _, agg in pools:
    f = agg["fall_rate"]
    lines.append(f"| {lbl} | {f['macro_avg']} | {f['worst_motion']} | {f['p10_motion']} |")

  lines += ["", "## Aggregates (joint err last)", "",
            "| pool | macro | worst | P10 |", "|---|---|---|---|"]
  for lbl, _, agg in pools:
    j = agg["joint_err_last"]
    lines.append(f"| {lbl} | {j['macro_avg']} | {j['worst_motion']} | {j['p10_motion']} |")

  Path(out_path).write_text("\n".join(lines) + "\n")
  print("\n".join(lines))
  print("saved", out_path)


if __name__ == "__main__":
  main()
