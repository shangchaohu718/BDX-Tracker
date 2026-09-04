"""P5.2 build the v13 training package (gesture whole-group holdout, G4).

Base = the v12 lineage-clean package (current campaign pool reality). v13
differs by ONE variable: entire gesture motion-groups are removed from train
and registered as gesture_heldout (their val members stay in val for
standard-gate comparability but are part of the heldout register — never
trained in any variant for v13).

Group selection rule FROZEN at P5.1 (before any G4 model eval):
  gesture motion groups (prefix lineage, see assess_gesture_holdout.py)
  sorted by total clip count ascending, accumulate WHOLE groups until
  >=15 held clips; train must keep >=20 gesture clips.

Motion-group function is copied verbatim from assess_gesture_holdout.py
(the frozen grouping).

Run: .venv/bin/python humanoidverse/scripts/build_v13_package.py
"""
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
            "bdx_planner_distill/20260904T1900Z_p321_v12data")
OUT = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
           "bdx_planner_distill/20260905T0200Z_p52_v13data")
HOLDOUT_MIN = 15
TRAIN_GESTURE_MIN = 20

VARIANT_RE = re.compile(r"(w\d+e\d+s\d+|_speed[\d.]+|_mirror)")
PREFIX_RE = re.compile(r"[_\-]?(v\d+|take\d*|t\d+|\d+[a-z]?|[a-z]\d+)$")


def family(name):
    return VARIANT_RE.sub("", name)


def motion_group(name):
    n = family(name)
    prev = None
    while prev != n:
        prev = n
        n = PREFIX_RE.sub("", n).rstrip("_-")
    return n or family(name)


def main():
    meta = json.loads((BASE / "meta.json").read_text())
    splits = json.loads((BASE / "splits.json").read_text())
    train0, val0 = splits["train"], splits["val"]
    tset, vset = set(train0), set(val0)
    cat = {c["name"]: c["category"] for c in meta["clips"]}

    gest = [c["name"] for c in meta["clips"] if c["category"] == "gesture"]
    grp = defaultdict(list)
    for n in gest:
        grp[motion_group(n)].append(n)

    # frozen rule: smallest-first whole groups until >=15 held clips
    held_groups, held_clips = [], set()
    for g, ns in sorted(grp.items(), key=lambda kv: (len(kv[1]), kv[0])):
        if len(held_clips) >= HOLDOUT_MIN:
            break
        held_groups.append(g)
        held_clips.update(ns)
    held_train = held_clips & tset
    train_gesture_left = sum(1 for n in train0
                             if cat.get(n) == "gesture" and n not in held_clips)
    assert len(held_clips) >= HOLDOUT_MIN, len(held_clips)
    assert train_gesture_left >= TRAIN_GESTURE_MIN, train_gesture_left

    train_new = [n for n in train0 if n not in held_train]
    # zero-leakage verification: no held-group member remains in train
    leak = [n for n in train_new if n in held_clips]
    assert not leak, leak[:5]

    print(f"v12 train {len(train0)} -> v13 train {len(train_new)} "
          f"(dropped {len(held_train)} gesture clips of {len(held_groups)} "
          f"held groups)")
    print(f"heldout register: {len(held_clips)} clips "
          f"({len(held_clips & vset)} were val-side); train gesture keeps "
          f"{train_gesture_left}")
    print(f"val unchanged: {len(val0)}")

    OUT.mkdir(parents=True, exist_ok=True)
    for p in BASE.iterdir():
        dst = OUT / p.name
        if not dst.exists():
            shutil.os.link(p, dst)
    (OUT / "splits.json").unlink()
    (OUT / "splits.json").write_text(json.dumps(
        {"source": "v12 lineage-clean [distill-campaign v13: gesture "
                   "whole-group holdout]",
         "train": train_new, "val": val0}, indent=0))

    register = {
        "built": "2026-09-05T02:00Z",
        "rule": "gesture motion groups (prefix lineage, frozen at P5.1) "
                "smallest-first until >=15 held clips; train keeps >=20 "
                "gesture clips; single variable vs v12",
        "heldout_groups": [{"group": g, "clips": sorted(grp[g]),
                            "n_train": len(set(grp[g]) & tset),
                            "n_val": len(set(grp[g]) & vset)}
                           for g in held_groups],
        "n_heldout_clips": len(held_clips),
        "train_gesture_remaining": train_gesture_left,
        "val_unchanged": len(val0) == len(val0),
        "leakage_check": "held-group members in v13 train: 0 (asserted)",
    }
    (OUT / "gesture_heldout.json").write_text(json.dumps(register, indent=1))
    print(f"-> {OUT}/ (splits.json replaced, gesture_heldout.json written)")


if __name__ == "__main__":
    main()
