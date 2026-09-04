"""P5.1 gesture lineage/source-group holdout feasibility assessment (G4
ruling 2026-09-04: no random frame splits, no twin-variant splits; if no
independent holdout can be formed -> GESTURE_TRAIN_FIT=VERIFIED +
GESTURE_GENERALIZATION=NOT_ESTABLISHED + G4=FROZEN_WITH_DISCLOSURE).

Enumerates gesture clips in the ACTIVE POOL of a data dir (default: v12
lineage-clean package; its splits are the current training reality),
maps each clip to its source family (stem with variant codes stripped),
and reports:
  - gesture clips total / in train / in val (twin leakage check)
  - family structure: families, per-family clip counts, val overlap
  - feasibility verdict under a decision rule FROZEN before inspection:
      a legal holdout exists iff >=2 gesture families AND at least
      GESTURE_HOLDOUT_MIN (=5) clips can be held out from families that
      have ZERO members in train after exclusion (no twins left behind),
      while >=20 gesture clips remain in train (train-fit still measurable).

Run: BDX_PLANNER_DATA=<v12data dir> ART_DIR=<out> \
     .venv/bin/python humanoidverse/scripts/assess_gesture_holdout.py
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA"))
ART = Path(os.environ["ART_DIR"])
GESTURE_HOLDOUT_MIN = 5
GESTURE_TRAIN_MIN = 20
VARIANT_RE = re.compile(r"(w\d+e\d+s\d+|_speed[\d.]+|_mirror)")
# gesture deep lineage: motion-level twin grouping. The walk-style stem rule
# leaves every gesture clip unique (no w/e/s codes in gesture names), but the
# names themselves carry motion identity: head_turn40/45/55, babble_1/2/3,
# re_dancegen_bounce_sway_3/15/31 are takes/variants of ONE motion source.
# Grouping = prefix with trailing numeric/take tokens stripped. This is the
# CONSERVATIVE direction (drops both-sides groups from the clean set).
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
    meta = json.loads((OUT_DIR / "meta.json").read_text())
    splits = json.loads((OUT_DIR / "splits.json").read_text())
    train, val = set(splits["train"]), set(splits["val"])

    gest = [c["name"] for c in meta["clips"] if c["category"] == "gesture"]
    g_train = [n for n in gest if n in train]
    g_val = [n for n in gest if n in val]
    grp_of = {n: motion_group(n) for n in gest}
    grp_clips = defaultdict(list)
    for n in gest:
        grp_clips[grp_of[n]].append(n)

    grp_stat = []
    for f, ns in sorted(grp_clips.items(), key=lambda kv: -len(kv[1])):
        grp_stat.append({
            "motion_group": f, "clips": len(ns),
            "in_train": sum(1 for n in ns if n in train),
            "in_val": sum(1 for n in ns if n in val),
            "example": ns[0]})

    twin_grps = [s for s in grp_stat if s["in_train"] and s["in_val"]]
    # OPTION A (no retrain): lineage-clean val gesture holdout = val clips
    # whose motion_group has ZERO train members
    clean_val = [n for n in g_val if grp_clips[grp_of[n]] and
                 all(m not in train for m in grp_clips[grp_of[n]])]
    clean_val_groups = sorted({grp_of[n] for n in clean_val})
    # OPTION B (retrain): hold out entire gesture motion_groups; smallest
    # first until >=HOLDOUT_MIN clips, keep >=TRAIN_MIN in train
    cand = sorted([s for s in grp_stat if s["in_train"] > 0],
                  key=lambda s: s["clips"])
    held, held_clips = [], 0
    for s in cand:
        if held_clips >= GESTURE_HOLDOUT_MIN:
            break
        held.append(s["motion_group"])
        held_clips += s["in_train"]
    remaining_train = len(g_train) - held_clips
    option_a_ok = len(clean_val) >= GESTURE_HOLDOUT_MIN
    option_b_ok = (len(grp_clips) >= 2 and held_clips >= GESTURE_HOLDOUT_MIN
                   and remaining_train >= GESTURE_TRAIN_MIN)

    out = {
        "artifact": "GESTURE_HOLDOUT_FEASIBILITY", "phase": "P5.1",
        "generated": "2026-09-05T01:00Z (distill campaign)",
        "pool": str(OUT_DIR.name),
        "lineage_rules": {
            "stem_rule": "walk-variant codes — INSUFFICIENT for gestures "
                         "(leaves all 441 unique)",
            "motion_group_rule": "prefix with trailing numeric/take tokens "
                                 "recursively stripped (head_turn40/45/55, "
                                 "babble_1/2/3, re_dancegen_<dance>_N = one "
                                 "motion source); CONSERVATIVE grouping "
                                 "chosen after seeing names, before any "
                                 "model evaluation",
            "sequencing_disclosure": "first pass with stem rule reported 0 "
                                     "leakage; name inspection revealed "
                                     "motion-level twins; the stricter rule "
                                     "can only shrink the clean set"},
        "decision_rule": {"families_ge": 2, "holdout_clips_ge":
                          GESTURE_HOLDOUT_MIN, "train_remain_ge":
                          GESTURE_TRAIN_MIN,
                          "frozen": "before inspection (G4 ruling boundary)"},
        "counts": {"gesture_total": len(gest), "in_train": len(g_train),
                   "in_val": len(g_val), "motion_groups": len(grp_clips)},
        "val_side": {
            "gesture_clips_in_val": len(g_val),
            "twin_leakage_groups": [s["motion_group"] for s in twin_grps],
            "lineage_clean_val_clips": sorted(clean_val),
            "lineage_clean_val_groups": clean_val_groups,
            "n_clean": len(clean_val)},
        "motion_group_table": grp_stat,
        "option_A_no_retrain": {
            "description": "evaluate student vs teacher on the lineage-clean "
                           "val gesture clips (groups absent from train)",
            "n_clips": len(clean_val), "feasible": option_a_ok},
        "option_B_retrain": {
            "description": "hold out entire gesture motion_groups, retrain "
                           "student (v13), evaluate on them",
            "candidate_groups": held, "clips": held_clips,
            "train_remaining": remaining_train, "feasible": option_b_ok},
        "feasible": option_a_ok or option_b_ok,
        "verdict": ("LINEAGE_CLEAN_VAL_EVALUABLE" if option_a_ok else
                    "RETRAIN_HOLDOUT_CONSTRUCTIBLE" if option_b_ok else
                    "FROZEN_WITH_DISCLOSURE"),
    }
    (ART / "GESTURE_HOLDOUT_FEASIBILITY.json").write_text(json.dumps(out, indent=1))
    print(f"gesture: total {len(gest)} train {len(g_train)} val {len(g_val)} "
          f"motion_groups {len(grp_clips)}")
    print(f"twin-leakage groups: {len(twin_grps)}; clean val clips: "
          f"{len(clean_val)} in groups {clean_val_groups}")
    print(f"option A (no retrain): {len(clean_val)} clips -> "
          f"{'FEASIBLE' if option_a_ok else 'insufficient'}")
    print(f"option B (retrain): {held_clips} clips from {len(held)} groups, "
          f"train keeps {remaining_train} -> "
          f"{'FEASIBLE' if option_b_ok else 'insufficient'}")
    print(f"VERDICT: {out['verdict']}")
    return out


if __name__ == "__main__":
    main()
