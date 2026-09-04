"""P3.2.1 build the v12 training package (backward A-line, user ruling).

Creates <TS>_p321_v12data/ :
  - hardlink copy of the canonical baseline dir (big npz shared, zero copy)
  - NEW splits.json  : train = original train
                        MINUS 30 anomalous _rev clips (defect fix, cause-listed)
                        MINUS heldout families' _rev clips
                        MINUS heldout families' forward twins (no variant leakage)
                      val = original val (unchanged, standard-gate comparability)
  - backward_heldout.json : heldout families + clip lists + exclusions + rationale
Heldout families are chosen from the 370 clean `_rev` at family level
(stem minus wXXeXXsX/speed/mirror variant codes), fixed seed.

G2 acceptance for v12 is measured ONLY on backward_heldout (never in training
in any variant form).

Run: .venv/bin/python humanoidverse/scripts/build_v12_package.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

BASE = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
            "bdx_planner_distill/20260904T1200Z_p0_baseline")
POC_POOL = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/bdx_planner_v2")
OUT = Path("/home/tcl/Desktop/start/BFM-zero/humanoidverse/data/"
           "bdx_planner_distill/20260904T1900Z_p321_v12data")
HELDOUT_FRAC = 0.15
SEED = 20260904


def fam(n):
    st = n.split("/")[-1].replace("_rev", "").replace("_mirror", "")
    st = re.sub(r"_w\d+e\d+s\d+", "", st)
    st = re.sub(r"_speed[\d.]+", "", st)
    return st


def main():
    meta = json.loads((BASE / "meta.json").read_text())
    names = [c["name"] for c in meta["clips"]]
    idx = {n: i for i, n in enumerate(names)}
    splits = json.loads((BASE / "splits.json").read_text())
    train0, val0 = splits["train"], splits["val"]
    tset, vset = set(train0), set(val0)

    # clean/anomalous partition (same contract as eval_backward_gate)
    zp = np.load(POC_POOL / "sparse_full_v2.npz", allow_pickle=True)
    lengths = zp["lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)])
    bp = zp["base_pos"]
    poc_meta = json.loads((POC_POOL / "meta.json").read_text())
    poc_names = [c["name"] for c in poc_meta["clips"]]
    poc_idx = {n: i for i, n in enumerate(poc_names)}
    revs = [n for n in poc_names
            if n.startswith("v2_backward/") and n.endswith("_rev")]
    clean, anomalous = [], []
    for n in revs:
        stem = n.split("/")[1][:-4]
        cands = [nm for nm in poc_names if nm.split("/")[-1] == stem
                 and not nm.startswith("v2_backward")]
        if not cands:
            anomalous.append((n, "no-original-pair"))
            continue
        io_, ir = poc_idx[cands[0]], poc_idx[n]
        dxo = bp[starts[io_ + 1] - 1, 0] - bp[starts[io_], 0]
        dxr = bp[starts[ir + 1] - 1, 0] - bp[starts[ir], 0]
        if dxo * dxr <= 0:
            clean.append(n)
        else:
            anomalous.append((n, f"dx-not-flipped({dxo:+.2f}->{dxr:+.2f})"))
    anomalous = [(n, r) for n, r in anomalous if r != "ok"]
    assert len(clean) == 370 and len(anomalous) == 30, (len(clean), len(anomalous))

    # family-level heldout selection among clean _rev.
    # Deterministic rule (fixed BEFORE any v12 result is seen):
    #   per category (of the _rev clip's original pair), eligible families are
    #   those <=30% of the category's clean total (size cap keeps the bulk of
    #   backward learning in train); pick smallest-first until the category's
    #   heldout reaches >=15% of its clean count or eligible families run out.
    poc_cat = {c["name"]: c["category"] for c in poc_meta["clips"]}
    by_fam = {}
    for n in clean:
        stem = n.split("/")[1][:-4]
        cat = poc_cat[next(nm for nm in poc_names
                           if nm.split("/")[-1] == stem
                           and not nm.startswith("v2_backward/"))]
        by_fam.setdefault(fam(n), {"cat": cat, "clips": []})["clips"].append(n)
    held_fams, n_held = [], 0
    for cat in sorted({v["cat"] for v in by_fam.values()}):
        tot = sum(len(v["clips"]) for v in by_fam.values() if v["cat"] == cat)
        cap = 0.30 * tot
        elig = sorted((f for f, v in by_fam.items()
                       if v["cat"] == cat and len(v["clips"]) <= cap),
                      key=lambda f: (len(by_fam[f]["clips"]), f))
        got = 0
        for f in elig:
            if got >= 0.15 * tot:
                break
            held_fams.append(f)
            got += len(by_fam[f]["clips"])
            n_held += len(by_fam[f]["clips"])
    held_clips = {n for f in held_fams for n in by_fam[f]["clips"]}

    # exclusions from train
    anom_set = {n for n, _ in anomalous}
    drop_rev = (held_clips | anom_set) & tset
    fwd_twins = [n for n in train0
                 if fam(n) in set(held_fams) and not n.startswith("v2_backward/")]
    # also drop anomalous _rev from VAL side? no — val stays untouched for
    # standard-gate comparability; the 30 anomalous are train-side defect fix.
    train_new = [n for n in train0
                 if n not in drop_rev and n not in set(fwd_twins)]
    print(f"train: {len(train0)} -> {len(train_new)} "
          f"(drop _rev heldout/anom {len(drop_rev)}, fwd twins {len(fwd_twins)})")
    held_by_cat={}
    for f in held_fams: held_by_cat.setdefault(by_fam[f]["cat"],[]).append((f,len(by_fam[f]["clips"])))
    print(f"heldout families {len(held_fams)}: {held_by_cat} "
          f"({n_held} clean _rev clips, {len(held_clips & vset)} were in val)")
    print(f"val unchanged: {len(val0)}")

    OUT.mkdir(parents=True, exist_ok=True)
    # hardlink-copy baseline contents (files not present in OUT yet)
    for p in BASE.iterdir():
        dst = OUT / p.name
        if not dst.exists():
            shutil.os.link(p, dst)
    # replace splits.json (break hardlink first) and write register
    (OUT / "splits.json").unlink()
    (OUT / "splits.json").write_text(json.dumps(
        {"source": "v1+planner_v2combo-merged [distill-campaign v12: "
                   "backward A-line lineage split]",
         "train": train_new, "val": val0}, indent=0))
    register = {
        "built": "2026-09-04T19:00Z", "seed": SEED,
        "heldout_fraction_target": HELDOUT_FRAC,
        "clean_rev": len(clean), "anomalous_rev": [{"clip": n, "reason": r}
                                                   for n, r in anomalous],
        "heldout_families": held_fams,
        "heldout_clips": sorted(held_clips),
        "n_heldout_clips": len(held_clips),
        "dropped_from_train": {"rev_heldout_or_anomalous": sorted(drop_rev),
                               "forward_twins": fwd_twins},
        "val_side_rev_kept_for_standard_gate": sorted(
            n for n in val0 if n.startswith("v2_backward/")),
        "rationale": "family = stem minus variant codes (wXXeXXsX/speed/mirror). "
                     "Heldout families excluded from v12 train in BOTH _rev and "
                     "forward-twin form (ruling: no twin-variant train/val split). "
                     "30 anomalous _rev (dx not flipped / no pair) excluded from "
                     "train as lineage defect fix; originals untouched.",
    }
    (OUT / "backward_heldout.json").write_text(json.dumps(register, indent=1))
    print(f"-> {OUT}/ (splits.json replaced, backward_heldout.json written)")


if __name__ == "__main__":
    main()
