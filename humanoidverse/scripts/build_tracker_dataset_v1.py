"""Build tracker_dataset_v1: semantic/activity-sampled replacement for the
VERSION_CAPS composition (ruling 2026-08-24 night 12).

Policy (supersedes VERSION_CAPS):
  eligibility : train split, schema/FK ok, frames >= 251 (seq_length 250 + 1)
  families    : semantic by version (walk / walk_head / walk_backward / perturb
                / augment / stand / gesture / planner / intervene / patch)
  family caps : explicit targets; unique command families favored over hours
                (receiver doctrine); static families (stand/gesture) included
                but small; derived mirror/speed variants capped (augment)
  lineage cap : per-group_id cap of 6 ONLY for multi-group families (intervene,
                patch) — v0/v1/v2 use a single shared group_id (no real lineage)
  within-family selection: activity-sorted linspace (deterministic, spreads
                activity; non-derived before derived)
Outputs:
  humanoidverse/data/bdx_14dof_train_v1.pkl / bdx_14dof_eval_v1.pkl
  results/bfmzero-bdx-full/tracker_dataset_v1_manifest.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from easydict import EasyDict

from humanoidverse.scripts.convert_dataset_to_pkl import MOTION_CFG, convert_clip

DATASET_ROOT = Path("/home/tcl/Desktop/start/dataset")
DRYRUN = Path("results/bfmzero-bdx-full/tracker_dryrun_eligibility.json")
OUT_TRAIN = Path("humanoidverse/data/bdx_14dof_train_v1.pkl")
OUT_EVAL = Path("humanoidverse/data/bdx_14dof_eval_v1.pkl")
OUT_MANIFEST = Path("results/bfmzero-bdx-full/tracker_dataset_v1_manifest.json")

FAMILY_OF = {
    "v0": "gesture", "v1_walk": "walk", "v1_walkhead": "walk_head",
    "v2_backward": "walk_backward", "v1_perturb": "perturb", "v1_augment": "augment",
    "v1_stand": "stand", "v3_planner": "planner", "v4_intervene": "intervene",
    "v5_intervene2": "intervene", "v6_patch": "patch", "v7_ikgait": "ikgait",
}
FAMILY_TARGETS = {
    "walk": 700, "walk_head": 183, "walk_backward": 341, "perturb": 800,
    "augment": 300, "stand": 230, "gesture": 185, "planner": 285,
    "intervene": 3000, "patch": 400,
}
GROUP_CAP_FAMILIES = {"intervene", "patch"}
GROUP_CAP = 6

_W = {}


def init_worker():
    import torch
    import yaml
    from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
    skel = Humanoid_Batch(MOTION_CFG, torch.device("cpu"))
    rcfg = yaml.safe_load(open(Path(__file__).parents[1] / "config/robot/bdx/bdx_14dof.yaml"))["robot"]
    skel.robot_joint_names = rcfg["dof_names"]
    _W["skel"] = skel
    _W["parents"] = skel._parents.numpy()
    _W["q_fixed"] = skel._local_rotation[0].numpy()


def convert_one(rec):
    f = DATASET_ROOT / rec["file"]
    try:
        entry, err = convert_clip(dict(np.load(f, allow_pickle=True)), _W["skel"], _W["parents"], _W["q_fixed"], 30)
        return rec["key"], entry, None
    except Exception as e:
        return rec["key"], None, f"{type(e).__name__}: {str(e)[:120]}"


def select_family(fam, clips, t1, t2):
    """Activity-sorted linspace; non-derived first; group cap for real-lineage families."""
    non_derived = sorted([c for c in clips if not c["derived"]], key=lambda c: c["activity_proxy_joint_speed"])
    derived = sorted([c for c in clips if c["derived"]], key=lambda c: c["activity_proxy_joint_speed"])
    target = FAMILY_TARGETS[fam]
    def take(pool, n):
        if len(pool) <= n:
            return pool
        return [pool[i] for i in np.linspace(0, len(pool) - 1, n).astype(int)]
    quota_nd = min(len(non_derived), target)
    sel = take(non_derived, quota_nd) + take(derived, target - quota_nd)
    if fam in GROUP_CAP_FAMILIES:
        per_group = {}
        capped = []
        for c in sel:
            g = c["group_id"]
            if per_group.get(g, 0) < GROUP_CAP:
                per_group[g] = per_group.get(g, 0) + 1
                capped.append(c)
        sel = capped
    return sel


def main():
    dry = json.load(open(DRYRUN))
    t1, t2 = dry["summary"]["activity_terciles_global"]
    ok = [c for c in dry["clips"] if c["ok"]]
    train_elig = [c for c in ok if c["split"] == "train" and c["frames"] >= 251]
    eval_elig = [c for c in ok if c["split"] == "eval" and c["frames"] >= 251]
    for c in ok:
        c["family"] = FAMILY_OF[c["version"]]

    by_fam = {}
    for c in train_elig:
        by_fam.setdefault(c["family"], []).append(c)
    selected = []
    for fam in FAMILY_TARGETS:
        sel = select_family(fam, by_fam.get(fam, []), t1, t2)
        selected.extend(sel)
        print(f"{fam}: {len(sel)}/{len(by_fam.get(fam, []))} selected", flush=True)

    acts = np.array([c["activity_proxy_joint_speed"] for c in selected])
    static_frac = float(np.mean([(c["family"] in ("stand", "gesture")) for c in selected]))
    print(f"total {len(selected)} clips, static(stand+gesture) {static_frac:.1%}, "
          f"activity terciles low/mid/high = "
          f"{np.mean(acts < t1):.1%}/{np.mean((acts >= t1) & (acts < t2)):.1%}/{np.mean(acts >= t2):.1%}", flush=True)

    init_worker()
    def rec_of(c):
        v, cid = c["key"].split("/", 1)
        return {"key": c["key"], "file": f"{v}/clips/{cid}.npz"}
    records = [rec_of(c) for c in selected] + [rec_of(c) for c in eval_elig]
    with mp.Pool(16, initializer=init_worker) as pool:
        results = list(pool.imap_unordered(convert_one, records, chunksize=16))

    train_out, eval_out, failed = {}, {}, []
    sel_keys = {c["key"] for c in selected}
    for key, entry, err in results:
        if err is not None:
            failed.append({"key": key, "error": err})
        elif key in sel_keys:
            train_out[key] = entry
        else:
            eval_out[key] = entry

    joblib.dump(train_out, OUT_TRAIN, compress=3)
    joblib.dump(eval_out, OUT_EVAL, compress=3)
    tr_fr = sum(v["root_trans_offset"].shape[0] for v in train_out.values())
    ev_fr = sum(v["root_trans_offset"].shape[0] for v in eval_out.values())
    manifest = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "builder": "build_tracker_dataset_v1.py",
            "supersedes": "legacy_tracker_pool_v0 (VERSION_CAPS composition)",
            "upstream": "canonical final-build2 (RELEASE.json) — read-only",
            "eligibility": "schema/FK ok (dry-run), frames>=251 (seq_length 250)",
            "policy": {
                "family_targets": FAMILY_TARGETS,
                "group_cap": {f"{GROUP_CAP} per group_id": sorted(GROUP_CAP_FAMILIES)},
                "within_family": "activity-sorted linspace, non-derived first",
                "note": "v0/v1/v2 share one group_id per version (no real lineage) — group cap not applied there",
            },
        },
        "train": {"clips": len(train_out), "frames": int(tr_fr), "hours_50fps": tr_fr / 50 / 3600,
                  "static_family_fraction": static_frac,
                  "activity_tercile_fractions": [float(np.mean(acts < t1)), float(np.mean((acts >= t1) & (acts < t2))), float(np.mean(acts >= t2))],
                  "family_counts": {f: sum(1 for c in selected if c["family"] == f) for f in FAMILY_TARGETS}},
        "eval": {"clips": len(eval_out), "frames": int(ev_fr)},
        "conversion_failures": failed,
        "selected_keys": sorted(sel_keys),
    }
    assert not OUT_MANIFEST.exists()
    OUT_MANIFEST.write_text(json.dumps(manifest) + "\n")
    print(f"train {len(train_out)} clips / {tr_fr} frames ({tr_fr/50/3600:.2f}h); eval {len(eval_out)} clips; failed {len(failed)}")


if __name__ == "__main__":
    main()
