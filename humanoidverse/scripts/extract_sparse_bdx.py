"""P0 for the BDX shared-latent planner: extract the sparse (29D) command
representation and full-motion (41D) targets from the source npz clips.

Sources:
  - clips:   DATASET_ROOT/{version}/clips/*.npz (mjlab exports; body order ==
             bdx_v4.xml MuJoCo body order, quats wxyz, 50 Hz)
  - clip list: humanoidverse/data/bdx_14dof_dataset.meta.json — the 1154 clips
             inside bdx_14dof_dataset.pkl, so planner data == tracker data.

Body indices (bdx_14dof.yaml body_names order):
  0 base_link, 5 left_ankle_link, 10 right_ankle_link, 12 neck_pitch_link.

Output (humanoidverse/data/bdx_planner/):
  sparse_full_v1.npz — per-frame arrays concatenated over clips, WORLD frame.
    The yaw-aligned window transform is a training-time concern (P1/P2), not
    baked in here. Contact thresholds mirror the unified bdx values in
    bdx_14dof.yaml (0.3 m/s, 0.04 m) and legged_robot_motions.foot_contact_detect.
  meta.json   — schema layout, thresholds, per-clip records, category stats.
  splits.json — clip-level train/val split, stratified by (version, category).

Feet/head poses and contact are deliberately NOT decoder targets downstream:
they are re-derived from q + base via FK at training time (see planner design
notes); foot_speed is stored for diagnostics only.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot

DATASET_ROOT = Path("/home/tcl/Desktop/start/dataset")
META = Path(__file__).parents[1] / "data" / "bdx_14dof_dataset.meta.json"
OUT_DIR = Path(__file__).parents[1] / "data" / "bdx_planner"

BODY_BASE, BODY_LFOOT, BODY_RFOOT, BODY_HEAD = 0, 5, 10, 12
FEET_IDX = [BODY_LFOOT, BODY_RFOOT]
NECK_DOF_NAMES = ["neck_forward", "neck_pitch", "neck_yaw", "neck_roll"]
FOOT_VEL_THRES = 0.3       # mirrors bdx_14dof.yaml foot_contact_vel_thres
FOOT_HEIGHT_THRES = 0.08  # measured: standing ankle-link origin z ≈ 0.0736 m,
                          # swing peaks ≈ 0.108 m — 0.04 (old yaml) never fired
VAL_FRACTION = 0.1
SEED = 0

SPARSE_LAYOUT = {  # 29D/frame
    "base_pos": [0, 3], "base_quat": [3, 7], "base_linvel": [7, 10],
    "base_angvel": [10, 13], "head_rotvec": [13, 16], "head_angvel": [16, 19],
    "foot_pos": [19, 25], "foot_yaw": [25, 27], "contact": [27, 29],
}
FULL_LAYOUT = {  # 41D/frame
    "q": [0, 14], "qdot": [14, 28], "base_pos": [28, 31], "base_quat": [31, 35],
    "base_linvel": [35, 38], "base_angvel": [38, 41],
}


def quat_wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


def extract_clip(d):
    q = d["joint_pos"].astype(np.float32)
    qdot = d["joint_vel"].astype(np.float32)
    bp = d["body_pos_w"].astype(np.float32)
    bq = d["body_quat_w"].astype(np.float32)
    blv = d["body_lin_vel_w"].astype(np.float32)
    bav = d["body_ang_vel_w"].astype(np.float32)
    T = q.shape[0]

    foot_pos = bp[:, FEET_IDX]                                   # (T,2,3)
    foot_speed = np.linalg.norm(blv[:, FEET_IDX], axis=-1)       # (T,2)
    foot_yaw = sRot.from_quat(quat_wxyz_to_xyzw(bq[:, FEET_IDX]).reshape(-1, 4)) \
        .as_euler("ZYX")[:, 0].reshape(T, 2).astype(np.float32)
    head_rotvec = sRot.from_quat(quat_wxyz_to_xyzw(bq[:, BODY_HEAD])) \
        .as_rotvec().astype(np.float32)
    contact = ((foot_speed < FOOT_VEL_THRES) &
               (foot_pos[..., 2] < FOOT_HEIGHT_THRES))

    # body-order sanity: head above base, feet near the ground
    assert (bp[:, BODY_HEAD, 2] > bp[:, BODY_BASE, 2] - 0.02).all(), "head not above base — body order suspect"
    assert foot_pos[..., 2].min() > -0.05 and foot_pos[..., 2].max() < 0.25, "foot heights out of range"

    return {
        "q": q, "qdot": qdot,
        "base_pos": bp[:, BODY_BASE], "base_quat": bq[:, BODY_BASE],
        "base_linvel": blv[:, BODY_BASE], "base_angvel": bav[:, BODY_BASE],
        "head_rotvec": head_rotvec, "head_angvel": bav[:, BODY_HEAD],
        "foot_pos": foot_pos, "foot_yaw": foot_yaw, "foot_speed": foot_speed,
        "contact": contact,
    }


def make_split(clip_records):
    strata = defaultdict(list)
    for i, c in enumerate(clip_records):
        strata[(c["version"], c["category"])].append(i)
    rng = np.random.default_rng(SEED)
    train, val = [], []
    for key in sorted(strata):
        idxs = sorted(strata[key])
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * VAL_FRACTION)))
        val.extend(idxs[:n_val])
        train.extend(idxs[n_val:])
    return sorted(clip_records[i]["name"] for i in train), \
           sorted(clip_records[i]["name"] for i in val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--full-pool", action="store_true",
                    help="extract ALL clips from every version manifest "
                         "(not just the 1154-clip pkl subset) and take the "
                         "train/val split from the official leak-grouped "
                         "eval_split.json instead of random stratification")
    ap.add_argument("--manifest", default=None,
                    help="explicit planner manifest pair "
                         "(TRAIN_MANIFEST:EVAL_MANIFEST); clips carry "
                         "command_fidelity/derived tags through to training")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        tr = json.loads(Path(args.manifest.split(":")[0]).read_text())
        ev = json.loads(Path(args.manifest.split(":")[1]).read_text())
        def expand(entry):
            version = entry["version"]
            man = json.load(open(DATASET_ROOT / version / "manifest.json"))
            m = next(c for c in man["clips"] if c["id"] == entry["id"])
            return {"name": f"{version}/{entry['id']}", "version": version,
                    "category": m.get("category"),
                    "frames": int(m["n_frames"]),
                    "fidelity": entry.get("command_fidelity"),
                    "derived": entry.get("derived", False),
                    "group": entry.get("group_id")}
        entry_list = [expand(e) for e in tr["clips"]]
        eval_names = {f"{e['version']}/{e['id']}" for e in ev["clips"]}
        # eval clips must ALSO be extracted (they become the val split);
        # expand against the eval manifest's own metadata
        have = {e["name"] for e in entry_list}
        for e in ev["clips"]:
            if f"{e['version']}/{e['id']}" not in have:
                entry_list.append(expand(e))
    elif args.full_pool:
        versions = sorted(p.name for p in DATASET_ROOT.iterdir()
                          if (p / "manifest.json").exists())
        entry_list = []
        for version in versions:
            for c in json.load(open(DATASET_ROOT / version / "manifest.json"))["clips"]:
                entry_list.append({"name": f"{version}/{c['id']}",
                                   "version": version, "category": c.get("category"),
                                   "frames": int(c["n_frames"])})
        eval_set = set(json.load(open(DATASET_ROOT / "eval_split.json"))["eval_clips"])
    else:
        pkl_meta = json.loads(META.read_text())
        entry_list = pkl_meta["clips"]
        eval_set = None

    manifests = {}
    for version in sorted({e["version"] for e in entry_list}):
        manifests[version] = {c["id"]: c for c in
                              json.load(open(DATASET_ROOT / version / "manifest.json"))["clips"]}

    buckets = defaultdict(list)
    clip_records, failed = [], []
    for entry in entry_list:
        version, clip_id = entry["name"].split("/", 1)
        try:
            man = manifests[version][clip_id]
            d = np.load(DATASET_ROOT / version / "clips" / man["file"])
            assert int(d["fps"]) == 50, f"fps {d['fps']}"
            f = extract_clip(d)
        except Exception as e:
            failed.append({"name": entry["name"], "error": str(e)[:120]})
            print(f"✗ {entry['name']}: {e}", flush=True)
            continue
        for k, v in f.items():
            buckets[k].append(v)
        rec = {
            "name": entry["name"], "version": version,
            "category": entry["category"], "frames": entry["frames"],
            "fidelity": entry.get("fidelity"), "derived": entry.get("derived"),
            "contact_frac": np.round(f["contact"].mean(axis=0), 4).tolist(),
            "single_support_frac": float(np.round(
                (f["contact"].sum(axis=1) == 1).mean(), 4)),
            "base_height_mean": float(np.round(f["base_pos"][:, 2].mean(), 4)),
            "neck_range": {n: [float(np.round(f["q"][:, 10 + j].min(), 3)),
                               float(np.round(f["q"][:, 10 + j].max(), 3))]
                           for j, n in enumerate(NECK_DOF_NAMES)},
        }
        clip_records.append(rec)
        print(f"✓ {entry['name']} ({entry['frames']}fr, "
              f"contact {rec['contact_frac']})", flush=True)

    if failed:
        print(f"\n⚠ {len(failed)} clips failed (excluded from output)")

    lengths = np.array([r["frames"] for r in clip_records], dtype=np.int32)
    clip_id = np.repeat(np.arange(len(clip_records), dtype=np.int16), lengths)
    frame_idx = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])
    data = {
        "clip_id": clip_id, "frame_idx": frame_idx, "lengths": lengths,
        "clip_fps": np.full(len(clip_records), 50, dtype=np.int16),
        "contact": np.concatenate(buckets["contact"]).astype(np.uint8),
    }
    for k in ["q", "qdot", "base_pos", "base_quat", "base_linvel", "base_angvel",
              "head_rotvec", "head_angvel", "foot_pos", "foot_yaw", "foot_speed"]:
        data[k] = np.concatenate(buckets[k]).astype(np.float32)

    npz_name = ("sparse_full_v1m.npz" if args.manifest else
                "sparse_full_v2.npz" if args.full_pool else "sparse_full_v1.npz")
    npz_path = out_dir / npz_name
    np.savez_compressed(npz_path, **data)

    if args.manifest:
        train = sorted(r["name"] for r in clip_records)
        val = sorted(n for n in train if n in eval_names)
        train = sorted(set(train) - set(val))
        split_info = {"source": args.manifest}
    elif args.full_pool:
        names = [r["name"] for r in clip_records]
        val = sorted(n for n in names if n in eval_set)
        train = sorted(set(names) - set(val))
        split_info = {"source": "eval_split.json (leak-grouped)"}
    else:
        train, val = make_split(clip_records)
        split_info = {"seed": SEED, "val_fraction": VAL_FRACTION}
    split_info["train"] = train
    split_info["val"] = val
    (out_dir / "splits.json").write_text(json.dumps(split_info, indent=1))

    # ---- category report ----
    cat_stats = defaultdict(lambda: Counter())
    for r in clip_records:
        s = cat_stats[r["category"]]
        s["clips"] += 1
        s["frames"] += r["frames"]
        s["contact_L"] += r["contact_frac"][0] * r["frames"]
        s["contact_R"] += r["contact_frac"][1] * r["frames"]
        s["single_support"] += r["single_support_frac"] * r["frames"]
    report = {}
    for cat, s in sorted(cat_stats.items()):
        report[cat] = {
            "clips": s["clips"], "frames": s["frames"],
            "contact_frac_L": round(s["contact_L"] / s["frames"], 4),
            "contact_frac_R": round(s["contact_R"] / s["frames"], 4),
            "single_support_frac": round(s["single_support"] / s["frames"], 4),
        }
    neck_all = data["q"][:, 10:14]
    neck_report = {
        "overall_range": {n: [float(np.round(neck_all[:, j].min(), 3)),
                              float(np.round(neck_all[:, j].max(), 3))]
                          for j, n in enumerate(NECK_DOF_NAMES)},
        "per_category_motion_std": {},
    }
    for cat in sorted(cat_stats):
        mask = np.isin(data["clip_id"], [i for i, r in enumerate(clip_records)
                                         if r["category"] == cat])
        neck_report["per_category_motion_std"][cat] = {
            n: float(np.round(neck_all[mask, j].std(), 4))
            for j, n in enumerate(NECK_DOF_NAMES)}

    meta = {
        "schema_version": "bdx_planner_v1",
        "source": {"dataset_root": str(DATASET_ROOT), "clip_list": str(META),
                   "clips": len(clip_records),
                   "total_frames": int(lengths.sum()),
                   "failed": failed},
        "body_indices": {"base_link": BODY_BASE, "left_ankle_link": BODY_LFOOT,
                         "right_ankle_link": BODY_RFOOT, "neck_pitch_link": BODY_HEAD},
        "contact_thresholds": {"vel": FOOT_VEL_THRES, "height": FOOT_HEIGHT_THRES,
                               "note": "mirrors bdx_14dof.yaml + foot_contact_detect fix"},
        "frame": "world; yaw-aligned window transform happens at training time",
        "sparse_layout": SPARSE_LAYOUT, "full_layout": FULL_LAYOUT,
        "category_report": report, "neck_report": neck_report,
        "clips": clip_records,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))

    print(f"\n{'='*64}")
    print(f"✓ {npz_path.name}: {len(clip_records)} clips, "
          f"{int(lengths.sum())} frames ({lengths.sum()/50/60:.1f} min)")
    print(f"  splits: {len(train)} train / {len(val)} val clips (stratified)")
    print("\ncategory report:")
    for cat, s in report.items():
        print(f"  {cat:<12} clips {s['clips']:>4}  frames {s['frames']:>7}  "
              f"contact L/R {s['contact_frac_L']:.2f}/{s['contact_frac_R']:.2f}  "
              f"single-support {s['single_support_frac']:.2f}")
    print("\nneck joint overall ranges (rad):")
    for n, (lo, hi) in neck_report["overall_range"].items():
        print(f"  {n:<13} [{lo:+.3f}, {hi:+.3f}]")
    print("neck per-category motion std:")
    for cat, d_ in neck_report["per_category_motion_std"].items():
        print(f"  {cat:<12} " + "  ".join(f"{n}={d_[n]:.3f}" for n in NECK_DOF_NAMES))


if __name__ == "__main__":
    main()
