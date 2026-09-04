"""Full read-only dry-run of the 13,619 canonical final-build2 NPZ clips for
tracker consumption compatibility (ruling 2026-08-24 night 12).

Per clip (train+eval manifests, never writes to the dataset):
  schema: required keys, dtypes, shapes, body/joint name coverage, fps
  FK:     convert-path forward kinematics on sampled frames (5 mm tolerance)
  length: frame count vs tracker constraints (seq_length 250 -> >=251;
          probe horizon 8+150+40=198 -> >=199 tier)
  lineage: version, group_id, derived, tags from canonical manifest
  activity proxy: mean |joint velocity| (finite diff on joint_pos, dof order)
Output: results/bfmzero-bdx-full/tracker_dryrun_eligibility.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as sRot

DATASET_ROOT = Path("/home/tcl/Desktop/start/dataset")
OUT = Path("results/bfmzero-bdx-full/tracker_dryrun_eligibility.json")
REQUIRED_BODIES = 17
MOTION_CFG = EasyDict({
    "asset": {"assetRoot": str(Path(__file__).parents[1] / "data" / "robots" / "bdx") + "/",
              "assetFileName": "bdx_v4.xml", "urdfFileName": "bdx_v4.urdf"},
    "extend_config": [],
})
_W = {}


def wxyz_to_scipy(q):
    q = np.asarray(q).reshape(-1, 4)
    return sRot.from_quat(np.concatenate([q[:, 1:4], q[:, :1]], axis=-1))


def init_worker():
    from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
    import yaml
    skel = Humanoid_Batch(MOTION_CFG, torch.device("cpu"))
    rcfg = yaml.safe_load(open(Path(__file__).parents[1] / "config/robot/bdx/bdx_14dof.yaml"))["robot"]
    skel.robot_joint_names = rcfg["dof_names"]
    _W["skel"] = skel
    _W["parents"] = skel._parents.numpy()
    _W["q_fixed"] = skel._local_rotation[0].numpy()


def audit_clip(rec):
    f = DATASET_ROOT / rec["file"]
    try:
        npz = np.load(f, allow_pickle=True)
        keys = set(npz.keys())
        missing = [k for k in ("body_pos_w", "body_quat_w", "joint_pos", "body_names", "joint_names", "fps") if k not in keys]
        if missing:
            return {**rec, "ok": False, "error": f"missing keys {missing}"}
        bn = [str(x) for x in npz["body_names"]]
        jn = [str(x) for x in npz["joint_names"]]
        skel = _W["skel"]
        try:
            order = [bn.index(n) for n in skel.body_names]
        except ValueError as e:
            return {**rec, "ok": False, "error": f"body name missing: {e}", "n_bodies_file": len(bn)}
        jorder = [jn.index(n) for n in skel.robot_joint_names] if all(n in jn for n in skel.robot_joint_names) else None

        body_pos = npz["body_pos_w"][:, order].astype(np.float32)
        body_quat = npz["body_quat_w"][:, order].astype(np.float32)
        joint_pos = npz["joint_pos"].astype(np.float32)
        if jorder is not None:
            joint_pos = joint_pos[:, jorder]
        fps = int(npz["fps"])
        T = body_pos.shape[0]
        if body_pos.shape[1] != REQUIRED_BODIES or body_quat.shape[:2] != (T, REQUIRED_BODIES):
            return {**rec, "ok": False, "error": f"shape {body_pos.shape} {body_quat.shape}"}
        if joint_pos.shape != (T, 14):
            return {**rec, "ok": False, "error": f"joint_pos shape {joint_pos.shape}"}
        if not np.isfinite(body_pos).all() or not np.isfinite(body_quat).all() or not np.isfinite(joint_pos).all():
            return {**rec, "ok": False, "error": "non-finite values"}

        # convert-path FK verification on sampled frames
        root_pos = body_pos[:, 0]
        root_rot = body_quat[:, 0]
        pose_aa = np.zeros((T, REQUIRED_BODIES, 3), dtype=np.float32)
        pose_aa[:, 0] = wxyz_to_scipy(root_rot).as_rotvec()
        R_world = wxyz_to_scipy(body_quat).as_matrix().reshape(T, REQUIRED_BODIES, 3, 3)
        R_fixed = wxyz_to_scipy(_W["q_fixed"]).as_matrix()
        R_parent = R_world[:, np.clip(_W["parents"], 0, None)]
        R_joint = np.linalg.inv(R_parent @ R_fixed[None]) @ R_world
        pose_aa[:, 1:] = sRot.from_matrix(R_joint.reshape(T * REQUIRED_BODIES, 3, 3)).as_rotvec().reshape(T, REQUIRED_BODIES, 3)[:, 1:]
        idx = np.linspace(0, T - 1, min(8, T)).astype(int)
        fk = skel.fk_batch(torch.from_numpy(pose_aa[idx][None]), torch.from_numpy(root_pos[idx][None]), return_full=False)
        err_mm = float(np.abs(fk.global_translation[0].numpy() - body_pos[idx]).max() * 1000)
        if err_mm >= 5.0:
            return {**rec, "ok": False, "error": f"FK error {err_mm:.2f}mm"}

        # activity proxy: mean |joint velocity| in the tracker dof order
        jvel = np.abs(np.diff(joint_pos, axis=0)).mean() * fps
        return {
            **{k: rec[k] for k in ("key", "version", "group_id", "derived", "duration_s", "tags", "split")},
            "ok": True, "frames": int(T), "fps": fps, "fk_err_mm_max": round(err_mm, 3),
            "activity_proxy_joint_speed": round(float(jvel), 4),
            "joint_order_match": jorder is not None,
        }
    except Exception as e:
        return {**rec, "ok": False, "error": f"{type(e).__name__}: {str(e)[:140]}"}


def main():
    train = json.load(open(DATASET_ROOT / "canonical_train_manifest.json"))["clips"]
    evalc = json.load(open(DATASET_ROOT / "canonical_eval_manifest.json"))["clips"]
    records = []
    for c in train:
        records.append({"key": f"{c['version']}/{c['id']}", "file": c["file"], "version": c["version"],
                        "group_id": c.get("group_id"), "derived": c.get("derived"),
                        "duration_s": c.get("duration_s"), "tags": c.get("tags", []), "split": "train"})
    for c in evalc:
        records.append({"key": f"{c['version']}/{c['id']}", "file": c["file"], "version": c["version"],
                        "group_id": c.get("group_id"), "derived": c.get("derived"),
                        "duration_s": c.get("duration_s"), "tags": c.get("tags", []), "split": "eval"})
    assert len(records) == 13619, len(records)

    with mp.Pool(16, initializer=init_worker) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(audit_clip, records, chunksize=32)):
            out.append(r)
            if (i + 1) % 1000 == 0:
                print(f"{i+1}/{len(records)} ok={sum(x['ok'] for x in out)}", flush=True)

    ok = [r for r in out if r["ok"]]
    bad = [r for r in out if not r["ok"]]
    def frac(pred): return sum(1 for r in ok if pred(r))
    summary = {
        "total": len(out), "ok": len(ok), "failed": len(bad),
        "eligible_seq250_251f": frac(lambda r: r["frames"] >= 251),
        "eligible_probe_199f": frac(lambda r: r["frames"] >= 199),
        "short_only": frac(lambda r: r["frames"] < 199),
        "by_version": {},
        "activity_terciles_global": None,
    }
    for v in sorted({r["version"] for r in ok}):
        rs = [r for r in ok if r["version"] == v]
        summary["by_version"][v] = {
            "ok": len(rs),
            "ge251": sum(r["frames"] >= 251 for r in rs),
            "frames_p50": float(np.median([r["frames"] for r in rs])),
            "activity_p50": float(np.median([r["activity_proxy_joint_speed"] for r in rs])),
            "groups": len({r["group_id"] for r in rs}),
        }
    acts = np.array([r["activity_proxy_joint_speed"] for r in ok])
    summary["activity_terciles_global"] = [float(x) for x in np.quantile(acts, (1/3, 2/3))]
    result = {
        "_provenance": {
            "created_at": datetime.now().isoformat(),
            "probe": "audit_tracker_dataset_dryrun.py (read-only)",
            "dataset": "canonical final-build2 (RELEASE.json, 13619 clips)",
            "checks": "keys/shapes/finiteness/fps/body+joint name coverage/FK<=5mm@8frames/activity proxy",
            "eligibility_tiers": {"tracker_seq_250": ">=251 frames", "probe_horizon": ">=199 frames"},
        },
        "summary": summary,
        "failures": bad,
        "clips": sorted(out, key=lambda r: r["key"]),
    }
    assert not OUT.exists()
    OUT.write_text(json.dumps(result) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
