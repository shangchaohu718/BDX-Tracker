"""Convert the BDX dataset (7 versions, self-describing 6-key npz clips) into
humanoidverse motion pkls, split by the held-out eval set.

Outputs:
  bdx_14dof_train.pkl  — eval_only=false clips, stratified caps per version
  bdx_14dof_eval.pkl   — the eval_split.json held-out clips (honest generalization set)

Uses each npz's own body_names for order mapping (self-describing since the
08-18 dataset rebuild), so mirror/backward clips convert identically to normal ones.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation as sRot

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

DATASET_ROOT = Path("/home/tcl/Desktop/start/dataset")

# per-version caps for the TRAIN pkl (stratified; mirrors are now usable, v2_backward new)
VERSION_CAPS = {
    "v0": 247,
    "v1_walk": 320,
    "v1_walkhead": 110,
    "v1_stand": 90,
    "v1_perturb": 420,
    "v1_augment": 150,   # mirror/speed variants (mirror convention fixed upstream)
    "v2_backward": 150,  # time-reversal backward walking
}

MOTION_CFG = EasyDict({
    "asset": {
        "assetRoot": str(Path(__file__).parents[1] / "data" / "robots" / "bdx") + "/",
        "assetFileName": "bdx_v4.xml",
        "urdfFileName": "bdx_v4.urdf",
    },
    "extend_config": [],
})


def wxyz_to_scipy(q):
    q = np.asarray(q).reshape(-1, 4)
    return sRot.from_quat(np.concatenate([q[:, 1:4], q[:, :1]], axis=-1))


def convert_clip(npz, skel, parents, q_fixed, verify_frames):
    # order mapping from the clip's own body_names (self-describing)
    bn = [str(x) for x in npz["body_names"]]
    order = [bn.index(n) for n in skel.body_names]
    jn = [str(x) for x in npz["joint_names"]]
    jorder = [jn.index(n) for n in skel.robot_joint_names] if hasattr(skel, "robot_joint_names") else None

    root_pos = npz["body_pos_w"][:, order[0]].astype(np.float32)
    root_rot = npz["body_quat_w"][:, order[0]].astype(np.float32)
    body_rot = npz["body_quat_w"][:, order].astype(np.float32)
    body_pos = npz["body_pos_w"][:, order].astype(np.float32)
    joint_pos = npz["joint_pos"].astype(np.float32)
    if jorder is not None:
        joint_pos = joint_pos[:, jorder]
    fps = int(npz["fps"])
    T, nb = body_rot.shape[:2]
    assert nb == len(skel.body_names)

    pose_aa = np.zeros((T, nb, 3), dtype=np.float32)
    pose_aa[:, 0] = wxyz_to_scipy(root_rot).as_rotvec()
    R_world = wxyz_to_scipy(body_rot).as_matrix().reshape(T, nb, 3, 3)
    R_fixed = wxyz_to_scipy(q_fixed).as_matrix()
    R_parent = R_world[:, np.clip(parents, 0, None)]
    R_joint = np.linalg.inv(R_parent @ R_fixed[None]) @ R_world
    pose_aa[:, 1:] = sRot.from_matrix(R_joint.reshape(T * nb, 3, 3)).as_rotvec().reshape(T, nb, 3)[:, 1:]

    idx = np.linspace(0, T - 1, min(verify_frames, T)).astype(int)
    fk = skel.fk_batch(torch.from_numpy(pose_aa[idx][None]), torch.from_numpy(root_pos[idx][None]), return_full=False)
    err_mm = np.abs(fk.global_translation[0].numpy() - body_pos[idx]).max() * 1000
    assert err_mm < 5.0, f"FK error {err_mm:.2f}mm"

    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": joint_pos,
        "root_rot": root_rot,
        "fps": fps,
    }, err_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(DATASET_ROOT))
    ap.add_argument("--out-train", default=str(Path(__file__).parents[1] / "data" / "bdx_14dof_train.pkl"))
    ap.add_argument("--out-eval", default=str(Path(__file__).parents[1] / "data" / "bdx_14dof_eval.pkl"))
    ap.add_argument("--verify-frames", type=int, default=30)
    args = ap.parse_args()

    skel = Humanoid_Batch(MOTION_CFG, torch.device("cpu"))
    # expose joint order for mapping (dof_names order used by the robot config)
    import yaml
    rcfg = yaml.safe_load(open(Path(__file__).parents[1] / "config/robot/bdx/bdx_14dof.yaml"))["robot"]
    skel.robot_joint_names = rcfg["dof_names"]
    parents = skel._parents.numpy()
    q_fixed = skel._local_rotation[0].numpy()

    eval_set = set(json.load(open(Path(args.dataset_root) / "eval_split.json"))["eval_clips"])

    train_out, eval_out, failed = {}, {}, []
    for version, cap in VERSION_CAPS.items():
        man = json.load(open(Path(args.dataset_root) / version / "manifest.json"))
        clips = sorted(man["clips"], key=lambda c: c["id"])
        train_pool = [c for c in clips if f"{version}/{c['id']}" not in eval_set]
        sel = np.linspace(0, len(train_pool) - 1, min(cap, len(train_pool))).astype(int) if len(train_pool) > cap else list(range(len(train_pool)))
        for j in sel:
            c = train_pool[j]
            name = f"{version}/{c['id']}"
            f = Path(args.dataset_root) / version / "clips" / c["file"]
            try:
                entry, err = convert_clip(dict(np.load(f, allow_pickle=True)), skel, parents, q_fixed, args.verify_frames)
                train_out[name] = entry
            except Exception as e:
                failed.append({"name": name, "error": str(e)[:120]})
                print(f"✗ {name}: {e}", flush=True)

    for name in sorted(eval_set):
        version, cid = name.split("/", 1)
        f = Path(args.dataset_root) / version / "clips" / f"{cid}.npz"
        try:
            entry, err = convert_clip(dict(np.load(f, allow_pickle=True)), skel, parents, q_fixed, args.verify_frames)
            eval_out[name] = entry
        except Exception as e:
            failed.append({"name": name, "error": str(e)[:120]})

    joblib.dump(train_out, args.out_train, compress=3)
    joblib.dump(eval_out, args.out_eval, compress=3)
    tr = sum(v["root_trans_offset"].shape[0] for v in train_out.values())
    ev = sum(v["root_trans_offset"].shape[0] for v in eval_out.values())
    print(f"\n✓ train: {args.out_train} — {len(train_out)} clips / {tr} frames ({tr/50/3600:.2f}h)")
    print(f"✓ eval : {args.out_eval} — {len(eval_out)} clips / {ev} frames ({ev/50/3600:.2f}h)")
    print(f"failed: {len(failed)}")


if __name__ == "__main__":
    main()
