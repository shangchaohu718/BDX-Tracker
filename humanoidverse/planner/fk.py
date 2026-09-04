"""Differentiable FK for the BDX kinematic tree (freejoint base + 14 hinges).

Parsed from bdx_v4.xml once (numpy constants), executed in torch so FK losses
backprop into predicted q / base pose. Semantics follow MuJoCo: each body's
frame in its parent = translate(body_pos) ∘ rotate(body_quat) ∘
[T(anchor) ∘ R(axis, q - jnt_ref) ∘ T(-anchor)] for its hinge, if any.

Verified against source-npz body_pos_w (see verify(), called at import time
by tests / at training start): max error must stay < 5 mm.
"""

from pathlib import Path

import mujoco
import numpy as np
import torch

XML = Path(__file__).parents[1] / "data" / "robots" / "bdx" / "bdx_v4.xml"

BODY_NAMES = [
    "base_link",
    "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
    "left_knee_link", "left_ankle_link",
    "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
    "right_knee_link", "right_ankle_link",
    "neck_forward_link", "neck_pitch_link", "neck_yaw_link", "neck_roll_link",
    "left_ear_link", "right_ear_link",
]
DOF_NAMES = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_joint",
    "neck_forward_joint", "neck_pitch_joint", "neck_yaw_joint", "neck_roll_joint",
]
def rotmat_to_rotvec(R):
    """(...,3,3) -> (...,3) differentiable log map."""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    ang = torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))
    axis = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                        R[..., 0, 2] - R[..., 2, 0],
                        R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    axis = axis / (2 * torch.sin(ang)[..., None]).clamp_min(1e-8)
    return axis * ang[..., None]


LFOOT, RFOOT, HEAD = 5, 10, 12


def _quat_wxyz_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)


class BDXFK:
    def __init__(self, xml_path=XML):
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        mj_id = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i
                 for i in range(m.nbody)}
        jnt_id = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j
                  for j in range(m.njnt)}
        assert [n for n in BODY_NAMES] == [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in sorted(mj_id[n] for n in BODY_NAMES)], "body set mismatch vs xml"

        self.parent = np.zeros(17, dtype=np.int64)
        self.body_pos = np.zeros((17, 3), dtype=np.float64)
        self.Rfix = np.zeros((17, 3, 3), dtype=np.float64)
        self.axis = np.zeros((17, 3), dtype=np.float64)
        self.anchor = np.zeros((17, 3), dtype=np.float64)
        self.qadr = np.full(17, -1, dtype=np.int64)   # index into dataset q(14)
        self.ref = np.zeros(17, dtype=np.float64)

        for i, name in enumerate(BODY_NAMES):
            b = mj_id[name]
            self.parent[i] = 0 if i == 0 else BODY_NAMES.index(
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]))
            self.body_pos[i] = m.body_pos[b]
            self.Rfix[i] = _quat_wxyz_to_mat(m.body_quat[b])
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    self.axis[i] = m.jnt_axis[j]
                    self.anchor[i] = m.jnt_pos[j]
                    self.ref[i] = m.qpos0[m.jnt_qposadr[j]]  # jnt_ref lands in qpos0
                    # map to the dataset dof column by joint name
                    self.qadr[i] = DOF_NAMES.index(
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))

        self.parent_t = torch.from_numpy(self.parent)
        self.body_pos_t = torch.from_numpy(self.body_pos).float()
        self.Rfix_t = torch.from_numpy(self.Rfix).float()
        self.axis_t = torch.from_numpy(self.axis).float()
        self.anchor_t = torch.from_numpy(self.anchor).float()
        self.qadr_t = torch.from_numpy(self.qadr)
        self.ref_t = torch.from_numpy(self.ref).float()

    def to(self, device):
        for k in ["parent_t", "body_pos_t", "Rfix_t", "axis_t",
                  "anchor_t", "qadr_t", "ref_t"]:
            setattr(self, k, getattr(self, k).to(device))
        return self

    def _rodrigues(self, axis, theta):
        # axis (3,) fixed, theta (...,) -> (...,3,3)
        K = torch.zeros(3, 3, device=theta.device, dtype=theta.dtype)
        K[0, 1], K[0, 2] = -axis[2], axis[1]
        K[1, 0], K[1, 2] = axis[2], -axis[0]
        K[2, 0], K[2, 1] = -axis[1], axis[0]
        I = torch.eye(3, device=theta.device, dtype=theta.dtype)
        th = theta.unsqueeze(-1).unsqueeze(-1)
        return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

    def forward(self, base_pos, base_R, q, return_rot=False):
        """base_pos (...,3), base_R (...,3,3), q (...,14) -> body_pos (...,17,3).
        With return_rot=True also returns body rotation matrices (...,17,3,3)."""
        Rs, ps = [None] * 17, [None] * 17
        Rs[0], ps[0] = base_R, base_pos
        for i in range(1, 17):
            p = int(self.parent_t[i].item())
            Rp, pp = Rs[p], ps[p]
            if self.qadr_t[i] >= 0:
                theta = q[..., self.qadr_t[i]] - self.ref_t[i]
                Rh = self._rodrigues(self.axis_t[i], theta.type(Rp.dtype))
                one_minus_Rh_a = self.anchor_t[i] - Rh @ self.anchor_t[i]
                local = self.body_pos_t[i] + one_minus_Rh_a @ self.Rfix_t[i].T
                Rs[i] = Rp @ (self.Rfix_t[i] @ Rh)
            else:
                local = self.body_pos_t[i]
                Rs[i] = Rp @ self.Rfix_t[i]
            ps[i] = pp + (Rp @ local.unsqueeze(-1)).squeeze(-1)
        pos = torch.stack(ps, dim=-2)
        if return_rot:
            return pos, torch.stack(Rs, dim=-3)
        return pos

    def verify(self, npz_path, frames=40):
        """FK from (base pose, joint_pos) vs stored body_pos_w — max err in mm."""
        d = np.load(npz_path)
        idx = np.linspace(0, d["joint_pos"].shape[0] - 1, frames).astype(int)
        base_R = torch.from_numpy(_quat_wxyz_to_mat(d["body_quat_w"][idx, 0]))
        pos = self.forward(torch.from_numpy(d["body_pos_w"][idx, 0]).float(),
                           base_R.float(), torch.from_numpy(d["joint_pos"][idx]).float())
        return float((pos.numpy() - d["body_pos_w"][idx]).max() * 1000)


if __name__ == "__main__":
    import json
    fk = BDXFK()
    root = Path("/home/tcl/Desktop/start/dataset")
    errs = []
    for ver, needle in [("v1_stand", "stand"), ("v1_walk", "walk"), ("v0", "angry")]:
        man = json.load(open(root / ver / "manifest.json"))
        c = next(c for c in man["clips"] if needle in c["id"])
        errs.append((f"{ver}/{c['id']}", fk.verify(root / ver / "clips" / c["file"])))
    for name, e in errs:
        print(f"  {name:<28} max FK err {e:.3f} mm")
    assert max(e for _, e in errs) < 5.0, "FK convention mismatch"
    print("FK verified < 5 mm on all probes")
