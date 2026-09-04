"""Window dataset for the BDX motion planner (P1/P2).

Reads humanoidverse/data/bdx_planner/sparse_full_v1.npz (world frame, see
scripts/extract_sparse_bdx.py) and yields canonicalized motion windows:

  window frame: origin = base position at window start, global yaw removed
                (Rz(-yaw) applied to positions/velocities/rotations)
  model representation per frame (43D, "full" 41D with quat→6D):
    q(14) qdot(14) base_pos(3) base_6d(6) base_linvel(3) base_angvel(3)

Normalization: per-dim mean/std over TRAIN-split windows only (sampled), stored
to stats.json and reused for val — val statistics must never touch preprocessing.
Dataset returns normalized model input + unnormalized physical GT (losses run in
physical units); FK targets are computed from GT q/base by planner.fk (verified
to 0.001 mm against the source data), so no FK fields are stored.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import os
DATA_DIR = Path(os.environ.get(
    "BDX_PLANNER_DATA", Path(__file__).parents[1] / "data" / "bdx_planner"))
WINDOW = 50
MODEL_DIM = 43  # 14+14+3+6+3+3


def quat_wxyz_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def mat_to_6d(R):
    """First two COLUMNS as [c1(3), c2(3)] — matches sixd_to_mat's layout.
    (Was row-major interleaved [r00,r01,r10,r11,r20,r21] before 2026-08-21,
    silently incompatible with sixd_to_mat; round-trip test added in tests/)"""
    return R.transpose(-1, -2)[..., :2, :].reshape(*R.shape[:-2], 6)


def sixd_to_mat(a):
    """Gram-Schmidt orthonormalization (Zhou et al. 6D rep), differentiable."""
    a1, a2 = a[..., 0:3], a[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def quat_yaw(q_wxyz):
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MotionData:
    """Concatenated frames + clip boundaries + split index."""

    def __init__(self, npz_path=None,
                 meta_path=None,
                 splits_path=None):
        npz_path = npz_path or next(
            (DATA_DIR / n for n in ("sparse_full_v1m.npz", "sparse_full_v2.npz",
                                    "sparse_full_v1.npz")
             if (DATA_DIR / n).exists()), DATA_DIR / "sparse_full_v1.npz")
        meta_path = meta_path or DATA_DIR / "meta.json"
        splits_path = splits_path or DATA_DIR / "splits.json"
        d = np.load(npz_path)
        self.d = {k: d[k] for k in d.files}
        hp = DATA_DIR / "head_pos.npy"   # v2: FK-derived head positions (T,3)
        if hp.exists():
            self.d["head_pos"] = np.load(hp)
        # intervention command-label fix (v1.1): manifest-requested commands
        self.cmd_override = self.d.get("cmd_override")
        self.cmd_source = self.d.get("cmd_source")
        meta = json.loads(Path(meta_path).read_text())
        self.clip_names = [c["name"] for c in meta["clips"]]
        self.clip_cat = [c["category"] for c in meta["clips"]]
        self.name2clip = {n: i for i, n in enumerate(self.clip_names)}
        splits = json.loads(Path(splits_path).read_text())
        self.split_names = splits
        lengths = self.d["lengths"]
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])

    def frames(self, clip, start, n):
        o = self.offsets[clip]
        keys = ("q", "qdot", "base_pos", "base_quat", "base_linvel",
                "base_angvel", "contact", "head_rotvec", "head_angvel",
                "foot_pos", "foot_yaw", "head_pos")
        return {k: v[o + start:o + start + n] for k, v in self.d.items()
                if k in keys}


class WindowDataset(Dataset):
    def __init__(self, md: MotionData, split="train", window=WINDOW,
                 stride=1, max_windows=None, stats=None, seed=0):
        self.md, self.window, self.stats = md, window, stats
        self.train = split == "train"
        names = md.split_names[split]
        rng = np.random.default_rng(seed)
        self.windows = []  # (clip_idx, start)
        for name in names:
            c = md.name2clip[name]
            T = int(md.d["lengths"][c])
            if T < window:
                continue
            starts = np.arange(0, T - window + 1, stride if not self.train else 1)
            for s in starts:
                self.windows.append((c, int(s)))
        if max_windows and len(self.windows) > max_windows:
            sel = rng.choice(len(self.windows), max_windows, replace=False)
            self.windows = [self.windows[i] for i in sorted(sel)]
        self.categories = np.array(
            [md.clip_cat[c] for c, _ in self.windows])

    def canonical(self, f):
        q = torch.from_numpy(f["q"].astype(np.float32))
        qdot = torch.from_numpy(f["qdot"].astype(np.float32))
        bp = torch.from_numpy(f["base_pos"].astype(np.float32))
        bq = torch.from_numpy(f["base_quat"].astype(np.float32))
        blv = torch.from_numpy(f["base_linvel"].astype(np.float32))
        bav = torch.from_numpy(f["base_angvel"].astype(np.float32))
        psi = quat_yaw(bq[0])
        c, s = torch.cos(psi), torch.sin(psi)
        Rz = torch.tensor([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])

        def rot(v):
            return v @ Rz.T

        bp_c = rot(bp - bp[0])
        R_c = Rz.unsqueeze(0) @ quat_wxyz_to_mat(bq)
        x = torch.cat([q, qdot, bp_c, mat_to_6d(R_c), rot(blv), rot(bav)], dim=-1)
        gt = {
            "q": q, "qdot": qdot, "base_pos": bp_c, "base_R": R_c,
            "base_linvel": rot(blv), "base_angvel": rot(bav),
            "contact": torch.from_numpy(f["contact"].astype(np.float32)),
        }
        return x, gt

    def __getitem__(self, i):
        c, s = self.windows[i]
        x, gt = self.canonical(self.md.frames(c, s, self.window))
        if self.stats is not None:
            x = (x - self.stats["mean"]) / self.stats["std"]
        return {"x": x, **gt,
                "clip": c, "start": s}

    def __len__(self):
        return len(self.windows)


def compute_stats(md: MotionData, n=20000, window=WINDOW, seed=0):
    """Per-dim mean/std of the canonical representation, TRAIN windows only."""
    ds = WindowDataset(md, "train", window=window, stride=25, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    xs = torch.stack([ds[int(i)]["x"] for i in idx])
    return {"mean": xs.mean(0), "std": xs.std(0).clamp_min(1e-6)}


SPARSE_DIM = 32  # base 13 + head 6 + feet 8 + contact 2 + head_pos 3 (v2:
                 # orientation alone hides neck_yaw/forward — head POSITION
                 # carries them, corr(height, neck_forward)=0.96)


PREPROCESS_VERSIONS = ("legacy_v0", "fixed_v1")
# FAIL-CLOSED: no silent default. Any consumer of canonical_sparse must either
# call set_preprocess_version() (registry does this) or pass it explicitly —
# otherwise the audit-2026-08-24 training/serving skew can recur silently.
_active_preprocess = None


def set_preprocess_version(v):
    global _active_preprocess
    assert v in PREPROCESS_VERSIONS
    _active_preprocess = v


def canonical_sparse(md_frames, psi):
    """(n,32) sparse window in the yaw-aligned frame of yaw `psi` (rad).

    head_rotvec handling is versioned:
      legacy_v0 — rotvec treated as a vector (R(-psi) applied to the 3
                  numbers); leaves absolute world yaw in the channel
                  (z std 1.53 on training data). Matches what the current
                  canonical checkpoint was TRAINED with.
      fixed_v1  — R(-psi) @ R_head then re-log (correct orientation math).
    """
    import numpy as _np
    if _active_preprocess is None:
        raise RuntimeError(
            "Preprocess version not configured. Load the canonical registry "
            "(planner.registry.resolve_canonical) or call "
            "set_preprocess_version('legacy_v0'|'fixed_v1') explicitly.")
    c, s = _np.cos(psi), _np.sin(psi)
    Rz = _np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]], dtype=_np.float32)

    def rot(v):
        return v @ Rz.T

    bp = md_frames["base_pos"].astype(_np.float32)
    bq = md_frames["base_quat"].astype(_np.float32)
    w = bq[:, 0]
    bq = bq * np.where(w < 0, -1.0, 1.0).astype(_np.float32)[:, None]  # w>=0
    # de-yaw the base quaternion (was raw world quat before 2026-08-21 —
    # leaked absolute heading into an otherwise yaw-canonical token set)
    from scipy.spatial.transform import Rotation as _sR
    _R_w = _sR.from_quat(np.roll(bq, -1, axis=-1)).as_matrix()
    bq = np.roll(_sR.from_matrix(Rz[None] @ _R_w).as_quat(),
                 1, axis=-1).astype(_np.float32)
    n = len(bp)
    fp = rot(md_frames["foot_pos"].astype(_np.float32).reshape(n * 2, 3)).reshape(n, 2, 3)
    yaw_c = _np.mod(md_frames["foot_yaw"].astype(_np.float32) - psi
                    + _np.pi, 2 * _np.pi) - _np.pi
    out = _np.concatenate([
        rot(bp - bp[0]), bq,
        rot(md_frames["base_linvel"].astype(_np.float32)),
        rot(md_frames["base_angvel"].astype(_np.float32)),
        (_sR.from_matrix(Rz[None] @ _sR.from_rotvec(
            md_frames["head_rotvec"].astype(_np.float32)).as_matrix()
        ).as_rotvec().astype(_np.float32)
         if _active_preprocess == "fixed_v1" else
         rot(md_frames["head_rotvec"].astype(_np.float32))),
        rot(md_frames["head_angvel"].astype(_np.float32)),
        fp[:, 0], fp[:, 1],
        yaw_c[:, :1], yaw_c[:, 1:2],
        md_frames["contact"].astype(_np.float32),
        rot(md_frames["head_pos"].astype(_np.float32) - bp[0]),  # v2 (29:32)
    ], axis=-1)
    assert out.shape[-1] == SPARSE_DIM
    return torch.from_numpy(out.astype(_np.float32))


class P2Dataset(WindowDataset):
    """Sparse future + sparse history -> predict the teacher latent / full motion.

    History is K frames before the future window start; both are expressed in
    the yaw-aligned frame anchored at the future window start."""

    def __init__(self, md, split="train", window=WINDOW, history=50,
                 stride=1, max_windows=None, stats=None, sparse_stats=None,
                 seed=0):
        self.history = history
        self.sparse_stats = sparse_stats
        names = md.split_names[split]
        rng = np.random.default_rng(seed)
        self.windows = []
        for name in names:
            ci = md.name2clip[name]
            T = int(md.d["lengths"][ci])
            if T < window + history:
                continue
            for s0 in range(history, T - window + 1,
                            25 if split != "train" else 1):
                self.windows.append((ci, s0))
        if max_windows and len(self.windows) > max_windows:
            sel = rng.choice(len(self.windows), max_windows, replace=False)
            self.windows = [self.windows[i] for i in sorted(sel)]
        self.categories = np.array([md.clip_cat[c] for c, _ in self.windows])
        self.md, self.window, self.stats = md, window, stats

    def __getitem__(self, i):
        c, s = self.windows[i]
        f_fut = self.md.frames(c, s, self.window)
        f_hist = self.md.frames(c, s - self.history, self.history)
        bq0 = f_fut["base_quat"][0].astype(np.float32)
        psi = float(np.arctan2(2 * (bq0[0] * bq0[3] + bq0[1] * bq0[2]),
                               1 - 2 * (bq0[2] ** 2 + bq0[3] ** 2)))
        sp_fut = canonical_sparse(f_fut, psi)
        sp_hist = canonical_sparse(f_hist, psi)
        if self.sparse_stats is not None:
            sp_fut = (sp_fut - self.sparse_stats["mean"]) / self.sparse_stats["std"]
            sp_hist = (sp_hist - self.sparse_stats["mean"]) / self.sparse_stats["std"]
        item = super().__getitem__(i)
        item["sp_fut"], item["sp_hist"] = sp_fut, sp_hist
        # explicit joint-space neck command (P2.8): observation-derived head
        # fields cannot carry neck_yaw (corr -0.005) — command it directly
        # v1.1 command-source routing (audit 2026-08-24: intervention labels
        # existed in manifests but were never consumed — planner trained only
        # on GT-derived window-mean commands, constant commands were OOD)
        src = int(self.md.cmd_source[c]) if (
            self.md.cmd_source is not None and c < len(self.md.cmd_source)) else 0
        if src == 1:
            cv = self.md.cmd_override[c]
            neck_const = torch.from_numpy(
                np.asarray(cv[0:4], np.float32)) / 1.7
            item["cmd"] = neck_const[None, :].expand(self.window, 4).contiguous()
            item["loco"] = torch.tensor([cv[4] / 1.0, cv[5] / 0.5, cv[6] / 1.5],
                                        dtype=torch.float32)
            item["command_source"] = 1   # manifest_intervention
        else:
            item["cmd"] = torch.from_numpy(
                f_fut["q"][:, 10:14].astype(np.float32)) / 1.7
            lv = torch.from_numpy(f_fut["base_linvel"].astype(np.float32))
            c2, sn2 = float(np.cos(psi)), float(np.sin(psi))
            Rzf = torch.tensor([[c2, sn2, 0.], [-sn2, c2, 0.], [0., 0., 1.]],
                               dtype=torch.float32)
            lv_c = lv @ Rzf.T
            av = torch.from_numpy(f_fut["base_angvel"].astype(np.float32)) @ Rzf.T
            item["loco"] = torch.stack([lv_c[:, 0].mean() / 1.0,
                                        lv_c[:, 1].mean() / 0.5,
                                        av[:, 2].mean() / 1.5])
            item["command_source"] = 0   # gt_derived
        return item


def compute_sparse_stats(md, n=20000, window=WINDOW, history=50, seed=0):
    ds = P2Dataset(md, "train", window=window, history=history, stride=25,
                   seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    xs = torch.stack([ds[int(i)]["sp_fut"] for i in idx])
    stats = {"mean": xs.mean(0), "std": xs.std(0).clamp_min(1e-6)}
    stats["mean"][:, 27:29] = 0.0       # contact(2) @ dims 27:29 stays raw
    stats["std"][:, 27:29] = 1.0        # (was [-2:] = head_pos[y:z] — wrong)
    return stats


def save_stats(stats, path=DATA_DIR / "p1_stats.json"):
    Path(path).write_text(json.dumps(
        {k: v.tolist() for k, v in stats.items()}))


def load_stats(path=DATA_DIR / "p1_stats.json"):
    s = json.loads(Path(path).read_text())
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in s.items()}
