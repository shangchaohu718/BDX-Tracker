"""P3.1.2 P1 reconstruction gate — full evaluation (user ruling 2026-09-04).

GATE: frozen teacher (p1_ae_l64, fixed_v1) reconstructs the clean `_rev`
backward clips at q <= 1.3x its own error on the SAME-CATEGORY forward val
windows of the canonical pool, with IDENTICAL windowing (non-overlapping
50-frame windows, canonical fixed_v1 x-construction).

PASS  -> G2 A-line (data fix + student retrain with backward)
FAIL  -> G2 B-line (FROZEN_WITH_DISCLOSURE + envelope/adapter narrowing)

Also reports: clean/anomalous partition of all 400 _rev clips, per-category
window counts, realized vx distribution (true-backward proof), and the 30
anomalous clip names for the data-fix track.

Run: BDX_PLANNER_DATA=<v2combo-baseline> .venv/bin/python \
     humanoidverse/scripts/eval_backward_gate.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, P2Dataset,
                                           WINDOW, load_stats, mat_to_6d,
                                           quat_wxyz_to_mat, quat_yaw)
from humanoidverse.planner.model import MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
POC_POOL = Path(__file__).parents[1] / "data" / "bdx_planner_v2"
GATE_X = 1.3


def canonical_x(f):
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
    return torch.cat([q, qdot, bp_c, mat_to_6d(R_c), rot(blv), rot(bav)], -1), bp_c


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(
        OUT_DIR / "p1_ae_l64.pt", map_location=device,
        weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    mean, std = stats["mean"].to(device), stats["std"].to(device)

    def q_recon(x):
        xn = ((x.to(device) - mean) / std).unsqueeze(0)
        p = teacher.decode(teacher.encode(xn))[0] * std + mean
        return float((p[..., 0:14].cpu() - x[..., 0:14]).abs().mean()) * 1000

    def clip_windows(clip):
        T = len(clip["q"])
        for s in range(0, T - WINDOW + 1, WINDOW):   # non-overlapping
            yield {k: v[s:s + WINDOW] for k, v in clip.items()}

    # ---- (1) canonical-pool val forward baseline, same windowing ----
    from humanoidverse.planner.dataset import set_preprocess_version
    set_preprocess_version("fixed_v1")
    md = MotionData()
    ds = P2Dataset(md, "val", stats=None, max_windows=8000)
    val_err = {}
    for i in range(len(ds.windows)):
        c, s = ds.windows[i]
        if s % WINDOW:                 # same non-overlapping grid
            continue
        f = md.frames(c, s, WINDOW)
        if len(f["q"]) < WINDOW:
            continue
        x, _ = canonical_x(f)
        cat = md.clip_cat[c]
        val_err.setdefault(cat, []).append(q_recon(x))
    val_base = {k: float(np.mean(v)) for k, v in val_err.items() if v}
    print("teacher val forward baseline (matched windowing):")
    for k, v in sorted(val_base.items()):
        print(f"  {k:<12} {v:6.1f} mrad  (n={len(val_err[k])})")

    # ---- (2) all 400 _rev: clean/anomalous partition + gate metric ----
    zp = np.load(POC_POOL / "sparse_full_v2.npz", allow_pickle=True)
    meta = json.load(open(POC_POOL / "meta.json"))
    names = [c["name"] for c in meta["clips"]]
    cats = [c["category"] for c in meta["clips"]]
    idx = {n: i for i, n in enumerate(names)}
    lengths = zp["lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)])
    base_pos_all = zp["base_pos"]
    revs = [n for n in names if n.startswith("v2_backward/") and n.endswith("_rev")]
    clean, anomalous = [], []
    for n in revs:
        stem = n.split("/")[1][:-4]
        cands = [nm for nm in names if nm.split("/")[-1] == stem
                 and not nm.startswith("v2_backward")]
        if not cands:
            anomalous.append((n, "no-original-pair"))
            continue
        io_, ir = idx[cands[0]], idx[n]
        so, sr = slice(starts[io_], starts[io_ + 1]), slice(starts[ir], starts[ir + 1])
        if lengths[io_] < 50 or lengths[ir] < 50:
            anomalous.append((n, "too-short"))
            continue
        dxo = base_pos_all[so][-1, 0] - base_pos_all[so][0, 0]
        dxr = base_pos_all[sr][-1, 0] - base_pos_all[sr][0, 0]
        if dxo * dxr <= 0:
            clean.append((n, ir, cats[io_]))
        else:
            anomalous.append((n, f"dx-not-flipped({dxo:+.2f}->{dxr:+.2f})"))
    print(f"_rev partition: clean={len(clean)} anomalous={len(anomalous)}")

    fields = ("q", "qdot", "base_pos", "base_quat", "base_linvel", "base_angvel")
    data = {k: zp[k] for k in fields}
    gate_err, vxs = {}, []
    per_clip = []
    for n, ic, cat in clean:
        sr = slice(starts[ic], starts[ic] + lengths[ic])
        clip = {k: data[k][sr] for k in fields}
        errs, vlist = [], []
        for w in clip_windows(clip):
            x, bp_c = canonical_x(w)
            errs.append(q_recon(x))
            dx = float(bp_c[-1, 0] - bp_c[0, 0])
            vlist.append(dx * 50 / (WINDOW - 1))
        if errs:
            gate_err.setdefault(cat, []).extend(errs)
            vxs.extend(vlist)
            per_clip.append({"clip": n, "cat": cat, "q_mrad": float(np.mean(errs)),
                             "n_win": len(errs), "vx_mean": float(np.mean(vlist))})
    vxs = np.array(vxs)
    print(f"backward windows total={len(vxs)}  realized vx mean {vxs.mean():+.3f} "
          f"m/s  neg-frac {(vxs < 0).mean() * 100:.1f}%")

    # ---- (3) gate: per matched category ----
    report = {"val_base": val_base, "partition": {"clean": len(clean),
              "anomalous": len(anomalous), "anomalous_list": anomalous},
              "per_category": {}, "vx_neg_frac": float((vxs < 0).mean()),
              "gate_x": GATE_X}
    gate_pass = True
    print(f"\nGATE (teacher q on backward <= {GATE_X}x same-cat forward val):")
    for cat in sorted(gate_err):
        if cat not in val_base:
            print(f"  {cat:<12} (no val baseline — report only)")
            report["per_category"][cat] = {
                "q_backward": float(np.mean(gate_err[cat])), "gate": "NO_BASELINE"}
            continue
        qb, qf = float(np.mean(gate_err[cat])), val_base[cat]
        ratio = qb / qf
        ok = ratio <= GATE_X
        gate_pass &= ok
        report["per_category"][cat] = {
            "q_backward": qb, "q_forward_val": qf, "ratio": ratio,
            "n_windows": len(gate_err[cat]), "gate": "PASS" if ok else "FAIL"}
        print(f"  {cat:<12} {qb:6.1f} / {qf:5.1f} = {ratio:5.3f}x  "
              f"[{'PASS' if ok else 'FAIL'}]  n={len(gate_err[cat])}")
    all_b = [e for v in gate_err.values() for e in v]
    report["overall_q_backward"] = float(np.mean(all_b))
    report["GATE"] = "PASS" if gate_pass else "FAIL"
    print(f"overall backward q {np.mean(all_b):.1f} mrad -> GATE {report['GATE']}")

    out = Path(os.environ.get("GATE_OUT", "backward_gate_result.json"))
    out.write_text(json.dumps(report, indent=1))
    pc_path = out.with_name("backward_gate_per_clip.json")
    pc_path.write_text(json.dumps(per_clip, indent=1))
    print(f"-> {out} (+ {pc_path.name})")


if __name__ == "__main__":
    main()
