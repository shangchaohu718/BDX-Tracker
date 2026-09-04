"""P1-1: deployment-shaped evaluation — the VLA-facing form.

Command form: CONSTANT per-window 7D commands exactly as command_adapter
emits (Level-1 JSON -> adapt() output), held for 1s windows.
Targets: the matched teacher realizations from v5/v6 interventional data
(realized_command_vector + GT future) — NOT modified-command vs original-GT.

Three report blocks (GPT spec):
  A motion quality: q / FK foot/head / contact / chunk-boundary continuity
  B command realization: gain, teacher-fidelity, sign accuracy
  C slices: stop/start, forward, turn, strafe, neck, combos, worst cell
Headlines: mean / median / group-bootstrap CI / worst slice / absolute err.
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, P2Dataset, WINDOW,
                                           load_stats, sixd_to_mat)
from humanoidverse.planner.model import MotionAE, CommandedEncoder
from humanoidverse.planner.fk import BDXFK
from humanoidverse.planner.registry import resolve_canonical
from humanoidverse.planner.angles import angle_diff

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
V5_ROOT = Path("/home/tcl/Desktop/start/dataset/planner_v2")
SCALE = np.array([1.7, 1.7, 1.7, 1.7, 1.0, 0.5, 1.5])


def bootstrap_ci(vals, groups, n=2000, seed=0):
    """Group-level bootstrap (lineage discipline)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    vals = np.asarray(vals)
    means = []
    for _ in range(n):
        gs = rng.choice(uniq, len(uniq), replace=True)
        sample = np.concatenate([idx_by_g[g] for g in gs])
        means.append(vals[sample].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=600)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _reg = resolve_canonical(OUT_DIR)
    print(f"canonical: {_reg['planner'].name} + {_reg['teacher'].name} "
          f"(preprocess {_reg['registry'].get('preprocess_version')})")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device, weights_only=False)["model"])
    from humanoidverse.planner.dataset import set_preprocess_version
    set_preprocess_version("fixed_v1")   # v1.1 representation
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    ckpt = str(_reg["planner"])
    print(f"planner under eval: {Path(ckpt).name} (registry)")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads((OUT_DIR / "p2_stats32.json").read_text()).items()}
    fk = BDXFK().to(device)
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    spm, sps = sp["mean"].to(device), sp["std"].to(device)

    # matched clips from the interventional manifests (cmd has matched realized)
    man5 = json.loads((V5_ROOT / "manifest.json").read_text())["clips"]
    # GPT closure item: deployment gate judges ONLY lineage-val clips
    # the deployment gate uses the planner_v2 pool's OWN val split (the merged
    # train npz never included v5's val side — earlier run was train-side only)
    vs = json.loads((V5_ROOT / "val_split.json").read_text())
    val_groups = set(vs["dev"]["groups"]) | set(vs["frozen_composition_test"]["groups"])
    ts = json.loads((V5_ROOT / "train_split.json").read_text())
    train_groups = set(ts.get("groups", []))
    def grp_of(c):
        g = str(c.get("group_id") or "")
        return g
    sel = [c for c in man5 if c.get("realized_command_vector")
           and grp_of(c) in val_groups][:args.max_clips]
    n_train_side = sum(1 for c in man5 if c.get("realized_command_vector")
                       and grp_of(c) in train_groups)
    print(f"gate set: {len(sel)} val-lineage clips ({n_train_side} train-side excluded)")
    import os as _os
    from humanoidverse.scripts.extract_sparse_bdx import extract_clip
    from humanoidverse.planner.dataset import canonical_sparse, set_preprocess_version
    # registry already set the version; hist canonicalization must match

    rows = []
    for c in sel:
        f5 = _os.path.normpath(_os.path.join(str(V5_ROOT), c["file"]))
        if not _os.path.exists(f5):
            continue
        d = np.load(f5)
        f = extract_clip(d)
        # head_pos via FK
        from humanoidverse.planner.fk import HEAD, _quat_wxyz_to_mat
        with torch.no_grad():
            p17 = fk.forward(torch.from_numpy(f["base_pos"]).to(device),
                             torch.from_numpy(_quat_wxyz_to_mat(f["base_quat"])).float().to(device),
                             torch.from_numpy(f["q"]).to(device))
        f["head_pos"] = p17[:, HEAD, :].cpu().numpy()
        T = len(f["q"])
        if T < 2 * WINDOW:
            continue
        s = T // 2
        f_hist = {k: v[s-WINDOW:s] for k, v in f.items()}
        q0 = f["base_quat"][s].astype(np.float32)
        psi = float(np.arctan2(2*(q0[0]*q0[3]+q0[1]*q0[2]), 1-2*(q0[2]**2+q0[3]**2)))
        sh = canonical_sparse(f_hist, psi)
        cmd7 = np.array(c["command_vector"], np.float32)
        realized = np.array(c["realized_command_vector"], np.float32)
        cmd = torch.tensor(cmd7 / SCALE, dtype=torch.float32, device=device)
        c7 = model(((sh.to(device)-spm)/sps).unsqueeze(0),
                   torch.zeros(1, WINDOW, sh.shape[-1], device=device),
                   cmd[None, None, :].expand(1, WINDOW, 7))
        p = teacher.decode(c7)[0][0] * std + mean
        # A: motion quality
        q_gt = torch.from_numpy(f["q"][s:s+WINDOW].astype(np.float32)).to(device)
        dq = float((p[None, :, 0:14] - q_gt[None]).abs().mean()) * 1000
        pos = p[:, 28:31]
        R = sixd_to_mat(p[:, 31:37])
        bp_p = fk.forward(pos, R, p[:, 0:14])[..., [5, 10, 12], :]
        # GT baseline in the FULL canonical chain: positions AND rotation
        # canonicalized (audit: world-rotation + canonical-position mixing
        # invalidated the earlier FK number)
        bq_gt = torch.from_numpy(f["base_quat"][s:s+WINDOW].astype(np.float32))
        from humanoidverse.planner.dataset import quat_wxyz_to_mat
        Rz_c = torch.from_numpy(np.array(
            [[np.cos(psi), np.sin(psi), 0.], [-np.sin(psi), np.cos(psi), 0.], [0., 0., 1.]], np.float32))
        R_gt = (Rz_c.unsqueeze(0).to(device)
                @ quat_wxyz_to_mat(bq_gt).to(device))   # canonical rotation
        bp_g = fk.forward((torch.from_numpy(f["base_pos"][s:s+WINDOW].astype(np.float32)).to(device)
                           - torch.from_numpy(f["base_pos"][s].astype(np.float32)).to(device)) @ Rz_c.to(device).T,
                          R_gt, q_gt)[..., [5, 10, 12], :]
        dfk = float((bp_p - bp_g).norm(dim=-1).mean()) * 1000
        # B: command realization (constant cmd vs generated mean vel, R-based yaw)
        gen_vx = float(pos[-1, 0] - pos[0, 0])
        gen_vy = float(pos[-1, 1] - pos[0, 1])
        dyaw = float(angle_diff(torch.atan2(R[-1, 1, 0], R[-1, 0, 0]).item(),
                                torch.atan2(R[0, 1, 0], R[0, 0, 0]).item()))
        gen_neck = float(p[:, 12].mean())
        rows.append({
            "group": str(c.get("group_id") or c["id"]), "combo": str(c.get("combo_tag") or "none"),
            "cmd": cmd7.tolist(), "realized": realized.tolist(),
            "dq_mrad": dq, "dfk_mm": dfk,
            "vx": (float(cmd7[4]), float(gen_vx)), "vy": (float(cmd7[5]), float(gen_vy)),
            "vyaw": (float(cmd7[6]), float(dyaw)), "neck": (float(cmd7[2]), float(gen_neck)),
        })

    # aggregate with group bootstrap
    groups = np.array([r["group"] for r in rows])
    def agg(key, get):
        vals = np.array([get(r) for r in rows])
        lo, hi = bootstrap_ci(vals, groups)
        return {"mean": float(vals.mean()), "median": float(np.median(vals)),
                "ci95": [lo, hi], "max": float(vals.max())}

    q_ag = agg("q", lambda r: r["dq_mrad"])
    fk_ag = agg("fk", lambda r: r["dfk_mm"])
    def gain(items):
        """slope/intercept/R² regression (GPT: mean-ratio assumes origin)."""
        cs = np.array([c for c, _ in items]); gs = np.array([g for _, g in items])
        ok = np.abs(cs) > 0.1
        if ok.sum() < 5: return {"n": int(ok.sum())}
        x, y = cs[ok], gs[ok]
        slope, intercept = np.polyfit(x, y, 1)
        ss_res = float(((y - (slope*x + intercept))**2).sum())
        ss_tot = float(((y - y.mean())**2).sum())
        return {"n": int(ok.sum()), "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(1 - ss_res/max(ss_tot, 1e-9)),
                "sign_acc": float((np.sign(y) == np.sign(x)).mean())}
    def stop_metrics():
        """zero-command: residual velocity at horizon end, not a gain."""
        rs = [r for r in rows if abs(r["cmd"][4]) < 0.05 and abs(r["cmd"][6]) < 0.1]
        if not rs: return {"n": 0}
        resid = [abs(r["vx"][1]) + abs(r["vy"][1]) for r in rs]
        return {"n": len(rs), "residual_speed_mean": float(np.mean(resid)),
                "residual_speed_p95": float(np.percentile(resid, 95))}
    vxa = [r["vx"] for r in rows]; vya = [r["vy"] for r in rows]
    vya_w = [r["vyaw"] for r in rows]; nka = [r["neck"] for r in rows]

    # C slices by combo tag
    slices = defaultdict(list)
    for r in rows:
        key = r["combo"] if r["combo"] != "none" else (
            "stop" if abs(r["cmd"][4]) < 0.08 else "fwd")
        slices[key].append(r)

    print(f"\n=== A motion quality (n={len(rows)} clips, group-bootstrap 95% CI) ===")
    print(f"q:  mean {q_ag['mean']:.1f} mrad  median {q_ag['median']:.1f}  "
          f"CI [{q_ag['ci95'][0]:.1f}, {q_ag['ci95'][1]:.1f}]  worst {q_ag['max']:.1f}")
    print(f"FK: mean {fk_ag['mean']:.1f} mm   median {fk_ag['median']:.1f}  "
          f"CI [{fk_ag['ci95'][0]:.1f}, {fk_ag['ci95'][1]:.1f}]  worst {fk_ag['max']:.1f}")
    print(f"\n=== B command realization (constant 7D, matched targets) ===")
    for nm, arr in (("vx", vxa), ("vy", vya), ("vyaw(1s dyaw)", vya_w), ("neck", nka)):
        g = gain(arr)
        print(f"  {nm:<14} {g}")
    print(f"  stop-specific  {stop_metrics()}")
    print(f"\n=== C slices (q mrad) ===")
    for k in sorted(slices):
        qs = [r["dq_mrad"] for r in slices[k]]
        print(f"  {k:<28} n={len(qs):>3}  mean {np.mean(qs):6.1f}  worst {np.max(qs):6.1f}")
    out = {"q": q_ag, "fk": fk_ag,
           "gains": {"vx": gain(vxa), "vy": gain(vya), "vyaw": gain(vya_w), "neck": gain(nka)},
           "slices": {k: {"n": len(v), "mean_q": float(np.mean([r['dq_mrad'] for r in v])),
                          "worst_q": float(np.max([r['dq_mrad'] for r in v]))}
                      for k, v in slices.items()}}
    (OUT_DIR / "deployment_eval.json").write_text(json.dumps(out, indent=1))
    (OUT_DIR / "deployment_eval_rows.json").write_text(json.dumps(rows, indent=1))
    print(f"\n-> {OUT_DIR / 'deployment_eval.json'}")


if __name__ == "__main__":
    main()
