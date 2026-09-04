"""Teacher-realization artifact — the ONE carry-over from planner closure.

Three-layer decomposition per channel (requested/teacher/planner), lineage-val
only, group-bootstrap CIs, per-bin tables, stop-specific metrics, provenance
hashes. Decides the next data-side task branch:
  teacher/req ~0.55 & planner/teacher ~1.0  -> fix teacher/calibration (NOT more data)
  teacher/req ~1.0  & planner/teacher ~0.55 -> planner sampling/training gap

Run: BDX_PLANNER_DATA=.../bdx_planner_v2combo .venv/bin/python \
     humanoidverse/scripts/eval_teacher_realization.py
"""
import json, os, sys, hashlib, datetime
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, WINDOW, canonical_sparse,
                                           load_stats, sixd_to_mat,
                                           set_preprocess_version)
from humanoidverse.planner.model import MotionAE, CommandedEncoder
from humanoidverse.planner.fk import BDXFK, HEAD, _quat_wxyz_to_mat
from humanoidverse.planner.angles import angle_diff
from humanoidverse.planner.registry import resolve_canonical
from humanoidverse.scripts.extract_sparse_bdx import extract_clip

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
V5 = Path("/home/tcl/Desktop/start/dataset/planner_v2")
SCALE = np.array([1.7, 1.7, 1.7, 1.7, 1.0, 0.5, 1.5])
CHANNELS = {"vx": 4, "vy": 5, "vyaw": 6, "neck_yaw": 2}


def gb_ci(vals, groups, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    vals = np.asarray(vals)
    ms = []
    for _ in range(iters):
        gs = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        ms.append(vals[idx].mean())
    return [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def slope_ci(pairs, groups):
    x = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs])
    ok = np.abs(x) > 0.1        # zero commands excluded from ratios (GPT spec)
    if ok.sum() < 5:
        return {"n": int(ok.sum()), "note": "insufficient nonzero samples"}
    sl, ic = np.polyfit(x[ok], y[ok], 1)
    ssr = ((y[ok] - (sl * x[ok] + ic)) ** 2).sum()
    sst = ((y[ok] - y[ok].mean()) ** 2).sum()
    # slope bootstrap: resample groups among the FILTERED samples only
    rng = np.random.default_rng(1)
    fg = groups[ok]
    ug = np.unique(fg)
    slopes = []
    for _ in range(1000):
        gs = rng.choice(ug, len(ug), replace=True)
        sel_idx = np.concatenate([np.where(fg == g)[0] for g in gs])
        xx, yy = x[ok][sel_idx], y[ok][sel_idx]
        s2, _ = np.polyfit(xx, yy, 1)
        slopes.append(s2)
    return {"n": int(ok.sum()), "slope": round(float(sl), 4),
            "intercept": round(float(ic), 4),
            "r2": round(float(1 - ssr / max(sst, 1e-9)), 4),
            "slope_ci95": [round(float(np.percentile(slopes, 2.5)), 4),
                            round(float(np.percentile(slopes, 97.5)), 4)]}


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = resolve_canonical(OUT_DIR)
    set_preprocess_version(reg["registry"]["preprocess_version"])
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    model.load_state_dict(torch.load(reg["planner"], map_location=device,
                                     weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    fk = BDXFK().to(device)
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    spm, sps = sp["mean"].to(device), sp["std"].to(device)

    man = json.loads((V5 / "manifest.json").read_text())["clips"]
    vs = json.loads((V5 / "val_split.json").read_text())
    val_groups = set(vs["dev"]["groups"]) | set(vs["frozen_composition_test"]["groups"])
    sel = [c for c in man if c.get("realized_command_vector")
           and str(c.get("group_id")) in val_groups]

    R = {}   # rows
    for c in sel:
        f5 = os.path.normpath(os.path.join(str(V5), c["file"]))
        if not os.path.exists(f5):
            continue
        f = extract_clip(np.load(f5))
        with torch.no_grad():
            p17 = fk.forward(torch.from_numpy(f["base_pos"]).to(device),
                             torch.from_numpy(_quat_wxyz_to_mat(f["base_quat"])).float().to(device),
                             torch.from_numpy(f["q"]).to(device))
        f["head_pos"] = p17[:, HEAD, :].cpu().numpy()
        T = len(f["q"])
        if T < 2 * WINDOW:
            continue
        s = T // 2
        fh = {k: v[s - WINDOW:s] for k, v in f.items()}
        q0 = f["base_quat"][s].astype(np.float32)
        psi = float(np.arctan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]),
                               1 - 2 * (q0[2] ** 2 + q0[3] ** 2)))
        sh = canonical_sparse(fh, psi)
        cv = np.array(c["command_vector"], np.float32)
        cmd = torch.tensor(cv / SCALE, dtype=torch.float32, device=device)
        c7 = model(((sh.to(device) - spm) / sps).unsqueeze(0),
                   torch.zeros(1, WINDOW, sh.shape[-1], device=device),
                   cmd[None, None, :].expand(1, WINDOW, 7))
        p = teacher.decode(c7)[0][0] * std + mean
        pos = p[:, 28:31]
        Rt = sixd_to_mat(p[:, 31:37])
        pl = {"vx": float(pos[-1, 0] - pos[0, 0]),          # 1s displacement
              "vy": float(pos[-1, 1] - pos[0, 1]),
              "vyaw": float(angle_diff(float(torch.atan2(Rt[-1, 1, 0], Rt[-1, 0, 0])),
                                        float(torch.atan2(Rt[0, 1, 0], Rt[0, 0, 0])))),
              "neck_yaw": float(p[:, 12].mean())}
        R[str(c["id"])] = {"group": str(c.get("group_id")),
                           "requested": cv.tolist(),
                           "teacher_realized": c["realized_command_vector"],
                           "planner_realized": pl}

    groups = np.array([r["group"] for r in R.values()])
    out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "provenance": {
               "manifest_sha256": hashlib.sha256((V5 / "manifest.json").read_bytes()).hexdigest()[:16],
               "checkpoint_sha256": hashlib.sha256(Path(reg["planner"]).read_bytes()).hexdigest()[:16],
               "stats_sha256": hashlib.sha256((OUT_DIR / "p2_stats32.json").read_bytes()).hexdigest()[:16],
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]},
           "realized_command_vector_note":
               "manifest field computed by the DATA agent during rollout export "
               "(mean realized base velocity over the command window, body frame); "
               "planner_realized here = planner output over the same window "
               "(canonical-frame displacement / R-differenced yaw / neck joint mean)",
           "three_layer": {}, "per_bin": {}, "stop": {}}

    for ch, idx in CHANNELS.items():
        req = [(r["requested"][idx], r["teacher_realized"][idx]) for r in R.values()]
        plq = [(r["requested"][idx], r["planner_realized"][ch]) for r in R.values()]
        tpc = [(r["teacher_realized"][idx], r["planner_realized"][ch]) for r in R.values()]
        out["three_layer"][ch] = {
            "teacher_vs_requested": slope_ci(req, groups),
            "planner_vs_requested": slope_ci(plq, groups),
            "planner_vs_teacher": slope_ci(tpc, groups)}
        # per-bin
        bins = {}
        for r in R.values():
            v = r["requested"][idx]
            b = round(float(v), 1)
            bins.setdefault(b, {"req": [], "tch": [], "pl": []})
            bins[b]["req"].append(v)
            bins[b]["tch"].append(r["teacher_realized"][idx])
            bins[b]["pl"].append(r["planner_realized"][ch])
        out["per_bin"][ch] = {str(b): {"n": len(v["req"]),
                                        "requested_mean": round(float(np.mean(v["req"])), 4),
                                        "teacher_mean": round(float(np.mean(v["tch"])), 4),
                                        "planner_mean": round(float(np.mean(v["pl"])), 4)}
                               for b, v in sorted(bins.items()) if abs(b) >= 0.1}
    # stop-specific
    stops = [r for r in R.values() if abs(r["requested"][4]) < 0.05 and abs(r["requested"][6]) < 0.1]
    if stops:
        resid = [abs(r["planner_realized"]["vx"]) + abs(r["planner_realized"]["vy"]) for r in stops]
        out["stop"] = {"n": len(stops),
                       "planner_residual_speed_mean": round(float(np.mean(resid)), 4),
                       "planner_residual_speed_p95": round(float(np.percentile(resid, 95)), 4)}

    dst = OUT_DIR / "teacher_realization.json"
    dst.write_text(json.dumps(out, indent=1))
    # headline print
    print(f"teacher-realization artifact — {len(R)} lineage-val clips")
    print(f"{'channel':<10} {'tch/req':>18} {'pln/req':>18} {'pln/tch':>18}")
    for ch in CHANNELS:
        t = out["three_layer"][ch]
        def f(k):
            d = t[k]
            return f"{d.get('slope','—')} [{d.get('slope_ci95',['?','?'])[0]},{d.get('slope_ci95',['?','?'])[1]}]" if 'slope' in d else "n/a"
        print(f"{ch:<10} {f('teacher_vs_requested'):>18} {f('planner_vs_requested'):>18} {f('planner_vs_teacher'):>18}")
    print(f"stop: {out['stop'].get('n',0)} clips, residual {out['stop'].get('planner_residual_speed_mean')}")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
