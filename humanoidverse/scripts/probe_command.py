"""P2.6/P2.8 acceptance probe: does the head COMMAND actually control the neck?

For explicit-cmd checkpoints (CommandedEncoder): sweeps the neck_yaw TARGET in
the command channel directly (everything else held at GT) and reports the
completed neck response — the deployment question verbatim.
Acceptance (external review): cmd +0.7 rad -> neck yaw output > +0.5 rad.

For legacy checkpoints (SparseEncoder): falls back to the sparse head-field
edit sweep (edit_both / cmd_only conditions).

Run: .venv/bin/python humanoidverse/scripts/probe_command.py [--checkpoint ...]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, MotionData, WINDOW,
                                           canonical_sparse, load_stats)
from humanoidverse.planner.model import (CommandedEncoder, MotionAE,
                                         SparseEncoder)

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="default: canonical registry")
    ap.add_argument("--clip", default="v0/forward_slow")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from humanoidverse.planner.registry import resolve_canonical
    _reg = resolve_canonical(OUT_DIR)
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(
        _reg["teacher"], map_location=device, weights_only=False)["model"])
    if args.checkpoint is None:
        args.checkpoint = str(_reg["planner"])
    print(f"canonical: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    explicit = bool(ckpt["config"].get("explicit_cmd"))
    student = (CommandedEncoder if explicit else SparseEncoder)(
        latent_dim=ckpt["config"]["latent"]).to(device).eval()
    student.load_state_dict(ckpt["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v).to(device) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    md = MotionData()
    ci = md.name2clip[args.clip]
    s = int(md.d["lengths"][ci]) // 2
    f_fut, f_hist = md.frames(ci, s, WINDOW), md.frames(ci, s - WINDOW, WINDOW)
    bq0 = f_fut["base_quat"][0].astype(np.float32)
    psi = float(np.arctan2(2 * (bq0[0] * bq0[3] + bq0[1] * bq0[2]),
                           1 - 2 * (bq0[2] ** 2 + bq0[3] ** 2)))
    sf, sh = canonical_sparse(f_fut, psi), canonical_sparse(f_hist, psi)
    cmd0 = torch.from_numpy(f_fut["q"][:, 10:14].astype(np.float32)) / 1.7

    def run(sf, sh, cmd):
        with torch.no_grad():
            c = student(((sf.to(device) - sp["mean"]) / sp["std"]).unsqueeze(0),
                        ((sh.to(device) - sp["mean"]) / sp["std"]).unsqueeze(0),
                        cmd.to(device).unsqueeze(0) if cmd is not None else None)
            p = teacher.decode(c)[0][0] * stats["std"].to(device) + stats["mean"].to(device)
        return p[:, 10:14].cpu().numpy()

    print(f"probe — {args.checkpoint} on {args.clip} (explicit_cmd={explicit})")
    if explicit:
        resp = []
        for yaw in (-0.7, -0.35, 0.0, 0.35, 0.7):
            cmd = cmd0.clone()
            cmd[:, 2] = yaw                       # neck_yaw target
            n = run(sf, sh, cmd)
            resp.append((yaw, n[:, 2].mean(), n[:, 0].mean(), n[:, 1].mean()))
        line = "  ".join(f"{y:+.2f}->yaw{w:+.3f}" for y, w, _, _ in resp)
        slope = np.polyfit([r[0] for r in resp], [r[1] for r in resp], 1)[0]
        print(f"  yaw sweep  {line}   gain {slope:.2f}")
        fwd_resp = []
        for fv in (0.0, 0.8, 1.5):
            cmd = cmd0.clone()
            cmd[:, 0] = fv                        # neck_forward (lift) target
            n = run(sf, sh, cmd)
            fwd_resp.append((fv, n[:, 0].mean()))
        print("  lift sweep " + "  ".join(f"{v:+.2f}->{w:+.3f}" for v, w in fwd_resp))
    else:
        for mode in ("edit_both", "cmd_only"):
            resp = []
            for yaw in (-0.7, 0.0, 0.7):
                sf2, sh2 = sf.clone(), sh.clone()
                sf2[:, 13:16] = torch.tensor([0., 0., yaw])
                sf2[:, 16:19] = 0
                if mode == "edit_both":
                    sh2[:, 13:16] = torch.tensor([0., 0., yaw])
                    sh2[:, 16:19] = 0
                else:
                    sh2[:, 13:19] = 0
                n = run(sf2, sh2, None)
                resp.append((yaw, n[:, 2].mean()))
            slope = np.polyfit([y for y, _ in resp], [w for _, w in resp], 1)[0]
            print(f"  {mode:<10} " + "  ".join(f"{y:+.2f}->{w:+.3f}"
                                               for y, w in resp) + f"   gain {slope:.2f}")


if __name__ == "__main__":
    main()
