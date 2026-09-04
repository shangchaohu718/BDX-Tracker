"""Composition holdout eval — the core experiment: does the planner
FACTORIZE commands and recombine skills, or memorize seen combinations?

Data: planner_v2's 4 strict-unseen combos (mirror twins NOT in train):
  strafe_l+look_l, strafe_l+turn_l, strafe_r+turn_r, turn_l+look_l
vs the 12 training combos. Metric: per-channel command tracking on the
holdout clips (planner conditioned on each clip's true command_vector),
compared against training-combo clips under the same procedure.

Run: BDX_PLANNER_DATA=.../bdx_planner_v2combo .venv/bin/python \
     humanoidverse/scripts/eval_combo_holdout.py
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from humanoidverse.planner.dataset import (MODEL_DIM, WINDOW, canonical_sparse,
                                           load_stats, sixd_to_mat)
from humanoidverse.planner.model import CommandedEncoder, MotionAE

OUT_DIR = Path(os.environ.get("BDX_PLANNER_DATA",
    Path(__file__).parents[1] / "data" / "bdx_planner"))
HOLD_NPZ = OUT_DIR / "v2_holdout.npz"
HOLD_REC = OUT_DIR / "v2_holdout_records.json"
V2_ROOT = Path("/home/tcl/Desktop/start/dataset/planner_v2")
SCALE = np.array([1.7, 1.7, 1.7, 1.7, 1.0, 0.5, 1.5])  # neck×4 / vx / vy / vyaw


def load_holdout():
    d = np.load(HOLD_NPZ)
    recs = json.loads(HOLD_REC.read_text())
    man = {c['id']: c for c in json.loads(
        (V2_ROOT / 'manifest.json').read_text())['clips']}
    offsets = np.concatenate([[0], np.cumsum(d['lengths'])])
    clips = []
    for i, r in enumerate(recs):
        cid = r['name'].split('/', 1)[1]
        m = man.get(cid)
        s, e = int(offsets[i]), int(offsets[i + 1])
        fr = {k: d[k][s:e] for k in ('q', 'qdot', 'base_pos', 'base_quat',
                                     'base_linvel', 'base_angvel', 'contact',
                                     'head_rotvec', 'head_angvel', 'foot_pos',
                                     'foot_yaw', 'head_pos')}
        clips.append({'frames': fr, 'cmd7': m['command_vector'],
                      'combo': str(m.get('combo_tag') or 'none'),
                      'name': r['name']})
    return clips


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: canonical registry")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from humanoidverse.planner.registry import resolve_canonical
    _reg = resolve_canonical(OUT_DIR)
    if args.checkpoint is None:
        args.checkpoint = str(_reg["planner"])
    print(f"canonical: {args.checkpoint}")
    teacher = MotionAE(in_dim=MODEL_DIM, latent_dim=64).to(device).eval()
    teacher.load_state_dict(torch.load(_reg["teacher"], map_location=device,
                                       weights_only=False)["model"])
    model = CommandedEncoder(latent_dim=64, cmd_dim=7).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=False)["model"])
    stats = load_stats(OUT_DIR / "p1_stats.json")
    sp = {k: torch.tensor(v).to(device) for k, v in json.loads(
        (OUT_DIR / "p2_stats32.json").read_text()).items()}
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    spm, sps = sp["mean"], sp["std"]

    clips = load_holdout()
    # also load a sample of TRAINING combos from the same pool for comparison
    d = np.load(OUT_DIR / "v2_train_pool.npz")
    recs = json.loads((OUT_DIR / "v2_train_records.json").read_text())
    man = {c['id']: c for c in json.loads(
        (V2_ROOT / 'manifest.json').read_text())['clips']}
    offsets = np.concatenate([[0], np.cumsum(d['lengths'])])
    import random
    rng = random.Random(0)
    train_combo_idx = [i for i, r in enumerate(recs)
                       if not str(r['tags'][0]).endswith('holdout')
                       and str(r['tags'][0]) != 'none']
    rng.shuffle(train_combo_idx)
    for i in train_combo_idx[:80]:
        r = recs[i]
        cid = r['name'].split('/', 1)[1]
        m = man.get(cid)
        s, e = int(offsets[i]), int(offsets[i + 1])
        fr = {k: d[k][s:e] for k in ('q', 'qdot', 'base_pos', 'base_quat',
                                     'base_linvel', 'base_angvel', 'contact',
                                     'head_rotvec', 'head_angvel', 'foot_pos',
                                     'foot_yaw', 'head_pos')}
        clips.append({'frames': fr, 'cmd7': m['command_vector'],
                      'combo': str(m.get('combo_tag') or 'none'),
                      'name': r['name']})

    agg = defaultdict(lambda: defaultdict(list))
    for c in clips:
        fr = c['frames']
        T = len(fr['q'])
        if T < 2 * WINDOW:
            continue
        s = T // 2
        f_hist = {k: v[s - WINDOW:s] for k, v in fr.items()}
        q0 = fr['base_quat'][s].astype(np.float32)
        psi = float(np.arctan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]),
                               1 - 2 * (q0[2] ** 2 + q0[3] ** 2)))
        sh = canonical_sparse(f_hist, psi)
        cmd = torch.tensor(np.array(c['cmd7']) / SCALE,
                           dtype=torch.float32, device=device)
        c7 = model(((sh.to(device) - spm) / sps).unsqueeze(0),
                   torch.zeros(1, WINDOW, sh.shape[-1], device=device),
                   cmd[None, None, :].expand(1, WINDOW, 7))
        p = teacher.decode(c7)[0][0] * std + mean
        pos = p[:, 28:31]
        R = sixd_to_mat(p[:, 31:37])
        gen = np.array([
            float(pos[-1, 0] - pos[0, 0]) * 50 / (WINDOW - 1),
            float(pos[-1, 1] - pos[0, 1]) * 50 / (WINDOW - 1),
            float(torch.atan2(R[-1, 1, 0], R[-1, 0, 0])
                  - torch.atan2(R[0, 1, 0], R[0, 0, 0])) * 50 / (WINDOW - 1)])
        gen_neck = float(p[:, 12].mean())
        cmd_phys = np.array(c['cmd7'])  # manifest vectors are physical units
        combo = c['combo']
        is_ho = combo.endswith('holdout')
        # direction group (GPT's 3-branch reporting): relationship between
        # neck_yaw cmd and the body-yaw-family cmd (vyaw or vy)
        neck_c = float(np.array(c['cmd7'])[2])
        body_c = float(np.array(c['cmd7'])[6]) if abs(np.array(c['cmd7'])[6]) > 0.15 \
            else float(np.array(c['cmd7'])[5])
        if abs(neck_c) < 0.15 or abs(body_c) < 0.05:
            dgrp = 'neutral'
        elif np.sign(neck_c) == np.sign(body_c):
            dgrp = 'same'
        else:
            dgrp = 'opposite'
        key = combo if is_ho else 'TRAIN(avg)'
        for k2 in ((f'{dgrp}|HO' if is_ho else f'{dgrp}|TR'),):
            pass
        agg[key]['dvx'].append(abs(gen[0] - cmd_phys[4]))
        agg[f'dgrp:{dgrp}|' + ('HO' if is_ho else 'TR')]['dneck'].append(
            abs(gen_neck - cmd_phys[2]))
        agg[f'dgrp:{dgrp}|' + ('HO' if is_ho else 'TR')]['dvyaw'].append(
            abs(gen[2] - cmd_phys[6]))
        agg[key]['dvy'].append(abs(gen[1] - cmd_phys[5]))
        agg[key]['dvyaw'].append(abs(gen[2] - cmd_phys[6]))
        agg[key]['dneck'].append(abs(gen_neck - cmd_phys[2]))
        agg[key]['q'].append(float((p[None, :, 0:14]
             - torch.from_numpy(fr['q'][s:s+WINDOW].astype(np.float32)).to(device))
             .abs().mean()))

    print(f"Composition holdout — {args.checkpoint}")
    print(f"{'combo':<28} {'n':>4} {'|Δvx|':>7} {'|Δvy|':>7} {'|Δvyaw|':>8} "
          f"{'|Δneck|':>8} {'q mrad':>7}")
    report = {}
    order2 = ['dgrp:same|HO', 'dgrp:same|TR', 'dgrp:neutral|HO',
              'dgrp:neutral|TR', 'dgrp:opposite|HO', 'dgrp:opposite|TR']
    print("  direction groups (GPT 3-branch):")
    for key in order2:
        if key not in agg:
            continue
        a = agg[key]
        r = {k2: float(np.mean(v)) for k2, v in a.items()}
        print(f"  {key:<22} n={len(a['dneck']):>4}  |Δneck| {r['dneck']:.3f}  "
              f"|Δvyaw| {r.get('dvyaw', 0):.3f}")
    for key in sorted(agg, key=lambda k: (not k.endswith('holdout'), k)):
        a = agg[key]
        if 'q' not in a or 'dvx' not in a:
            continue   # direction-group keys carry only dneck/dvyaw
        r = {k2: float(np.mean(v)) for k2, v in a.items()}
        report[key] = {'n': len(a['q']), **r}
        print(f"{key:<28} {len(a['q']):>4} {r['dvx']:7.3f} {r['dvy']:7.3f} "
              f"{r['dvyaw']:8.3f} {r['dneck']:8.3f} {r['q']*1000:7.0f}")
    # headline: holdout avg vs train avg
    ho_keys = [k for k in agg if k.endswith('holdout')]
    if ho_keys and 'TRAIN(avg)' in agg:
        for m_ in ('dvx', 'dvy', 'dvyaw', 'dneck', 'q'):
            tr_v = np.mean(agg['TRAIN(avg)'][m_])
            ho_v = np.mean([np.mean(agg[k][m_]) for k in ho_keys])
            print(f"  {m_}: train {tr_v:.4f} -> holdout {ho_v:.4f} "
                  f"(×{ho_v/max(tr_v,1e-9):.2f})")
    (OUT_DIR / "combo_holdout_result.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
