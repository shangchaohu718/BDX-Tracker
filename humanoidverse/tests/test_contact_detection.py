"""Regression test for the BDX contact-detection bug fixed 2026-08-18.

History: standing ankle-link origin sits at z≈0.0736 m, so thresholds below
that (old yaml 0.04, G1 default 0.07) made contact_binary ≈ all zeros for BDX
— the obs channel was dead in every tracker training run before the fix.
This test pins the three places that must stay consistent:
  1. bdx_14dof.yaml thresholds (source of truth at runtime),
  2. scripts/extract_sparse_bdx.py constants (planner-side labels),
  3. foot_contact_detect reads config, not hardcoded values,
and re-derives contact on one stand + one walk clip to check behavior.

Run: .venv/bin/python humanoidverse/tests/test_contact_detection.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "humanoidverse" / "scripts"))

DATASET = Path("/home/tcl/Desktop/start/dataset")
HEIGHT_THRES_MIN_STANCE = 0.0736  # measured standing ankle-link height


def load_env_yaml():
    text = (ROOT / "humanoidverse/config/robot/bdx/bdx_14dof.yaml").read_text()
    return text, yaml.safe_load(text)


def test_yaml_thresholds():
    _, cfg = load_env_yaml()
    motion = cfg["robot"]["motion"]
    assert motion["foot_contact_height_thres"] == 0.08, "yaml height thres drifted"
    assert motion["foot_contact_vel_thres"] == 0.3, "yaml vel thres drifted"


def test_extractor_matches_yaml():
    _, cfg = load_env_yaml()
    motion = cfg["robot"]["motion"]
    import extract_sparse_bdx as ex
    assert ex.FOOT_HEIGHT_THRES == motion["foot_contact_height_thres"], \
        "extractor height thres != yaml"
    assert ex.FOOT_VEL_THRES == motion["foot_contact_vel_thres"], \
        "extractor vel thres != yaml"


def test_env_reads_config_not_hardcoded():
    src = (ROOT / "humanoidverse/envs/legged_robot_motions/legged_robot_motions.py").read_text()
    m = re.search(r"def foot_contact_detect.*?return contact_mask", src, re.S)
    assert m, "foot_contact_detect not found"
    body = m.group(0)
    assert "foot_contact_vel_thres" in body and "foot_contact_height_thres" in body, \
        "foot_contact_detect no longer config-driven"
    assert not re.search(r"vel_thres\s*=\s*0\.\d+\s*$", body, re.M), \
        "hardcoded threshold returned"


def contact_from_npz(path, h_thres, v_thres):
    d = np.load(path)
    z = d["body_pos_w"][:, [5, 10], 2]
    v = np.linalg.norm(d["body_lin_vel_w"][:, [5, 10]], axis=-1)
    return (v < v_thres) & (z < h_thres)


def first_clip(version, needle):
    man = json.load(open(DATASET / version / "manifest.json"))
    c = next(c for c in man["clips"] if needle in c["id"])
    return DATASET / version / "clips" / c["file"]


def test_stand_clip_in_contact():
    ct = contact_from_npz(first_clip("v1_stand", "stand"), 0.08, 0.3)
    assert ct.mean() > 0.9, f"stand clip contact only {ct.mean():.2f}"


def test_walk_clip_has_both_states():
    ct = contact_from_npz(first_clip("v1_walk", "walk"), 0.08, 0.3)
    for foot in range(2):
        f = ct[:, foot]
        assert 0.05 < f.mean() < 0.95, f"walk foot {foot} contact frac {f.mean():.2f} not bimodal"
        assert (f == 0).any() and (f == 1).any(), f"walk foot {foot} never toggles"


def test_threshold_covers_measured_stance_height():
    # the original bug: any threshold below the standing ankle height can
    # never fire — guard the invariant, not just the current value
    _, cfg = load_env_yaml()
    assert cfg["robot"]["motion"]["foot_contact_height_thres"] > HEIGHT_THRES_MIN_STANCE, \
        "height threshold below measured standing ankle-link height -> contact dead"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"✓ {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
