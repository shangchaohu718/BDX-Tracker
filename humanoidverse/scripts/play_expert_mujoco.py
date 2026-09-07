"""[expert-playback] Real-time MuJoCo viewer for GT expert clips (50 Hz).

Plays a curated playlist of expert motions from the planner data pool
(BDX_PLANNER_DATA) directly in an interactive mujoco.viewer window:
kinematic pose replay (qpos teleport per frame, no physics), wall-clock
paced at the data's native 50 fps. Close the window to stop.

Run: BDX_PLANNER_DATA=<pool dir> DISPLAY=:1 MUJOCO_GL=glfw \
     .venv/bin/python humanoidverse/scripts/play_expert_mujoco.py [--list]
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO.parents[0]))

import mujoco
import mujoco.viewer

from humanoidverse.planner.dataset import MotionData

MODEL_PATH = REPO / "data" / "robots" / "bdx" / "scene_bdx_freebase_mujoco.xml"
FPS = 50
MAX_SEC_PER_CLIP = 14.0

# one representative clip per action group; entries are regex patterns matched
# against the full clip name — longest matching clip wins
PLAYLIST = [
    (r"re_dancegen_bounce_sway_\d+$",  "bounce_sway 弹跳摆动"),
    (r"re_dancegen_curious_tilt_\d+$", "curious_tilt 好奇侧倾"),
    (r"re_dancegen_excited_wiggle_\d+$", "excited_wiggle 兴奋扭动"),
    (r"re_dancegen_hop_bounce_\d+$",  "hop_bounce 蹦跳"),
    (r"re_dancegen_nod_greet_\d+$",   "nod_greet 点头问候"),
    (r"re_dancegen_peek_around_\d+$", "peek_around 探头张望"),
    (r"re_dancegen_slide_step_\d+$",  "slide_step 滑步"),
    (r"re_dancegen_side_step_\d+$",   "side_step 侧步"),
    (r"re_dancegen_t_step_\d+$",      "t_step T步"),
    (r"re_dancegen_box_step_\d+$",    "box_step 方形步"),
    (r"re_dancegen_kick_step_\d+$",   "kick_step 踢步"),
    (r"re_dancegen_running_man$",     "running_man 跑步舞"),
    (r"v0/babble_1$",                 "babble 咿呀手势"),
    (r"v0/head_turn45w$",             "head_turn45w 转头"),
]


def pick(md, pattern, min_len):
    import re
    rx = re.compile(pattern)
    cand = [(int(md.d["lengths"][md.name2clip[n]]), n)
            for n in md.split_names["train"] + md.split_names.get("val", [])
            if rx.search(n)]
    if not cand:
        return None
    cand.sort(reverse=True)
    return max(cand, key=lambda c: min(c[0], min_len * 3))[1]


def load_clip(md, name):
    T = int(md.d["lengths"][md.name2clip[name]])
    n = min(T, int(MAX_SEC_PER_CLIP * FPS))
    f = md.frames(md.name2clip[name], 0, n)
    return {"q": f["q"].astype(np.float64),
            "bp": f["base_pos"].astype(np.float64),
            "bq": f["base_quat"].astype(np.float64),   # wxyz, MuJoCo order
            "name": name, "T": n}


def main():
    md = MotionData()
    clips = []
    print("playlist:")
    for pattern, label in PLAYLIST:
        name = pick(md, pattern, 400)
        if name is None:
            print(f"  [skip] {pattern}: no clips")
            continue
        c = load_clip(md, name)
        clips.append(c)
        print(f"  {label:28s} -> {name}  ({c['T']} frames = {c['T']/FPS:.1f}s)")
    if not clips:
        raise SystemExit("no clips selected")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    if model.nq != 21:
        raise SystemExit(f"unexpected nq={model.nq} (expected 21: 3+4+14)")
    mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0.0, 0.0, 0.2]
        v.cam.distance = 1.6
        v.cam.azimuth, v.cam.elevation = 130.0, -18.0
        while v.is_running():
            for c in clips:
                print(f">>> now playing: {c['name']}  "
                      f"({c['T']/FPS:.1f}s)", flush=True)
                t0 = time.perf_counter()
                for i in range(c["T"]):
                    if not v.is_running():
                        return
                    data.qpos[:3] = c["bp"][i]
                    data.qpos[3:7] = c["bq"][i]
                    data.qpos[7:] = c["q"][i]
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    v.cam.lookat[:] = [c["bp"][i][0], c["bp"][i][1], 0.18]
                    v.sync()
                    target = t0 + (i + 1) / FPS
                    dt = target - time.perf_counter()
                    if dt > 0:
                        time.sleep(dt)
                # hold the last pose briefly between clips
                time.sleep(0.4)
        print("viewer closed, bye")


if __name__ == "__main__":
    main()
