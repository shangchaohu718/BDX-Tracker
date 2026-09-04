"""One-off timing probe: per-injection and per-step wall time for the _cf path.

Decides between (a) contention (scan running alongside the smoke) and
(b) structural slowness in the injection path. Standalone process, cuda env,
identical construction to fds_train_dance_core_cf.py.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BFM_ZERO_HARD_LIMIT_RESET", "0")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

ENV_CFG = ROOT / "results/bfm-dynamics-baseline/config.json"
os.environ["BFM_POOL_DATA"] = str(ROOT / "humanoidverse/data/fds_re_dancegen_side_step_4.pkl")

from humanoidverse.scripts.generate_expert_sim_pool import build_cpu_env, inject_read_exact_lean

cfg = json.loads(ENV_CFG.read_text())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped = build_cpu_env(cfg, device)
inner = wrapped._env
dev = torch.device(inner.device)
IS_CUDA = dev.type == "cuda"
print(f"[probe] device={dev}")

ids = torch.zeros(1, dtype=torch.long, device=dev)

# warm-up (motion lib lazy paths, allocator, cuda context)
inject_read_exact_lean(wrapped, ids, torch.full((1,), 0.02, device=dev), settle_reps=2)
torch.cuda.synchronize() if IS_CUDA else None

N_INJ, N_STEP = 30, 200
inj_s = []
for i in range(N_INJ):
    t = torch.full((1,), (i * 7 % 292) / 50.0, device=dev)
    t0 = time.perf_counter()
    inject_read_exact_lean(wrapped, ids, t, settle_reps=2)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    inj_s.append(time.perf_counter() - t0)

a = torch.zeros(1, 14, device=dev)
step_s = []
for i in range(N_STEP):
    t0 = time.perf_counter()
    wrapped.step(a, to_numpy=False)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    step_s.append(time.perf_counter() - t0)

import numpy as np
inj, st = np.array(inj_s), np.array(step_s)
print(f"[probe] injection: mean={inj.mean()*1e3:.1f}ms p50={np.percentile(inj,50)*1e3:.1f}ms "
      f"max={inj.max()*1e3:.1f}ms over {N_INJ}")
print(f"[probe] step:      mean={st.mean()*1e3:.1f}ms p50={np.percentile(st,50)*1e3:.1f}ms "
      f"max={st.max()*1e3:.1f}ms over {N_STEP}")
print(f"[probe] implied batch(2048 steps, ~700 resets): "
      f"{(st.mean()*2048 + inj.mean()*700):.1f}s  (v1 reference ~50s, smoke observed ~600s)")
