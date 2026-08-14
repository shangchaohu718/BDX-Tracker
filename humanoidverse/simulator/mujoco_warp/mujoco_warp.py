import os
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
from loguru import logger
from pathlib import Path

from humanoidverse.utils.torch_utils import *
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator


class MuJoCoWarp(BaseSimulator):
    """GPU-accelerated MuJoCo simulator backend using MuJoCo Warp.

    Provides vectorized simulation of `num_envs` parallel worlds on CUDA,
    exposing the same BaseSimulator interface as IsaacSim but without
    requiring RT cores.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_cfg = config.robot
        self.domain_rand_config = config.domain_rand
        self.freebase = True
        self.visualize_viewer = False
        self.renderer = None
        self.viewer = None
        self._graph = None
        self._step_count = 0
        self._use_cuda_graph = self.simulator_config.sim.get("use_cuda_graph", True)
        # [BFM-DIAG-NAN] allow disabling the CUDA graph. The graph replays mjw.step on Warp's stream;
        # an eager pre-step clamp (BFM_ZERO_HARD_LIMIT_CLAMP, a PyTorch op) issued on PyTorch's stream
        # may race the graph launch so the past-limit state slips through un-clamped. Disabling the
        # graph makes every substep an eager mjw.step on Warp's stream (serialized with the clamp) and
        # is the diagnostic + fallback if the clamp fails under the graph. ~2x slower. Off by default.
        if os.environ.get("BFM_ZERO_NO_CUDA_GRAPH", "0") == "1":
            self._use_cuda_graph = False

        # [BFM-DIAG-NAN] observe-only sim-level probe (added by Claude). Gated by
        # BFM_ZERO_NAN_PROBE=1; default OFF so normal runs are untouched. When on, it detects
        # the FIRST physics substep that writes a non-finite generalized state and dumps the
        # pre-step |qvel| plus post-step qacc / qfrc_constraint / nefc to pinpoint whether the
        # NaN comes from the constraint solver, the mass-matrix solve, or a buffer overflow.
        # Diagnostic only -- it never sanitizes, clamps, or halts the simulation.
        self._diag_probe = os.environ.get("BFM_ZERO_NAN_PROBE", "0") == "1"
        self._diag_nan_dumped = False
        self._diag_probe_checks = 0
        self._diag_pre_qvel_absmax = None
        self._diag_pre_qvel = None       # [BFM-DIAG-NAN] per-env pre-step qvel snapshot (for divergence dump)
        self._diag_pre_ctrl = None       # [BFM-DIAG-NAN] per-env pre-step ctrl (action) snapshot
        # [BFM-DIAG-NAN] how often (in physics substeps) the probe scans for non-finite state.
        # Default once per control step; set BFM_ZERO_NAN_PROBE_DECIM=1 to catch the EXACT first-NaN
        # substep (needed to see the triggering pre-step state, since the default 4 catches it late).
        self._diag_decimation = int(os.environ.get(
            "BFM_ZERO_NAN_PROBE_DECIM", self.simulator_config.sim.get("control_decimation", 4)))
        self._diag_njmax = None
        self._diag_nconmax = None
        # [BFM-DIAG-NAN] observe-only reset/init tracking (added by Claude). Records the global
        # physics-substep (_step_count) at which each env's qpos/qvel was last (re)written via
        # set_*_state_tensor (the reset/init path, legged_robot_base.py:321-322). Lets the
        # first-NaN dump report steps_since_last_set for the diverging envs, distinguishing a
        # freshly-(re)initialized blowup (implicates the state we fed) from a mid-rollout one.
        # Lazy-created on first set_* call (num_envs known then). Diagnostic only.
        self._diag_last_set_step = None

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def setup(self):
        hv_root = Path(__file__).parents[2]
        # Scene path is config-driven: robot.asset.scene_file (relative to data/robots/).
        # Falls back to the G1 scene so existing G1 runs are byte-identical.
        scene_file = self.robot_cfg.asset.get("scene_file", None)
        if scene_file is None:
            scene_file = "g1/scene_29dof_freebase_mujoco.xml"
        self.model_path = str(hv_root / "data/robots" / scene_file)

        # Load standard MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.mj_model.opt.timestep = 1.0 / self.simulator_config.sim.fps
        self.sim_dt = self.mj_model.opt.timestep

        # [BFM-DIAG-NAN] Arm A (integrator) — NOT the fix; kept as opt-in defense-in-depth only.
        # The REAL fix is Arm B (deterministic hard-joint-limit reset in _update_reset_buf), which
        # prevents the trigger from ever arising. The NaN trigger is a joint starting a step ALREADY
        # PAST its hard jnt_range -> the LIMIT CONSTRAINT solve produces NaN qfrc_constraint. Proven
        # via state surgery on the captured divergence-onset state (env 6218): switching Euler ->
        # implicitfast does NOT prevent the NaN; only clamping the violating joint does. So the
        # integrator is downstream of the real problem. It's left switchable here (default Euler, to
        # preserve the baseline) purely as a robustness knob to try alongside Arm B if a residual
        # solver fragility shows up:   BFM_ZERO_MJW_INTEGRATOR=implicitfast
        # choices: euler | rk4 | implicit | implicitfast (mujoco.mjtIntegrator). Note mjwarp does
        # not implement implicit+fluid models (io.py:184) but the G1 has no fluid -> fine here.
        _integ = os.environ.get("BFM_ZERO_MJW_INTEGRATOR", "euler").strip().lower()
        _integ_map = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }
        if _integ in _integ_map:
            self.mj_model.opt.integrator = _integ_map[_integ]
            if _integ != "euler":
                logger.info(f"[BFM-DIAG-NAN] mujoco_warp integrator set to {_integ} "
                            f"(was Euler) via BFM_ZERO_MJW_INTEGRATOR")

        # Store defaults for domain randomization
        self._default_dof_frictionloss = self.mj_model.dof_frictionloss.copy()
        self._default_body_mass = self.mj_model.body_mass.copy()
        self._default_geom_friction = self.mj_model.geom_friction.copy()

        # These are created lazily in create_envs when we know num_envs
        self.m = None  # mjw.Model
        self.d = None  # mjw.Data
        self.mjm = None  # mujoco.MjModel (possibly modified by domain rand)

        # [BFM-DIAG-NAN] Fix #3: trim full-body collisions to feet + fall-detection bodies BEFORE
        # the model is first uploaded. contype/conaffinity are never touched by domain
        # randomization (which only resets mass/frictionloss/friction from saved defaults), so
        # this one-time edit on mj_model persists across every later mjw.put_model re-upload.
        self._apply_collision_trim()

        logger.info(f"MuJoCoWarp setup complete. Model: {self.model_path}")

    def _apply_collision_trim(self):
        # [BFM-DIAG-NAN] Fix #3 (added by Claude): make non-essential robot geoms visual-only by
        # zeroing their contype/conaffinity bitmasks (a geom with contype=0 & conaffinity=0 collides
        # with nothing). Kept = feet (env contact_bodies: ankle_roll_link) + fall-detection bodies
        # (env terminate/penalize_contacts_on: pelvis, shoulder, hip), matched by SUBSTRING so a
        # single "shoulder"/"hip" token covers all six L/R links. The floor plane (world body 0) is
        # always preserved. Directly removes the full-body contact pile-up (logged "increase njmax
        # to 1117") that overflowed the constraint buffer -- the upstream trigger of the sim NaN.
        # Gated by sim.trim_collision (default off); reversible by setting it false in the yaml.
        if not self.simulator_config.sim.get("trim_collision", False):
            return
        keep = [str(s) for s in self.simulator_config.sim.get("trim_collision_keep_bodies", [])]
        if not keep:
            return
        m = self.mj_model
        static_words = ("floor", "ground", "terrain", "plane")
        disabled = 0
        kept = 0
        for g in range(m.ngeom):
            bid = int(m.geom_bodyid[g])
            gname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            # Never disable the world body (holds the floor/terrain plane) or any obviously static
            # environment geom matched by name -- belt-and-suspenders on top of the bodyid==0 guard.
            if bid == 0 or any(w in gname for w in static_words):
                kept += 1
                continue
            if any(k in bname for k in keep):
                kept += 1
                continue
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
            disabled += 1
        logger.info(
            f"[BFM-DIAG-NAN] collision trim: {kept} geom(s) still colliding, "
            f"{disabled} made visual-only; keep-substrings={keep}"
        )

    def setup_terrain(self, mesh_type):
        if mesh_type == "plane":
            pass  # Already defined in the XML
        elif mesh_type in ("heightfield", "trimesh"):
            logger.warning(f"MuJoCoWarp: terrain type '{mesh_type}' not yet supported, using plane")
        elif mesh_type is not None:
            raise ValueError(f"Unknown terrain mesh type: {mesh_type}")

    # ------------------------------------------------------------------ #
    #  Assets
    # ------------------------------------------------------------------ #

    def load_assets(self):
        self.num_dof = self.mj_model.nv - 6
        self.num_bodies = self.mj_model.nbody - 1

        self.dof_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.mj_model.njnt)
        ][1:]
        self.body_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in range(self.mj_model.nbody)
        ][1:]

        self.body_id = np.arange(self.num_bodies, dtype=np.int32) + 1

        # If the loaded model has more bodies than the robot config expects, prune the
        # extras (e.g. G1's fakehand bodies). Generalized from the old `"29" in model_path`
        # check so any robot works. The asserts below still catch a true mismatch.
        expected_bodies = list(self.robot_cfg.body_names)
        if self.num_bodies > len(expected_bodies):
            for b in range(self.mj_model.nbody):
                name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
                if name and name not in expected_bodies and name in self.body_names:
                    self.body_names.remove(name)
                    self.num_bodies -= 1
                    self.body_id = np.delete(self.body_id, np.where(self.body_id == b))

        assert self.num_dof == len(self.robot_cfg.dof_names), (
            f"DOF count mismatch: model={self.num_dof}, config={len(self.robot_cfg.dof_names)}"
        )
        assert self.dof_names == self.robot_cfg.dof_names, "DOF names mismatch"
        assert self.body_names == self.robot_cfg.body_names, "Body names mismatch"

        # Alias used by LeggedRobotMotions._init_motion_extend
        self._body_list = self.body_names

    # ------------------------------------------------------------------ #
    #  Environment creation
    # ------------------------------------------------------------------ #

    def create_envs(self, num_envs, env_origins, base_init_state):
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        nconmax = self.simulator_config.sim.get("nconmax", 50)
        njmax = self.simulator_config.sim.get("njmax", 100)
        # [BFM-DIAG-NAN] record configured buffer caps for the probe's overflow report
        self._diag_nconmax = nconmax
        self._diag_njmax = njmax

        # Apply domain randomization before creating MJWarp model
        self.episodic_domain_randomization(None)

        # Create MJWarp model and data
        self.m = mjw.put_model(self.mj_model)
        self.d = mjw.make_data(self.mj_model, nworld=num_envs, nconmax=nconmax, njmax=njmax)

        # Initialize all worlds to default state
        self._init_all_envs(base_init_state)

        # Run a forward pass to populate derived quantities
        mjw.forward(self.m, self.d)

        logger.info(
            f"MuJoCoWarp created {num_envs} parallel worlds "
            f"(nconmax={nconmax}, njmax={njmax})"
        )
        return [self.d], []

    def _init_all_envs(self, base_init_state):
        """Initialize all worlds to the given base state."""
        nq = self.mj_model.nq
        nv = self.mj_model.nv

        # Default joint positions (from init_state.default_joint_angles in robot config)
        default_joint_angles = self.robot_cfg.init_state.default_joint_angles
        dof_order = self.dof_names  # ordered list from MuJoCo model
        default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in dof_order],
            dtype=torch.float32, device=self.sim_device
        )

        # Build initial qpos for all worlds: [nworld, nq]
        # Free base: qpos[0:3] = position, qpos[3:7] = quaternion (wxyz)
        qpos_init = np.zeros((self.num_envs, nq), dtype=np.float32)
        qpos_init[:, 0:3] = base_init_state[0:3].cpu().numpy()
        # Quaternion: config uses xyzw, MuJoCo uses wxyz
        quat_xyzw = base_init_state[3:7].cpu().numpy()
        qpos_init[:, 3] = quat_xyzw[3]  # w
        qpos_init[:, 4] = quat_xyzw[0]  # x
        qpos_init[:, 5] = quat_xyzw[1]  # y
        qpos_init[:, 6] = quat_xyzw[2]  # z
        qpos_init[:, 7:] = default_dof_pos.cpu().numpy()

        # Build initial qvel for all worlds: [nworld, nv]
        qvel_init = np.zeros((self.num_envs, nv), dtype=np.float32)
        qvel_init[:, 0:6] = base_init_state[7:13].cpu().numpy()

        # Copy to Warp arrays
        wp_qpos = wp.array(qpos_init, dtype=wp.float32)
        wp_qvel = wp.array(qvel_init, dtype=wp.float32)
        wp.copy(self.d.qpos, wp_qpos)
        wp.copy(self.d.qvel, wp_qvel)

        # Zero ctrl
        wp_zero_ctrl = wp.zeros((self.num_envs, self.mj_model.nu), dtype=wp.float32)
        wp.copy(self.d.ctrl, wp_zero_ctrl)

        # [BFM-DIAG-NAN] cache the finite standing-pose state (qpos/qvel) per env. Used by the
        # per-substep NaN-reset safety net below (simulate_at_each_physics_step) to project any
        # env whose state went non-finite back to a known-good finite state before the env layer
        # reads obs/reward/gradient. Live on sim_device so the reset is a pure-GPU masked write.
        self._nan_safe_qpos = torch.from_numpy(qpos_init).to(self.sim_device).contiguous()
        self._nan_safe_qvel = torch.from_numpy(qvel_init).to(self.sim_device).contiguous()
        self._nan_resets_gpu = torch.zeros(1, dtype=torch.int64, device=self.sim_device)  # cumulative count, no per-substep sync

    # ------------------------------------------------------------------ #
    #  Domain randomization
    # ------------------------------------------------------------------ #

    def episodic_domain_randomization(self, env_ids):
        # Reset to defaults
        self.mj_model.dof_frictionloss[:] = self._default_dof_frictionloss
        self.mj_model.body_mass[:] = self._default_body_mass
        self.mj_model.geom_friction[:] = self._default_geom_friction

        if self.domain_rand_config.get("randomize_link_mass", False):
            dmass = np.random.uniform(
                low=self.domain_rand_config["link_mass_range"][0],
                high=self.domain_rand_config["link_mass_range"][1],
                size=(self.mj_model.nbody,),
            )
            self.mj_model.body_mass[:] *= dmass

        if self.domain_rand_config.get("randomize_friction", False):
            njnt = self.mj_model.njnt
            frictionloss = self.mj_model.dof_frictionloss[6:] * np.random.uniform(
                low=abs(self.domain_rand_config["friction_range"][0]),
                high=abs(self.domain_rand_config["friction_range"][1]),
                size=(njnt - 1,),
            )
            self.mj_model.dof_frictionloss[6:] = frictionloss

            friction = self.mj_model.geom_friction * np.random.uniform(
                low=abs(self.domain_rand_config["friction_range"][0]),
                high=abs(self.domain_rand_config["friction_range"][1]),
                size=(self.mj_model.ngeom, 1),
            )
            self.mj_model.geom_friction[:] = friction

        if self.domain_rand_config.get("randomize_base_com", False) and self.robot_cfg.has_torso:
            torso_id = self.mj_model.body(self.robot_cfg.torso_name).id
            assert torso_id > -1
            x_range = self.domain_rand_config["base_com_range"]["x"]
            dmass_torso = np.random.uniform(low=x_range[0], high=x_range[1])
            self.mj_model.body_mass[torso_id] += dmass_torso

        # If MJWarp model already exists, re-upload randomized parameters
        if self.m is not None:
            self.m = mjw.put_model(self.mj_model)
            self._graph = None  # Invalidate CUDA graph

    # ------------------------------------------------------------------ #
    #  Simulation stepping
    # ------------------------------------------------------------------ #

    def prepare_sim(self):
        mjw.forward(self.m, self.d)
        self.refresh_sim_tensors()

    def refresh_sim_tensors(self):
        pass  # Warp arrays are always live on GPU

    def simulate_at_each_physics_step(self):
        # [BFM-DIAG-NAN] DEFENSE-IN-DEPTH (NOT the NaN fix): per-substep hard-joint-limit CLAMP.
        # Investigation proved the captured divergence state has joints at LEGAL angles (max 2.737 rad
        # vs hip/shoulder limit ~2.97) -- the real NaN is a floating-BASE blowup, not a joint-limit
        # violation, so this clamp cannot fix it (that is handled by the NaN-reset net below). It is
        # kept because it is harmless (identity no-op when in-bounds -> no host sync to fight the CUDA
        # graph) and does protect a SEPARATE failure mode: if a joint ever genuinely exceeds its
        # hard jnt_range, this projects it back in-range and kills the outbound velocity, which is
        # what PhysX effectively does. The torch in-place ops (clamp_/masked_fill_) DO propagate to
        # the live Warp memory (DLPack zero-copy view; proven by a direct propagation test). qpos=
        # [7 base + 29 joint], qvel=[6 base + 29 joint]. Gated by BFM_ZERO_HARD_LIMIT_CLAMP
        # (default 1 = on; =0 disables). THE actual NaN fix is BFM_ZERO_NAN_RESET below.
        #
        # [ARCH-CHECK 2026-07-07] Verified mujoco_warp's solver ALREADY enforces joint limits via the
        # standard soft unilateral constraint (_limit_slide_hinge kernel in mujoco_warp/_src/constraint.py:
        # `active = (qpos - range) < 0` -> restoring force via jnt_solref/jnt_solimp). The solver does NOT
        # hard-clamp qpos -- it applies forces, so it cannot rescue an already-infeasible (past-limit)
        # start state, which is the only thing this custom clamp did. Joint-limit enforcement is the
        # simulator's job and the solver does it; this wrapper-level qpos patch was duplicating/overriding
        # that responsibility in the wrong layer. Commented out accordingly. If a past-limit start state
        # ever produces NaN again, the proper fix is the NaN-reset net (below) or an upstream solver
        # change, not an env/qpos clamp.
        # if os.environ.get("BFM_ZERO_HARD_LIMIT_CLAMP", "1") != "0" and getattr(self, "hard_dof_pos_limits", None) is not None:
        #     _qp = wp.to_torch(self.d.qpos)               # [N, nq]
        #     _qv = wp.to_torch(self.d.qvel)               # [N, nv]
        #     _jp = _qp[:, 7:7 + self.num_dof]             # [N, 29] joint pos (LIVE view of d.qpos)
        #     _lo = self.hard_dof_pos_limits[:, 0]
        #     _hi = self.hard_dof_pos_limits[:, 1]
        #     _past_hi = _jp > _hi                          # outbound mask BEFORE clamping
        #     _past_lo = _jp < _lo
        #     _jp.clamp_(min=_lo, max=_hi)                  # project joint pos into hard range (identity if in-bounds)
        #     _jv = _qv[:, 6:6 + self.num_dof]             # [N, 29] joint vel (LIVE view of d.qvel)
        #     _jv.masked_fill_(_past_hi & (_jv > 0), 0.0)  # kill velocity pushing further past the HI limit
        #     _jv.masked_fill_(_past_lo & (_jv < 0), 0.0)  # kill velocity pushing further past the LO limit

        # [BFM-DIAG-NAN] observe-only probe: snapshot pre-step |qvel| (async, no host sync) so
        # that IF this substep produces a non-finite state we can report the extreme state that
        # triggered the divergence. Gated by BFM_ZERO_NAN_PROBE=1.
        if self._diag_probe:
            _qv = wp.to_torch(self.d.qvel)
            self._diag_pre_qvel_absmax = _qv.abs().max()
            # [BFM-DIAG-NAN] snapshot per-env pre-step qpos/qvel + ctrl (action) so the divergence dump
            # can show the EXACT state/action fed into the first-NaN step (garbage vs extreme-finite),
            # AND so the replayable npz below can capture the divergence-onset state for cross-backend
            # replay (diff_sim_io). With decim=1 these are the inputs to the just-stepped substep.
            self._diag_pre_qpos = wp.to_torch(self.d.qpos).clone()
            self._diag_pre_qvel = _qv.clone()
            self._diag_pre_ctrl = wp.to_torch(self.d.ctrl).clone()

        if self._use_cuda_graph and self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            mjw.step(self.m, self.d)

        # Capture CUDA graph after warmup
        if (
            self._use_cuda_graph
            and self._step_count == 5
            and self._graph is None
        ):
            try:
                with wp.ScopedCapture() as capture:
                    mjw.step(self.m, self.d)
                self._graph = capture.graph
                logger.info("MuJoCoWarp: CUDA graph captured")
            except Exception as e:
                logger.warning(f"MuJoCoWarp: CUDA graph capture failed ({e}), continuing without")
                self._use_cuda_graph = False

        self._step_count += 1

        # [BFM-DIAG-NAN] THE fix: per-substep NaN-reset safety net (mjwarp only). mjwarp's constraint
        # solver intermittently emits a NON-FINITE state on rare extreme configs that PhysX tolerates;
        # at 8192 envs this rare-per-env event becomes near-certain per batch, and a single NaN'd env
        # poisons the whole training gradient. Right after mjw.step, project any env whose qpos/qvel
        # became non-finite back to the cached finite standing pose (self._nan_safe_*), BEFORE the env
        # layer reads obs/reward (which happens after all control_decimation substeps complete). This
        # is heal-and-continue, NOT terminate: the bad env gets a clean state and training proceeds.
        # The trigger is STRICTLY isfinite -- a definitive "the math is invalid" signal, no threshold
        # and no false positives (this is MuJoCo's own documented divergence detector: "Nan, Inf or
        # huge value in QACC"). A position/velocity-magnitude guard was considered and REJECTED: envs
        # spawn on a grid (env_origins spaced 5 m apart -> legit base xy reaches ~50 m at 64 envs and
        # far more at 8192), so any world-frame magnitude threshold conflates "far down the grid" with
        # "diverged" -- an ungrounded magic number. Only non-finite is reset. Implementation is a
        # pure-GPU masked write (torch.where -> .copy_), CUDA-graph-safe and ZERO host-sync; .copy_ on
        # the wp.to_torch view propagates to the live Warp memory the next substep reads (proven:
        # set_dof_state_tensor uses the same __setitem__ path). When every env is finite the mask is
        # all-False and the write is an identity. qpos=[7 base + 29 joint], qvel=[6 base + 29 joint].
        # Gated by BFM_ZERO_NAN_RESET (default 1 = on; =0 reproduces the baseline divergence for A/B).
        if os.environ.get("BFM_ZERO_NAN_RESET", "1") != "0" and getattr(self, "_nan_safe_qpos", None) is not None:
            _qp = wp.to_torch(self.d.qpos)  # [N, nq] live view of d.qpos (post-step, possibly non-finite)
            _qv = wp.to_torch(self.d.qvel)  # [N, nv] live view of d.qvel
            _bad = ~(torch.isfinite(_qp).all(dim=-1) & torch.isfinite(_qv).all(dim=-1))  # [N] bool, GPU
            _nbad = _bad.sum()
            # Unconditional masked write (no host-sync branch): identity when all envs finite,
            # projects bad envs to the safe pose otherwise. A few elementwise kernels per substep,
            # negligible vs mjw.step, and stays CUDA-graph-safe.
            _qp.copy_(torch.where(_bad[:, None], self._nan_safe_qpos, _qp))
            _qv.copy_(torch.where(_bad[:, None], self._nan_safe_qvel, _qv))
            self._nan_resets_gpu += _nbad  # GPU accumulate; no per-substep host sync
            # one sync per 2000 substeps (~10s) so we can see how often mjwarp diverges
            if self._step_count % 2000 == 0:
                _tot = int(self._nan_resets_gpu.item())
                if _tot:
                    print(f"[BFM-DIAG-NAN] NaN-reset safety net: {_tot} env-reset(s) so far "
                          f"(step_count={self._step_count}, profile_envs={self.num_envs})", flush=True)

        # [BFM-DIAG-NAN] observe-only: detect+dump the FIRST non-finite state. Checked once per
        # env step (every control_decimation substeps) to limit host-sync overhead; NaN persists
        # across substeps so this never misses it.
        if self._diag_probe and (self._step_count % max(1, self._diag_decimation) == 0):
            self._diag_check_after_step()

    def _diag_check_after_step(self):
        # [BFM-DIAG-NAN] observe-only probe body (added by Claude). One host sync per call to
        # detect the first substep whose generalized state (qpos/qvel) became non-finite, then a
        # rich one-shot dump. Diagnostic only -- never alters state.
        # [BFM-DIAG-NAN] observe-only probe (added by Claude). Strictly observe-only: the whole
        # method is try/except-wrapped and every per-world read is shape-guarded, so a legacy/
        # scalar data field (e.g. nacon/nefc not stored per-world in some mjwarp builds) degrades
        # to "n/a" instead of an index-out-of-bounds that would abort the run. Never alters state.
        try:
            qpos_t = wp.to_torch(self.d.qpos)
            qvel_t = wp.to_torch(self.d.qvel)
            finite = torch.isfinite(qpos_t).all(dim=-1) & torch.isfinite(qvel_t).all(dim=-1)
            self._diag_probe_checks += 1
            n_bad = int((~finite).sum().item())
            if n_bad == 0 or self._diag_nan_dumped:
                return
            self._diag_nan_dumped = True
            self._diag_dump_nan(finite, n_bad)
        except Exception as e:  # noqa -- the probe must never crash training
            if not getattr(self, "_diag_probe_err", False):
                print(f"[BFM-DIAG-NAN] probe error (ignored, run continues): {e}", flush=True)
                self._diag_probe_err = True

    def _diag_dump_nan(self, finite, n_bad):
        # [BFM-DIAG-NAN] rich one-shot dump for the first non-finite substep. Every read is guarded
        # so a non-per-world field prints "n/a"/its shape rather than OOB-indexing.
        bad_ids = torch.nonzero(~finite, as_tuple=False).flatten()
        k = bad_ids[: min(8, len(bad_ids))].to(torch.int64)
        kmax = int(k.max().item()) if k.numel() else -1

        def _stat(name, t):
            # per-bad-env slice if t is per-world (shape[0] > max bad id); else whole-tensor stats
            try:
                if t is None or t.numel() == 0:
                    return f"{name}: n/a"
                sub = t[k] if (t.dim() >= 1 and t.shape[0] > kmax) else t
                nf = int((~torch.isfinite(sub)).sum().item())
                amax = float(sub.abs().max().item()) if sub.numel() else 0.0
                return f"{name}: nonfinite={nf}  absmax={amax:+.6e}"
            except Exception as e:  # noqa
                return f"{name}: <read err {e}>"

        def _gmax(t):  # global max of a per-world COUNT tensor; safe for scalar/legacy shapes
            try:
                if t is None or t.numel() == 0:
                    return "n/a"
                v = t.max().item()
                return str(int(v)) if float(v).is_integer() else f"{v}"
            except Exception:  # noqa
                return "n/a"

        def _bad_list(t):  # per-bad-env values of a per-world count tensor; else report shape
            try:
                if t is None or t.numel() == 0:
                    return "n/a"
                if t.dim() >= 1 and t.shape[0] > kmax:
                    return str(t[k].tolist())
                return f"<shape={tuple(t.shape)} not per-world>"
            except Exception as e:  # noqa
                return f"<read err {e}>"

        qacc_t = wp.to_torch(self.d.qacc)                 # generalized acceleration (solver output)
        qfc_t = wp.to_torch(self.d.qfrc_constraint)        # constraint forces (contact/limit solver)
        qfa_t = wp.to_torch(self.d.qfrc_actuator)          # actuator forces (the torques we fed in)
        qvel_t = wp.to_torch(self.d.qvel)
        nefc_t = wp.to_torch(self.d.nefc)                  # constraint rows (vs njmax)
        nacon_t = wp.to_torch(self.d.nacon)                # narrowphase contacts (vs nconmax)
        ncol_t = wp.to_torch(self.d.ncollision)            # broadphase candidate pairs

        pre = self._diag_pre_qvel_absmax
        pre_absmax = float(pre.item()) if pre is not None else float("nan")

        # [BFM-DIAG-NAN] contact penetration at the divergence step (added by Claude).
        # d.contact.dist[:nacon] are the active contacts (neg = penetration depth in m);
        # worldid attributes each contact to a world, so we report the deepest penetration per
        # diverging env. Confirms/refutes the "we fed a penetrating init/reset state" hypothesis.
        try:
            nacon_v = int(wp.to_torch(self.d.nacon).item())
            dist_full = wp.to_torch(self.d.contact.dist)
            wid_full = wp.to_torch(self.d.contact.worldid)
            if nacon_v > 0 and dist_full.numel() >= nacon_v and wid_full.numel() >= nacon_v:
                dist = dist_full[:nacon_v]
                wid = wid_full[:nacon_v]
                gmin = float(dist.min().item())
                pen_by_env = {}
                for eid in k.tolist():
                    msk = wid == eid
                    if bool(msk.any()):
                        pen_by_env[eid] = round(float(dist[msk].min().item()), 5)
                pen_str = (
                    f"global_min_dist={gmin:+.5f} m (neg=penetration), n_active={nacon_v}, "
                    f"bad-env deepest penetration (env:min_dist)={pen_by_env}"
                )
            else:
                pen_str = f"no active contacts at this substep (nacon={nacon_v})"
        except Exception as e:  # noqa -- diagnostic must never crash a run
            pen_str = f"<contact-penetration read err {e}>"

        # [BFM-DIAG-NAN] reset/init timing for the bad envs (added by Claude): how many physics
        # substeps elapsed between the env's last set_*_state_tensor (reset/init) and this NaN.
        try:
            if self._diag_last_set_step is not None:
                since = (self._step_count - self._diag_last_set_step[k]).tolist()
                reset_str = (
                    f"steps_since_last_set={since}  "
                    f"(<= {self._diag_decimation} => diverged within one control step of a "
                    f"reset/init; -1 => never set since probe start)"
                )
            else:
                reset_str = "steps_since_last_set: n/a (no set_* since probe start)"
        except Exception as e:  # noqa -- diagnostic must never crash a run
            reset_str = f"<reset-timing read err {e}>"

        print(
            "\n[BFM-DIAG-NAN] >>> FIRST non-finite sim state after mjw.step  "
            f"(step_count={self._step_count}, probe_checks={self._diag_probe_checks}, "
            f"n_bad={n_bad}/{self.num_envs}, sample_env_ids={k.tolist()})\n"
            f"    PRE-step  |qvel| max (all envs)        = {pre_absmax:+.6e}\n"
            f"    POST-step qacc       [bad envs]        = {_stat('qacc', qacc_t)}\n"
            f"    POST-step qfrc_constraint [bad envs]   = {_stat('qfc', qfc_t)}\n"
            f"    POST-step qfrc_actuator  [bad envs]    = {_stat('qfa', qfa_t)}\n"
            f"    POST-step qvel       [bad envs]        = {_stat('qvel', qvel_t)}\n"
            f"    nacon/world max={_gmax(nacon_t)}  nconmax={self._diag_nconmax}  "
            f"(real contacts; bad-envs={_bad_list(nacon_t)})\n"
            f"    ncollision/world max={_gmax(ncol_t)}  (broadphase candidate pairs)\n"
            f"    nefc/world  max={_gmax(nefc_t)}  njmax={self._diag_njmax}  "
            f"(constraint rows; bad-envs={_bad_list(nefc_t)})\n"
            f"    CONTACT penetration at divergence      = {pen_str}\n"
            f"    RESET timing for bad envs              = {reset_str}\n"
            "[BFM-DIAG-NAN] interpretation:\n"
            "      nacon small but nefc huge  -> API inflating constraints (wrapper bug), not real contacts\n"
            "      nacon huge (~100s)         -> genuine full-body-collision pile-up (reduce geoms / excludes / buffers)\n"
            "      qacc non-finite + finite qfrc_actuator -> mass-matrix / Euler solve diverged\n"
            "      qfrc_constraint non-finite             -> contact/limit constraint solver diverged\n"
            "      nefc > njmax                           -> constraint buffer overflow (dropped rows)\n"
            "      huge PRE-step |qvel|                   -> extreme-state trigger (no dof-vel bound)\n"
            "      deep CONTACT penetration (min_dist << 0)      -> a kept body is through the floor (the lie_down_init / penetrating-init hypothesis)\n"
            "      steps_since_last_set <= control_decimation    -> diverged right after a reset/init (implicates the state WE fed the solver)",
            flush=True,
        )

        self._diag_per_env_breakdown(k)  # [BFM-DIAG-NAN] exact per-env pre-step state + action
        self._diag_save_replay(k)  # [BFM-DIAG-NAN] replayable npz of divergence-onset state for cross-backend replay

    def _diag_save_replay(self, k):
        # [BFM-DIAG-NAN] (added by Claude) save a REPLAYABLE npz of the divergence-onset state for
        # the first bad env, so it can be replayed through diff_sim_io on mujoco_warp AND isaacsim to
        # test whether PhysX also NaNs on that exact (qpos, qvel, ctrl) — the decisive solver-robustness
        # test. With BFM_ZERO_NAN_PROBE_DECIM=1, _diag_pre_qpos/qvel/ctrl are the inputs to the just-
        # stepped substep, i.e. the true divergence onset. Observe-only, fully guarded, opt-in via
        # BFM_ZERO_NAN_REPLAY_OUT (path to write; if unset, this is a no-op print).
        try:
            out = os.environ.get("BFM_ZERO_NAN_REPLAY_OUT", "")
            eid = int(k[0].item())
            pre_qpos = self._diag_pre_qpos[eid].detach().cpu().numpy() if self._diag_pre_qpos is not None else None
            pre_qvel = self._diag_pre_qvel[eid].detach().cpu().numpy() if self._diag_pre_qvel is not None else None
            pre_ctrl = self._diag_pre_ctrl[eid].detach().cpu().numpy() if self._diag_pre_ctrl is not None else None
            if out:
                import numpy as _np
                _np.savez(
                    out,
                    env_id=_np.array(eid),
                    step_count=_np.array(self._step_count),
                    pre_qpos=pre_qpos,   # [nq=36] divergence-onset generalized position (wxyz base quat)
                    pre_qvel=pre_qvel,   # [nv=35] divergence-onset generalized velocity
                    pre_ctrl=pre_ctrl,   # [nu=35] ctrl (action) applied in the diverging substep
                )
                print(f"[BFM-DIAG-NAN] saved REPLAYABLE divergence-onset state to {out} "
                      f"(env_id={eid}, step_count={self._step_count})", flush=True)
            else:
                print(f"[BFM-DIAG-NAN] set BFM_ZERO_NAN_REPLAY_OUT=<path.npz> to dump a replayable "
                      f"divergence-onset state (env_id={eid}, step_count={self._step_count})", flush=True)
        except Exception as e:  # noqa -- the probe must never crash training
            print(f"[BFM-DIAG-NAN] replay-save error (ignored, run continues): {e}", flush=True)

    def _diag_per_env_breakdown(self, k):
        # [BFM-DIAG-NAN] (added by Claude) per-bad-env breakdown of the EXACT pre-step state and
        # action fed into the first-NaN step. Distinguishes a solver gap on an extreme-but-finite
        # state (huge |qvel| / a joint pinned at its limit / huge ctrl) from a DATA bug (garbage
        # qpos/qvel/ctrl from a wrong write-path semantic/dimension). Observe-only, fully guarded.
        try:
            # joint names + range (29 joints == qpos[7:]/qvel[6:]); jnt_range is XML-fixed (DR-proof)
            if getattr(self, "_diag_jnames", None) is None:
                self._diag_jnames = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                                     for j in range(1, self.mj_model.njnt)]
                self._diag_jrange = np.asarray(self.mj_model.jnt_range[1:])  # [29,2] lo,hi
            jnames = self._diag_jnames
            jrange = self._diag_jrange
            qpos_t = wp.to_torch(self.d.qpos)       # [N,36]
            post_qvel_t = wp.to_torch(self.d.qvel)  # [N,35]
            lines = ["[BFM-DIAG-NAN] per-env breakdown (pre-step = state/action fed INTO the diverging step):"]
            for eid in k.tolist():
                pre_q = self._diag_pre_qvel[eid] if self._diag_pre_qvel is not None else None
                pre_c = self._diag_pre_ctrl[eid] if self._diag_pre_ctrl is not None else None
                pre_abs = float(pre_q.abs().max()) if pre_q is not None else float("nan")
                pre_j = int(pre_q.abs().argmax()) if pre_q is not None else -1
                jn = jnames[pre_j - 6] if (pre_q is not None and 6 <= pre_j < 6 + len(jnames)) else "?"
                ctrl_joints = pre_c[6:] if pre_c is not None else None
                ctrl_abs = float(ctrl_joints.abs().max()) if ctrl_joints is not None else float("nan")
                ctrl_jmax = int(ctrl_joints.abs().argmax()) if ctrl_joints is not None else -1
                ctrl_name = jnames[ctrl_jmax] if (ctrl_joints is not None and 0 <= ctrl_jmax < len(jnames)) else "?"
                jq = qpos_t[eid, 7:]                  # [29] joint positions
                near = []
                if jrange is not None and jq.numel() == jrange.shape[0]:
                    for j in range(jq.numel()):
                        lo, hi = float(jrange[j, 0]), float(jrange[j, 1])
                        qv = float(jq[j])
                        if (qv - lo) < 0.10 or (hi - qv) < 0.10:
                            near.append(f"{jnames[j]}={qv:+.3f}[{lo:+.2f},{hi:+.2f}]")
                nan_dofs = int((~torch.isfinite(post_qvel_t[eid])).sum())
                lines.append(
                    f"  env {eid}: PRE|qvel|max={pre_abs:+.4e} @dof[{pre_j}]={jn}; "
                    f"PRE ctrl(joints)|max|={ctrl_abs:+.4e} @ {ctrl_name}; "
                    f"post-NaN-doFs={nan_dofs}/35; near-limit: {', '.join(near) if near else 'none'}"
                )
            print("\n".join(lines), flush=True)
        except Exception as e:  # noqa -- diagnostic must never crash a run
            print(f"[BFM-DIAG-NAN] per-env breakdown error (ignored): {e}", flush=True)

    def _diag_mark_set(self, set_env_ids):
        # [BFM-DIAG-NAN] observe-only (added by Claude): record that these envs' qpos/qvel were
        # (re)written by the env via set_*_state_tensor (reset/init path) at the current physics
        # substep. Never alters sim state. Called from set_actor_root_state_tensor and
        # set_dof_state_tensor so the first-NaN dump can report steps_since_last_set per bad env.
        if not self._diag_probe:
            return
        try:
            if self._diag_last_set_step is None and hasattr(self, "num_envs"):
                self._diag_last_set_step = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.sim_device
                )
            if self._diag_last_set_step is not None:
                self._diag_last_set_step[set_env_ids.reshape(-1)] = self._step_count
        except Exception:  # noqa -- diagnostic must never crash a run
            pass

    # ------------------------------------------------------------------ #
    #  Control
    # ------------------------------------------------------------------ #

    def apply_torques_at_dof(self, torques):
        if isinstance(torques, torch.Tensor):
            torques = torques.contiguous()
        # Convert to warp array (zero-copy if already on CUDA)
        wp_torques = wp.from_torch(torques, dtype=wp.float32)
        if self.freebase:
            # G1's MJCF defines 6 virtual base actuators (fx..tz) before the joint
            # motors, so ctrl = [6 base | 29 joints] and the offset is 6. BDX has no
            # base actuators, so ctrl = [14 joints] and the offset is 0. Derive it as
            # nu - num_dof (works for both; equals 6 for G1, 0 for BDX).
            ctrl_offset = self.mj_model.nu - self.num_dof
            wp.copy(self.d.ctrl[:, ctrl_offset:], wp_torques)
        else:
            wp.copy(self.d.ctrl, wp_torques)

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        if isinstance(set_env_ids, torch.Tensor):
            set_env_ids = set_env_ids.reshape(-1)
        if isinstance(root_states, torch.Tensor):
            root_states = root_states.reshape(-1, 13).clone()
        else:
            root_states = torch.tensor(root_states, dtype=torch.float32, device=self.sim_device).reshape(-1, 13)

        # Caller may pass the FULL buffer (e.g. legged_robot_base.py:321 passes
        # self.target_robot_root_states [num_envs, 13] with a subset of env_ids).
        # Extract only the rows we need.
        if root_states.shape[0] != set_env_ids.shape[0]:
            root_states = root_states[set_env_ids].clone()

        # Convert angular velocity from root_states format back to MuJoCo qvel format.
        # robot_root_states stores quat_rotate(base_quat, qvel[3:6]) for angular velocity,
        # so we must undo that: qvel[3:6] = quat_rotate_inverse(base_quat, root_states[:, 10:13])
        # This matches the MuJoCo backend's convention exactly.
        bq_sub = self.base_quat[set_env_ids]
        root_states[:, 10:13] = quat_rotate_inverse(
            bq_sub, root_states[:, 10:13], w_last=True
        )

        # GPU-direct write via wp.to_torch() (shared memory, zero-copy)
        qpos_t = wp.to_torch(self.d.qpos)  # [N, nq] on CUDA
        qvel_t = wp.to_torch(self.d.qvel)  # [N, nv] on CUDA

        # Position
        qpos_t[set_env_ids, 0:3] = root_states[:, 0:3]
        # Quaternion: root_states xyzw → MuJoCo wxyz
        qpos_t[set_env_ids, 3] = root_states[:, 6]   # w
        qpos_t[set_env_ids, 4] = root_states[:, 3]   # x
        qpos_t[set_env_ids, 5] = root_states[:, 4]   # y
        qpos_t[set_env_ids, 6] = root_states[:, 5]   # z
        # Velocity
        qvel_t[set_env_ids, 0:6] = root_states[:, 7:13]

        # [BFM-DIAG-NAN] observe-only: record that these envs were (re)initialized at this substep
        self._diag_mark_set(set_env_ids)

        # Invalidate CUDA graph after state changes
        self._graph = None

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        if isinstance(dof_states, np.ndarray):
            dof_states = torch.from_numpy(dof_states).to(self.sim_device)
        elif not isinstance(dof_states, torch.Tensor):
            dof_states = torch.tensor(dof_states, dtype=torch.float32, device=self.sim_device)

        qpos_t = wp.to_torch(self.d.qpos)
        qvel_t = wp.to_torch(self.d.qvel)

        qpos_t[set_env_ids, 7:] = dof_states[set_env_ids, :, 0]
        qvel_t[set_env_ids, 6:] = dof_states[set_env_ids, :, 1]

        # [BFM-DIAG-NAN] observe-only: record that these envs were (re)initialized at this substep
        self._diag_mark_set(set_env_ids)

        self._graph = None

    # ------------------------------------------------------------------ #
    #  Properties — all return torch.Tensor shape [num_envs, ...]
    # ------------------------------------------------------------------ #

    @property
    def dof_state(self):
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    @property
    def robot_root_states(self):
        # [num_envs, 13]: pos(3) + quat_xyzw(4) + linvel(3) + angvel_body(3)
        qpos = wp.to_torch(self.d.qpos)
        qvel = wp.to_torch(self.d.qvel)

        pos = qpos[:, 0:3]                            # [N, 3]
        quat_wxyz = qpos[:, 3:7]                       # [N, 4] wxyz
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]        # → xyzw

        lin_vel = qvel[:, 0:3]                          # [N, 3]
        ang_vel_world = qvel[:, 3:6]                    # [N, 3] angular vel in world frame
        # Convert to body frame
        ang_vel_body = quat_rotate(quat_xyzw, ang_vel_world, w_last=True)

        return torch.cat([pos, quat_xyzw, lin_vel, ang_vel_body], dim=-1)

    @property
    def all_root_states(self):
        return self.robot_root_states

    @property
    def base_quat(self):
        qpos = wp.to_torch(self.d.qpos)
        return qpos[:, 3:7][:, [1, 2, 3, 0]]  # wxyz → xyzw

    @property
    def dof_pos(self):
        qpos = wp.to_torch(self.d.qpos)
        return qpos[:, 7:]  # [N, 29]

    @property
    def dof_vel(self):
        qvel = wp.to_torch(self.d.qvel)
        return qvel[:, 6:]  # [N, 29]

    @property
    def contact_forces(self):
        # cfrc_ext: external forces on bodies [nworld, nbody, 6]
        # Last 3 components are force (torque, force)
        cfrc = wp.to_torch(self.d.cfrc_ext)
        return cfrc[:, 1:, 3:6]  # [N, num_bodies, 3] — skip world body

    @property
    def _rigid_body_pos(self):
        xpos = wp.to_torch(self.d.xpos)
        return xpos[:, self.body_id, :]  # [N, num_bodies, 3]

    @property
    def _rigid_body_rot(self):
        xquat = wp.to_torch(self.d.xquat)
        return xquat[:, self.body_id, :][:, :, [1, 2, 3, 0]]  # wxyz → xyzw

    @property
    def _rigid_body_vel(self):
        cvel = wp.to_torch(self.d.cvel)
        return cvel[:, self.body_id, 3:6]  # [N, num_bodies, 3] linear velocity

    @property
    def _rigid_body_ang_vel(self):
        cvel = wp.to_torch(self.d.cvel)
        return cvel[:, self.body_id, 0:3]  # [N, num_bodies, 3] angular velocity

    # ------------------------------------------------------------------ #
    #  DOF limits
    # ------------------------------------------------------------------ #

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device)

        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_cfg.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]

            # Soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.config.rewards.reward_limit.soft_dof_pos_limit

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    # ------------------------------------------------------------------ #
    #  Misc
    # ------------------------------------------------------------------ #

    def find_rigid_body_indice(self, body_name):
        if body_name not in self.body_names:
            raise ValueError(f"Rigid body '{body_name}' not found in model.")
        return self.body_names.index(body_name)

    def setup_viewer(self):
        # MuJoCo Warp doesn't have a built-in interactive viewer for batched sims.
        # For visualization, use the standard MuJoCo viewer with a single-world copy.
        logger.warning("MuJoCoWarp: interactive viewer not supported for batched simulation")

    def render(self, sync_frame_time=True):
        pass  # No rendering in batched GPU mode