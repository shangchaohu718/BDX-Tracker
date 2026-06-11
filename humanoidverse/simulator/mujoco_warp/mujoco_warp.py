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

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def setup(self):
        hv_root = Path(__file__).parents[2]
        self.model_path = str(hv_root / "data/robots/g1/scene_29dof_freebase_mujoco.xml")

        # Load standard MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.mj_model.opt.timestep = 1.0 / self.simulator_config.sim.fps
        self.sim_dt = self.mj_model.opt.timestep

        # Store defaults for domain randomization
        self._default_dof_frictionloss = self.mj_model.dof_frictionloss.copy()
        self._default_body_mass = self.mj_model.body_mass.copy()
        self._default_geom_friction = self.mj_model.geom_friction.copy()

        # These are created lazily in create_envs when we know num_envs
        self.m = None  # mjw.Model
        self.d = None  # mjw.Data
        self.mjm = None  # mujoco.MjModel (possibly modified by domain rand)

        logger.info(f"MuJoCoWarp setup complete. Model: {self.model_path}")

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

        # For 29-DOF model, exclude hand bodies
        if "29" in self.model_path:
            for b in range(self.mj_model.nbody):
                name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
                if name and "hand" in name:
                    if name in self.body_names:
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

        if self.domain_rand_config.get("randomize_base_com", False):
            torso_id = self.mj_model.body("torso_link").id
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

    # ------------------------------------------------------------------ #
    #  Control
    # ------------------------------------------------------------------ #

    def apply_torques_at_dof(self, torques):
        if isinstance(torques, torch.Tensor):
            torques = torques.contiguous()
        # Convert to warp array (zero-copy if already on CUDA)
        wp_torques = wp.from_torch(torques, dtype=wp.float32)
        if self.freebase:
            # Skip first 6 ctrl entries (free base), write joint torques
            wp.copy(self.d.ctrl[:, 6:], wp_torques)
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