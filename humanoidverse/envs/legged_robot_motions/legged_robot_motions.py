import torch
import numpy as np
from pathlib import Path
import os
import random
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.torch_utils import (
    my_quat_rotate,
    quat_to_tan_norm, 
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_conjugate,
    quat_to_angle_axis,
    quat_from_angle_axis,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw
)
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from loguru import logger
from collections import OrderedDict


class LeggedRobotMotions(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        self.debug_viz = False
        
        super().__init__(config, device)
        self._init_motion_lib()
        self._init_motion_extend()
        self._init_tracking_config()
        # read from robot.motion (where the yaml key lives), not env root —
        # the old path never matched and contact was silently excluded
        self.use_contact_in_obs_max = self.config.robot.motion.get(
            "use_contact_in_obs_max", False)
        self.init_done = True
        self.debug_viz = True
        self.viewer_focus = False

        if self.config.use_teleop_control:
            self.teleop_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            self.node = Node("legged_motions")
            self.teleop_sub = self.node.create_subscription(Float64MultiArray, "vision_pro_data", self.teleop_callback, 1)
        
        if not self.headless:
            if self.simulator.simulator_config.name  == "isaacsim":
                self.simulator.add_keyboard_callback("R", self.reset_all)
            elif self.simulator.simulator_config.name == "isaacgym":
                from isaacgym import gymtorch, gymapi, gymutil
                self.simulator.add_keyboard_callback(gymapi.KEY_R, self.reset_all, "Reset")

    def _init_motion_lib(self):
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
            
        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
        res = self._resample_motion_time_and_ids(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
        
    def _init_motion_extend(self):
        if "extend_config" in self.config.robot.motion and len(self.config.robot.motion.extend_config) > 0:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                # extend_parent_ids.append(self.simulator.find_rigid_body_indice(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)
        else:
            self.num_extend_bodies = 0
            
        self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.marker_coords = torch.zeros(self.num_envs, 
                                        self.num_bodies + self.num_extend_bodies, 
                                        3, 
                                        dtype=torch.float, 
                                        device=self.device, 
                                        requires_grad=False) # extend

    def _init_buffers(self):
        super()._init_buffers()
        self.vr_3point_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long).to(self.device) 
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        
    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        self._resample_motion_time_and_ids(env_ids) # need to resample before reset root states

    def _update_tasks_callback(self):
        super()._update_tasks_callback() # Push robots. 
        
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()

    def set_is_evaluating(self, global_rank = 0):
        super().set_is_evaluating(global_rank)
        self.begin_seq_motion_samples(global_rank)
        
    def set_is_training(self):
        self.is_evaluating = False
        self.resample_motion()

    def _check_termination(self):
        super()._check_termination()
        if self.config.termination.terminate_when_motion_far:
            
            if self.is_evaluating:
                reset_buf_motion_far = torch.norm(self.dif_global_body_pos, dim=-1).mean(dim=-1) > 0.5 # Evaluation distance threshold
            else:
                reset_buf_motion_far = torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
            
            reset_buf_motion_far[self.push_robot_recovery_counter > 0] = False #
                
            self.reset_buf |= reset_buf_motion_far
            
            # log current motion far threshold
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        # [BFM-REF-CURRICULUM] BDX diagnostic/canonical exploration curriculum.
        #
        # This environment already resets to a motion-library state.  The
        # missing piece was *frequency*: the normal 10 s episode lets an
        # untrained policy spend hundreds of steps far away from the expert
        # manifold, which teaches the discriminator the spurious rule
        # "policy motion == fake".  While enabled, start with short anchored
        # rollouts and linearly restore the original horizon.  All envs share
        # the same horizon, preserving the vector wrapper's synchronized-reset
        # invariant when history_actor is enabled.
        import os as _os
        if _os.environ.get("BFM_REF_STATE_CURRICULUM", "0") == "1" and not self.is_evaluating:
            total_env_steps = max(1, int(_os.environ.get("BFM_REF_CURRICULUM_STEPS", "300000")))
            progress = min(1.0, (self.common_step_counter * self.num_envs) / total_env_steps)
            horizon_start = max(2, int(_os.environ.get("BFM_REF_HORIZON_START", "8")))
            horizon_end = max(horizon_start, int(_os.environ.get("BFM_REF_HORIZON_END", str(self.max_episode_length))))
            horizon = int(round(horizon_start + progress * (horizon_end - horizon_start)))
            self.time_out_buf |= self.episode_length_buf >= horizon
            self.log_dict["ref_curriculum_progress"] = torch.tensor(progress, dtype=torch.float32)
            self.log_dict["ref_curriculum_horizon"] = torch.tensor(float(horizon), dtype=torch.float32)
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_buf |= current_time > self.motion_len

    def _resample_motion_time_and_ids(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])

        # [BFM-REF-CURRICULUM] Prefer genuinely dynamic reference frames early
        # in training.  Merely resetting to an arbitrary expert frame is not
        # sufficient: roughly one fifth of the corpus is stand/pose data, and
        # the remaining rollout can immediately leave the dynamic manifold.
        # Rejection sampling is local to reset and uses the same motion library
        # that subsequently constructs root/DOF target states.
        import os as _os
        if _os.environ.get("BFM_REF_STATE_CURRICULUM", "0") == "1" and not self.is_evaluating:
            total_env_steps = max(1, int(_os.environ.get("BFM_REF_CURRICULUM_STEPS", "300000")))
            progress = min(1.0, (self.common_step_counter * self.num_envs) / total_env_steps)
            prob_start = float(_os.environ.get("BFM_REF_DYNAMIC_PROB_START", "1.0"))
            prob_end = float(_os.environ.get("BFM_REF_DYNAMIC_PROB_END", "0.2"))
            dynamic_prob = prob_start + progress * (prob_end - prob_start)
            dynamic_prob = max(0.0, min(1.0, dynamic_prob))
            activity_min = float(_os.environ.get("BFM_REF_ACTIVITY_MIN", "2.0"))
            max_tries = max(1, int(_os.environ.get("BFM_REF_MAX_TRIES", "12")))

            force_dynamic = torch.rand(len(env_ids), device=self.device) < dynamic_prob

            # [BFM-REF-Z-ALIGN] When a simulator-domain expert pool is supplied
            # by Train, select the physical reset state from an exact pool
            # window.  Train later encodes z from the same window.  Without this
            # coupling, frequent expert resets pair motion A with an independent
            # latent B and can worsen the conditional discriminator signal.
            pool = getattr(self, "_bfm_ref_pool_payload", None)
            if _os.environ.get("BFM_REF_Z_ALIGNED", "0") == "1" and pool is not None:
                if not hasattr(self, "_bfm_ref_pool_rows"):
                    self._bfm_ref_pool_rows = torch.full(
                        (self.num_envs,), -1, dtype=torch.long, device=self.device
                    )
                    self._bfm_ref_pool_offsets = torch.zeros(
                        self.num_envs, dtype=torch.long, device=self.device
                    )
                    self._bfm_ref_aligned_mask = torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device
                    )
                    # The runtime motion library holds a sampled subset (local
                    # ids 0..N-1), while pool metadata uses global dataset ids.
                    # Build an exact global->(local,pool-row) map once.
                    pool_row_by_global = {
                        int(meta["motion_id"]): row for row, meta in enumerate(pool["meta"])
                    }
                    current_globals = self._motion_lib._curr_motion_ids.detach().cpu().tolist()
                    eligible = [
                        (local, pool_row_by_global[int(global_id)])
                        for local, global_id in enumerate(current_globals)
                        if int(global_id) in pool_row_by_global
                    ]
                    if not eligible:
                        raise RuntimeError("BFM_REF_Z_ALIGNED found no overlap between runtime motions and pool")
                    self._bfm_ref_eligible_local = [item[0] for item in eligible]
                    self._bfm_ref_eligible_rows = [item[1] for item in eligible]
                self._bfm_ref_aligned_mask[env_ids] = False
                # Alignment is a semantic invariant for every reset episode;
                # `force_dynamic` only controls curriculum sampling bias.  At
                # late training the other rows may sample stand/pose windows,
                # but their state and z must still come from the same window.
                aligned_local = torch.arange(len(env_ids), device=self.device)
                if aligned_local.numel() > 0:
                    aligned_envs = env_ids[aligned_local.to(env_ids.device)]
                    chosen_rows_list = []
                    chosen_offsets = []
                    seq_len = int(_os.environ.get("BFM_REF_Z_SEQ", "8"))
                    for _row_i in range(aligned_local.numel()):
                        require_dynamic = bool(force_dynamic[_row_i].item())
                        # Select a genuinely dynamic pool frame.  BDX state is
                        # [14 dof_pos, 14 dof_vel, gravity(3), base_ang_vel(3)].
                        # The pool was built from the simulator pipeline, so
                        # this gate uses the exact data later encoded into z.
                        best_row, best_offset, best_activity = 0, 0, -1.0
                        for _attempt in range(max_tries):
                            eligible_i = int(torch.randint(0, len(self._bfm_ref_eligible_rows), (1,)).item())
                            row = self._bfm_ref_eligible_rows[eligible_i]
                            length = len(pool["episodes"][row]["truncated"])
                            max_offset = max(0, length - seq_len)
                            offset = int(torch.randint(0, max_offset + 1, (1,)).item()) if max_offset else 0
                            state_row = pool["episodes"][row]["observation"]["state"][offset]
                            activity_row = float(state_row[14:28].float().norm().item())
                            if activity_row > best_activity:
                                best_row, best_offset, best_activity = row, offset, activity_row
                            if (not require_dynamic) or activity_row >= activity_min:
                                break
                        chosen_rows_list.append(best_row)
                        chosen_offsets.append(best_offset)
                    chosen_rows = torch.tensor(chosen_rows_list, dtype=torch.long, device=self.device)
                    chosen_offsets = torch.tensor(chosen_offsets, dtype=torch.long, device=self.device)
                    meta_rows = chosen_rows.detach().cpu().tolist()
                    global_to_local = {
                        int(global_id): local
                        for local, global_id in enumerate(self._motion_lib._curr_motion_ids.detach().cpu().tolist())
                    }
                    motion_ids = torch.tensor(
                        [global_to_local[int(pool["meta"][row]["motion_id"])] for row in meta_rows],
                        dtype=torch.long, device=self.device,
                    )
                    start_times = torch.tensor(
                        [float(pool["meta"][row]["t0"]) for row in meta_rows],
                        dtype=torch.float32, device=self.device,
                    ) + chosen_offsets.float() * self.dt
                    self.motion_ids[aligned_envs] = motion_ids
                    self.motion_len[aligned_envs] = self._motion_lib.get_motion_length(motion_ids)
                    self.motion_start_times[aligned_envs] = start_times
                    self._bfm_ref_pool_rows[aligned_envs] = chosen_rows
                    self._bfm_ref_pool_offsets[aligned_envs] = chosen_offsets
                    self._bfm_ref_aligned_mask[aligned_envs] = True

            unresolved = force_dynamic.clone()
            accepted_activity = torch.zeros(len(env_ids), device=self.device)
            for _ in range(max_tries):
                if not unresolved.any():
                    break
                local = unresolved.nonzero(as_tuple=False).flatten()
                chosen_envs = env_ids[local.to(env_ids.device)]
                # The first pass evaluates the already sampled frames.  Later
                # passes replace only unresolved rows, avoiding biasing rows
                # that have already met the activity gate.
                if _ > 0 and not (
                    _os.environ.get("BFM_REF_Z_ALIGNED", "0") == "1" and pool is not None
                ):
                    self.motion_ids[chosen_envs] = self._motion_lib.sample_motions(len(chosen_envs))
                    self.motion_len[chosen_envs] = self._motion_lib.get_motion_length(self.motion_ids[chosen_envs])
                    self.motion_start_times[chosen_envs] = self._motion_lib.sample_time(self.motion_ids[chosen_envs])
                state = self._motion_lib.get_motion_state(
                    self.motion_ids[chosen_envs], self.motion_start_times[chosen_envs]
                )
                activity = state["dof_vel"].norm(dim=-1)
                accepted_activity[local] = activity
                unresolved[local[activity >= activity_min]] = False
                # Pool-aligned rows must not be independently resampled: doing
                # so would invalidate the episode/offset used to construct z.
                if _os.environ.get("BFM_REF_Z_ALIGNED", "0") == "1" and pool is not None:
                    break

            self.log_dict["ref_curriculum_dynamic_prob"] = torch.tensor(dynamic_prob, dtype=torch.float32)
            self.log_dict["ref_curriculum_dynamic_fraction"] = (
                force_dynamic & (accepted_activity >= activity_min)
            ).float().mean().detach().cpu()
            if force_dynamic.any():
                self.log_dict["ref_curriculum_forced_activity"] = accepted_activity[force_dynamic].mean().detach().cpu()
            
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # offset = self.env_origins
        # motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

    def begin_seq_motion_samples(self, global_rank = 0):
        # For evaluation
        print("Loading motions for evaluation")
        self.start_idx = 0 + global_rank * self.num_envs
        self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
        self.reset_all()
        
    def resample_motion(self):
        # import ipdb; ipdb.set_trace()
        self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
        self.reset_all()
        
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        
        offset = self.env_origins
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

        self.ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        self.ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        self.ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
        
        env_batch_size = self.simulator._rigid_body_pos.shape[0]
        num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]
        
        ################### EXTEND Rigid body POS #####################
        if self.num_extend_bodies > 0:
            rotated_pos_in_parent = my_quat_rotate(
                self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                self.extend_body_pos_in_parent.reshape(-1, 3)
            )
            extend_curr_pos = my_quat_rotate(
                self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                rotated_pos_in_parent
            ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
            self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)

            ################### EXTEND Rigid body Rotation #####################
            extend_curr_rot = quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                        self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                        w_last=True).view(self.num_envs, -1, 4)
            self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, extend_curr_rot], dim=1)
            
            ################### EXTEND Rigid Body Angular Velocity #####################
            self._rigid_body_ang_vel_extend = torch.cat([self.simulator._rigid_body_ang_vel, self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)
        
            ################### EXTEND Rigid Body Linear Velocity #####################
            self._rigid_body_ang_vel_global = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
            angular_velocity_contribution = torch.cross(self._rigid_body_ang_vel_global, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
            extend_curr_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
            self._rigid_body_vel_extend = torch.cat([self.simulator._rigid_body_vel, extend_curr_vel], dim=1)
        else:
            self._rigid_body_vel_extend = self.simulator._rigid_body_vel
            self._rigid_body_ang_vel_extend = self.simulator._rigid_body_ang_vel
            self._rigid_body_pos_extend = self.simulator._rigid_body_pos
            self._rigid_body_rot_extend = self.simulator._rigid_body_rot

        #### Heading quantatives ####
        heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)

        heading_rot = calc_heading_quat(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)
        
        
        if self.config.get("local_ref_motion", False):
            ref_heading_inv_rot = calc_heading_quat_inv(self.ref_body_rot_extend[:, 0], w_last=True)
            ref_heading_inv_rot_expand = ref_heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)
            
            self.ref_body_pos_extend = self.ref_body_pos_extend - self.ref_body_pos_extend[:, 0:1]
            self.ref_body_pos_extend = my_quat_rotate(ref_heading_inv_rot_expand, self.ref_body_pos_extend.view(-1, 3)) # remove heading from reference motion
            self.ref_body_pos_extend = my_quat_rotate(heading_rot_expand, self.ref_body_pos_extend.view(-1, 3)).view(env_batch_size, -1, 3) # add heading current to reference motion
            self.ref_body_pos_extend = self.ref_body_pos_extend + self._rigid_body_pos_extend[:, 0:1]
            
            self.ref_body_rot_extend = quat_mul(heading_rot_expand, quat_mul(ref_heading_inv_rot_expand, self.ref_body_rot_extend.view(-1, 4), w_last=True), w_last=True).view(env_batch_size, -1, 4)
            
            self.ref_body_vel_extend = my_quat_rotate(ref_heading_inv_rot_expand, self.ref_body_vel_extend.view(-1, 3))
            self.ref_body_vel_extend = my_quat_rotate(heading_rot_expand, self.ref_body_vel_extend.view(-1, 3)).view(env_batch_size, -1, 3)
            
            self.ref_body_ang_vel_extend = my_quat_rotate(ref_heading_inv_rot_expand, self.ref_body_ang_vel_extend.view(-1, 3))
            self.ref_body_ang_vel_extend = my_quat_rotate(heading_rot_expand, self.ref_body_ang_vel_extend.view(-1, 3)).view(env_batch_size, -1, 3)
        
        
        self.ref_body_pos_extend = self.ref_body_pos_extend # for visualization and analysis
        self.ref_body_rot_extend = self.ref_body_rot_extend 

        ################### Compute differences #####################
        
        ## diff compute - kinematic position
        self.dif_global_body_pos = self.ref_body_pos_extend - self._rigid_body_pos_extend
        
        # import ipdb; ipdb.set_trace()
        ## diff compute - kinematic rotation
        self.dif_global_body_rot = quat_mul(self.ref_body_rot_extend, quat_conjugate(self._rigid_body_rot_extend, w_last=True), w_last=True)
        
        ## diff compute - kinematic velocity
        self.dif_global_body_vel = self.ref_body_vel_extend - self._rigid_body_vel_extend
        ## diff compute - kinematic angular velocity
        
        self.dif_global_body_ang_vel = self.ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend
        # ang_vel_reward = self._reward_teleop_body_ang_velocity_extend()


        ## diff compute - kinematic joint position
        self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos
        ## diff compute - kinematic joint velocity
        self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel


        # marker_coords for visualization
        self.marker_coords[:] = self.ref_body_pos_extend.reshape(B, -1, 3)

        if "dif_local_rigid_body_pos" in self.all_obs_name:
            dif_global_body_pos_for_obs_compute = self.ref_body_pos_extend.view(env_batch_size, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, -1, 3)
            dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_pos_for_obs_compute.view(-1, 3))
            self._obs_dif_local_rigid_body_pos = dif_local_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        
        if "dif_local_rigid_body_rot" in self.all_obs_name:
            dif_global_body_rot_for_obs_compute = quat_mul(self.ref_body_rot_extend, quat_conjugate(self._rigid_body_rot_extend, w_last=True), w_last=True)
            dif_local_body_rot_flat =  quat_mul(quat_mul(heading_inv_rot_expand.view(-1, 4), dif_global_body_rot_for_obs_compute.view(-1, 4), w_last=True), heading_rot_expand.view(-1, 4), w_last=True) 
            self._obs_dif_local_rigid_body_rot = quat_to_tan_norm(dif_local_body_rot_flat, w_last=True).view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
            
        if "dif_local_rigid_body_vel" in self.all_obs_name:
            dif_global_body_vel_for_obs_compute = self.ref_body_vel_extend.view(env_batch_size, -1, 3) - self._rigid_body_vel_extend.view(env_batch_size, -1, 3)
            dif_local_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_vel_for_obs_compute.view(-1, 3))
            self._obs_dif_local_rigid_body_vel = dif_local_body_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        
        if "dif_local_rigid_body_ang_vel" in self.all_obs_name:
            dif_global_body_ang_vel_for_obs_compute = self.ref_body_ang_vel_extend.view(env_batch_size, -1, 3) - self._rigid_body_ang_vel_extend.view(env_batch_size, -1, 3)
            dif_local_body_ang_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_ang_vel_for_obs_compute.view(-1, 3))
            self._obs_dif_local_rigid_body_ang_vel = dif_local_body_ang_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        
        if "local_ref_rigid_body_pos" in self.all_obs_name:
            global_ref_rigid_body_pos = self.ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)  # preserves the body position
            local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
            self._obs_local_ref_rigid_body_pos = local_ref_rigid_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
            
        if "local_ref_rigid_body_rot" in self.all_obs_name:
            local_ref_rigid_body_rot_flat = quat_mul(heading_inv_rot_expand.view(-1, 4), self.ref_body_rot_extend.view(-1, 4), w_last=True)
            self._obs_local_ref_rigid_body_rot = quat_to_tan_norm(local_ref_rigid_body_rot_flat, w_last=True).view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        if "local_ref_rigid_body_vel" in self.all_obs_name:
            global_ref_body_vel = self.ref_body_vel_extend.view(env_batch_size, -1, 3)
            local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))
            self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
            
        if "local_ref_rigid_body_ang_vel" in self.all_obs_name:
            global_ref_body_ang_vel = self.ref_body_ang_vel_extend.view(env_batch_size, -1, 3)
            local_ref_rigid_body_ang_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_ang_vel.view(-1, 3))
            self._obs_local_ref_rigid_body_ang_vel = local_ref_rigid_body_ang_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
            
            
        ######################VR 3 point ########################
        if not self.config.use_teleop_control:
            ref_vr_3point_pos = self.ref_body_pos_extend.view(env_batch_size, -1, 3)[:, self.motion_tracking_id, :]
        else:
            ref_vr_3point_pos = self.teleop_marker_coords
        
        vr_2root_pos = (ref_vr_3point_pos - self.simulator.robot_root_states[:, 0:3].view(env_batch_size, 1, 3))
        heading_inv_rot_vr = heading_inv_rot.repeat(len(self.motion_tracking_id),1)
        self._obs_vr_3point_pos = my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(env_batch_size, -1)
        
        #################### Deepmimic phase ###################### 
        if "ref_motion_phase" in self.all_obs_name:
            self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
            self._ref_motion_phase = motion_times / self._ref_motion_length
            if not (torch.all(self._ref_motion_phase >= 0) and torch.all(self._ref_motion_phase <= 1.05)): # hard coded 1.05 because +1 will exceed 1
                max_phase = self._ref_motion_phase.max()
                # import ipdb; ipdb.set_trace()
            self._ref_motion_phase = self._ref_motion_phase.unsqueeze(1)
            
        if "max_local_self" in self.all_obs_name:
            with torch.no_grad():
                if self.use_contact_in_obs_max:
                    contact_binary = self.foot_contact_detect(self._rigid_body_pos_extend, self._rigid_body_vel_extend)
                    max_local_self_dict = compute_humanoid_observations_max_with_contact(self._rigid_body_pos_extend, self._rigid_body_rot_extend, self._rigid_body_vel_extend, self._rigid_body_ang_vel_extend, True, self.config.obs.get("root_height_obs", True), contact_binary=contact_binary)
                else:
                    max_local_self_dict = compute_humanoid_observations_max(self._rigid_body_pos_extend, self._rigid_body_rot_extend, self._rigid_body_vel_extend, self._rigid_body_ang_vel_extend, True, self.config.obs.get("root_height_obs", True))
                self._max_local_self = torch.cat([v for v in max_local_self_dict.values()], dim=-1)
        
        if "max_local_ref" in self.all_obs_name:
            with torch.no_grad():
                if self.use_contact_in_obs_max:
                    contact_binary = self.foot_contact_detect(self.ref_body_pos_extend, self.ref_body_vel_extend)
                    max_local_ref_dict = compute_humanoid_observations_max_with_contact(self.ref_body_pos_extend, self.ref_body_rot_extend, self.ref_body_vel_extend, self.ref_body_ang_vel_extend, True, True, self.feet_indices, contact_binary=contact_binary)
                else:
                    max_local_ref_dict = compute_humanoid_observations_max(self.ref_body_pos_extend, self.ref_body_rot_extend, self.ref_body_vel_extend, self.ref_body_ang_vel_extend, True, True, self.feet_indices)
                self._max_local_ref = torch.cat([v for v in max_local_ref_dict.values()], dim=-1)
                
        # print(f"ref_motion_phase: {self._ref_motion_phase[0].item():.2f}")
        # print(f"ref_motion_length: {self._ref_motion_length[0].item():.2f}")
        self._log_motion_tracking_info()
    
    def _compute_reward(self):
        super()._compute_reward()
        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()

    def _log_motion_tracking_info(self):
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        joint_pos_diff = self.dif_joint_angles

        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()

        self.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        self.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm
        # upper/lower body split is G1-specific (upper/lower_body_link); robots without
        # those links (e.g. BDX: has_upper_body_dof=False) skip the two extra metrics
        if hasattr(self, "upper_body_id") and hasattr(self, "lower_body_id"):
            upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
            lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
            self.log_dict["upper_body_diff_norm"] = upper_body_diff.norm(dim=-1).mean()
            self.log_dict["lower_body_diff_norm"] = lower_body_diff.norm(dim=-1).mean()
    
    def _draw_debug_vis(self):
        if self.config.simulator.config.name in ('mujoco', 'mujoco_warp'):
            return
        if not self.headless:
            self.simulator.clear_lines()
        self._refresh_sim_tensors()
        
        if self.config.simulator.config.name == 'isaacsim':
            scales = torch.ones_like(self.marker_coords) * 0.5
            self.simulator.draw_spheres_batch(self.marker_coords.reshape(-1, 3), scales = scales.view(-1, 3))
        else:
            for env_id in range(self.num_envs):
                if not self.config.use_teleop_control:
                    # draw marker joints
                    for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                        if self.config.robot.motion.visualization.customize_color:
                            color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                        else:
                            color_inner = (0.3, 0.3, 0.3)
                        color_inner = tuple(color_inner)

                        self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)


                else:
                    # draw teleop joints
                    for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
                        self.simulator.draw_sphere(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)

        if self.viewer_focus:
            root_pos = self.simulator._rigid_body_pos[0, 0]
            eye = root_pos + torch.tensor([2, 2, 1], device=self.device)
            self.simulator.viewer.update_view_location(eye.cpu(), root_pos.cpu())

    def _reset_root_states(self, env_ids, target_state=None):
        # reset root states according to the reference motion
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.custom_origins: # trimesh
            self.target_robot_root_states[env_ids, :3] = target_state[..., 0]
            self.target_robot_root_states[env_ids, 3:7] = target_state[..., 1]
            self.target_robot_root_states[env_ids, 7:10] = target_state[..., 2]
            self.target_robot_root_states[env_ids, 10:13] = target_state[..., 3]
        elif target_state is not None:
            # Same logic as in legged_robot_base (reset to this target state)
            self.target_robot_root_states[env_ids] = target_state
            self.target_robot_root_states[env_ids, :3] += self.env_origins[env_ids]
        else:
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins
            motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

            root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
            root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
            root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
            root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

            ref_root_pos = motion_res['root_pos'][env_ids]
            ref_root_rot = motion_res['root_rot'][env_ids]
            ref_root_vel = motion_res['root_vel'][env_ids]
            ref_root_ang_vel = motion_res['root_ang_vel'][env_ids]
            
            if self.config.get("lie_down_init", False):
                prob = getattr(self.config, "lie_down_init_prob", 0)  # default 50% probability

                # Generate a boolean mask of environments that will lie down
                mask = (torch.rand(len(env_ids), device=ref_root_rot.device) < prob)

                # Adjust root position: set z-coordinate to 0.5 for lying-down cases
                ref_root_pos = motion_res['root_pos'][env_ids].clone()
                ref_root_pos[mask, 2] = 0.5

                # Define a quaternion rotation: rotate -90° around the X-axis (standing -> lying on back)

                sign = 1 if random.random() < 0.5 else -1
                rot_quat = quat_from_angle_axis(
                    torch.tensor(sign * (-torch.pi/2), device=ref_root_rot.device),
                    torch.tensor([1.0, 0.0, 0.0], device=ref_root_rot.device),
                    w_last=True
                )

                # Apply the extra rotation only to environments selected by the mask
                ref_root_rot[mask] = quat_mul(
                    rot_quat.expand_as(ref_root_rot[mask]),
                    ref_root_rot[mask],
                    w_last=True
                )


            self.target_robot_root_states[env_ids, :3] = ref_root_pos + torch.randn_like(ref_root_pos) * root_pos_noise
            if self.config.simulator.config.name == 'isaacgym':
                self.target_robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(ref_root_rot.shape[0], root_rot_noise), ref_root_rot, w_last=True)
            elif self.config.simulator.config.name == 'isaacsim':
                self.target_robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(ref_root_rot.shape[0], root_rot_noise), ref_root_rot, w_last=True))
            elif self.config.simulator.config.name == 'genesis':
                self.target_robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(ref_root_rot.shape[0], root_rot_noise), ref_root_rot, w_last=True)
            elif self.config.simulator.config.name == 'mujoco':
                self.target_robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(ref_root_rot.shape[0], root_rot_noise), ref_root_rot, w_last=True)
            elif self.config.simulator.config.name == 'mujoco_warp':
                self.target_robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(ref_root_rot.shape[0], root_rot_noise), ref_root_rot, w_last=True)
            else:
                raise NotImplementedError
            self.target_robot_root_states[env_ids, 7:10] = ref_root_vel + torch.randn_like(ref_root_vel) * root_vel_noise
            self.target_robot_root_states[env_ids, 10:13] = ref_root_ang_vel + torch.randn_like(ref_root_ang_vel) * root_ang_vel_noise


    def small_random_quaternions(self, n, max_angle):
        axis = torch.randn((n, 3), device=self.device)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
        angles = max_angle * torch.rand((n, 1), device=self.device)
        
        # Convert angle-axis to quaternion
        sin_half_angle = torch.sin(angles / 2)
        cos_half_angle = torch.cos(angles / 2)
        
        q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)  
        return q

    def _reset_dofs(self, env_ids, target_state=None):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        if target_state is not None:
            self.target_robot_dof_state[env_ids, :, 0] = target_state[..., 0]
            self.target_robot_dof_state[env_ids, :, 1] = target_state[..., 1]
        else:   
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins
            motion_res = self._motion_lib.get_motion_state(self.motion_ids[env_ids], motion_times[env_ids], offset=offset[env_ids])

            dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
            dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
            dof_pos = motion_res['dof_pos'].clone()
            dof_vel = motion_res['dof_vel'].clone()
            
            target_dof_pos = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
            target_dof_vel = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise
            self.target_robot_dof_state[env_ids, :, 0] = target_dof_pos
            self.target_robot_dof_state[env_ids, :, 1] = target_dof_vel
        


    def _post_physics_step(self):
        super()._post_physics_step()

    # ############################################################
    
    def _get_obs_max_local_self(self):
        return self._max_local_self

    ######################### Observations #########################
    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    ###############################################################

    def _reward_penalty_undesired_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        undesired_contact = torch.any(torch.abs(self.simulator.contact_forces[:, self.penalised_contact_indices, :]) > 1, dim=(1, 2))
        # Penalize undesired contact
        # import ipdb; ipdb.set_trace()
        res[undesired_contact] = 1.0
        # print(res)
        return res
    
    def _reward_penalty_ankle_roll(self):
        # Compute the penalty for ankle roll.
        # [BDX-NOTE] BDX has a single ankle DOF per foot (no roll/pitch split),
        # so indices[1:2] is an empty slice and this penalty is identically 0.
        # Kept structurally correct for G1-style feet; for single-DOF ankles the
        # term is not applicable and its coefficient should stay 0 (external
        # review ruling 2026-08-21 — do not silently remap it to the only DOF,
        # that would change reward semantics).
        if len(self.left_ankle_dof_indices) < 2 or len(self.right_ankle_dof_indices) < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        left_ankle_roll = self.simulator.dof_pos[:, self.left_ankle_dof_indices[1:2]]
        right_ankle_roll = self.simulator.dof_pos[:, self.right_ankle_dof_indices[1:2]]
        return torch.sum(torch.square(left_ankle_roll) + torch.square(right_ankle_roll), dim=1)
    
    def foot_contact_detect(self, positions, velocity):
        foot_vel = velocity[:, self.feet_indices]
        foot_height = positions[:, self.feet_indices, 2]
        # config-driven (bdx yaml: 0.3/0.08 — VALIDATED against measured standing
        # ankle height 0.0736 m; hardcoded 0.4/0.7 never fires on BDX and made
        # contact obs ~all zeros in pre-2026-08-18 training). Regression test:
        # tests/test_contact_detection.py
        try:
            motion_cfg = self.config.robot.motion
            vel_thres = motion_cfg.get("foot_contact_vel_thres", 0.4)
            height_thres = motion_cfg.get("foot_contact_height_thres", 0.07)
        except AttributeError:
            vel_thres, height_thres = 0.4, 0.07
        foot_speed = torch.norm(foot_vel, dim=-1)  # [num_envs, num_feet]
        contact_mask = (foot_speed < vel_thres) & (foot_height < height_thres)  # [num_envs, num_feet]
        return contact_mask


@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Dict[str, Tensor]
    obs_dict = OrderedDict()
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(root_rot, w_last=True)

    if root_height_obs:
        obs_dict['root_height'] = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last=True)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = quat_to_tan_norm(root_rot, w_last=True) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_dict['local_body_pos'] = local_body_pos
    obs_dict['local_body_rot'] = local_body_rot_obs
    obs_dict['local_body_vel'] = local_body_vel
    obs_dict['local_body_ang_vel'] = local_body_ang_vel

    return obs_dict


@torch.jit.script
def compute_humanoid_observations_max_with_contact(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_binary):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor) -> Dict[str, Tensor]
    obs_dict = OrderedDict()
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(root_rot, w_last=True)

    if root_height_obs:
        obs_dict['root_height'] = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last=True)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = quat_to_tan_norm(root_rot, w_last=True) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_dict['contact_binary'] = contact_binary
    obs_dict['local_body_pos'] = local_body_pos
    obs_dict['local_body_rot'] = local_body_rot_obs
    obs_dict['local_body_vel'] = local_body_vel
    obs_dict['local_body_ang_vel'] = local_body_ang_vel

    return obs_dict
