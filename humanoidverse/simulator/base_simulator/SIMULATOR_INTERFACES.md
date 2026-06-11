# Simulator Backend Tensor Interfaces

Reference for the three simulator backends implementing `BaseSimulator`. All expose the same interface but differ in quaternion convention, angular velocity frame, and data location.

---

## 1. IsaacLab (IsaacSim) — `isaacsim/isaacsim.py`

All tensors live on GPU via IsaacLab's `Articulation` class. Batched `[num_envs, ...]`.

### Read Properties

| Property | Source | Output Shape | Notes |
|----------|--------|--------------|-------|
| `robot_root_states` | `root_state_w[:, [0,1,2,4,5,6,3,7,8,9,10,11,12]]` | `[N, 13]` | wxyz→xyzw reorder at index 3 |
| `all_root_states` | same as `robot_root_states` | `[N, 13]` | — |
| `base_quat` | `root_state_w[:, [4,5,6,3]]` | `[N, 4]` | **xyzw** |
| `dof_pos` | `joint_pos[:, dof_ids]` | `[N, 29]` | — |
| `dof_vel` | `joint_vel[:, dof_ids]` | `[N, 29]` | — |
| `dof_state` | `cat([dof_pos[...,None], dof_vel[...,None]], -1)` | `[N, 29, 2]` | — |
| `contact_forces` | `contact_sensor.net_forces_w[:, body_idx, :]` | `[N, B, 3]` | — |
| `_rigid_body_pos` | `body_pos_w[:, body_ids, :]` | `[N, B, 3]` | — |
| `_rigid_body_rot` | `body_quat_w[:, body_ids][:,:,[1,2,3,0]]` | `[N, B, 4]` | wxyz→xyzw |
| `_rigid_body_vel` | `body_lin_vel_w[:, body_ids, :]` | `[N, B, 3]` | world frame |
| `_rigid_body_ang_vel` | `body_ang_vel_w[:, body_ids, :]` | `[N, B, 3]` | world frame |

### root_states layout `[N, 13]`

```
[0:3]   position (x, y, z)
[3:7]   quaternion (x, y, z, w)          ← xyzw
[7:10]  linear velocity (vx, vy, vz)     ← world frame
[10:13] angular velocity (wx, wy, wz)    ← WORLD frame (no conversion)
```

### Write Methods

```python
set_actor_root_state_tensor(set_env_ids, root_states):
    write_root_pose_to_sim(root_states[set_env_ids, :7], set_env_ids)    # pos + quat_xyzw
    write_root_velocity_to_sim(root_states[set_env_ids, 7:], set_env_ids) # lin + ang (world frame)
    # NO angular velocity conversion needed

set_dof_state_tensor(set_env_ids, dof_states):
    dof_pos, dof_vel = dof_states[set_env_ids, :, 0], dof_states[set_env_ids, :, 1]
    write_joint_state_to_sim(dof_pos, dof_vel, dof_ids, set_env_ids)

apply_torques_at_dof(torques):
    set_joint_effort_target(torques, joint_ids=dof_ids)
```

---

## 2. MuJoCo CPU — `mujoco/mujoco.py`

Single environment. Data in numpy arrays on CPU. No batch dim.

### MuJoCo Internal Layout

| Field | Layout | Shape | Convention |
|-------|--------|-------|------------|
| `qpos[0:3]` | base position | `[3]` | — |
| `qpos[3:7]` | base quaternion | `[4]` | **wxyz** |
| `qpos[7:36]` | joint positions (29 DOF) | `[29]` | — |
| `qvel[0:3]` | base linear velocity | `[3]` | world frame |
| `qvel[3:6]` | base angular velocity | `[3]` | **world frame** |
| `qvel[6:35]` | joint velocities | `[29]` | — |
| `ctrl[0:6]` | free base actuators (unused) | `[6]` | — |
| `ctrl[6:35]` | joint torques | `[29]` | — |

### Read Properties

| Property | Computation | Output Shape | Notes |
|----------|------------|--------------|-------|
| `robot_root_states` | `[pos, quat_xyzw, lin_vel, quat_rotate(quat, qvel[3:6])]` | `[13]` | ang vel in **body frame** |
| `base_quat` | `qpos[3:7][[1,2,3,0]]` | `[4]` | **xyzw** |
| `dof_pos` | `qpos[7:]` | `[29]` | — |
| `dof_vel` | `qvel[6:]` | `[29]` | — |
| `contact_forces` | `cfrc_ext[1:, 3:6]` | `[B, 3]` | skip world body (index 0) |

### root_states layout `[13]`

```
[0:3]   position (x, y, z)
[3:7]   quaternion (x, y, z, w)          ← xyzw (converted from wxyz)
[7:10]  linear velocity (vx, vy, vz)     ← world frame
[10:13] angular velocity (wx, wy, wz)    ← BODY frame (quat_rotate(quat, qvel[3:6]))
```

### Write Methods

```python
set_actor_root_state_tensor(set_env_ids, root_states):
    # Convert body-frame angular velocity BACK to world frame
    root_states[:, 10:13] = quat_rotate_inverse(base_quat, root_states[:, 10:13], w_last=True)
    qpos[0:3] = root_states[0, 0:3]                    # position
    qpos[3:7] = root_states[0, 3:7][..., [3,0,1,2]]    # xyzw → wxyz
    qvel[0:3] = root_states[0, 7:10]                    # linear velocity
    qvel[3:6] = root_states[0, 10:13]                   # angular velocity (now world frame)

set_dof_state_tensor(set_env_ids, dof_states):
    qpos[7:] = dof_states[0, :, 0]   # joint positions
    qvel[6:] = dof_states[0, :, 1]   # joint velocities

apply_torques_at_dof(torques):
    data.ctrl[6:] = torques.cpu().numpy()   # skip free base actuators
```

---

## 3. MuJoCo Warp — `mujoco_warp/mujoco_warp.py`

Batched `[num_envs, ...]`. Data on GPU as Warp arrays, accessed via `wp.to_torch()` (zero-copy shared memory). Convention follows MuJoCo exactly + batch dim.

### Warp Internal Layout

| Field | Shape (torch) | Convention |
|-------|---------------|------------|
| `d.qpos` | `[N, 36]` | pos(3) + quat**wxyz**(4) + joints(29) |
| `d.qvel` | `[N, 35]` | lin_vel(3) + ang_vel**world**(3) + joint_vel(29) |
| `d.ctrl` | `[N, 35]` | free_base(6) + joint_torques(29) |
| `d.xpos` | `[N, nbody, 3]` | body positions |
| `d.xquat` | `[N, nbody, 4]` | body quats **wxyz** |
| `d.cvel` | `[N, nbody, 6]` | `[angular(3), linear(3)]` |
| `d.cfrc_ext` | `[N, nbody, 6]` | `[torque(3), force(3)]` |

> **Warning:** Warp `.shape` attribute lies for spatial dimensions. `d.xpos.shape` reports `(N, 33)` but `wp.to_torch(d.xpos)` returns `[N, 33, 3]`. Always use `wp.to_torch()` for correct shapes.

### Read Properties

| Property | Computation | Output Shape | Notes |
|----------|------------|--------------|-------|
| `robot_root_states` | `cat([pos, quat_xyzw, lin_vel, quat_rotate(quat_xyzw, qvel[:,3:6])])` | `[N, 13]` | ang vel in **body frame** |
| `all_root_states` | same as `robot_root_states` | `[N, 13]` | — |
| `base_quat` | `qpos[:,3:7][:,[1,2,3,0]]` | `[N, 4]` | **xyzw** |
| `dof_pos` | `qpos[:,7:]` | `[N, 29]` | — |
| `dof_vel` | `qvel[:,6:]` | `[N, 29]` | — |
| `dof_state` | `cat([dof_pos[...,None], dof_vel[...,None]], -1)` | `[N, 29, 2]` | — |
| `contact_forces` | `cfrc_ext[:,1:,3:6]` | `[N, B, 3]` | skip world body |
| `_rigid_body_pos` | `xpos[:,body_id,:]` | `[N, B, 3]` | — |
| `_rigid_body_rot` | `xquat[:,body_id,:][:,:,[1,2,3,0]]` | `[N, B, 4]` | wxyz→xyzw |
| `_rigid_body_vel` | `cvel[:,body_id,3:6]` | `[N, B, 3]` | linear velocity |
| `_rigid_body_ang_vel` | `cvel[:,body_id,0:3]` | `[N, B, 3]` | angular velocity |

### root_states layout `[N, 13]`

```
[0:3]   position (x, y, z)
[3:7]   quaternion (x, y, z, w)          ← xyzw (converted from wxyz)
[7:10]  linear velocity (vx, vy, vz)     ← world frame
[10:13] angular velocity (wx, wy, wz)    ← BODY frame (quat_rotate(quat_xyzw, qvel[:,3:6]))
```

### Write Methods

```python
set_actor_root_state_tensor(set_env_ids, root_states):
    # Reshape to ensure correct dimensions
    set_env_ids = set_env_ids.reshape(-1)
    root_states = root_states.reshape(-1, 13).clone()

    # Convert body-frame angular velocity BACK to world frame (same as MuJoCo)
    base_quat_subset = self.base_quat[set_env_ids]  # [n, 4] xyzw
    root_states[:, 10:13] = quat_rotate_inverse(base_quat_subset, root_states[:, 10:13], w_last=True)

    # GPU-direct write via wp.to_torch() (shared memory, zero-copy)
    qpos_t = wp.to_torch(d.qpos)  # [N, 36] on CUDA
    qvel_t = wp.to_torch(d.qvel)  # [N, 35] on CUDA

    qpos_t[set_env_ids, 0:3] = root_states[:, 0:3]         # position
    qpos_t[set_env_ids, 3]   = root_states[:, 6]            # w
    qpos_t[set_env_ids, 4]   = root_states[:, 3]            # x
    qpos_t[set_env_ids, 5]   = root_states[:, 4]            # y
    qpos_t[set_env_ids, 6]   = root_states[:, 5]            # z
    qvel_t[set_env_ids, 0:6] = root_states[:, 7:13]         # velocity
    self._graph = None  # invalidate CUDA graph after state changes

set_dof_state_tensor(set_env_ids, dof_states):
    qpos_t = wp.to_torch(d.qpos)
    qvel_t = wp.to_torch(d.qvel)
    qpos_t[set_env_ids, 7:] = dof_states[set_env_ids, :, 0]   # joint positions
    qvel_t[set_env_ids, 6:] = dof_states[set_env_ids, :, 1]   # joint velocities
    self._graph = None

apply_torques_at_dof(torques):
    wp_torques = wp.from_torch(torques.contiguous())   # zero-copy if on CUDA
    wp.copy(d.ctrl[:, 6:], wp_torques)                 # skip free base actuators
```

---

## Key Convention Differences

| Aspect | IsaacSim | MuJoCo | MuJoCo Warp |
|--------|----------|--------|-------------|
| **Environments** | Batched `[N, ...]` | Single `[...]` | Batched `[N, ...]` |
| **Data location** | GPU (IsaacLab tensors) | CPU (numpy) | GPU (Warp arrays, `wp.to_torch()`) |
| **Quat internal** | wxyz | wxyz | wxyz |
| **Quat exposed** | xyzw | xyzw | xyzw |
| **Ang vel in root_states** | **world** frame | **body** frame | **body** frame |
| **Ang vel write-back** | direct (no conversion) | `quat_rotate_inverse` | `quat_rotate_inverse` |
| **CUDA graph** | N/A | N/A | ✅ after 5 warmup steps |

## Critical Notes

1. **`robot_root_states` is a read-only property.** Every call constructs a new tensor. Writing to the returned tensor does NOT modify simulation state. All state changes must go through `set_actor_root_state_tensor` / `set_dof_state_tensor`.

2. **MuJoCo Warp `.numpy()` returns a host COPY.** On CUDA warp arrays, `d.qpos.numpy()` creates a CPU-side copy — writes to it are silently lost. Always use `wp.to_torch(d.qpos)` which shares GPU memory.

3. **Warp `.shape` lies.** `d.xpos.shape` reports `(N, 33)` but actual shape is `(N, 33, 3)`. Use `wp.to_torch()` for correct shapes.

4. **Angular velocity conversion is the #1 source of bugs.** MuJoCo and MuJoCo Warp store world-frame angular velocity in `qvel[3:6]` but expose body-frame in `robot_root_states[10:13]` via `quat_rotate`. The inverse must be applied when writing back. Forgetting this produces silently wrong physics.