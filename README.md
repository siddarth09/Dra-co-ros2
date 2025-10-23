# DRA-CO ROS2
**Franka Panda Arm Implementation using MuJoCo + Drake + ROS 2 (ROS 2 support coming soon)**

---

## Overview

This repository provides a unified controller for the Franka Emika Panda manipulator, integrating:

* **MuJoCo** for high-fidelity dynamics and visualization
* **Drake** for analytical kinematics, inverse kinematics (IK), and trajectory generation
* **ROS 2 (planned)** for future integration with action servers and topics

The core functionality is implemented in `panda_move.py`, which can be executed and tested directly without ROS 2.
This system enables users to plan, solve IK, and execute smooth joint-space trajectories within the MuJoCo simulator while leveraging Drake’s robust solver stack.

---

## File Structure

```
panda_mujoco/
├── panda_move.py        # Core MuJoCo + Drake integration
└── robot.py             # Example demo script for running motion sequences
```

---

## 1. panda_move.py — Unified MuJoCo + Drake Controller

### Class: PandaMove

`PandaMove` encapsulates all functionality needed to load a MuJoCo model, build a Drake multibody representation for kinematic computations, solve inverse kinematics, and execute trajectories in simulation.

#### Initialization

```python
panda = PandaMove(xml_path, visualization=True)
```

**Parameters**

| Parameter       | Type   | Description                                                 |
| --------------- | ------ | ----------------------------------------------------------- |
| `xml_path`      | `str`  | Path to the MuJoCo XML model of the Franka Panda.           |
| `visualization` | `bool` | If `True`, initializes Meshcat visualization through Drake. |

The constructor sequentially:

1. Loads the MuJoCo model (`_load_mujoco_model`)
2. Converts it for Drake (`_setup_drake_model`)
3. Optionally starts Meshcat (`_setup_meshcat`)

---

### Internal Setup and Utility Functions

#### `_load_mujoco_model()`

Loads the Panda XML model into a MuJoCo `MjModel` and creates its simulation data.

```python
self.model = mu.MjModel.from_xml_path(self.xml_path)
self.data = mu.MjData(self.model)
```

This step initializes the simulation backend.

---

#### `_setup_drake_model()`

Converts the MuJoCo XML into a Drake-compatible XML and constructs a Drake `MultibodyPlant`.

Steps performed:

1. Uses `MakeDrakeCompatibleModel` to generate a `.drake.xml` model.
2. Builds a `MultibodyPlant` and parses the converted model.
3. Detects the Panda’s end-effector frame automatically (`_detect_end_effector`).

This enables Drake’s solver APIs (IK, planning, etc.) to operate on the same robot geometry as MuJoCo.

---

#### `_setup_meshcat()`

Starts a Meshcat server for Drake visualization if `visualization=True`.

Creates a `RobotDiagram`, initializes a `Simulator`, and stores a mutable `Context` for further visualization or simulation.

---

#### `_detect_end_effector()`

Searches for common Franka end-effector frame names (`hand`, `link7`, `panda_link8`) and returns the first found frame.

Raises `RuntimeError` if none are found.

---

#### `get_current_joint_positions()`

Returns the current MuJoCo joint positions as a NumPy array.

```python
q = panda.get_current_joint_positions()
```

---

#### `get_ee_pose()`

Queries Drake’s plant for the end-effector pose in world coordinates.

```python
R, p = panda.get_ee_pose()
```

Returns:

* `R`: `RotationMatrix` (orientation)
* `p`: `np.ndarray` (translation vector `[x, y, z]`)

---

### Motion Planning and Execution APIs

#### `solve_ik(target_pose: RigidTransform)`

Solves inverse kinematics using Drake for a given end-effector pose.

```python
pose = RigidTransform(RotationMatrix.MakeZRotation(np.pi/2), [0.5, 0.0, 0.5])
q_sol = panda.solve_ik(pose)
```

This sets up:

* A position constraint on the end-effector translation
* An orientation constraint on its rotation

Returns a NumPy array of joint angles that satisfy the constraints within tolerance.

Raises a `RuntimeError` if the IK problem has no feasible solution.

---

#### `plan_cubic_joint_trajectory(q_start, q_end, steps=150)`

Generates a smooth, time-parameterized cubic joint-space trajectory between two configurations with zero start and end velocities.

```python
q_traj, qd_traj = panda.plan_cubic_joint_trajectory(q_start, q_end, steps=200)
```

Returns:

* `q_traj`: NxJ array of interpolated joint positions
* `qd_traj`: NxJ array of interpolated joint velocities

This produces continuous acceleration profiles (C²-continuous trajectories) suitable for smooth motion execution.

---

#### `run_sequence(targets, duration=2.0)`

Executes a list of joint targets sequentially in the MuJoCo viewer.

```python
panda.run_sequence(q_targets, duration=3.0)
```

Steps:

1. Launches a passive MuJoCo viewer.
2. For each target:

   * Plans a cubic trajectory from the current position to the next target.
   * Iteratively updates `self.data.qpos` and advances the simulation.
3. Prints progress for each target reached.
4. Keeps the viewer open until manually closed.

This method enables continuous trajectory playback and can be extended later to publish commands via ROS 2 actions.

---

### Internal Algorithm Summary

| Function                      | Purpose               | Description                              |
| ----------------------------- | --------------------- | ---------------------------------------- |
| `_load_mujoco_model`          | Simulation setup      | Loads XML, initializes MuJoCo backend    |
| `_setup_drake_model`          | Kinematics setup      | Creates Drake plant for IK/solvers       |
| `_setup_meshcat`              | Visualization         | Starts optional Meshcat session          |
| `solve_ik`                    | Inverse kinematics    | Computes joint config for pose           |
| `plan_cubic_joint_trajectory` | Trajectory generation | Produces smooth time-based interpolation |
| `run_sequence`                | Execution             | Streams trajectory through MuJoCo        |

---

## 2. robot.py — Example Usage and Test Script

This file demonstrates how to use `PandaMove` for multi-pose IK and trajectory execution.

```python
#!/usr/bin/env python3
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix
from panda_move import PandaMove

xml_path = "src/panda_mujoco/franka_emika_panda/scene.xml"

panda = PandaMove(xml_path, visualization=True)

# Define multiple Cartesian target poses
targets = [
    RigidTransform(RotationMatrix.MakeZRotation(np.pi/2), [0.5, 0.0, 0.5]),
    RigidTransform(RotationMatrix.MakeZRotation(np.pi/2), [0.3, 0.0, 0.6]),
    RigidTransform(RotationMatrix.MakeZRotation(np.pi/2), [0.4, 0.2, 0.4]),
]

# Compute IK for each pose
q_targets = []
for i, pose in enumerate(targets):
    print(f"\n Solving IK for Target {i+1}: {pose.translation()}")
    q_sol = panda.solve_ik(pose)
    print(f" IK Solution {i+1}: {np.round(q_sol, 4)}")
    q_targets.append(q_sol)

# Execute all targets in sequence
print("\n Executing full motion sequence...")
panda.run_sequence(q_targets, duration=3.0)
```

**To run the demo:**

```bash
python3 panda_mujoco/panda_mujoco/robot.py
```

The script:

1. Initializes the PandaMove controller and loads the MuJoCo + Drake models.
2. Defines three end-effector poses.
3. Solves IK for each pose using Drake.
4. Executes the smooth cubic trajectories between all poses in MuJoCo.

The MuJoCo viewer opens automatically, and the Panda moves through each target pose sequentially.

---

## 3. Current Status and Future Work

| Component                 | Status   | Description                                      |
| ------------------------- | -------- | ------------------------------------------------ |
| MuJoCo Integration        | Complete | Physics simulation and viewer active             |
| Drake IK and Trajectories | Complete | Fully functional solver integration              |
| ROS 2 Support             | Planned  | Action server interface to follow                |
| Trajectory Blending       | Planned  | Continuous velocity transition between waypoints |
| Collision-Aware Planning  | Future   | Integration with RRT*/PRM planners               |

---

## 4. Dependencies

Ensure that the following Python packages are installed:

* `mujoco` ≥ 3.1
* `drake` ≥ 1.24
* `manipulation` (Drake manipulation utilities)
* `numpy`

Install via pip if required:

```bash
pip install mujoco numpy
pip install drake  # or use official Drake binaries
```

---

## 5. Summary

`panda_move.py` provides a unified Python interface to:

* Load and simulate a Franka Panda in MuJoCo
* Compute inverse kinematics with Drake
* Generate and execute smooth joint trajectories

`robot.py` serves as the main test entry point for running the full motion sequence.
This implementation will later be extended with ROS 2 interfaces to allow real-time trajectory execution through ROS 2 action servers.

---
