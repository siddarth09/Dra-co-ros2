# DRA-CO ROS 2

**Franka Panda Arm Implementation using MuJoCo + Drake + ROS 2**

---
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/281ca3ff-ffbf-4864-bbfb-2464e0a9510d" />

---

## Overview

This repository provides a unified simulation and control framework for the **Franka Emika Panda manipulator**, integrating:

* **MuJoCo 3.x** – for high-fidelity physics and real-time rendering
* **Drake** – for analytical kinematics, inverse kinematics (IK), and trajectory generation
* **ROS 2 (Humble +)** – for camera streaming, topic publication, and future action-server interfaces

It enables users to:

* Load the Panda MuJoCo model
* Compute IK and generate smooth trajectories with Drake
* Stream simulated RGB + Depth data from multiple virtual RealSense-style cameras in real time

---

## File Structure

```
panda_mujoco/
├── panda_move.py          # Core MuJoCo + Drake controller
├── camera_publisher.py    # ROS 2 node for multi-camera RGB/Depth streaming
└── robot.py               # Example demo for motion sequences
```

---

## 1. panda_move.py — Unified MuJoCo + Drake Controller

### Class: `PandaMove`

`PandaMove` encapsulates all logic required to load a MuJoCo model, build a Drake multibody plant, solve IK, and execute trajectories directly in simulation.

#### Initialization

```python
panda = PandaMove(xml_path, visualization=True)
```

| Parameter       | Type | Description                           |
| --------------- | ---- | ------------------------------------- |
| `xml_path`      | str  | Path to the Panda MuJoCo XML file.    |
| `visualization` | bool | Enables Drake Meshcat viewer if True. |

#### Core Pipeline

1. Load MuJoCo model → `_load_mujoco_model()`
2. Build Drake plant → `_setup_drake_model()`
3. Optionally start Meshcat → `_setup_meshcat()`
4. Detect end-effector frame → `_detect_end_effector()`

---

### Key Functions

| Function                                      | Purpose               | Description                                          |
| --------------------------------------------- | --------------------- | ---------------------------------------------------- |
| `solve_ik(target_pose)`                       | Inverse Kinematics    | Solves IK using Drake for a given end-effector pose. |
| `plan_cubic_joint_trajectory(q_start, q_end)` | Trajectory Generation | Produces C²-continuous joint-space interpolation.    |
| `run_sequence(targets, duration)`             | Execution             | Streams trajectories through MuJoCo viewer.          |

---

## 2. camera_publisher.py — Multi-Camera ROS 2 Node

`camera_publisher.py` provides a **ROS 2 node (`multi_realsense_node`)** that automatically discovers all RealSense-like cameras defined in the MuJoCo scene XML and publishes their **RGB + Depth** streams as ROS 2 topics.

### Run the Node

```bash
ros2 run panda_mujoco camera_publisher
```

### Published Topics

For each discovered camera prefix (e.g. `wrist`, `table`, `side`):

```
/<prefix>/camera/color/image_raw   # RGB stream
/<prefix>/camera/depth/image_raw   # Depth stream
```

### Interactive Keyboard Control

While the node is running, you can toggle which cameras publish live streams:

| Key | Camera | Stream | Description               |
| --- | ------ | ------ | ------------------------- |
| `1` | wrist  | RGB    | Toggle wrist RGB camera   |
| `2` | wrist  | Depth  | Toggle wrist Depth camera |
| `3` | table  | RGB    | Toggle table RGB camera   |
| `4` | table  | Depth  | Toggle table Depth camera |
| `5` | side   | RGB    | Toggle side RGB camera    |
| `6` | side   | Depth  | Toggle side Depth camera  |

Pressing a key once **starts** publishing that stream; pressing it again **stops** it.
Default: all RGB streams ON, all Depth streams OFF.

The node runs at the parameterized frame rate (`camera_fps`, default 10 Hz) and uses a 320×240 render resolution for efficient simulation.

---

### Performance Notes

* Each camera pair (RGB + Depth) performs separate MuJoCo renders per tick.
* To improve FPS: lower resolution, reduce the number of active streams, or split cameras across threads.
* Typical throughput at 320×240 ≈ 3–5 FPS per camera depending on hardware.

---

## 3. robot.py — Example IK and Trajectory Execution

```python
python3 panda_mujoco/panda_mujoco/robot.py
```

Demonstrates:

1. Initializing `PandaMove`
2. Defining multiple Cartesian targets
3. Solving IK for each pose
4. Executing smooth cubic trajectories in MuJoCo

The MuJoCo viewer opens automatically and the Panda moves through all target poses sequentially.

---

## 4. Current Status and Roadmap

| Component                  | Status     | Description                                |
| -------------------------- | ---------- | ------------------------------------------ |
| MuJoCo Simulation          | ✅ Complete | Physics and rendering integrated           |
| Drake IK & Trajectories    | ✅ Complete | Analytical solver integration              |
| ROS 2 Camera Node          | ✅ Active   | RealSense-style RGB/Depth streaming        |
| ROS 2 Action Servers       | 🔄 Planned | Trajectory execution and feedback          |
| Motion Blending / Planning | 🔄 Future  | Smooth transitions and collision awareness |

---

## 5. Summary

This project unifies:

* **MuJoCo** – for dynamics and rendering
* **Drake** – for kinematics and trajectory planning
* **ROS 2** – for communication and camera streaming

You can now:

* Execute IK and motion sequences with `panda_move.py`
* Stream RGB + Depth feeds interactively with `ros2 run panda_mujoco camera_publisher`
* Toggle individual camera topics live using keyboard shortcuts (1–6)

---

**Author:** Siddarth Dayasagar
**Repository:** `panda_mujoco` — DRA-CO ROS 2 (Drake + MuJoCo Integration for Franka Panda)
