#!/usr/bin/env python3
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix
from panda_move import PandaMove

# Path to your MuJoCo model
xml_path = "/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/scene.xml"

panda = PandaMove(xml_path, visualization=True)

# Define multiple target poses
targets = [
    RigidTransform(
        RotationMatrix.MakeXRotation(np.pi / 2),
        [0.5, 0.0, 0.0]
    ),
    RigidTransform(
        RotationMatrix.MakeZRotation(np.pi / 2),
        [0.5, 0.5, 0.5]
    )
 
 
]

# Solve IK for each target
q_targets = []
for i, pose in enumerate(targets):
    print(f"\n Solving IK for Target {i+1}: {pose.translation()}")
    q_sol = panda.solve_ik(pose)
    print(f" IK Solution {i+1}: {np.round(q_sol, 4)}")
    q_targets.append(q_sol)

# Run the full motion sequence in one MuJoCo viewer session
print("\n Executing full motion sequence...")
panda.run_sequence(q_targets, duration=3.0)

