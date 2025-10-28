#!/usr/bin/env python3
import numpy as np
from pydrake.all import RigidTransform
from panda_move import PandaMove

def main():
    MODEL_PATH = "/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/scene.xml"
    panda = PandaMove(MODEL_PATH, visualization=False)

    # Get current pose
    R_fixed, p0 = panda.get_ee_pose()
    print("Initial EE position:", p0)

    # Define translation-only IK targets (no rotation change)
    targets = [
        RigidTransform(R_fixed, p0 + np.array([0.1, 0.0, 0.0])),  # move forward
        RigidTransform(R_fixed, p0 + np.array([0.1, 0.1, 0.0]))   # move diagonal
    ]

    q_targets = []
    for pose in targets:
        q_sol = panda.solve_ik(pose)
        q_targets.append(q_sol)

    # Run the sequence
    panda.run_sequence(q_targets, duration=2.0)

if __name__ == "__main__":
    main()
