#!/usr/bin/env python3
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix
from panda_move import PandaMove

def main():
    MODEL_PATH = "/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/scene.xml"
    panda = PandaMove(MODEL_PATH, visualization=False)
    initial_pose= panda.get_ee_pose()
    print(initial_pose)
    # Define IK targets
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

    # Solve IK
    q_targets = []
    for pose in targets:
        q_sol = panda.solve_ik(pose)
        q_targets.append(q_sol)

    # Run motion + live RGB|Depth streaming
    panda.run_sequence(q_targets, duration=2.0)


if __name__ == "__main__":
    main()
