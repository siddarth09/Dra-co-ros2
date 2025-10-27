#!/usr/bin/env python3
import sys, select, termios, tty, time, signal, os
import numpy as np
import mujoco as mu
from mujoco import viewer
from pydrake.all import RigidTransform, RotationMatrix
from panda_move import PandaMove

# step sizes
DELTA_POS = 0.02
DELTA_ROT = np.deg2rad(5)


class PandaTeleop:
    def __init__(self, panda: PandaMove):
        self.panda = panda
        self.R, self.t = panda.get_ee_pose()
        self.R = self.R.matrix()
        self.t = np.array(self.t)
        self.settings = termios.tcgetattr(sys.stdin)
        self.last_pressed = {}

        print("\n=== Panda Keyboard Teleop ===")
        print("Move EE in Cartesian space:")
        print("  W/S : +Z / -Z")
        print("  A/D : +Y / -Y")
        print("  Q/E : +X / -X")
        print("  R/F : roll + / -")
        print("  T/G : pitch + / -")
        print("  Y/H : yaw + / -")
        print("  SPACE : reset home")
        print("  ESC or CTRL+C : quit")
        print("==============================\n")

    # ----------- util: read one key without blocking ------------
    def _get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    # ----------- main teleop loop ------------
    def run(self):
        with viewer.launch(self.panda.model, self.panda.data) as sim_viewer:
            while sim_viewer.is_running():
                key = self._get_key()

                if key == '\x1b' or key == 'q':  # ESC or q
                    print("Exiting teleop.")
                    break
                elif key == ' ':
                    print("Resetting to home keyframe.")
                    try:
                        key_id = mu.mj_name2id(
                            self.panda.model, mu.mjtObj.mjOBJ_KEY, "home"
                        )
                        mu.mj_resetDataKeyframe(self.panda.model, self.panda.data, key_id)
                    except Exception:
                        mu.mj_resetData(self.panda.model)
                    self.R, self.t = self.panda.get_ee_pose()
                    self.R = self.R.matrix()
                    self.t = np.array(self.t)
                    continue

                # ----- translations -----
                if key == 'q':
                    self.t[0] += DELTA_POS
                elif key == 'e':
                    self.t[0] -= DELTA_POS
                elif key == 'a':
                    self.t[1] += DELTA_POS
                elif key == 'd':
                    self.t[1] -= DELTA_POS
                elif key == 'w':
                    self.t[2] += DELTA_POS
                elif key == 's':
                    self.t[2] -= DELTA_POS

                # ----- rotations -----
                elif key == 'r':
                    self.R = self.R @ RotationMatrix.MakeZRotation(DELTA_ROT).matrix()
                elif key == 'f':
                    self.R = self.R @ RotationMatrix.MakeZRotation(-DELTA_ROT).matrix()
                elif key == 't':
                    self.R = self.R @ RotationMatrix.MakeYRotation(DELTA_ROT).matrix()
                elif key == 'g':
                    self.R = self.R @ RotationMatrix.MakeYRotation(-DELTA_ROT).matrix()
                elif key == 'y':
                    self.R = self.R @ RotationMatrix.MakeXRotation(DELTA_ROT).matrix()
                elif key == 'h':
                    self.R = self.R @ RotationMatrix.MakeXRotation(-DELTA_ROT).matrix()

                if key:
                    self._apply_pose_update()

                sim_viewer.sync()
                time.sleep(0.01)

    # ----------- apply target pose via IK ------------
    def _apply_pose_update(self):
        target_pose = RigidTransform(RotationMatrix(self.R), self.t)
        try:
            q_sol = self.panda.solve_ik(target_pose)
            self.panda.data.qpos[:] = q_sol
            mu.mj_forward(self.panda.model, self.panda.data)
        except Exception:
            print("IK failed for this step.")


# --------------- entry point ---------------
def main():
    MODEL_PATH = "/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/scene.xml"
    panda = PandaMove(MODEL_PATH, visualization=False)
    teleop = PandaTeleop(panda)

    # graceful exit on Ctrl+C
    def _signal_handler(sig, frame):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, teleop.settings)
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_handler)

    teleop.run()


if __name__ == "__main__":
    main()
