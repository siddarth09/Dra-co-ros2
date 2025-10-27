#!/usr/bin/env python3
import mujoco
import numpy as np
import cv2
from mujoco import viewer


def main():
    MODEL_PATH = "/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/scene.xml"

    # ------------------------------------------------------------------
    # Load model + data
    # ------------------------------------------------------------------
    print(f" Loading MuJoCo model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Reset to 'home' keyframe if available
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except Exception:
        mujoco.mj_resetData(model)

    # ------------------------------------------------------------------
    # Camera setup
    # ------------------------------------------------------------------
    H, W = 480, 640
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.enable_depth_rendering()

    rgb_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
    depth_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")

    if rgb_cam_id == -1 or depth_cam_id == -1:
        raise RuntimeError("Both rgb_camera and depth_camera must exist in the XML!")

    print("\n--- Viewer started (manual mode) ---")
    print("Use the Control panel sliders to move joints.")
    print("Press 'q' in OpenCV window or ESC in viewer to quit.\n")

    # ------------------------------------------------------------------
    # Launch interactive viewer
    # ------------------------------------------------------------------
    with viewer.launch_passive(model, data) as sim_viewer:
        counter = 0
        while sim_viewer.is_running():
            mujoco.mj_step(model, data)  # physics step
            counter += 1

            # Render camera every few steps to maintain speed (~20 Hz)
            if counter % 12 == 0:
                # --- RGB ---
                renderer.disable_depth_rendering()
                renderer.update_scene(data, camera=rgb_cam_id)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # --- Depth ---
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=depth_cam_id)
                depth = renderer.render()
                renderer.disable_depth_rendering()

                depth = np.nan_to_num(depth)
                dmin, dmax = depth.min(), depth.max()
                if dmax > dmin:
                    depth_norm = (depth - dmin) / (dmax - dmin)
                else:
                    depth_norm = np.zeros_like(depth)
                depth_vis = (depth_norm * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

                combined = np.hstack((bgr, depth_color))
                cv2.imshow("MuJoCo Camera (RGB | Depth)", combined)

                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord("q")]:
                    break

            sim_viewer.sync()

    print("\n--- Simulation complete ---")
    renderer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
