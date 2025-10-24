import mujoco
import numpy as np
from mujoco import viewer
from PIL import Image

# -------------------------------------------------------------
# Load model and data
# -------------------------------------------------------------
MODEL_PATH = "/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/panda_with_camera.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Reset to keyframe "home" (so the arm is not folded)
try:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
except Exception:
    mujoco.mj_resetData(model)

# -------------------------------------------------------------
# Create an offscreen renderer for camera capture
# -------------------------------------------------------------
renderer = mujoco.Renderer(model, height=480, width=640)

camera_name = "rgb_camera"
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

if camera_id == -1:
    raise RuntimeError(f"Camera '{camera_name}' not found in model!")

# -------------------------------------------------------------
# Start the interactive viewer
# -------------------------------------------------------------
with viewer.launch_passive(model, data) as sim_viewer:
    print("\n--- Viewer started ---")
    print("Press ESC or close the window to quit.\n")

    while sim_viewer.is_running():
        mujoco.mj_step(model, data)

        
        renderer.update_scene(data, camera=camera_id)
        rgb_image = renderer.render()

       
        # Update the interactive viewer window
        sim_viewer.sync()

    print("--- Simulation complete ---")

renderer.close()
