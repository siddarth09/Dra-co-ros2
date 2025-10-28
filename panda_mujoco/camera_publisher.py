#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import mujoco as mu
import numpy as np
import cv2
import re
import threading, sys, termios, tty


class MultiRealsenseNode(Node):
    """
    Unified publisher for all RealSense-like cameras defined in a MuJoCo scene.
    Automatically detects and publishes RGB+Depth topics for each camera name.
    """

    def __init__(self):
        super().__init__('multi_realsense_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('xml_path', '/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/scene.xml')
        self.declare_parameter('camera_fps', 10.0)

        self.xml_path = self.get_parameter('xml_path').get_parameter_value().string_value
        self.fps = self.get_parameter('camera_fps').get_parameter_value().double_value
        self.dt = 1.0 / self.fps

        self.get_logger().info(f"Loading MuJoCo model: {self.xml_path}")
        self.model = mu.MjModel.from_xml_path(self.xml_path)
        self.data = mu.MjData(self.model)

        # ---------------- Renderer ----------------
        self.img_h = 480
        self.img_w = 640
        self.renderer = mu.Renderer(self.model, height=self.img_h, width=self.img_w)
        self.renderer.enable_depth_rendering()

        # ---------------- Camera discovery ----------------
        self.bridge = CvBridge()
        self.cameras = self._discover_cameras()

        if not self.cameras:
            raise RuntimeError("No valid RGB/Depth camera pairs found in the MuJoCo model!")

        self.get_logger().info(f"Discovered {len(self.cameras)} camera sets:")
        for cam_name in self.cameras.keys():
            self.get_logger().info(f"  - {cam_name}")

        # ---------------- Publishers ----------------
        self.camera_publishers = self._create_publishers()
        self.active = {prefix: {'rgb': True, 'depth': False}
                       for prefix in self.cameras.keys()}
        self.key_map = {
            '1': ('wrist', 'rgb'),
            '2': ('wrist', 'depth'),
            '3': ('table', 'rgb'),
            '4': ('table', 'depth'),
            '5': ('side', 'rgb'),
            '6': ('side', 'depth'),
        }
        self.key_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.key_thread.start()

        # ---------------- Timer ----------------
        self.timer = self.create_timer(self.dt, self.publish_all_cameras)
        self.get_logger().info(f"Publishing all camera streams at {self.fps:.1f} FPS")

    # ------------------------------------------------------------------
    # CAMERA DISCOVERY + PUBLISHER SETUP
    # ------------------------------------------------------------------
    
    def _keyboard_listener(self):
        """Listen for keypresses and toggle publishers."""
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.get_logger().info("Press numeric keys (1-6) to toggle camera streams.")
        try:
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if ch in self.key_map:
                    prefix, stream = self.key_map[ch]
                    self.active[prefix][stream] = not self.active[prefix][stream]
                    state = "ON" if self.active[prefix][stream] else "OFF"
                    self.get_logger().info(f"Toggled {prefix}/{stream} → {state}")
        except Exception as e:
            self.get_logger().error(f"Keyboard thread error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
    def _discover_cameras(self):
        """
        Discover all camera name pairs (rgb/depth) based on naming pattern:
        e.g., table_rgb / table_depth → key 'table'
        """
        all_cams = [mu.mj_id2name(self.model, mu.mjtObj.mjOBJ_CAMERA, i)
                    for i in range(self.model.ncam)]

        # Group by prefix
        pairs = {}
        for name in all_cams:
            if name.endswith('_rgb'):
                prefix = name.replace('_rgb', '')
                if prefix not in pairs:
                    pairs[prefix] = {}
                pairs[prefix]['rgb'] = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_CAMERA, name)

            elif name.endswith('_depth'):
                prefix = name.replace('_depth', '')
                if prefix not in pairs:
                    pairs[prefix] = {}
                pairs[prefix]['depth'] = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_CAMERA, name)

        # Keep only valid pairs
        valid_pairs = {k: v for k, v in pairs.items() if 'rgb' in v and 'depth' in v}
        return valid_pairs

    def _create_publishers(self):
        """Create publishers for all discovered cameras."""
        pubs = {}
        for prefix in self.cameras.keys():
            pubs[prefix] = {
                'rgb': self.create_publisher(Image, f'/{prefix}/camera/color/image_raw', 10),
                'depth': self.create_publisher(Image, f'/{prefix}/camera/depth/image_raw', 10)
            }
        return pubs

    # ------------------------------------------------------------------
    # RENDERING + PUBLISHING
    # ------------------------------------------------------------------

    def _render_camera(self, cam_id, depth=False):
        """Render an RGB or depth frame from a camera ID."""
        if depth:
            self.renderer.enable_depth_rendering()
        else:
            self.renderer.disable_depth_rendering()

        self.renderer.update_scene(self.data, camera=cam_id)
        img = self.renderer.render().copy()
        return img

    def publish_all_cameras(self):
        mu.mj_step(self.model, self.data)
        for prefix, ids in self.cameras.items():
            # RGB
            if self.active[prefix]['rgb']:
                rgb = self._render_camera(ids['rgb'], depth=False)
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
                rgb_msg.header.stamp = self.get_clock().now().to_msg()
                self.camera_publishers[prefix]['rgb'].publish(rgb_msg)

            # Depth
            if self.active[prefix]['depth']:
                depth = np.nan_to_num(self._render_camera(ids['depth'], depth=True))
                depth_mm = (depth * 1000.0).astype(np.uint16)
                depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                self.camera_publishers[prefix]['depth'].publish(depth_msg)

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info("Shutting down MultiRealsenseNode.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiRealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
