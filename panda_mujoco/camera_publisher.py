#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import mujoco as mu
import open3d as o3d
import struct

class PandaCameraPublisher(Node):
    def __init__(self):
        super().__init__("panda_camera_publisher")

        # --- Parameters ---
        self.declare_parameter("xml_path", "/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/scene.xml")
        xml_path = self.get_parameter("xml_path").get_parameter_value().string_value

        # --- Load MuJoCo model ---
        self.model = mu.MjModel.from_xml_path(xml_path)
        self.data = mu.MjData(self.model)
        self.camera_name = "rgb_camera"
        self.cam_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_CAMERA, self.camera_name)
        self.renderer = mu.Renderer(self.model)

        # --- Camera intrinsics (approximate RealSense D435i) ---
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.depth_scale = 1.0  # already in meters

        # --- Publishers ---
        self.rgb_pub = self.create_publisher(Image, "/panda/camera/rgb", 10)
        self.depth_pub = self.create_publisher(Image, "/panda/camera/depth", 10)
        self.pc_pub = self.create_publisher(PointCloud2, "/panda/camera/pointcloud", 10)

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.publish_camera_data)  # 10 Hz
        self.get_logger().info("✅ Panda camera publisher (MuJoCo + Open3D) started.")

    # ----------------------------------------------------------
    # Core Loop
    # ----------------------------------------------------------
    def publish_camera_data(self):
        mu.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam_id)

        rgb = self.renderer.render()            # RGB image (H×W×3, uint8)
        depth = self.renderer.render(depth=True)  # Depth map (H×W, float32)

        # --- Convert to ROS Image messages ---
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        rgb_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.header.stamp = rgb_msg.header.stamp
        rgb_msg.header.frame_id = "panda_camera_link"
        depth_msg.header.frame_id = "panda_camera_link"

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)

        # --- Generate and publish Open3D point cloud ---
        self.publish_pointcloud_open3d(rgb, depth, rgb_msg.header)

    # ----------------------------------------------------------
    # Open3D Point-Cloud Conversion
    # ----------------------------------------------------------
    def publish_pointcloud_open3d(self, rgb, depth, header):
        H, W, _ = rgb.shape

        # Convert to Open3D images
        depth_o3d = o3d.geometry.Image((depth * self.depth_scale).astype(np.float32))
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, self.fx, self.fy, self.cx, self.cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # Optional filtering
        pcd = pcd.voxel_down_sample(voxel_size=0.002)

        # Convert to ROS PointCloud2
        pts = np.asarray(pcd.points)
        cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        rgb_packed = np.array([
            struct.unpack('I', struct.pack('BBBB', r, g, b, 255))[0] for r, g, b in cols
        ], dtype=np.uint32)

        pc = np.zeros(pts.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'u4')
        ])
        pc['x'], pc['y'], pc['z'], pc['rgb'] = pts[:,0], pts[:,1], pts[:,2], rgb_packed

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = pts.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=7, count=1),
            PointField(name='y', offset=4, datatype=7, count=1),
            PointField(name='z', offset=8, datatype=7, count=1),
            PointField(name='rgb', offset=12, datatype=7, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * pts.shape[0]
        msg.is_dense = True
        msg.data = pc.tobytes()

        self.pc_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PandaCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
