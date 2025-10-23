#!/usr/bin/env python3
import mujoco as mu
from mujoco import viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory


class PandaStatePublisher(Node):
    def __init__(self):
        super().__init__('panda_state_publisher')
        self.xml_path = '/home/siddarth/manipulation_ws/src/panda_mujoco/franka_emika_panda/scene.xml'
        self.model = mu.MjModel.from_xml_path(self.xml_path)
        self.data = mu.MjData(self.model)
        self.pub = self.create_publisher(JointState, '/panda/joint_states', 10)
        self.get_logger().info(" Panda State Publisher node started.")

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            mu.mj_id2name(self.model, mu.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        msg.position = self.data.qpos.tolist()
        self.pub.publish(msg)

    def start_visualization(self):
        with viewer.launch_passive(self.model, self.data) as v:
            while rclpy.ok() and v.is_running():
                mu.mj_step(self.model, self.data)
                v.sync()
                self.publish_joint_states()


def main(args=None):
    rclpy.init(args=args)
    node = PandaStatePublisher()
    node.start_visualization()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
