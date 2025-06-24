#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from dualarm_custom_msgs.msg import TrajStatus      # Custom message

class StatusPublisher(Node):
    def __init__(self):
        super().__init__('status_publisher')

        # Create a publisher for StatusArray messages
        self.publisher_ = self.create_publisher(TrajStatus, 'status_array_topic', 10)

        # Publish every second
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.counter = 0

    def timer_callback(self):
        msg = TrajStatus()
        msg.status = f"Running {self.counter}"

        # Populate Float64MultiArray data
        msg.data.data = [float(i) for i in range(3)]

        self.publisher_.publish(msg)
        self.get_logger().info(f"Published: status='{msg.status}', data={msg.data.data}")

        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = StatusPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
