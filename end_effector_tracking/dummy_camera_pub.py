# aruco_tracker/image_publisher/image_and_camera_info_pub.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import cairosvg
from io import BytesIO

class ImageAndCameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('image_and_camera_info_publisher')

        # Publishers
        self.image_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.5, self.timer_callback)  # 2 Hz

        # === Load and convert SVG ===
        svg_path = '/home/kd/Documents/urs_ws/src/end_effector_tracking/data/4x4_50-0.svg'
        with open(svg_path, 'rb') as f:
            png_data = BytesIO()
            cairosvg.svg2png(file_obj=f, write_to=png_data)
            nparr = np.frombuffer(png_data.getvalue(), np.uint8)
            self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # === Camera Info ===
        self.camera_info = CameraInfo()
        self.camera_info.header.frame_id = 'camera_color_optical_frame'
        self.camera_info.height = self.image.shape[0]
        self.camera_info.width = self.image.shape[1]
        self.camera_info.distortion_model = 'plumb_bob'
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        fx = fy = 525.0
        cx = self.camera_info.width / 2.0
        cy = self.camera_info.height / 2.0

        self.camera_info.k = [fx, 0.0, cx,
                              0.0, fy, cy,
                              0.0, 0.0, 1.0]
        self.camera_info.r = [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0]
        self.camera_info.p = [fx, 0.0, cx, 0.0,
                              0.0, fy, cy, 0.0,
                              0.0, 0.0, 1.0, 0.0]

    def timer_callback(self):
        # Create and publish image message
        image_msg = self.bridge.cv2_to_imgmsg(self.image, encoding='bgr8')
        stamp = self.get_clock().now().to_msg()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = 'camera_color_optical_frame'

        self.image_pub.publish(image_msg)

        # Publish camera info with same timestamp
        self.camera_info.header.stamp = stamp
        self.camera_info_pub.publish(self.camera_info)


def main():
    rclpy.init()
    node = ImageAndCameraInfoPublisher()
    rclpy.spin(node)
    rclpy.shutdown()
