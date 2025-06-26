import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
import tf2_ros
import numpy as np

class ArucoTracker(Node):
    def __init__(self, show_detection = False):
        super().__init__('aruco_detection_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tag_size = 0.05  # 5 cm tag
        self.show_detection = show_detection

        self.create_subscription(Image, '/robot1/D435_1/color/image_raw', self.image_callback, 10)
        self.create_subscription(CameraInfo, '/robot1/D435_1/color/camera_info', self.camera_info_callback, 10)

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera parameters received.")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("Detected ArUco Marker", cv_image)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = ArucoTracker()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()
