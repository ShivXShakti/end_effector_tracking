import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
import numpy as np
from std_msgs.msg import Float64MultiArray
from dualarm_custom_msgs.msg import TrajStatus 




class ArucoTracker(Node):
    def __init__(self, show_detection = True):
        super().__init__('aruco_tracker_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tag_size = 0.1  # 5 cm
        self.show_detection = show_detection
        self.centroid_pub = self.create_publisher(Float64MultiArray, '/aruco/left_right/ee_pose', 10)

        self.create_subscription(Image, '/robot1/D435_1/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/robot1/D435_1/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/robot1/D435_1/color/image_raw', self.image_callback, 10)
        #self.create_subscription(CameraInfo, '/robot1/D435_1/color/camera_info', self.camera_info_callback, 10)

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera parameters received.")
    def depth_callback(self, msg):
        """Convert depth image from ROS2 message to OpenCV format"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(cv_image, corners, ids)
            for i, marker_corners in enumerate(corners):
                pts = marker_corners[0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                self.get_logger().info(f"Marker {ids[i][0]} centroid: ({cx}, {cy})")

                cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(cv_image, f'ID {ids[i][0]}', (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                
                depth_img = self.depth_image
                Z = depth_img[cy, cx] * 0.001  # Depth in mm
                if Z == 0:  # Ignore invalid depth points
                    self.get_logger().warn("Invalid depth at selected point!")
                    return

                # Convert to 3D coordinates
                X = (cx - self.camera_matrix[0, 2]) * Z / self.camera_matrix[0, 0]
                Y = (cy - self.camera_matrix[1, 2]) * Z / self.camera_matrix[1, 1]
                print(f"xyz: {X, Y, Z}")

                """TWC = np.array([[1,0,0, 0.400],
                        [0,0,-1, 1.000],
                        [0,-1,0,0.110],
                        [0,0,0,1]])
                poc = np.array([X, Y, Z, 1]).reshape(4,1)
                p = TWC@poc
                #labels.append(class_name)
                #detected_objs.append(p.flatten())
                p = p.flatten()"""
                centroid_msg = Float64MultiArray()
                centroid_msg.data = [0.0,0.0,0.0,X,Y,Z]
                self.centroid_pub.publish(centroid_msg)

        else:
            self.get_logger().info("No markers detected.")

        if self.show_detection:
            cv2.imshow("ArUco Tracking", cv_image)
            cv2.waitKey(1)

def main():
    rclpy.init()
    node = ArucoTracker()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()
