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
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)
             
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        #print(f"dictionary: {dictionary},   parameters: {parameters},  detector: {detector}, ids: {ids}")

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            for marker_corners in corners:
                pts = marker_corners[0]  # shape (4,2)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                print(f"cx: {cx}, cy:{cy}")

                cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(cv_image, f'Centroid: ({cx}, {cy})', (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        else:
            print("No markers detected.")
        if self.show_detection:
            cv2.imshow("Detected ArUco Marker", cv_image)
            print("Press any key in the image window to exit...")
            while True:
                if cv2.waitKey(10) != -1:
                    break
            cv2.destroyAllWindows()

    def rotation_matrix_to_quaternion(self, R):
        return list(cv2.Rodrigues(R)[0].flatten()) + [1.0] 

def main():
    rclpy.init()
    node = ArucoTracker()
    rclpy.spin(node)
    rclpy.shutdown()
