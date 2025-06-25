import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
import numpy as np
from std_msgs.msg import Float64MultiArray



class ArucoTracker(Node):
    def __init__(self, show_detection = True, leftarm_f = True, rightarm_f = True):
        super().__init__('aruco_tracker_node')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tag_size = 0.1  # 5 cm
        self.show_detection = show_detection
        self.leftarm_f = leftarm_f
        self.rightarm_f = rightarm_f
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
        depth_img = self.depth_image
        Xl,Yl,Zl,Xr,Yr,Zr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if self.leftarm_f and self.rightarm_f:
            dictionary_l = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
            parameters_l = aruco.DetectorParameters()
            detector_l = aruco.ArucoDetector(dictionary_l, parameters_l)
            corners_l, ids_l, _ = detector_l.detectMarkers(gray)

            dictionary_r = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
            parameters_r = aruco.DetectorParameters()
            detector_r = aruco.ArucoDetector(dictionary_r, parameters_r)
            corners_r, ids_r, _ = detector_r.detectMarkers(gray)
            
            if ids_l is not None:
                aruco.drawDetectedMarkers(cv_image, corners_l, ids_l)
                aruco.drawDetectedMarkers(cv_image, corners_r, ids_r)
                for i, (marker_corners_l, marker_corners_r) in enumerate(zip(corners_l,corners_r)):
                    ptsl = marker_corners_l[0]
                    cxl = int(np.mean(ptsl[:, 0]))
                    cyl = int(np.mean(ptsl[:, 1]))
                    self.get_logger().info(f"Marker {ids_l[i][0]} centroid: ({cxl}, {cyl})")

                    cv2.circle(cv_image, (cxl, cyl), 5, (0, 255, 0), -1)
                    cv2.putText(cv_image, f'ID {ids_l[i][0]}', (cxl + 10, cyl),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                    
                    Zl = depth_img[cyl, cxl] * 0.001  # Depth in mm
                    if Zl == 0:  # Ignore invalid depth points
                        self.get_logger().warn("Invalid depth at selected point!")
                        return

                    # Convert to 3D coordinates
                    Xl = (cxl - self.camera_matrix[0, 2]) * Zl / self.camera_matrix[0, 0]
                    Yl = (cyl - self.camera_matrix[1, 2]) * Zl / self.camera_matrix[1, 1]
                    print(f"xyz: {Xl, Yl, Zl}")

                    ptsr = marker_corners_r[0]
                    cxr = int(np.mean(ptsr[:, 0]))
                    cyr = int(np.mean(ptsr[:, 1]))
                    self.get_logger().info(f"Marker {ids_r[i][0]} centroid: ({cxr}, {cyr})")

                    cv2.circle(cv_image, (cxr, cyr), 5, (0, 255, 0), -1)
                    cv2.putText(cv_image, f'ID {ids_r[i][0]}', (cxr + 10, cyr),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                    
                    Zr = depth_img[cyr, cxr] * 0.001  # Depth in mm
                    if Zr == 0:  # Ignore invalid depth points
                        self.get_logger().warn("Invalid depth at selected point!")
                        return

                    # Convert to 3D coordinates
                    Xr = (cxr - self.camera_matrix[0, 2]) * Zr / self.camera_matrix[0, 0]
                    Yr = (cyr - self.camera_matrix[1, 2]) * Zr / self.camera_matrix[1, 1]
                    print(f"xyz: {Xr, Yr, Zr}")
                    
                    centroid_msg = Float64MultiArray()
                    centroid_msg.data = [Xl,Yl,Zl,Xr,Yr,Zr]
                    self.centroid_pub.publish(centroid_msg)
            else:
                self.get_logger().info("No markers detected.")

        elif self.leftarm_f:
            dictionary_l = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
            parameters_l = aruco.DetectorParameters()
            detector_l = aruco.ArucoDetector(dictionary_l, parameters_l)
            corners_l, ids_l, _ = detector_l.detectMarkers(gray)
            if ids_l is not None:
                aruco.drawDetectedMarkers(cv_image, corners_l, ids_l)
                for i, marker_corners_l in enumerate(corners_l):
                    ptsl = marker_corners_l[0]
                    cxl = int(np.mean(ptsl[:, 0]))
                    cyl = int(np.mean(ptsl[:, 1]))
                    self.get_logger().info(f"Marker {ids_l[i][0]} centroid: ({cxl}, {cyl})")

                    cv2.circle(cv_image, (cxl, cyl), 5, (0, 255, 0), -1)
                    cv2.putText(cv_image, f'ID {ids_l[i][0]}', (cxl + 10, cyl),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                    
                    Zl = depth_img[cyl, cxl] * 0.001  # Depth in mm
                    if Zl == 0:  # Ignore invalid depth points
                        self.get_logger().warn("Invalid depth at selected point!")
                        return

                    # Convert to 3D coordinates
                    Xl = (cxl - self.camera_matrix[0, 2]) * Zl / self.camera_matrix[0, 0]
                    Yl = (cyl - self.camera_matrix[1, 2]) * Zl / self.camera_matrix[1, 1]
                    print(f"xyz: {Xl, Yl, Zl}")
                    centroid_msg = Float64MultiArray()
                    centroid_msg.data = [Xl,Yl,Zl,Xr,Yr,Zr]
                    self.centroid_pub.publish(centroid_msg)
            else:
                self.get_logger().info("No markers detected.")
        elif self.rightarm_f:
            dictionary_r = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
            parameters_r = aruco.DetectorParameters()
            detector_r = aruco.ArucoDetector(dictionary_r, parameters_r)
            corners_r, ids_r, _ = detector_r.detectMarkers(gray)
            if ids_r is not None:
                aruco.drawDetectedMarkers(cv_image, corners_r, ids_r)
                for i, marker_corners_r in enumerate(corners_r):
                    ptsr = marker_corners_r[0]
                    cxr = int(np.mean(ptsr[:, 0]))
                    cyr = int(np.mean(ptsr[:, 1]))
                    self.get_logger().info(f"Marker {ids_r[i][0]} centroid: ({cxr}, {cyr})")

                    cv2.circle(cv_image, (cxr, cyr), 5, (0, 255, 0), -1)
                    cv2.putText(cv_image, f'ID {ids_r[i][0]}', (cxr + 10, cyr),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                    
                    Zr =  -depth_img[cyr, cxr] * 0.001  # Depth in mm
                    if Zr == 0:  # Ignore invalid depth points
                        self.get_logger().warn("Invalid depth at selected point!")
                        return

                    # Convert to 3D coordinates
                    Xr = (cxr - self.camera_matrix[0, 2]) * Zr / self.camera_matrix[0, 0]
                    Yr = (cyr - self.camera_matrix[1, 2]) * Zr / self.camera_matrix[1, 1]
                    print(f"xyz: {Xr, Yr, Zr}")
                    centroid_msg = Float64MultiArray()
                    centroid_msg.data = [Xl,Yl,Zl,Xr,Yr,Zr]
                    self.centroid_pub.publish(centroid_msg)
            else:
                self.get_logger().info("No markers detected.")



        """dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
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

                '''TWC = np.array([[1,0,0, 0.400],
                        [0,0,-1, 1.000],
                        [0,-1,0,0.110],
                        [0,0,0,1]])
                poc = np.array([X, Y, Z, 1]).reshape(4,1)
                p = TWC@poc
                #labels.append(class_name)
                #detected_objs.append(p.flatten())
                p = p.flatten()'''
                centroid_msg = Float64MultiArray()
                centroid_msg.data = [0.0,0.0,0.0,X,Y,Z]
                self.centroid_pub.publish(centroid_msg)

        else:
            self.get_logger().info("No markers detected.")"""

        if self.show_detection:
            cv2.imshow("ArUco Tracking", cv_image)
            cv2.waitKey(1)

def main():
    rclpy.init()
    node = ArucoTracker()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()
