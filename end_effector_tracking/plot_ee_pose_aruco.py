import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import numpy as np

class PosePlotter(Node):
    def __init__(self):
        super().__init__('pose_plotter')
        
        # Lists to store the trajectory
        self.x_vals = []
        self.y_vals = []
        self.z_vals = []

        # Smoothing buffer
        self.window_size = 5  # Adjust as needed
        self.x_window = deque(maxlen=self.window_size)
        self.y_window = deque(maxlen=self.window_size)
        self.z_window = deque(maxlen=self.window_size)

        self.create_subscription(Float64MultiArray, '/ur/ee_pose_aruco', self.pose_callback, 10)
        self.get_logger().info("PosePlotter node started.")

    def pose_callback(self, msg):
        if len(msg.data) >= 6:
            x = msg.data[3]
            z = msg.data[4]
            y = msg.data[5]

            self.x_window.append(x)
            self.y_window.append(y)
            self.z_window.append(z)

            # Compute moving average
            smooth_x = np.mean(self.x_window)
            smooth_y = np.mean(self.y_window)
            smooth_z = np.mean(self.z_window)

            self.x_vals.append(smooth_x)
            self.y_vals.append(smooth_y)
            self.z_vals.append(smooth_z)

            self.get_logger().info(f'Smoothed: x={smooth_x:.3f}, y={smooth_y:.3f}, z={smooth_z:.3f}')


def main():
    rclpy.init()
    node = PosePlotter()

    # Enable interactive mode
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize empty plot
    trajectory, = ax.plot([], [], [], label='Smoothed Trajectory', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Aruco: Live 3D Smoothed End Effector Trajectory")
    ax.legend()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.x_vals:
                trajectory.set_data(node.x_vals, node.y_vals)
                trajectory.set_3d_properties(node.z_vals)

                ax.relim()
                ax.autoscale_view()

                plt.draw()
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("Shutting down...")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
