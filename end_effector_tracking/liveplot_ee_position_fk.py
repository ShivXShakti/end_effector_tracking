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
        self.window_size = 20  # Adjust as needed
        self.x_window = deque(maxlen=self.window_size)
        self.y_window = deque(maxlen=self.window_size)
        self.z_window = deque(maxlen=self.window_size)

        self.create_subscription(Float64MultiArray, '/fk/left_right/ee_pose', self.fk_pose_callback, 10)
        self.create_subscription(Float64MultiArray, '/desired/left_right/ee_pose', self.desired_pose_callback, 10)
        self.get_logger().info("PosePlotter node started.")

    def fk_pose_callback(self, msg):
        if len(msg.data) >= 6:
            x = msg.data[0]
            y = msg.data[1]
            z = msg.data[2]

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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize empty plot
    trajectory, = ax.plot([], [], [], label='Smoothed Trajectory', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("FK: Live 3D Smoothed End Effector Trajectory")
    ax.legend()

    ax.set_xlim([-1, 5])  # Set X axis limits
    ax.set_ylim([-3, 3])  # Set Y axis limits
    ax.set_zlim([-3, 1])   # Set Z axis limits

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

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
