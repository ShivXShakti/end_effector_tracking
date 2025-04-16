import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt

class PosePlotter(Node):
    def __init__(self):
        super().__init__('pose_plotter')
        self.x_vals = []
        self.y_vals = []
        self.z_vals = []

        self.create_subscription(Float64MultiArray, '/ur/ee_pose_aruco', self.pose_callback, 10)
        self.get_logger().info("PosePlotter node started.")

    def pose_callback(self, msg):
        if len(msg.data) >= 6:
            x = msg.data[3]
            y = msg.data[4]
            z = msg.data[5]
            self.x_vals.append(x)
            self.y_vals.append(y)
            self.z_vals.append(z)
            self.get_logger().info(f'Received: x={x}, y={y}, z={z}')

def main():
    rclpy.init()
    node = PosePlotter()

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.x_vals:
                ax.clear()
                ax.plot(node.x_vals, node.y_vals, node.z_vals, label='Trajectory', color='blue')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title("Live 3D End Effector Trajectory")
                ax.legend()
                plt.draw()
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("Shutting down...")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
