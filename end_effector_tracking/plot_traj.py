import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
import threading

class CentroidPlotter(Node):
    def __init__(self):
        super().__init__('centroid_plotter')
        self.subscription = self.create_subscription(
            Point,
            '/aruco_centroid',
            self.listener_callback,
            10
        )
        self.x_data = []
        self.y_data = []
        self.get_logger().info("Centroid plotter node started.")

    def listener_callback(self, msg):
        self.x_data.append(msg.x)
        self.y_data.append(msg.y)
        self.get_logger().info(f'Received centroid: ({msg.x}, {msg.y})')

def ros_thread():
    rclpy.spin(node)

def start_plotting(node):
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'go-', label='Centroid Trajectory')
    ax.set_title('ArUco Marker Centroid Trajectory')
    ax.set_xlabel('cx (pixels)')
    ax.set_ylabel('cy (pixels)')
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()

    while rclpy.ok():
        if node.x_data:
            line.set_data(node.x_data, node.y_data)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        plt.pause(0.1)

def main():
    global node
    rclpy.init()
    node = CentroidPlotter()

    # Start ROS in a background thread
    thread = threading.Thread(target=ros_thread, daemon=True)
    thread.start()

    try:
        start_plotting(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
