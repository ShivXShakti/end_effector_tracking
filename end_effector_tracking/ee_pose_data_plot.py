import rclpy
from rclpy.node import Node
import os
import csv
import matplotlib.pyplot as plt
import time
from dualarm_custom_msgs.msg import TrajStatus      # Custom message



class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('ee_sub')
        self.create_subscription(TrajStatus, '/fk/left_right/ee_pose', self.fk_ee_callback, 10)
        self.create_subscription(TrajStatus, '/desired/left_right/ee_pose', self.desired_ee_callback, 10)
        
        self.fk_ee_pose_f = False
        self.desired_ee_pose_f =False
        
        self.execution_f = False

        self.create_timer(2, self.timer_callback) # 2sec

        self.file_path_fk = '/home/cstar/Documents/dual_arm_ws/src/dual_arm_control_py/data/fk_ee_pose_1.csv'
        self.file_path_desired = '/home/cstar/Documents/dual_arm_ws/src/dual_arm_control_py/data/desired_ee_pose_1.csv'
        self.create_csv(self.file_path_fk)
        self.create_csv(self.file_path_desired)

        #-----plot ------
        self.xal_data = []
        self.yal_data = []
        self.zal_data = []
        self.xar_data = []
        self.yar_data = []
        self.zar_data = []

        self.xdl_data = []
        self.ydl_data = []
        self.zdl_data = []
        self.xdr_data = []
        self.ydr_data = []
        self.zdr_data = []
        self.counter = 0
        self.t = []

        self.fk_pose_ee_status = None
        self.desired_pose_ee_status= None
        self.plot_2d = True
        self.plot_3d = True
        self.plot = True
        
        
    def create_csv(self, file_path):
        if not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['pose_{}'.format(i) for i in range(6)])

    def fk_ee_callback(self, msg):
        self.fk_ee_pose_f = True
        self.fk_pose_ee_status = msg.status
        with open(self.file_path_fk, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(msg.data.data)
        self.xal_data.append(msg.data.data[0])
        self.yal_data.append(msg.data.data[1])
        self.zal_data.append(msg.data.data[2])
        self.xar_data.append(msg.data.data[6])
        self.yar_data.append(msg.data.data[7])
        self.zar_data.append(msg.data.data[8])
        self.get_logger().info(f"fk saved pose: {msg.data.data}")
    
    def desired_ee_callback(self, msg):
        self.desired_ee_pose_f = True
        self.desired_pose_ee_status = msg.status
        with open(self.file_path_desired, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(msg.data.data)
        self.xdl_data.append(msg.data.data[0])
        self.ydl_data.append(msg.data.data[1])
        self.zdl_data.append(msg.data.data[2])
        self.xdr_data.append(msg.data.data[6])
        self.ydr_data.append(msg.data.data[7])
        self.zdr_data.append(msg.data.data[8])
        
        self.t.append(self.counter)
        self.counter +=1
        self.get_logger().info(f"desired saved pose: {msg.data.data}")

    def plot_2d_data(self):
        fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3)
        ax1.plot(self.t, self.xdl_data)
        ax1.plot(self.t, self.xal_data)
        ax1.set_title("Left arm: X")

        ax2.plot(self.t, self.ydl_data)
        ax2.plot(self.t, self.yal_data)
        ax2.set_title("Left arm: Y")

        ax3.plot(self.t, self.zdl_data)
        ax3.plot(self.t, self.zal_data)
        ax1.set_title("Left arm: Z")

        ax4.plot(self.t, self.xdr_data)
        ax4.plot(self.t, self.xar_data)
        ax4.set_title("Right arm: X")

        ax5.plot(self.t, self.ydr_data)
        ax5.plot(self.t, self.yar_data)
        ax5.set_title("Right arm: Y")

        ax6.plot(self.t, self.zdr_data)
        ax6.plot(self.t, self.zar_data)
        ax6.set_title("Right arm: Z")
        fig.tight_layout()
        plt.show()
    def plot_3d_data(self):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D traj')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.plot3D(self.xdl_data,self.ydl_data,self.zdl_data, color='red', linewidth=2)
        ax.plot3D(self.xal_data,self.yal_data,self.zal_data, color='blue', linewidth=2)
        plt.show()

    def timer_callback(self):
        if not self.fk_ee_pose_f and not self.desired_ee_pose_f:
            self.get_logger().info(f"topic: '/fk/left_right/ee_pose' and '/desired/left_right/ee_pose' are not publishing data.")
        elif not self.fk_pose_ee_status and not self.desired_pose_ee_status:
            self.get_logger().info(f"Plotting....")
            if self.plot_2d:
                self.plot_2d_data()
            if self.plot_3d:
                self.plot_3d_data()
            self.plot_2d = False
            self.plot_3d = False
            time.sleep(5)

def main(args = None):
    rclpy.init(args=args)
    node = PoseSubscriber()
    try:
        rclpy.spin(node=node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == "__main__":
    main()