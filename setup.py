from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'end_effector_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kd',
    maintainer_email='kuldeeplakhansons@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dummy_camera_pub = end_effector_tracking.dummy_camera_pub:main',
            'aruco_tracker = end_effector_tracking.aruco_tracker:main',
            'msg_test = end_effector_tracking.msg_test:main',
            'plot_ee_position = end_effector_tracking.plot_ee_position:main',
            'plot_ee_position_fk = end_effector_tracking.plot_ee_position_fk:main',
            'ee_pose_data_plot = end_effector_tracking.ee_pose_data_plot:main',
        ],
    },
)
 