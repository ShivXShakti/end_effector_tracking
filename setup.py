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
            'aruco_tracker = end_effector_tracking.aruco_tracker:main',
            'plot_ee_pose_aruco = end_effector_tracking.plot_ee_pose_aruco:main',
            'liveplot_ee_position_fk = end_effector_tracking.liveplot_ee_position_fk:main',
            'ee_pose_savedata_plot = end_effector_tracking.ee_pose_savedata_plot:main',
        ],
    },
)
 