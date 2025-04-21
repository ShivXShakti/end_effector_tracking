import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='end_effector_tracking',
            executable='aruco_tracker',
            output='screen'
        ),
        launch_ros.actions.Node(
            package='end_effector_tracking',
            executable='plot_ee_position',
            output='screen'
        ),
        launch_ros.actions.Node(
            package='end_effector_tracking',
            executable='plot_ee_position_fk',
            output='screen'
        ),
    ])

