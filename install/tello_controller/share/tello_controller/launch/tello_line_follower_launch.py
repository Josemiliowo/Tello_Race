import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    controller_share = get_package_share_directory('tello_controller')
    line_follower_share = get_package_share_directory('tello_line_follower')

    controller_params = os.path.join(controller_share, 'config', 'controller_params.yaml')
    line_follower_params = os.path.join(
        line_follower_share, 'config', 'line_follower_params.yaml')

    simulate_arg = DeclareLaunchArgument(
        'simulate',
        default_value='false',
        description='Run tello_controller without connecting to a real Tello.',
    )

    enable_motors_arg = DeclareLaunchArgument(
        'enable_motors',
        default_value='true',
        description=(
            'If false, tello_controller still connects and streams video but ignores '
            'takeoff requests and RC control commands (safe for camera-only checks).'
        ),
    )

    controller_node = Node(
        package='tello_controller',
        executable='tello_controller',
        name='tello_controller',
        output='screen',
        parameters=[
            controller_params,
            {
                'simulate': LaunchConfiguration('simulate'),
                'enable_motors': LaunchConfiguration('enable_motors'),
            },
        ],
    )

    tag_args = [
        DeclareLaunchArgument(
            'left_turn_ids', default_value='[2]',
            description='AprilTag IDs that trigger a 90 deg left turn, e.g. "[2,5]". [-1] disables.'),
        DeclareLaunchArgument(
            'right_turn_ids', default_value='[3]',
            description='AprilTag IDs that trigger a 90 deg right turn. [-1] disables.'),
        DeclareLaunchArgument(
            'land_ids', default_value='[4]',
            description='AprilTag IDs that make the drone land. [-1] disables.'),
    ]

    line_follower_node = Node(
        package='tello_line_follower',
        executable='tello_line_follower',
        name='tello_line_follower',
        output='screen',
        parameters=[
            line_follower_params,
            {
                'left_turn_ids': LaunchConfiguration('left_turn_ids'),
                'right_turn_ids': LaunchConfiguration('right_turn_ids'),
                'land_ids': LaunchConfiguration('land_ids'),
            },
        ],
    )

    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Open the tello_gui window (video, live gain sliders, takeoff/land).',
    )

    gui_node = Node(
        package='tello_line_follower',
        executable='tello_gui',
        name='tello_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    return LaunchDescription([
        simulate_arg,
        enable_motors_arg,
        *tag_args,
        gui_arg,
        controller_node,
        line_follower_node,
        gui_node,
    ])
