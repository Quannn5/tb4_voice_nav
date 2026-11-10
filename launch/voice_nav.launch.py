"""Launch the semantic navigator and print voice commander instructions."""

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tb4_voice_nav',
            executable='semantic_navigator',
            name='semantic_navigator',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        LogInfo(
            msg=(
                '\nSemantic navigator is starting with simulated time.\n'
                'In a separate interactive terminal, run:\n'
                '  source /opt/ros/humble/setup.bash\n'
                '  source ~/tb4_voice_nav/install/setup.bash\n'
                '  ros2 run tb4_voice_nav voice_commander\n'
                'Click the "Ghi âm 5 giây" button in the voice window.'
            )
        ),
    ])
