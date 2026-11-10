"""Publish semantic map markers and execute voice-commanded Nav2 routes."""

import json
import math
from pathlib import Path
import time
from typing import Any

import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav2_simple_commander.robot_navigator import BasicNavigator
from action_msgs.msg import GoalStatus
from rclpy.duration import Duration
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.task import Future
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from tb4_voice_nav.reporting import append_test_result, default_results_path


_ARRIVAL_TOLERANCE = 0.4
_FEEDBACK_LOG_INTERVAL = 2.0
_AMCL_POSE_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


def _default_config_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory('tb4_voice_nav'))
            / 'config'
            / 'semantic_regions.yaml'
        )
    except (ImportError, LookupError):
        source_root = Path(__file__).resolve().parent.parent
        return source_root / 'config' / 'semantic_regions.yaml'


def _load_regions(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open(encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get('regions'), list):
        raise ValueError('Semantic region configuration must contain a regions list.')

    regions: dict[str, dict[str, Any]] = {}
    for region in config['regions']:
        if not isinstance(region, dict):
            raise ValueError('Each semantic region must be a mapping.')
        route_id = region.get('route_id')
        name = region.get('name')
        pose = region.get('pose')
        if not isinstance(route_id, str) or not route_id:
            raise ValueError('Each semantic region must have a non-empty route_id.')
        if not isinstance(name, str) or not name:
            raise ValueError(f'Semantic region {route_id!r} has no display name.')
        if (
            not isinstance(pose, list)
            or len(pose) != 3
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in pose
            )
        ):
            raise ValueError(
                f"Semantic region '{route_id}' pose must be [x, y, yaw]."
            )
        if route_id in regions:
            raise ValueError(f'Duplicate semantic region route_id: {route_id}')
        regions[route_id] = region
    return regions


def _decode_route(
    payload: str,
    regions: dict[str, dict[str, Any]],
) -> list[str]:
    route = json.loads(payload)
    if isinstance(route, dict):
        route = route.get('route')
    if not isinstance(route, list) or not route:
        raise ValueError('Route command must be a non-empty JSON array.')
    if not all(isinstance(route_id, str) for route_id in route):
        raise ValueError('Every route destination must be a string route_id.')
    unknown_ids = [route_id for route_id in route if route_id not in regions]
    if unknown_ids:
        raise ValueError(
            'Route contains unknown semantic region(s): '
            + ', '.join(unknown_ids)
        )
    return route


def _decode_route_command(
    payload: str,
    regions: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    decoded = json.loads(payload)
    metadata: dict[str, str] = {}
    if isinstance(decoded, dict):
        for key in ('test_id', 'command', 'transcription'):
            value = decoded.get(key, '')
            if not isinstance(value, str):
                raise ValueError(f'Route metadata field {key!r} must be a string.')
            metadata[key] = value
        route_payload = json.dumps(decoded.get('route'))
    else:
        route_payload = payload
    return _decode_route(route_payload, regions), metadata


def _make_navigation_goal(pose: PoseStamped) -> NavigateToPose.Goal:
    goal = NavigateToPose.Goal()
    goal.pose = pose
    return goal


def _make_label_markers(
    marker_id: int,
    name: str,
    x: float,
    y: float,
    stamp: Any,
    lifetime: Any,
) -> tuple[Marker, Marker]:
    offsets = (
        (0.0, 0.8),
        (-1.5, 0.2),
        (0.5, 0.8),
        (0.0, 0.8),
    )
    offset_x, offset_y = offsets[marker_id % len(offsets)]
    label_x = x + offset_x
    label_y = y + offset_y

    background = Marker()
    background.header.frame_id = 'map'
    background.header.stamp = stamp
    background.ns = 'semantic_region_label_backgrounds'
    background.id = marker_id
    background.type = Marker.CUBE
    background.action = Marker.ADD
    background.pose.position.x = label_x
    background.pose.position.y = label_y
    background.pose.position.z = 0.62
    background.pose.orientation.w = 1.0
    background.scale.x = max(1.2, len(name) * 0.19 + 0.35)
    background.scale.y = 0.48
    background.scale.z = 0.04
    background.color.r = 0.08
    background.color.g = 0.12
    background.color.b = 0.18
    background.color.a = 0.92
    background.lifetime = lifetime

    label = Marker()
    label.header.frame_id = 'map'
    label.header.stamp = stamp
    label.ns = 'semantic_region_labels'
    label.id = marker_id
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose.position.x = label_x
    label.pose.position.y = label_y
    label.pose.position.z = 0.78
    label.pose.orientation.w = 1.0
    label.scale.z = 0.32
    label.color.r = 1.0
    label.color.g = 1.0
    label.color.b = 1.0
    label.color.a = 1.0
    label.text = name
    label.lifetime = lifetime

    return background, label


class SemanticNavigator(BasicNavigator):
    """Render semantic regions and navigate validated routes sequentially."""

    def __init__(self) -> None:
        super().__init__(node_name='semantic_navigator')
        self.declare_parameter(
            'semantic_regions_file',
            str(_default_config_path()),
        )
        self.declare_parameter(
            'report_results_file',
            str(default_results_path()),
        )
        config_path = Path(
            str(self.get_parameter('semantic_regions_file').value)
        ).expanduser()
        self.report_results_file = Path(
            str(self.get_parameter('report_results_file').value)
        ).expanduser()
        try:
            self.regions = _load_regions(config_path)
        except (OSError, yaml.YAMLError, ValueError) as error:
            self.get_logger().error(
                f'Cannot load semantic regions from {config_path}: {error}'
            )
            raise

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/semantic_map_markers',
            10,
        )
        self.route_subscription = self.create_subscription(
            String,
            '/voice_nav/route_command',
            self._route_callback,
            10,
        )
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self._pose_callback,
            _AMCL_POSE_QOS,
        )
        self._marker_timer = self.create_timer(1.0, self._publish_markers)
        self._navigation_timer = self.create_timer(0.1, self._navigation_tick)

        self._current_pose: PoseStamped | None = None
        self._route: list[str] = []
        self._route_metadata: dict[str, str] = {}
        self._route_started_at: float | None = None
        self._route_index = 0
        self._goal_handle: ClientGoalHandle | None = None
        self._goal_request_pending = False
        self._latest_distance_remaining: float | None = None
        self._last_feedback_log_at = 0.0
        self._waiting_for_pose_logged = False
        self._waiting_for_server_logged = False
        self._preview_delete_pending = False
        self.get_logger().info(
            f'Loaded {len(self.regions)} semantic regions from {config_path}.'
        )

    def _pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        pose = PoseStamped()
        pose.header = message.header
        pose.pose = message.pose.pose
        self._current_pose = pose
        if self._waiting_for_pose_logged:
            self.get_logger().info('Received robot localization from AMCL.')
            self._waiting_for_pose_logged = False

    def _make_marker(self, namespace: str, marker_id: int, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=1.5).to_msg()
        return marker

    def _publish_markers(self) -> None:
        marker_array = MarkerArray()
        colors = (
            (0.15, 0.55, 1.0),
            (0.15, 0.8, 0.3),
            (1.0, 0.55, 0.1),
            (0.75, 0.2, 0.9),
        )
        for marker_id, (route_id, region) in enumerate(self.regions.items()):
            x, y, _ = region['pose']
            red, green, blue = colors[marker_id % len(colors)]

            destination = self._make_marker(
                'semantic_region_goals',
                marker_id,
                Marker.CYLINDER,
            )
            destination.pose.position.x = float(x)
            destination.pose.position.y = float(y)
            destination.pose.position.z = 0.12
            destination.scale.x = 0.28
            destination.scale.y = 0.28
            destination.scale.z = 0.24
            destination.color.r = red
            destination.color.g = green
            destination.color.b = blue
            destination.color.a = 0.9
            destination.text = route_id
            marker_array.markers.append(destination)

            label_background, label = _make_label_markers(
                marker_id,
                str(region['name']),
                float(x),
                float(y),
                self.get_clock().now().to_msg(),
                Duration(seconds=1.5).to_msg(),
            )
            marker_array.markers.append(label_background)
            marker_array.markers.append(label)

        if self._route:
            preview = self._make_marker('planned_route', 0, Marker.LINE_STRIP)
            preview.pose.position.z = 0.03
            preview.scale.x = 0.06
            preview.color.r = 1.0
            preview.color.g = 0.1
            preview.color.b = 0.1
            preview.color.a = 0.9
            if self._current_pose is not None:
                preview.points.append(
                    self._point_from_pose(self._current_pose)
                )
            for route_id in self._route[self._route_index:]:
                x, y, _ = self.regions[route_id]['pose']
                preview.points.append(
                    Point(x=float(x), y=float(y), z=0.03)
                )
            if len(preview.points) >= 2:
                marker_array.markers.append(preview)
        elif self._preview_delete_pending:
            preview = self._make_marker('planned_route', 0, Marker.LINE_STRIP)
            preview.action = Marker.DELETE
            marker_array.markers.append(preview)
            self._preview_delete_pending = False

        self.marker_publisher.publish(marker_array)

    @staticmethod
    def _point_from_pose(pose: PoseStamped) -> Point:
        point = Point()
        point.x = pose.pose.position.x
        point.y = pose.pose.position.y
        point.z = 0.03
        return point

    def _route_callback(self, message: String) -> None:
        try:
            route, metadata = _decode_route_command(
                message.data,
                self.regions,
            )
        except (json.JSONDecodeError, ValueError) as error:
            self.get_logger().error(f'Invalid route command: {error}')
            return

        if self._route or self._goal_handle is not None or self._goal_request_pending:
            self.get_logger().warning(
                'A route is already in progress; the new route was rejected.'
            )
            self._append_test_result(
                route,
                metadata,
                'Báo lỗi: robot đang thực hiện lộ trình khác',
                0.0,
            )
            return

        self._route = route
        self._route_metadata = metadata
        self._route_started_at = time.monotonic()
        self._route_index = 0
        self._latest_distance_remaining = None
        self._preview_delete_pending = False
        self.get_logger().info(
            'Accepted route: ' + ' -> '.join(route)
        )
        self._publish_markers()
        self._navigation_tick()

    def _navigation_tick(self) -> None:
        if not self._route:
            return
        if self._goal_handle is not None or self._goal_request_pending:
            self._log_feedback_if_due()
            return
        if self._current_pose is None:
            if not self._waiting_for_pose_logged:
                self.get_logger().warning(
                    'Waiting for robot localization on amcl_pose.'
                )
                self._waiting_for_pose_logged = True
            return

        if self._route_index >= len(self._route):
            self.get_logger().info('Route completed successfully.')
            self._finish_test_result('Thành công')
            self._clear_route()
            return

        if not self.nav_to_pose_client.server_is_ready():
            if not self._waiting_for_server_logged:
                self.get_logger().info(
                    "Waiting for Nav2 'navigate_to_pose' action server."
                )
                self._waiting_for_server_logged = True
            return
        self._waiting_for_server_logged = False

        route_id = self._route[self._route_index]
        region = self.regions[route_id]
        x, y, _ = region['pose']
        if self._route_index == 0:
            distance = math.hypot(
                self._current_pose.pose.position.x - float(x),
                self._current_pose.pose.position.y - float(y),
            )
            if distance <= _ARRIVAL_TOLERANCE:
                self.get_logger().info(
                    f"Already within {_ARRIVAL_TOLERANCE:.1f} m of "
                    f"'{region['name']}' ({distance:.2f} m); "
                    'continuing to the next destination.'
                )
                self._route_index += 1
                return

        goal = _make_navigation_goal(self._make_goal(region))
        self.get_logger().info(
            f"Navigating to '{region['name']}' "
            f'({self._route_index + 1}/{len(self._route)}).'
        )
        self._publish_markers()
        self._goal_request_pending = True
        goal_future = self.nav_to_pose_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback,
        )
        goal_future.add_done_callback(self._goal_response_callback)

    def _make_goal(self, region: dict[str, Any]) -> PoseStamped:
        x, y, yaw = region['pose']
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.z = math.sin(float(yaw) / 2.0)
        goal.pose.orientation.w = math.cos(float(yaw) / 2.0)
        return goal

    def _goal_response_callback(self, future: Future) -> None:
        self._goal_request_pending = False
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(f'Failed to send Nav2 goal: {error}')
            self._finish_test_result('Báo lỗi: không gửi được goal Nav2')
            self._clear_route()
            return
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the destination goal.')
            self._finish_test_result('Báo lỗi: Nav2 từ chối goal')
            self._clear_route()
            return

        self._goal_handle = goal_handle
        self._latest_distance_remaining = None
        self._last_feedback_log_at = time.monotonic()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _feedback_callback(self, feedback_message: Any) -> None:
        self._latest_distance_remaining = float(
            feedback_message.feedback.distance_remaining
        )

    def _log_feedback_if_due(self) -> None:
        if self._goal_handle is None or self._latest_distance_remaining is None:
            return
        now = time.monotonic()
        if now - self._last_feedback_log_at < _FEEDBACK_LOG_INTERVAL:
            return
        self.get_logger().info(
            f'Distance remaining: {self._latest_distance_remaining:.2f} m'
        )
        self._last_feedback_log_at = now

    def _goal_result_callback(self, future: Future) -> None:
        self._goal_handle = None
        try:
            result = future.result()
        except Exception as error:
            self.get_logger().error(f'Failed to receive Nav2 result: {error}')
            self._finish_test_result('Báo lỗi: không nhận được kết quả Nav2')
            self._clear_route()
            return

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            region = self.regions[self._route[self._route_index]]
            self.get_logger().info(f"Reached '{region['name']}'.")
            self._route_index += 1
            self._latest_distance_remaining = None
            return
        if result.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning('Nav2 navigation was canceled.')
            outcome = 'Báo lỗi: điều hướng bị hủy'
        else:
            self.get_logger().error(
                f'Nav2 navigation failed with action status {result.status}.'
            )
            outcome = f'Báo lỗi Nav2 (status {result.status})'
        self._finish_test_result(outcome)
        self._clear_route()

    def _append_test_result(
        self,
        route: list[str],
        metadata: dict[str, str],
        outcome: str,
        duration: float,
    ) -> None:
        coordinates = []
        for route_id in route:
            region = self.regions.get(route_id)
            if region is not None:
                x, y, _ = region['pose']
                coordinates.append(
                    f"{region['name']}: ({float(x):.2f}, {float(y):.2f})"
                )
        try:
            append_test_result(
                self.report_results_file,
                {
                    'command': metadata.get('command', 'Không rõ'),
                    'transcription': metadata.get(
                        'transcription',
                        'Lệnh ROS trực tiếp',
                    ),
                    'route': ' -> '.join(route),
                    'coordinates': '; '.join(coordinates) or '—',
                    'duration': f'{duration:.2f}',
                    'outcome': outcome,
                },
            )
        except OSError as error:
            self.get_logger().error(
                f'Cannot append test result to '
                f'{self.report_results_file}: {error}'
            )

    def _finish_test_result(self, outcome: str) -> None:
        if not self._route:
            return
        duration = (
            time.monotonic() - self._route_started_at
            if self._route_started_at is not None
            else 0.0
        )
        self._append_test_result(
            self._route,
            self._route_metadata,
            outcome,
            duration,
        )

    def _clear_route(self) -> None:
        self._route.clear()
        self._route_index = 0
        self._route_metadata = {}
        self._route_started_at = None
        self._goal_handle = None
        self._goal_request_pending = False
        self._latest_distance_remaining = None
        self._preview_delete_pending = True


def main(args: list[str] | None = None) -> None:
    """Run the semantic marker publisher and Nav2 route executor."""
    rclpy.init(args=args)
    navigator = None
    try:
        navigator = SemanticNavigator()
        rclpy.spin(navigator)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if navigator is not None:
            navigator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
