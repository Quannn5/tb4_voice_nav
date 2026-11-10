# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest

from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker

from tb4_voice_nav.semantic_navigator import (
    _AMCL_POSE_QOS,
    _decode_route,
    _decode_route_command,
    _load_regions,
    _make_navigation_goal,
    _make_label_markers,
)
from rclpy.qos import QoSDurabilityPolicy, QoSReliabilityPolicy


def _regions():
    config_path = (
        Path(__file__).resolve().parent.parent
        / 'config'
        / 'semantic_regions.yaml'
    )
    return _load_regions(config_path)


def test_load_semantic_region_config():
    regions = _regions()
    assert set(regions) == {
        'khu_vuc_a',
        'khu_vuc_b',
        'khu_vuc_c',
        'phong_thi_nghiem',
        'hanh_lang_c',
    }


def test_decode_route_preserves_order():
    route = [
        'khu_vuc_a',
        'hanh_lang_c',
        'khu_vuc_b',
    ]
    assert _decode_route(json.dumps(route), _regions()) == route


def test_decode_route_command_preserves_voice_report_metadata():
    command = {
        'test_id': 'sample-id',
        'command': 'Đi đến khu vực A',
        'transcription': 'Đi đến khu vực A',
        'route': ['khu_vuc_a'],
    }

    route, metadata = _decode_route_command(
        json.dumps(command, ensure_ascii=False),
        _regions(),
    )

    assert route == ['khu_vuc_a']
    assert metadata['test_id'] == 'sample-id'
    assert metadata['command'] == 'Đi đến khu vực A'
    assert metadata['transcription'] == 'Đi đến khu vực A'


def test_navigation_goal_uses_nav2_action_goal_type():
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = 2.0
    pose.pose.position.y = -1.0

    goal = _make_navigation_goal(pose)

    assert isinstance(goal, NavigateToPose.Goal)
    assert goal.pose == pose


def test_amcl_pose_subscription_accepts_latched_pose():
    assert _AMCL_POSE_QOS.reliability == QoSReliabilityPolicy.RELIABLE
    assert _AMCL_POSE_QOS.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL


def test_region_labels_have_contrasting_text_and_background():
    background, label = _make_label_markers(
        0,
        'Khu vực A',
        -7.4,
        7.6,
        Time(),
        Duration(sec=1, nanosec=500000000),
    )

    assert background.type == Marker.CUBE
    assert background.color.r < 0.2
    assert background.color.g < 0.2
    assert background.color.b < 0.2
    assert label.type == Marker.TEXT_VIEW_FACING
    assert label.text == 'Khu vực A'
    assert label.color.r == label.color.g == label.color.b == 1.0
    assert label.pose.position.y > 7.6


@pytest.mark.parametrize(
    'payload',
    (
        '{not json}',
        '[]',
        '["khu_vuc_a", 1]',
        '["khu_vuc_z"]',
    ),
)
def test_decode_route_rejects_malformed_or_unknown_destination(payload):
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _decode_route(payload, _regions())
