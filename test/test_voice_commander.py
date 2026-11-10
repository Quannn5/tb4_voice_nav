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

from pathlib import Path

import yaml

from tb4_voice_nav.voice_commander import normalize_text, parse_route


def _regions():
    config_path = (
        Path(__file__).resolve().parent.parent
        / 'config'
        / 'semantic_regions.yaml'
    )
    with config_path.open(encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)['regions']


def test_normalize_text_removes_vietnamese_diacritics():
    assert normalize_text('ĐI ĐẾN Phòng thí nghiệm') == (
        'di den phong thi nghiem'
    )


def test_parse_single_and_two_destination_routes():
    regions = _regions()
    assert parse_route('Đi đến phòng thí nghiệm', regions) == (
        ['phong_thi_nghiem'],
        [],
    )
    assert parse_route(
        'Di chuyển từ khu vực A đến khu vực B',
        regions,
    ) == (['khu_vuc_a', 'khu_vuc_b'], [])


def test_parse_multi_destination_route_in_spoken_order():
    assert parse_route(
        'Đi từ phòng A qua hành lang C rồi đến phòng B',
        _regions(),
    ) == (['khu_vuc_a', 'hanh_lang_c', 'khu_vuc_b'], [])


def test_parse_new_region_c_aliases():
    assert parse_route('Đi đến khu vực C', _regions()) == (
        ['khu_vuc_c'],
        [],
    )


def test_unknown_destination_is_reported_with_valid_destination():
    route, unknown = parse_route(
        'Đi từ khu vực A đến khu vực Z',
        _regions(),
    )

    assert route == ['khu_vuc_a']
    assert unknown == ['khu vuc z']
