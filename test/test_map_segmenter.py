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

import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from tb4_voice_nav.map_segmenter import _pixel_to_map, process_map


def test_pixel_to_map_flips_image_y_axis():
    assert _pixel_to_map(4, 1, 10, 0.05, [-2.0, 3.0, 0.0]) == [
        -1.8,
        3.4,
    ]


def test_process_map_writes_five_regions(tmp_path):
    maps_directory = tmp_path / 'maps'
    maps_directory.mkdir()
    image_path = maps_directory / 'map.pgm'
    map_yaml_path = maps_directory / 'map.yaml'
    output_image_path = maps_directory / 'segmented_map.png'
    output_yaml_path = tmp_path / 'config' / 'semantic_regions.yaml'
    image = np.zeros((240, 320), dtype=np.uint8)
    image[10:230, 10:310] = 254
    image[115:125, 100:220] = 0
    assert cv2.imwrite(str(image_path), image)
    map_yaml_path.write_text(
        yaml.safe_dump(
            {
                'image': 'map.pgm',
                'resolution': 0.05,
                'origin': [-2.0, 1.0, 0.0],
            }
        ),
        encoding='utf-8',
    )

    regions = process_map(
        map_yaml_path,
        output_image_path,
        output_yaml_path,
    )

    assert len(regions) == 5
    assert [region['name'] for region in regions] == [
        'Khu vực A',
        'Khu vực B',
        'Khu vực C',
        'Phòng thí nghiệm',
        'Hành lang',
    ]
    assert [region['route_id'] for region in regions] == [
        'khu_vuc_a',
        'khu_vuc_b',
        'khu_vuc_c',
        'phong_thi_nghiem',
        'hanh_lang_c',
    ]
    assert all(len(region['polygon']) >= 3 for region in regions)
    assert output_image_path.is_file()
    saved_data = yaml.safe_load(output_yaml_path.read_text(encoding='utf-8'))
    assert len(saved_data['regions']) == 5
    assert saved_data['frame_id'] == 'map'


def test_checked_in_goals_are_free_and_spread_across_the_map():
    package_root = Path(__file__).resolve().parent.parent
    with (package_root / 'maps' / 'map.yaml').open(encoding='utf-8') as file:
        map_metadata = yaml.safe_load(file)
    map_image = cv2.imread(
        str(package_root / 'maps' / map_metadata['image']),
        cv2.IMREAD_GRAYSCALE,
    )
    with (package_root / 'config' / 'semantic_regions.yaml').open(
        encoding='utf-8'
    ) as file:
        semantic_config = yaml.safe_load(file)

    assert map_image is not None
    regions = semantic_config['regions']
    assert len(regions) == 5
    goals = []
    for region in regions:
        x, y, _ = region['pose']
        pixel_x = round((x - map_metadata['origin'][0]) / map_metadata['resolution'])
        pixel_y = map_image.shape[0] - 1 - round(
            (y - map_metadata['origin'][1]) / map_metadata['resolution']
        )
        assert map_image[pixel_y, pixel_x] >= 250, region['route_id']
        goals.append((x, y))

    distances = [
        math.dist(first, second)
        for index, first in enumerate(goals)
        for second in goals[index + 1:]
    ]
    assert min(distances) >= 5.0
