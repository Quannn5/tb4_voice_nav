"""Segment a ROS occupancy map into navigable semantic regions."""

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


_DEFAULT_REGIONS = (
    {
        'route_id': 'khu_vuc_a',
        'name': 'Khu vực A',
        'aliases': ['khu vuc a', 'khu vực a', 'khu vực A', 'phong a', 'phòng A'],
        'seed_fraction': (0.13, 0.12),
    },
    {
        'route_id': 'khu_vuc_b',
        'name': 'Khu vực B',
        'aliases': ['khu vuc b', 'khu vực b', 'khu vực B', 'phong b', 'phòng B'],
        'seed_fraction': (0.35, 0.35),
    },
    {
        'route_id': 'khu_vuc_c',
        'name': 'Khu vực C',
        'aliases': ['khu vuc c', 'khu vực c', 'khu vực C', 'phong c', 'phòng C'],
        'seed_fraction': (0.90, 0.05),
    },
    {
        'route_id': 'phong_thi_nghiem',
        'name': 'Phòng thí nghiệm',
        'aliases': ['phong thi nghiem', 'phòng thí nghiệm', 'lab'],
        'seed_fraction': (0.20, 0.69),
    },
    {
        'route_id': 'hanh_lang_c',
        'name': 'Hành lang',
        'aliases': [
            'hanh lang',
            'hành lang',
            'corridor',
            'hanh lang c',
            'hành lang C',
        ],
        'seed_fraction': (0.80, 0.60),
    },
)


def _package_root() -> Path:
    source_root = Path(__file__).resolve().parent.parent
    if (source_root / 'maps' / 'map.yaml').is_file():
        return source_root

    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return source_root
    return Path(get_package_share_directory('tb4_voice_nav'))


def _read_map_metadata(map_yaml_path: Path) -> tuple[Path, float, list[float]]:
    with map_yaml_path.open(encoding='utf-8') as yaml_file:
        metadata = yaml.safe_load(yaml_file)
    if not isinstance(metadata, dict):
        raise ValueError(f'Map YAML must contain a mapping: {map_yaml_path}')

    resolution = metadata.get('resolution')
    origin = metadata.get('origin')
    if not isinstance(resolution, (int, float)) or not math.isfinite(resolution):
        raise ValueError('Map YAML must contain a finite numeric resolution.')
    if resolution <= 0:
        raise ValueError('Map resolution must be greater than zero.')
    if (
        not isinstance(origin, list)
        or len(origin) != 3
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in origin
        )
    ):
        raise ValueError(
            'Map YAML origin must contain three finite numbers: [x, y, yaw].'
        )

    image_value = metadata.get('image', 'map.pgm')
    if not isinstance(image_value, str) or not image_value:
        raise ValueError('Map YAML image must be a non-empty path.')
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = map_yaml_path.parent / image_path
    return image_path, float(resolution), [float(value) for value in origin]


def _select_seeds(
    free_mask: np.ndarray,
    distance: np.ndarray,
    region_count: int,
) -> list[tuple[int, int]]:
    height, width = free_mask.shape
    free_y, free_x = np.where(free_mask[1:-1, 1:-1] != 0)
    free_y += 1
    free_x += 1
    if free_x.size < region_count:
        raise ValueError(
            f'Free space has only {free_x.size} pixels; '
            f'cannot seed {region_count} regions.'
        )

    seeds: list[tuple[int, int]] = []
    for region in _DEFAULT_REGIONS[:region_count]:
        target_x = int(round(region['seed_fraction'][0] * (width - 1)))
        target_y = int(round(region['seed_fraction'][1] * (height - 1)))
        window_x = max(1, int(width * 0.10))
        window_y = max(1, int(height * 0.10))
        candidates = (
            (np.abs(free_x - target_x) <= window_x)
            & (np.abs(free_y - target_y) <= window_y)
        )
        candidate_indices = np.flatnonzero(candidates)
        if candidate_indices.size == 0:
            raise ValueError(
                f'No free-space pixels near the configured seed for '
                f"{region['name']}."
            )
        candidate_clearance = distance[
            free_y[candidate_indices],
            free_x[candidate_indices],
        ]
        best_index = int(candidate_indices[int(np.argmax(candidate_clearance))])
        seeds.append((int(free_y[best_index]), int(free_x[best_index])))

    while len(seeds) < region_count:
        nearest_seed_distance = np.full(free_x.shape, np.inf, dtype=np.float64)
        for seed_y, seed_x in seeds:
            squared_distance = (
                (free_x - seed_x) ** 2 + (free_y - seed_y) ** 2
            )
            np.minimum(
                nearest_seed_distance,
                squared_distance,
                out=nearest_seed_distance,
            )
        seed_score = nearest_seed_distance * np.maximum(
            distance[free_y, free_x],
            1.0,
        )
        for seed_y, seed_x in seeds:
            seed_score[(free_y == seed_y) & (free_x == seed_x)] = -1
        seed_index = int(np.argmax(seed_score))
        seeds.append((int(free_y[seed_index]), int(free_x[seed_index])))

    return seeds


def _segment(
    image: np.ndarray,
    region_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    binary_mask = np.where(image >= 250, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    threshold_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    free_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)
    if not np.any(free_mask):
        raise ValueError('No free-space pixels (intensity >= 250) were found.')

    padded_mask = cv2.copyMakeBorder(
        free_mask,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[
        1:-1,
        1:-1,
    ]
    seeds = _select_seeds(free_mask, distance, region_count)

    normalized_distance = cv2.normalize(
        distance,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    ).astype(np.uint8)
    elevation = 255 - normalized_distance
    markers = np.where(free_mask == 0, 1, 0).astype(np.int32)
    for label, (seed_y, seed_x) in enumerate(seeds, start=2):
        seed_support = np.zeros(free_mask.shape, dtype=np.uint8)
        cv2.circle(seed_support, (seed_x, seed_y), 3, 255, -1)
        seed_pixels = (seed_support != 0) & (free_mask != 0) & (markers == 0)
        markers[seed_pixels] = label

    seed_markers = markers.copy()
    watershed_image = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_image, markers)
    return (
        markers,
        distance,
        seeds,
        threshold_mask,
        free_mask,
        seed_markers,
    )


def _pixel_to_map(
    u: int,
    v: int,
    image_height: int,
    resolution: float,
    origin: list[float],
) -> list[float]:
    return [
        origin[0] + u * resolution,
        origin[1] + (image_height - 1 - v) * resolution,
    ]


def _region_metadata(
    markers: np.ndarray,
    distance: np.ndarray,
    resolution: float,
    origin: list[float],
    seeds: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[tuple[np.ndarray, tuple[int, int]]]]:
    height, _ = markers.shape
    regions: list[dict[str, Any]] = []
    render_data: list[tuple[np.ndarray, tuple[int, int]]] = []

    for region_id, (seed_y, seed_x) in enumerate(seeds, start=1):
        label = region_id + 1
        if region_id <= len(_DEFAULT_REGIONS):
            region_config = _DEFAULT_REGIONS[region_id - 1]
            region_name = region_config['name']
            route_id = region_config['route_id']
            aliases = list(region_config['aliases'])
        else:
            region_name = f'Khu vực {region_id}'
            route_id = f'khu_vuc_{region_id}'
            aliases = [f'khu vuc {region_id}', region_name.lower()]

        region_mask = np.where(markers == label, 255, 0).astype(np.uint8)
        if not np.any(region_mask):
            continue

        goal_v, goal_u = seed_y, seed_x
        if markers[goal_v, goal_u] != label:
            region_y, region_x = np.where(region_mask != 0)
            nearest_index = int(np.argmin(
                (region_y - seed_y) ** 2 + (region_x - seed_x) ** 2
            ))
            goal_v = int(region_y[nearest_index])
            goal_u = int(region_x[nearest_index])
        if distance[goal_v, goal_u] <= 0:
            raise ValueError(
                f"Selected goal for '{region_name}' "
                'is not in free space.'
            )
        goal_xy = _pixel_to_map(goal_u, goal_v, height, resolution, origin)

        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        polygon_pixels = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        polygon = [
            [
                float(value)
                for value in _pixel_to_map(
                    int(point[0]), int(point[1]), height, resolution, origin
                )
            ]
            for point in polygon_pixels
        ]
        if len(polygon) < 3:
            polygon = [
                [
                    float(value)
                    for value in _pixel_to_map(
                        int(point[0]), int(point[1]), height, resolution, origin
                    )
                ]
                for point in contour.reshape(-1, 2)
            ]

        regions.append(
            {
                'id': region_id,
                'route_id': route_id,
                'name': region_name,
                'aliases': aliases,
                'pose': [goal_xy[0], goal_xy[1], 0.0],
                'polygon': polygon,
            }
        )
        render_data.append((region_mask, (goal_u, goal_v)))

    return regions, render_data


def _region_statistics(
    regions: list[dict[str, Any]],
    render_data: list[tuple[np.ndarray, tuple[int, int]]],
    distance: np.ndarray,
    resolution: float,
) -> list[dict[str, Any]]:
    statistics = []
    for region, (region_mask, (goal_u, goal_v)) in zip(regions, render_data):
        moments = cv2.moments(region_mask, binaryImage=True)
        if moments['m00'] <= 0:
            raise ValueError(
                f"Region '{region['name']}' has no pixels for its centroid."
            )
        centroid_u = moments['m10'] / moments['m00']
        centroid_v = moments['m01'] / moments['m00']
        goal_x, goal_y, goal_yaw = region['pose']
        statistics.append(
            {
                'region_id': region['id'],
                'region_name': region['name'],
                'area_pixels': int(moments['m00']),
                'area_m2': float(moments['m00']) * resolution**2,
                'centroid_u': centroid_u,
                'centroid_v': centroid_v,
                'goal_x': goal_x,
                'goal_y': goal_y,
                'goal_yaw': goal_yaw,
                'safe_distance_m': float(distance[goal_v, goal_u]) * resolution,
                'centroid_pixel': (centroid_u, centroid_v),
            }
        )
    return statistics


def _write_report_table(
    statistics: list[dict[str, Any]],
    data_directory: Path,
) -> None:
    data_directory.mkdir(parents=True, exist_ok=True)
    csv_path = data_directory / 'region_statistics.csv'
    columns = [
        ('region_id', 'Region ID'),
        ('region_name', 'Tên khu vực'),
        ('area_pixels', 'Diện tích (pixel)'),
        ('area_m2', 'Diện tích thực (m²)'),
        ('centroid_pixel', 'Tọa độ tâm Pixel (u, v)'),
        ('goal_pose', 'Tọa độ đích ROS (x, y, yaw) (m)'),
        ('safe_distance_m', 'Khoảng cách an toàn tới vật cản gần nhất (m)'),
    ]

    def format_pixel(value: dict[str, Any]) -> str:
        return f"({value['centroid_u']:.2f}, {value['centroid_v']:.2f})"

    def format_pose(value: dict[str, Any]) -> str:
        return (
            f"({value['goal_x']:.3f}, {value['goal_y']:.3f}, "
            f"{value['goal_yaw']:.3f})"
        )

    rows = []
    for statistic in statistics:
        row = dict(statistic)
        row['centroid_pixel'] = format_pixel(statistic)
        row['goal_pose'] = format_pose(statistic)
        rows.append(row)

    with csv_path.open('w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[key for key, _ in columns],
            extrasaction='ignore',
            lineterminator='\n',
        )
        writer.writerow({key: title for key, title in columns})
        writer.writerows(rows)

    def latex_escape(value: str) -> str:
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
        }
        return ''.join(replacements.get(character, character) for character in value)

    tex_path = data_directory / 'region_table.tex'
    headers = [latex_escape(title) for _, title in columns]
    with tex_path.open('w', encoding='utf-8') as tex_file:
        tex_file.write('\\begin{table}[htbp]\n\\centering\n')
        tex_file.write('\\caption{Thống kê các vùng ngữ nghĩa trên bản đồ}\n')
        tex_file.write('\\label{tab:region-statistics}\n')
        tex_file.write('\\begin{tabular}{r l r r l l r}\n\\hline\n')
        tex_file.write(' & '.join(headers) + r' \\' + '\n\\hline\n')
        for statistic in statistics:
            values = [
                str(statistic['region_id']),
                latex_escape(statistic['region_name']),
                str(statistic['area_pixels']),
                f"{statistic['area_m2']:.3f}",
                format_pixel(statistic),
                format_pose(statistic),
                f"{statistic['safe_distance_m']:.3f}",
            ]
            tex_file.write(' & '.join(values) + r' \\' + '\n')
        tex_file.write('\\hline\n\\end{tabular}\n\\end{table}\n')


def _write_report_figures(
    figures_directory: Path,
    image: np.ndarray,
    threshold_mask: np.ndarray,
    morphology_mask: np.ndarray,
    distance: np.ndarray,
    seed_markers: np.ndarray,
    regions: list[dict[str, Any]],
    render_data: list[tuple[np.ndarray, tuple[int, int]]],
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figures_directory.mkdir(parents=True, exist_ok=True)

    def save_figure(filename: str) -> None:
        plt.tight_layout()
        plt.savefig(
            figures_directory / filename,
            dpi=300,
            bbox_inches='tight',
            metadata={'Software': 'tb4_voice_nav map_segmenter'},
        )
        plt.close()

    height, width = image.shape
    pixel_extent = (-0.5, width - 0.5, height - 0.5, -0.5)

    plt.figure(figsize=(8, 6))
    axes = plt.gca()
    raw_image = axes.imshow(
        image,
        cmap='gray',
        vmin=0,
        vmax=255,
        extent=pixel_extent,
    )
    axes.set_title('01 - Bản đồ gốc (PGM)')
    axes.set_xlabel('Pixel u (cột)')
    axes.set_ylabel('Pixel v (hàng)')
    axes.annotate(
        'gốc ảnh (0, 0)',
        xy=(0, 0),
        xytext=(max(8, width * 0.08), max(8, height * 0.08)),
        color='yellow',
        arrowprops={'arrowstyle': '->', 'color': 'yellow'},
    )
    plt.colorbar(raw_image, ax=axes, label='Cường độ pixel')
    save_figure('01_raw_map.png')

    for filename, title, mask in (
        (
            '02_binary_threshold.png',
            '02 - Free space sau ngưỡng và Opening khử nhiễu',
            threshold_mask,
        ),
        (
            '03_morphology.png',
            '03 - Free space sau Opening và Closing',
            morphology_mask,
        ),
    ):
        plt.figure(figsize=(8, 6))
        axes = plt.gca()
        plotted = axes.imshow(
            mask,
            cmap=ListedColormap(['black', '#f4f4f4']),
            vmin=0,
            vmax=255,
            extent=pixel_extent,
        )
        axes.set_title(title)
        axes.set_xlabel('Pixel u (cột)')
        axes.set_ylabel('Pixel v (hàng)')
        colorbar = plt.colorbar(plotted, ax=axes, ticks=[0, 255])
        colorbar.ax.set_yticklabels(['Occupied / unknown', 'Free'])
        save_figure(filename)

    plt.figure(figsize=(8, 6))
    axes = plt.gca()
    heatmap = axes.imshow(
        distance,
        cmap='viridis',
        extent=pixel_extent,
    )
    axes.set_title('04 - Euclidean Distance Transform')
    axes.set_xlabel('Pixel u (cột)')
    axes.set_ylabel('Pixel v (hàng)')
    plt.colorbar(heatmap, ax=axes, label='Khoảng cách tới vật cản (pixel)')
    save_figure('04_distance_transform.png')

    plt.figure(figsize=(8, 6))
    axes = plt.gca()
    axes.imshow(
        image,
        cmap='gray',
        vmin=0,
        vmax=255,
        extent=pixel_extent,
    )
    seed_labels = seed_markers.copy()
    seed_labels[seed_labels <= 1] = 0
    marker_map = axes.imshow(
        np.ma.masked_where(seed_labels == 0, seed_labels),
        cmap='tab10',
        alpha=0.8,
        extent=pixel_extent,
        interpolation='nearest',
    )
    seed_y, seed_x = np.where(seed_labels > 1)
    axes.scatter(seed_x, seed_y, marker='+', c='red', s=45, label='Seed center')
    for index, (region, (_, (goal_u, goal_v))) in enumerate(
        zip(regions, render_data),
        start=1,
    ):
        axes.annotate(
            f"{index}: {region['name']}",
            (goal_u, goal_v),
            xytext=(5, 5),
            textcoords='offset points',
            color='white',
            fontsize=8,
            bbox={'facecolor': 'black', 'alpha': 0.65, 'pad': 2},
        )
    axes.set_title('05 - Seed markers trước Watershed')
    axes.set_xlabel('Pixel u (cột)')
    axes.set_ylabel('Pixel v (hàng)')
    axes.legend(loc='lower right')
    plt.colorbar(marker_map, ax=axes, label='Nhãn seed')
    save_figure('05_watershed_markers.png')

    palette = [
        '#f6a04d',
        '#75cf75',
        '#5da9e9',
        '#d582d5',
        '#e6d65c',
        '#49c5b6',
        '#d97565',
        '#a38be8',
    ]
    label_image = np.zeros(image.shape, dtype=np.uint8)
    for region, (region_mask, _) in zip(regions, render_data):
        label_image[region_mask != 0] = region['id']
    colors = ['#d3d3d3'] + [
        palette[(region['id'] - 1) % len(palette)] for region in regions
    ]
    plt.figure(figsize=(10, 8))
    axes = plt.gca()
    axes.imshow(
        image,
        cmap='gray',
        vmin=0,
        vmax=255,
        extent=pixel_extent,
    )
    axes.imshow(
        np.ma.masked_where(label_image == 0, label_image),
        cmap=ListedColormap(colors),
        alpha=0.55,
        vmin=0,
        vmax=max(1, len(regions)),
        extent=pixel_extent,
        interpolation='nearest',
    )

    legend_handles = []
    for region, (region_mask, (goal_u, goal_v)) in zip(regions, render_data):
        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            contour_points = contour.reshape(-1, 2)
            axes.plot(
                contour_points[:, 0],
                contour_points[:, 1],
                color='red',
                linewidth=0.8,
            )
        moments = cv2.moments(region_mask, binaryImage=True)
        centroid_u = moments['m10'] / moments['m00']
        centroid_v = moments['m01'] / moments['m00']
        axes.scatter(
            centroid_u,
            centroid_v,
            marker='x',
            color='black',
            s=55,
            linewidths=1.5,
        )
        axes.scatter(
            goal_u,
            goal_v,
            marker='o',
            facecolors='none',
            edgecolors='blue',
            s=65,
            linewidths=1.5,
        )
        axes.annotate(
            region['name'],
            (goal_u, goal_v),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=8,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2},
        )
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                color=palette[(region['id'] - 1) % len(palette)],
                marker='s',
                linestyle='',
                label=region['name'],
            )
        )
    legend_handles.extend(
        [
            plt.Line2D([], [], color='red', label='Ranh giới Watershed'),
            plt.Line2D(
                [],
                [],
                color='black',
                marker='x',
                linestyle='',
                label='Centroid hình học',
            ),
            plt.Line2D(
                [],
                [],
                color='blue',
                marker='o',
                markerfacecolor='none',
                linestyle='',
                label='Safe goal',
            ),
        ]
    )
    axes.set_title('06 - Bản đồ phân vùng ngữ nghĩa')
    axes.set_xlabel('Pixel u (cột)')
    axes.set_ylabel('Pixel v (hàng)')
    axes.legend(handles=legend_handles, loc='upper left', fontsize=8)
    save_figure('06_final_semantic_map.png')


def process_map(
    map_yaml_path: Path,
    output_image_path: Path,
    output_yaml_path: Path,
    region_count: int = 5,
    report_directory: Path | None = None,
) -> list[dict[str, Any]]:
    """Segment a map and write its visualization and semantic region YAML."""
    map_yaml_path = map_yaml_path.expanduser().resolve()
    image_path, resolution, origin = _read_map_metadata(map_yaml_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Could not read map image: {image_path}')
    if region_count < 1:
        raise ValueError('Region count must be at least one.')

    (
        markers,
        distance,
        seeds,
        threshold_mask,
        morphology_mask,
        seed_markers,
    ) = _segment(image, region_count)
    regions, render_data = _region_metadata(
        markers,
        distance,
        resolution,
        origin,
        seeds,
    )
    statistics = _region_statistics(
        regions,
        render_data,
        distance,
        resolution,
    )
    if report_directory is None:
        report_directory = map_yaml_path.parent.parent / 'report_assets'
    report_directory = report_directory.expanduser().resolve()
    _write_report_figures(
        report_directory / 'figures',
        image,
        threshold_mask,
        morphology_mask,
        distance,
        seed_markers,
        regions,
        render_data,
    )
    _write_report_table(statistics, report_directory / 'data')

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = [
        (50, 180, 255),
        (80, 220, 100),
        (255, 150, 70),
        (220, 90, 200),
        (60, 220, 220),
    ]
    for region, (region_mask, goal) in zip(regions, render_data):
        color = colors[(region['id'] - 1) % len(colors)]
        overlay = np.zeros_like(visualization)
        overlay[:] = color
        selected = region_mask != 0
        visualization[selected] = cv2.addWeighted(
            visualization[selected],
            0.45,
            overlay[selected],
            0.55,
            0,
        )
        cv2.circle(visualization, goal, 7, (0, 0, 255), -1)
        label = region['aliases'][0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(label, font, 0.5, 2)
        text_x = goal[0] + 8
        if text_x + text_size[0] >= visualization.shape[1]:
            text_x = max(0, goal[0] - text_size[0] - 8)
        text_y = max(
            text_size[1],
            min(goal[1] - 8, visualization.shape[0] - baseline),
        )
        cv2.putText(
            visualization,
            label,
            (text_x, text_y),
            font,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_image_path), visualization):
        raise OSError(f'Could not write segmented map: {output_image_path}')
    with output_yaml_path.open('w', encoding='utf-8') as yaml_file:
        yaml.safe_dump(
            {
                'frame_id': 'map',
                'resolution': resolution,
                'regions': regions,
            },
            yaml_file,
            allow_unicode=True,
            sort_keys=False,
        )

    return regions


def main(args: list[str] | None = None) -> None:
    """Run map segmentation from the command line."""
    package_root = _package_root()
    parser = argparse.ArgumentParser(
        description='Segment free space in a ROS map into semantic regions.'
    )
    parser.add_argument(
        '--map-yaml',
        type=Path,
        default=package_root / 'maps' / 'map.yaml',
        help='ROS map YAML file (default: maps/map.yaml).',
    )
    parser.add_argument(
        '--output-image',
        type=Path,
        default=package_root / 'maps' / 'segmented_map.png',
        help='Visualization output (default: maps/segmented_map.png).',
    )
    parser.add_argument(
        '--output-yaml',
        type=Path,
        default=package_root / 'config' / 'semantic_regions.yaml',
        help='Semantic regions output (default: config/semantic_regions.yaml).',
    )
    parser.add_argument(
        '--regions',
        type=int,
        default=5,
        help='Number of distributed semantic regions to seed (default: 5).',
    )
    parser.add_argument(
        '--report-dir',
        type=Path,
        default=package_root / 'report_assets',
        help='Directory for 300-DPI report figures and statistics.',
    )
    options = parser.parse_args(args)

    regions = process_map(
        options.map_yaml,
        options.output_image,
        options.output_yaml,
        options.regions,
        options.report_dir,
    )
    for region in regions:
        x, y, yaw = region['pose']
        print(
            f"{region['id']}: {region['name']} "
            f"map=({x:.3f}, {y:.3f}, {yaw:.3f})"
        )
    print(f'Saved visualization: {options.output_image}')
    print(f'Saved semantic regions: {options.output_yaml}')
    print(f'Saved report figures: {options.report_dir / "figures"}')
    print(f'Saved report data: {options.report_dir / "data"}')


if __name__ == '__main__':
    main()
