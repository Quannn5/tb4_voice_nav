# TurtleBot 4 Voice Navigation

ROS 2 Humble package for Vietnamese voice commands, semantic destinations, and
sequential Nav2 navigation. The package is designed for Ubuntu 22.04 and
TurtleBot 4.

## System architecture

| Component | Responsibility |
| --- | --- |
| `map_segmenter` | Segments the free space in a map and writes the visualization and `config/semantic_regions.yaml`. |
| `voice_commander` | Captures a Vietnamese command (or reads terminal text), resolves destination aliases, and publishes a JSON route. |
| `semantic_navigator` | Loads semantic poses, publishes RViz markers, validates the route, and sends Nav2 goals one at a time. |
| Nav2 / AMCL | Localizes the robot and plans and executes each destination goal. |
| RViz2 | Displays the map, semantic destination markers, labels, and planned route. |

### Data flow

```text
Microphone (Enter / push-to-talk)        Terminal text (debug mode)
                  \                           /
                   +------ voice_commander ---+
                              |
                              | std_msgs/msg/String
                              | /voice_nav/route_command
                              | JSON: ["khu_vuc_a", "hanh_lang_c"]
                              v
semantic_regions.yaml --> semantic_navigator --> Nav2 NavigateToPose
                              |                         |
                              | MarkerArray              | feedback / result
                              v                         v
                 /semantic_map_markers            AMCL / robot
                              |
                              v
                       RViz2 (map frame)
```

The navigator publishes a `MarkerArray` at 1 Hz to `/semantic_map_markers`.
Region labels use `TEXT_VIEW_FACING`, destination poses use cylinders, and an
active route is shown as a `LINE_STRIP` from the localized robot pose through
the remaining destinations. Route commands are JSON arrays of `route_id`
strings in travel order. A destination not present in the semantic YAML is
rejected before navigation.

## Map coordinates: pixels to ROS `map`

`map_segmenter` reads `resolution` and `[origin_x, origin_y, yaw]` from the map
YAML. For image pixel `(u, v)`, image height `H`, and resolution `r`, this
package uses:

```text
x_map = origin_x + u * r
y_map = origin_y + (H - 1 - v) * r
```

Image `u` increases rightward and `v` increases downward, so the `H - 1 - v`
term flips the image's vertical axis into the ROS map's upward-positive `y`.
The generated semantic pose is `[x_map, y_map, 0.0]` and its polygon vertices
are converted to meters with the same mapping. This formula assumes the map
origin yaw is zero, as it is in the supplied map. For a nonzero origin yaw, the
pixel offsets must additionally be rotated by that yaw before adding the
origin.

For example, with resolution `0.05 m/pixel`, origin `[-10, -10, 0]`, image
height `402`, and pixel `(u, v) = (52, 49)`:

```text
x_map = -10 + 52 * 0.05             = -7.40 m
y_map = -10 + (402 - 1 - 49) * 0.05 =  7.60 m
```

The pose, aliases, and `route_id` values consumed by the nodes are stored in
[`config/semantic_regions.yaml`](config/semantic_regions.yaml).

### Scientific report assets

Running `map_segmenter` also creates `report_assets/figures/` and
`report_assets/data/` alongside the input map's parent directory by default.
Pass `--report-dir PATH` to choose another output directory. The figures
`01_raw_map.png` through `06_final_semantic_map.png` are exported at 300 DPI;
the last figure distinguishes the geometric centroid (`x`) from the safe
navigation goal (blue circle). Region area is computed as pixel count times
resolution squared, and safety distance is the distance-transform value at the
safe goal multiplied by resolution.

The report tables are `region_statistics.csv` (UTF-8 with BOM for spreadsheet
compatibility) and `region_table.tex` (UTF-8 LaTeX source). The LaTeX table
contains Vietnamese text and can be included in a Unicode-capable LaTeX
document such as one compiled with XeLaTeX.

The checked-in map has five distributed destinations. “Khu vực C” is the
upper-right area of the map; the other destinations are Khu vực A (upper-left),
Khu vực B (near the middle), Phòng thí nghiệm (lower-left), and Hành lang
(lower-right). Their configured map poses are:

| Destination | `route_id` | Map pose `(x, y)` in meters |
| --- | --- | --- |
| Khu vực A | `khu_vuc_a` | `(-7.40, 7.60)` |
| Khu vực B | `khu_vuc_b` | `(-3.00, 3.05)` |
| Khu vực C | `khu_vuc_c` | `(7.05, 8.00)` |
| Phòng thí nghiệm | `phong_thi_nghiem` | `(-5.95, -3.75)` |
| Hành lang | `hanh_lang_c` | `(6.05, -2.00)` |

These are goal poses, not guarantees that Nav2 will use a particular path.
The map loaded by Nav2 must match the map used to generate the semantic YAML;
otherwise the displayed destinations and navigation goals may not line up.

## Build and run in TurtleBot 4 Gazebo

The example below uses the TurtleBot 4 simulation and combined
Gazebo + localization + Nav2 launch available in the
`tb4_navigation_practice` workspace. Make sure ROS 2 Humble, TurtleBot 4
simulation/navigation packages, and that workspace are installed and built.

Build this package:

```bash
source /opt/ros/humble/setup.bash
cd ~/tb4_voice_nav
colcon build --packages-select tb4_voice_nav
```

### Terminal 1: Gazebo, localization, and Nav2

```bash
source /opt/ros/humble/setup.bash
source ~/tb4_ws/install/setup.bash
ros2 launch tb4_navigation_practice tb4_navigation.launch.py \
  world:=maze \
  map:="$(ros2 pkg prefix tb4_navigation_practice)/share/tb4_navigation_practice/maps/map.yaml" \
  use_sim_time:=true \
  spawn_assignment_objects:=false
```

Wait for Gazebo and Nav2 to finish starting and for localization to be
available. The map used for localization must match the simulated world.
If using a different Gazebo world or map, replace `world` and `map` together.

### Terminal 2: RViz2

```bash
source /opt/ros/humble/setup.bash
source ~/tb4_ws/install/setup.bash
ros2 launch turtlebot4_viz view_robot.launch.py use_sim_time:=true
```

In RViz2, set **Fixed Frame** to `map` and add a **MarkerArray** display for
`/semantic_map_markers`. The map and robot displays may be provided by the
TurtleBot visualization launch.

### Terminal 3: semantic navigation

```bash
source /opt/ros/humble/setup.bash
source ~/tb4_voice_nav/install/setup.bash
ros2 launch tb4_voice_nav voice_nav.launch.py
```

This starts `semantic_navigator` with `use_sim_time: true` and prints the
instructions for the interactive voice process.

### Terminal 4: voice commands

Open a separate terminal. `voice_commander` opens a window with a button; click
**Ghi âm 5 giây**, speak your destination, then the recognized text appears in
the window and a valid route is sent to the navigator:

```bash
source /opt/ros/humble/setup.bash
source ~/tb4_voice_nav/install/setup.bash
ros2 run tb4_voice_nav voice_commander
```

For text-only debugging without the window:

```bash
ros2 run tb4_voice_nav voice_commander --ros-args -p use_text_input:=True
```

Google Speech Recognition (`vi-VN`) is the default and needs network access.
Vosk or faster-whisper can be selected as an offline backend; install the
chosen Python package and model before configuring the corresponding
`stt_backend`, `offline_fallback`, or `vosk_model_path` parameter.

Each input attempt is recorded in
[`report_assets/data/test_results.md`](report_assets/data/test_results.md):
unknown or unrecognized commands are logged by `voice_commander`, while valid
routes are logged by `semantic_navigator` when navigation finishes. The output
path can be overridden on either node with the `report_results_file` parameter.

## Topics and parameters

| Topic / parameter | Type / default | Purpose |
| --- | --- | --- |
| `/voice_nav/route_command` | `std_msgs/msg/String` | JSON array of semantic `route_id` values. |
| `/semantic_map_markers` | `visualization_msgs/msg/MarkerArray` | Region labels, destination markers, and active route line in `map`. |
| `semantic_regions_file` | Installed `config/semantic_regions.yaml` | Semantic labels, aliases, poses, and polygons. |
| `use_text_input` (`voice_commander`) | `False` | Set `True` to type commands instead of recording audio. |

Build and focused tests:

```bash
cd ~/tb4_voice_nav
colcon build --packages-select tb4_voice_nav
source /opt/ros/humble/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  test/test_map_segmenter.py \
  test/test_voice_commander.py \
  test/test_semantic_navigator.py
```
