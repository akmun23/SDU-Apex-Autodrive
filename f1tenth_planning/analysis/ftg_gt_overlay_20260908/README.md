# FTG mapping run: ground-truth and LiDAR overlay

This directory contains the diagnostics extracted from `/tmp/ftg_odom_lap_bag` and the saved map `autodrive_track_ftg_centroid_20260908`.

The main plot is [gt_overlay_exact_map_values.png](gt_overlay_exact_map_values.png). It uses the map values exactly:

- black = occupied (`0`)
- grey = unknown (`205`)
- white = free (`254`)

The ground-truth path is plotted in the saved-map coordinate frame. The car did **not** complete a valid directed lap: it came within 0.14 m of the start after 62.03 m, but its heading differed from the departure heading by 3.02 rad (approximately pi), proving that this was a U-turn/retrace. Collision count stayed zero. The simulator lap-count topic stayed at zero, so it was not used as the completion signal.

Useful files:

- [gt_path_unknown_cell_mask.png](gt_path_unknown_cell_mask.png): unknown cells highlighted purple, with GT path.
- [gt_finite_lidar_on_saved_map.png](gt_finite_lidar_on_saved_map.png): raw finite LiDAR hits and processed hits against the exact map values.
- [gt_world_path_and_lidar.png](gt_world_path_and_lidar.png): GT trajectory and LiDAR returns without an occupancy background.
- [gt_trajectory.csv](gt_trajectory.csv): timestamped GT pose, heading, and velocity.
- [ftg_diagnostics.csv](ftg_diagnostics.csv): one FTG decision record per processed scan.
- [lidar_scan_summary.csv](lidar_scan_summary.csv) and [processed_scan_summary.csv](processed_scan_summary.csv): per-scan valid-beam counts and range extrema.
- `raw_lidar_hit_points.npz` and `processed_hit_points.npz`: GT-transformed finite-return point clouds (`x_m`, `y_m`, `time_s`, `scan_index`).
- `collision_count.csv`, `lap_count.csv`, [cmd_speed.csv](cmd_speed.csv), and [actuator_feedback.csv](actuator_feedback.csv): run-state and command evidence.

LiDAR geometry in this run was 1081 beams over 270 degrees, 0.06--10 m range, with the mapping mount offset `(0.2733, 0, 0)` m. Odom and LiDAR each contributed 2859 records; their recorded mean rate was 14.59 Hz, median 14.80 Hz, and maximum inter-message gap 0.152 s.

The NPZ point clouds are generated diagnostics; simulator ground truth was not used by the runtime controller.
