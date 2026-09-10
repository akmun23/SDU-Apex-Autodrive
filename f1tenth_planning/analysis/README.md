# Local analysis outputs

This directory is intentionally empty in the repository. Live CSV files,
rosbags, logs, plots, and score reports belong here while a run is being
investigated, but they are ignored and must not become source inputs.

The production inputs are kept separately in:

- `f1tenth_planning/maps/autodrive_track_ftg_commit_20260909_025m.yaml`
- `f1tenth_planning/trajectories/autodrive_track_ftg_commit_20260909_025m_mintime_raceline.csv`

Use a new subdirectory for each actual simulator run. Ground truth may be
recorded for offline scoring only; it must not be fed into odometry,
localization, or control.
