# raceline_optimizer

Offline minimum-time raceline optimizer. It takes an occupancy map of the track and
produces a raceline together with the steering and throttle commands that realize it in
the vehicle model the MPC uses.

`track_optimizer.py` coordinates the pipeline:

1. `map_track.py` reads the PGM and ROS YAML map and extracts the closed track corridor and
   its centerline, with wall clearance for the real rear-axle footprint.
2. `min_curvature_seed.py` builds geometric starting lines (minimum curvature, optional
   length term) inside the corridor.
3. `command_ocp.py` solves the periodic problem with IPOPT through CasADi: the car state
   follows the 25 ms MPC step map exactly, with a C++ kernel (`command_ocp_kernel.cpp`,
   built on first use from `f1tenth_mpc/src/unity_source_plant.cpp`) providing values
   and derivatives of the plant.
4. `command_campaign.py` repeats this over several lap durations and starting points,
   checkpointing results so a campaign can stop and resume. `steering_seed.py`
   perturbs steering to generate starts; `adaptive_corner_search.py` selects
   corner perturbations; `equal_time_refinement.py` refines witnesses at fixed
   duration; `rigid_pose_polish.py` adjusts their placement in the map. The
   optional hybrid time predictor is part of `command_ocp.py`.
5. `command_realizability.py` and `path_frame.py` check a result against the plant and the
   reference frame the MPC uses. Only results that pass are exported.

`plant_library.py` builds and loads the production C++ plant through its C ABI.
`vehicle_model.py` builds the CasADi vehicle model and resolves parameters from
the production plant, reset data and Unity project
(`third_party/autodrive-unity-source/`).

## Setup and usage

```bash
python3 -m venv raceline_optimizer/.venv
raceline_optimizer/.venv/bin/pip install -r raceline_optimizer/requirements.txt

tools/raceline/run_raceline_campaign.sh f1tenth_planning/maps/competition.pgm \
    f1tenth_planning/maps/competition.yaml \
    --driving-reference f1tenth_planning/trajectories/competition/raceline.csv
```

Running the same command again resumes from the checkpoints. `--status`, `--watch` and
`--stop` report progress or stop the campaign; the dashboard in `tools/raceline/` shows it
live. A user systemd session is required by the launcher, including its
`--foreground` preflight. Results are written under `live_runs/` and are not copied into
`f1tenth_planning/trajectories/` automatically. Options, outputs and the validity checks are
described in [../docs/raceline-optimization.md](../docs/raceline-optimization.md).

## Tests

```bash
cd raceline_optimizer && .venv/bin/python -m pytest
```
