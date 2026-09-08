# F1Tenth Planning

Trajectory optimization for the AutoDRIVE RoboRacer. It takes the five-lap
SLAM map and produces a raceline CSV consumed directly by Pure Pursuit (and
also usable by MPC).

## Quick Start

```bash
# 1. Drive five FTG laps; the robot stops, waits for loop closure to settle,
#    and SLAM Toolbox writes both files.
ros2 launch sdu_apex_autodrive mapping.launch.py

# 2. Generate the AutoDRIVE mintime raceline using the in-code settings.
python3 f1tenth_planning/scripts/optimize_trajectory.py

# 3. Follow the saved raceline with encoder/IMU odometry and custom AMCL.
ros2 launch sdu_apex_autodrive controller.launch.py controller:=pure_pursuit
```

The output lands in `f1tenth_planning/trajectories/<track>_raceline.csv`.

## What the Pipeline Does

`optimize_trajectory.py` runs four steps automatically:

1. **Extract centerline** — reads the `.pgm` map, finds track boundaries, computes the GVD centerline, and measures track widths via ray-cast.
2. **Optimize raceline** — feeds the centerline + widths into the [TUM global trajectory optimizer](global_racetrajectory_optimization/) in `mintime` mode, using the BachelorProject settings and the AutoDRIVE vehicle model.
3. **Convert to MPC format** — rotates headings by π/2 (TUM→F1Tenth convention), clamps velocities, and writes a comma-separated CSV.
4. **Add wall distances** — ray-casts perpendicular to each waypoint to get `d_left` and `d_right` for the MPC's lateral error tracking.

## Optimizer settings

`optimize_trajectory.py` is the runnable mintime entry point. It intentionally
uses the same in-code settings as the BachelorProject implementation and does
not accept command-line overrides. Edit the user-settings block in the script
only when deliberately changing an experiment.

```bash
python3 f1tenth_planning/scripts/optimize_trajectory.py
```

The two smoothing values have different scopes and both match the
BachelorProject setup:

- `racecar.ini:reg_smooth_opts.s_reg = 3.0` is TUM's spline regression value.
- `optimizer_smoothing_s = 6.0` is the prepared centerline smoother before TUM.
- `optimizer_smoothing_k = 2`, `stepsize_prep = 0.02`, `stepsize_reg = 0.08`,
  and final waypoint spacing `0.02 m` are also fixed in the settings block.

## Output Format

The raceline CSV has 9 columns:

```
# s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2, d_left_m, d_right_m
0.000, 4.317, -4.834, -3.035, 0.012, 3.50, 1.20, 0.18, 0.22
...
```

| Column | Description |
|--------|-------------|
| `s_m` | Arc length along the raceline [m] |
| `x_m`, `y_m` | World coordinates [m] |
| `psi_rad` | Heading angle [rad] |
| `kappa_radpm` | Curvature [rad/m] |
| `vx_mps` | Target velocity [m/s] |
| `ax_mps2` | Target longitudinal acceleration [m/s²] |
| `d_left_m` | Raw center-point distance to left wall [m] |
| `d_right_m` | Raw center-point distance to right wall [m] |

The wall-distance columns are not reduced by half the car width.  For example,
with a 0.35 m wide car and 0.05 m desired side clearance, each raw wall
distance must be at least `0.35 / 2 + 0.05 = 0.225 m`.

## Vehicle Parameters

Vehicle parameters for the optimizer live in two places:

- **`config/autodrive_sim_vehicle.yaml`** — readable simulator vehicle profile.
- **`global_racetrajectory_optimization/params/racecar.ini`** — values consumed by TUM mintime. The pipeline temporarily patches only experiment-level track settings and restores the file afterwards.

### AutoDRIVE simulator vehicle profile

The simulator interface uses a 3.2 rad/s steering-rate limit and 0.5236 rad
steering saturation. The corresponding minimum geometric turn radius is
0.561 m for the 0.324 m kinematic wheelbase. TUM mintime uses its Pacejka
`B/C/E` tire model; fixed `C_alpha` values in N/rad are not available from the
simulator interface and are not inserted as active simulator parameters. The
profile retains the BachelorProject `C_alpha` values only in a
`physical_reference` block for comparison.

Generate the route with:

```bash
f1tenth_planning/.venv/bin/python f1tenth_planning/scripts/optimize_trajectory.py
```

The verifier recomputes heading/curvature from the actual exported XY path and
fails if the route exceeds the simulator steering curvature limit.

## Files

```
f1tenth_planning/
├── scripts/
│   ├── optimize_trajectory.py    # Main pipeline — this is what you run
│   ├── optimize_trajectory_mintime.py  # Canonical mintime reference copy
│   └── compute_wall_distances.py # Wall ray-cast helper (called by pipeline)
├── global_racetrajectory_optimization/  # TUM optimizer (submodule)
│   ├── main_globaltraj.py
│   └── params/racecar.ini
├── trajectories/                 # Output directory for raceline CSVs
├── maps/                         # SLAM Toolbox map output (.yaml + .pgm)
└── CMakeLists.txt
```

## Dependencies

```bash
pip install numpy opencv-contrib-python scipy pyyaml matplotlib

# Required by the mintime optimizer:
pip install casadi
```
