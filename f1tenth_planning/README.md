# f1tenth_planning

Track maps and reference racelines. The package contains data only.

| Track | Map | Raceline | Lap |
|---|---|---|---|
| Practice | `maps/practice` | `trajectories/practice/raceline.csv` | 26.6 m, 4.800 s |
| Competition | `maps/competition` | `trajectories/competition/raceline.csv` | 42.8 m, 6.925 s |

Each map is an occupancy grid (`.pgm` plus ROS map `.yaml`). The competition grid is
238 x 903 cells at 0.025 m, the practice grid 105 x 312 cells at 0.05 m.

A raceline is a "source witness" CSV, one row per millisecond of the reference lap:

```
s_m, x_m, y_m, psi_rad, kappa_radpm, body_forward_speed_mps, acceleration_mps2,
d_left_m, d_right_m, source_heading_error_rad, source_v_rear_mps,
source_yaw_rate_radps, source_physical_steering_rad,
source_steering_target_rad, source_throttle
```

The first nine columns are the path, its speed profile and the distance to the left and
right wall. The last six are the plant state and the steering and throttle commands that
realize the profile in the vehicle model. The MPC loads all fifteen columns.

`trajectories/competition/` also holds the geometric starting line
(`initial_raceline.csv`) and the map-extraction report the optimizer started from.
`trajectories/practice/` holds the validation report and the
witness arrays of the practice raceline.

Keep a map paired with its raceline. The development launch chooses the MPC profile from the map filename. The
competition launch always uses `mpc_competition.yaml`; `SDU_APEX_MAP_REL`
and `SDU_APEX_TRAJECTORY_REL` choose its map and raceline. The competition
image sets those paths from the build script's `AUTODRIVE_MAP_REL` and
`AUTODRIVE_TRAJECTORY_REL` variables.
The optimizer that produces these files is described in
[../docs/raceline-optimization.md](../docs/raceline-optimization.md).
