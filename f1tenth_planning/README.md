# Competition planning data

This package contains the ICRA competition map and exactly one installed
raceline. The raceline is consumed by Pure Pursuit, AMCL global initialization,
and the MPC; keeping one file avoids silently testing different trajectories.

Runtime files:

- `maps/autodrive_track_ftg_commit_20260909_025m.yaml` and its PGM image
- `trajectories/autodrive_mintime_sim_5p0_dense/autodrive_mintime_raceline.csv`
- the trajectory manifest in the same directory

Trajectory optimization and experimental outputs are intentionally outside
this competition runtime repository. A replacement raceline must be reviewed
and replace the canonical CSV in one change; do not add parallel candidates.

The reviewed exact minimum-time generator is retained under
`SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/`. It writes checkpoints to
the ignored `artifacts/raceline_generation/` directory by default and uses the
canonical CSV only as an optional warm start. It does not create additional
installed runtime racelines.
