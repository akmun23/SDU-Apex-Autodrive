# Production code map

## Runtime path

`docker/competition_entrypoint.sh` starts
`sdu_apex_autodrive/launch/competition.launch.py`. The development launch is
`controller.launch.py`; it shares the runtime controller and localization
nodes but supports choosing a map, trajectory, and parameter profile.

- **Simulator bridge and commands:** `sdu_apex_autodrive/`
- **Vehicle odometry and source-derived plant state:** `f1tenth_localization/`
- **AMCL/EKF configuration:** `f1tenth_localization/config/`
- **Throttle/steering MPC, plant rollout, reference parser:** `f1tenth_mpc/`
- **Maps and the active practice/competition trajectories:** `f1tenth_planning/`
- **ROS interfaces:** `sdu_apex_msgs/`

## Active practice configuration

`tools/run_mpc_practice_trial.sh` composes the practice map, the 4.800 s
source-command trajectory, the practice MPC profile, production AMCL defaults,
and the fixed practice simulator image. It starts the recorder and supervised
collision/lap watcher. Its output is local under ignored `live_runs/`.

`tools/build_competition.sh` and `tools/run_competition.sh` build and launch the
separate competition profile. The practice fastline profile is not silently
substituted into the competition build.

## Maintained development utilities

- `tools/evaluate_mpc_bag.py` decodes and scores captured runs.
- `tools/export_mpc_run_report.py` and `tools/seal_mpc_run_manifest.py` produce
  reproducible run records.
- `tools/watch_sim_run.sh` stops a supervised run on collision/fault or
  completion.
- `tools/verify_runtime_topic_policy.py` checks the runtime topic boundary.
- `SDU_Apex_IPOPT_PhysX_MinTime_v2/` contains the source-based raceline
  optimization implementation. Its generated solver trees, outputs, and
  caches are not source and are not retained in the workspace.

Unit tests beside each maintained package protect its current implementation;
they are not runtime nodes or installed executables.

The optimizer's test-only source-analysis helpers live in
`SDU_Apex_IPOPT_PhysX_MinTime_v2/tests/diagnostic_fixtures/`. Focused tests use
them to protect source-derived force, slip, suspension, contact, and IPOPT
feasibility behavior; production optimizer entrypoints do not import them.
Unreferenced diagnostic launchers and one unused diagnostic module have been
removed.
