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

## Trials and development utilities

`tools/run_trial.sh {practice|competition} RUN_ID` builds the working tree in the
development image and runs a supervised closed-loop trial (controller, recorder,
Unity simulator, collision/lap watcher); output is local under ignored
`live_runs/`. `tools/build_competition.sh` and `tools/run_competition.sh` build
and launch the separate competition image.

- `tools/summarize_trial.py`, `tools/evaluate_mpc_bag.py`,
  `tools/export_mpc_run_report.py` and `tools/seal_mpc_run_manifest.py` decode,
  score and record runs.
- `tools/watch_sim_run.sh` stops a supervised run on collision, fault or
  completion; `tools/verify_runtime_topic_policy.py` checks the runtime topic
  boundary.
- `f1tenth_mpc/util/` holds the diagnostics publisher and the frozen-problem
  export/benchmark (see `docs/MPC.md`).
- `SDU_Apex_IPOPT_PhysX_MinTime_v2/` contains the source-based raceline
  optimization. Its generated solver trees, outputs and caches are not source.
