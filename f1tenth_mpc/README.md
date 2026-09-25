# AutoDRIVE MPC

This package contains the single production MPC controller and its compact
Riccati/ADMM core. It uses only the team-derived `/odom` and
`/current_map_pose` inputs plus the canonical installed raceline. Simulator
truth, hidden Unity state, and source timestamps are not controller inputs.

The controller runs a 30-stage horizon at 40 Hz with the competition model
and weights in `config/mpc_competition.yaml`. There is no shadow
controller or alternate runtime model in this repository.

`rejected_nonlinear_rollout` means the exact predicted horizon failed its
corridor check; it does not by itself mean the controller withdrew the first
command. A best-effort first action is now checked with the exact nonlinear
one-step vehicle model and footprint corridor; if needed it is backtracked
toward the shifted nominal action. If no safe first step is found, that
best-effort action is withheld. `best_effort_action_repaired` makes the
correction visible in the diagnostic record. `max_solver_iterations` remains
capped at 100 by the controller,
so larger YAML values are not effective.

For quick real-simulator weight screens, put only the candidate parameter(s)
in a ROS parameter YAML under `live_runs/`, then start the development stack
with `SDU_APEX_MPC_PARAMETER_OVERLAY="$PWD/live_runs/candidate.yaml"`.
This layers over the canonical MPC configuration without changing the
competition profile or rebuilding for weight-only changes. A partial overlay
looks like:

```yaml
/**:
  ros__parameters:
    weight_e_psi: 2.0
```

Record a debug bag, then run:

```bash
tools/watch_sim_run.sh <controller-container> <simulator-container> \
  <recorder-container> 2
```

Start this watcher before launching the simulator in another terminal. It
stops the simulator and closes the recorder on a collision or after one
warmup plus one racing lap. Use short screens to reject poor candidates, then
run the full 12-count acceptance only on finalists.

Build and unit-test it in the ROS 2 Humble image:

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-select f1tenth_mpc
colcon test --merge-install --packages-select f1tenth_mpc \
  --event-handlers console_direct+
```
