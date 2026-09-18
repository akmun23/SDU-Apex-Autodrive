# AutoDRIVE source-command MPC

This package implements the AutoDRIVE MPC rewrite described in
`docs/SIMULATOR_NATIVE_MODEL_WORKLIST.md` and the 2026-09-18 coding handoff.
It models the simulated vehicle's accepted observable command-to-response
behavior; it is not a real-car model. The handoff's frozen model and
localization parameters remain unchanged. Unity physics, scene geometry,
sensor behavior, and timing are out of scope for edits.

The RTI controller uses seven plant states
`[e_y, e_psi, u, v, r, target_speed, steering_command]`, two previous-rate
history states, and two optimizer inputs
`[steering_command_rate, target_speed_rate]`. Its horizon is 30 stages at
25 ms (0.75 s). Raceline projection and references are continuous in arc
length. The ROS adapter synchronizes legal `/current_map_pose` and `/odom`
data and uses controller command history; simulator truth and hidden Unity
state are not runtime inputs.

The MPC corridor has a hard configured inset for predicted states. The measured
state `x0` is not corridor-gated because it is immutable for the current solve.
`first_prediction_corridor_margin_m` applies only to `x1`; it defaults to the
normal `corridor_margin_m` and must be no larger. Any reduction is an explicit
controller-policy change and must be checked on captured simulator runs before
use; it does not change the track geometry or simulator behavior.

The old `mpc_compute_optimal_control` implementation is now isolated in a
`BUILD_TESTING`-only compatibility library. It is linked only by its historical
regression test/benchmark, is not linked into `mpc_core` or the ROS component,
and its `mpc.h` API is not installed. The production solver path is the 9-state
RTI implementation described above.

## Runtime modes

- Default (`enabled: false`, `shadow_mode: false`): inert, with no command
  publisher.
- Shadow (`controller:=pure_pursuit with_mpc_shadow:=true`): runs beside Pure
  Pursuit, subscribes to its `/cmd/speed`, publishes diagnostics on
  `/mpc_shadow/diagnostics`, and has no `/cmd/speed` publisher.
- Command authority (`controller:=mpc mpc_enabled:=true`): available for
  controlled validation, but not yet declared driving-ready. By default, the
  launch waits 2 seconds after AMCL starts before creating the MPC container;
  adjust with `mpc_start_delay_sec:=...` or set it to zero to disable.

Use batchmode for simulator testing and do not add `-no-graphics`. A passing
unit test or offline replay is not a live-driving acceptance result. The
current status and evidence are recorded in
`artifacts/mpc_rewrite_845e621/` and the worklist document.

Build and test in the ROS 2 Humble workspace container:

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-select f1tenth_mpc
colcon test --merge-install --packages-select f1tenth_mpc \
  --event-handlers console_direct+
```
