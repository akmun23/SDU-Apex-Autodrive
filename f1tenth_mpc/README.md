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

One conformance/cleanup item remains: the old `mpc_compute_optimal_control`
compatibility implementation and its 10-state types/tests are still in the
package build for regression coverage. The ROS component does not call that
API; it calls the 9-state RTI path. Do not treat the legacy API as an alternate
production controller, and retire or consolidate it before calling the full
rewrite finished.

## Runtime modes

- Default (`enabled: false`, `shadow_mode: false`): inert, with no command
  publisher.
- Shadow (`controller:=pure_pursuit with_mpc_shadow:=true`): runs beside Pure
  Pursuit, subscribes to its `/cmd/speed`, publishes diagnostics on
  `/mpc_shadow/diagnostics`, and has no `/cmd/speed` publisher.
- Command authority (`controller:=mpc mpc_enabled:=true`): intentionally not
  accepted yet. Keep MPC disabled until the shadow and staged live gates pass.

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
