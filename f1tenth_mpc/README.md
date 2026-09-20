# AutoDRIVE source-command MPC

This package implements the AutoDRIVE MPC rewrite described in
`docs/SIMULATOR_NATIVE_MODEL_WORKLIST.md` and the 2026-09-18 coding handoff.
It models the simulated vehicle's accepted observable command-to-response
behavior; it is not a real-car model. The handoff's frozen model and
localization parameters remain unchanged. Unity physics, scene geometry,
sensor behavior, and timing are out of scope for edits.

The RTI controller uses ten plant states
`[e_y, e_psi, u, v, r, target_speed, steering_command,
delayed_steering_command_1, delayed_steering_command_2,
actual_steering_angle]`, two previous-rate history states, and two optimizer inputs
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

When adaptive/R2 mode is enabled, a finite R1 that reaches the iteration cap
above the normal degraded residual gate may be retained only as an exact
nonlinear-checked R2 seed when its residual is at most
`rti2_residual_recovery_limit` (default `0.25`). R1-only mode remains strict,
and only a converged R2 candidate can be selected or published. This recovery
was added for the observed simulator stop transition; it is not a timing
acceptance rule and does not permit an unchecked residual candidate to control
the car.

## Runtime modes

- Normal development (`./tools/start_dev.sh`): MPC command authority, AMCL
  warm-up, first-lap ramp, and terminal collision handling are enabled by the
  unified launcher.
- Shadow (`controller:=pure_pursuit with_mpc_shadow:=true`): an explicit
  diagnostic mode beside Pure Pursuit. It has no `/cmd/speed` authority and is
  not valid evidence for MPC closed-loop driving.
- Direct command authority (`controller:=mpc`): uses the same MPC path. The
  launch waits 2 seconds after AMCL starts before creating the MPC container;
  adjust with `mpc_start_delay_sec:=...` or set it to zero to disable.
- Competition authority (`competition.launch.py`): fixed MPC-only launch with
  the official API bridge, no shadow/recorder/override/RViz switches, and the
  immutable `mpc_iros_2026_competition.yaml` profile. This is the only launch
  used by the competition image.

Use batchmode for simulator testing and do not add `-no-graphics`. A passing
unit test or offline replay is not a live-driving acceptance result. The
Current tuning evidence is summarized in
`docs/MPC_WEIGHT_EVALUATION.md`. Historical replay dumps are kept outside the
runtime repository and are not required to start the controller.

Per-cycle MPC JSON diagnostics are disabled on the normal authority path to
avoid serialization/DDS work at 40 Hz. Enable them only for a diagnostic run:

```bash
SDU_APEX_MPC_PUBLISH_DIAGNOSTICS=true ./tools/start_dev.sh
```

Build and test in the ROS 2 Humble workspace container:

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-select f1tenth_mpc
colcon test --merge-install --packages-select f1tenth_mpc \
  --event-handlers console_direct+
```
