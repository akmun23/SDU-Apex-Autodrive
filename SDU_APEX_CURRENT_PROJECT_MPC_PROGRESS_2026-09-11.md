# MPC progress note — 2026-09-11

This note records the state reached before pausing work. The current review file is the governing technical checklist. The Unity simulator source and physics were not edited.

## Completed

- Removed the stale MPC node references from `sdu_apex_autodrive/launch/controller.launch.py`. The launch file no longer advertises the nonexistent `mpc_shadow` or `control_state_node` path.
- Kept `f1tenth_mpc` as the BachelorProject-style pure C core: no vendor directory and no custom ROS wrapper files were added to that package.
- Removed `-ffast-math` from the MPC build.
- Added the first native core test target in `f1tenth_mpc/test/test_vehicle_model.c`.
- Unified global and Frenet prediction around the shared nonlinear body-dynamics force path instead of maintaining a second local lateral-force implementation.
- Added runtime vehicle-parameter access/validation, including steering time constant, tire parameters, gravity, slip-velocity regularization, and acceleration limits.
- Preserved physical longitudinal velocity at zero. The low-speed denominator floor is now numerical regularization only; it is not a fake physical speed.
- Added the MPC Frenet rollout API and updated MPC code to use runtime parameters.
- Added a diagnostics-only raw steering override to the model-identification path. It is enabled only by `model_identification.launch.py`; it does not alter the simulator or normal driving path.
- Rebuilt the ROS packages successfully.

## Validation completed

Native MPC test:

```text
vehicle_model_test: Passed
100% tests passed, 0 tests failed
```

Fresh open-plane batch model-identification run:

```text
run: sdu_apex_autodrive/artifacts/model_id_work/open_plane_competition_steering_response_raw2_review_20260911
raw packets: 1504
transitions: 1503
source rate median: 40.000000000002 Hz
source dt median: 0.025000 s
source dt p95: 0.026001 s
source gaps: 0
duplicate/reverse rows: 0
selected body-frame displacement error median: 0.03495 m/s
yaw-rate consistency median error: 0.0000395 rad/s
quality status: timing_and_kinematics_gate_passed
```

The recorded command sequence now contains the requested steering levels (`-0.50, -0.25, 0, 0.25, 0.50`). The simulator-applied steering also responds over the run, approximately `-0.2618` to `+0.2618`, rather than being overwritten to zero.

The validation command was:

```bash
PYTHONPATH=/workspace/src/sdu_apex_autodrive \
python3 tools/model_id/assemble_transitions.py \
  sdu_apex_autodrive/artifacts/model_id_work/open_plane_competition_steering_response_raw2_review_20260911 \
  --twist-frame body
```

## Not completed / not promoted

- There is not yet an accepted steering-dynamics fit. The fresh run passes the timing and frame gates, but it is only one run and covers one operating regime. Repeatability, multiple speeds, steering grids, delay/pole fitting, and blind validation are still required.
- The earlier longitudinal polynomial candidate is not accepted for recursive prediction. It must not be used as a production MPC model.
- The MPC core is not connected to a live ROS controller and has not passed shadow, solver-timing, closed-loop, or racing acceptance.
- The direct normalized-throttle versus physical-acceleration boundary still needs to be made explicit. The preferred handoff direction is direct normalized throttle, with any acceleration model kept as an explicitly identified virtual layer.
- Steering time constant is still a default placeholder until it is identified from data.
- No production odometry or AMCL code was changed in this work. A separate legal `mpc_state_estimator` using only dev-branch topics is still future work; simulator truth remains offline-only.
- No simulator physics, vehicle behavior, scene geometry, timing configuration, or Unity source was changed.

## Important data caveat

The previous `...steering_response_raw_review_20260911` run is retained but is invalid for steering fitting because the calibration steering command was still being overwritten during that run. The `...raw2_review_20260911` run is the first corrected run and should be used for further analysis.

The temporary calibration file is:

```text
sdu_apex_autodrive/config/model_id_steering_grid_temp.yaml
```

The other model-identification artifact directories are preserved as evidence and were not deleted.

## Recommended next work

1. Repeat the corrected steering-response experiment at least three times and add speed/regime coverage.
2. Implement an offline candidate steering fit with explicit delay, first-order steering dynamics, residuals, and blind recursive scoring. Do not promote it automatically.
3. Add the remaining native tests for Jacobians, steering discretization, solver constraints, NaN handling, and determinism.
4. Finish the throttle/input-boundary decision and replay API before connecting MPC to ROS.
5. Add the separate legal MPC state estimator for causal lateral velocity, then validate odometry/AMCL improvements against ground truth offline.
6. Only after the blind model/state gates pass, create the separate `f1tenth_mpc_ros` adapter and run MPC in shadow mode.

