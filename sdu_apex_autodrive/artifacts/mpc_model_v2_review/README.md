# MPC model v2 review

This directory contains the current offline model-identification phase. It is
not a runtime model release and it does not modify Unity physics, simulator
timing, production MPC parameters, AMCL, EKF, or odometry behavior.

## Acceptance boundary

Ground truth is used only for offline fitting and scoring. Recursive prediction
uses the initialized state, recorded applied controls, source `dt`, and fixed
candidate parameters. Future measured `u`, `v`, `r`, pose, acceleration, and
wheel speed are not inputs to recursive propagation.

The native plant is a replay target only. It must pass causal recursive
validation, native parity, an observer replay, and one untouched blind track
run before it can replace the existing production MPC plant.

## Current evidence

- `../model_id_work/model_fits_v2/longitudinal_model_benchmark_v2.json`:
  causal longitudinal comparison; wheel-state candidate remains promising but
  does not meet the final MPC precision target.
- `../model_id_work/model_fits_v2/vehicle_model_candidate_v4_true_open_loop.json`:
  causal empirical body-model baseline; it remains rejected at racing horizons.
- `../model_id_work/model_fits_v2/lateral_model_benchmark_v2.json`:
  Y0/Y1/Y2 lateral comparison with continuous 2 ms steering-ramp integration.
- `../model_id_work/model_fits_v2/native_replay_v1.json`:
  full validation replay through the native provisional plant.
- `../model_id_work/model_fits_v2/python_vs_c_parity_v2.json`:
  Python/native fixture parity.
- `../model_id_work/model_fits_v2/lateral_identifiability_v1.json`:
  profile and local-conditioning check for tire stiffness versus yaw inertia.

The current Y1 lateral candidate is not accepted. A fit that moves `I_z` and
tire stiffnesses to bounds is reported as non-identifiable, not as a physical
measurement.

The current standalone C sources compile cleanly and all 91 Python tests pass.
The ROS/Humble colcon build was attempted in the existing workspace container
but is blocked by that container's missing `ament_package` Python module.

## Next work

1. Use the identifiability result to constrain or redesign lateral excitation.
2. Add tire relaxation only if residual slicing shows dynamic force lag after
   steering-ramp integration.
3. Refit and re-score the composed plant causally.
4. Validate the exact native plant and finite-difference Jacobians.
5. Run the legal sensor-only observer replay and an untouched blind track run.
6. Integrate into MPC only after every gate passes.
