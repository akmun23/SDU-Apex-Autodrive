# N30 error-reduction campaign — pinned to `44421c3`

This campaign follows `SDU_Apex_Autodrive_N30_Error_Reduction_Coding_Handoff_2026-09-16.md` in order. The target remains 30 commands at 40 Hz (`dt=0.025 s`, `T=0.75 s`, speed ceiling `16 m/s`). Simulator physics, production MPC, localization runtime, and applied control are not changed here.

## Current gate

- Phase 0: **passed for the current pinned implementation**. The 67 relevant tests pass; the fitted Unity-controller candidate and all six native-wheel horizons replay exactly; the v5 cross-schedule force screen matches its pinned summary. The older `corrected_source` report predates commit `44421c3`, which introduced the explicit Unity-wheel model/scorer. It has no implementation/input hashes, is retained as a historical pre-commit result, and is not evidence of an improvement. See `00_environment/baseline_44421c3.json` and `00_environment/native_wheel_replay_provenance.json`.
- Phase 1: existing-data inventory confirms there was no synchronized 1 kHz/40 Hz full-state track trace. A short live batchmode PP pilot established the per-fixed-step `applied_command_sequence` join, but a later source-cadence fault invalidated the pilot for modeling. See `01_data_inventory/existing_field_inventory.csv` and `01_data_inventory/phase1_inventory.md`.
- User acceptance screen: the legal-input live N30 replay has CORE p95 `e_cross=0.083 m` and `u=0.129 m/s`; truth-initialized `u` p95 is `0.131 m/s`. This is promising cross-track accuracy, but it misses the requested `u<0.10 m/s` threshold. The run reached only `8.41 m/s`, so it does not validate the 12–16 m/s range. It is not a production-MPC acceptance result.
- Phase 2 onward: not started. In particular, zero-initialized suspension scores are not accepted as oracle ablations. Oracle initialization must use measured hidden states at the origin only and propagate recursively thereafter.

## Evidence so far

`00_environment/native_wheel_static_075.json` matches every reported horizon score in the existing `direct_unity_wheel_collider_benchmark_075_exact_unity_curve_20260916.json`. Its 0.75 s p95 is `0.48347 m` position, `0.92810 m/s` longitudinal velocity, `0.02959 m/s` lateral velocity, and `0.15642 rad/s` yaw rate. High-speed constant-speed and braking runs show longitudinal prediction bias; mixed-throttle/steering runs have a different lateral/yaw failure signature. These remain offline oracle-initialized diagnostics, not evidence of runtime benefit.

The cross-schedule force re-run under `00_environment/axle_force_cross_schedule_v4_reproduced.json` is exactly equal to the pinned v4 summary. It does not support promoting a universal axle-force gain.

The handoff's §1.2 values match `validation_canonical_scalar_baseline` in the candidate JSON, not `validation_profile`. The reproduced fitted-profile CORE values at 0.75 s are `e_cross=0.04135 m`, `heading=0.04020 rad`, `u=0.13666 m/s`, `v=0.01030 m/s`, and `r=0.09187 rad/s`. Do not label the canonical baseline as the latest fitted candidate.

The native-score difference is now classified: the `corrected_source` artifact was produced before the pinned commit introduced the explicit Unity-wheel dynamics and its scorer; the exact-curve artifact is from the current implementation. The current implementation was replayed over all `11,145` origins, six horizons, and five primary states with maximum p95 difference `0.0`. The older result is not comparable and remains non-promoted; no improvement is claimed from the numerical delta.

## Next required action

Run a bounded timing pilot with the newly built asynchronous diagnostic writer. Stop both Unity and ROS together on the first bridge timing fault; accept no trace if it drops fixed-step rows or source cadence is invalid. If that pilot passes, gather the Phase 1 synchronized straight and combined/track captures; if it does not, do not start the 12-run campaign. Do not run the Phase 2 oracle ablation until accepted straight and combined/track data contain exact four-wheel and reconstructable suspension origin state, with no interpolation across resets or gaps. Every Unity run must use `-batchmode`; do not use `-no-graphics`.
