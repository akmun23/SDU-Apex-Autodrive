# Model-identification data index

Only accepted or directly useful model-identification runs remain in this
directory. Raw CSV files are kept inside `accepted/`; comparison plots and
reports are inside `reports/`.

## Accepted data

- `accepted/01_longitudinal_throttle_from_rest_40hz_20260913/`
  - 3,750 source packets and 3,749 causal transitions.
  - Source timing and body-frame kinematics passed; no timing gaps or bridge
    faults.
  - The exploratory longitudinal candidate is rejected and must not be used:
    validation one-step relative p95 was 5.06% and recursive relative p95 was
    132.02%. The wide legacy grid-summary CSV was removed because it exceeded
    Git's 100 MB per-file limit; the canonical recorder partitions remain.
- `accepted/02_steering_actuator_stationary_source_step_40hz_20260913/`
  - 921 source packets; stationary steering steps, reversals, slew and
    saturation data.
- `accepted/03_steering_response_multispeed_40hz_20260913/`
  - 4,048 source packets; speed-regime steering response data.
- `accepted/04_steering_response_single_regime_40hz_20260911/`
  - 1,504 source packets; corrected single-regime baseline.
- `accepted/30_zero_throttle_coast_map_40hz_20260913/`
  - 1,581 source packets and 1,580 causal transitions covering three
    acceleration/coast ranges (approximately 6, 12, and 14.7 m/s).
  - Source timing/ordering and body-frame kinematic gates passed: median
    source rate 40.00 Hz, 22--28 ms source intervals, no gaps or faults, and
    zero collisions.
  - The valid source-indexed replay summary is
    `source_replay_20260913_115806.csv`; the failed preflight summary was
    removed.
- `accepted/50_steering_multisine_6mps_40hz_20260913/`
  - 1,041 source packets and 1,040 causal transitions with bounded,
    sign-changing steering excitation around 6 m/s.
  - Source timing/ordering and body-frame kinematic gates passed: median
    source rate 40.00 Hz, 22--28 ms source intervals, no gaps or bridge
    faults, and zero collisions.
- `accepted/60_e1_corner_throttle_40hz_20260913/`
  - 1,001 source packets and 1,000 causal transitions with throttle
    increase/decrease while holding a bounded corner.
  - Source timing/ordering and body-frame kinematic gates passed: median
    source rate 40.00 Hz, 22--28 ms source intervals, no gaps or bridge
    faults, and zero collisions.
- `accepted/61_e2_steering_acceleration_40hz_20260913/`
  - 1,001 source packets and 1,000 causal transitions with signed steering
    steps under fixed acceleration.
  - Source timing/ordering and body-frame kinematic gates passed: median
    source rate 40.00 Hz, 19--30 ms source intervals, no gaps or bridge
    faults, and zero collisions.
- `accepted/62_e3_steering_reversal_speed_40hz_20260913/`
  - 1,081 source packets and 1,080 causal transitions with repeated signed
    steering reversals at bounded speed.
  - Source timing/ordering and body-frame kinematic gates passed: median
    source rate 40.00 Hz, 22--28 ms source intervals, no gaps or bridge
    faults, and zero collisions.
- `accepted/70_d1_d2_steady_corner_40hz_20260913/`
  - 1,921 source packets and 1,920 causal transitions covering mirrored
    steady left/right cornering at approximately 6.08 m/s.
  - Source timing/ordering and body-frame kinematic gates passed: median
    source rate 40.00 Hz, 22--28 ms source intervals, no gaps or bridge
    faults, and zero collisions.
- `accepted/75_d3_lateral_envelope_60hzheadroom_40hz_20260913/`
  - 1,761 source packets and 1,760 causal transitions covering a progressive
    mirrored lateral-demand sweep at approximately 6.08 m/s.
  - Source timing/ordering and body-frame kinematic gates passed: median
    source rate 40.00 Hz, 22--28 ms source intervals, no gaps or bridge
    faults, and zero collisions.
  - Offline diagnostic: at normalized steering 0.20, measured curvature was
    approximately 0.277 1/m versus 0.324 1/m from the nominal kinematic
    relation, indicating about 15% understeer/nonlinearity at that point.
- `accepted/10_static_repeatability_run01_40hz_20260913/` through run05
  - Five clean static repeatability runs.
- `accepted/20_combined_throttle_steering_run01_40hz_20260913/` through run05
  - Five clean source-step combined-input runs.
- `accepted/30_straight_replay_run01_40hz_20260913/` through run05
  - Clean straight replay supplementary runs.
- `accepted/40_steering_replay_run01_40hz_20260913/` through run05
  - Clean steering replay supplementary runs.
- `accepted/00_source_step_timing_smoke_40hz_20260913/`
  - Small valid source-step smoke artifact kept as a timing reference.

## Current identification status

The data-collection and timing gates have passed for the retained runs. The
canonical transition file is now `assembled/model_transition_v4.csv`. It has
explicit normalized-command and physical-radian steering fields, source
derived wheel speed, `reset_epoch`, and `segment_id`. Reset commands and
detected pose teleports are hard rollout boundaries; their crossing
transitions are excluded rather than scored as plant error. The v4 assembler
has unit coverage for steering semantics, source timing, reset boundaries,
and pose discontinuities.

A production-quality MPC model has not yet been identified. The corrected
structured vehicle candidate is recorded at
`model_fits/vehicle_model_candidate_v2_steering_corrected.json`. It uses
16,109 training transitions and 6,388 whole-run validation transitions after
warm-up/boundary handling. Its steering actuator submodel still selects zero
source-step lag and 3.2 rad/s rate limiting, with 0.00004 rad validation p95
feedback error. The body-dynamics candidate remains rejected: its validation
recursive p95 errors at 0.50 s are approximately 0.44 m position, 1.21 m/s
longitudinal speed, 0.93 m/s lateral speed, and 1.12 rad/s yaw rate. Correcting
the steering field therefore fixed the identification contract but did not,
by itself, fix the plant model. No fitted parameters have been copied into the
C MPC model.

The longitudinal benchmark is recorded at
`model_fits/longitudinal_model_benchmark_v1.json`. The wheel-slip candidate
identifies approximately `17.72 N` force saturation, `2.26 1/(m/s)` slip gain,
and `0.89 N/(m/s)` speed drag. With an explicitly fitted discrete wheel-speed
state, its conditional longitudinal validation p95 is approximately 0.17 m/s
at 0.10 s and 0.33 m/s at 0.50 s, compared with 0.49 m/s and 1.58 m/s for the
direct throttle surface. This is encouraging evidence for the wheel-slip
architecture, but the score conditions on recorded lateral `r*v` and is not a
full six-state or blind MPC acceptance result.

The next identification step is corrected lateral fitting with steering-ramp
integration, followed by native recursive replay of the same candidate. A
fresh untouched track-domain run remains required before freezing any model or
changing MPC parameters.

The dev-side odometry candidate now rejects only a low-speed coherent encoder
recovery that is also above the launch wheel-spin threshold. This was driven
by repeated accepted recordings where wheel rate was about 6 m/s while body
speed was below 1 m/s. Existing observer unit tests pass; representative
replay still exposes separate braking/turn transients that require more data
before further tuning.
Generated
wide `identification_grid_*.csv` summaries are intentionally excluded; the
source-event recorder partitions are authoritative.

Ground truth remains offline-only. No Unity simulator physics or behavior was
changed during this cleanup.

## Causal scoring and structured-plant phase

The updated handoff identified future-state leakage in both previous
recursive scorers. That is now fixed in `tools/model_id/fit_longitudinal_models.py`
and `tools/model_id/fit_vehicle_model.py`. Recursive prediction consumes only
the initialized state, applied inputs, and source `dt`; the required 25 ms and
750 ms horizons are reported explicitly. Anti-leak regression tests are in
`test/test_model_id_recursive_causality.py`.

Versioned reports are stored under `model_fits_v2/` because the historical
`model_fits/` directory is Docker-owned:

- `longitudinal_model_benchmark_v2.json` is causal and reports approximately
  0.83 m/s validation p95 at 0.50 s for the wheel-state candidate. The earlier
  0.33 m/s result was GT-conditioned and is not accepted.
- `vehicle_model_candidate_v3_true_open_loop.json` is causal and remains
  rejected; at 0.50 s it reports approximately 0.375 m position, 1.05 m/s
  `u`, 1.18 m/s `v`, and 0.77 rad/s yaw-rate p95.
- `lateral_model_benchmark_v1.json` compares Y0 kinematic, Y1 saturated-linear,
  and Y2 tanh candidates using 2 ms steering-ramp integration. Y1 is currently
  the best candidate, but it does not meet the recursive precision gates.
- `python_vs_c_parity.json` compares the same provisional Y1 plant equations
  against `f1tenth_mpc/src/vehicle_plant.c`; four fixtures pass with maximum
  absolute error approximately `1.3e-6`.

`vehicle_plant.c` is an offline native replay target only. It is not wired into
the production MPC and no MPC parameters or simulator behavior were changed.
The next unresolved work is model improvement/relaxation analysis, native
replay over the complete V4 dataset, observer replay, and a fresh untouched
blind track run before production MPC integration.

The lateral inertia question is now measured rather than assumed. The profile
diagnostic in `model_fits_v2/lateral_identifiability_v1.json` refits tire
stiffness while sweeping `I_z`; its optimum is at the upper profile bound
`0.08 kg m^2` and the local sensitivity condition number is approximately
`7,484`. This means `I_z` and tire stiffness are not separately identifiable
under the current candidate and excitation. The value is therefore not a
measurement of Unity's rigid-body inertia and must not be frozen into MPC.
The wheel state follows the same rule: it is an effective causal drive state,
not a claim that its fitted coefficients equal a real wheel's mechanical
rotational model.
