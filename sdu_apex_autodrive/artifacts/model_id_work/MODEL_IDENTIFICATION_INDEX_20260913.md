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

The data-collection and timing gates have passed for the retained runs. A
production-quality MPC model has not yet been identified: the first simple
longitudinal model failed recursive validation, and steering/combined data
still require candidate fitting plus blind recursive validation.

The next identification step is steering actuator and lateral-dynamics fitting
from the retained stationary, multispeed, combined, replay, and multisine runs.
Generated
wide `identification_grid_*.csv` summaries are intentionally excluded; the
source-event recorder partitions are authoritative.

Ground truth remains offline-only. No Unity simulator physics or behavior was
changed during this cleanup.
