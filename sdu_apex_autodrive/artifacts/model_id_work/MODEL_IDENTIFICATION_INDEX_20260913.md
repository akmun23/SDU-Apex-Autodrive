# Model-identification data index

## Active 2026-09-15 freeze path

The current implementation follows the 0.75 s handoff: N=30 commands at
0.025 s, with 16 m/s as the hard project ceiling. The active model-selection
artifacts are:

- `raceline_operating_envelope_v1.json` and `.csv` — coupled production
  raceline CORE/GUARD/STRESS occupancy;
- `model_fits_v2/speed_regime_vehicle_candidate_core_guard_075_20260915.json`
  — canonical-lateral candidate;
- `model_fits_v2/speed_regime_vehicle_candidate_core_guard_fitted_075_20260915.json`
  — fitted-lateral candidate;
- `model_fits_v2/speed_regime_vehicle_candidate_first_order_fitted_075_20260915.json`
  — current first-order-steering fitted-lateral candidate;
- `model_fits_v2/speed_regime_vehicle_candidate_first_order_residual_fitted_075_20260915.json`
  — rejected first-order steering-transition-residual experiment;
- `model_fits_v2/speed_regime_vehicle_candidate_delayed_steering_raceline_fitted_075_20260916.json`
  — current causally corrected, raceline-filtered fitted-lateral candidate;
- `model_fits_v2/tire_peak_bound_diagnostic_20260916.md`, plus the `peak120`
  and `peak240` reports — tire-peak capacity-bound diagnostics; not approved
  candidates;
- `model_fits_v2/raceline_residual_diagnostic_first_order_fitted_075_20260916.json`
  — CORE/GUARD one-step residual attribution;
- `model_fits_v2/ablation_A0_A5_first_order_core_guard_075_20260915.json`,
  `model_fits_v2/ablation_A0_A6_first_order_core_guard_075_20260915.json`,
  and `model_fits_v2/ablation_A0_A6_first_order_residual_core_guard_075_20260915.json`
  — first-order A0–A6 attribution, conditional one-parameter A6 result, and
  the nonzero steering-residual attribution;
- `steering_semantics_v2_train_validation_20260916.json` — command/applied/
  feedback semantics over the union of train and validation runs;
- `model_fits_v2/runtime_control_state_live_20260915.json` — live state score;
- `model_fits_v2/speed_command_pipeline_live_20260915.json` — live command
  pipeline evidence.

These are offline or diagnostic candidates only. No candidate coefficients
have been migrated into production MPC. Historical reports below remain for
audit provenance and are not active promotion gates; in particular, old 2 s
and >16 m/s results must not be used to select the current model.

### 2026-09-16 causal timing and tire-peak update

The assembled transition rows now use the previous packet's steering command
(`steering_target_norm_k`) as the command consumed by the transition; the raw
`commanded_steering_norm_k1` field remains available for provenance but is not
used as the transition input. The corrected raceline-filtered candidate passes
the provisional 0.75 s CORE numerical gates: p95 `e_cross=0.0327 m`,
`heading=0.0406 rad`, `u=0.135 m/s`, `v=0.0127 m/s`, and `r=0.0966 rad/s`.
It is still offline-only pending blind validation, native parity, and live
acceptance.

The low-regime rear tire peak reached the former 60 N optimizer ceiling. A
120 N sweep moved it to 114.3 N with negligible CORE improvement and a small
GUARD yaw improvement; a 240 N sweep failed to converge, and an effectively
unbounded 1000 N diagnostic moved it to 179.8 N while worsening holdout
GUARD errors. The peak is therefore weakly identifiable with the current
model/data combination. The 60 N value remains a conservative offline
regularizer, not a claimed simulator physics parameter. Full details are in
`model_fits_v2/tire_peak_bound_diagnostic_20260916.md`.

### 2026-09-16 exact Unity value capture

The disposable diagnostic player was built from the exact competition scene
in batchmode with `-batchmode` and without `-nographics`; Unity physics and
scene behavior were not changed. The stationary capture contains 2,003
1-kHz rows and the combined excitation contains 12,000 1-kHz rows, with no
fixed-step gaps. Both captures include the new read-only `GetWorldPose`
columns.

The exact runtime yaw inertia is `I_z = 0.0961907506 kg m^2` (`body_y`),
matching the canonical manifest value `0.0961908`; the old `0.0276976` value
was the rejected wrong-axis projection. Exact runtime sprung masses are
`[0.8171950, 0.8160194, 0.9189917, 0.9177940] kg`, corresponding to static
wheel loads `[8.01668, 8.00515, 9.01531, 9.00356] N`.

The stationary pose/load audit reproduces the configured suspension closely
(descriptive front screens about `531 N/m`, `107 N s/m`; rear about `589 N/m`,
`112 N s/m`, versus configured `500 N/m`, `100 N s/m`). The combined-run
screen has low explanatory power because wheel contact load changes with
acceleration, steering, contact geometry, and wheel rotational state. No
dynamic load-transfer coefficient or slip scale has been promoted. See
`model_fits_v2/unity_exact_competition_static_suspension_audit_20260916.json`,
`model_fits_v2/unity_exact_competition_combined_suspension_audit_20260916.json`,
and `tools/model_id/UNITY_MODEL_REQUIRED_VALUES_20260916.md`.

A delayed post-settling runtime snapshot was added to check for hidden
WheelCollider rewrites. It matches the serialized `500/100/.05` suspension
and zero force-app-point values; the optional repository `Suspension.cs` is
not attached to the F1TENTH competition prefab.

The exact open-plane drive excitation reached `15.36 m/s` and retained all
111,266 fixed-step rows in three CSV parts, each below 100 MB. The recorded
forward slip is reproduced by the simulator-specific coordinate
`(wheel_surface_speed-ground_speed)/max(abs(wheel_surface_speed),abs(ground_speed))`,
using collider radius `0.059 m` and saturation at `+-1`. Settled positive-drive
plateaus have at most about `0.0008` slip-unit reconstruction error per wheel.
Large post-command transient differences remain and are explicitly attributed
to wheel rotational state; they are not hidden in the forward friction curve.
See `model_fits_v2/unity_exact_open_drive_wheel_drive_audit_20260916.json` and
the trace README in
`model_fits_v2/unity_exact_open_drive_excitation_20260916/`.

The explicit forward-curve recursion was screened against the same complete
0.75 s set and rejected: CORE cross-track p95 `0.0486 m` and longitudinal
speed p95 `0.214 m/s`, versus `0.0431 m` and `0.135 m/s` for the identified
force screen. The direct curve and forward-slip coordinate are retained, but
the wheel rotational and dynamic-load states are not yet sufficient for
recursive prediction.

The repeated powered-drive trace is
`model_fits_v2/unity_exact_open_powered_drive_repeat_wheel_drive_audit_20260916.json`;
its source CSV is in
`model_fits_v2/unity_exact_open_powered_drive_repeat_20260916/` and is 57 MB.
The explicit torque-balance screen separates positive-drive transitions from
the zero-throttle CAWB brake lock. It identifies effective wheel-state
coefficients of approximately `I=0.000366 kg m^2` below `6 m/s`,
`I=0.000441 kg m^2` at `6--12 m/s`, and `I=0.000436 kg m^2` at `12--16 m/s`,
with `c=0.250 N m s`. Per-regime held-out balance error is approximately
`0.11--0.14 N m`. The speed dependence is retained as an explicit Unity
solver-response diagnostic; no single inertia has been promoted and no
wheel-state behavior has been hidden inside tire force.

### Latest PR1/PR2/PR3/PR4/PR5/PR6 evidence

- The complete first-order stride-1 ablation is in
  `model_fits_v2/ablation_A0_A5_first_order_core_guard_075_20260915.json`.
  At 0.75 s, A2 first-order gives CORE cross-track/heading/yaw-rate p95 of
  `0.0404 m / 0.0439 rad / 0.1328 rad/s`; A4/A5 give
  `0.0403 m / 0.0393 rad / 0.1290 rad/s`. A4 improves both substantial
  held-out track runs in lateral/yaw error, so the handoff's A6 condition was
  met. The conditional one-parameter result is in
  `ablation_A0_A6_first_order_core_guard_075_20260915.json`; it fits a gain
  of `0.4261` but is worse than A4 on CORE heading, lateral velocity, and
  yaw-rate, so it is rejected. No candidate passes the complete provisional
  CORE gate and nothing is frozen or migrated. All variants have zero
  nonfinite or unstable-state failures; the 233 rejected origins are 175
  envelope-crossing transitions plus 58 incomplete end-of-run windows.
- The explicit first-order residual attribution is in
  `ablation_A0_A6_first_order_residual_core_guard_075_20260915.json`. The
  fitted transition gains (`-0.6966 N/(rad/s)`, `0.0210 Nm/(rad/s)`) do not
  improve the held-out candidate: A5 reaches CORE yaw-rate p95 `0.1294`
  rad/s versus `0.1290` rad/s for no-residual A4. The residual is therefore
  retained as rejected evidence only.
- The active residual diagnostic contains 5,613 CORE/GUARD one-step
  transitions. The two long track holdouts dominate the remaining yaw error
  (`r` one-step p95 `0.126` and `0.103` rad/s respectively), while the
  isolated 14--16 m/s experiments are effectively zero in this score. This
  points to a track-run/state/actuator-semantic issue to resolve before
  adding another tire coefficient; no one-step feature has been promoted as a
  runtime correction.
- `model_fits_v2/mpc_N30_discrete_stage_map_core_guard_20260915.json` compares
  only the active N30/25 ms grid. The 2 ms reference has 0.0456 m overall
  cross-track p95, 4x6.25 ms has 0.0477 m, 2x12.5 ms has 0.0524 m, and one
  25 ms step has 0.0833 m. This is an offline map-resolution result, not a
  simulator-physics change.
- `model_fits_v2/n30_stage_map_validation_20260915.json` passes finite,
  bounded, finite-difference-Jacobian, and scalar Python/native parity
  checks; maximum native reference error is 3.1e-6.
- `steering_semantics_v2_train_validation_20260916.json` scores 36,162 causal
  steering transitions, including the decisive track holdouts. First-order
  remains best overall (`0.00278 rad` p95 versus `0.00733 rad` for both
  rate-limited and instantaneous), and is also best on holdouts 86 (`0.00477`
  rad) and 103 (`0.00543` rad). This still is not a promotion decision until
  command/applied/feedback angle semantics are reviewed explicitly.
- `model_fits_v2/runtime_control_state_live_20260915.json` reports current
  map pose cross-track p95 about 0.024 m for `/current_map_pose` and marks
  future-horizon state data unavailable. Raw `/odom` is therefore not used
  as a global-pose acceptance metric.
- `model_fits_v2/speed_command_pipeline_live_20260915.json` finds a measured
  speed peak of 8.441 m/s but no timestamped PP target/raceline/curvature or
  acceleration-limiter fields in the retained capture, so limiter attribution
  is correctly unavailable rather than guessed.

The active candidate-selection path is the artifact list above. Older raw
CSV files, fit reports, and comparison outputs remain in their existing
locations for reproducibility; they are audit evidence only and must not be
used as a current model input. Legacy raw CSV files are kept inside
`accepted/`; the current 2026-09-15 high-speed set is under
`accepted_20260915/`. Rejected captures remain recoverable under
`failed_20260915/`; comparison plots and reports are inside `reports/`.

## Project speed envelope

The project operating and model-identification ceiling is now **16 m/s**.
New controller commands, calibration safety checks, and model-fitting data
must stay at or below this speed. Existing 18/20 m/s captures are retained as
historical offline evidence only; they are out of the current operating
envelope and must not be used to define new production candidates. This limit
is implemented in the dev command/diagnostic paths only and does not modify
Unity physics, vehicle parameters, or simulator behavior.

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

The dev-side odometry observer now has a causal wheel-slew/slip gate at
`40 m/s^2`. It rejects abrupt wheel-rate changes that are inconsistent with
the current body-speed estimate, requires causal-speed agreement before
recovery, and automatically enables the validated dynamic bicycle turn
propagation only while the wheel measurement is rejected. Normal turns still
use the stable wheel-anchored mode. The rebuilt C++ observer and Python
reference match over all retained source replays to machine precision with
zero flag mismatches. Across moving samples, body-forward p95 error is about
`0.37 m/s` and body-lateral p95 error about `0.26 m/s`; the E1 slip run is
approximately `0.50/0.18 m/s`. This is an observer candidate improvement,
not a claim of the 2% final acceptance target. The runtime remains dev-side;
no Unity physics or behavior was changed.
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

## Simulator-native strategy phase

The latest simulator-native strategy is now represented by the following
offline-only tools and artifacts:

- `tools/model_id/wheel_friction_curve.py` implements the documented two-piece
  cubic-Hermite surrogate for the public Unity friction breakpoints. It does
  not claim to reproduce Unity's hidden interpolation exactly.
- `tools/model_id/simulator_native_model.py` implements the VS1 virtual axle
  and VS2 four-contact/Ackermann candidate structures. It supports the
  compatibility mean-wheel state and a causal left/right encoder-state form
  for four-contact replay. Its state stores encoder-derived wheel surface
  speed in m/s; the slip calculation therefore does not multiply that state by
  wheel radius a second time.
- `tools/model_id/fit_simulator_native_model.py` fits only effective
  translational/lateral/yaw gains. It does not fit or invent `I_z`, and it
  keeps simulator truth offline. Recursive scoring now propagates each origin
  once to 2 s and samples every requested horizon, avoiding repeated replay
  of the same source transitions.
- `tools/model_id/analyze_simulator_diagnostics.py` validates the required
  diagnostic-only Unity parameter dump and derives yaw inertia from the full
  principal inertia tensor and tensor rotation.

The dump parser independently projects Unity's principal inertia tensor onto
the body vertical axis and rejects a dump when its optional convenience scalar
disagrees. It also preserves all four wheel friction curves and applies a
non-zero pose/velocity lever arm only when it comes from the diagnostic dump.

`model_fits_v2/simulator_native_model_benchmark_v4.json` is the corrected
full-horizon comparison over 21,774 causal training transitions and 12 held-out
runs. It includes all required horizons from 25 ms through 2 s and samples 200
deterministic origins per validation run for recursive scoring. VS2 is only
marginally better than VS1 at the longer horizons; at 0.50 s its validation
position p95 is approximately 0.339 m versus 0.346 m, and at 2.0 s it is
approximately 6.07 m versus 6.19 m. Both candidates fail the precision gate
and remain unaccepted. No C replay or production MPC update has been made.

`reference_point_audit_v2.json` audits all 32 accepted runs. The best
candidate is that the recorded position point is approximately 0.15532 m
behind the reported body-velocity point: aggregate midpoint-kinematic p95 is
0.0112 m/s, versus 0.2282 m/s when both are treated as the same point. This is
offline evidence only; runtime frames were not changed. A full 1,500-origin
candidate replay of that correction improved the 25 ms position p95 from
approximately 0.010 m to 0.003 m, but worsened the 0.50 s and 2.0 s position
p95 to approximately 0.402 m and 6.52 m from 0.339 m and 6.07 m. It is
therefore retained as an explicit diagnostic-derived option, not enabled for
the accepted replay candidate.

The four-contact side-wheel screen uses the two legal encoder streams to avoid
collapsing left/right surface speeds. It is currently a candidate-only path;
the reduced side state did not yet beat the mean-wheel VS2 validation profile
in the sampled screening run, so it has not been promoted or ported to C.

The mixed one-step/recursive effective-gain fit is recorded in
`model_fits_v2/simulator_native_model_benchmark_v7_mixed_prefab_screen.json`.
It uses the checked-in F1TENTH prefab geometry/curve profile, 232 dynamic
training origins per candidate, robust residuals at 0.10--1.00 s, and a
held-out screen of 200 origins per validation run. The best screen result
reduced VS1 validation position p95 to approximately 0.14 m at 0.50 s,
0.60 m at 1.00 s, and 2.84 m at 2.00 s; this was a genuine recursive
held-out improvement over the derivative-only fit, but the optimizer stopped
at its configured evaluation cap.

The fixed-gain full replay is recorded in
`model_fits_v2/simulator_native_model_benchmark_v8_mixed_prefab_full_replay.json`.
It replays the exact v7 gains without refitting over the complete deterministic
origin population (up to 1,500 origins per validation run) and all horizons
from 25 ms through 2 s. VS1 is the better full-replay candidate: validation
position p95 is approximately 0.157 m at 0.50 s, 0.628 m at 1.00 s, and
3.067 m at 2.00 s. VS2 is approximately 0.162 m, 0.669 m, and 3.412 m at
the same horizons. These are offline effective-model results, not a physical
parameter identification or a production acceptance. The fixed replay tool
now separates optimizer fitting from high-population validation so this
comparison is reproducible.

The Unity diagnostic exporter is present in the external simulator checkout at
`Assets/Scripts/ModelIdentificationDiagnostics.cs`, with a disposable-scene
build path in `Assets/Editor/BuildLinuxPlayer.cs`. The Unity 2022.3.52f1 editor
was found and used in batchmode to build the disposable player; no
`-nographics` flag was used. The source competition scene was restored after
the build. The build succeeded despite pre-existing missing-prefab and
LLMUnity asset-copy warnings.

The first no-command diagnostic run exposed a CSV logger schema bug and was
not used for fitting. The logger was corrected to write the root quaternion
that its header declared. A second run used the opt-in offline-only
`ramp_sweep_v1` command profile and is preserved under
`diagnostics_20260914/`. It is 51.856 s of simulator time, 51,857 fixed-step
rows, 110/110 valid columns, and 42 MB, below the 100 MB repository limit.
The player ran in batchmode with graphics enabled and the competition physics
was not changed.

The static dump now establishes Gate B for this build: Unity fixed timestep is
0.001 s, dumped yaw inertia is 0.0276986 kg m^2, all four wheel curves and
wheel dynamic settings are present, and the dump is marked
`simulatorPhysicsUnmodified=true`. It also records differences from the guide:
the controller radius is 0.0325 m while WheelCollider radius is 0.059 m, and
the longitudinal curve values are 0.90/0.58 rather than the guide's 0.72/0.464.
These values must remain provenance-labelled rather than silently reconciled.

The dynamic analyzer is
`tools/model_id/analyze_wheel_contact_diagnostics.py`. The sweep confirms
monotonic 1 kHz physics timing, forward speed up to 15.3 m/s, raw Unity
Ackermann side/sign behavior, and highly correlated but non-identical front
/rear wheel RPM states. The stream contains `WheelHit.force` and direction
data but no separate tire Fx/Fy magnitudes, so it reports a normal-load proxy
and deliberately does not fabricate normalized friction curves. Gate C is
therefore only partially addressed; a force-law fit still requires either
force components exposed by Unity or an explicitly effective acceleration /
moment inversion.

The exporter was then completed to schema v2 and rebuilt in Unity batchmode.
The fresh artifacts are under `diagnostics_20260914_v2/`; the trace contains
52.053 s of simulator time, 52,054 rows, 110 columns, exact one-step
fixed-step continuity, and median `dt` approximately 0.001 s. All four wheels
now report sprung mass and suspension spring/damper/target-position values,
and the Rigidbody reports `maxAngularVelocity=7.0 rad/s`. The model parser
preserves these values as offline structural provenance. The small serialized-
float discrepancy in the dumped convenience yaw-inertia scalar is accepted by
a bounded cross-check tolerance; material disagreement is still rejected. No
simulator physics or runtime control path was changed.

## Latest push review: residual slices and VS2.5 preparation

The latest review is
`SDU_Apex_Latest_Push_Model_Analysis_and_Development_Plan_2026-09-13.md`.
Its main conclusion is followed here: the current error tail is a lateral/yaw
structure problem, so no new broad generic dataset was collected and no
production MPC changes were made.

`model_fits_v2/simulator_native_model_benchmark_v9_event_preserving_screen.json`
records the event-preserving screen fit. It deliberately retains steering
slew/reversal, high-yaw, and lateral-state origins before uniform fill. The
optimizer still stopped at its configured screen cap; validation position p95
for VS1/VS2 at 0.50 s was approximately 0.118/0.113 m and at 2.00 s
approximately 1.751/1.762 m. Neither candidate is accepted.

`model_fits_v2/simulator_native_residual_slices_v8.json` ranks the failure
regimes from the immutable v8 replay. At 2.00 s, the largest VS1 run p99
errors were steering reversal run 62 (11.01 m) and combined
throttle/steering runs 04/05 (10.65/10.55 m). The high-`|Sy|` slice had a
9.70 m p95, while low-demand origins were approximately 0.11 m p95. This
supports prioritizing contact-force/moment structure over another optimizer
budget on the unchanged VS1/VS2 equations.

The offline model now includes a VS2.5 four-contact normalized force/moment
basis in `tools/model_id/simulator_native_model.py`. It evaluates all four
wheel slip responses, rotates each tire force into the body frame, and forms
`qX`, `qY`, and `qM = sum(x_i*qY_i - y_i*qX_i)`. Its fitted coefficients remain
explicit effective simulator gains; no `I_z` is invented or promoted. The
corresponding fit path and candidate are in
`tools/model_id/fit_simulator_native_model.py`; this candidate is still
offline-only and awaits the screen/full-replay result.

The converged VS2.5 screen is recorded in
`model_fits_v2/simulator_native_model_benchmark_v11_vs25_converged_screen.json`.
The optimizer terminated by `xtol` after 24 evaluations and the yaw-moment gain
was no longer at the temporary bound. The complete fixed-gain replay is
`model_fits_v2/simulator_native_model_benchmark_v12_vs25_full_replay.json`.
At up to 1,500 origins per validation run, VS2.5 validation position p95 is
approximately 0.123 m at 0.50 s, 0.433 m at 1.00 s, and 1.385 m at 2.00 s.
The same replay reports `v` p95 of approximately 0.059/0.079/0.154 m/s and
`r` p95 of approximately 0.192/0.302/0.353 rad/s at those horizons. This is a
material offline improvement over v8 VS1, but it remains above the desired
precision and has no native parity, observer, or blind-run acceptance.

The remaining tail is recorded in
`model_fits_v2/simulator_native_residual_slices_v12_vs25.json`. The worst
validation runs are combined throttle/cornering runs 04/05, with 2.00 s
position p95 approximately 6.04/5.99 m in the 200-origin slice; the top
origins reach approximately 13.25 m. Low-demand origins are approximately
0.08 m p95 at 2.00 s. This is the expected signature of unresolved
load/contact dynamics and is not addressed by promoting the effective model
or by changing the simulator.

The diagnostic analyzer now reports all wheel curves, contact geometry,
mass/load semantics, suspension/damping, rigid-body damping, solver/timing,
and inertia cross-checks. The exporter source is versioned at
`tools/model_id/unity/ModelIdentificationDiagnostics.cs` and matches the
external diagnostic component. It remains a disposable diagnostic artifact;
it does not alter the competition simulator.

## Steering-map experiment and current model decision

The dynamic trace shows that the raw Unity `WheelCollider.steerAngle` sign and
left/right Ackermann ordering differ from the API-frame steering convention
used by the accepted replay data. Both conventions are now explicit in
`simulator_native_model.py`; the raw Unity formula is retained as a diagnostic
reference and is not silently applied to API-frame states.

The corrected raw-map VS2.5 fit is recorded in
`model_fits_v2/simulator_native_model_benchmark_v13_vs25_unity_steering_map.json`.
It terminated after 11 evaluations with poor conditioning and worsened held-
out 2.00 s position p95 to approximately 4.89 m, versus approximately 1.38 m
for the prior API-frame v11/v12 candidate. It is rejected. The prior effective
API-frame convention remains the current offline baseline, and no production
MPC change has been made.

The optional effective yaw-damping audit is recorded in
`model_fits_v2/simulator_native_model_benchmark_v14_vs25_yaw_damping.json`.
The damping-enabled VS2.5 fit drove the new coefficient to the lower bound
(`approximately 5e-11 1/s`) and did not improve held-out replay: the 200-origin
screen gave position p95 of approximately 0.119/0.419/1.411 m at 0.50/1.00/2.00
s, versus 0.118/0.409/1.399 m for v11. VS1 and VS2 also remained worse in the
multi-step comparison. This is evidence against adding an unobserved residual
damping term to the candidate; it is retained only as an offline audit field,
not as a production or baseline parameter.

The same fixed gains were then scored over the full deterministic validation
population in
`model_fits_v2/simulator_native_model_benchmark_v15_yaw_damping_full_replay.json`.
VS2.5 with damping reached position p95 approximately
`0.123/0.434/1.390 m` at 0.50/1.00/2.00 s, compared with
`0.123/0.433/1.385 m` for the retained v12 replay. The full replay therefore
confirms the screen decision: damping is rejected and the v12 VS2.5 gains
remain the best current offline baseline.

The v2 contact trace also showed strong measured lateral load-transfer
correlation, so a causal lateral-load-transfer screen was run without changing
the v12 gains. Its results are preserved in
`model_fits_v2/simulator_native_load_transfer_screen_v1.json`. With the
effective `a_y ~= r*u` estimate, heights of 0.08 m and 0.16 m worsened 2.00 s
position p95 from 1.399 m to 3.132 m and 3.758 m respectively. The structure is
rejected: contact-load correlation alone is not enough to infer a usable
reduced-model load law, and the default height remains zero.

The next model step is to use targeted diagnostic data to identify effective
force/moment behavior without conflating raw Unity axes with API axes. The
v2 dump and sweep pass the provenance/timing/static-field checks, but they do
not yet pass
the contact-law, native-parity, observer, blind-run, or shadow-MPC gates.

## VS2.75 split-contact structural screen

The next structural experiment retained the full four-contact body-frame
geometry but split the lateral and yaw terms by front/rear axle. It explicitly
keeps the complete longitudinal yaw contribution `-y_i*Fx_i` rather than
discarding left/right longitudinal asymmetry. The implementation is an
offline-only `moment_basis_split` candidate in
`tools/model_id/simulator_native_model.py` and
`tools/model_id/fit_simulator_native_model.py`.

The screen fit is recorded in
`model_fits_v2/simulator_native_model_benchmark_v16_vs275_split_moment_basis.json`;
the full fixed-gain replay is
`model_fits_v2/simulator_native_model_benchmark_v17_vs275_split_moment_basis_full_replay.json`.
On the complete deterministic validation population, position p95 improved
from v12's `0.123/0.433/1.385 m` to `0.121/0.392/1.234 m` at 0.50/1.00/2.00 s,
and 2.00 s position p99 improved from `5.614 m` to `3.975 m`. The residual
slice report
`model_fits_v2/simulator_native_residual_slices_v17_vs275.json` shows lower
combined throttle/cornering p95 (`3.391` to `3.005 m`) and high-`|Sy|` p95
(`9.485` to `8.895 m`), but steering-reversal p95 worsens (`1.389` to
`2.128 m`) and the optimizer reports high numerical optimality. VS2.75 is
therefore the best provisional offline structure, not an accepted plant; it
still requires optimizer refinement, native parity, observer replay, and
blind-track acceptance before any MPC integration.

The optimizer refinement is recorded in
`model_fits_v2/simulator_native_model_benchmark_v18_vs275_split_moment_basis_refined.json`.
It used 70 event-preserving origins per training run and allowed 60 optimizer
evaluations; the optimizer converged after 23 evaluations. Compared with v17,
it improved validation position p95 at 0.50 s from `0.121` to `0.119 m`, was
effectively unchanged at 1.00 s (`0.392` to `0.392 m`), but worsened the
long-horizon 2.00 s value from `1.234` to `1.277 m`. The 2.00 s residual
slice is also mixed: steering reversal improved from `2.128` to `1.752 m`
and low-demand p95 from `0.084` to `0.079 m`, while combined corner/throttle
and high-`|Sy|` tails did not improve. Its optimizer optimality remains high.
The refinement is therefore retained as an audit candidate, while v17
remains the best provisional long-horizon structure. The residual comparison
is in `model_fits_v2/simulator_native_residual_slices_v18_vs275_refined.json`.

No candidate is promoted to production MPC: native C parity, observer replay,
and blind-track acceptance are still outstanding, and all simulator truth
continues to be used offline only.

## 2026-09-14 latest-plan E0 superseding addendum

The latest review and recovery plan changes the priority order: vehicle-axis
semantics and the hybrid longitudinal actuator must be corrected before tire
law tuning or MPC tuning. The earlier `0.0276986 kg m^2` convenience value is a
local-Z projection and is now explicitly legacy; it must not be interpreted as
the physical yaw inertia.

The disposable Unity diagnostic exporter was corrected to emit all body-frame
inertia projections and `yawAxis=body_y`. A fresh batchmode-with-graphics
diagnostic run passed the parser/analyzer gate with
`I_body_x=0.09352724`, `I_body_y=0.09619075`, and `I_body_z=0.02769764 kg m^2`.
The native plant and MPC constants were intentionally not changed. The durable
E0 report is
`diagnostics_20260914_v2/inertia_axes_e0_20260914.json`.

The corrected existing force/moment screen still gives a lateral effective
scale near unity (`0.973`) but a longitudinal scale near `0.085`, so the next
campaign is the repeated near-zero throttle/brake map and hybrid continuous
wheel model. The corrected inversion summary is
`diagnostics_20260914_v2/body_force_moment_inversion_axis_y_e0_20260914.json`.

The new sequence is: E1 near-zero actuator semantics, E2 repeated powered
drive, E3 corrected force/moment inversion, E4 cross/along-track localization
decomposition, E5 source-time controller contract, then only after accepted
high-fidelity replay derive the reduced MPC model, observer, and shadow MPC.
No simulator physics or competition runtime behavior was changed.

The canonical E1 campaign was then completed with 105/105 successful
normal-graphics batchmode runs: 5 commanded speeds, 7 near-zero commands, and
3 repeats per condition. A 0.1 s disposable 0.65-throttle precondition was used
so WheelCollider RPM was initialized before the measurement command; the
measurement interval is explicitly marked in each trace. The compact report is
`diagnostics_20260914_v2/near_zero_actuator_e1_20260914.json`; raw traces remain
outside the repository under `/tmp/autodrive-model-id-near-zero-e1-clean-20260914`.

E1 confirms the hybrid actuator mapping exactly: command zero applies 428 Nm
brake torque to every wheel, while each positive tested command applies
`107 * command` Nm motor torque per wheel and zero brake torque. No tested
positive command behaved as coast; wheel RPM and body speed collapse after the
command transition. The actuator gate is therefore passed, but the continuous
longitudinal plant remains not accepted and the next step is repeated powered
drive identification with explicit drive-only windows. No simulator physics or
competition runtime behavior was changed.

E2 then completed five independent 51.017 s positive-drive repeats using 17
 three-second upstep/downstep plateaus. All repeats contain 51,018 rows, have
 approximately 1 ms source timing, and contain no brake torque. Fitting the
 existing effective wheel-state basis on repeats 1--3 and validating on repeats
 4--5 gives held-out wheel replay p95 errors of approximately 0.54--0.70
 rad/s at 25 ms, with similar results at 50 ms. The compact report is
 `diagnostics_20260914_v2/powered_drive_e2_20260914.json`.

The optimizer drives the contact-force coefficient to approximately zero on all
four wheels and identifies repeatable motor/damping coefficients instead. This
passes the independent wheel replay gate only; it does not pass body-u
recursive prediction, force/moment parity, or MPC promotion. The measured
contact force/slip regressors remain offline-only, and no simulator physics or
competition runtime behavior was changed.

E3 was rerun with the corrected body-Y inertia and is recorded in
`diagnostics_20260914_v2/body_force_moment_inversion_e3_20260914.json`. The
global screen remains consistent: lateral scale approximately `0.973`,
longitudinal scale approximately `0.085`, and yaw-moment scale approximately
`0.00513`. This confirms that correcting inertia semantics does not explain the
longitudinal mismatch; the actuator/wheel/contact model remains the dominant
open physics issue. E3 is an offline structural screen only and does not
promote a tire law or MPC plant.

## 2026-09-14 latest-plan E4 localization decomposition

E4 was completed offline against the accepted full-speed Pure Pursuit run
`full_speed_mpc_recovery_v1/runtime_track/e1_pp_full_speed_yaw_20260914_1220`.
The new tool is
`sdu_apex_autodrive.odometry_analysis.localization_error_decomposition`; it
uses the existing source-stamped localization report, the bridge source-time
stream, and the saved raceline. It writes per-sample signed cross-track,
along-track, wrapped raceline-s, yaw, speed, curvature, and segment data. The
raw decomposition CSV is kept outside the repository; the compact result is
`diagnostics_20260914_v2/localization_decomposition_e4_20260914.json`.

The source stream contains 6,703 packets with `dt` p95 `26.001 ms` and max
`28 ms`. AMCL is predominantly along-track biased: `/amcl_pose` has cross-track
p95 `0.0313 m` and along-track p95 `0.2133 m`; `/current_map_pose` has
`0.0328 m` and `0.2196 m`. Local `/odom` and `/ekf_odom` are identical in this
run and have mixed cross/along p95 errors of `0.8884/0.8709 m`.

AMCL correction age is normally approximately `25 ms` and scan alignment has
zero source error and zero queue/drop counters. The health diagnostics have no
source header, however, so exact correction age cannot yet be joined to an
individual estimator row. This establishes the E5 requirement for a
source-stamped state-age contract; no AMCL or EKF tuning is promoted from E4.

## 2026-09-14 E5 source-stamped AMCL health contract

The E4 result identified a missing observability contract: AMCL correction age
was available, but the health `Float64MultiArray` had no source stamp. The AMCL
publisher now appends the scan/source timestamp as field 15 while preserving
the original fourteen fields. The diagnostics recorder stores that field and
uses it as `source_event_stamp_s`; old 14-field AMCL binaries retain the
arrival-time fallback. The offline decomposition joins the correction age to
source rows within a 1 ms timestamp tolerance when the extension is present.
The compact implementation record is
`diagnostics_20260914_v2/source_stamped_health_contract_e5_20260914.json`.

This is a dev-side diagnostic/estimator observability change only: it does not
alter the particle filter, simulator, physics, controller inputs, or command
path. A ROS 2 Humble build with CUDA AMCL enabled passed all three localization
C++ tests, and the Python analysis suite passed 119 tests. The existing E4
recording predates field 15, so it correctly remains marked
`state_age_exactly_joined=false`; a new clean localization recording is still
required to validate the source-stamped health join live.

## 2026-09-14 E5 source-time contract live validation

The source-epoch bridge contract was validated in a clean track run using Unity
batchmode with normal graphics. The compact evidence is
`diagnostics_20260914_v2/source_time_contract_e5_live_20260914.json`; the raw
run is `accepted/82_e5_source_epoch_fault_diagnostic_track_40hz_20260914`.

The bridge recorded 1,331 valid simulator packets over 33.249 s of source time.
Source intervals were 17.000--33.002 ms, with zero samples outside the strict
15--35 ms window. Request and telemetry sequences had zero gaps or reversals,
and the applied command lag was exactly one packet. The two source intervals
above 30 ms were still valid 33 ms intervals, not 40 Hz failures. Pure Pursuit
reached repeated `lap_fraction=1.000` with zero recorded collisions. The only
timing fault was the expected Socket.IO disconnect when the test player was
stopped.

This validates the source-time/header contract and the fail-closed diagnostic
path, but it does not promote the current continuous body-dynamics candidate
or MPC tuning. The next implementation phase is a dev-side dynamic observer
and replay score using legal sensor/runtime inputs only; simulator truth and
embedded simulator dynamics remain offline diagnostics.

## 2026-09-14 E6 MPC dynamic observer prototype

The source-time observer prototype is implemented in
`f1tenth_mpc/include/mpc_observer.h` and `f1tenth_mpc/src/mpc_observer.c`.
It carries `[u, v, r, delta, q_drive]`, rejects source intervals outside
15--35 ms, exposes source state age for a future solve/application epoch, and
uses no simulator truth at runtime. Its source-time update consumes odometry
speed, IMU yaw rate/lateral acceleration, source-associated applied steering,
and applied throttle. The bridge timing record is used for the latter two
until typed feedback messages carry source headers.

The offline fit/evaluation tool is `tools/model_id/evaluate_mpc_observer.py`.
The compact evidence is
`diagnostics_20260914_v2/mpc_dynamic_observer_e6_20260914.json`: run 81 was
the training trace and the independent run 82 was held out. On run 82, `r`
matched the recorded IMU/vehicle yaw rate exactly, `u` had p95 absolute error
`0.342 m/s` after the causal IMU-`a_x`/odometry correction, and the prototype
`v` estimate had p95 absolute error `0.00404 m/s` and maximum error
`0.01072 m/s`. These are absolute metrics
because lateral velocity crosses zero; they must not be reported as a
misleading percentage.

The observer remains unaccepted: `u`, `v`, and `r` are all marked false until
open-scene and cross-regime data are available, and the current `v` mapping is
an empirical observer prototype rather than an accepted high-fidelity plant.
The native MPC model also no longer uses the invalid `0.01434 m` Unity-local
COM coordinate as a physical load-transfer height; that parameter is now
explicitly unresolved and load transfer is disabled pending measurement.

## 2026-09-14 E7 LiDAR transport-cache validation

The two rejected long runs E6-83 and E6-84 both lost simulator telemetry at
the same sequence transition (`1010 -> 1012`). Bridge callback serialization
did not change that result. Source inspection identified a deterministic
performance hotspot: the Unity socket recompressed the complete LiDAR array on
every 40 Hz reply, although the sensor scans at 20 Hz. The Unity-side fix
caches only the compressed transport string per completed scan. It does not
change raycasts, range values, scan timing, collision geometry, rigid-body
settings, or controller behavior. The player was rebuilt from a temporary copy
of the checked-out competition scene, leaving the source scene untouched.

The accepted validation run is
`accepted/85_e7_lidar_cache_validation_track_40hz_20260914`. It used
batchmode with normal graphics and Pure Pursuit. It captured 3,121 valid
bridge packets through source time `78.135 s`; source intervals were
`21.999--27.999 ms` with zero intervals above 30 ms, zero request gaps, zero
telemetry gaps, zero reversals, and exactly one packet of applied-command lag.
The prior `1010 -> 1012` loss did not recur. The compact evidence is
`diagnostics_20260914_v2/timing_cache_e7_20260914.json`, and the raw event log
and finalized timing partitions remain in the accepted run directory.

Using the same dev-side observer fit as E6, the independent E7 validation gives
`u` p95 absolute error `0.357 m/s`, `v` p95 absolute error `0.00407 m/s`, and
`r` p95 absolute error `0`. These values are offline scoring against embedded
simulator truth only; no simulator truth enters runtime. The observer and MPC
plant remain unpromoted pending the planned open-scene, cross-regime, and
longitudinal-model validation.

An independent repeat, `accepted/86_e7_lidar_cache_repeat2_track_40hz_20260914`,
captured 3,403 valid packets with source intervals `21.998--28.000 ms`, zero
source/request/telemetry gaps, zero reversals, and one-packet command lag. Its
observer score was `u` p95 `0.332 m/s`, `v` p95 `0.00428 m/s`, and `r` p95 `0`.
The repeat report is
`diagnostics_20260914_v2/timing_cache_e7_repeat2_20260914.json`; the combined
two-run observer report is
`diagnostics_20260914_v2/mpc_dynamic_observer_e7_cross_run_20260914.json`.
Together E7 provides 6,524 clean source-time validation packets after the
cache change, but it is still one track regime and therefore not a promotion
gate for the observer or MPC plant.

## 2026-09-14 E8 request-identity and track regression

The bridge now associates every accepted telemetry response with the exact
Bridge request that produced it. Batchmode no longer emits an unsolicited
bootstrap telemetry packet, while interactive GUI startup retains its manual
connection behavior. A response without a request identity is ignored as
bootstrap traffic; a response with the wrong identity is a fatal transport
fault. This prevents reconnect/bootstrap packets from being mistaken for a
missing 40 Hz sample. The change is transport bookkeeping only: it does not
alter Unity physics, sensor rays, scene geometry, or vehicle behavior.

The clean open-scene run
`accepted/91_e8_open_scene_speed_steps_rebuilt_workspace_20260914` completed
1,200 requested speed-step packets with source intervals
`21.999--28.002 ms`, zero source/request/telemetry gaps, zero duplicate or
reverse sequences, and one-packet command lag. All four commanded speed phases
completed. Its causal transition assembly passed with 1,199 transitions and a
body-frame displacement p95 of `0.00423 m/s`.

The independent track regression
`accepted/92_e8_track_transport_regression_20260914` completed 3,583 packets,
six raceline crossings, and no recorded collision events. Source intervals
were `20.990--29.000 ms`, with zero source gaps above 30 ms, zero request or
telemetry gaps, zero sequence reversals, and exactly one packet of applied
command lag. Pure Pursuit stayed on the raceline; the run is a transport and
regression validation, not the full-speed MPC gate because its measured peak
speed was `8.34 m/s`.

The compact evidence is
`diagnostics_20260914_v2/track_regression_e8_20260914.json`. The legal-input
observer scored the track holdout at `u` p95 `0.3525 m/s`, `v` p95
`0.00442 m/s`, and `r` p95 `0`. A new mixed-recursive VS2.75 split-contact
candidate scored the independent track run at position p95 `0.0866 m` at
0.50 s, `0.2025 m` at 1.00 s, and `0.4314 m` at 2.00 s. The optimizer reached
its evaluation limit and native C parity/blind full-speed acceptance are still
outstanding, so no model or production MPC parameters were changed.

## 2026-09-14 PR1 explicit drive/brake controller semantics

The speed-controller actuator contract is now explicit in
`sdu_apex_autodrive/sdu_apex_autodrive/speed_controller.py`. It exposes
`LongitudinalMode.DRIVE`, `LongitudinalMode.BRAKE`, and
`LongitudinalMode.STOP` through `LongitudinalCommand`, while retaining the
existing normalized-throttle return path for compatibility. Positive throttle
is `DRIVE`; zero throttle selected for overspeed or target downshift is
`BRAKE`; watchdog/reset/target-zero output is `STOP`. The measured simulator
maps both brake and stop to zero on the legacy wire topic, so this is a
dev-side semantic correction and does not alter simulator behavior.

The Humble package rebuilt successfully and the focused Python suite passed
`59` tests, including the new semantic and immediate-brake tests. The first
runtime smoke run used the calibration node's built-in `0.5--2.0 m/s` sequence
and completed its brake confirmations. Its temporary parameter-file mistake
was corrected without changing tracked runtime configuration.

The intended independent `4--20 m/s` regression is
`accepted/94_pr1_speed_steps_4_to_20mps_20260914`. It captured `2,412`
bridge packets with source intervals `21.999--28.000 ms`, zero sequence gaps,
zero telemetry gaps, and exactly one packet of applied-command lag. Targets
`4, 6, 8, 10, 12, and 14 m/s` settled. The `16 m/s` target did not: the legal
conditioned speed estimate reached the target while embedded simulator truth
peaked at `15.5615 m/s`; the trace shows growing wheel/ground-speed
disagreement and no collision-count increase. This is an odometry/longitudinal
validity failure, not a transport failure. The compact evidence is
`diagnostics_20260914_v2/speed_controller_pr1_evidence_20260914.json`.

PR1 is therefore implemented and unit/runtime-smoke validated, but the
high-speed controller is not promoted and no MPC weights, plant constants, or
simulator physics were changed. The next step is to use this clean failure to
improve the legal longitudinal observer/hybrid plant, then repeat the
independent speed/brake holdout.

## 2026-09-14 PR5 odometry dropout and coherent-pose validation

The observer no longer integrates longitudinal braking twice when a repeated
encoder packet activates the turn-dropout dynamic path. The second frozen-wheel
braking branch is now mutually exclusive with the RK2 dropout propagation, and
`test_turn_dropout_does_not_integrate_braking_twice` covers the regression.
On `accepted/102_pr5_turn_dropout_single_integration_20260914`, the recorded
turn-dropout longitudinal error fell to approximately `0.121 m/s` p95 from the
`1.1--1.3 m/s` p95 errors in the earlier runs. The run remained on the track
until the reproducible Unity CPU stall at source time `113.105 s`; it is not a
full-lap acceptance result.

The active observer also has a bounded coherent-current-packet path used only
for pose trapezoidal integration. The rolling 100 ms encoder estimate remains
the published longitudinal state, so burst rejection and controller speed
semantics are unchanged. The current packet is used only when it is nonzero,
non-burst, and agrees with the rolling estimate. This is a legal sensor-side
estimator change and does not modify simulator physics.

Source-time replay of the same observer on runs 97, 98, and 102 reduced pose
position p95 from `0.692` to `0.472 m`, `1.413` to `1.001 m`, and `0.765` to
`0.482 m`, respectively. A clean live validation,
`accepted/103_pr5_pose_packet_live_20260914`, captured `1,914` packets at a
median source rate of `40.000 Hz`, source intervals `21.999--28.000 ms`, zero
request/telemetry gaps, zero sequence reversals, and one-packet command lag.
Its active `/odom` scores are position p95 `0.466 m`, longitudinal
pose-point velocity p95 `0.174 m/s`, lateral pose-point velocity p95
`0.086 m/s`, and yaw-rate p95 `0`. `/current_map_pose` position p95 was
`0.116 m`; AMCL remained stable enough for Pure Pursuit to complete multiple
laps, but the run does not pass the `0.05 m` map-position target.

The full-speed report now distinguishes the simulator rigidbody/COM lateral
velocity from the GPS/reference-point velocity represented by the odometry
pose. Raw COM lateral velocity is retained as a diagnostic; it is not used as
an odometry error target for the pose point.

The expanded mixed-recursive VS2.75 fit
`model_fits_v2/simulator_native_model_benchmark_pr5_full_train_validate103_20260914.json`
was rejected. Adding run 102 to the training set reduced local training error
but diverged on the independent run 103 (`1.313 m` position p95 at `1.00 s`,
`6.439 m` at `2.00 s`). The prior cross-regime candidate replayed in
`model_fits_v2/simulator_native_model_benchmark_pr5_replay_e8_on_103_20260914.json`
remained the better holdout (`0.196 m` at `1.00 s`, `0.419 m` at `2.00 s`),
but still requires native C parity and blind closed-loop validation. No MPC
model, weights, simulator physics, or runtime ground-truth input were changed.

## 2026-09-15 source-time wheel model and odometry recovery fixes

The longitudinal identification tool now fits an explicit source-time
continuous wheel-speed derivative model in addition to the legacy discrete
wheel model. Both models use the measured source `dt`; neither consumes
future ground truth during recursive rollout. The same choice is represented
in the Python reference plant and native `vehicle_plant.c`, including the
offline hybrid zero-throttle hard-brake force fit. The open/track fit estimated
`18.9627 N` hard-brake force from the current v6 fit (`82` qualifying
transitions). Native/Python
parity remains clean: the v6 fixture report is below `8.3e-7` maximum absolute
error, and the fixed-inertia lateral replay parity report is below `1.7e-6`
against a `3.0e-5` tolerance.

The observer recovery path now re-anchors `body_u` to a fresh synchronized
wheel packet after a turn dropout instead of allowing the same packet's IMU
RK2 update to overwrite the recovered speed. Turn-dropout RK2 propagation now
also applies the calibrated braking acceleration (`decel_ax_scale` and
`decel_ax_offset`) that was previously bypassed in that branch. The Python
reference and C++ implementation are covered by native regression tests and
remain runtime-input compliant; simulator truth is used only in offline
scoring.

The clean post-fix track runs were
`accepted/119_pr14_turn_recovery_reanchor_20260915` and
`accepted/120_pr15_turn_dropout_calibrated_braking_20260915`. Both preserved
the 40 Hz source contract with zero request/telemetry sequence gaps, one
packet of command lag, no collision events, and repeated full-speed Pure
Pursuit laps. Run 119 scored `/odom` position p95 `0.2822 m`, longitudinal
velocity p95 `0.1268 m/s`, lateral velocity p95 `0.0656 m/s`, and current-map
position p95 `0.1059 m`. Run 120 scored `0.3743 m`, `0.1320 m/s`,
`0.0659 m/s`, and `0.0925 m`, respectively. The run-to-run variation means
the braking correction is a safety/no-regression fix, not a claimed universal
2% result. The source-time braking sweep retained the active
`1.005 / 0.020 m/s^2` calibration as the best tested safe point.
A separate bounded dropout-onset reuse of the preceding coherent current
packet was tested in the Python reference and rejected: over both runs it
extended the dropout state and increased position p95 from `0.338 m` to
`1.546 m`. That candidate was removed from both the C++ observer and the
reference replay, so it cannot silently affect runtime.

## 2026-09-15 fixed-inertia lateral candidate replay

The lateral fitter now supports keeping the corrected body-Y yaw inertia fixed
at `0.0961908 kg m^2` rather than re-fitting it inside the old `0.003--0.080`
bound. The resulting candidates are in
`model_fits_v2/lateral_model_benchmark_fixed_iz_20260915.json`. On the
combined clean 119/120 recordings, the fixed-inertia Y2 tanh candidate is the
best current offline composed-plant candidate. Its native recursive position
p95 is `0.0396 m` at `0.10 s`, `0.1884 m` at `0.50 s`, `0.3701 m` at `1.00 s`,
and `0.7447 m` at `2.00 s`; lateral-velocity p95 is `0.0187`, `0.0211`,
`0.0216`, and `0.0217 m/s`. The old unconstrained Y2 candidate reaches
`2.249 m` at `1.00 s` and `7.492 m` at `2.00 s` on the same replay. The
fixed candidate is therefore selected for the next blind validation, but it
has not been loaded into runtime MPC and is not promoted: the 2-second error
still exceeds the project target, and closed-loop blind validation remains
required.

The corresponding native replay is
`model_fits_v2/native_replay_Y2_fixed_iz_wheel_continuous_v6_on_119_120_20260915.json`;
the parity evidence is
`model_fits_v2/native_plant_parity_fixed_iz_20260915.json`. The replay is
offline-only and does not modify simulator physics, runtime odometry, MPC
weights, or the simulator data path. Focused Humble tests pass `44` tests
with `3` non-fatal dependency warnings.

## 2026-09-15 residual-driven plant consistency implementation

The structured offline plant and its native C mirror now use the same current
offline candidate defaults: corrected Unity body-Y inertia
`0.0961908 kg m^2`, mass `3.47 kg`, measured contact geometry, continuous
wheel dynamics, tanh lateral forces, the v6 wheel/brake fit, and the fixed-
inertia damped Y2 lateral fit. The recorded pose point is propagated with the
measured `-0.155320086 m` rear-position/COM-velocity lever arm. The lever arm
changes pose propagation only; it does not change body `u`, `v`, or `r`.

Unity Rigidbody drag and angular drag are represented explicitly in the
offline plant as `0.273 /s` and `0.1 /s`, with the fitted longitudinal drag
force term set to zero in the measured-damping profile so it is not counted
twice. A legacy fitted-drag profile remains selectable for A/B comparison.
This is offline model code only: production MPC, runtime odometry, simulator
physics, scene geometry, and simulator data inputs are unchanged.

On the available clean runs 92 and 103, the lever-arm A/B reduced aggregate
0.50 s position p95 from `0.1887 m` to `0.0558 m` without changing body-state
errors. The measured-damping profile then reduced aggregate 0.50 s position
p95 to `0.0533 m`, yaw-rate p95 to `0.1611 rad/s`, and longitudinal-speed
bias to `0.0196 m/s`; both individual runs improved or held their primary
metrics. Adding measured linear drag on top of the fitted drag was rejected
because it over-damped speed.

Native/Python parity now also checks the default parameter structure, not only
fixture propagation. The damped fixed-inertia parity report passes with a
maximum propagation difference of `8.1e-7` and default-parameter difference
of `7.5e-9`, against tolerances of `3e-5` and `2e-4`.

The residual attribution tool is
`tools/model_id/analyze_native_plant_residuals.py`. Its 92/103 report is
`model_fits_v2/native_residual_analysis_damped_92_103_20260915.json` with the
transition table in
`model_fits_v2/native_residuals_damped_92_103_20260915.csv`. Leave-one-run-out
regression identifies `v`/`r` as the strongest held-out yaw residual terms and
rear slip as the strongest lateral-acceleration residual term. No residual
regression term has been inserted into the plant; the next change must first
be tested as a physical mechanism on independent runs.

The odometry reference replay was also corrected to load the deployed
`f1tenth_localization/config/sensor_odometry.yaml` instead of silently using
its separate constructor defaults. The standalone C++ replay now uses the
same deployment profile, including the speed table, coherent pose packet,
bounded turn correction, and bounded sideslip. On the available 980-packet
source replay from run 20, Python/C++ outputs matched to below `4e-15` in
position and velocity, with zero flag mismatches. This changes offline replay
consistency only; the sensor node still receives its active values from the
same YAML file.

The node's parameter-declaration fallbacks and the direct Python
`ReferenceObserver()` defaults are now sourced from the same deployment
profile as the active YAML values. This removes the remaining stale fallback
copies (including the old disabled coherent-pose and sideslip settings). The
Humble build and native tests pass after this change; the complete Python
suite passes `130` tests.

The residual diagnostic was then expanded from the two clean replay runs to
`33` dynamic accepted runs and `58,709` transitions. The full composed native
replay of the fixed-inertia damped Y2 plus continuous-wheel candidate gives a
0.50 s position p95 of `0.1358 m`, heading p95 `0.0698 rad`, yaw-rate p95
`0.1776 rad/s`, and body-speed p95 `0.2170 m/s`. These are archive-wide
offline scores, not a live-racing acceptance claim. Leave-one-run-out
regressions do not improve yaw residuals consistently across the archive;
therefore no residual correction term has been forced into the physical plant.
The full reports are
`model_fits_v2/native_replay_Y2_fixed_iz_damped_wheel_continuous_all_dynamic_20260915.json`,
`model_fits_v2/native_residual_analysis_damped_all_dynamic_20260915.json`, and
`model_fits_v2/native_residuals_damped_all_dynamic_20260915.csv`.

## 2026-09-15 model-authority gate

The active authorities are now explicit in
`f1tenth_mpc/config/vehicle_model_manifest_v1.json`:

- `production_mpc_baseline` is the only active runtime MPC profile. It remains
  the BachelorProject six-state Frenet/augmented model and is unchanged by the
  offline candidate work.
- `offline_candidate_fixed_iz_damped_y2_continuous_wheel` is the only canonical
  replay candidate. Its resolved Python and native C defaults now use the same
  fixed-inertia damped Y2 lateral parameters, continuous wheel derivative,
  active zero-throttle braking, measured body damping, and rear-pose lever arm.
- Historical model reports and simulator-native diagnostic models remain
  offline evidence only; they are not active runtime or canonical replay
  authorities.

The manifest gate checks production constants/structure, candidate report
provenance, Python/native offline defaults, and the complete deployed odometry
YAML/C++/Python mapping. The clean Humble milestone passed `7` tests with
`0` errors and `0` failures. Candidate/runtime compatibility is deliberately
still `false`: the candidate has an eight-state normalized-throttle/wheel-state
architecture while production MPC has a six-state direct-acceleration
architecture. No candidate parameter has therefore been copied into runtime
MPC.

The compact live capture
`run_live/compact_capture_baseline_20260915/` passed the source timing gate
for `30.13 s` and `556` packets: mean source interval `25.002 ms`, maximum
`28.001 ms`, zero source gaps over `30 ms`, and no sequence gaps. It is a
transport/capture acceptance artifact, not a high-speed model-identification
run. The next data step is a compact high-speed lateral excitation campaign,
followed by whole-run held-out residual scoring and only then the explicit
plant-to-MPC architecture migration.

## 2026-09-15 high-speed longitudinal identification and source shutdown fix

The open-plane model-identification player was run in `-batchmode` with the
existing disposable no-LiDAR diagnostic build. No Unity physics, timestep,
scene geometry, or runtime controller behavior was changed. The accepted
high-speed data now includes independent 0.70, 0.75, 0.80, 0.85, and 0.90
throttle regimes, 16 m/s and 20 m/s zero-throttle braking runs, a continuous
0.70 -> 0.80 -> 0.85 -> 0.90 -> 0.75 -> 0.00 throttle sweep, and a repeated
0.75 run. Every accepted run has one causal source segment and passes the
40 Hz source gate. The new sweep has `1,461` packets and `1,460` transitions,
mean source interval `25.000 ms`, range `21.999--28.000 ms`, zero sequence
gaps, zero source gaps over `30 ms`, and one-packet command lag. The isolated
constant-throttle runs each have `1,120` transitions and zero collisions.

The isolated 0.75 repeat reaches `17.5685` versus `17.5691 m/s`, confirming
repeatability of that equilibrium point. Refitting the pre-gate high-speed
data selected `wheel_continuous` again. The superseded v3 report is
`model_fits_v2/longitudinal_model_benchmark_measured_damping_holdout20brake20driven_20260915.json`.
It uses the measured `0.273 /s` Unity body damping once and sets the fitted
speed-drag term to zero; the independent 20 m/s braking and driven runs are
both held out. The combined validation is `5.27%` relative p95 at `0.50 s`
and `9.22%` at `2.00 s`. Those values remain historical offline evidence only;
the v4 clean-data report below is the current candidate authority.

The report-driven native residual analysis is
`model_fits_v2/native_plant_residuals_measured_damping_20260915.csv` and its
JSON report. Across `11,550` transitions, the corrected plant has
longitudinal derivative residual MAE `0.250 m/s^2`, p95 `0.837 m/s^2`, lateral
derivative MAE `0.418 m/s^2`, and yaw derivative MAE `0.276 rad/s^2`. The
measured-damping correction reduces longitudinal bias from `0.259` to
`0.047 m/s^2` and reduces the high-speed (>16 m/s) bias to `0.131 m/s^2`.
The remaining high-speed error points to an unmodeled driven transient or
state-memory effect, not a reason to force an arbitrary correction term into
MPC.

The manifest was temporarily version `2026-09-15.v3` for this historical
comparison and has since moved to v4 after the off-plane data gate. Native C
and Python offline defaults are now standardized against the clean v4 report;
the manifest checker passes candidate-report provenance, native/Python parity,
production MPC constants, and odometry mapping. Canonical replay now resolves
the manifest only when `--use-manifest` is explicit; otherwise it evaluates
the exact reports supplied for an offline A/B comparison. The production
six-state BachelorProject MPC remains unchanged and is still the only active
runtime profile. The v3 native replay remains at
`model_fits_v2/native_replay_report_driven_measured_damping_20260915.json` as
historical offline evidence.

Finally, the recurring false fatal message at normal finite-run shutdown was
fixed in `sdu_apex_autodrive/bridge_40hz.py`: a buffered Socket.IO response
after the request FIFO is cleared is ignored only after shutdown has begun;
real source gaps still fail closed. The live repeat completed without that
false timing fault, and the focused Humble tests pass `10` tests. The next
model step is a new high-speed powered transient/release experiment designed
to distinguish wheel-state lag from speed-dependent drive force. In parallel,
fresh live track `/odom` versus ground-truth scoring should be used to tune
the observer; no MPC migration should occur until both the candidate model and
deployed odometry pass blind acceptance.

## 2026-09-15 clean-data gate and standardized candidate v4

The high-speed acceptance rule now includes an offline open-plane validity
check in `tools/model_id/assemble_transitions.py`. A capture is rejected when
the ground-truth vertical position deviates more than `1.0 m` from its initial
plane, even if its source packets remain perfectly regular at 40 Hz. This
identified two previously retained but invalid captures: the 18 m/s and 20
m/s lateral runs left the plane with maximum vertical deviations of `11.3 m`
and `27.0 m`; both are preserved under `failed_20260915/` and are excluded
from all current fitting.

The clean powered transient/release capture is
`accepted_20260915/high_speed_powered_transient_080_070_060_release_nolidar_40hz_20260915/`:
`1,061` packets, `1,060` transitions, median source rate `40.0006 Hz`, no
source gaps, no reset/teleport boundary, peak speed `18.66 m/s`, and vertical
deviation `0.006 m`. It was run with the normal-graphics Unity player in
`-batchmode`; `-no-graphics` was not used and simulator physics was not edited.

The clean refit is
`model_fits_v2/longitudinal_model_benchmark_measured_damping_clean_holdout_brake20_lateral16_20260915.json`.
It uses `7,880` training transitions and `2,180` clean validation transitions,
with the 20 m/s braking and 16 m/s lateral runs held out. The selected
continuous-wheel longitudinal candidate scores `4.03%` relative p95 at `0.50
s` and `6.45%` at `2.00 s` in its causal longitudinal score. The corresponding
native full-plant replay is
`model_fits_v2/native_replay_report_driven_clean_holdout_brake20_lateral16_20260915.json`;
at `0.50 s` it scores position p95 `0.223 m`, body-forward-speed p95 `0.260
m/s`, lateral-speed p95 `0.983 m/s`, and yaw-rate p95 `0.141 rad/s`. At `2.00
s`, the full-plant scores are position `4.633 m`, `u` `2.911 m/s`, `v` `6.318
m/s`, and `r` `1.128 rad/s`. These are offline candidate results, not a 2%
runtime acceptance.

The canonical manifest is now `2026-09-15.v4` and points to this clean report;
native C defaults and the Python reference use the same resolved parameters,
with the manifest checker passing. The active runtime remains the unchanged
BachelorProject six-state MPC. The offline candidate remains explicitly
`not_promoted` because the full-plant lateral/yaw error is still the dominant
limitation and clean 18/20 m/s lateral holdouts must be recollected before
claiming high-speed blind validation.

## 2026-09-15 lateral/yaw holdouts and live odometry validation

Clean lateral holdouts were recollected at the high-speed 18/20 m/s operating
points:

- `accepted_20260915/high_speed_lateral_18mps_clean_nolidar_40hz_20260915/`
- `accepted_20260915/high_speed_lateral_20mps_clean_nolidar_40hz_20260915/`

Each contains `861` packets and `860` transitions, has a `40.0006 Hz` median
source rate, no source cadence/sequence gaps, no reset boundary, no collision,
and less than `0.007 m` vertical deviation. The explicit speed/combined-slip
candidate was tested on both blind holdouts and did not improve the canonical
Y2 tanh model: the combined-slip gain converged to approximately zero, and its
recursive position p95 was worse at both `0.50 s` and `2.00 s`. It remains an
offline experiment only.

An independent normal-range repeat is at
`run_live/lateral_speed_bands_2_12_repeat_20260915/`. It passed with `1,941`
packets, `40.0000 Hz` median source rate, `21.999--28.000 ms` source intervals,
zero source gaps/cadence violations, no reset boundary, and no collision. A
blind Y2/Y3 refit using the original 2--12 m/s run as training and this repeat
as validation selected Y2 again. On the repeat, Y2 scored one-step p95
`0.0224 m/s` lateral velocity and `0.0827 rad/s` yaw rate; recursive full-plant
position p95 was `0.306 m` at `0.50 s` and `6.401 m` at `2.00 s`. Y3 was not a
robust improvement (`0.301 m` and `6.533 m` at those horizons). The remaining
long-horizon error is therefore not evidence for promoting the current
speed/combined-slip extension; it points to state-memory/actuator/longitudinal
coupling that needs a separate model revision.

The live track validation was run with the rebuilt competition player in
`-batchmode` using normal graphics; `-no-graphics` was not used and no Unity
physics, scene geometry, or vehicle behavior was edited. The accepted run is
`run_live/live_odom_ekf_amcl_pp_12mps_lateral_candidate_20260915_valid/`.
It completed multiple laps without collision. The 40 Hz bridge recorded `8,633`
packets with no sequence loss, one-command applied lag, and no mid-run timing
fault; the only timing-fault row is the expected final socket disconnect during
shutdown. The sub-100 MB validation bag is retained under its `rosbags/`
directory.

The new causal runtime odometry lateral model uses only mapped wheel speed and
IMU yaw rate:
`v = yaw_rate * (0.167 - 0.0063 * forward_speed)`, clamped to `+-0.35 m/s`.
Against the recorded simulator packet truth offline, its direct body-lateral
velocity p95 absolute error was `0.056 m/s`, versus `0.337 m/s` for the prior
observer. In the matched `174.569 s` live comparison, raw odometry position p95
fell from `2.266 m` to `1.370 m`; AMCL position p95 was `0.129 m`, and raw
odometry yaw p95 was `0.0095 rad`. The C++ observer, YAML, Python reference,
manifest mapping, and tests use the same lateral parameters; Humble build and
observer tests pass.

The lateral/yaw plant candidate has not been migrated into MPC. The next model
revision must isolate and reduce the remaining recursive yaw/lateral drift
with a causal state/actuator treatment, followed by another independent blind
holdout and fresh live odometry/EKF/AMCL validation.

## 2026-09-15 steering-transition identification revision

The transition schema was audited before adding another lateral correction.
At a commanded reversal the source packet reports the applied physical
steering angle directly: for example, `-0.1047 rad` becomes `0.0 rad` in the
next 25 ms packet. The previous offline plant nevertheless imposed a
`3.2 rad/s` ramp, which created artificial yaw residuals at exactly those
reversals. This was an identification-model mismatch, not a Unity-physics
change.

`tools/model_id/structured_vehicle_plant.py` now makes the choice explicit
with `steering_dynamics_kind`: the existing rate-limited behavior remains the
default, while the data-supported instantaneous transition is available only
to offline candidate fits. The fitter and residual analyzer record and honor
that choice; focused causality tests pass `11/11`.

The instantaneous high-speed benchmark is
`model_fits_v2/lateral_model_benchmark_instantaneous_highspeed_holdout_20260915.json`.
Its Y2 tanh candidate, trained on clean 14/16 m/s data and held out at clean
18/20 m/s, scores one-step p95 `0.0249 m/s` lateral velocity and `0.0404
rad/s` yaw rate. Full-plant p95 at 2 s is `2.485 m` position, `0.0475 m/s`
lateral velocity, and `0.0906 rad/s` yaw rate. The direct-steering Y3
speed/combined-slip candidate is worse on the same holdout.

The independent normal-range repeat benchmark is
`model_fits_v2/lateral_model_benchmark_instantaneous_normal_repeat_20260915.json`.
The high-speed Y2 parameter set also cross-scores that repeat at 2 s with
`2.259 m` position, `0.0191 m/s` lateral velocity, and `0.1311 rad/s` yaw
rate. A normal-range-only fit does not extrapolate to 18/20 m/s: its 2 s
lateral-velocity p95 reaches `4.953 m/s` on that holdout. The old combined
throttle/steering manoeuvre remains a coupled longitudinal regime and is not
being hidden by the lateral fit.

Free-`I_z` and free effective yaw-damping experiments were also rejected as
promotions: free `I_z` hit its lower search bound and traded high-speed yaw
against the normal repeat, while fitted damping converged to approximately
zero and did not improve the fixed measured-damping candidate. The canonical
manifest, native runtime, production MPC, and Unity simulator remain
unchanged; this steering candidate is still offline and not promoted.

## 2026-09-15 positive wheel-burst guard and fresh track validation

The source-time replay exposed a coherent positive cumulative-encoder burst at
established speed. Both the rolling and current encoder rates could agree with
each other while remaining above the causal body-speed prediction, so the old
straight-line recovery branch could incorrectly accept the burst. The C++
observer and Python reference now keep the recovery pending until the current
packet is also inside the bounded positive-increase gate. A focused regression
test covers this established-speed straight burst; the Humble localization
build and all three localization tests pass.

On the clean 18 m/s and 20 m/s corner replays, the guarded candidate reduced
the offline raw-odometry position p95 from `4.884 m` to `0.523 m` and from
`1.875 m` to `0.672 m`, respectively. Its lateral velocity p95 remained
approximately `0.100/0.112 m/s`, so this is a longitudinal/pose-burst
improvement, not a claim that lateral odometry is solved.

The fresh competition-track validation is retained at
`run_live/live_odom_ekf_amcl_pp_16mps_positive_burst_guard_20260915/`. It used
the required Unity `-batchmode` player with normal graphics; `-no-graphics` was
not used, and no Unity physics, geometry, or vehicle behavior was changed. It
recorded `8,798` source packets over approximately `220 s`, source median
`40.0000 Hz`, source interval `21--29 ms`, zero sequence gaps, one-command
applied lag, and zero collisions. Pure Pursuit completed repeated laps; the
recorded track speed reached approximately `8.44 m/s`.

The fresh source-time localization report shows AMCL p95 `0.180 m`, current
map p95 `0.110 m`, and raw `/odom`/EKF p95 `1.662 m`. The odometry yaw error is
below numerical precision, while the track-frame decomposition is dominated
by cross-track drift (`1.492 m` p95) with a smaller along-track component
(`1.274 m` p95). This separates the next work from timing and global AMCL:
audit the runtime/reference point and the causal lateral pose integration,
then validate any observer change on a new blind track run. The guarded
observer and any lateral/yaw plant candidate remain unpromoted into MPC.

## 2026-09-15 capped 14 m/s lateral/yaw holdout

The next clean identification run used the open-plane model-identification
player and the existing single-regime steering excitation at the new project
ceiling. It is retained at
`run_live/lateral_yaw_14_single_openplane_20260915/`: `1,281` packets and
`1,280` transitions, median source rate `40.0000 Hz`, source interval
`22--28 ms`, zero cadence gaps, zero reset boundaries, zero collisions, and
maximum measured speed `14.24 m/s`. The v4 timing/kinematic gate passed; the
selected body-frame displacement consistency median was `0.0091 m/s`.

The independent blind fit is
`model_fits_v2/lateral_model_benchmark_instantaneous_highspeed_14_holdout_20260915.json`.
On this 14 m/s holdout, fixed-`I_z` Y2 tanh gives one-step p95 errors of
`0.0392 m/s` lateral velocity and `0.0555 rad/s` yaw rate; causal recursive
position p95 is `0.313 m` at `0.50 s` and `5.368 m` at `2.00 s`. The Y3
speed/combined-slip extension gives `0.323 m` and `5.326 m` at those horizons,
so it is not a robust improvement. Neither candidate is runtime-validated or
promoted to MPC.

Two preliminary attempts were excluded from fitting: a competition-track
source replay without a path follower hit a barrier and is stored under
`failed_20260915/lateral_yaw_16_single_competition_barrier_20260915/`; the
open-plane 16 m/s attempt terminated at the speed guard after a measured
`16.0668 m/s` overshoot and is stored under
`failed_20260915/lateral_yaw_16_single_openplane_speed_guard_20260915/`. No
simulator physics or behavior was changed by either attempt.

## 2026-09-15 recursive-position regression and model-kind correction

The `5.368 m` two-second position result was traced to a model-selection
regression in the offline lateral fitter. The longitudinal report contains
both a discrete wheel-state model and a continuous wheel-derivative model.
The fitter had selected `wheel_dynamic` while the structured plant integrated
the coefficients as a derivative. At the measured 14 m/s operating point this
made the predicted wheel state grow by approximately `15 m/s` per second. The
resulting wheel p95 error reached `32.3 m/s` at two seconds and contaminated
the longitudinal force, yaw, and pose scores.

`tools/model_id/structured_vehicle_plant.py` now rejects unknown wheel-model
kind strings and maps the discrete and continuous identified kinds explicitly.
`tools/model_id/fit_lateral_model.py` now defaults to the canonical
`wheel_continuous` report entry and records the selected source-model key.
The regression suite covers both mappings and rejects an ambiguous kind;
`sdu_apex_autodrive/test/test_model_id_recursive_causality.py` passes `10/10`.

The corrected 14 m/s replay is
`model_fits_v2/lateral_model_benchmark_instantaneous_highspeed_14_holdout_corrected_wheel_20260915.json`.
It removes the wheel blow-up and reduces Y2 two-second position p95 from
`5.368 m` to `3.708 m`; wheel p95 falls from `32.3 m/s` to `0.023 m/s`.
The remaining error is not primarily lateral-velocity magnitude: a causal
decomposition gives two-second position p95 `3.706 m` with the full plant,
`3.671 m` with measured longitudinal state, and `0.577 m` with measured
heading. The remaining dominant error is accumulated yaw/heading, with a
smaller longitudinal speed bias (`0.504 m/s` p95).

The first recursive raceline objective was added to the lateral fitter. It
uses contiguous, reachable windows only: project speed `1--16 m/s`, steering
within `+-0.45 rad`, and measured lateral acceleration within `14 m/s^2`.
The fit uses measured longitudinal state only while identifying lateral
dynamics offline; complete recursive scoring remains causal. A track-only
fit improved its own held-out track score to approximately `0.84 m` position
p95 at two seconds but extrapolated to approximately `22 m` on the 14 m/s
holdout, so it was rejected.

A mixed track plus capped-14 m/s speed-dependent fit was also rejected after
per-run cross-checking: the Y3 candidate was approximately `0.83--0.85 m`
on the 8 m/s track holdouts but approximately `5.0 m` on the 14 m/s holdout,
worse than the corrected baseline. Likewise, a combined longitudinal refit
reduced track speed error to approximately `0.17 m/s` p95 but increased the
14 m/s speed error to approximately `1.68 m/s` p95. These results show why a
single regime-specific model must not be promoted or wired into MPC.

The completed candidate reports are retained for audit:

- `model_fits_v2/lateral_model_raceline_recursive_track_holdout_20260915.json`
- `model_fits_v2/lateral_model_speed_dependent_recursive_raceline_20260915.json`
- `model_fits_v2/lateral_model_recursive_highspeed14_repeat_20260915.json`
- `model_fits_v2/longitudinal_model_benchmark_raceline_and_14_holdout_20260915.json`

The corrected instantaneous-steering residual CSV was regenerated after the
fit so its p95 values match the JSON exactly (`0.03922 m/s` for lateral
velocity and `0.05554 rad/s` for yaw rate).

None is promoted. The next model revision must fit the speed-dependent
longitudinal slip/force relation and the causal yaw response jointly, with
balanced per-regime weighting and separate track/14 m/s holdouts. Unity,
production MPC, odometry, EKF, and AMCL were not modified in this analysis.

## 2026-09-15 equal-horizon regime-model implementation

The next offline-only model implementation is
`tools/model_id/fit_speed_regime_vehicle_model.py`. It adds a causal
effective-steering state with a fitted first-order lag diagnostic, smooth
low/mid/high speed blending at 2--8, 8--12, and 12--16 m/s, combined-slip
lateral-force scaling using wheel/body longitudinal slip, and speed-dependent
drive-force/slip-gain profiles. Unity's measured `0.273 1/s` linear damping
remains the only longitudinal drag term; the fitted residual-drag term is
zero. Rows above the project `16 m/s` ceiling, full-lock commands, and
excessive lateral acceleration are excluded from identification.

The retained candidate report is
`model_fits_v2/speed_regime_vehicle_candidate_recursive_instantaneous_20260915.json`.
It is explicitly not runtime validated or promoted. The steering-lag fit is
approximately `0.0475 s`, but the selected replay uses the measured effective
steering state instantaneously because the retained command/feedback data do
not independently establish that lag. The combined-slip gain is driven to
approximately zero in the fitted regimes, so it is not being forced into the
model without evidence.

On the actual track holdouts at the MPC-relevant `0.60 s`, the new profile
reduced position p95 from approximately `0.213 m` to `0.089 m` and longitudinal
speed p95 from `0.496 m/s` to `0.191 m/s`. Heading p95 remained approximately
`0.071 rad`, so this is a useful raceline-speed improvement but not a yaw
solution. On the capped 14 m/s turn, lateral-velocity p95 improved to roughly
`0.018 m/s`, while position/heading remained approximately `0.514 m`/`0.125
rad` at `0.60 s`; the high-speed lateral candidate is therefore still not
accepted.

The equal-physical-horizon benchmark is
`model_fits_v2/mpc_equal_horizon_discretization_candidate_20260915.json`.
For the combined track and 14 m/s sample at `0.60 s`, the coarse
`N=20, dt=0.030, internal=0.030` grid produced approximately `0.430 m`
position p95 and `0.313 rad` heading p95. Matched internal half-steps reduced
this to `0.359 m` and `0.097 rad` for `N=20`, and `0.357 m` and `0.094 rad`
for `N=24, dt=0.025`. `N=40, dt=0.015` gave `0.359 m` and `0.092 rad`, so
doubling the stage count adds little after the integration is resolved.
No simulator or production MPC constants were changed. The next fit must
focus on the remaining causal yaw response and be validated on an independent
12--16 m/s turning holdout before any controller migration.

## 2026-09-15 regime-separated lateral screening and MPC horizon gate

The offline plant now supports explicit effective steering, smooth regime
profiles, and separate lateral training pools through
`tools/model_id/fit_speed_regime_vehicle_model.py`. Straight samples are no
longer allowed to dominate the lateral fit: a lateral sample must contain
nontrivial steering or lateral acceleration, while straight powered samples
remain available to the longitudinal fit. This keeps the requested
2--8/8--12/12--16 m/s bands tied to turning behaviour.

The first regime-separated screening artifact is
`model_fits_v2/speed_regime_vehicle_candidate_regime_screen_20260915.json`.
It used an 8-origin screening stride and is not an acceptance result. On the
actual track holdouts its `.60 s` position p95 was approximately `0.10 m`,
versus `0.21 m` for the canonical scalar baseline. On the capped 14 m/s
temporal turn holdout it was approximately `0.34 m`/`0.08 rad` for
position/heading, versus `0.29 m`/`0.10 rad` for the scalar baseline. On the
2--12 m/s repeat it was worse (`0.49 m`/`0.29 rad` versus `0.25 m`/`0.09
rad`), so the candidate is rejected pending a better mid-speed yaw model.

The earlier equal-horizon benchmark remains as a historical
`model_fits_v2/mpc_equal_horizon_discretization_candidate_20260915.json`.
Its result is unchanged: matched internal half-steps produce nearly the same
`.60 s` error for `N=20, dt=.03`, `N=24, dt=.025`, and `N=40, dt=.015`; the
coarse single-step integrator is the bad case. It is superseded as an active
gate by the `.75 s`/30-command result recorded below. No Unity, production
MPC, odometry, EKF, or AMCL behavior was changed. Focused model tests remain
green (`24 passed`).

## 2026-09-15 active 40 Hz horizon standardization and full holdout score

The active prediction contract is now standardized everywhere the model is
fit, replayed, scored, or compiled into the MPC: `30` commands at `40 Hz`,
with `0.025 s` per stage and a physical horizon of `0.75 s`. The production
MPC defaults in `f1tenth_mpc/include/mpc_types.h` and the canonical manifest
now use this contract, and the manifest checker rejects timing drift. The
legacy one- and two-second prediction targets were removed from executable
fitters, repeatability reports, replay scoring, and tests. Historical JSON
reports that contain those old metrics remain unchanged as audit evidence and
are not re-run.

The full stride-1 offline candidate score is
`model_fits_v2/speed_regime_vehicle_candidate_canonical_lateral_steering_rate_075_20260915.json`.
It uses the canonical lateral law, the fitted longitudinal speed profile, and
a causal steering-transition residual. The simulator was not modified and no
candidate parameters were copied into runtime MPC. On `8,364` valid holdout
origins at the active `.75 s` horizon, the candidate gives position p95
`0.268 m`, heading p95 `0.106 rad`, longitudinal-speed p95 `0.151 m/s`,
lateral-speed p95 `0.141 m/s`, and yaw-rate p95 `0.182 rad/s`. The scalar
baseline gives position p95 `0.306 m`, heading `0.099 rad`, longitudinal speed
`0.506 m/s`, lateral speed `0.076 m/s`, and yaw rate `0.192 rad/s`; therefore
the candidate is a useful longitudinal/position improvement but is not yet a
uniform lateral/yaw improvement or a 2% model. It remains rejected for
runtime migration until an independent blind turning holdout and native
parity pass.

The equal-horizon discretization result is
`model_fits_v2/mpc_equal_horizon_discretization_candidate_075_20260915.json`.
The requested `N=30, dt=.025 s` grid gives position p95 `0.323 m`; the
`N=50, dt=.015 s` resolution diagnostic gives `0.289 m`, while the coarse
`N=25, dt=.030 s` single-step case gives `0.339 m`. The matched half-step
`N=30` diagnostic gives `0.285 m`. This indicates that the 40 Hz grid is
adequate and that integration resolution, not a longer horizon, is the
important numerical factor.

The latest clean Pure Pursuit track run remains
`run_live/live_odom_ekf_amcl_pp_16mps_positive_burst_guard_20260915/`:
`8,798` source packets over about `808 m`, median source interval `25 ms`,
maximum interval `29 ms`, zero sequence gaps, zero collisions, and maximum
measured speed `8.44 m/s`. Current-map localization is approximately
`0.110 m` p95, AMCL `0.180 m` p95, while raw `/odom` and `/ekf_odom` position
are approximately `1.66 m` p95 on the path-relative localization score.
The run demonstrates repeatable collision-free driving and clean timing, but
does not demonstrate 16 m/s operation or MPC acceptance. Odom velocity is
approximately `0.152 m/s` p95 longitudinal and `0.311 m/s` p95 at the
GPS/pose point laterally (`0.063 m/s` p95 against the COM diagnostic).

## 2026-09-16 direct Unity WheelCollider model screen

The offline plant now includes a separate direct Unity lateral-law path using
the F1TENTH WheelCollider sideways curve, four-wheel geometry, and the
VehicleController Ackermann equations. The first screen exposed and fixed a
left/right Ackermann sign reversal. The corrected full holdout artifact is
`model_fits_v2/direct_unity_wheel_collider_benchmark_075_ackermann_fix_20260916.json`;
its 0.75 s CORE p95 is `0.0431 m` cross-track, `0.0528 rad` heading,
`0.0100 m/s` lateral velocity, and `0.1098 rad/s` yaw rate. It remains
offline-only and is not an MPC candidate.

The companion
`model_fits_v2/unity_wheel_contact_trace_audit_20260916.json` finds that
state-reconstructed `sidewaysSlip` agrees with recorded Unity wheel slip,
while per-wheel contact loads vary materially from static sprung-mass loads.
An instantaneous load-transfer screen was unstable in recursive replay and
is rejected. The next model-identification action is a fresh exact
competition-scene wheel-contact trace with a causal load/suspension state;
no tire peak is to be inflated to hide this mechanism.

## 2026-09-16 exact Unity value contract and collision mechanism

The required virtual-car values are consolidated in
`tools/model_id/UNITY_MODEL_REQUIRED_VALUES_20260916.md`. This contract keeps
serialized Unity values, runtime values, and diagnostic-only effective values
separate. Important exact inputs include mass `3.47 kg`, serialized COM
`(-0.00008,0.06434,-0.00468) m`, WheelCollider radius `0.059 m`, wheel mass
`0.109 kg`, suspension `500/100/0.5/0.05`, the serialized Unity friction
curves, wheelbase/track `0.324/0.236 m`, active scene controller
`CAWD/CAWB/FrontWheelSteer`, scene-overridden motor torque `428 Nm`, and the
project physics/timing values `fixedDeltaTime=.001 s`, gravity `-9.81 m/s^2`,
solver `6/1`, contact offset `.01 m`, and maximum angular speed `7 rad/s`.
The operational speed ceiling remains `16 m/s`.

The disposable exact open-plane runtime-state capture is
`model_fits_v2/unity_exact_open_combined_slip_runtime_state_v2_20260916/`.
It records 70,028 fixed-step rows with per-step inertia and sprung mass plus
290,569 collision callbacks. Runtime body-y inertia ranges
`0.09557827--0.09619075 kg m^2`; this must not be replaced by a single static
`I_z`. The unchanged compound vehicle colliders also produce `10,231` stable
`Chassis-1-solid1`--`Floor` contact callbacks. The full contact audit found
positive separation and zero solver impulse for all chassis callbacks, so the
lower residual obtained after excluding those rows is a data-regime effect,
not proof of a chassis-force channel or a reason to tune a tire peak. No
simulator physics or production MPC behavior was changed.

### 2026-09-16 WheelCollider force API boundary audit

The disposable open-plane `combined_slip_matrix_v1` was repeated with only
read-only `Rigidbody.GetAccumulatedForce(Time.fixedDeltaTime)` and
`GetAccumulatedTorque(Time.fixedDeltaTime)` fields added to the diagnostic
trace. The 70,028-row batchmode run completed with normal graphics and no
`-nographics`; all six accumulator components were exactly zero on every
row. This is an API boundary result: the active `VehicleController` does not
call `Rigidbody.AddForce`, while the WheelCollider contact solver applies its
forces internally during the physics step. The result is stored in
`model_fits_v2/unity_exact_open_combined_slip_accumulated_force_v4_20260916/`
and is not used as a model input.

The required force quantity is therefore the causal internal WheelCollider
response, reconstructed only from the already available body acceleration,
WheelHit slip/contact state, wheel RPM, motor/brake torque, steering angle,
runtime sprung mass, and runtime inertia. The front/rear grouped gains from
the previous screen vary with speed and are retained only as an observability
diagnostic; they are not promoted as tire parameters or used to hide a missing
solver mechanism. No Unity physics, vehicle behavior, or production MPC was
changed.

The next diagnostic is now reproducible as
`tools/model_id/identify_unity_axle_force_response.py`. It removes forward
force using the explicit previously identified wheel rotational state, then
solves body lateral force and yaw moment for front/rear axle forces. Two
independent captures of the same unchanged schedule gave front/rear proxy
gains `0.6489 / 0.6869` with chronological holdout RMSE `0.179 / 0.195 N`,
repeating to below `1e-4`. The reports are
`model_fits_v2/unity_exact_open_axle_force_response_v1_20260916.json` and
`model_fits_v2/unity_exact_open_axle_force_response_v1_repeat_20260916.json`.
These are explicit Unity-solver observability results only; no friction,
cornering-stiffness, or MPC parameter was created or promoted.

A fresh `raceline_relevant_holdout_v1` was then run in the disposable player.
It reached `15.36 m/s` with applied steering limited to `+-3.6 deg`; the
selected turning rows covered `5.94--15.31 m/s` after lowering the offline
minimum-steering screen to `0.005 rad`, including high-speed small-steering
raceline conditions. Its axle inversion gave front/rear proxy gains
`0.7932/0.6781` and chronological holdout RMSE `0.234/0.183 N`, unlike the
previous `0.6489/0.6869` pair. This falsifies a
universal front/rear gain and confirms that the response must be split by an
observable Unity state or regime. The trace is stored in two sub-100 MB CSV
parts under
`model_fits_v2/unity_exact_open_raceline_relevant_holdout_v1_20260916/`;
the report is
`model_fits_v2/unity_exact_open_raceline_axle_force_response_v1_20260916.json`.

### 2026-09-16 cross-schedule mechanism screen

The axle inversion now exports compact selected-row tables through the
optional `--rows-output` argument. The new
`tools/model_id/screen_unity_axle_force_cross_schedule.py` fits only explicit
Unity-trace quantities on one complete experiment and validates on the other
complete experiment in both directions. The tested bases are the direct
curve/contact-load proxy, proxy plus speed, proxy plus measured slip, proxy
plus serialized curve demand, and an explicit load-power hypothesis.

The independent old combined-slip and raceline-relevant schedules do not
support one universal gain or any of those added terms: the best cross-
schedule test RMSE remains approximately `1.01--1.08 N` for the front axle
and `0.17--0.27 N` for the rear axle for the simple candidates, while the
slip and load-power terms extrapolate substantially worse. The report is
`model_fits_v2/unity_exact_axle_force_cross_schedule_screen_v1_20260916.json`;
the compact row tables have matching
`cross_schedule_*_rows_v1_20260916.csv` names. No term is promoted.

A second disposable, normal-graphics batchmode speed-aware sweep was also
completed for the raceline-relevant envelope. It recorded `90,039` fixed
steps over `90.038 s` with no timing gaps, a maximum speed of `15.22 m/s`,
and maximum applied steering of `6 deg`; the project ceiling was not
exceeded. The lower-speed steering plateaus produced normalized sideways
slips up to `0.998`, and the recovered axle forces showed large transient/
schedule dependence. That result is retained as a rejection artifact, not a
vehicle parameter fit. Its trace is split into three sub-100 MB CSV parts
under `model_fits_v2/unity_exact_open_raceline_relevant_speed_sweep_v1_20260916_clean/`;
the force-response report is
`model_fits_v2/unity_exact_open_raceline_axle_force_response_speed_sweep_v1_20260916.json`.
The incomplete first attempt is outside the repository under `/tmp` and is
not part of the accepted dataset.

The current required-value conclusion is unchanged: serialized Unity values
and per-step runtime states are required inputs, while the internal
WheelCollider tangent-force response is still the unidentified mechanism.
The next useful run must use a lower-acceleration, non-spinning raceline
holdout and must be accepted only if the same explicit mapping predicts both
the existing small-steering holdout and the new holdout without a schedule-
specific gain.

### 2026-09-16 exact-value use and mechanical load-transfer screen

The captured values now have an explicit usage boundary in
`tools/model_id/UNITY_MODEL_REQUIRED_VALUES_20260916.md`: source/scene values
are direct plant inputs; per-step Unity observations identify missing causal
states; simulator truth is an offline scoring reference for odometry and is
not a legal runtime `/odom`/EKF/AMCL input. This prevents a diagnostic value
from silently becoming an odometry correction or a fitted tire parameter.

A separate physics-derived load-transfer screen was benchmarked using only
Unity mass `3.47 kg`, COM height `0.06434 m`, measured contact wheelbase
`0.33000004 m`, and track width `0.236 m`. It conserved total supported load
and introduced no fitted coefficient. On the identical blind direct benchmark
it improved 0.75 s CORE p95 cross-track from `0.043080` to `0.042580 m`,
heading from `0.052810` to `0.051882 rad`, and yaw rate from `0.109821` to
`0.108606 rad/s`; longitudinal p95 was unchanged. The improvement is small
and does not prove that suspension/contact state has been identified, so the
screen remains diagnostic and static measured sprung masses remain canonical.

The data can therefore be used immediately for three concrete purposes:
exact offline plant replay, causal wheel/suspension mechanism identification,
and offline calibration/scoring of the sensor-only odometry observer. The
remaining required model values are the four-wheel rotational transitions,
causal suspension/contact transitions, and the WheelCollider local-slip to
wrench mapping. Those must be validated on a non-spinning raceline holdout
before any model or odometry candidate is promoted.

The v2 compact row exports now preserve per-wheel angular speed and
derivative, actual wheel steering angle, sprung mass, and `GetWorldPose`
vertical position. The explicit runtime-state cross-schedule basis is recorded
in `model_fits_v2/unity_exact_axle_force_cross_schedule_screen_v2_20260916.json`.
It failed to generalize—the front old-high-steering-to-raceline direction
became numerically poor, and the reverse/front and rear tests did not beat the
stable simpler bases. This does not make the fields useless; it shows that
they must enter through a causal wheel/suspension transition model, with
deflection relative to a settled reference, rather than as another fitted
force gain. No model or odometry value was promoted.
