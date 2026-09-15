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
