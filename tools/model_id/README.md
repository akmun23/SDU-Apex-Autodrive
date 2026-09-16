# Active model-identification path

The active handoff contract is **N=30 commands at 40 Hz**: `0.025 s` per
command and `0.75 s` physical horizon. The project ceiling is `16 m/s`.

Use these files for the current recovery/freeze work:

1. `build_raceline_operating_envelope.py` — production raceline plus PP
   limits, producing coupled CORE/GUARD/STRESS occupancy.
2. `fit_speed_regime_vehicle_model.py` — offline candidate fit and causal
   score; its 0.75 s `e_cross_m` result is the primary model metric.
3. `score_raceline_model.py` — generic Frenet/corridor scorer.
4. `ablate_speed_regime_vehicle_model.py` — fixed A0–A5 ablation plus the
   conditional, one-parameter A6 lateral speed-scale experiment.
5. `score_steering_semantics.py` — command/applied/feedback semantics across
   the union of train and validation runs before any steering-transition
   residual is considered.
6. `benchmark_mpc_horizon_discretization.py` — N30 stage-map integration
   comparison: 1, 2, 4 substeps and the 2 ms reference.
7. `n30_stage_map.py` and `validate_n30_stage_map.py` — canonical 25 ms
   stage map, finite/bounded/Jacobian checks, and scalar native parity.
8. `score_runtime_control_state.py` — runtime state/reference-point score.
9. `analyze_speed_command_pipeline.py` — evidence-based speed-limiter
   attribution.
10. `analyze_raceline_model_residuals.py` — CORE/GUARD one-step residual
    conditioning and leave-one-run-out mechanism diagnosis.
11. `identify_unity_axle_force_response.py` and
    `screen_unity_axle_force_cross_schedule.py` — diagnostic-only Unity
    WheelCollider wrench inversion and complete-schedule mechanism checks.

The canonical outputs are under
`sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/`, except for the
raceline envelope and live-run reports kept beside their source artifacts.
The simulator and production MPC are not modified by these tools.

The exact value contract is in
`UNITY_MODEL_REQUIRED_VALUES_20260916.md`. It is the authority for the
virtual Unity car: serialized WheelCollider/controller values, initial and
per-fixed-step runtime inertia, sprung mass, compound collision geometry,
and simulator-specific slip coordinates. Do not replace those values with
real-car tire assumptions or use a fitted gain to hide a missing collision or
runtime-state channel.

The current best offline candidate is the causally corrected,
raceline-filtered fitted-lateral profile in
`speed_regime_vehicle_candidate_delayed_steering_raceline_fitted_075_20260916.json`.
Its 0.75 s CORE p95 values are `e_cross=0.0327 m`, `heading=0.0406 rad`,
`u=0.135 m/s`, `v=0.0127 m/s`, and `r=0.0966 rad/s`; this passes the current
provisional CORE gate. It remains offline-only and is not approved for MPC
migration because blind, native-parity, and live acceptance are incomplete.

The lateral peak-force upper bound is an optimizer diagnostic, not a simulator
physics value. The active default remains 60 N per axle as a conservative
regularizer. A 120 N sweep moved the low-regime rear peak to 114.3 N with only
negligible CORE improvement and a small GUARD yaw improvement. A 240 N sweep
did not converge, and an effectively unbounded 1000 N low-regime diagnostic
drove the rear peak to 179.8 N while worsening the holdout GUARD yaw and
lateral-velocity errors. These results do not justify accepting an unbounded
or inflated tire peak; see
`model_fits_v2/tire_peak_bound_diagnostic_20260916.md`.

The residual diagnostic confirms that the remaining yaw error is concentrated
in the two long track holdouts (`86` and `103`); the isolated high-speed
experiments have negligible yaw residual. That is a reason to investigate
track-run state/actuator semantics and data quality before adding another
free tire parameter.

Older fitters (`fit_vehicle_model.py`, `fit_lateral_model.py`,
`fit_longitudinal_models.py`, and `fit_simulator_native_model.py`) and their
reports are retained as historical/audit evidence because earlier reviews
refer to them. They are not active candidate-selection paths. Do not add a
new report there; use the active files above and record the result in the
model-identification index.

There is no active 2 s model gate. Any older 2 s report is historical only.

## Direct Unity WheelCollider model screen (2026-09-16)

The offline plant now has a separate `unity_wheel_collider` path. It uses the
serialized F1TENTH sideways WheelFrictionCurve (`extremumSlip=0.01`,
`extremumValue=1`, `asymptoteSlip=0.1`, `asymptoteValue=0.5`, `stiffness=1`),
four measured wheel locations, Unity's Ackermann equations, and the exact
competition-scene runtime sprung-mass capture. `WheelCollider.sprungMass` is
calculated by Unity at runtime and is not serialized in the prefab; the
captured static values are tied to that scene/configuration. It does not use a fitted lateral peak,
cornering-stiffness value, friction coefficient, or speed-dependent tire gain.

The full stride-1 comparison is
`model_fits_v2/direct_unity_wheel_collider_benchmark_075_ackermann_fix_20260916.json`.
At 0.75 s its CORE p95 is `0.0431 m` cross-track, `0.0528 rad` heading,
`0.0100 m/s` lateral velocity, and `0.1098 rad/s` yaw rate. The two long
track holdouts dominate the error; the isolated steering/acceleration and
14--16 m/s straight/lateral runs are much tighter. This is an offline model
screen and is not approved for MPC migration.

The corrected Ackermann sign was important: the first direct screen had the
left/right wheel curvature reversed and produced approximately `0.136 m`
CORE cross-track p95. The corrected screen is therefore the relevant result.

The contact-trace audits are
`model_fits_v2/unity_exact_competition_combined_suspension_audit_20260916.json`
and `model_fits_v2/unity_exact_competition_combined_excitation_audit_20260916.json`.
The fresh exact-scene trace shows that
the body-state reconstruction of Unity `sidewaysSlip` is close, while
`WheelHit.contact_force_n` varies substantially from static sprung-mass loads.
The separate stationary capture provides the per-wheel settled `GetWorldPose`
reference. A transparent spring/damper load screen has low explanatory power
on the combined run, so it is not promoted or used to hide a tire-model error.
An algebraic load-transfer screen was tested and rejected because its
instantaneous feedback became unstable in recursive replay. A causal
suspension/load state must be identified from additional excitation and blind
holdouts before any load correction is considered.

The open-plane runtime-state capture is
`model_fits_v2/unity_exact_open_combined_slip_runtime_state_v2_20260916/`.
It records per-step body-y inertia and wheel sprung mass together with a
read-only compound-collision trace. It found `10,231` stable rows with
`Chassis-1-solid1` contact callbacks, but all had positive separation and zero
solver impulse. They do not establish a body-force channel; excluding them is
only a diagnostic regime split.

A delayed post-settling runtime snapshot was also verified against the
serialized snapshot. The F1TENTH competition prefab keeps the serialized
`500/100/.05` suspension and zero force-app-point values; the optional
repository `Suspension.cs` is not attached to this prefab.

The exact open-plane drive audit is
`model_fits_v2/unity_exact_open_drive_wheel_drive_audit_20260916.json`. The
trace reached `15.36 m/s` and retained all 111,266 fixed-step rows in three
CSV parts. The recorded forward slip is reproduced by the simulator-specific
coordinate
`(wheel_surface_speed-ground_speed)/max(abs(wheel_surface_speed),abs(ground_speed))`,
with collider radius `0.059 m` and saturation at `+-1`. On settled positive-
drive plateaus the remaining per-wheel reconstruction error is at most about
`0.0008` slip units. The remaining large mismatch immediately after
throttle/brake changes is therefore a transient wheel-rotation state, not a
reason to alter the forward-friction curve. The active CAWD controller sends
`428 Nm * throttle / 4` to each wheel and applies `428 Nm` brake torque to all
four wheels at zero throttle; these semantics are now recorded as required
model inputs.

The direct screen also records the active competition-scene controller
contract: `CAWD`, front-wheel steering, `MotorTorque=428`, and the serialized
`183.346 deg/s` steering rate. A `sidewaysSlip` scale of `1.10` was screened
against the same holdouts; it changed CORE yaw p95 from `0.1098` to `0.1049
rad/s` but worsened CORE cross-track and GUARD heading, so it is not accepted
as a tuning constant. The direct model therefore remains at scale `1.0` until
the exact-scene contact trace can distinguish slip-coordinate error from
dynamic suspension/load behavior.

The explicit forward-curve recursion is recorded in
`model_fits_v2/direct_unity_wheel_collider_benchmark_075_explicit_forward_curve_20260916.json`.
It is rejected for now: CORE cross-track p95 is `0.0486 m` and longitudinal
speed p95 is `0.214 m/s`, worse than the identified-force screen (`0.0431 m`
and `0.135 m/s`). The serialized curve and simulator slip coordinate are
therefore not the problem by themselves; the missing causal wheel/load state
must be resolved first.

### Wheel rotational-state identification (2026-09-16)

The repeated powered-drive trace is
`model_fits_v2/unity_exact_open_powered_drive_repeat_wheel_drive_audit_20260916.json`,
with its source CSV in
`model_fits_v2/unity_exact_open_powered_drive_repeat_20260916/`. It contains
51,018 internal fixed-step rows (57 MB) and reaches the project ceiling
without exceeding it. The audit uses the transparent balance

`I*d(angular_velocity)/dt + c*angular_velocity = motor_torque - tire_torque`

with the serialized forward curve and recorded `WheelHit.force`; zero-throttle
CAWB braking is kept separate as a hybrid wheel-lock event.

The result is a speed-dependent effective transition state, not one universal
inertia: approximately `0.000366 kg m^2` below `6 m/s`, `0.000441 kg m^2`
from `6--12 m/s`, and `0.000436 kg m^2` at `12--16 m/s`. The rotational
damping remains approximately `0.250 N m s` in all regimes. The per-regime
held-out torque-balance errors are about `0.11--0.14 N m`. These are
diagnostic effective coefficients for the Unity solver response, not claims
about a real-car wheel inertia and not yet runtime parameters. They must be
represented explicitly if the wheel state is added; they must not be absorbed
into a longitudinal force peak.

## Value-use boundary (2026-09-16)

The exact value contract in `UNITY_MODEL_REQUIRED_VALUES_20260916.md` classifies
the data explicitly: serialized Unity values are direct offline plant inputs;
per-step Unity observations identify missing causal states; simulator truth is
only an offline reference for scoring sensor-only odometry. It is never a
runtime `/odom`, EKF, AMCL, or controller input.

The separate `mechanical_cg_transfer` screen uses only Unity mass, COM height,
contact wheelbase, and track width. It changes the identical blind benchmark's
0.75 s CORE cross-track p95 from `0.043080` to `0.042580 m`, which is too small
to establish a correct suspension model. It remains diagnostic. The next
required model values are the four wheel rotational transitions, causal
suspension/contact transitions, and the local WheelCollider slip-to-wrench
mapping, validated on a non-spinning raceline holdout.
