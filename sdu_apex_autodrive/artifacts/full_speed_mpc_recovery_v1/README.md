# Full-speed MPC recovery v1

Status: **in progress; no plant or MPC candidate is accepted yet.**

This artifact folder follows `SDU_Apex_Full_Speed_MPC_Recovery_Handoff_2026-09-14.md`.
It is intentionally separate from the older simulator-native fits.

## Completed in this phase

- Audited the Unity mass semantics: Rigidbody/body mass is `3.47 kg`; the four
  WheelCollider masses are `0.109 kg` each and are not silently added to the
  body mass.
- Audited wheel-radius semantics: the physical WheelCollider radius is
  `0.059 m`; `VehicleController.WheelRadius = 0.0325 m` is used by the
  skid-steer `ExtendedDifferentialDrive` branch, not the F1TENTH `CAWD` car
  branch.
- Built and ran a disposable diagnostic player in Unity batchmode with normal
  graphics enabled. `-nographics` was not used.
- Collected a straight drive-excitation trace at approximately 1 kHz:
  `55,849` rows, `110` columns, `0--55.848 s`, continuous 1 ms source timing,
  and about `42 MB`.
- Separated the observed zero-throttle brake event from the continuous powered
  wheel-state fit. The trace applies `428 N m` brake torque to every wheel and
  reaches zero wheel RPM in about `1 ms`; this is not passive-coast data.
- Corrected the diagnostic inertia-axis semantics in E0. Unity body Y is the
  physical vertical/yaw axis used by the dynamic trace; the fresh diagnostic
  dump reports `I_body_y = 0.09619075 kg m^2`. The former `0.0277 kg m^2`
  local-Z projection is retained only as legacy evidence and is not accepted
  by the offline parser or as an MPC constant.
- Recovered the 40 Hz request/response cadence without changing simulator
  physics: the bridge now uses an explicitly bounded two-entry FIFO because
  the competition-track Socket.IO round trip is about `50 ms`. The source
  sequence and simulation-time contract remains strict; no sample is repeated
  or interpolated.

## E1 full-speed Pure Pursuit acceptance

The accepted batchmode-with-graphics run is
`runtime_track/e1_pp_full_speed_accept_20260914_1145/`. It ran the recovered
competition track with the native Pure Pursuit controller, AMCL, odometry, EKF,
and diagnostics-only ground truth enabled:

- `6,531` source packets over `163.250 s` at `40.00024 Hz`;
- source intervals `21.995--28.000 ms`, with `0` gaps over `30 ms`;
- `0` request-sequence gaps, `0` duplicate/reverse packets, and command lag
  exactly one packet;
- `11` complete raceline crossings, with ground-truth raceline distance p95
  `0.0862 m` and maximum `0.1655 m`;
- diagnostics-only collision counter stayed at `0` throughout the run.

The earlier `e1_pp_full_speed_clean_20260914_1140` run reached `9` complete
crossings with the same zero-gap timing result. Both runs were stopped by the
operator after the campaign; the resulting bridge disconnect fault is an
expected end-of-run shutdown marker, not a simulator timing failure.

## Not accepted yet

The powered four-wheel fit is an effective offline basis only. Its contact-term
coefficient is not identifiable from this straight sweep (the design is badly
conditioned and the nonnegative fit drives that term to zero). It must not be
used by runtime MPC. The trace also contains no lateral/combined-slip
identification.

## Artifact rules

- Simulator ground truth is offline-only.
- The diagnostic component is inert unless its environment variable is set and
  does not feed commands to ROS or the runtime controller.
- No Unity physics, vehicle controller, scene geometry, or production MPC
  files were changed in this phase. The Unity-side changes used for these
  runs are limited to batchmode socket activation, telemetry diagnostics, and
  execution pacing; `-nographics` was not used.
- The source-timestamp contract is used for derivatives and replay; callback
  time is not used as a dynamics timestamp.

## Latest follow-on evidence (2026-09-14)

- Fresh E1 replay:
  `runtime_track/e1_pp_full_speed_yaw_20260914_1220/`. It contains `6,703`
  source packets, zero sequence gaps, median source interval `25.0009 ms`,
  p95 `26.001 ms`, maximum `28 ms`, and constant one-packet command lag.
  The car completed repeated raceline circuits without a collision report.
  The corrected offline localization score remains **not accepted**:
  `/current_map_pose` position p95 `0.220 m`; `/odom` position p95
  `1.124 m`; `/odom` longitudinal-speed p95 `0.537 m/s`; lateral-speed p95
  `0.302 m/s`. Yaw p95 is effectively zero after the local-odom frame is
  converted by the initial map yaw.
- Fresh E3 diagnostic-only combined-slip capture:
  `diagnostics_20260914_v2/combined_slip_matrix_v1_20260914/`. It contains
  `86,410` one-millisecond rows with no fixed-step gaps and the previously
  sparse `0.10 <= |Sx| < 0.25` band populated across lateral-slip bins.
  Contact-load analysis measures left/right load difference versus lateral
  acceleration correlation `0.9995`; this is a load-transfer observation,
  not a promoted tire-force law.
- The continuous-drive fit and the causal load-transfer fit remain
  **offline-only**. Their reports deliberately retain the `not_accepted`
  status because the available trace does not expose separate tire-frame
  force components and the recursive plant gates have not passed.
- E0 corrected the physical inertia metadata without changing native MPC
  behavior: Rigidbody mass `3.47 kg`, body-frame inertia projections
  `I_x=0.09352724`, `I_y=0.09619075`, `I_z=0.02769764 kg m^2`, with
  `yawAxis=body_y`. The native plant/MPC constants remain provisional until
  the later model gates pass. Contact wheelbase is `0.33000004 m`, steering
  geometry wheelbase `0.324 m`, and contact radius `0.059 m`; controller
  radius `0.0325 m` remains skid-steer-only.
- Re-running the existing combined-slip force/moment screen with the corrected
  body-Y inertia leaves the lateral effective scale near unity (`0.973`) but
  the longitudinal scale near `0.085`; this confirms that actuator/wheel-force
  identification is still the dominant modeling task. The compact result is
  `../model_id_work/diagnostics_20260914_v2/body_force_moment_inversion_axis_y_e0_20260914.json`.
- Regime cross-score:
  `../model_id_work/model_fits_v2/regime_cross_score_v1.json`. The same
  split-contact structure was fit separately to low-demand, straight,
  corner-only, and combined corner/throttle runs, then replayed across all
  four domains. The diagonal fits still reach approximately `0.31--0.45 m`
  position p95 at `0.50 s`, while off-diagonal fits can exceed `1 m` quickly.
  This supports gain scheduling or an additional causal state, but does not
  justify hard switching or runtime promotion yet.
- AMCL tangent-gain A/B was **invalid**, not accepted: the isolated override
  `f1tenth_localization/config/experiments/amcl_full_speed_along_track_ab.yaml`
  raised the fast tangent gain to `1.0`, but the fresh run stopped after one
  collision followed by the strict bridge's `41 ms` source-cadence fault.
  The captured packets before the fault were otherwise ordered and near
  `40 Hz`; this run must not be compared as an AMCL accuracy result.

## Next handoff actions

1. Run the E1 near-zero throttle/brake map at several initial speeds and
   determine whether a coast-equivalent command exists.
2. Collect repeated powered-drive sequences for the hybrid continuous wheel
   model and score arbitrary integration steps.
3. Re-run body force/moment inversion with corrected body-Y inertia, then
   decompose full-speed localization error into cross-track and along-track
   components before observer/MPC work.

The `plant/`, `observer/`, `mpc/`, `runtime_track/`, `contact_model/`,
`load_model/`, `regimes/`, and `blind/` directories are reserved for those
later, separately validated artifacts.
