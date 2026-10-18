# v2.3.15 — PhysX native tire FORCE basis conversion

**Observation.** User-provided v2.3.14 report showed initial max strict
vehicle-model defect **8.761113** (v2.3.12: **1.428774**) after correcting
native lateral-slip sign. Native slip parity, however, improved to 0.000547 rad
RMS. This isolated the omitted inverse lateral-force coordinate transform.

**Physical fix, generated wheel source only.** PhysX Unity callback returns
longitudinal force along the steered forward axis and lateral force along the
steered right-positive axis. The original CasADi body transform treated its
lateral callback force as left-positive. Apply right-to-left conversion in
`unity_wheel_model.tire_forces_casadi`:

- `fx_body = fx*cos(angle) + fy*sin(angle)`
- `fy_body = fx*sin(angle) - fy*cos(angle)`

Raw `fy_wheels` remains native right-positive for callback parity, and all
steering geometry, longitudinal slip, lateral slip, wheel torque, masses and
stiffness coefficients are unchanged. The original source is not modified.
Known v2.3.14/v2.3.12 cached working wheel-model revisions upgrade byte-
exactly; unknown local edits fail closed.

**Native-data cross-check.** From 481 timestamps (1924 wheel measurements),
rotating recorded native callback forces into the model's forward/left/up
coordinate system lowered lateral-acceleration RMS error from ~4.028 to
~0.201 m/s² when compared with 25ms finite-difference velocity and yaw.
This is a kinematic audit, not exact native force/state timing parity: the
contact callback is sampled pre-scene whereas body speeds post-scene, with
scene contact impulses omitted by the symbolic model.

**Commands.** Run `bash diagnose_force_frame_v2315.sh` first; it fails unless
the *actual generated* CasADi function rotates the raw callback forces in
the corrected direction. Then `bash diagnose_phase1_v2315.sh` (40 iterations,
210s timeout) tests full-lap OCP feasibility and refuses any trajectory with
nonzero original physical-equation residual. No successful IPOPT racing lap
or sub-4.9s native/Unity validation yet.
