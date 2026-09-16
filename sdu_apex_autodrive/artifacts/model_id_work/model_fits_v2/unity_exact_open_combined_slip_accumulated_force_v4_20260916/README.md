# Exact Unity open-plane accumulated-force boundary audit

- Unity: `2022.3.52f1` Linux player
- Run mode: `-batchmode`, normal graphics; `-nographics` was not used
- Experiment: unchanged `combined_slip_matrix_v1`
- Duration: `70.027 s`
- Rows: `70,028` at the Unity `0.001 s` fixed step
- Simulator behavior/physics: unchanged

This disposable diagnostic build added read-only calls to
`Rigidbody.GetAccumulatedForce(Time.fixedDeltaTime)` and
`Rigidbody.GetAccumulatedTorque(Time.fixedDeltaTime)` immediately before the
physics step. All six exported components are exactly zero on every row.

This is not evidence that the car has no force. The active `VehicleController`
does not call `Rigidbody.AddForce`; the WheelCollider solver applies its wheel
forces internally during the physics simulation and those forces are not
exposed by this Rigidbody accumulator at this boundary. The result is a
confirmed API limitation, not a model parameter.

The trace remains useful for body-state inversion, WheelHit slip/load,
wheel RPM, motor/brake torque, steering, suspension pose, and runtime inertia.
No accumulator value is used in the plant or MPC.
