# Phase 4 — exact seven-state nonlinear stage

`mpc_vehicle_model_step` is the single stage function used by recursive model
prediction and by the numerical Jacobian module. It advances the accepted
steering and target-speed commands over the same modeled interval, applies the
current speed-response map and measured speed-dependent braking envelope,
updates yaw response using that stage's steering command, and propagates the
Frenet geometry with midpoint/RK2 equations. It reports saturation branches,
along-track progress, and body acceleration; an invalid Frenet denominator
fails the stage rather than being silently repaired.

This is an implemented, unit-tested nominal model, not a claim of exact Unity
WheelCollider prediction. Lateral velocity is still held constant. Live
controller integration and recursive holdout validation remain pending. No
Unity or simulator files were changed and no simulator was started.
