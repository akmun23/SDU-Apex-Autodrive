# Phase 12 — MPC authority probe (blocked)

This was an actual MPC-authority trial, not a shadow run. The ICRA simulator
image was started in batchmode (with Xvfb; no `-no-graphics` flag), and the
unified stack launched `controller:=mpc mpc_enabled:=true` on the accepted
51.72 m raceline. The requested speed ceiling was 4.0 m/s, with the configured
1.5 m/s initial target and one-lap speed ramp. Adaptive rho was enabled and
the ADMM iteration cap remained 100.

The controller loaded all 2,586 raceline samples, but did not drive. Its first
state handoff reported `missing_map_pose`; after localization began publishing
poses, the component repeatedly logged RTI candidate rejection and published
only zero-speed, zero-steering commands. All 490 recorded `/cmd/speed` values
were zero. Offline-only simulator telemetry shows the car remained within
about 1.2 mm of its start point and registered no collision.

The sim stream itself was not a valid sustained-40-Hz run. There were 490
sequential responses to request sequences 1–490, with no duplicate/reversed
sequence numbers; the selected quantized IMU/pose tuple changed on 273 of 490
packets while the car was stationary. The image exposes no simulator source
timestamp/frame ID, so this is not proof of 490 distinct physics frames. The
arrival-interval median/p95/max was 25.07/26.67/96.46 ms. Then the bridge
received no further simulator response for 150 ms, latched neutral, and stopped
the stream. The simulator and ROS stack were stopped. Do not call this a
driving, lap, or 40-Hz acceptance result, and do not use it as model-fit data.

An offline replay of the same recorded legal inputs, using the active
adaptive-rho/100-iteration/prefactorized profile, reported 484/484 solver
cycles accepted (p99 replay solve time 0.716 ms). That disagrees with the live
component's repeated rejection and zero commands. It is a debugging clue, not
proof that the controller works: the run artifact lacks per-cycle live solver
status. The controller now logs rejection class, residuals, regularization,
and nonlinear failure stage at a throttled rate so a later permitted live run
can resolve this discrepancy.

## Gate and next action

Status: **BLOCKED / NOT ACCEPTED**. Before another authority trial, resolve
both (1) why the ICRA simulator stopped answering after 490 responses and (2)
why the live ROS component rejected cycles that the replay accepted. Do not
start another simulator run until the bridge data stream can be kept at 40 Hz;
keep the initial authority cap at 4 m/s for the next valid trial.

The complete, sub-100-MB diagnostic capture is in
`mpc_authority_4mps_attempt2_20260918/`. Ground-truth fields in those files are
offline diagnostics only and were not consumed by localization or MPC.
