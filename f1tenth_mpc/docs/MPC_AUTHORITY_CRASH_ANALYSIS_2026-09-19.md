# MPC authority crash analysis — 2026-09-19

Baseline reviewed: `a597eca504d1dd3c2233709eb7d9eed301bfae18`
(`Offline analysis ready`).

This note records the evidence used for the authority-state fix. It deliberately
separates behaviour seen while Pure Pursuit owns the car from behaviour that
exists only when MPC owns the command.

## 1. Longitudinal runaway is no longer the primary fault

The latest baseline already carries the raceline acceleration, a direct
target-speed-state reference/cost, and the inverse identified longitudinal
response. The exact raceline peaks at about 10.16 m/s and the corresponding
inverse-model target is about 10.52 m/s. The new live 3 m/s authority test also
stayed near the intended speed range, yet still collided repeatedly.

Therefore this patch does not retune the longitudinal weights or reopen the
already-fixed 16 m/s issue.

## 2. The PP-driven shadow state is normally solvable

The clean shadow trace contains 13,036 MPC diagnostic callbacks. The large
block of 1,287 `rejected_residual` rows is not a moving-car failure sequence:
after the PP run stops, progress freezes near 863.79 m while `u=0`,
target-speed and steering commands are zero, and the same low-speed/high-
curvature QP is retried repeatedly.

The adaptive offline replay of the moving trace has only the known stop-event
residual failure. That event is already covered by the existing same-cycle R2
residual-recovery path.

The stopped tail must therefore not be used to tune racing Q/R weights.

## 3. MPC and PP steering agree when MPC observes PP's command history

On 9,141 moving accepted shadow cycles:

- PP vs MPC steering correlation: **0.99934**
- median absolute steering difference: **0.00423 rad**
- p95 absolute steering difference: **0.01632 rad**
- maximum absolute difference: **0.04207 rad**
- meaningful steering-sign mismatches: **0**

This is strong evidence that the raceline projection, lateral objective and
steering direction are broadly correct for states actually visited by PP.

It also identifies the key limitation of the shadow experiment: MPC is handed
PP's already-working command history. Shadow mode therefore does not prove that
MPC can maintain its own actuator state after it receives authority.

## 4. AutoDRIVE steering command is not the instantaneous steering state

The PP shadow actuator trace contains both normalized steering commands and
measured steering feedback in radians. After conversion to the same units,
8,149 moving command/feedback pairs show:

- median absolute command-vs-feedback error: **0.01258 rad**
- p95 absolute error: **0.04551 rad** (~2.61 deg)
- p99 absolute error: **0.05811 rad**
- typical command age at feedback: median **14.14 ms**, p95 **23.23 ms**

Pure Pursuit already compensates this measured servo lag with
`/autodrive/roboracer_1/steering` feedback and a bounded lead term.

The baseline MPC did not consume steering feedback. It propagated
`steering_command` from its own command history and used that value directly
in the yaw-response model. Under authority this lets the optimizer believe a
requested steering angle has become the physical steering state before the
simulator has actually reached it.

The authority fix therefore uses fresh measured steering feedback as the
current MPC steering state. The optimizer still chooses a bounded steering
rate and publishes the next target angle; the next cycle closes the state loop
again from feedback.

## 5. Baseline MPC could drive before any accepted MPC solution existed

This is a concrete controller bug.

Before this patch, `publish_driving_fallback()` derived a fallback speed from
the startup/local-raceline cap even when `target_speed_initialized_` was
false and MPC had never accepted a solve.

Consequently a missing/unqualified map pose at startup could result in:

- nonzero speed request,
- zero/old steering,
- no accepted MPC trajectory.

This is opposite to the established PP startup contract, which stays neutral
until localization is qualified.

The fix introduces an explicit `has_accepted_command_` authority latch.
Before the first accepted solve, every state/localization/solver fallback is
neutral: **0 steering, 0 target speed**. Only after an accepted MPC cycle may a
temporary fallback retain a bounded previous command.

## 6. Baseline MPC accepted the first finite map pose; PP does not

Pure Pursuit requires:

- XY covariance <= 0.25 m^2,
- yaw covariance <= 0.12 rad^2,
- 5 consecutive good pose updates,

before it drives.

The baseline MPC checked only finite pose/frame values. A fixed process-start
delay is not equivalent to an AMCL-lock condition.

The authority fix copies the already-proven PP covariance qualification into
MPC. The synchronizer may continue receiving pose updates, but command
authority stays neutral until the qualification gate is satisfied.

## 7. Changes in branch `mpc-authority-state-fix-20260919`

- neutral fallback until the first accepted MPC command;
- PP-equivalent localization covariance gate and five-good-update requirement;
- steering-feedback subscription and fresh measured steering as MPC current
  steering state;
- explicit diagnostic telemetry for localization readiness, authority-command
  establishment and steering feedback;
- unit-testable authority policy helpers;
- ROS 2 Humble CI job that builds and tests `f1tenth_mpc`.

No Q/R weights, AMCL gains, corridor margins, vehicle-model coefficients or
solver tolerances are changed by this patch.

## 8. Required next live test

Run MPC authority with the same exact raceline and a 3 m/s global cap.

Record at minimum:

- `/mpc_shadow/diagnostics` (or the configured MPC diagnostics topic),
- `/current_map_pose`,
- `/odom`,
- `/ekf_odom`,
- `/cmd/speed`,
- `/autodrive/roboracer_1/steering`,
- steering/throttle commands,
- collision count for offline scoring only.

Acceptance for the first run:

1. no motion before `localization.ready=true`;
2. no motion before `authority_command_established=true`;
3. measured steering is fresh during moving MPC cycles;
4. no collision for one complete lap;
5. if a collision remains, preserve the final 2 s of MPC diagnostics and
   actuator feedback so the first divergence can be identified before tuning.

If this state-contract fix removes the low-speed crashes, the next optimization
stage is controller performance relative to PP (lap time, lateral error,
command smoothness and solver cost). If it does not, the next change should be
driven by the first captured authority divergence rather than by broad weight
sweeps.
