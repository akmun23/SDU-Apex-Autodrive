# MPC mathematical and epoch specification

Revision: 2026-11-01. This describes the maintained runtime implementation,
including its remaining limitations. It is the reference for formulation and
derivative tests. It does **not** certify Unity parity, recursive feasibility,
real-time qualification or racing performance.

## 1. Time and causality

Use three distinct clocks:

* `t_n`: integer ROS nanoseconds captured when the host emits Bridge request n.
  The exact rounded outgoing command and all associated response sensor headers
  carry this stamp. Association is FIFO; the stock simulator does not echo a
  request sequence or a physical sample time.
* `t_c`: the predicted next request epoch after expected command publication.
  `next_bridge_request_epoch` extrapolates from `t_n` with a 25 ms period.
* Monotonic time: callback, solve and publication deadlines. ROS time is never
  subtracted from a monotonic deadline.

The discrete source interval is 25 **simulated** milliseconds, integrated as
25 one-millisecond source steps. Host request spacing is approximately 25 ms;
network latency and host phase correction are not additional simulated steps.
Odometry intentionally advances its observer clock by one 25 ms interval per
complete packet. Do not replace that clock with jittery host timestamp deltas.
Missing packets must therefore never silently remove integration intervals.

The sensor assembler waits for an earlier partially received packet before
emitting a newer complete packet. The actuator join waits for exact command,
feedback, encoder and IMU stamps. Queue exhaustion fails closed. A completely
missing sensor packet still requires transport sequence evidence; host stamps
alone cannot distinguish packet loss from host scheduling jitter.

The map pose is an anchor for a map/odometry rigid transform, rather than a
current twist observation. Given odometry poses `O(t_a), O(t_s)` and map anchor
`M(t_a)`, synchronization computes `M(t_s)=M(t_a) O(t_a)^-1 O(t_s)`.
Both odometry poses must have interpolation brackets. Nearest-timestamp
substitution and relabeling older twist with a future pose timestamp are forbidden.
The output pose and twist are at `t_s`, with the anchor stamp retained separately.

Hidden-state snapshots are placed in tracking coordinates at their own epoch,
propagated to `t_s`, and fused there with observable motion. The sensor-epoch
snapshot is retained; a separate copy advances to `t_c`. All SQP passes, QP
Jacobians, exact costs and nonlinear validation use that **one frozen copy**.
Callbacks arriving during the solve are inputs to the next cycle.

Control-time propagation consumes `/odom/bridge_request_command`, containing
the actual rounded request payload. ROS publication time is not application
time. Its event boundaries are quantized relative to the initial snapshot's
one-millisecond grid, with less than one millisecond discarded over the whole
prediction, rather than once per command segment.

**Unclosed epoch obligation:** `t_c` remains a host phase/latency prediction.
The middleware chain and stock Unity player do not acknowledge the exact
FixedUpdate that applies a newly published command. The unused
`max_command_start_lateness_ms` setting does not establish this guarantee.
Current code cannot honestly claim physical command-epoch equality end to end.

## 2. Coordinates, complete state and optimization coordinates

The body frame is rear-axle referenced: x forward, y left, yaw positive left.
Map pose is reconstructed from path position/tangent plus signed lateral error
`e_y` and heading error `e_psi`. Angles use radians; distance uses metres;
linear/angular velocities use m/s and rad/s; wheel rates use rad/s.
The source plant internally uses COM lateral velocity:
`v_COM = v_rear + r * rear_axle_from_COM`.

The nonlinear state is:

```
z = (e_y, e_psi, u, v_rear, r, rack,
     omega_FL, omega_FR, omega_RL, omega_RR,
     height, height_rate, roll, roll_rate, pitch, pitch_rate,
     jounce_FL, jounce_FR, jounce_RL, jounce_RR,
     forward_sticky_timer[4], side_sticky_timer[4],
     steering_history[2], throttle_history[2], initialization_flags)
```

Progress `s` is propagated separately and is unwrapped across laps. Wheel
rotation angles are encoder-output accumulators, not force-generating MPC states.
`UnityPlantState` schema 11 is a sensor-corrected **estimate**, never Unity truth.
Its `sensor_corrected_state_mask` records which mechanical channels are corrected.

The QP has 23 state coordinates:

| Indices | Coordinates |
| --- | --- |
| 0–4 | e_y, e_psi, u, v_rear, r |
| 5 | physical rack angle |
| 6–9 | wheel angular rates FL, FR, RL, RR |
| 10–15 | height/rate, roll/rate, pitch/rate |
| 16–19 | suspension jounces FL, FR, RL, RR |
| 20 | progress deviation `s - s_nominal[k]` |
| 21–22 | previous absolute steering and throttle targets |

The eight sticky timers are carried by the nonlinear rollout but are not QP
decision coordinates. Branch-local derivatives do not establish accuracy across
sticky/contact activation boundaries. This reduced chart is not a proof that
all finite-step effects of hidden memory are captured by the QP.

Two controls are absolute physical steering target `a_delta` and normalized
throttle `a_q`. Increment mode uses `d_k=a_k-a_(k-1)` with the measured/confirmed
command before stage zero held fixed. This is a coordinate change, not a trust
region. Full Hessian state/input coupling must transform with the dynamics.

## 3. Transition and actuators

Let `P_1ms` denote the checked-in `unity_source_plant_step_1ms` transition,
including wheel forces, motor/brake branches, steering slew, suspension, body
attitude, wheel spin and sticky timers. The physical 25 ms transition is its
25-fold composition under held actuator target. The source integrator also
accumulates body displacement and yaw at one-millisecond resolution.

`mpc_trajectory_advance_tracking` maps that integrated displacement into the
actual sampled path geometry. Endpoint Euler twist integration and an unrelated
constant-curvature bicycle model are not interchangeable with this transition.

`mpc_vehicle_model_step(z,a,dt,kappa)` applies the stage decision directly to
the source plant. Its command histories are bookkeeping, not an additional
two-stage physical FIFO. `mpc_vehicle_model_propagate` instead holds the newest
confirmed request command while moving an existing state to a later epoch.
Odometry integrates the interval just completed using the preceding request,
then queues the command associated with the new packet. Endpoint rack feedback
corrects the completed interval's endpoint; it must not be backdated over that
interval. Headerless throttle feedback never replaces its completed input.

The physical rack limit is ±pi/6 rad, with source slew 183.346 degrees/s.
Outgoing steering is normalized by pi/6 and rounded to three decimals by the
Bridge; throttle is also rounded to three decimals. Feedback `V1 Steering`
is already radians, unlike the outgoing normalized steering command.
The actuator interface validates a pair and publishes it atomically on
`/actuators/drive_command`; the Bridge snapshots both under one lock. Separate
Float32 actuator topics remain hardware/recording outputs, not competing
Unity request inputs.

Exact zero throttle selects Unity's active-brake branch. Normal production
optimization and publication use `[0.001,1]`; this excludes active braking by
design. Reducing positive torque is not mathematically equivalent to braking.
Any future corner-anticipation analysis must account for this restriction.

The public source model is not the native Unity WheelCollider/PhysX binary.
One-step derivative agreement cannot certify multiple-step simulator fidelity.

## 4. Finite-horizon problem

Normal horizon: N=30, h=0.025 s, T=0.750 s. Initial `z_0,s_0,a_-1` are fixed
at the predicted command epoch. Choose `a_0...a_29` subject to the nonlinear
transition and the constraints below.

At the first nominal, freeze stage tracking targets, speed-weight branches
and reference steering increments in one `MpcRtiObjectiveSchedule_t`. R2 and
exact merit evaluate that same objective. Future **constraint geometry** is
resampled at candidate physical progress; frozen targets cannot freeze walls.
An uncapped source witness supplies its validated state/command references.
A capped-speed phase drops witness state references and uses geometric targets.

Define `D(e,t)=sign(e)*max(abs(e)-t,0)` for optional tracking dead zones,
`b_k=terminal_multiplier` at k=N and 1 otherwise. State cost is

```
L_k = w_y D(e_y-y_ref,t_y)^2 + w_psi D(e_psi-psi_ref,t_psi)^2
    + w_speed[k](u-u_ref)^2 + w_v(v-v_ref)^2 + w_r(r-r_ref)^2
    + w_rack(rack-delta_ref)^2 + w_rack(a_prev_delta-delta_ref)^2.
```

Overspeed weight is `w_over*(1+g_over*abs(kappa_ref))`; underspeed weight is
`w_under/(1+g_under*abs(kappa_ref))`. Selection is frozen from the first
nominal, so a line-search comparison does not change objectives mid-cycle.
There is no independently optimized lap-time or progress reward; speed
tracking and the loaded trajectory encode the current racing preference.

For each input stage add:

```
U_k = w_target(a_delta-delta_ref[k])^2 + w_throttle*a_q^2
    + w_steer_slew*((a_delta-a_prev_delta)-Delta_delta_ref[k])^2
    + pseudo_Huber_steering_correction_variation(...)
    + w_throttle_slew*(a_q-a_prev_q)^2.
```

`Delta_delta_ref` comes from consecutive rack/feedforward targets, not from
an unexecuted proposal. Add four squared positive wheel surface-speed excess
residuals measured over each complete incoming source interval, with curvature
weighted slip coefficient and the receiving stage's `b_k`. This is a soft
traction penalty, not a friction-circle hard constraint.

Total objective is `sum(k=0..N) b_k L_k + sum(k=0..N-1) U_k + slip_cost`,
plus the optional terminal continuation value described below. Weights have
units of cost divided by their squared residual units; they are tuning choices,
not measured physical parameters. Their physical adequacy remains unqualified.

## 5. Hard constraints and approximations

* Physical steering target/rack envelope; normal throttle in `[.001,1]`.
* Body speed within the source command ceiling, with the documented 8 mm/s
  reverse transient tolerance in state validation. Profile speed targets are
  separate from the physical/project command ceiling.
* QP heading chart ±pi. This is a coordinate-chart restriction, not a tire-force
  limit. Wheel-rate bounds use the standard unconstrained sentinel. Former ±350 rad/s
  entries also exceeded BIG_BOUND=50 and were already unconstrained, not
  active wheel-rate limits. Source and validator impose no such saturation.
* Future body footprint clearance. Body geometry: width .273 m, rear overhang
  .080 m, rear-axle-to-front-bumper .430 m. Active profile supplies margins.
  Stage zero is fixed measured input; the optimizer cannot repair it.
* QP corridor planes linearize all four body corners. Exact candidate checking
  reprojects source displacement through candidate future geometry.
* Optional world-map checker tests swept interpolated body poses between stage
  endpoints. It is not continuous verification of every source/Unity substep.

Command increment bounds in runtime must span the complete command range.
Narrow .10 rad/.25 defaults were removed; runtime rejects artificial narrower
settings. The generic offline QP API can still describe explicit rate-constrained
problems, which must not be confused with this Unity runtime problem.

## 6. SQP and numerical contract

At each exact nonlinear nominal, build source-derived A/B, affine offsets,
complete stage Q/q/R/r/N, coupled footprint constraints and terminal terms.
The QP convention is `0.5*x'Q*x+q'x+x'N*u+0.5*u'R*u+r'u`, with symmetric
off-diagonal arrays stored once per matrix entry, and progress in nominal-error
coordinates. Source-slip Gauss–Newton outer products are retained in full.

Riccati-ADMM and independent OSQP solve **the frozen QP**, not the nonlinear
problem. Variable scaling transforms dynamics, objective and constraints
exactly. Positive-definiteness repair of a 2x2 control Schur block uses bounded
diagonal regularization; counts and maximum regularization are diagnostics.
It is not a physical cost term and may not be hidden as convergence evidence.

There are at most two full SQP passes. R1 reserves a complete R2 opportunity
when the remaining budget permits. R2 uses the same initial state, command
epoch and frozen objective; it must not shift commands or progress seeds.
Progress warm split states are rebased to the new nominal and scaling;
same-epoch R2 keeps stage indices. Runtime warm controls and solver primal/dual/penalty arrays advance by elapsed
command stages. Repeated epochs advance zero stages; skipped epochs advance
multiple stages; reversed, unknown or expired epochs cold-start. Host request
jitter is rounded to the nearest fixed source stage. Retained plans and
three-stage prefixes use the same epoch rule and expire, rather than consuming
one action per callback. Legacy offline callers with epoch zero explicitly
retain one-stage-per-call semantics.

Every direction is tried at full scale first. The bounded nonlinear search
tests alpha=.5,.25,.125,.0625 when useful and affordable; alpha=0 may retain an
already verified incumbent. These are finite sampling/budget choices, not a
bound on distance from nominal. A shared scalar alpha couples both inputs and
can miss a useful mixed steering/throttle direction.

Optional source-informed R2 curvature uses the adjoint of the normal QP on
the exact source trajectory. Its positive secant correction is PSD and centered
at the new nominal, preserving the gradient. It is an approximation, not a
global curvature bound. The separate source optimizer is an offline oracle;
runtime source-primary and independent post-QP optimization are removed.

Restoration reuses the same Jacobians, relaxes **linearized** corridor bounds
and adds PSD wall penalties. Exact collision/physical acceptance remains
unchanged. Unsupported terminal heading/slip funnels were removed from primary
restoration acceptance. This is not yet a complete elastic SQP/filter formulation.

Numerical optimality, capped finite feasibility, nonlinear validity, deadline
exhaustion and publication are distinct outcomes. A capped QP can supply a
verified action but cannot be reported as an optimal nonlinear solution.
Mathematical memory committed by a solve is discarded if publication misses;
only an actually issued retained sequence may seed that recovery.

## 7. Terminal conditions

Default normal optimization has a terminal tracking multiplier and final
footprint checks, without a recursively invariant terminal safe set.
`terminal_viability` is experimental and default-off. It appends a three-step
75 ms held-final-command source continuation to the normal objective and
terminal inequalities: future body clearance, physical speed and nonnegative
forward progress. Exact nonlinear suffix verification is mandatory.

This finite backup is a useful viability approximation, not proof of safe
progress for all future MPC updates. The current live terminal experiment
failed after three lap boundaries; it is not qualified. External terminal
screening/recovery ranking remains until an integrated replacement is validated.

## 8. Component conformance and test authority

| Component | Problem implemented / obligation |
| --- | --- |
| `vehicle_model.cpp`, `unity_source_plant.cpp` | Common nonlinear source transition; hidden timers and branch changes remain important. |
| `mpc_rti_qp.cpp` | Sole normal QP builder and complete increment-coordinate transform. |
| `mpc_source_adjoint.cpp` | Adjoint of that frozen quadratic at the nonlinear nominal; finite-difference agreement with the complete frozen nonlinear cost is the required test. |
| `riccati_solver.cpp`, `riccati_factorization.cpp` | Frozen affine QP; scaling and regularization must be reported. |
| `mpc_rti.cpp` | Frozen objective, normal SQP, exact rollout/merit and restoration. |
| `mpc_terminal_viability.cpp` | Optional local terminal continuation model and exact suffix test. |
| `mpc_source_nonlinear_optimizer.cpp` | Offline shooting oracle; no runtime output authority. |
| Controller recovery | Additional source-verified candidate generators/selection, not demonstrated redundant. |
| Trajectory/IPOPT tooling | Witness or offline trajectory optimization. A witness using this source plant is not automatically the same online objective, constraints or uncertainty model. |
| Bridge and odometry | Host request association and state estimation; neither is physical Unity ground truth. |

Reference tests: `test_vehicle_model_source_rollout`, `test_unity_source_plant`,
`test_mpc_sqp_convergence`, `test_mpc_feasibility_restoration`,
`test_mpc_state_synchronizer`, `test_bridge_epoch_assembler`,
`test_sensor_packet_assembler`, frozen-QP fixtures and independent OSQP/LP tools.
Do not use a passing internal consistency test as evidence of external model
fidelity or completed five-race qualification.
