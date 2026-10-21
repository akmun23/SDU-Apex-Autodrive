# MPC runtime timing, failure behavior, and bag decoding — 2026-10-21

## Scope and runtime inputs

The development MPC stack subscribes to `/odom`, `/current_map_pose`, and the
odometry-owned `/odom/unity_plant_state`. Production odometry subscribes to:

- left and right wheel encoders;
- IMU;
- steering and throttle feedback;
- steering and throttle commands.

The production topic-policy guard passed. Ground truth, collision/lap topics,
simulator reset, and transform topics are not controller or odometry inputs.
The MPC consumes the plant state published by odometry; it does not receive
ground truth or the development actuator timing stream.

The permitted feedback provides a useful, current packet snapshot of steering
and throttle and lets odometry correct the corresponding actuator state. The
command history is also known locally. Neither feedback topic carries a
command sequence ID or Unity physics-step ID, so the exact command consumed by
each hidden `FixedUpdate` cannot be recovered from the permitted runtime
topics. The debug bridge stream can characterize request/response timing
offline, but is not used by the runtime controller. This remaining timing
uncertainty is explicit, not treated as a known per-step schedule.

The recorder additionally captures
`/autodrive/roboracer_1/development_actuator_timing` for offline diagnosis. The
bag evaluator now decodes it and reports command sequence continuity and
interface-receive-to-actuator-publish delay. It is diagnostic output, not a
control input.

## Code changes in this update

- End-to-end MPC publication gate is strictly `<20 ms`. Command-start
  prediction lateness remains a diagnostic measurement, not a second rejection
  gate.
- A finite iteration-limited candidate is eligible only after the same exact
  nonlinear rollout and corridor acceptance used for an optimal candidate.
- If the current solve is rejected, MPC shifts the last accepted plan by one
  action and revalidates the remaining sequence from the current state against
  the exact plant, command envelope, and corridor before publishing it. It
  consumes only one plan action per cycle. It does not issue an unchecked stale
  command or synthesize a brake command. The reuse envelope uses the same
  numerical tolerance and clamps as the actual publisher, so float rounding at
  the steering limit cannot incorrectly block a valid saved action.
- If neither the current candidate nor a remaining prior plan passes exact
  validation, the MPC latches a fault and publishes no replacement command.
  In the development run, the supervisor aborts the simulator on that fault.
  The production path has no external development supervisor; no claim is made
  that a rejected-plan fault is a complete actuator-stop lifecycle.
- Corrected an extra quote in MPC JSON diagnostics. The bag decoder recovers
  this one known historical serialization typo and preserves unknown malformed
  diagnostics as explicit decode errors rather than silently dropping them.
- The bag evaluator directly decodes `UnityPlantState` schemas 9–11 and
  `DriveCommand` CDR without requiring generated Python message classes.

## First real practice simulator run

The established practice simulator was launched in batch mode with Xvfb and
Vulkan; no GUI and no `-no-graphics` option were used. The development stack
compiled the mounted workspace source before startup. Controller max speed was
capped at 3 m/s for this supervised runtime/failure-path capture.

Capture: `live_runs/mpc_runtime_policy_reuse_20261021/run`

- 533 truth/odometry/plant samples over 13.32 s of sensor time; peak ground-
  truth speed 2.968 m/s.
- No collision and no completed lap. This is a short capped practice test, not
  a racing-performance result.
- MPC diagnostics: 489 `accepted_optimal`, 29 `accepted_feasible` after the
  solver iteration limit but after exact nonlinear validation, one
  `accepted_prior_plan`, one nonlinear corridor rejection, plus the terminal
  fault diagnostic. Thus 519/520 attempted cycles published a validated
  current or revalidated prior-plan action.
- The prior-plan path was exercised once and published a validated next
  action. At a later cycle the candidate failed exact corridor validation at
  stage 26; its lateral state exceeded the tolerance-adjusted bound by only
  `0.000076 m`. The remaining old plan also failed current-state corridor
  validation, so the controller faulted and the run supervisor stopped the
  simulator. No fallback/braking command was issued. The bag recorded the
  bridge connection closing after the supervised stop.
- Whole-cycle time p50/p95/max: `10.654/13.288/16.563 ms`; end-to-end control
  time max: `16.886 ms`. This run meets the requested strict 20 ms ceiling.
  QP build p50/p95/max was `5.864/6.941/7.593 ms`; exact rollout was
  `1.331/1.530/2.141 ms`.
- ADMM converged in 491 cycles and hit the iteration limit in 29; every
  iteration-limited current candidate counted as accepted only after exact
  nonlinear validation. This establishes that the current acceptance path is
  exercised, not that all future operating conditions will converge.
- The actuator diagnostic stream had 519 forwarded commands, no sequence gaps,
  and interface-receive-to-actuator-publish delay median/p95/max
  `0.131/0.208/1.849 ms`. This is host-side publication timing, not proof of
  Unity's per-fixed-step application sequence.

## Follow-up live validation after decoder and reuse fixes

Two more short supervised practice runs were used only to verify fresh JSON
diagnostics and the evidence-driven actuator-envelope correction. Both used
the same practice map/raceline, 3 m/s configured speed ceiling, established
batch-mode/Xvfb/Vulkan launch, full recording, and immediate stop on collision
or MPC rejection.

### Fresh-diagnostics run

Capture: `live_runs/mpc_runtime_policy_jsonfix_20261021/run`

- New MPC diagnostic JSON decoded without repairs or errors. The watcher
  recognized and stopped on the first rejected MPC cycle; no collision or lap
  occurred.
- 277/278 attempted MPC cycles produced an exactly validated command; the
  remaining candidate failed corridor validation at stage 26. Its predicted
  lateral state exceeded the already tolerance-adjusted bound by `2.813 mm`.
- End-to-end control time max `16.157 ms`; cycle time p50/p95/max
  `10.055/12.176/15.886 ms`.
- Peak speed `3.004 m/s`; sensor span `8.73 s`.
- Fixed-input source-plant rollout scored 226 complete 750 ms windows. All
  endpoints and full paths were under 5 cm; worst full-path position error was
  `31.8 mm`. This remains low-speed-only evidence.
- The prior-plan attempt was rejected by an inconsistent precheck: stored
  steering `0.52359879 rad` was just `1.4e-8 rad` above the double-precision
  limit, even though the normal publisher accepts and clamps that float.

### Prior-plan envelope correction run

Capture: `live_runs/mpc_runtime_priorplan_clamp_20261021/run`

- The corrected reuse path was exercised in the real simulator: 16
  `accepted_prior_plan` commands were published after exact current-state
  validation. The rounding-only rejection disappeared.
- Bag diagnostics were valid JSON with zero decode repairs/errors; all 403
  recorded actuator publication entries were forwarded, sequence gaps `0`,
  publication latency median/p95/max `0.114/0.161/0.302 ms`.
- 408/409 attempted MPC cycles published an exactly validated current or prior
  action. Statuses: 372 optimal current candidates, 20 exactly validated
  iteration-limited candidates, 16 revalidated prior actions, and one
  corridor-rejected candidate.
- Whole-cycle p50/p95/max `9.962/12.591/15.245 ms`; maximum end-to-end control
  time `15.771 ms`. It meets the `<20 ms` requirement in this run.
- The final candidate failed exact corridor validation at stage 19 by about
  `20.6 mm` beyond the tolerance-adjusted bound. The remaining prior sequence
  also failed current-state corridor validation, so the supervisor stopped
  the simulator. No collision and no lap occurred. This is a genuine
  feasibility failure, not the earlier floating-point limit artifact.
- The car reached `3.378 m/s` despite the configured 3 m/s speed ceiling;
  that parameter is a reference ceiling, not a hard speed clamp.
- Odometry relative-pose error over the 9.38 s sensor span was `1.2 cm` at
  endpoint, `2.9 cm` p95, `3.4 cm` max. AMCL control-pose error was `5.0 cm`
  RMSE, `7.1 cm` p95, `8.2 cm` max.
- Fixed-input source-plant rollout scored 346 complete 750 ms windows:
  XY RMSE/p95/max `4.99/11.35/25.51 mm`; all endpoints and all full paths
  stayed below 5 cm. Maximum one-step yaw-rate residual was `0.448 rad/s`,
  while 750 ms yaw error RMSE/max was `0.00770/0.04448 rad`; retain both
  metrics rather than letting the integrated score hide the one-step tail.
- The controller's receding-horizon diagnostic at 750 ms showed XY RMSE/max
  `0.251/1.305 m`. Because commands are replanned at 25 ms, this is not the
  fixed-input plant rollout metric; it remains evidence that closed-loop
  forecast/trajectory consistency needs work.

### Motion accuracy from this capture

The evaluator decoded all 533 `UnityPlantState` and 519 `DriveCommand`
messages directly from CDR. It recovered 520 MPC diagnostic messages that had
the known duplicate-quote defect; no diagnostic payload remained undecodable.

- Odometry forward-speed RMSE `0.0487 m/s`, lateral-speed RMSE `0.0175 m/s`;
  yaw rate is the direct gyro observation in this stack and therefore its
  zero residual here is not an independent predictive-accuracy result.
- Relative odometry pose error at the end of the recorded sensor span was
  `0.070 m`, with `0.104 m` p95 and `0.106 m` maximum. This does not meet the
  separate 5 cm long-run pose target.
- AMCL control-pose error versus truth: `0.059 m` XY RMSE,
  `0.084 m` p95, `0.197 m` maximum.
- Fixed-input source-plant rollout: 503 scored windows at 25 ms intervals,
  each using logged initial plant state and logged command history without
  future-state correction. At 750 ms, XY endpoint RMSE/p95/max was
  `0.00638/0.01292/0.03831 m`; all 503 endpoints and all 503 full paths stayed
  below 5 cm. Maximum start speed in this capture was below 3 m/s, so this
  does not validate high-speed rollout.
- One-step yaw-rate RMSE/max was `0.02361/0.18797 rad/s`; 750 ms yaw error
  RMSE/max was `0.00254/0.01900 rad`. These are separate metrics; the low
  integrated-yaw error does not erase the one-step tail.
- The MPC's receding-horizon diagnostic compared the evolving closed-loop run
  against each cycle's forecast; at 750 ms its XY RMSE/max was `0.0589/0.268 m`.
  This is not a fixed-command rollout score, because commands were replanned
  every 25 ms. It does show the closed-loop prediction/trajectory discrepancy
  remains large and requires its own diagnosis.
- `/ekf_pose` scored with a large map-frame discrepancy (`7.540 m` XY RMSE,
  `1.571 rad` yaw p95). MPC uses `/current_map_pose`, not `/ekf_pose`; this
  needs a separate frame-alignment investigation and must not be mistaken for
  the MPC's pose input error.

## Verification

- `python3 tools/verify_runtime_topic_policy.py`: PASS.
- `python3 -m py_compile tools/evaluate_mpc_bag.py tools/watch_sim_run.py`:
  PASS.
- `colcon build --packages-select f1tenth_mpc --parallel-workers 4`:
  PASS after the diagnostic serialization fix.
- Focused `test_vehicle_model_source_rollout`: 28/28 tests PASS, including
  exact validation of iteration-limited candidates and prior-plan suffixes.
- The retained practice bag decodes custom plant/control messages and the
  prior malformed MPC JSON records; diagnostic statuses and timings are now
  recoverable instead of silently absent.
- All simulator/dev/recorder containers from this run have stopped.

## Remaining work / interpretation

The next useful run should follow a root-cause correction for the stage-26
corridor rejection and rejected prior suffix, not repeat the unchanged capped
trial. Inspect the logged state/reference/bounds and the minimum-clearance
geometry before changing any tolerance or weights. Then validate a faster
practice case with the same full-bag decoder and immediate
collision/rejection supervision.

The current prior-plan behavior is bounded by exact current-state validation:
it continues the prior converged intent only while that suffix is still
feasible. If no valid candidate exists, it refuses to publish an invalid
continuation and latches a fault. Because no emergency/braking fallback was
authorized, a competition deployment still needs an explicit external fault
lifecycle before it can be called robust to a no-feasible-plan event.
