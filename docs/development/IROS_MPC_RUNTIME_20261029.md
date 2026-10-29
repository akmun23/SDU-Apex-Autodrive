# IROS competition MPC: runtime and validation progress

## Current gate

The best source-certified reference is **7.400 s**, witness
`f8c5b636134d8b64`. **a09 completed one warmup lap (9.314 s), then failed in
lap two. No scored lap or repeatable racing success yet.** The practice records
are not IROS competition results. Source certification, QP convergence,
accepted nonlinear horizons and actual completed driving are separate claims.

Single dashboard: <http://127.0.0.1:8765/>. Single optimizer workspace:
`live_runs/optimizer_iros_conditioned_20261029`. The validator pauses that
optimizer during builds, driving and replay, then resumes the same iterate.

## Implemented corrections

1. **Track-specific configuration.** The launch file had unconditionally
   applied the practice fast-line profile to IROS. Non-practice maps now get
   the competition base plus an explicit optional overlay. Parameter-file
   precedence is logged and tested.
2. **Replay ordering.** Diagnostics put rack angle after queued steering
   targets; C++ replay expects it before them. The canonical exporter now
   explicitly reorders them. Prefix replay uses that same header/exporter.
   Old results from the swapped exporter are not current solver evidence.
3. **Steering objective consistency.** Source-witness steering increments
   use commanded-target changes, not slower physical-rack changes. The first
   target error uses the actual prior command. QP and nonlinear merit match.
   Geometric-reference behavior is unchanged. Consistency does not prove the
   chosen weights or witness command pattern are optimal for tracking.
4. **Capped warmup.** Capped samples use geometry-only reference targets,
   not lateral/yaw/throttle states from a different-speed native witness.
   The same full physical plant predicts motion. Uncapped samples restore
   native references without modifying the trajectory file.
5. **Feedback-stabilized trials.** Optional `feedback_rollout` reuses final
   augmented Riccati gains already calculated by ADMM, transformed to physical
   units. It generates an explicit corrected command sequence during nonlinear
   trial evaluation. Its linear reference trajectory uses projected controls,
   not unconstrained primal states paired with different inputs. Commands keep
   the same actuator/trust boxes and exact plant/corridor/deadline checks.
   Backtracking starts from original QP controls, not an already corrected
   sequence. No future measurements, new solver coordinates or factorization.
   These are augmented Riccati gains, not constrained-optimum sensitivities.
   Default off; these IROS trials explicitly enable it.
6. **Soft clearance reference.** Optional `reference_clearance_reserve_m`
   moves the lateral target into the footprint corridor where zero offset
   leaves insufficient reserve. Uses desired body heading and rear-axle bumper
   anchors. Hard **5 cm** clearance is unchanged. Default zero; a08/a09 use
   3 cm. This does not certify a replacement raceline.

No physical source-plant equations or parameters were fitted/retuned here.
MPC still cannot choose zero throttle; odom/replay retain its hard-brake model.

## Unity evidence

All trials use production MPC, odom and AMCL on the IROS competition map.
Complete artifacts are in `live_runs/apex_mpc_f8c5b636134d8b64_aNN/`.
Progress is unwrapped along the approximately 42.97 m circuit; spawn is near
39.1 m. Until a09, no warmup lap completed.

| Attempt | Change | Terminal progress | Full-horizon accepts | Publication gate max |
| --- | --- | ---: | ---: | ---: |
| a01 | Leaked practice profile | 62.295 m | 165 | 18.514 ms |
| a02 | Correct map profile | 65.574 m | 155 | 23.928 ms |
| a03 | Native-target increment correction | 64.842 m | 158 | 18.479 ms |
| a04 | Cost scaling and warmup ramp | 65.007 m | 202 | 19.132 ms |
| a05 | More overspeed/yaw authority, less throttle damping | 76.574 m | 261 | 20.897 ms |
| a06 | Stronger heading/yaw bundle; reverted | 64.836 m | 210 | 18.904 ms |
| a07 | a05 plus feedback trial rollout | 62.937 m | 218 | See retained report |
| a08 | a07 plus soft clearance reserve | 77.284 m | 309 | 19.079 ms |
| a09 | Stronger path tracking, less native-increment damping | 107.140 m | 511 | 19.350 ms |

a07 used no recovery prefix or retained-plan actions, then rejected at a
tight wall with +2.3 cm lateral error. a08 passed there but accumulated -9 cm
error near the next turn; its last three-step prediction crossed its permitted
interval by about 1.4 cm. It used five retained-plan and two prefix actions.
a09 passed that turn and warmup, then developed heading/lateral error during a
chain of retained/prefix plans in lap two (20 retained, four prefixes overall).
Do not count those as full-horizon accepts.

The watcher aborts on the first MPC fault. Some bags also contain a collision
during shutdown roughly 0.1 s later. Metrics cut at the **first** fault; the
dashboard shows later collision evidence separately. These are not successful
or universally collision-free runs.

## Fixed-issued-command physics replay

Each 750 ms recursive prediction uses a logged production observer state and
actual subsequent issued commands, with **no future state correction**. GT is
scoring only. This is not an unexecuted MPC plan compared to Unity replanning.
Overlapping windows are not independent. Windows end before the first fault.

| Attempt | Windows | Endpoint XY RMSE | XY max anywhere in horizon | Paths over 5 cm |
| --- | ---: | ---: | ---: | ---: |
| a03 | 170 | 0.986 cm | 2.990 cm | 0 |
| a05 | 287 | 0.761 cm | 5.278 cm | 1 |
| a06 | 214 | 0.902 cm | 3.982 cm | 0 |
| a07 | 195 | 0.559 cm | 1.479 cm | 0 |
| a08 | 293 | 1.333 cm | 3.582 cm | 0 |
| a09 | 512 | 0.772 cm | 2.175 cm | 0 |

a09 forward/lateral speed RMSE: 0.00805/0.00143 m/s; yaw-rate RMSE:
0.0106 rad/s. Its loss of tracking is not explained by huge fixed-command
prediction error on this recording. This does not prove accuracy for arbitrary
unobserved sequences/regimes. The freshly compiled evaluator SHA was
`7de79b79a487fbcdc954c0719f5fcec11e3f16b7af0e2111d64a9efd2dccb1f8`.

## Offline alternatives already tried

- Feedback rollout improved same-state full-horizon acceptance **226/236 →
  233/236** on a04 and **281/311 → 287/311** on a05, with identical weights and
  two passes. Generated explicit controls reproduce the policy's plant rollout.
- Three/four SQP passes did not improve both corpora consistently; the trial
  extension was removed. Production retains at most two passes.
- Wider/disabled steering trust regions did not transfer consistently; kept
  0.08 rad. Cold/reference/warm initialization and input-penalty variants did
  not justify globally replacing initialization. a06 regressed online distance.
- Saved-state replay cannot restore all live ADMM/certified-plan memory.
  Counterfactual acceptance counts are not online driving proof.
- On the complete a09 corpus, warm initialization accepted 512/536 horizons;
  reference-seeded and fully cold initialization each accepted 514/536.
  Raising heading weight from 30 to 200 (and lateral from 200 to 300) reduced
  acceptance to 507/536. These comparisons used the same current physical
  plant, feedback trial rollout and 3 cm soft reserve. They do not justify
  another blind online heading-weight increase or a global cold-start switch.

## Automation and next gate

- One enabled GUI/optimizer; old reverse campaign stopped and disabled, files
  recoverable. Retry is same-origin JSON into a serialized queue, not arbitrary
  command execution. No automatic weight search or infinite retries.
- Per-attempt overlays/SHA, map, trajectory and controller image are retained.
  Fresh production build; monitored Unity; failed bags decoded too.
- Automatic report builds/fingerprints the source-only plant evaluator and
  shows fixed-command prediction metrics beside timing/tracking/lap results.
- Missing recorded images allow an explicitly documented decoder-only image
  substitution (`analysis_environment.json`), not silent binary equivalence.
- Unknown optional metadata becomes JSON `null`, not NaN; regression-tested.
- Warmup completion remains visible even when no scored lap finishes.
- Pause leases check owner PID and process start time. Supervisor restart does
  not thaw a still-live offline replay. Normal SIGTERM releases owned work.

Reproduction: a09's immutable overlay is
`mpc_validation/inputs/f8c5b636134d8b64/attempt_09_mpc.yaml` in the optimizer
workspace; it is also copied to the trial's `mpc_overlay.yaml`. Relative
priorities: lateral/heading 200/30, underspeed/overspeed 0.35/2, yaw 2, wheel
slip/lateral speed 0.05/0.05, steering-command/target 0.03/0.05, throttle command
0.001, steering/throttle increments 1/0.1. Feedback enabled, soft reserve 0.03 m,
startup 3 m/s ramping over one warmup lap. Base solver: two SQP passes,
150 ADMM iterations/pass, eight Jacobian workers, 20 ms publication budget.
This incomplete racing profile is not promoted over the practice baseline.

Verification: 120 focused Python regressions passed; 47 C++ plant/solver tests
passed, including gain scaling/unchanged QP solution, explicit generated-control
replay and clearance geometry. Shell/JavaScript syntax and runtime-topic-policy
checks passed. Source model fidelity is measured by the replay table above,
not inferred from those unit tests.

Next: use a09's states **before** the recovery chain to distinguish a bad
nominal basin from weights/ADMM convergence. Re-test a demonstrated change at
full reference speed. Keep hard clearance intact. The required result remains
repeated scored laps, not just a warmup or accepted QPs.
