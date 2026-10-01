# Open-plane throttle surface: final capture and first fit — 2026-09-30

## Capture status and integrity

The randomized 13-angle, reset-isolated throttle step-response design is
complete across the original and supplemental captures:

- r04: 770 fit-usable conditions; one interrupted condition; the process was
  stopped before the full design finished.
- r05 supplement: 738/738 scheduled conditions complete and fit-usable;
  zero invalid conditions. Its closed-bag integrity gate passed.
- Combined: all **1,508 expected replicate keys** are represented exactly
  once by a fit-usable condition. These form 754 steering/throttle
  combinations with exactly two valid replicates each. The interrupted r04
  condition was completed by r05; the incomplete attempt is not counted.

The r05 bag audit reports 738 phase starts and ends, no duplicate, missing, or
unexpected scheduled keys, 738 command/throttle-feedback/steering-feedback
tracking passes, and 738 usable response conditions. There were 738 requested
resets and 739 recoveries (including the initial spawn reset); maximum reset
position error was 1.6 mm and the maximum condition-start spread from spawn
was 0.1 mm. Collision count and bridge timing-fault count were both zero; the
experiment ended normally.

All nine active streams passed the 40 Hz gate. Measured rates were about
39.981–40.004 Hz; p95 message gaps were about 25.1–25.8 ms and the largest
observed gap was 55.3 ms. Throttle feedback endpoint error was approximately
`2.4e-8` normalized at p95. The uncapped test reached 20.7951 m/s and
173.68 m maximum displacement from spawn without a collision or timing fault.
These are observed outcomes, not a claim that every speed/steering combination
is safe or feasible.

The original r04 whole-run integrity flag remains false because that capture
was intentionally incomplete. Its 770 individually valid phases are retained
and were used only after the cross-capture replicate-key audit established the
complete union with r05. Do not interpret `source_passed_integrity_gates=false`
in the combined fit as failure of its selected rows: it records that r04 as a
whole was partial.

## First response-surface fit

The exploratory fit uses all 1,508 fit-usable condition summaries. It predicts
the measured body-speed change from the throttle-step moment to the end of the
8-second response window. Grouped out-of-fold validation keeps both replicates
of a steering/throttle condition together.

- Holding out complete throttle transitions (58 groups), ExtraTrees with
  command inputs and measured baseline body state achieved 0.827 m/s RMSE,
  0.378 m/s MAE, and R² 0.912. Commands plus speed alone achieved 0.986 m/s
  RMSE. The baseline body state therefore adds predictive information within
  the sampled steering support.
- Holding out replicate groups (754 groups), the body-state ExtraTrees fit
  achieved 0.778 m/s RMSE and R² 0.923.
- Holding out entire steering commands (13 groups), its RMSE rose to
  2.829 m/s with R² -0.023, barely better than the constant baseline
  (2.903 m/s). This is not reliable generalization to unseen steering
  regimes; the present fit has not resolved the high-angle modelling gap.
- Throttle-feedback half-response time was not predictively improved:
  ExtraTrees RMSE was 7.67 ms versus 7.42 ms for the mean baseline on held-out
  throttle transitions. The feedback stream closely tracks its command, so
  this test's useful dynamics appear in the vehicle response, not a complex
  throttle-feedback delay.

These are condition-level cross-validation scores from reset-isolated trials,
not independent continuous-run validation and not a recursive plant rollout.
They have no run-cluster confidence intervals. The response target is an
8-second endpoint difference; fitting it does not show that intermediate
motion, yaw, lateral velocity, wheel speeds, or pose can be predicted
accurately. Do not use this surface fit in MPC, odometry, EKF, or AMCL.

## Artifacts and code changes

- Closed r05 bag:
  `live_runs/openplane_throttle_5pct_5deg_20260930_r05_resume/run/run_0.db3`
- r05 integrity report:
  `live_runs/openplane_throttle_5pct_5deg_20260930_r05_resume/throttle_transition_analysis.json`
- Combined grouped-fit report:
  `live_runs/openplane_throttle_5pct_5deg_20260930_r05_resume/preliminary_fit.json`
- The scheduled-condition audit in
  `tools/analyze_open_plane_throttle_transitions.py` was fixed after its
  automatic post-run analysis hit a `NameError`. The closed bag was re-read
  offline and the successful report above was produced; no capture data was
  changed.
- `tools/fit_open_plane_throttle_surface.py` now accepts multiple analysis
  reports, rejects duplicate fit-usable replicate keys, and writes the
  replicate coverage distribution alongside its grouped scores.

## Next work

1. Export full 40 Hz, reset-delimited sensor/command/feedback trajectories for
   all fit-usable conditions, preserving phase, replicate, reset, and quality
   labels. Keep baseline and response windows distinct; never train across a
   reset.
2. Resolve `/ips` semantics and align it with `/odom` using timestamp/frame
   evidence before treating either as truth. Until then, report odometry as the
   measured label rather than simulator ground truth.
3. Compare the full recursive teacher models only after sequence export:
   predict effective body accelerations with explicit rigid-body coupling and
   learn actuator/wheel/history residual state. At rollout, use commands, `dt`,
   and the model's own state—not future recorded sensors or feedback.
4. Preserve whole conditions and entire continuous runs as separate test
   levels. The present 13 steering angles are not enough to establish
   interpolation or generalization at high angle; use fresh feasible
   trajectories for final validation.
