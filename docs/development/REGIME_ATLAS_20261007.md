# Regime Atlas — 2026-10-07

## Purpose and scope

This is the WP-R0 evidence map, updated with the Y1 specialist fit, its narrow
default-MPC integration, and a read-only source-order reconstruction from the
two clean P0 bags. See the repository [modeling rules](../../MODELING_RULES.md).
It is not a general vehicle model, runtime router, capability envelope, or
recommendation to raise race speeds. Odometry, localization, simulator
physics, and the protected P0 trajectory remain unchanged.

The structured source is [regime_atlas.yaml](../../config/racing/regime_atlas.yaml).
The machine-readable mirror is
[`support_atlas.json`](../../live_runs/racing_model_diagnostics_20261007/regime_atlas/support_atlas.json).

## What the current evidence supports

| Specialist | Existing evidence and exact support | Split / status | What it does not establish |
|---|---|---|---|
| S0 shared transition | Existing 25 ms production C transition, 0.324 m wheelbase, ±0.523598 rad steering, 3.2 rad/s steering-rate limit, one command-step delay, target-speed state and Frenet integration; C/Python/CasADi parity artifact exists. | Existing parity checked; trusted implementation baseline. | Equation parity is not simulator-truth accuracy or a full-lap plant. |
| Y0 ordinary yaw | Two unchanged P0 sessions, 20 scored laps; best 5.7378 s, pooled mean 5.7904 s. | P0 parent is the protected baseline. | The P0 samples do not cover truth speed ≥6 m/s jointly with |steering| ≥0.2 rad. |
| Y1 low-speed/high-steer authority loss | Fitted equilibrium yaw response at 2.5/3.0/3.5 m/s plus a hold-phase first-order response; support is 2.5–3.5 m/s, 0.30–0.42 rad, low steering rate. Narrow support-gated surface enabled in default MPC; optimizer overlay is wired. | Two independent whole-run captures: yaw RMSE 1.872→0.211 rad/s (r04) and 1.866→0.173 (r05); zero wrong-direction predictions. P0 screen had zero active samples. High-steer live treatment collided at 7.573 s versus 7.502 s for control; no lap-time gain demonstrated. | Does not support turn-in/unwind tau, below-0.30-rad response, recursive full-lap prediction, or broad practice benefit. |
| Y2 moderate/high-speed, low/moderate-steer yaw | Existing high-speed response atlas and race-domain swerve captures; a practice pre-impact observation near 6.0–6.1 m/s and 0.13 rad had predicted yaw about 2.12 rad/s versus measured about 1.4 rad/s. | Diagnostic mismatch evidence; no Y2 specialist or untouched candidate score. | One failure point is not a boundary or a response surface. |
| Y3 reversal/unwind | A continuous steering training capture records 2,022 samples, with marginal observed ranges u=0–7.7801 m/s, physical steering ±0.5236 rad, yaw rate −1.9513–1.9511 rad/s, throttle feedback 0–0.5. Separate phase-labeled high-steer turn-in/unwind captures have whole-run train and validation cohorts. | Train and validation data exist; no Y3 specialist fit. | Marginal extrema are not joint support; no response-derived Y3 boundary is published. |
| Y4 high-speed/high-steer | Open-plane only: accepted train r02 and r03 plus validation r01/r02/r03; one training run includes 695 samples at 7–9 m/s and |steering|≥0.30 rad (326 at ≥0.40). The two 7.5 m/s held-out captures used for the cited count each include 316 samples with u≥7.5 m/s and |steering|≥0.40, up to 8.072 m/s. | Diagnostic-only; no production integration. | Not practice transfer, odometry validation, or a safe capability limit. |
| Y5 very-high-speed/low-steer | Reset-isolated throttle-reduction pairs at 9.5 and 10.5 m/s with steering magnitudes 0.14/0.18/0.20 rad, and at 11.1 m/s with 0.12/0.16/0.20 rad, both turn directions; three independent 11.1 m/s replications. | Validation analyses are opened; diagnostic-only. | This intervention is not a yaw model, generalized steering response, or racing feasibility surface. |
| L0 ordinary longitudinal drive | The fine throttle sweep has 754 fixed-steering/throttle condition keys, with two reset-isolated repeats per key. | Existing development evidence; production feed-forward remains baseline. | Fixed-steering transitions alone do not identify a recursive plant; held-out-steering endpoint-speed fit failed. |
| L1 throttle pickup/wheelspin transient | Exact-condition paired response fit: 333 training pairs across 13 captures, 96 held-out validation pairs across four captures. A narrow 6.5 m/s, 0.08 rad, +0.08 throttle, 0.15 s ramp condition had matching training/validation action direction in two independent validation captures. | Offline response candidate; test/final-test splits for this artifact were not loaded. Runtime action shaper remains unpromoted. | It predicts a local step-minus-ramp contrast, not absolute motion; acceleration and lap-time benefit are not established. |
| L2 high-steer throttle action cost | Existing matched interventions span 4.5/6.5/7.5/8.0 m/s, both turn signs, multiple throttle increments/ramp durations, and high-steer 0.30/0.42 rad cells. At 8 m/s, +0.08 and a 0.30 s ramp, the acceleration contrast changes sign between 0.30 and 0.42 rad. | Offline action-table evidence only. | The sign change brackets observed conditions; it does not justify a midpoint threshold or universal slew policy. |
| L3 braking/release | Fourteen exact regression fixtures (seven train, seven validation) include active-brake and release events; observed pre-brake speeds are 7.0139–11.0741 m/s and commanded steering spans −0.14 to +0.14 rad. | Dedicated train/validation fixtures; replay/scoring gate remains pending. Keep braking separate. | Existing fixtures do not yet certify a learned braking model. Zero throttle is active braking, not coast. |
| L4 frontier throttle reduction | Same frontier cohorts as Y5; per-cell/run estimates are retained, including the one invalid packet-gap pair excluded from the 53-pair aggregate and separate 11.1 m/s replication. | Diagnostic-only; no runtime policy. | Pooled effects must not erase turn/steering dependence or be treated as a universal control law. |

The strongest already-implemented local response result is L1. Its exact-cell
lookup scores held-out whole captures at 0.01375 m/s RMSE for the paired
wheel/body mismatch effect versus 0.13327 m/s for the coarse baseline; for the
early 0.10–0.35 s window the mismatch RMSE is 0.04174 versus 0.20101 m/s, and
the wheel-slip-proxy RMSE is 0.00603 versus 0.02542. Integrated acceleration
RMSE is 0.06394 versus 0.44399 m/s², but its run-level gain is not confirmed.
This is concrete evidence that the captured local response can be predicted
more usefully than by the coarse baseline. It still predicts a *difference
between two actions*, not the car's absolute movement, and practice lap-time
transfer has not improved.

## Measured response transitions — candidate evidence, not guards

- At 10.5 m/s the earlier steering-response summary reports median yaw rate
  near 1.00 rad/s at 0.04 rad, 0.81 at 0.08 rad, and about 0.56 at 0.12 rad,
  then approximately 0.56–0.58 through 0.20 rad. This is a measured,
  non-monotonic response change at that speed anchor. It is not a general
  steering cutoff; cross-speed and phase-conditioned confirmation is still
  required.
- At 8 m/s, +0.08 throttle and a 0.30 s ramp, the body-acceleration
  step-minus-ramp effect changes sign between the measured 0.30 and 0.42 rad
  steering conditions. This is an observed bracket, not a continuous steering
  threshold.
- The low-speed 0.30–0.40 rad practice mismatch and the 6.0–6.1 m/s, 0.13 rad
  yaw mismatch justify keeping Y1 and Y2 distinct. The available report does
  not locate their numeric boundaries.

No midpoint, fixed rectangular speed/steering bins, global lateral-acceleration
cap, or capability claim is introduced by this atlas. Actual runtime routing
remains unchanged.

## Y1 lower steering edge and temporal response

The response below 0.30 rad is not a constant plateau. Existing training-only
steady-response data are speed-dependent and non-monotone: near 4 m/s the
absolute yaw response rises from about 1.76 rad/s at 0.15 rad to 2.24 at
0.20 rad, then falls to about 1.23 at 0.21 rad. The independent r04/r05
captures have support at 0.15 and 0.30/0.42 rad, but essentially no samples in
the 0.20–0.25 interval. That gap prevents validating a lower-edge fit.

The runtime candidate therefore preserves the legacy model through 0.29 rad,
uses a narrow smooth 0.29–0.30 continuity taper, and is full-weight only from
0.30–0.42 rad (with an outer fade to 0.45). The taper is a support guard, not a
claim that the measured response is constant or learned below 0.30 rad.

Inside the measured low-speed/high-steer support, the candidate uses the
first-order recurrence
`r_next = r_eq(u, steering) + (r - r_eq) exp(-dt/tau)`. The hold-phase tau
is 0.1293 s and is gated to near-steady steering. Turn-in and unwind fitted
taus (0.071 s and 0.401 s) have only 52 and 27 training samples respectively,
so runtime keeps the legacy transient response for those phases. A
step-response-like temporal model is suitable in this local form; a hard
step in steering or a single constant response below 0.30 rad is not supported.

## Odometry source-order replay finding

The two clean P0 bags were reconstructed in bag receipt order from left
encoder, right encoder, and IMU messages. `practice_r0_current_safe_r01`
contains 3,092 complete, source-monotone packets and no earlier partial packet
being passed. `practice_r0_current_safe_r02` contains 3,060 complete packets
and one right-encoder-only startup packet from before left encoder/IMU
recording began. All 3,060 emitted complete packets remain source-monotone;
the startup partial never completes late. Thus strict-drop and bounded-reorder
variants would produce the same integration sequence on these P0 captures,
apart from dropping that inert startup fragment. No source-order defect or
P0 odometry gain is demonstrated, so the branch is closed for this track and
no live odometry A/B is warranted by these bags.

## Provenance and split handling

The detailed run IDs and local artifact paths are in the YAML and JSON. Key
source families include the 2026-10-02 steering-frontier captures, 2026-10-04
high-steer transient train/validation captures, 2026-10-06/07 swerve and
throttle-rate captures, the dedicated braking fixtures, and the two P0
baseline sessions. Saved reports distinguish simulator-truth labels from
sensor/command inputs; truth remains development-side scoring only.

One archive needs an explicit integrity note:

`replacement_teacher_dataset_v2_20261004/openplane_dynamics.npz` contains
train, validation, test, and final-test run labels in shared arrays. During
schema inspection, reading shapes materialized the NPZ arrays. No test-row
values were inspected, filtered, or scored, but the archive cannot be claimed
as a sealed blind holdout after that access. The atlas excludes that archive's
held-out rows from its evidence claims. The separately saved reports and
artifacts cited above remain the source for the listed results.

## Promotion and next-step boundary

Y1 passes its measured whole-run one-step validation and C/Python/CasADi parity
checks, and is now enabled in the default MPC only behind its measured support
gates. That integration is not a lap-time result: the P0 screen never entered
the gate, while the high-steer treatment and control both collided at nearly
the same time. On the Y1-disabled control bag, only three samples were
materially active and improved active-sample yaw RMSE; the whole pre-impact
run regressed from 0.445 to 0.542 rad/s. The treatment bag cannot be used for
the replay A/B because its logged baseline already includes Y1. See
[`REGIME_SPECIFIC_LIMIT_PUSH_PROGRESS_20261007.md`](REGIME_SPECIFIC_LIMIT_PUSH_PROGRESS_20261007.md)
and the collision timelines in
`live_runs/racing_candidate_failure_atlas_y1_20261007/`.

The optimizer's partial vehicle-model overlay is now consumed by
`run_autodrive_mintime.py`, and its self-test confirms that Y1 is loaded. The
optimizer's existing scalar lateral envelope has not been revised here, and
no new trajectory has yet been scored from the Y1 overlay. In parallel, the
handoff's next performance package is a small sector speed-schedule change on
frozen P0 geometry, scored by measured sector time, exit speed, tracking, wall
margin, MPC rejects, and wheel mismatch—not the schedule proxy.

No new open-plane capture is justified by the current evidence. Test/final-test
partitions remain unopened.
