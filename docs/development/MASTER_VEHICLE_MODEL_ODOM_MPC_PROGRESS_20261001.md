# Master vehicle model / odometry / MPC handoff progress

**Status:** active; not complete.  
**Scope:** the 2026-10-01 master handoff, followed in its listed order, within
the validated 0–12 m/s racing domain.  
**Current checkout:** `2d391cdb4a5269646f0e0ebb810ccf434d9d366c` on `main`.  
**Handoff checksum and frozen artifact checksums:**
[`baseline_registry.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/baseline_registry.json).

This document is the running record of work, tests, outcomes, and remaining
steps. A checked step means its stated evidence exists; it does not imply a
later product gate passed. Do not integrate learned plant/observer/MPC code
into production until its own gates pass. Simulator truth remains offline-only.

## Operating rules retained from the handoff

- Optimize the race model and planner only over 0–12 m/s; treat higher-speed
  data as diagnostics, never as dominant training or optimization evidence.
- Keep whole-run splits immutable. The stored `test` and `final_test` splits
  remain unscored until architecture and hyperparameters are frozen.
- Preserve command semantics: positive throttle is drive, zero throttle is
  active-brake/idle-brake behavior, and negative throttle is outside the
  production contract. No fictitious coast input.
- Keep body motion and rear-wheel rotation as separate dynamic states. Rear
  wheel speed is not body speed during drive wheelspin or active braking.
- Use the existing pinned practice simulator and established Xvfb `-batchmode`
  launcher; no GUI and no `-no-graphics` mode. Do not alter simulator physics,
  production odometry/localization/MPC, or competition topic policy for model
  convenience.
- New race validation runs are validation-only from capture time. Failed
  whole-run gates are not repaired by bridging gaps or weakening thresholds.
- MPC rejection counts are diagnostics; plant quality is judged first by
  actual prediction errors and transfer, then closed-loop safety/performance.
- The primary acceptance target is a recursively accurate offline simulator
  over the measured feasible speed/steering envelope, so raceline generation
  can exploit that envelope. Tracking the existing conservative line or
  avoiding a crash is not a substitute for this plant-accuracy gate.

## Frozen baseline (WP0)

The schema-8 race view has 657,621 rows and 3,761 continuous sequence pieces:
490,450 train rows from 12 runs, 85,250 validation rows from 5 runs, 40,958
test rows, and 40,963 final-test rows. The dataset, split manifest, response
atlas, candidate checkpoints, production model files, test files, simulator
image digests, and prior practice-capture disposition are fingerprinted in the
baseline registry. Test/final-test have not been scored for this task.

The earlier three-lap captures remain **preliminary evaluation only** and are
not substitutes for the fresh six-lap evidence. Their frozen-model scores were
not used for training or checkpoint choice.

The working tree was already dirty before this handoff work. Those pre-existing
changes are listed in the registry and are being preserved. No simulator or
Docker container was active when the registry was created.

## Ordered execution plan (WP0–WP23)

| WP | Required work | Status / evidence |
|---:|---|---|
| 0 | Freeze git, checkpoints, race dataset/splits, atlas, branch harness | **Done.** Registry records paths and SHA-256 checksums; 26 referenced paths plus model sidecars verified; checkout matches handoff HEAD. |
| 1 | Two fresh, strict-gated practice validation captures, unchanged stack, <=6 laps | **Done.** `practice_model_validation_r02` and replacement `r03` each passed the exact 0→6 gate; both completed six laps with zero collisions/faults, contiguous aligned packet samples, complete sensor/GT/odom labels, and 40 Hz streams. `r01` was rejected because recording began at lap count 2 and is retained as a failed-capture audit. The two admitted bags are immutable validation-only inputs. |
| 2 | Deterministic, common-start practice benchmark stratified by phase/regime/speed | **Done.** Frozen `practice_transfer_benchmark_v1.json` has 135 common starts across two whole-run validation captures, with empty strata kept unsupported. |
| 3 | Score historical GRU, lead RSSM, direct predictor, production MPC on common starts | **Partial.** Historical GRU, direct predictor, lead RSSM, high-steer 2 s RSSM, high-steer 5 s RSSM, and the first 3D residual teacher have been scored on the same 135 starts / two independent practice captures. Production MPC0 is not yet included. Candidate transfer findings are below. |
| 4 | Freeze replicated braking/wheel regression fixtures | **Fixtures built; model replay gate pending.** `braking_wheel_regression_fixtures_v1.json` contains 14 exact 25 ms train/validation events, including straight and moderate-steering braking and release. Builder checks and targeted unit tests pass. Each plant still needs to be replayed/scored against them. |
| 5 | Simplify/refit stable race-domain nominal grey-box | **Not accepted.** The available four-wheel fit has a nonphysical free-run failure and cannot be the nominal model yet; details below. |
| 6 | Train hybrid nominal + 2 s history/latent + residual teacher | **Prototype attempted; required hybrid not yet implemented/accepted.** The 3D residual teacher has explicit rigid-body integration and balanced recursive training but lacks an identified, stable nominal tire/wheel-force model. It fails practice transfer. |
| 7 | Compare hybrid vs RSSM on identical practice starts and fixtures | **In progress.** The available candidates are compared on identical practice windows and open-plane high-steer probes; neither the 3D teacher nor the longer-rollout RSSM passes promotion. Braking-fixture model replay is pending. |
| 8 | Five-member winner ensemble and calibrated support/error | **Not started.** |
| 9 | Run serious plant candidates in synthetic-sensor → production odom → localization surrogate → MPC branches | **Partial infrastructure/evidence.** H128/H256 branches already exist at 0.5/1/2 s on preliminary runs; starts/regimes and accepted capture gates are incomplete. |
| 10 | Track boundary, heading-aware footprint clearance, known safe/collision tests | **Pending verification and implementation.** |
| 11 | Historical configuration ranking, including known safe/unsafe and faster/slower cases | **Pending.** |
| 12 | Legal-sensor longitudinal production-odom residual/gate baseline | **Not started in the handoff form.** Production odom remains authoritative. |
| 13 | High-capacity legal-sensor observer upper bound vs residual observer | **Prior observer research exists; required paired fresh validation is pending.** |
| 14 | MHE prototype after residual baseline | **Not started.** |
| 15 | Compare production/residual/MHE dead reckoning with AMCL disabled and sparse corrections | **Not started.** |
| 16 | Freeze and score production MPC model as MPC0 | **Pending.** |
| 17 | Distill/test nominal-residual, bounded scheduling, LPV, optional latent MPC students | **Not started.** |
| 18 | Student Jacobian, runtime, solver validation | **Not started.** |
| 19 | Paired MPC prediction-model closed-loop A/B | **Partial branch harness exists, but no accepted student comparison.** |
| 20 | Real-simulator candidate shadow-prediction validation | **Not started.** |
| 21 | Supported empirical/teacher planner envelope only over 0–12 m/s | **Partial.** Race-domain response atlas exists; uncertainty/support and accepted-teacher checks remain. |
| 22 | Offline grouped MPC-weight optimization, only after historical ranking passes | **Blocked by earlier gates; do not start yet.** |
| 23 | Speed-profile then raceline geometry optimization, only after validated envelope | **Blocked by earlier gates; do not start yet.** |

## Product architecture and acceptance status

### A. Offline plant (A0–A12; Gates A–F)

- **A0 freeze:** done in the registry.
- **A1 fresh transfer before architecture claims:** pending strict-gated runs;
  current two runs are preliminary only.
- **A2–A4 hybrid nominal + latent residual:** pending. Target residuals are
  `Δax`, `Δay`, `Δyaw_accel`, and left/right `Δwheel_accel`; initialize latent
  size 32 and compare 16/64 only if indicated.
- **A5 effective-parameter/DynSSM branch:** pending; bounded, data-supported
  parameters only.
- **A6–A9 training:** pending. Use 0.25/0.75/2 s primary losses, 5 s
  secondary, family→run→condition→window sampling, and 0–5/5–7/7–9/9–12 m/s
  and slip-balanced influence.
- **A10–A11 robustness/ensemble:** pending. Calibrate support and ensemble
  spread against held-out run error; do not call uncalibrated spread uncertainty.
- **A12 API:** existing offline plant API is present; verify every accepted
  candidate obeys the command-only future-input contract.
- **Gates A–F:** local dynamics, replicated braking, fresh practice transfer,
  production-stack branches, collision geometry, and historical ranking have
  not all passed. No full-lap safety claim is authorized yet.

### B. Odometry/observer (B0–B13)

Production odometry is preserved. The physical premise—wheel rotation can
over-read body speed under drive and under-read it during active braking—is
supported by existing race-domain captures. The handoff-form residual `Δu` plus
filtered gate, high-capacity legal-sensor upper bound, MHE, AMCL-disabled/sparse
correction replay, and shadow-mode promotion gates remain to be executed.
Truth is permitted only as an offline label/evaluation target, never as an
observer input at runtime.

### C. MPC model (C0–C9)

The production MPC source/config is fingerprinted. Its model has not yet been
scored as MPC0 on the common race-domain and fresh practice starts. Compact
students, derivative/runtime checks, same-plant/same-weight A/B, and real
shadow-prediction runs remain pending. The full recurrent teacher is not to be
placed directly in the RTI solver.

## Existing numerical evidence (not a completion claim)

Frozen RSSM h128 vs h256 metrics on two unseen practice runs are summarized in
[`RACE_DOMAIN_UNSEEN_MODEL_VALIDATION_20261001.md`](RACE_DOMAIN_UNSEEN_MODEL_VALIDATION_20261001.md).
They show promising short-horizon transfer and lower 2 s position/heading
error, but are based on two drives, speeds only to about 7.8 m/s, and are not
full-lap evaluation. The production stack itself completed three laps without
collision in those sessions; the model was not integrated and did not cause
that result.

## Fresh capture disposition

The six-lap path uses the handoff's exact 0→6 requirement and does not weaken
the old 0→12 evaluator. `validate_practice_capture.py` requires zero collisions
and timing faults, continuous bridge packet IDs, 40 Hz streams, complete legal
sensor and simulator/odom labels, and no reset during the scored interval. It
excludes only the measured startup prefix before packet identity/causal encoder
history is available (2 odometry samples in r02; 1 sensor sample in r03),
before the car has begun moving. The recorder is stopped before the simulator,
so teardown cannot contaminate the active-interval timing gate.

Both successful runs used the pinned practice simulator and unchanged
production development stack, with the pre-registered practice map and
speed-headroom trajectory. r02 scored 1,515 contiguous packet-aligned samples
over 37.94 s; r03 scored 1,516 over 37.96 s. Both had zero collision/timing
faults and 100% sensor/GT/odom completeness after the startup prefix. Required
streams measured 39.979–40.000 Hz. Peak speed was 7.956 and 7.963 m/s,
respectively; steering feedback peaked at 0.283 rad. These are clean transfer
captures, not evidence above their measured speed/steering coverage.

The first attempted `r01` recording started after lap count had already reached
2 (transitions `[2,3,4,5,6]`); it was rejected and preserved. No training or
checkpoint selection used any fresh practice capture.

No independent subagent audit has been requested yet: the handoff says to do
that only after all implementation and validation steps are complete.

## New high-steering offline-plant evidence (2026-10-02)

This work targets the primary product: a recursively accurate offline plant
over the car's empirically feasible speed/steering envelope. A conservative
raceline or a crash-free live lap is not the plant acceptance criterion.

Three fresh Explore open-plane captures were added under the pre-registered
`isolated_highsteer_75_long` profile: `openplane_highsteer_75_train_r02` is
train-only; `validation_r01` and `validation_r02` are held out from both fitting
and checkpoint selection. All three completed 19/19 phases with zero collision
and bridge-timing faults and approximately 39.985 Hz aligned odometry, encoder,
steering, throttle, IMU, and packet streams. The earlier `train_r01` pilot was
aborted for a 315 ms timing gap and remains excluded.

The two independent held-out captures cover 17 scheduled steering conditions
at approximately 7.51–7.55 m/s, including both signs through ±0.5236 rad. On
the full-lock hold, median measured yaw rate is 1.0107 rad/s, curvature is
0.1340 m⁻¹, lateral IMU acceleration is 7.627 m/s², and peak roll proxy is
0.0772 rad. Mirrored conditions reproduce closely across runs. The empirical
steady response is nonlinear: an odd cubic fit of curvature against actual
steering, trained on validation run r01 and checked on r02, reduced grid RMSE
from 0.0212 to 0.0100 m⁻¹ versus a through-origin linear fit. This is a
single-speed response description, not a full plant or a proof of individual
tire forces. Raw per-condition measurements are in
[`highsteer_75_response_atlas_20261002.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/highsteer_75_heldout_validation_race_20261002/highsteer_75_response_atlas_20261002.json).

The frozen h256/z32, 2-second RSSM performs very poorly on these new
command-only free runs. Across the 17 probes and two independent runs, mean
per-probe 5.5-second free-run position RMSE is 9.05 m, heading RMSE 1.385 rad,
and speed RMSE 0.322 m/s. A targeted h256/z32 RSSM trained with the added
high-steering run (2-second training rollout, best step 2000) reduces those
means to 2.16 m, 0.261 rad, and 0.089 m/s, respectively. Its 2-second joint
position RMSE in the 7–9 m/s, 0.4–0.524 rad cell falls from 6.13 m to 1.12 m.
That is substantial improvement in the exact target regime, but still far too
large to call the offline simulator accurate over a lap. Its frozen ordinary
validation selection score is slightly worse than the original (0.0802 vs
0.0772), and low-steering short-horizon errors also regress; do not promote it
as a global teacher or integrate it into MPC/odometry.

The prescribed symmetry-augmentation trial was run once and rejected. It
produced 6.14 m mean per-probe free-run position RMSE on the same held-out
captures, substantially worse than the targeted non-augmented model (2.16 m).
The option and physical-unit mirror tests remain available; this checkpoint is
not retained as a candidate. A 5-second-rollout RSSM was also completed and
rejected: it did not improve either the 5.5-second high-steering free run or
practice transfer.

An extractor discrepancy was isolated before training: re-extracting all 19
accepted historical bags with the current exporter changes row counts in four
legacy runs, including two that previously had only 98–147 rows on re-export.
The frozen 669,042-row source was therefore preserved byte-for-byte and only
the 5,107 rows from the new training capture were appended. The resulting
662,728-row race view reproduces every one of the original 43 schema-8 arrays
exactly as a prefix; only new train rows/sequences are appended. A one-run
append utility now enforces split, quality, schema, and prefix-preservation
checks. The new high-steer run is correctly assigned to the
`steering_transition_slew` sampling family (15% family mass; two train runs,
23 conditions). The mistaken re-extracted archive and intermediate family
views are not used for training or scoring.

Important remaining coverage gap: the new high-steering evidence is at 7.5
m/s. Existing clean training/validation data above 9 m/s has little steering
coverage, and no held-out captures establish the high-speed feasible steering
boundary. Do not extrapolate full lock to 9–12 m/s. The offline plant is not
yet suitable for aggressive raceline optimization.

## 2026-10-02 follow-up — plant is the target, not conservative-line tracking

The optimization objective is a recursively useful offline simulator across
the car's *empirically feasible* 0–12 m/s speed/steering envelope. Reproducing
the current conservative raceline, or merely completing a collision-free live
lap, does not pass the plant gate. The simulator must predict body motion and
wheel state well enough that a planner can explore faster, higher-curvature
lines without relying on unsupported or physically impossible combinations.

### Full-range support audit

The current augmented race-domain training view was audited without changing
its split or source rows:

- 671,932 samples, 3,949 sequence pieces, 22 independent runs; source SHA-256
  `ef11c955e45fe95bf9ec264beaa99a13158e7920d583c6b3dee2e97e6f40a542`.
- At 7–9 m/s and `|steering| >= 0.3 rad`, the training split has only three
  independent runs. Separate held-out high-steering captures add two more
  runs, but they are steady 7.5 m/s probes, not varied-speed cornering traces.
- At 9–10 m/s, training support is mainly `|steering| < 0.2 rad`; cells above
  0.2 rad have no support. At 10–12 m/s, the observed data is almost entirely
  `|steering| < 0.2 rad`; there is no evidence for high-speed full-lock use.
- The corresponding validation support is similarly sparse in high-steering
  7–9 m/s cells and absent above 9 m/s for `|steering| >= 0.2 rad`.

The machine-readable audit is
[`race_domain_coverage_full_range_20261002.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/race_domain_coverage_full_range_20261002.json).
Its counts distinguish independent runs from correlated 40 Hz rows. This is
the evidence for the current coverage limit; row volume alone is not adequate.

### New recursive candidate comparisons

The high-steering-focused 2 s RSSM (h256/z32, trained with one added
high-steering run) and its 5 s-rollout variant were scored on the same frozen
135 practice windows from both unseen practice runs. The high-steering-focused
2 s candidate improves the short practice transition metric at 0.25 s
(`u` RMSE 0.126 vs 0.144 m/s; yaw-rate RMSE 0.107 vs 0.136 rad/s) and at
0.5 s (`u` 0.166 vs 0.188 m/s; yaw-rate 0.148 vs 0.168 rad/s). Its lateral
velocity error is worse, however, and by 2 s it regresses: `u` 0.275 vs
0.222 m/s, `v` 0.050 vs 0.035 m/s, yaw rate 0.265 vs 0.220 rad/s, heading
0.165 vs 0.120 rad, and XY component RMSE 0.416/0.175 vs 0.328/0.161 m.
The brief short-horizon gain does not establish a good multi-step offline
plant.

The 5 s-rollout variant is worse than the frozen lead RSSM on the practice
benchmark at 0.75–2 s. On the separate held-out 7.5 m/s high-steering probes,
its mean 5.5 s per-probe position RMSE is 2.45 m (versus 2.10 m for the
high-steer-focused 2 s candidate); speed RMSE is 0.098 versus 0.087 m/s.
Longer rollout training by itself did not close the gap and this branch is not
promoted.

The first 3D rigid-body residual teacher also fails broad practice transfer:
the frozen 135-window benchmark's aggregate position error grows to about
1.9 m at 1 s and about 7.7 m by 2 s. Its high-steer probe score is much better
at 1 s than at 5.5 s, showing that short local accuracy is not a substitute
for recursive full-run accuracy.

Reports:

- [`highsteer_75_candidate_eval_20261002/practice_transfer_benchmark_score.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/highsteer_75_candidate_eval_20261002/practice_transfer_benchmark_score.json)
- [`highsteer_75_5s_candidate_eval_20261002/practice_transfer_benchmark_score.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/highsteer_75_5s_candidate_eval_20261002/practice_transfer_benchmark_score.json)
- [`hybrid_teacher_3d_h2_f5_latent32_balanced_20261002/practice_transfer_benchmark_score.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/hybrid_teacher_3d_h2_f5_latent32_balanced_20261002/practice_transfer_benchmark_score.json)

### Why the current grey-box cannot be the nominal plant

The best current four-wheel fit has motor torque scale 0.506, wheel damping
0.00096 N·m·s, idle-brake torque 0.0495 N·m, and approximately 50/50 front
drive. In an unseen 7.5 m/s straight, constant-throttle replay, its predicted
rear-wheel speed falls by about 1.25 m/s in the first 25 ms and predicted body
speed decays toward about 1.2 m/s over 5 s, while simulator truth remains near
7.53 m/s. This violates the measured drive/wheel-speed balance, so adding a
learned residual on top without fixing the nominal would hide a structural
error. The checkpoint is rejected as a plant/nominal, and wheel radius or
simulator physics was not changed.

### Model/data conclusion and next gate

These results do not show that the car is inherently unmodelable. They do show
that repeating the same constant-speed/constant-steering probes or increasing
optimizer steps is not resolving the missing behavior. The main remaining
evidence gap is independent, recursive command sequences that couple steering
transitions, throttle changes, body yaw/lateral motion, and wheel/body
decoupling through the measured feasible envelope—especially the 7.5–8.3 m/s
transition and higher-speed moderate-steering regions. Static open-plane
surfaces constrain steady response, while the practice benchmark currently
has neither full-lock steering nor speeds above about 8 m/s.

Next plant work remains handoff WP5–WP7: refit a stable simplified nominal,
then add the 2 s latent-history residual to nominal body and wheel
accelerations; score exact practice starts and frozen braking fixtures. New
captures, if required by the support audit, must be independent whole runs
with dynamic command sequences spanning supported speed/steering cells—not a
larger pile of repeated identical holds. No candidate is yet accepted for
offline raceline optimization, MPC, or odometry.

## 2026-10-02 objective correction and latest teacher result

The deliverable is an accurate, recursive offline plant from which a new
raceline optimizer can exploit the vehicle's *full feasible* speed/steering
envelope. Demonstrating that the current conservative raceline can be tracked
without a crash is not progress toward this deliverable and is not an
acceptance criterion. Feasibility still matters to the planner: impossible
speed/steering combinations must be excluded, but the current test schedule is
not itself evidence of the vehicle's true limits.

The data gap is more specific than “no high-speed data.” Existing accepted
captures include speeds through approximately 11.1 m/s and moderate steering
there, plus full-lock steering at approximately 7.5 m/s. What remains weak is
independent, dynamic coverage of the *coupled* speed/steering boundary and the
recursive teacher's ability to predict state and pose under those transitions.
Counts of 40 Hz rows do not substitute for distinct whole-run trajectories.

The pose-aware RSSM experiment implemented deterministic-prior training and
added integrated pose loss/checkpoint selection. On one held-out 48.6 s
open-plane replay, it reduced radial position RMSE from 66.9 m to 20.4 m and
endpoint drift from 103.9 m to 20.6 m versus the dynamic-steering RSSM. This
is a real improvement on that trace, but remains unusable as a lap simulator;
it also worsened body-state RMSE (speed 0.117 to 0.417 m/s, yaw-rate 0.401 to
0.976 rad/s) and was worse than the lead h256 RSSM on the independent practice
transfer benchmark through 2 s. It is therefore a rejected comparator, not a
new teacher. The full replay's error begins growing within the first few
seconds, so its endpoint gain does not establish stable long-horizon dynamics.

The suspected rear-axle/COM pose-origin mismatch was checked independently:
integrating held-out simulator rear-axle velocity and yaw rate reproduces
simulator pose to 0.20 m full-run RMSE. That coordinate mapping is not the
source of the large learned-model drift.

The next plant work remains the handoff's WP5–WP7 sequence: obtain a stable,
reduced nominal; add the specified history/latent residual over generalized
body and wheel accelerations; then compare it against the lead RSSM on identical
whole-run and practice windows. Do not start broad architecture/weight sweeps
or raceline optimization while no teacher meets recursive-accuracy gates. Any
additional capture must target a demonstrated support gap in coupled,
feasible speed/steering transitions rather than repeat static holds or merely
prove another collision-free lap.

### 2026-10-02 high-speed frontier: plane-edge artifact isolated

The first independent high-speed frontier capture (`validation_r01`) ended at
10.5 m/s and +0.18 rad with an apparent 10.3-degree tilt. Inspection of the
packet-level simulator truth showed this was not a vehicle-response outlier:
the car had traveled to world x≈500 m, the edge of the finite Explore plane.
At the final packets its z-velocity changed from about +0.18 to −1.48 m/s and
vertical acceleration reached −78.6 m/s²; the collision counter remained
zero. Yaw response before leaving the plane matched the earlier training run.
The run is therefore rejected as a whole-run validation capture; its terminal
fall samples are not vehicle-model data. This exposed a design flaw in the
previous continuous sweep: it matched velocity before probes but never reset
the vehicle's position.

The earlier `train_r02` archive remains a clean stream candidate but is not an
independent validation set. Its simulator position reached about 363 m from
the map origin while z stayed within 0.0545–0.0596 m, so it had not yet
reached the plane edge. Do not treat its within-run rows as independent
replicates.

The frontier experiment now starts each steering condition with the built-in
simulator reset, then re-approaches the target speed and waits for a matched
near-straight state before measuring. The reset harness learns the actual
post-reset spawn coordinates from the first successful reset (rather than
assuming the odometry/world origin is zero) and checks later resets against
that point within 0.25 m. The 8-degree tilt and 11.9 m/s hard-speed aborts are
unchanged. Reset edges are explicitly recorded in the bag so offline sequence
construction can break at every reset.

The corrected independent frontier capture
(`openplane_steering_frontier_validation_r03`, seed 17047) completed all 61
conditions / 122 approach-and-probe phases. It had no abort, collision,
timing fault, packet-alignment loss, or quality failure. The recorded command
stream was 39.87 Hz; packet alignment was 17051/17051. There were exactly 61
reset-separated sequences. Every reset returned to the same measured odometry
spawn `[-0.1584, 0.0]` m (0.1 mm peak-to-peak x variation), and simulator truth
remained on the plane (maximum radial displacement 57.94 m; z=0.0537–0.0600 m).
The run used the interim zero-origin tolerance; the later source refinement
that learns the first reset's actual position has not yet had its own complete
run. The exported validation archive has 16,781 samples and 61 sequences.

Across paired signed steering/speed cells in the earlier clean `train_r02`
and this unseen `validation_r03`, median vehicle response agrees closely:
yaw-rate RMSE 0.0097 rad/s and signed lateral-acceleration RMSE 0.090 m/s².
The steering feedback matches the requested angles through ±0.20 rad, so the
observed response roll-off is not an actuator-command tracking artifact. At
10.5 m/s, median yaw response is about 1.00 rad/s at 0.04 rad, 0.81 at 0.08,
0.56 at 0.12, then roughly 0.56–0.58 through 0.20 rad. This is strong
two-capture repeatability evidence for a non-monotonic high-speed response,
not yet an independently replicated physical limit or a full feasible
steering envelope. It is also a quasi-steady response map, not a recursive
plant model.

### Frozen teacher comparison on the new unseen run

The lead race-domain RSSM and high-steering-focused RSSM were scored on the
same 61 reset-separated sequences from `validation_r03`, using 1.975 s of
initial history and then commands only. No future truth or measured feedback
was fed back. The exact outputs are
[`rssm_h256_baseline_free_run.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/steering_frontier_validation_r03_20261002/rssm_h256_baseline_free_run.json)
and
[`rssm_highsteer_candidate_free_run.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/steering_frontier_validation_r03_20261002/rssm_highsteer_candidate_free_run.json).

Across the 61 condition windows (descriptive within-run medians, not 61
independent run replicates), baseline vs high-steering-candidate errors were:

| Horizon | Model | Position error | Speed error | Heading error |
|---|---|---:|---:|---:|
| 0.25 s | Lead RSSM | 0.008 m | 0.042 m/s | 0.0085 rad |
| 0.25 s | High-steering RSSM | 0.037 m | 0.266 m/s | 0.0016 rad |
| 0.75 s | Lead RSSM | 0.016 m | 0.002 m/s | 0.010 rad |
| 0.75 s | High-steering RSSM | 0.256 m | 0.378 m/s | 0.085 rad |
| 2.0 s | Lead RSSM | 0.993 m | 0.093 m/s | 0.164 rad |
| 2.0 s | High-steering RSSM | 1.533 m | 0.195 m/s | 0.064 rad |

The high-steering candidate improves 2 s heading error but worsens the
position and speed error; it does not give a consistent whole-plant gain.
Across full available rollouts (4.6–5.25 s), median path position RMSE is
still 6.20 m for the lead and 5.32 m for the candidate, with median speed
RMSE 0.134 vs 0.235 m/s. At 5 s, only 19 of the 61 sequences are long enough;
their median point-position error is 18.43 m vs 13.59 m and their 90th
percentile is 27.68 m vs 31.18 m. That mixed result is not suitable for
raceline simulation. Neither checkpoint is promoted.

The validation archive is schema 7, which exposed a loader-version bug:
`_load_dataset` incorrectly required the run-local `condition_run_index`
catalog introduced in schema 8. The loader now validates that field only for
schema 8, while still checking schema-7 condition IDs and sequence/frame
alignment. A focused regression test passes. This changed no data or model.

The earlier `validation_r01` plane-edge fall and `validation_r02` reset-pause
timeout remain rejected as described above. The r03 original archive remains
unchanged and validation-only; it has not been appended to training. The new
scores reinforce the current handoff decision: continue WP5–WP7, but do not
select either current neural plant for raceline optimization. The plant is not
yet accurate enough for the user objective of constructing an aggressive
line from the full feasible 0–12 m/s envelope.

### WP5 root-cause checks: command alignment, parameter sensitivity, and hidden wheel state

The first four-wheel capture scorer used command rows starting one packet after
the trainer's `(state[k], command[k]) -> state[k+1]` convention, while the
nominal already includes an internal one-packet actuator delay. The scorer now
uses the trainer-aligned command sequence; its report is
[`greybox_highspeed_steering_free_run_command_aligned.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/steering_frontier_validation_r03_20261002/greybox_highspeed_steering_free_run_command_aligned.json).
Correcting this did not materially change the failure: across 61 reset-separated
conditions from the one held-out capture, macro-sequence position RMSE is
14.40 m, speed RMSE 7.28 m/s, and endpoint position error 26.35 m. The
command-index error is fixed, but it was not the cause of poor prediction.

A run-balanced local sensitivity analysis of the fitted grey-box used 11
independent training captures and 2 s command rollouts. Its normalized
parameter Jacobian has rank 20/20 and a smallest/largest singular-value ratio
of 0.479, with no parameter-sensitivity pair correlated above the tool's
0.95 cutoff. Throttle rate limit is the only near-inactive parameter: its
run-balanced sensitivity RMS is 0.00127 (bootstrap 95% interval 0–0.00381),
orders of magnitude below the other terms. This does not prove physical
parameters are individually true; it does show that another broad parameter
sweep is not the obvious route to fixing the forecast error. The report is
[`identifiability_train_runs.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/greybox_highspeed_steering_h0p5_20261002/identifiability_train_runs.json).

The specific hypothesis that unmeasured front-wheel speed explains the
high-throttle mismatch was tested on the untouched `validation_r03` run. For
each of its 61 reset-separated condition sequences, a 122-point grid over the
two front-wheel surface speeds (0–20 m/s) was scored only against the
preceding 0.5 s of body motion, rear encoders, and actuator feedback; then the
resulting latent state was carried into a command-only future rollout. This
uses truth labels only in the past-history fit and is an optimistic
identifiability diagnostic, not a deployable observer. The grid reduces the
median normalized history loss from 4.921 to 4.646, but 78/122 inferred wheel
values hit the 20 m/s search ceiling. More importantly, future prediction
regresses:

| Metric | Pure-roll initialization | History-fit front-wheel latent |
|---|---:|---:|
| Median 0.25 s position error | 0.108 m | 0.130 m |
| Median 0.25 s speed error | 1.00 m/s | 1.14 m/s |
| Median 2 s position error | 7.87 m | 7.99 m |
| Median 2 s speed error | 6.34 m/s | 6.36 m/s |
| Median full-rollout position RMSE | 14.76 m | 15.01 m |
| Median full-rollout speed RMSE | 7.29 m/s | 7.69 m/s |

The per-condition results are in
[`front_wheel_history_latent_grid.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/steering_frontier_validation_r03_20261002/front_wheel_history_latent_grid.json).
The large inferred speeds are not measurements of the front wheels. Their
frequent ceiling saturation plus worse future rollouts indicate that changing
these two hidden initial values cannot repair the current nominal force and
wheel dynamics. Do not build the next model around a literal front-wheel-speed
estimate on this evidence alone.

**WP5 conclusion:** the four-wheel model is numerically stable but rejected as
a useful race-domain nominal. Command alignment is corrected; the weak
inactive throttle-rate-limit term can be fixed/removed in the simplification,
but this alone cannot explain the large acceleration error. Per the handoff,
stop trying to recover exact Unity tire parameters. The required next plant
candidate is the WP6 effective hybrid: a simple stable nominal, a 2 s history
encoder/latent state, and learned residuals for `Δax`, `Δay`, `Δyaw_accel`, and
left/right rear-wheel acceleration. It must be judged on fresh whole-run
command-only rollouts across the supported speed/steering cells; it is not
ready for raceline optimization or production integration yet.
