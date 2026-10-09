# Full-spectrum yaw response progress — 2026-10-09

> Status note: this document is chronological. Later continuation sections
> supersede earlier “running”, proposed-test, and coverage statements. The
> latest measured coverage and active work are recorded at the end.

## Objective and admission rules

Build a simulator-GT-trained, sensor-only one-step yaw-rate predictor across the
car's measured 0–12 m/s and full steering range. Simulator truth is a label and
offline score only; it is never an input. A model score is `GT yaw rate at the
next 25 ms packet - current IMU yaw rate`, in rad/s. The current target is not
met: the best broad exact-two event model still has large reversal/unwind
outliers above 0.1 rad/s. No candidate from this work is integrated into
production odometry or MPC, and no simulator physics were changed.

For yaw-response training/scoring, command-to-actuator feedback must be exactly
two contiguous simulator packets apart. One- and three-packet cases are
excluded. Coverage is binned by *measured GT speed and measured steering*;
requested targets alone do not establish coverage. The full Cartesian
0–12 m/s × ±0.5 rad rectangle may contain physically unreachable states; those
must be identified from measurements, not filled with synthetic labels.

## Work completed

### Reset-isolated edge capture

The Explore simulator was run in the repository's established batch-mode
procedure. Capture:

`live_runs/openplane_yaw_exact_two_edge_gapfill_train_20261009_r01/run/run_0.db3`

- 180/180 phases and 60 reset-isolated signed steering conditions completed;
  no collision, timing fault, quality failure, or abort.
- 14,680 bridge packet samples; packet IDs were contiguous. During active
  intervals the rate was 39.962 Hz, median gap 24.950 ms, p95 25.864 ms, and
  maximum active gap 48.890 ms. The 60 intervals over 60 ms were reset
  boundaries, not packet-sequence losses.
- 227/240 steering transition events had exactly two contiguous packets.
  The remaining 13 were three-packet events and are excluded.
- The audited exact-two windows occupied 121/504 measured 0.5 m/s × 0.05 rad
  cells; 47 cells had at least 20 rows. This is an edge-only capture, not a
  full grid, and confirms why command-target combinations cannot be counted
  as measured coverage.
- Steering feedback reached each requested 0.05–0.50 rad magnitude. The
  high-speed targets did not mean the car held that speed through the response:
  exact-two windows around turns at 11.35/11.60 m/s targets were typically
  9.7–9.9 m/s. The complete phase briefly reached up to 11.60 m/s. Thus this
  capture adds low/high-edge transients but does not by itself fill every
  high-speed/high-steering measured-state cell.

### Reuse of the existing 1,508-phase throttle surface

Source dataset:
`live_runs/derived_dynamics_learning_20260928/throttle_surface_40hz_dataset_20260930/`

The command-to-feedback packet audit found 1,497 exact-two sequences, 7
one-packet sequences, and 4 three-packet sequences. The latter 11 were
excluded. Six training and three validation sequences with internal sensor
gaps were also excluded. Training used capture r04 (756 admitted sequences,
352,337 rows in the GT 0–12 m/s domain); capture r05 remained whole-run held
out (732 sequences, 340,830 input samples in-domain; 337,214 scored yaw rows).
Encoder surface speed came from causal encoder-position history; invalid direct
encoder-velocity fields and simulator truth were not model inputs.

The exact-two throttle-surface augmentation was evaluated with the same
58-feature contract on all corpora. The earlier accidental feature-layout
mismatch was caught before scoring and corrected; no result from that invalid
fit was retained.

| Held-out set | Baseline yaw RMSE | With throttle data | p95 absolute error | Max absolute error | Samples over 0.1 rad/s |
|---|---:|---:|---:|---:|---:|
| Existing yaw probes, 20 runs / 41,143 rows | 0.03474 | 0.03500 | 0.04252 → 0.04362 | 1.558 → 1.558 | 674 → 706 |
| Throttle continuation r05, 337,214 rows | 0.09758 | 0.02352 | 0.24395 → 0.03634 | 1.643 → 0.983 | 33,958 → 3,193 |

On the 20 yaw-probe runs, the paired run-macro RMSE delta was +0.000145
rad/s (95% bootstrap interval −0.000279 to +0.000538). This is not a broad
improvement and the candidate is not promoted globally. On the unseen throttle
continuation, the improvement is large, so the throttle captures are useful
for a distinct throttle-response domain. They do not resolve reversal: the
throttle sweep provided no exact-two reversal holdout rows; the yaw-probe
reversal maximum remains 1.558 rad/s. Preserve this model/data as a comparator
while the complete surface is collected.

Machine-readable report and predictions:
`live_runs/racing_model_diagnostics_20261008/yaw_throttle_surface_exact_two_augmentation/`

Per-bag measured-state coverage auditor:
`tools/racing/specialists/audit_yaw_full_spectrum_capture.py`; its edge-run
output is the `yaw_exact_two_coverage.json` file beside the bag.

## Full-spectrum captures and analysis now underway

The new `yaw_full_spectrum_grid` development profile is implemented in
`tools/open_plane_excitation.py` and `tools/run_open_plane_experiment.sh`.
Schedule validation confirmed:

- 480 randomized, reset-isolated conditions: 24 speed targets from 0.25 to
  11.75 m/s in 0.5 m/s increments × 10 steering magnitudes from 0.05 to
  0.50 rad in 0.05 rad increments × both steering signs.
- 1,440 phases total. Each condition has approach, settle, then turn-in,
  unwind, reversal, and return-to-center transitions.
- Profile-specific speed governor is 11.95 m/s; hard abort remains 12.0 m/s.
  Existing collision and tilt interlocks remain enabled. Reset occurs between
  every condition. No physics, production controller, odometry, or localization
  behavior is changed.
- Nominal schedule is 6,480 s; including the existing reset timeout budget,
  the required experiment timeout is 8,837 s. The launcher allows 12,000 s.

The first training capture is complete:

`live_runs/openplane_yaw_full_spectrum_grid_train_20261009_r01/run/run_0.db3`

- All 1,440 phases / 480 conditions completed; `aborted=False`, zero quality
  failures, zero timing faults, zero collisions.
- 106,368 packet samples; active interval rate 39.966 Hz; p50/p95/p99/max
  active gaps 24.934/25.885/26.249/49.299 ms; packet IDs are contiguous.
  Long intervals occur at reset boundaries, not as missing sequence IDs.
- Of 1,920 steering transition measurements, 1,789 were exactly two contiguous
  packets. The remaining 3 one-packet, 124 three-packet, and 4 four-packet
  responses are excluded from yaw fitting and scoring.
- Measured GT state spans 0.2296–11.7441 m/s and 0–0.50 rad absolute steering.
  Exact-two response windows occupy 466/504 0.5 m/s × 0.05 rad signed cells;
  407 cells have at least 20 rows. The 38 empty cells are concentrated at
  high speed/high steering, and weak cells cluster near crawl and the upper
  speed edge. These are coverage observations, not yet proof that every empty
  state is physically unreachable.
- The experiment command loop averaged 39.69 Hz. Recorder and bridge were
  healthy for the full capture.

An event-time check was made against raw steering-command samples. The atlas
commands are linear ramps: their response onset is at 0.00/0.70/1.30/2.10 s,
and their waypoint endpoints follow 0.20–0.25 s later. The existing event
parser uses ramp-start times, which the bag confirms; scoring against endpoint
times would miss already-started actuator responses. The raw bag remains
unchanged.

A separately seeded whole-run validation capture is currently running:

`live_runs/openplane_yaw_full_spectrum_grid_validation_20261009_r02/`

It uses the same 480-condition grid and now forces the built-in simulator reset
before the first condition as well as between conditions. It remains held out
from fitting. The capture uses no simulator physics or production runtime
changes.

A dedicated exact-two evaluator is implemented at
`tools/racing/specialists/evaluate_yaw_full_spectrum_exact_two.py`. It admits
only two-packet steering windows, uses causal sensors/commands as features, and
compares historical-training-only, full-spectrum-trained, causal-event, and
measured-speed/steering-local models. It reports per-speed/steering errors and
strata for turn event, throttle slew/gap, roll, wheel split, and wheel/body
mismatch. It opens no test/final-test arrays and writes top outliers for
diagnosis.

The evaluator now also consumes the exact-two portion of the existing
1,508-sequence throttle sweep under the same 58-feature/four-history-lag
contract. The verified source split is r04 training (348,600 usable rows in
756 contiguous exact-two sequences) and r05 held-out validation (337,214 rows
in 732 sequences). Packet-delay audit after adding an explicit intervening
packet-contiguity check: 1,497 two-packet sequences are contiguous; 7 one-packet
and 4 three-packet sequences are rejected. No noncontiguous two-packet
sequence was found. The evaluator compares historical-only, yaw-grid-only,
historical-plus-throttle, and all-data training, then scores across prior yaw
validation, throttle continuation, and the independent full-grid capture. It
does not treat the throttle sweep as evidence of steering-domain coverage.
This integration is implemented and schema-checked, but model fitting and
scoring remain pending until the current validation bag closes.

After both captures, empty measured-state cells will be classified as either
missing reachable data (requiring a targeted test) or empirically unsupported
states. Held-out residuals above 0.1 rad/s will then be traced by speed,
steering, turn-in/unwind/reversal, throttle slew, roll, wheel split, and
wheel/body-speed mismatch before any further model change.

## Current status

### Continuation note — sparse-cell run governor defect (2026-10-09)

The first targeted sparse-cell support capture,
`openplane_yaw_sparse_cell_support_train_20261009_r01`, was safely aborted at
phase 54/198 without a collision or timing fault. Its nine completed 0.25 m/s
conditions are retained as diagnostics but are not admitted into the new model
fit because the capture ended aborted. The next approach requested 11.75 m/s
and stalled at 11.282 m/s. Bag phase events confirmed the request was correctly
11.75 m/s; the experiment interlock was applying the generic 11.2 m/s governor
to this sparse-support profile. This is a test-harness bug, not evidence that
the car cannot reach the target. The independent high-speed envelope captures
had already reached approximately 11.68 m/s under their intended governor.

The harness now assigns the sparse-support profile the same 11.95 m/s governor
and 12.0 m/s hard cutoff as the full-grid/high-speed-envelope profiles. No
simulator physics or vehicle behavior changed. The evaluator's sparse training
and held-out run IDs were moved to the corrected captures
`openplane_yaw_sparse_cell_support_train_20261009_r02` and
`openplane_yaw_sparse_cell_support_validation_20261009_r03`; the aborted r01
bag remains excluded. Next: verify the patched schedule in the pinned API
container, run corrected r02, audit exact-two measured-state coverage, then
run independent r03 before fitting/scoring.

- Exact-two edge capture: complete and quality-checked.
- Existing throttle sweep: used; useful for its own held-out throttle domain,
  not a general yaw-model promotion.
- Full-spectrum training capture: complete and audited; no partial-bag model
  scores are treated as complete results.
- Full-spectrum validation capture: complete in Explore batch mode under seed
  `202610093`; it has not been used for fitting.
- It completed 1,440/1,440 phases and 480/480 conditions, with `aborted=False`,
  zero collisions, zero timing faults, and zero quality failures. The bag closed
  successfully at 441 MiB; command loop averaged 39.69 Hz.
- Independent audit: active packet rate 39.97 Hz, packet IDs contiguous,
  measured GT speed 0.2307–11.7441 m/s, measured steering 0–±0.50 rad, and
  466/504 exact-two measured-state cells occupied (408 with at least 20 rows).
- Combining r01 and r02 leaves 466/504 cells occupied and 405 cells with at
  least 20 rows in both runs. The exact same 38 cells are empty in both runs;
  all are at GT speeds ≥10.5 m/s with high steering (the highest occupied
  steering bins are ±0.30 rad in 10.5–11.0 m/s, ±0.20 rad in 11.0–11.5 m/s,
  and ±0.05 rad in 11.5–12.0 m/s). These are repeatable gaps under the tested
  approach-then-turn protocol, not yet proven physically unreachable.
- Shared-feature throttle loading and strict contiguous exact-two filtering
  were exercised successfully; training and held-out row/sequence counts are
  as listed above. No model results from the new combined evaluator exist yet.
- The unified four-lag feature path passed a small fit/predict schema smoke
  check; this is only a shape/causality-path check, not an accuracy result.
- Broad yaw model: still fails the requested maximum-error target; no production
  change has been made.

## Next steps

1. Finish the validation capture and verify 40 Hz active packet transfer,
   exact-two counts, reset integrity, collision/tilt status, and achieved GT
   coverage.
2. Export the clean captures into a run-grouped train/validation dataset,
   preserving raw bags and fixed 25 ms packet timebase.
3. Fit exact-two causal yaw models using training runs only. Report every
   measured speed × signed-steering cell, sample/run support, RMSE, p95,
   maximum absolute error, and count above 0.1 rad/s. Do not substitute target
   speed for GT speed or treat an empty cell as validated by interpolation.
4. Use the held-out capture to compare the frozen candidate. For remaining errors,
   investigate the relevant causal state/history and add only the missing
   targeted data or model term; do not promote on aggregate RMSE while hiding
   large maxima.

## Continuation — residual diagnosis and history test (2026-10-09)

This section supersedes the earlier “analysis pending” statements above.

### First full-spectrum comparison and residual diagnosis

The completed four-lag evaluation saved its predictions and outlier table even
though its final report serialization initially failed on a NumPy integer. The
serializer is fixed for future runs; the already-saved arrays were analyzed
without refitting. On independent full-grid capture r02 (40,828 exact-two
rows), the strongest current model is the measured-wheel-speed/steering local
expert:

| Model | RMSE (rad/s) | p95 | Max | Rows over 0.1 | Supported cells with p95 over 0.1 |
|---|---:|---:|---:|---:|---:|
| Historical global | 0.10581 | 0.25376 | 0.70358 | 6,349 | 185 |
| Full-grid yaw-only global | 0.05601 | 0.12319 | 0.78103 | 2,744 | 138 |
| Causal-event experts | 0.03488 | 0.07340 | 0.90726 | 1,190 | 97 |
| Measured-speed/steering local experts | **0.03426** | **0.06035** | **1.04456** | **852** | **75** |

This is a substantial improvement in typical error, but fails the required
every-sample <=0.1 rad/s criterion. The 852 violating rows are preserved in
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_exact_two/heldout_exact_two_yaw_outliers.csv`.
Errors concentrate in turn-in/unwind/reversal, particularly 200–300 ms after
the command transition. One worst row is at 9.44 m/s, with 0.045 rad measured
steering and -3.604 rad/s steering rate 228 ms into unwind: measured yaw
residual +0.227 versus predicted -0.818 rad/s. This supports a history or
transient-state hypothesis but does not prove the mechanism. Errors also occur
at low-speed/high-steering and near-zero-steering high-speed states; wheel/body
mismatch alone does not explain them. Roll and roll-rate history are already
inputs, so adding those signals again is not justified by this result.

### Repeated coverage gaps

Training r01 and validation r02 have the same 38 empty signed measured-state
cells under the current straight-approach-then-steer protocol:

- 10.5–11.0 m/s: ±0.35, ±0.40, ±0.45, ±0.50 rad (8 cells).
- 11.0–11.5 m/s: ±0.25 through ±0.50 rad in 0.05-rad increments (12 cells).
- 11.5–12.0 m/s: ±0.10 through ±0.50 rad in 0.05-rad increments (18 cells).

These are observed coverage gaps, not established physical limits. The current
profile changes steering after reaching speed and has only a 2.75 s turn / unwind
window. A reset-isolated envelope experiment is needed to distinguish transient
speed loss from sustained speed/steering limits: establish steering before or
during the speed approach, then hold measured steering while the speed
controller operates. Score only achieved GT speed and feedback steering. Keep
existing collision, tilt, and 12 m/s interlocks. If a cell cannot be occupied
safely through controlled approaches, record that measured limit rather than
inventing a training value.

The targeted profile `yaw_high_speed_envelope_gapfill` is now implemented in
`tools/open_plane_excitation.py` and `tools/run_open_plane_experiment.sh`.
It has 38 signed target conditions (10.75 m/s at ±0.35–0.50 rad; 11.25 m/s at
±0.25–0.50 rad; 11.75 m/s at ±0.10–0.50 rad), each with a 12 s pre-steered
approach, 1 s settle, and a 2.75 s final 0.05-rad steering step/hold. A
schedule check confirmed 114 phases, all 38 signed cells and labels parsed by
the exact-two packet auditor. Run it twice with separate run IDs/seeds for
independent replication. It preserves the existing 11.95 m/s governor,
12 m/s hard cutoff, collision stop, tilt stop, and reset-before-condition
behavior. The first train capture was started alongside the current CPU
intensive fit after
confirming 16 logical CPUs and memory headroom, using the already-running
pinned Explore batch-mode simulator under
`openplane_yaw_high_speed_envelope_gapfill_train_20261009_r01`. The bridge has
connected and the recorder/experiment are progressing through reset-isolated
conditions alongside the fit; no timing fault, collision, or tilt abort has
been observed so far. The second capture is planned as
`openplane_yaw_high_speed_envelope_gapfill_validation_20261009_r02`.
After both bags close, export them together with
`tools/vehicle_dynamics_learning/prepare_dataset.py` using the fixed packet
timebase and explicit train/validation split overrides, then include that
mixed-train/validation-only archive in the exact-two evaluator. No test or
final-test arrays are needed or opened.

### Expanded causal-history candidate

The four-snapshot model sees only 0, 25, 50, and 100 ms of history. A targeted
candidate now uses sensor snapshots at 0, 25, 50, 100, 200, 300, and 500 ms.
The same lag set is applied to the exact-two yaw captures and existing
throttle-surface archive; no future GT or receipt-timing-only features are
introduced. A schema smoke check produced 91 features (7 × 11 sensor
observations + 14 derived features). The throttle source audit reconfirmed
1,497 contiguous two-packet sequences admitted, with 7 one-packet and 4
three-packet sequences excluded. The full train/held-out fit is running in
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_long_history_exact_two/`.
Only training runs enter fitting; validation runs are used for comparison and
diagnosis, not fitting. Since prior residual analysis informed this revision,
any candidate promotion still requires a fresh unseen run.

After the two envelope captures are exported, rerun the same seven-lag
evaluator into a separate `yaw_full_spectrum_long_history_plus_envelope_exact_two`
output directory. That second fit will add only the envelope training capture;
the envelope validation capture remains held out and receives its own
prediction/outlier artifacts. Keep both model reports so the effect of the
new domain data is attributable rather than conflated with the longer history.

No production odometry/MPC integration or simulator-physics change has been
made. Retain the long-history candidate only if whole-run and per-cell held-out
errors improve without hiding maxima. Otherwise, use the residuals to design
the next targeted model/data experiment.

## Continuation — first envelope attempt audit and model result (2026-10-09)

This section supersedes the earlier statement that the first envelope capture
was still running and the earlier pre-steered approach plan.

### Seven-lag held-out score

The completed seven-lag evaluator wrote its prediction arrays and outlier CSV,
then failed while serializing the final report because it referenced a JSON
helper on the wrong module. The report writer now uses the existing atlas
serializer. The saved predictions were scored directly, without another fit.
On the same 40,828-row full-grid validation capture, the seven-lag local expert
did not improve on the four-lag comparator:

| Candidate | RMSE (rad/s) | p95 | Max | Rows >0.1 rad/s |
|---|---:|---:|---:|---:|
| Four-lag local expert | 0.03426 | 0.06035 | 1.04456 | 852 |
| Seven-lag local expert | 0.03535 | 0.06286 | 1.08398 | 950 |

The extra 200–500 ms history therefore does not solve the current transient
error. The seven-lag largest residual is still an unwind at 9.44 m/s, 0.045 rad
feedback steering, and 228 ms event age. Overall 97.67% of rows are within
0.1 rad/s, but the 950 violations and 1.084 rad/s maximum fail the all-sample
criterion. Keep the seven-lag model as a comparator; do not promote it.

### Empirical gap-fill test failure and correction

The first envelope bag was closed before inspection and audited at
`live_runs/openplane_yaw_high_speed_envelope_gapfill_train_20261009_r01/yaw_exact_two_coverage.json`.
It completed 114/114 phases, with no collision, no timing fault, and a
contiguous 40-Hz packet stream. The probe phases themselves were valid; the
38 quality failures were all the 1-second settle phases, each with fewer than
30 post-settling samples. Only 36/38 steering transitions had exactly two
contiguous packets; the two three-packet responses are excluded. Most
importantly, achieved GT speed topped out at 9.776 m/s, so none of the intended
>=10.5 m/s holes were tested. The pre-steered-from-rest approach was therefore
an invalid design for the stated coverage objective, even though the run was
collision-free. Preserve its raw data as protocol/frontier diagnostics only;
do not count it as a clean gap-fill training capture.

The profile was changed based on those measurements: approach at zero steering
with `reach_speed_target`, hold straight for 1.5 s, then apply the requested
steering step. The 38 conditions remain ordered from lower to higher combined
speed/steering demand, with resets and the existing speed/tilt/collision
interlocks. Corrected training capture
`openplane_yaw_high_speed_envelope_gapfill_train_20261009_r02` is now running.
If it reaches the intended bins cleanly, a separately seeded validation
capture
`openplane_yaw_high_speed_envelope_gapfill_validation_20261009_r03` will repeat
the same achieved-state probes. A third whole-spectrum capture will be
considered for a final fresh evaluation after the model revision is selected;
the previously used full-grid r02 remains diagnostic, not final proof.

## Continuation — corrected envelope audit and replicated validation (2026-10-09)

The corrected training capture r02 is complete and closed. It ran 114/114
phases across all 38 signed high-speed conditions, with no collision, timing
fault, phase-quality failure, or abort. It measured GT speed 10.741–11.781 m/s
and steering through ±0.50 rad. The active bridge stream averaged 39.958 Hz;
37/38 response transitions were exactly two contiguous packets and the other
was a three-packet response, excluded from fitting. One packet sequence skip
occurred in a settle interval; active-gap p50/p95/p99/max was
24.948/25.908/26.343/58.036 ms. The exception was `v11.25_a0.350_turn-1`,
which had three packets and therefore supplied no exact-two response rows.

Combined exact-two audit of full-grid training r01, full-grid validation r02,
and high-speed envelope training r02 now measures 503/504 cells occupied on
the 0.5 m/s × 0.05 rad signed grid. The sole empty cell is
11.0–11.5 m/s at −0.35 rad. 431 cells have at least 20 rows; 405 have at least
20 rows in each of two independent captures. The repeated 38-cell gap has thus
been reduced to one empty cell, but sparse/one-run support remains and is not
being declared complete. Low-speed near-zero and high-speed/high-steering bins
need interpretation against achieved state and response quality, rather than
being filled with target-command labels.

The two earlier full-grid captures remain 466/504 occupied each. The failed
envelope r01 is excluded. Its failure and the corrected r02 protocol are
documented above. A fresh, separately seeded high-speed envelope validation
capture is now running as
`openplane_yaw_high_speed_envelope_gapfill_validation_20261009_r03` (seed
202610094), using the established Explore batch simulator, the 40 Hz bridge,
reset isolation, and unchanged speed/tilt/collision interlocks. It is
validation-only and will remain out of fitting.

The initial frozen-GRU scoring attempt did not produce a score: the new full-grid
validation run had not yet been exported into an admitted run-grouped archive,
so the scorer correctly rejected its unknown run ID. The run has now been
exported separately to
`live_runs/derived_yaw_full_spectrum_gru_validation_dataset_20261009_r02/`
with an explicit validation split, fixed 25 ms packet timebase, and no test or
final-test data. The frozen-GRU comparison is being rerun against this admitted
capture. This is an evaluation setup correction, not a model result.

The combined coverage JSON is
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_coverage_after_envelope_train_r02.json`.
Next: finish r03, audit every target/event for exact-two status and measured
cell support, and add a narrowly targeted repeat only if the −0.35-rad missing
cell remains absent. Then export envelope training r02 and validation r03,
rerun the causal full-spectrum evaluator with the high-speed training capture
admitted, and compare all per-cell errors on held-out runs. Keep model changes
offline until they demonstrate a whole-run gain; the current broad model still
has substantial errors above 0.1 rad/s, and no production odometry/MPC or
physics changes have been made.

## Continuation — full-grid edge support and yaw residual response (2026-10-09)

### Independent high-speed validation and updated coverage

The high-speed envelope validation r03 is complete and closed:

`live_runs/openplane_yaw_high_speed_envelope_gapfill_validation_20261009_r03/run/run_0.db3`

It completed 114/114 phases and all 38 signed conditions with no collision,
timing fault, quality failure, or abort. The active stream averaged 39.958 Hz;
packet sequence IDs were contiguous. It measured 10.741–11.782 m/s and 0–±0.50
rad. Of 38 transitions, 34 were exact-two and four were three-packet; the four
three-packet events were excluded. The previously empty
11.0–11.5 m/s, −0.35 rad cell now has 15 exact-two rows from this independent
run. The individual audit is
`live_runs/racing_model_diagnostics_20261009/yaw_high_speed_envelope_validation_r03_coverage.json`.

Combining the two full-grid runs and both corrected high-speed envelope runs
now gives 504/504 occupied cells, 459 cells with at least 20 exact-two rows,
and 412 cells with at least 20 rows in two independent captures. The remaining
45 low-support cells are not being treated as adequate just because they are
nonempty: 14 lie below 0.5 m/s, and 31 lie from 10–12 m/s. The middle
0.5–10 m/s bands are densely supported; each has at least 20 rows in every
signed steering cell in the combined data, and almost all have that support in
two runs. The row-by-row counts are in
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_coverage_after_envelope_validation_r03.json`.

### Frozen GRU comparison

The old trajectory GRU checkpoint was scored on the full-grid validation r02
using only exact-two windows and causal sensor/command inputs. It is a poor
full-spectrum predictor and is not promoted:

| Event | Rows | RMSE (rad/s) | p95 absolute | Maximum absolute | Rows >0.1 |
|---|---:|---:|---:|---:|---:|
| Turn-in | 9,117 | 0.13686 | 0.27170 | 1.31096 | 1,943 |
| Unwind | 10,110 | 0.12403 | 0.28006 | 1.07213 | 2,068 |
| Reversal | 9,825 | 0.13321 | 0.27516 | 1.32033 | 1,819 |

Across these 29,052 exact-two rows its RMSE is 0.13127 rad/s, 5,830 rows exceed
0.1, and only 79.93% are within 0.1. This is much worse than the current
tree/local-expert result on the same full-grid bag (about 0.034 rad/s RMSE and
852 rows above 0.1 in the four-lag version). The GRU remains a comparator only;
no GRU retraining is being done on unrestricted packet cases.

Artifacts and the generic exact-two scorer are
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_long_history_exact_two/frozen_gru_full_grid_r02_exact_two.json` and
`tools/racing/specialists/score_yaw_gru_exact_two_probe_runs.py`.

### What the remaining error looks like

The best current measured-speed/steering local model still misses the requested
per-sample 0.1 rad/s threshold. Its 7-lag variant scores 0.03535 RMSE, p95
0.06286, maximum 1.08398, and 950/40,828 rows above 0.1 on full-grid r02; the
4-lag comparator is slightly better at 0.03426 RMSE and 852 violations. The
largest failures are transient, not an unfilled broad steady-state patch:
unwind contributes 557/950 violations; 251 violations occur 200–300 ms after
the command. Large errors appear from crawl to 10 m/s and near neutral
steering as well as at high speed. One representative failure at 9.44 m/s,
0.045 rad feedback and 228 ms into unwind has a target yaw-rate residual of
+0.227 rad/s but prediction −0.857 rad/s while steering is unwinding at
−3.604 rad/s. Adding 200–500 ms of tree history did not resolve it, and the
frozen GRU performs worse. The evidence points to transient/history-dependent
response not represented adequately by these fits; it does not yet prove
whether the missing cause is latent lateral motion, actuator/command history,
or target/sensor timing.

### Targeted sparse-cell experiment now running

The packet-qualified audit motivated a narrow reset-isolated test rather than
another full-grid duplicate. Profile `yaw_sparse_cell_support` is implemented
in `tools/open_plane_excitation.py`; the auditor now recognizes each new event
label, and the full-spectrum evaluator admits separate train and validation
runs. Its 33 speed/magnitude conditions target 0.25 m/s and only the weak
10.25–11.75 m/s cells. Each condition runs turn-in, unwind, reversal, and a
second unwind at 40 Hz so both steering signs and transient directions get
separate exact-two measurements. The conditions are ordered from lower to
higher speed/steering demand; every condition resets, and existing collision,
tilt, and speed guards remain enabled. Schedule/parser validation confirmed
198 phases and 132 auditable probe events (33 turn-ins, 66 unwinds, 33
reversals), with 574.5 s nominal schedule duration.

Training run currently active:
`openplane_yaw_sparse_cell_support_train_20261009_r01` (seed 202610095).
After it closes, it will be audited before a separately seeded validation run
`openplane_yaw_sparse_cell_support_validation_20261009_r02` is launched. Both
are explicitly assigned train/validation splits on 25 ms packet time, and
three-packet responses remain excluded. Then the exact-two model evaluator
will be rerun with the sparse training capture and scored separately on the
full-grid, envelope, and sparse validation captures. This comparison will
tell whether stronger coverage actually reduces error; it will not be called
success on cell counts or aggregate RMSE alone. Production odometry/MPC and
simulation physics remain unchanged.

### Sparse-support continuation — corrected capture and validation (2026-10-09)

The original sparse-support r01 did not reach the 11.75 m/s target because the
profile accidentally inherited the general 11.2 m/s test governor. This was a
test-harness defect. The sparse profile now uses the same 11.95 m/s governor
and 12.0 m/s hard cutoff as the upper-edge grid tests; no plant or production
behavior changed. A schedule smoke check confirmed the 198-phase plan and the
first upper-edge approach target before relaunch.

Corrected training capture, closed and audited:
`live_runs/openplane_yaw_sparse_cell_support_train_20261009_r02/run/run_0.db3`

- 198/198 phases completed, no abort, zero collisions, zero bridge timing
  faults; 39.977 Hz active packet rate.
- Measured GT speed was 0.208–11.771 m/s and steering reached ±0.50 rad.
  The 11.75 m/s approaches reached 11.678 m/s; a high-speed settle measured
  up to 11.783 m/s.
- All 132 response phases were valid. There were 127 exactly-two contiguous
  command/feedback transitions, one one-packet and four three-packet
  transitions; only the 127 exact-two cases are admitted. The fixed-timebase
  dataset has zero within-sequence packet gaps.
- The run-level audit flags nine crawl approach phases as `samples<30`; these
  are speed-setup phases, not yaw-response windows. No probe phase failed. The
  dataset's whole-bag stream/collision gate passes, and the evaluator admits
  only this exact set of non-response setup warnings for the paired sparse
  captures. The schedule now disables sample-count validation for those crawl
  approaches in future captures; validation r03 had already started with the
  earlier setting, so the same narrowly scoped exception is applied to its
  setup phases only.
- The capture's 127 exact-two transitions occupied 124 measured 0.5 m/s ×
  0.05 rad cells, with 27 cells reaching 20 rows in this run alone.
- Fixed-timebase training archive:
  `live_runs/derived_yaw_sparse_cell_support_train_20261009_r02/`.
- Per-run exact-two audit:
  `live_runs/racing_model_diagnostics_20261009/yaw_sparse_support_train_r02_coverage.json`.

Independent validation is now running in the same Explore batch-mode
simulator, with a separate seed:
`openplane_yaw_sparse_cell_support_validation_20261009_r03` (seed 202610098).
It uses the same 33 weak-cell conditions and remains held out. After closure,
the next steps are to audit and export it as validation, combine its cell
counts with the four earlier captures, then fit the exact-two local/event
comparators and score full-grid, upper-envelope, and sparse held-out runs.
Only after those results will remaining >0.1 rad/s errors be localized and
turned into the next targeted model/data revision.

### Sparse high-speed validation and final crawl coverage gap (2026-10-09)

Independent sparse high-speed validation r03 completed and was audited:
`live_runs/openplane_yaw_sparse_cell_support_validation_20261009_r03/run/run_0.db3`.
It completed 198/198 phases, with no abort, collision, timing fault, or phase
quality failure; active transfer was 39.98 Hz. The run measured 0.223–11.771
m/s and ±0.50 rad; 127 of 132 response events were exact-two contiguous
transitions (one one-packet and four three-packet events were excluded). Its
fixed-25-ms dataset is in
`live_runs/derived_yaw_sparse_cell_support_validation_20261009_r03/`, and its
audit is
`live_runs/racing_model_diagnostics_20261009/yaw_sparse_support_validation_r03_coverage.json`.

Combining the four original grid/envelope captures and sparse train r02 plus
held-out r03 now gives 504/504 measured cells occupied, 494 cells with at
least 20 exact-two rows, and 430 cells with at least 20 rows in at least two
independent captures. The only ten cells still below 20 rows are all below
0.5 m/s at high steering: +0.30 through +0.50 rad and −0.20, −0.35 through
−0.50 rad. The formerly weak 11.0–11.5 m/s, −0.35 rad cell now meets support.
Machine-readable counts:
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_coverage_after_sparse_validation_r03.json`.

The broad sparse profile did not hold its 0.25 m/s target during the response
probes: its measured crawl response windows drifted upward, so the remaining
ten cells cannot be closed by repeating that protocol. A focused
`yaw_sparse_crawl_cell_support` profile has been added. It uses the existing
measured 1% throttle anchor (0.244 m/s), only the seven steering magnitudes
that contain remaining weak cells, and turn-in/unwind/reversal at reset-isolated
conditions. Probe phases enforce the normal speed and steering quality gates.
The checked schedule contains 42 phases and 28 auditable exact-two response
events. Separate training and validation run IDs are wired into the evaluator.
This test has not started yet; first finish it as a train/validation pair, then
combine coverage and run the full held-out model comparison. No model candidate
has yet been promoted, and no production odometry/MPC or simulator physics have
changed.

### Crawl-gap completion and model evaluation (2026-10-09)

The focused crawl train/validation pair is complete:

- Train bag: `live_runs/openplane_yaw_sparse_crawl_cell_support_train_20261009_r01/run/run_0.db3`.
- Validation bag: `live_runs/openplane_yaw_sparse_crawl_cell_support_validation_20261009_r02/run/run_0.db3`.
- Both completed all 42 phases with no quality failures, collisions, timing
  faults, or packet-sequence gaps. Active rates were 39.947 and 39.968 Hz.
- Training GT speed was 0.216–0.273 m/s; validation was 0.222–0.275 m/s.
  All 28 response phases passed their speed/steering gates in each capture.
  Each run yielded 26 exact-two contiguous transitions; two 3-packet events
  per run were excluded.
- Fixed-timebase train and validation archives:
  `live_runs/derived_yaw_sparse_crawl_cell_support_train_20261009_r01/` and
  `live_runs/derived_yaw_sparse_crawl_cell_support_validation_20261009_r02/`.
- Audits:
  `live_runs/racing_model_diagnostics_20261009/yaw_sparse_crawl_train_r01_coverage.json`
  and `.../yaw_sparse_crawl_validation_r02_coverage.json`.

After adding both crawl captures, all 504 measured speed/signed-steering
cells have at least 20 exact-two rows combined; 432 have at least 20 in two
independent captures. No cell is empty or below 20 combined rows. The
aggregate is
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_coverage_after_crawl_validation_r02.json`.

The exact-two expanded model comparison is currently running at
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_sparse_crawl_exact_two_r01/`.
It includes the original full-grid and high-speed held-outs plus these sparse
upper-speed and crawl held-outs. Its output will determine whether residuals
remain above 0.1 rad/s, which speed/steering/event cells fail, and whether the
current local experts have enough training support or need another targeted
capture. No production model has been changed.

### Full-spectrum model results and error diagnosis (2026-10-09)

The expanded exact-two fit completed. Machine-readable results and held-out
predictions are in
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_sparse_crawl_exact_two_r01/`.
It used 10 training captures and 7 held-out captures (386,799 held-out rows,
including 337,214 rows from the throttle-surface continuation). No test or
final-test data were opened. The large pooled score is not a substitute for
the separate open-plane run results:

| Held-out capture | Local model RMSE | p95 | Max | Samples >0.1 rad/s |
|---|---:|---:|---:|---:|
| Full-spectrum grid r02 (40,828 rows) | 0.03225 | 0.05786 | 1.0673 | 783 |
| High-speed envelope r03 (388 rows) | 0.01658 | 0.03694 | 0.0831 | 0 |
| Sparse-cell support r03 (2,375 rows) | 0.05703 | 0.08750 | 0.6105 | 106 |
| Crawl support r02 (531 rows) | 0.03815 | 0.06590 | 0.2974 | 13 |

These are one-step yaw-rate residual errors in rad/s, not integrated yaw-angle
errors. The model is not yet below the requested 0.1 maximum. On the primary
full-grid held-out capture, the causal-event specialist scored RMSE 0.03630,
p95 0.07816, max 0.8872, with 1,347 rows above 0.1; measured-speed/steering
local experts reduce the overall errors, but still fail the maximum criterion.
The high-speed envelope result is promising but has only 388 exact-two rows
and cannot stand in for full-domain validation.

Error diagnosis from the full-grid run:

- 481/783 threshold violations occur in scheduled unwind windows; errors are
  also present in turn-in and reversal. The sensor-derived unwind expert has
  468/783 violations, so event classification alone is not sufficient.
- Large residuals cluster 100–250 ms after steering changes. One extreme at
  9.44 m/s, 0.045 rad measured steering, and 228 ms into unwind has a target
  residual of +0.227 rad/s; the local prediction is −0.840 rad/s (1.067 error).
  The causal event model also predicts the wrong sign (−0.508), so this is not
  only a local-cell selector problem.
- The held-out high-speed-envelope run has no >0.1 errors, while the sparse
  transition run has 70/106 violations during unwind and the crawl run has
  12/13 during unwind. Thus the remaining failure is transient/history
  dependent, not simply a missing steady speed/steering cell.
- Full-grid violations concentrate in approximately 5.5–7.5 m/s near-neutral
  steering during unwind, with another group around 1–2.5 m/s at high steering.
  Error counts rise in extreme steering-command/feedback-gap and throttle-gap
  quartiles and at both tails of rear-wheel/body-speed mismatch. Roll correlates
  with some errors but does not explain the crawl and near-neutral failures.

The coverage audit is complete only at the stated measured grid resolution:
all 504 bins (0–12 m/s in 0.5 m/s bins by signed steering −0.50…+0.50 rad in
0.05 rad bins) have at least 20 exact-two rows combined. Only 432 bins have
that support in at least two independent captures. This is not a claim that
every continuous speed/angle pair is tested or that every bin is accurate.

The evaluation exposed a support bug in the local-expert gate: it checked row
and phase counts but did not require independent capture count. Of 549 local
cells marked supported across the measured training domain, 278 were supported
by only one capture. An offline selector replay on the already-held-out full
grid, using the causal-event expert instead of the pooled fallback for
unsupported cells, reduced RMSE from 0.03225 to 0.02907 and violations from
783 to 629; max remained 1.067. Requiring two captures reduced the maximum to
0.887 but raised violations to 709. These are development diagnostics on an
already-inspected validation run—not final promotion evidence. The evaluator
now records both selector variants and requires independent-capture support
as an explicit candidate criterion.

A diagnostic-only upper-bound check is being added to test whether current
simulator-truth body-frame u/v and sideslip explain the unresolved errors.
Those values are forbidden inference inputs; they are used only as labels for
this information test. The alignment path is being checked against the same
25-ms rows before any score is accepted. Depending on that result, the next
action is either (a) estimate the missing lateral/body-motion state from
permitted sensor history, or (b) add a narrowly targeted independent capture
for cells whose local experts lack replicated training support. No broad sweep
is justified by the current evidence, and no production odometry, MPC, physics,
or runtime behavior has changed.

### Under-replicated-bin replication captures

The 72 signed bins below the two-capture/20-row support criterion reduce by
left/right symmetry to 38 speed/magnitude conditions. The new
`yaw_exact_two_support_replication` development profile targets precisely
those conditions (0.25, 6.75, 10.25, 10.75, 11.25, and 11.75 m/s bands), with
onset, unwind, reversal, and a second unwind at each condition. The schedule
was checked in the pinned API container: 228 phases, 38 reset-isolated
conditions, 152 probes, 667 s nominal drive time and an 853.2 s schedule/reset
budget. It retains the existing 11.95 m/s governor, 12.0 m/s hard cutoff, and
tilt/collision interlocks; physics and runtime odometry are unchanged.

Training capture r01 completed in the already-started Explore batch-mode
simulator:
`live_runs/openplane_yaw_exact_two_support_replication_train_20261009_r01/`.
It completed all 228/228 phases with zero experiment quality failures, zero
collisions (0 at start and end), zero bridge timing faults, no non-unit packet
sequence steps, and a measured active interval rate of 39.92 Hz (median
25.06 ms). Of 152 probe transitions, 140 were contiguous exactly-two-packet
responses; 11 three-packet and one one-packet transitions were excluded from
the admitted response set. The 38 approach phases are deliberately unscored.
Coverage/audit output is
`live_runs/racing_model_diagnostics_20261009/yaw_exact_two_support_replication_train_r01_coverage.json`.
The intended speed range was reached (measured 0.229–11.786 m/s), and steering
feedback spanned 0–0.50 rad. This validates the capture procedure and packet
quality, not yet whether each target cell now has sufficient independent
exact-two support; that is checked in the combined corpus audit after r02.

The separately seeded validation capture r02 completed with seed 202610092 at
`live_runs/openplane_yaw_exact_two_support_replication_validation_20261009_r02/`.
It also completed 228/228 phases with zero quality failures, zero collisions,
zero timing faults, and no packet-sequence gaps. Its measured active rate was
39.95 Hz (median 24.99 ms); 143/152 transitions had exactly two contiguous
packets and nine three-packet transitions were excluded. Its derived archive
is assigned validation only and is not used in fitting.

Both captures were converted with `prepare_dataset.py` using the fixed 25-ms
packet timebase and `--continuous-whole-run`. Each produced 38 sequences and
about 14,000 frames. Their schema-7 archives and manifests are under
`live_runs/racing_model_diagnostics_20261009/yaw_exact_two_support_replication_{train_r01,validation_r02}_dataset/`;
the corpus discovery gate confirms r01 is train and r02 is validation. The raw
bags now have an analysis-ready, split-safe path into the model corpus.

The detailed two-run transition audit covered all 152 intended event-stage
keys. 132 passed the exact-two contiguous-packet gate in both captures, 19 in
only one, and one high-speed (+0.50-rad onset at 11.75 m/s) passed in neither
(both attempts were three-packet responses and excluded). These are exact
event-stage support counts, not a claim that each of the other cells lacks
historical support. The combined audit is saved at
`live_runs/racing_model_diagnostics_20261009/yaw_exact_two_support_replication_combined_coverage.json`.
An aggregate exact-two coverage audit over the 15 prior yaw bags plus these two
new bags reports all 504 measured grid cells occupied with at least 20 rows
pooled across runs. 448/504 have at least 20 rows in at least two independent
runs. The remaining 56 are not empty: each has 54–103 pooled rows from 4–8
independent runs, while zero or one individual run reaches 20 rows. Thus no
measured grid cell is currently a zero-data hole; the
56 are weaker replication cells, and their relevance will be judged against
held-out model errors and local expert support. This is not a claim of dense
continuous coverage between grid cells. Full audit:
`live_runs/racing_model_diagnostics_20261009/yaw_full_domain_exact_two_coverage_after_supportrep_r01_r02.json`.

The completed pre-replication evaluator compared two hybrid selector
candidates: route unsupported local cells through the causal-event model, with
one variant requiring at least two independent training captures. It also
evaluated a diagnostic-only GT u/v/sideslip upper bound; exact row alignment
was confirmed on the full-grid archive (80,117/80,117). Simulator truth is
not used by either legal candidate. The final-test arrays remain sealed.
After r02 passed the same quality audit, the evaluator was rerun with r01 as
training data and r02 as whole-run held-out data. Any remaining
>0.1 rad/s errors will be re-stratified by the newly replicated speed/steering
cells and transition events; they will not be treated as solved merely because
the planned grid is covered.

The post-replication evaluator completed with 11 train and 8 whole-run
validation sources (412,981 training rows). On the unchanged original full-grid
holdout, the two-capture local/event hybrid changed from 709 to 713 rows over
0.1 rad/s and from a 0.887 to 0.890 rad/s maximum: effectively no gain. On the
new replication holdout it scored 2,276 rows: RMSE 0.04684 rad/s, p95 0.03427,
48/2,276 above 0.1, and max 0.99171. The violations split into 40/231 unwind,
8/408 turn-in, and 0/1,637 hold samples. Thus the new coverage data are useful
for an independent transfer check but did not solve the transient yaw model.
The report is
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_supportrep_r01_train_r02_validation/report.json`.

The extended-history comparison is now running with the same train/validation
split. Its output target is
`live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_supportrep_history500ms_r01_train_r02_validation/`.

The pre-replication evaluation has now completed. It used 10 training runs and
7 whole-run validation captures (386,799 held-out rows overall); the full-grid
validation alone contained 40,828 rows. For that full-grid run, the local
speed/steering/event model had RMSE 0.03225 rad/s, p95 0.05786 rad/s, and 783
rows above 0.1 rad/s (98.08% within threshold), but a 1.0673 rad/s maximum.
The local/event hybrid routed by whether local support existed reduced this to
RMSE 0.02907, p95 0.05126, and 629 rows above 0.1, with the same 1.0673 maximum.
Requiring two independent training runs reduced the maximum to 0.8872 but left
709 rows above 0.1. Thus none meets the requested all-samples threshold, despite
the strong central-distribution accuracy.

Error attribution on that full-grid validation showed 468 of the 783 local
model violations during unwind, 239 during turn-in, and 76 during hold. Errors
were disproportionately concentrated at both extremes of steering
command/feedback mismatch, rather than growing monotonically with one simple
variable. The worst row was a 9.442 m/s unwind at 0.045 rad measured steering,
with 0.427 m/s rear-wheel/body-speed mismatch and −3.604 rad/s steering rate:
GT target residual +0.227 rad/s, model −0.840 rad/s. This is a wrong-sign
transient, not merely a small coefficient error.

The GT u/v/sideslip upper-bound diagnostic barely changes the full-grid result:
RMSE 0.05869 to 0.05673 rad/s; p95 0.12461 to 0.12237; violations 2,853 to
2,703, while the maximum slightly worsens (0.755 to 0.777). These features are
forbidden model inputs and were used only diagnostically. Current body velocity
and sideslip therefore do not explain most remaining large errors; model
structure/event history and the extreme steering-mismatch regimes remain the
next targets. No production odometry or MPC code was changed.

### Extended-history result and current error investigation

The full-domain support audit after adding r01/r02 is complete (17 raw yaw
captures, 504/504 measured cells with at least 20 pooled exact-two rows; 448
with at least 20 rows in two runs). The remaining 56 lower-replication cells
are now being cross-checked against the observed held-out error map and the
local-expert support counts before deciding whether another simulator capture
is warranted.

The 500-ms-history comparison completed using the same 11-train/8-validation
split. It reduced causal-event-specialist errors on the new replication run
from 12 to 7 of 2,276 samples (RMSE 0.01280 to 0.01213 rad/s; maximum 0.1868
to 0.1781). On the 40,828-row full-grid holdout it changed 1,289 to 1,263
samples above 0.1 rad/s, but p95 slightly worsened (0.07543 to 0.07703) and
maximum changed 0.89013 to 0.89168. The two-run local/event hybrid also
worsened on that full-grid holdout (713 to 744 threshold violations) while
improving slightly on the replication run (48 to 43). Longer sensor memory
alone is therefore not a general solution; neither history candidate reaches
the requested all-samples bound. Results:

- Standard history (0–100 ms):
  `live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_supportrep_r01_train_r02_validation/`
- Extended history (0–500 ms):
  `live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_supportrep_history500ms_r01_train_r02_validation/`

The exact-two aggregate support audit was joined to full-grid held-out error
bins. Seven of the 56 lower-replication bins contained causal-event-model
violations; these were all at crawl speed with 0.35–0.50 rad steering, with
1–2 violations per bin. Most other large-error bins had adequate aggregate
occupancy, so data scarcity alone cannot explain the full error tail. Separately,
the training-only local selector has fewer than 80 rows in several of the
replication-run failure bins around 6.5 m/s at ±0.5/0.1 rad and 10.5–11.5 m/s
at ±0.05 rad. This difference between pooled-split coverage and actual training
support is important: some rows exist only in validation captures and must not
be leaked into fitting.

A focused diagnostic on both the full-grid and support-replication heldouts
refit the already compared sensor-only global, event, local, and hybrid
candidates, then exported every row exceeding 0.1 rad/s with condition/event
age, speed, steering, wheel/body mismatch, actuator gaps and rates, yaw, roll,
and signed residual. The output is:
`live_runs/racing_model_diagnostics_20261009/yaw_error_rows_fullgrid_and_replication/`.
The reusable diagnostic source is:
`tools/racing/specialists/diagnose_yaw_support_replication_errors.py`;

The full-grid event model has 1,289/40,828 samples above 0.1 rad/s. Of these,
843 are in the requested unwind phase; top misses repeatedly occur about
225–250 ms after a high-angle steering command starts returning to center.
For example, at 8.135 m/s, steering feedback is 0.045 rad, steering rate is
−1.804 rad/s, and yaw rate is 1.058 rad/s. The next GT yaw residual is
 +0.023 rad/s, but the model predicts −0.867 rad/s (absolute error 0.890).
The vehicle retains yaw momentum while the learned model predicts decay or
sign reversal too quickly. Other large errors show the same pattern at
4.7–10.8 m/s, including both signs. Extended history does not remove it.

In the extended-history report, 789/1,263 failures fall in the causal-sensor
unwind class. Errors are elevated at both extremes of steering command/feedback
gap (525 in the lowest-gap quartile and 530 in the highest). Wheel/body-speed
mismatch quartiles are comparatively flat, so wheelspin alone is not an
adequate explanation. Roll is already an input; the highest absolute-roll
quartile has 536 failures versus 172 in the next quartile, but low-roll also
has 325, so roll alone cannot explain the tail.

The independent replication capture separates model structure from local
support. The causal-event model has only 12/2,276 errors over 0.1 rad/s (8
unwind, 4 turn-in), while the local model has 56 and the two-run hybrid 48.
In particular, local high-steer unwind predictions at 6.75 m/s miss by about
0.98–0.99 rad/s despite 950–1,294 training rows in those wheel-speed/steering
cells; the pooled causal-event model predicts those rows within the threshold.
That is a model/selector failure, not a data-quantity gap. Conversely, six of
the causal-event model's 12 replication failures lie in cells with fewer than
80 local training rows, and the full-grid event model has 139 failures in
locally unsupported cells. Some additional support is justified there.

I also swept the global independent-run gate using the saved full-grid
predictions (no refit): minimum run counts 1/2/3/4/5 produced 645/713/1,048/
1,149/1,168 samples above 0.1 rad/s. Raising one global threshold therefore
hurts the full-grid holdout; support must be evaluated per operating cell and
transition type, not by a single more conservative cutoff.

### New full-spectrum training capture

The next experiment is a second independently seeded full-grid training
capture, rather than another general-purpose data sweep. It repeats the
established reset-isolated 0.25–11.75 m/s × 0.05–0.50 rad grid in both turn
directions. Its purpose is to provide training support for the error cells
that currently rely on a single capture and to supply additional transition
histories for regime-specific fitting. Run ID:
`openplane_yaw_full_spectrum_grid_train_20261009_r03`, seed `202610093`.
It is running through `tools/run_open_plane_experiment.sh` against the existing
batch-mode Explore simulator. The existing collision, tilt, speed, timing, and
reset abort checks remain active. An independent, staggered validation will
follow: `YAW_FULL_SPECTRUM_MIDPOINT_PROFILE` with run ID
`openplane_yaw_full_spectrum_midpoint_validation_20261009_r04`, seed
`202610094`. The schedule has 1,400 independently reset conditions (4,200
phases): every point omitted by the current 0.5 m/s × 0.05 rad grid. A
schedule-level check verified the 1,400 conditions are unique and exactly
complement the existing 480 points to form the complete 1,880-point
0.25 m/s × 0.025 rad lattice over 0.25–11.75 m/s and both steering signs up
to 0.50 rad. The scheduled grid does not exceed the existing 12 m/s hard
limit. This is held-out validation, not training data.

After r03 has closed and passed its audit, start r04 with the established
batch-mode workflow:

```sh
SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode \
SDU_APEX_EXPERIMENT_PROFILE=yaw_full_spectrum_midpoint_validation \
SDU_APEX_EXPERIMENT_RUN_ID=openplane_yaw_full_spectrum_midpoint_validation_20261009_r04 \
SDU_APEX_EXPERIMENT_SEED=202610094 SDU_APEX_EXPERIMENT_TIMEOUT_S=28000 \
./tools/run_open_plane_experiment.sh
```

When converting its closed bag, assign the explicit split override
`openplane_yaw_full_spectrum_midpoint_validation_20261009_r04=validation`;
the generic filename classifier does not yet recognize the 2026-10-09 run
name as validation.

An empirical recheck of the closed r02 bag verified the exact-two audit
transition timing. For all 480 conditions, measured command departures were
within 3–14 ms median of the configured transition offsets (0.00, 0.70, 1.30,
2.10 s). The apparent 0.20–0.25 s discrepancy was a mistaken interpretation
of linear-ramp waypoint endpoints: the configured times are the ramp start
times, which is what the audit detects. Across 1,920 transitions, 1,777 were
exactly two contiguous packets; 140 had three packets and 3 had four. The
non-two-packet transitions remain excluded. No audit timing correction is
needed. This timing check was run on closed r02; the separately audited r03
capture also admits only exactly-two contiguous response transitions.

The first cell-by-event candidate has now completed. On the independent
support-replication run it reduced the local model's 56 violations to 13,
nearly matching the pooled event specialist's 12 (RMSE 0.01369 vs 0.01280
rad/s), and improved p95 to 0.01957 rad/s. But on the broad full-grid holdout
it had 1,162 violations versus 1,289 for the pooled event model and 713 for
the existing local/event hybrid. It is therefore not a general improvement
and is not being promoted. Candidate output:
`live_runs/racing_model_diagnostics_20261009/yaw_cell_event_specialist_candidate_current_train/`.

To resolve the local-vs-pooled routing failure without tuning on validation
labels, the diagnostic now supports a run-grouped, training-only out-of-fold
selector. It compares both candidates on entire training captures excluded
from fitting, then selects a local cell/event expert only when at least three
independent held-out runs show repeated RMSE and >0.1-rad/s-tail wins. An
initial offline attempt was stopped during source loading after I found that
it would refit the same fold experts for each held-out run. The implementation
now fits each expert once per fold, and the bounded offline evaluation has
restarted. Its decisions will be tested only on the sealed full-grid and
replication runs. It is a model-selection diagnostic, not runtime code.

The completed grouped-OOF selector chose the pooled causal-event model for
every cell; no local cell/event expert met the training-only gate. On sealed
r02, it therefore exactly matches the pooled specialist: full-grid RMSE
0.03557 rad/s, p95 0.07543, and 1,289/40,828 rows above 0.1; support-
replication RMSE 0.01280, p95 0.02314, and 12/2,276 above 0.1. This rejects
cell routing as a current improvement; the remaining errors need a different
causal feature/model explanation, not more permissive per-cell thresholds.

The previously successful frozen causal GRU has been scored unchanged on the
sealed full-grid and support-replication captures. Its training report has no
overlap with either run. Its one-step output is causal (past sensor/actuator
history plus the current command); later command steps cannot affect its
first decoded output. It transfers poorly: on the full-grid capture, 5,830 of
29,052 scored rows exceed 0.1 rad/s (pooled RMSE 0.13127, max 1.32033); on the
support-replication capture, 556 of 1,953 exceed 0.1 rad/s (RMSE 0.23715,
max 0.81296). It is not a promotion candidate. It scores fewer rows than the
tree evaluation because its 1.6 s history/future-validity windows are stricter;
the percentages are useful as a transfer check but not yet an exactly
row-matched model comparison.

The r03 full-grid training capture has completed and its bag is closed:
`live_runs/openplane_yaw_full_spectrum_grid_train_20261009_r03/run/run_0.db3`.
All 480/480 conditions and 1,440/1,440 phases completed with zero experiment
quality failures, zero start/end collisions, zero timing faults, and no
abort. The runner reported 39.69 Hz command delivery; the closed-bag audit
measured 39.928 Hz active packet rate, p95 interval 25.97 ms, and no packet
sequence discontinuities. Of 1,920 response transitions, 1,686 had exactly
two contiguous packets and are admitted; 8 one-packet, 217 three-packet, and
9 four-packet transitions are excluded. This confirms the need to retain the
strict exact-two gate. The single-capture coarse grid occupied 466/504
measured GT cells, with 401 cells having at least 20 rows. Its fixed-25-ms
whole-run training export is
`live_runs/derived_yaw_full_spectrum_grid_train_20261009_r03/` (104,847
samples, 481 sequences); the sealed r02 and final/test arrays were not changed.

The grouped-OOF selector is being rerun with r03 included as training data;
its earlier r01-only result is retained as a baseline. Meanwhile the held-out
midpoint validation r04 is running in the existing batch-mode Explore
simulator. At the latest check it had completed 158/1,400 approach phases
(11.3% of conditions started through their approach; settle/probe still follow);
reset, collision, tilt, speed, timing, and recorder guards remain enabled.
The run is expected to take several hours. Its active bag must not be opened
until the run closes.

The midpoint profile is now also recognized by the experiment wrapper: its
help text, 28,000 s timeout, reset-enabled profile list, and strict profile
allowlist are wired. `bash -n tools/run_open_plane_experiment.sh` passes.
During the completed r03 invocation, a shell syntax diagnostic appeared after
the container had already reported a clean bag close and driver status 0. The
script file had been edited while that invocation was still running; the
current file validates, and it was left untouched before launching r04. If
the diagnostic recurs after r04 closes, investigate the wrapper independently
before another run.

No production odometry or MPC code, physics, or runtime topic behavior has
changed. The final-test arrays remain sealed. The batch-mode Explore simulator
remains running, and r04 is the only active driving experiment; the r03-aware
offline OOF fit is CPU-capped and reads only closed training/validation data.
No runtime candidate is promoted until it improves the independent whole-run
results and the remaining >0.1-rad/s cases have been re-audited.

### Residual-model hypothesis to test

Feature inspection shows the current causal event model receives sensor and
actuator history at 0/25/50/100 ms plus a four-class event label, but does not
receive explicit elapsed time since the steering transition. The broad
holdout's largest remaining cluster is unwind (879/1,289 >0.1-rad/s rows).
Of those unwind errors, 712 occur 100–300 ms after the measured command edge
(345 at 100–200 ms and 367 at 200–300 ms); 136 are within 100 ms and 31
precede the edge. A representative miss is 225–250 ms after a high-angle
return toward center. This distribution supports testing explicit response
age, but is not itself proof of a gain. Extending generic history to 500 ms
barely changed the full-grid tail (1,289 to 1,263 rows), so simply adding
more lags is not established as a solution.

The targeted offline candidate is now running: add a causal steering-event-age state,
derived from the observed command/feedback edge and history—not phase labels
or future sensors—and allow its response to vary with measured speed and
steering. Fit using training captures only, with run-grouped OOF selection;
score unchanged on sealed r02 now and midpoint r04 after its bag is closed and
archived. This is a hypothesis, not yet evidence of improvement. Any runtime
use must separately pass the competition topic-policy audit. The r04
midpoint experiment is intended first to establish whether the same error
pockets persist at unseen lattice points.

### Current validation accuracy snapshot

The latest completed broad holdout is still coarse-grid r02; r03 is training
only and r04 midpoint validation is not complete. On r02 the pooled causal
event model has 40,828 exact-two rows, yaw-rate RMSE 0.03557 rad/s, absolute
error p95 0.07543 rad/s, maximum 0.89013 rad/s, and 1,289 rows (3.16%) above
0.1 rad/s. Thus the model is not yet below 0.1 rad/s everywhere. By event,
unwind accounts for 879/1,289 threshold violations, versus 265 in turn-in
and 145 in reversal. Worst repeated pockets include unwind around 5–5.5 m/s
at roughly 0.10–0.15 rad steering; turn-in also contributes around 6–8.5 m/s.
These are yaw-rate prediction errors (rad/s), not accumulated yaw-angle
errors (rad). The r04 midpoint run currently has 255/1,400 approach phases
complete (18.2% of conditions have passed the approach stage; settle/probe
still follow); its unseen-point accuracy cannot be claimed until the closed
bag is audited and scored.

### What “unwind” means, with raw validation examples

Here, unwind means the steering input is being brought back toward zero after
the car has been turned into a corner (or after a reversal). The wheels are
nearing straight, but the chassis can still have substantial yaw rate. The
model is asked for the next 25-ms simulator-truth yaw rate relative to the
current IMU yaw rate; it is not being asked to predict a full-lap angle in
these rows.

Two largest exact-two validation misses illustrate the failure:

| Quantity | 8.25 m/s, 0.45-rad turn, unwind | 10.75 m/s, 0.45-rad turn, unwind |
|---|---:|---:|
| Time after steering command began returning | 250 ms | 228 ms |
| Current measured steering | +0.045 rad | +0.045 rad |
| Steering feedback rate | -1.804 rad/s | -3.604 rad/s |
| Current IMU yaw rate | +1.058 rad/s | +0.738 rad/s |
| GT next yaw rate | +1.081 rad/s | +0.965 rad/s |
| Model next yaw rate | +0.191 rad/s | +0.210 rad/s |
| Prediction error (model minus GT) | -0.890 rad/s | -0.755 rad/s |
| Rear-wheel mean minus GT body speed | +0.313 m/s | +0.427 m/s |

For example, in the first row the steering command has returned to zero and
feedback is almost straight, but the car is still rotating at about +1.06
rad/s. Ground truth says it remains near +1.08 rad/s one packet later. The
model instead expects it to fall to about +0.19 rad/s. It therefore loses
track of retained yaw motion during the steering return. The wheel/body speed
gaps show that wheel slip is present in these examples, but these two rows do
not prove that slip caused the yaw error.

Across all 879 unwind violations, 712 occur 100–300 ms after the measured
steering-command edge (345 at 100–200 ms and 367 at 200–300 ms). That timing
pattern is why explicit response age is being tested. It is evidence of a
transient timing/history association, not proof of a single physical cause:
the unwind errors are not all the same sign (434 predictions are low and 445
are high). The likely issue is that the present inputs/model do not describe
the changing yaw response precisely enough through the transition, rather
than one constant yaw-rate bias. The two raw rows are in
`live_runs/racing_model_diagnostics_20261009/yaw_cell_event_oof_gate_train_r01_validation_r02/openplane_yaw_full_spectrum_grid_validation_20261009_r02_causal_event_specialists_errors_over_0p1.csv`.

### Left/right rear encoder split in the same holdout

The large yaw-error export contains 1,289 rows over the 0.1-rad/s threshold.
Their absolute rear-left/right surface-speed difference has median 0.012 m/s,
p95 0.062 m/s, and maximum 0.099 m/s. In the two largest yaw misses above,
the splits are only about 0.002 and 0.003 m/s, so one rear encoder running
far faster than the other does not explain those worst cases.

There are nevertheless moderate differentials elsewhere: the largest
differences among the error rows (about 0.094–0.099 m/s) occur mainly in
1.75–2.25 m/s, 0.30–0.50 rad steering unwind samples, with yaw-rate errors
around 0.10–0.20 rad/s. Across all validation rows, the lowest and highest
signed wheel-split quartiles each have about 4.1% yaw errors above 0.1 rad/s,
versus about 2.1–2.5% in the middle quartiles. This is an association with
extreme left/right speed difference, not proof of one-wheel tire slip: a
normal turn also gives the outside rear wheel a different path speed. The
current export does not compare each wheel's measured speed against its
ground-truth contact-patch speed, so it cannot classify these cases as
one-wheel slipping.

### Can ground truth identify slip as a yaw-error factor?

Yes, kinematic slip proxies can be reconstructed offline from the documented
wheel radius, wheelbase/track, Ackermann steering, rear encoders, and
ground-truth body velocity/yaw rate. For each wheel, shift body velocity to
the wheel contact point (`u_i=u-r*y_i`, `v_i=v+r*x_i`), rotate into the
wheel's steered frame, then compare rear `R*omega_i` against local longitudinal
contact speed for `Sx`; compute `Sy=v_y,i/abs(v_x,i)` for lateral slip. The
guide's longitudinal/lateral curve landmarks are useful reference points, not
direct measurements of force or the simulator's hidden tire state.

This has been done on earlier open-plane captures. The documented full-input
holdout showed rear encoder surface speed exceeding GT rear-contact speed by
median 7.86 m/s in the high-throttle/high-steering, below-6-m/s subset (97.5%
of samples exceeded 3 m/s), while the low-throttle/high-steering subset stayed
near zero excess. That is strong evidence of common-mode rear wheelspin in
that specific excitation. At 4 m/s, the reconstructed front lateral-slip
proxy rose from about 0.017 at 0.20-rad steering to 0.161 at 0.25 rad, while
steady yaw rate dropped about 49.5%; the high-angle result is consistent with
front lateral authority loss. A rear-slip yaw-gain feature improved held-out
fits at 2.2 and 3.0 m/s, but failed at 5.0 m/s and the 6.5-m/s trough. Thus
slip is a demonstrated regime-specific factor, not a universal explanation.
See [`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md)
for the calculations, validation splits, and rejected model fits.

The current full-spectrum unwind failures have not yet been scored against
these corrected per-wheel proxies; prior rows only reported rear wheel/body
mismatch and left/right split. The closed r02 bag already contains the needed
GT motion, steering, and rear encoders, so this attribution can be tested
offline without another simulator run. Keep GT-derived slip as a diagnostic
or training target only; a deployable observer/MPC feature must be computable
from permitted causal inputs without future GT.

### Continuation — tire-slip landmarks and yaw-error attribution (2026-10-09)

The official [AutoDRIVE vehicle dynamics guide, §1.3.2](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics)
defines longitudinal slip as `Sx=(R*omega - Vx)/Vx` and lateral slip as
`Sy=tan(alpha)=Vy/abs(Vx)`. Its two-piece cubic tire curve is anchored at an
extremum and then an asymptote; the published slip landmarks are:

| Tire direction | Peak/extremum | Second/asymptote landmark |
|---|---:|---:|
| Longitudinal | `|Sx|=0.15`, normalized force 0.72 | `|Sx|=0.25`, normalized force 0.464 |
| Lateral | `|Sy|=0.01`, normalized force 1.0 | `|Sy|=0.10`, normalized force 0.50 |

The second point is not a measured minimum or an instantaneous force sample.
These coordinates tell us which part of the stated tire curve a kinematic
slip proxy occupies; this bag still lacks the per-wheel force, normal load,
front-wheel RPM, and hidden spline output needed to claim direct force
reconstruction.

Implemented a research-only exact-two-packet analysis in
`tools/racing/specialists/analyze_yaw_slip_regime_factor.py`. It derives
wheel-contact velocities from CG truth `(u,v,r)`, published wheelbase/track/CG
geometry and Ackermann steering; rear `Sx` uses the recorded rear-wheel
surface speeds, and all four `Sy` values use wheel-frame velocity. Near-zero
wheel-forward-speed rows are explicitly marked invalid. Training uses the two
complete full-grid training captures r01/r03; scoring uses only the separately
randomized, closed full-grid validation r02 (40,828 admitted rows, exactly two
contiguous actuator-feedback packets). A matched comparison separates the
sensor-only predictor, a current GT-motion-state oracle `(u,v,r)`, and the
same GT-state oracle plus GT-derived slip. This control prevents attributing a
gain from simply providing better current motion state to “slip.”

| r02 predictor | Yaw-residual RMSE | p95 absolute | Rows over 0.1 rad/s |
|---|---:|---:|---:|
| Sensor-only | 0.03720 | 0.08016 | 1,300 / 40,828 |
| GT motion-state oracle | 0.03643 | 0.07979 | 1,258 / 40,828 |
| GT state + slip oracle | **0.03477** | **0.07485** | **1,058 / 40,828** |

The slip features add a measurable but not universal signal beyond GT state:
RMSE falls another 0.00167 rad/s (4.6%); the paired 479-condition bootstrap
interval is `[-0.00184, -0.00105]` rad/s. The overall maximum remains 0.940
rad/s, so this does not solve the large outliers. The effect is concentrated
where the tire-curve landmarks predict reduced lateral authority:

| Regime | GT-state RMSE | GT state + slip RMSE | >0.1 rad/s, state → state+slip |
|---|---:|---:|---:|
| Front `|Sy|<0.01` | 0.03798 | 0.03917 | 605 → 632 |
| Front `0.01≤|Sy|<0.10` | 0.05410 | **0.04736** | 504 → **330** |
| Front `|Sy|≥0.10` | 0.01977 | **0.01579** | 126 → **72** |
| Rear mean `|Sx|≥0.25` | 0.04408 | 0.04764 | 144 → 184 |

The front lateral peak-to-asymptote improvement is consistent across 415
conditions (condition-bootstrap RMSE delta `[-0.01172, -0.00643]` rad/s);
the at/beyond-asymptote improvement is consistent across 295 conditions
(`[-0.00456, -0.00337]` rad/s). In contrast, the rear high-longitudinal-slip
cohort does not improve over the GT-state model. Event-wise, state+slip
improves unwind RMSE from 0.03630 to 0.03260 rad/s and reduces >0.1 errors
from 678 to 458, but does not improve reversal (0.03252 to 0.03332) and is
only a small gain on turn-in. This supports a front-lateral-slip/unwind
interaction, not one general slip correction for every regime.

The causal sensor-only observer is implemented in
`tools/racing/specialists/evaluate_sensor_only_slip_observer.py`. It receives
only current/past encoder, IMU, actuator-feedback and command history plus a
causal sensor-derived event. Its yaw model is trained on leave-one-whole-run-
out slip predictions from r01/r03, then evaluated on closed r02. Held-out
per-wheel slip errors are:

| Estimated quantity | RMSE | p95 absolute error |
|---|---:|---:|
| Front lateral `Sy`, left/right | 0.00363 / 0.00366 | 0.00753 / 0.00749 |
| Rear lateral `Sy`, left/right | 0.00111 / 0.00111 | 0.00231 / 0.00230 |
| Rear longitudinal `Sx`, left/right | 0.03062 / 0.03048 | 0.03195 / 0.03155 |

Front lateral estimates stay fairly accurate even beyond the guide's
asymptote (two-wheel pooled RMSE 0.00418, p95 0.00774). In the peak-to-
asymptote band the pooled RMSE is 0.00574, p95 0.01090. Rear longitudinal
slip is much harder under severe wheelspin: at/beyond `|Sx|=0.25`, its pooled
RMSE is 0.0925 and p95 0.201. That distinction agrees with the yaw ablation:
front lateral slip is the useful feature here; high rear longitudinal slip is
not.

The sensor-only slip-augmented yaw model, without a gate, lowers r02 RMSE from
0.03720 to 0.03622 rad/s and reduces rows over 0.1 rad/s from 1,300 to 1,196,
but it worsens the front-below-peak and turn-in/reversal cohorts. A simple
regime screen uses the published lateral extremum as a fixed boundary:
select the slip-augmented yaw prediction only when the *estimated*
`max(|Sy_front-left|, |Sy_front-right|) >= 0.01`; otherwise retain the
sensor-only baseline. On r02 this selects 54.5% of rows, has 94.5% precision
and 99.4% recall for the GT-derived `max(|Sy_front|)>=0.01` regime, and gives:

| r02 sensor-only yaw estimate | RMSE | p95 absolute | Rows over 0.1 rad/s |
|---|---:|---:|---:|
| Baseline | 0.03720 | 0.08016 | 1,300 |
| Slip augmentation, ungated | 0.03622 | 0.07922 | 1,196 |
| Fixed `|Sy|≥0.01` regime gate | **0.03473** | **0.07436** | **1,085** |

The gated-vs-baseline paired 479-condition bootstrap gives an equal-condition
macro-RMSE delta of `-0.00235` rad/s (95% interval `[-0.00267,-0.00204]`);
gated-vs-ungated is `-0.00130` (95% interval `[-0.00153,-0.00108]`). This is
a promising *offline candidate*, not production promotion: we inspected r02
to identify the gate, so r02 is now development evidence rather than a final
blind test. The midpoint r04 whole-run capture remains the untouched transfer
check and its active bag is still not being read. After r04 closes, evaluate
the frozen sensor observer and this unretuned `0.01` gate; only then decide
whether the gain is robust enough to port into odometry. No production
odometry/MPC, tire physics, or runtime has been changed.

At this progress check r04 has completed 706/1,400 approach phases; those are
not completed test conditions because their settle/probe phases still follow.
The recording process is active, so its bag remains unopened.

The saved slip-error breakdown at
`live_runs/racing_model_diagnostics_20261009/sensor_slip_observer_r02/slip_error_by_speed_steering.json`
checks the full measured speed and steering bands, with exact-two r02 row
alignment verified against the closed bag. Front lateral slip-estimation RMSE
is 0.00330–0.00414 over 2–10 m/s; it rises to 0.00576 at 10–12 m/s, where
there are only 445 held-out rows. By steering magnitude, its RMSE is 0.00268
below 0.1 rad and 0.00360–0.00467 from 0.1–0.5 rad. The worst populated
speed/steering cell is 2–4 m/s at 0.3–0.4 rad (RMSE 0.00755, p95 0.01574),
so the lateral estimate is not uniformly precise at the low-speed/high-angle
corner.

Rear longitudinal slip is particularly unreliable near low forward speed:
the 0–2 m/s cohort has RMSE 0.0843 and p95 0.180, consistent with the
`Sx` ratio becoming ill-conditioned as wheel-local `Vx` approaches the 0.5
m/s validity threshold. The front-slip gate's yaw RMSE improves across the
2–10 m/s cohorts, is nearly neutral at 0.4–0.5 rad, slightly worsens at
0–2 m/s (0.04623→0.04652), and has only 445 rows at 10–12 m/s. This reinforces
that the current gate is a promising high-slip candidate, not a universal
correction. Do not add the rear `Sx` estimate or the gate to production until
the frozen model/rule also transfers on the closed r04 midpoint capture.

### Continuation — front-slip-specific observer candidate (2026-10-09)

The shared slip observer trained all six slip values and their validity flags
as one multi-output regression problem. A focused offline candidate now
predicts only front-left/right lateral `Sy`, and only from rows where both
front GT-derived slip labels are valid. The other slip outputs and validity
estimates remain those of the shared observer. This isolates the slip
quantity that helped yaw prediction instead of letting the poorly estimated
rear-longitudinal `Sx` outputs and validity flags influence the same tree
splits.

The implementation is
[`compare_front_slip_observer.py`](../../tools/racing/specialists/compare_front_slip_observer.py).
It keeps the same causal feature history, reflection augmentation, phase
weights, and ExtraTrees family. It selects the approach only by whole-run
out-of-fold results on training r01/r03; already-inspected r02 is development
scoring, not a blind final test. The closed exact-two row extracts are cached
at `live_runs/racing_model_diagnostics_20261009/front_slip_observer_candidate_r02/exact_two_row_cache/`.
The active midpoint r04 bag was not opened.

| Evaluation | Shared front-slip RMSE / p95 | Front-specific RMSE / p95 | RMSE change |
|---|---:|---:|---:|
| r01/r03 whole-run OOF, 78,924 paired wheel labels | 0.003804 / 0.007903 | **0.001657 / 0.002572** | **−56.4%** |
| r02 development capture, 40,518 paired wheel labels | 0.003647 / 0.007518 | **0.001619 / 0.002494** | **−55.6%** |

The gain appears across the measured domain: whole-run OOF RMSE at 0.3–0.4
rad falls 0.00476→0.00226, at 0.4–0.5 rad 0.00360→0.00174, and at 10–12
m/s 0.00688→0.00396. The 10–12 m/s band has only 1,734 paired OOF wheel
labels (890 in r02), so this high-speed estimate still needs transfer support.

With the same fixed published front-slip peak gate
(`max(|Sy_front|) >= 0.01`), the revised observer gives r02 yaw residual
RMSE 0.03371 and 978 samples over 0.1 rad/s, versus baseline 0.03720 and
1,300 such samples. That is 24.8% fewer >0.1-rad/s errors than baseline and
better than the previous shared-slip gate (0.03473 RMSE / 1,085 errors). The
estimated gate has 98.8% precision and 99.6% recall for the GT-derived
front-slip-peak regime; p95 yaw error falls 0.08016→0.07129 rad/s. Maximum
error remains about 0.949 rad/s, slightly above baseline's 0.932, so the
candidate does not solve the worst outliers or establish a universal
correction.

The paired, equal-condition bootstrap compares the front-specific gated
candidate with baseline across 479 r02 conditions: median macro-RMSE delta is
−0.00333 rad/s, 95% interval [−0.00374, −0.00293]. This is consistency across
conditions within one capture, not uncertainty across independent runs. By
true front-slip region, the gate changes >0.1-rad/s counts from 597→351 in
the peak-to-asymptote band, 129→50 at/beyond the asymptote, and 542→545 below
the peak; the low-slip change is essentially neutral in RMSE (0.03735→0.03714)
but not an improvement in threshold-exceedance count.

By commanded phase event, the gate reduces unwind errors >0.1 from 764 to
448 (unwind RMSE 0.03856→0.03229), and turn-in errors from 298 to 291. It
does **not** reduce reversal threshold errors (238→239), though reversal RMSE
improves slightly (0.03255→0.03175). Front-slip estimate RMSE itself is
0.00139 on unwind and 0.00156 on reversal. That points to a remaining
reversal yaw-response issue rather than simply a poor front-slip estimate;
do not claim the slip feature solves reversal.

Treat this as a frozen research candidate, not a production change. Score it
once on the closed, untouched r04 midpoint capture without retuning. If it
transfers, expose the front-slip estimator as a separate sensor-only branch;
keep rear `Sx` out of the yaw correction until its high-wheelspin estimate
improves. If it fails transfer, retain the current observer and diagnose the
specific r04 slip regime. No production odometry/MPC, simulator physics, or
controller behavior has changed.

At the latest safe log check, r04 had completed 1,134/1,400 approach phases.
These are not 1,054 complete conditions: each still requires settle/probe
work. The live bag remains sealed.

### Hidden-factor screen: slip interactions and relaxation memory (2026-10-09)

The 24.8% figure above is a reduction in the count of samples exceeding
0.1 rad/s on one development capture. It is **not** 25% of total error energy
explained, and the residual count cannot be interpreted as one 75% hidden
factor. This distinction matters when deciding what the remaining errors mean.

Using only the closed r01/r03/r02 exact-two row caches, I screened two
physically motivated additions to the GT-state-plus-current-slip oracle:
(1) rear combined-slip interactions and slip-modulated roll/IMU loading, and
(2) causal 25/50/100/200-ms slip relaxation state. Each was mirror-augmented;
r01 and r03 were held out one whole run at a time. r02 remains development
evidence, not a blind final test. The active r04 bag was not read.

| Diagnostic oracle | r01 OOF RMSE | r03 OOF RMSE | r02 RMSE | r02 >0.1 | r02 maximum |
|---|---:|---:|---:|---:|---:|
| GT state + current slip | 0.03354 | 0.03554 | 0.03477 | 1,058 | 0.940 |
| + rear combined-slip products/utilization | 0.03420 | 0.03446 | 0.03515 | 1,060 | 0.945 |
| + roll/ay/steering-rate × slip proxies | 0.03317 | 0.03472 | 0.03636 | 1,165 | 0.950 |
| + 25-ms prior slip and slip rate | **0.03225** | 0.03395 | **0.03257** | **876** | 0.956 |
| + 50-ms prior slip and slip rate | 0.03483 | 0.03477 | 0.03543 | 1,072 | 0.977 |
| + 100-ms prior slip and slip rate | 0.03234 | **0.03369** | 0.03373 | 948 | 0.990 |
| + 200-ms prior slip and slip rate | 0.03253 | 0.03404 | 0.03537 | 1,102 | 0.978 |

The 25-ms relaxation candidate improves RMSE over the current-slip oracle in
both independent training-run folds and on r02; its r02 RMSE delta is
−0.00220 rad/s and it removes 182 additional >0.1 samples relative to that
oracle. The 100-ms candidate also improves all three RMSE comparisons, but
its r02 gain is smaller. The 50/200-ms variants are inconsistent or worse on
r02. Rear combined-slip terms and roll/IMU modulation do not beat the
instantaneous-slip oracle on r02; therefore neither is presently supported as
the missing dominant factor. The maximum error remains about 0.956 rad/s even
with the best lag candidate, so the large wrong-sign tail is not solved.

The lag terms are derived from prior rows within the same response phase, on
the fixed 25-ms sample index and admitted only when event-age separation is
within 12 ms of the requested lag. They are **GT oracle histories**, not
runtime measurements or a deployable observer. Their predictive value is
evidence for a short slip-dependent response-memory state, but not proof of a
specific tire-force relaxation law: slip itself, model residual structure,
and phase can still be correlated. The next check stratifies the frozen
current-slip, 25-ms, and 100-ms oracle scores at the measured 1-m/s ×
0.05-rad × event resolution. Only if the gain localizes consistently will a
sensor-only estimator for this lag state be evaluated; the per-regime atlas
and r04 transfer check remain necessary before promotion.

Analysis implementation:
`tools/racing/specialists/evaluate_yaw_hidden_tire_factors.py`.
Initial full-domain screen output:
`live_runs/racing_model_diagnostics_20261009/hidden_tire_factor_screen_r02/report.json`.

The focused region score is now complete at 1-m/s × 0.05-rad absolute
steering × event resolution. This is a pooled model scored separately in each
region—not yet a collection of separately fitted per-cell atlas models. Of
396 possible speed/steering/event combinations, 335 have at least one r02
row, 257 have at least 20 rows, and 61 have no admitted response rows. The
full table, including every count, RMSE, tail count, and maximum, is saved at
`live_runs/racing_model_diagnostics_20261009/hidden_tire_factor_region_atlas_r02/report.json`.

The factor is event-specific. On r02, the sensor-only → current-slip oracle →
current-plus-25-ms-slip oracle changes were:

| Event | RMSE | Samples over 0.1 rad/s |
|---|---:|---:|
| Turn-in | 0.03931 → 0.04048 → **0.03291** | 298 → 336 → **247** |
| Unwind | 0.03856 → 0.03260 → **0.03204** | 764 → 458 → **373** |
| Reversal | **0.03255** → 0.03332 → 0.03321 | **238** → 264 → 256 |

There are no hold rows in these selected transition windows. The 25-ms term
helps most in near-neutral-steering turn-in from roughly 3–12 m/s and in
moderate-steering unwind, particularly around 2–5 m/s. For example, at
3–4 m/s and `|steering|=0.20–0.25 rad` unwind (164 samples), RMSE falls
0.0657→0.0541 and >0.1 errors fall by 23. Conversely, at 0–2 m/s with
`|steering|≥0.30 rad`, some turn-in, reversal, and unwind cells worsen; e.g.
the 1–2 m/s, 0.50-rad unwind cell (44 samples) changes 0.0963→0.1066 RMSE.
Therefore this factor is not a safe global correction and cannot be applied
uniformly across the atlas.

This answers the scope question precisely: the preceding fit used all admitted
training rows to learn one pooled mapping; this follow-up reports its score
for each measured regime. It has **not** yet independently fit or promoted a
model for every region. The next useful step is to replace the GT slip-history
oracle with causal observer estimates and test that feature in the supported
regions, retaining the baseline where it loses. Only then does a
regime-conditioned atlas candidate become implementable; r04 remains the
untouched transfer check.

### Causal sensor-only slip-memory candidate (2026-10-09)

The front-slip observer was then used to estimate both current front `Sy` and
its prior 25-ms value from only the allowed sensor/actuator history. The
observer's predicted prior value is taken from the previous row in the same
phase; both the current and previous estimates are causal. Training yaw
features use whole-run out-of-fold front-slip estimates (r01 predicted from
r03 and vice versa), and r02 is scored without fitting on it.

The front-slip estimator itself remains accurate: whole-run OOF RMSE is
0.00147 on r01 and 0.00184 on r03; r02 RMSE is 0.00162 with p95 absolute
error 0.00249. For r02 yaw prediction:

| Sensor-only yaw candidate | RMSE | p95 | >0.1 rad/s | Maximum |
|---|---:|---:|---:|---:|
| No slip feature | 0.03720 | 0.08016 | 1,300 | 0.932 |
| Estimated current front slip | 0.03445 | 0.07439 | 996 | 0.968 |
| Estimated current + prior 25-ms slip | **0.03322** | 0.07296 | 958 | 0.952 |
| Current-slip gate at fixed `|Sy|=0.01` | 0.03323 | 0.06978 | 918 | 0.968 |
| Current+prior gate at fixed `|Sy|=0.01` | 0.03330 | **0.07001** | **911** | 0.952 |

The estimated 25-ms state retains much of the GT-oracle gain (oracle front-
slip current+prior: RMSE 0.03255, 879 >0.1 samples) without using GT at
inference. However, this is still r02 development evidence and it still has
958 errors above the threshold; it does not meet the requested maximum-error
criterion. Event decomposition shows why a single global rule is inadequate:

| Event | Sensor-only | Estimated current + prior | Interpretation |
|---|---:|---:|---|
| Turn-in | 0.03931 / 298 | **0.03408 / 253** | Prior-slip state materially helps |
| Unwind | 0.03856 / 764 | 0.03194 / 401 | Current slip alone is slightly better (0.03106 / 399) |
| Reversal | **0.03255 / 238** | 0.03474 / 304 | Ungated slip-history candidate is worse |

Each pair is RMSE / number of samples above 0.1 rad/s. The fixed peak gate
reduces the reversal count slightly (to 234) but does not explain or eliminate
the reversal dynamics. The per-speed/steering/event table is recorded at
`live_runs/racing_model_diagnostics_20261009/sensor_slip_relaxation_r02/report.json`;
low-speed/high-steering and near-neutral-steering reversal cells remain
counterexamples. Do not deploy one pooled correction across these regimes.

The observer feature is now practical enough to carry forward as a
regime-conditioned *research candidate*, not a production odometry change.
Next: score a frozen rule that uses the prior-slip estimate on training-only
whole-run folds to choose turn-in versus unwind/reversal handling, then test
that rule once on r04. No tuning may use r04 labels. If event-specific routing
does not preserve transfer and the large-error tail, keep the current
production model and continue diagnosing the unresolved reversal/high-steer
regions. Implementation:
`tools/racing/specialists/evaluate_sensor_only_slip_relaxation.py`.

### Midpoint transfer and remaining-factor screen (2026-10-09)

The midpoint validation run r04 is now closed, audited, and scored. It was not
used for fitting or selecting the candidate. This supersedes the earlier
statement that r04 remained sealed. The run has 112,276 exact-two yaw rows;
the complete 1 m/s × 0.05-rad absolute-steering × event score tables are in
the reports below. There are 334 measured cells with at least 20 rows.

The slip observer was trained on r01/r03, and branch choices were selected on
r02 before r04 was opened. The global row below requires no phase routing;
the phase-routed row is retrospective only:

| r04 model | RMSE (rad/s) | p95 | Samples >0.1 | Maximum |
|---|---:|---:|---:|---:|
| Full-input sensor-only baseline | 0.03825 | 0.08271 | 3,798 | 0.8234 |
| Estimated front slip, current + prior 25 ms, global (no phase routing) | **0.03442** | 0.07521 | **2,976** | 0.9139 |
| Same branches routed by causal command/feedback event feature | 0.03425 | 0.07465 | 2,977 | 0.9151 |
| Estimated slip, post-hoc phase-event-routed upper bound | 0.03317 | **0.06979** | 2,732 | 0.9151 |

The global, no-phase-routing sensor model transfers with 10.0% lower RMSE,
19.0% lower squared error, and 21.6% fewer threshold-exceeding samples; p95
improves, but its maximum is worse. This is the valid full-input sensor
candidate. The 0.03317 score is a post-hoc routing upper bound: the branch
selector reads `phase_event` from the scripted capture schedule (turn-in,
unwind, reversal). That label is unavailable online, so this score must not be
presented as a deployable result. The causal event feature was separately
scored below; it does not recover the phase-routed upper bound. Neither result
meets the requested maximum-error bound.

Using the existing causal command/feedback event feature for the same routing
rule gives 0.03425 RMSE, p95 0.07465, and 2,977 threshold errors—essentially
the same as the simpler global model, and not the phase-label upper bound. In
this r04 capture the causal classifier assigned all 30,167 schedule-reversal
rows to either `hold` (19,007) or `turn_in` (11,160); it emitted no `reversal`
class. This is why the phase-routed branch must not be used as evidence that
reversal has been solved. Keep the global candidate as the cleaner atlas
entry; a more accurate causal reversal detector would need separate work.

The global model's event-stratified scores explain why routing looked
attractive, and why reversals remain the key failure: turn-in changes from
0.04005 RMSE / 762 errors to 0.03506 / 679, unwind from 0.03929 / 2,345 to
0.03291 / 1,433, but reversal worsens from 0.03453 / 691 to 0.03656 / 864.
These event labels are used here only to diagnose the held-out residuals; the
global model itself did not use the labels to select a branch.

The global sensor-only branch is an offline research candidate in
`evaluate_sensor_only_slip_relaxation.py`; it is **not yet in production
odometry**. This is the first unseen-run transfer evidence for that feature,
so preserve it in the yaw-response atlas rather than discard it. It still needs
an implementation-compatible runtime form and a timestamp-aligned odometry
evaluation before integration. No simulator physics or runtime behavior was
changed.

There is a concrete runtime-interface gap: `SensorPacket` and
`OdometryObservation` currently carry encoder angles and IMU values only. The
slip observer also uses steering feedback/command history (and throttle
feedback/command history), so it cannot be dropped into today's odometry
observer as-is. Any port must first establish that these actuator channels are
permitted for the competition odometry process, add source-time-paired inputs,
and re-evaluate without introducing an output timestamp lead/lag.

That interface concern was tested directly. The same observer/yaw pipeline
was retrained with an input mask matching the current C++ packet: rear encoder
speeds plus IMU ax/ay/yaw-rate histories and derivations only; steering,
throttle, roll, and phase-event inputs were removed. On held-out r04, the
front-slip observer RMSE was 0.05446 (whole-run OOF 0.05233/0.05040), versus
0.00476 with the full causal input set. Yaw scores were:

| Current-odometry-input model | RMSE | p95 | >0.1 samples | Maximum |
|---|---:|---:|---:|---:|
| Sensor-only baseline | 0.08774 | 0.14003 | 8,090 | 0.9054 |
| Add estimated current + prior 25-ms front slip | **0.08560** | **0.12686** | **7,235** | 0.9263 |

This is only 2.4% lower RMSE / 4.8% lower squared error and 10.6% fewer
threshold errors; p95 improves, while the maximum gets worse. The resulting
error is much larger than with the full actuator/roll feature set. Therefore
the slip observer is not directly usable in the current production odometry
interface. The held-out result supports adding policy-permitted, timestamped
actuator/steering (and, if allowed, roll) inputs before attempting a runtime
port; merely copying the model into the present encoder/IMU-only node would
discard most of the measured benefit.

To test the hidden-factor hypothesis without using r04 for tuning, a frozen
GT-oracle factor screen was also trained on r01+r03 and scored once on r04.
GT state/slip is diagnostic-only here, not a usable online input:

| Diagnostic model | r04 RMSE | r04 p95 | >0.1 | Maximum |
|---|---:|---:|---:|---:|
| GT body state + current per-wheel slip | 0.03535 | 0.07678 | 3,067 | 0.7305 |
| + rear combined-slip interactions | 0.03580 | 0.07578 | 3,030 | 0.8196 |
| + roll/ay/steering-rate × slip proxies | 0.03678 | 0.07933 | 3,284 | 0.7033 |
| + 25-ms prior slip and slip rate | **0.03354** | **0.07166** | **2,507** | 0.7438 |
| + 100-ms prior slip and slip rate | 0.03466 | 0.07482 | 2,827 | 0.7706 |
| + 50-ms prior slip and slip rate | 0.03640 | 0.07723 | 3,158 | 0.7426 |
| + 200-ms prior slip and slip rate | 0.03615 | 0.07960 | 3,253 | 0.8861 |

The 25-ms history improves on the current-slip oracle by 5.1% RMSE and removes
560 >0.1 samples on r04. This independently supports a short, slip-associated
response-memory state. It does not prove the simulator's exact tire relaxation
equation: slip, turn phase, and other unobserved dynamics can be correlated.
The rear-combined terms make no useful RMSE gain (37 fewer threshold errors at
the cost of worse RMSE and a larger maximum); roll/ay interactions are worse
on both RMSE and threshold count. The tested *proxies* for load transfer are
therefore not supported as the dominant remaining factor. Direct per-wheel
normal loads, tire forces, front-wheel speeds, and suspension deflection were
not recorded, so this screen cannot rule out those actual internal states.

The gain is not universal. The GT-oracle 25-ms term lowers RMSE versus the
sensor-only model in 236/334 supported cells, but loses in the remainder. Its
largest counterexamples are 1–2 m/s at 0.45–0.50 rad during unwind/reversal
(for example, 1–2 m/s, 0.50-rad unwind: RMSE 0.0900→0.1025, >0.1 count 14→20)
and near-neutral 11–12 m/s turn-in (0.1001→0.1023, count remains 42). The
sensor-only event route is less harmful in these regions because it can retain
the baseline on unsupported conditions.

Interpretation of the remaining errors:

- Slip is a real predictive factor. The independent r04 global sensor model
  shows 19.0% lower squared error and 21.6% fewer >0.1 samples without using
  experiment phase labels. The 24.8% MSE / 28.1% threshold-count reductions
  belong to the non-deployable phase-routed upper bound. Neither model removes
  the maximum outlier. The GT-oracle 25-ms model removes 34.0% of >0.1 samples,
  but cannot be used online.
- Simple rear combined-slip and roll/ay load-transfer proxies did not explain
  the remainder on the independent run. More data by itself is not the
  conclusion; the current candidate family is missing state or response
  structure, especially around reversals and the low-speed/high-steering edge.
- Reversal remains the clearest distinct response: even with GT slip history,
  the r04 25-ms oracle has RMSE 0.03527 and 720 >0.1 samples, versus the
  sensor-only baseline's 0.03453 and 691. Slip history is not the reversal fix.
- The next targeted analysis should inspect the largest r04 residuals in the
  failed cells against causal steering onset/reversal age, front-vs-rear and
  left-vs-right slip state/rate, and available wheel/body-speed mismatch. Do
  not add a global correction or fill absent wheel-load signals by assumption.

Machine-readable outputs:

- `live_runs/racing_model_diagnostics_20261009/sensor_slip_relaxation_r04/report.json`
  — sensor-only observer, frozen event route, and full region table.
- `live_runs/racing_model_diagnostics_20261009/hidden_tire_factor_screen_r04/report.json`
  — frozen current/25/100-ms oracle transfer comparison.
- `live_runs/racing_model_diagnostics_20261009/hidden_tire_factor_full_screen_r04/report.json`
  — r04 transfer scores for rear combined slip, roll/ay proxies, and all tested
  slip-memory horizons.

### Actuator response-state residual expert (2026-10-09)

This iteration uses the two full-spectrum training captures r01/r03 for
whole-run out-of-fold residual training and regime selection; r04 is scored as
the midpoint transfer capture. It does **not** fit all bags in the repository
as one homogeneous dataset. The 1,508-phase throttle surface remains a separate
domain/comparator because its excitation and transition coverage differ.

The held-out r04 residual audit showed a repeatable actuator-response state:
the derived feature
`steering_feedback_age - steering_command_age` is at least 50 ms when the
command has changed more recently than its measured feedback. The existing yaw
model already receives this causal feature, so the revision is not an added
input; it fits a residual expert specifically for that state. The gate has no
scripted phase label and uses only current/past command and feedback samples.

On r04, the stale-steering-feedback state occupies 9,362/112,276 rows (8.3%).
The previous selected-regional model had 430/939 of its >0.1-rad/s errors in
those rows (4.59% error rate there versus 0.49% outside). The state also has
ample training support: each held-run fold has 3,309–3,523 gated rows across
1,657–1,767 phases.

The gated residual expert passed both r01↔r03 folds. Pooled OOF correction
residual metrics in the gated state improved from RMSE 0.04488, p95 0.10065,
and 348 samples >0.1 to RMSE 0.02315, p95 0.04864, and 45 samples >0.1 rad/s.
These are cross-run *residual-correction* errors, not full-domain yaw scores.
On r04, the full selected regional candidate reduced errors within the gated
rows from 430 to 145 and reduced whole-run metrics relative to the previous
selected regional model:

| r04 candidate | RMSE (rad/s) | p95 | >0.1 | Maximum |
|---|---:|---:|---:|---:|
| Previous selected regional residual model | 0.02286 | 0.04365 | 939 | 0.93095 |
| Add cross-run-selected steering-feedback-lag expert | **0.02053** | **0.03712** | **654** | 0.93095 |

That is a 10.2% lower RMSE, 15.0% lower p95, and 30.4% fewer threshold errors
on r04. The maximum did not improve; the explicit direct-yaw regional overlay
also remains rejected because it raises the combined threshold count to 698.
No production odometry/MPC model has been changed. The r04 residual inspection
motivated the gate, so r04 is transfer evidence rather than a pristine,
untouched final benchmark. The same frozen gate is now being scored on existing
r02 data before it is treated as a stable atlas entry.

The high-error tail is now visibly split into different response regimes, not
one scalar slip correction. Remaining large examples include late 8–10 m/s
unwind near center steering (the maximum, r04 row 18,903: 9.08 m/s, 226 ms
after the command edge, steering feedback −0.042 rad, yaw target changes from
−0.784 to −1.020 rad/s), as well as near-zero-feedback turn-in/reversal rows
with large command-feedback differences. The worst unwind row has a steering
feedback age gap of −25 ms, so the new lagging-feedback gate correctly does
not claim to fix it. Next, score r02, then isolate the remaining onset/reversal
and late-unwind residuals by their causal command/feedback-age state and
measured speed/steering cells. Do not merge those into the new lag expert.

Machine-readable artifacts:

- `live_runs/racing_model_diagnostics_20261009/actuator_age_gap_residual_r04/report.json`
  — cross-run gate selection, r04 whole-run metrics, and per-regime diagnostics.
- `live_runs/racing_model_diagnostics_20261009/actuator_age_gap_residual_r04/residual_diagnostics/`
  — r04 threshold-error rows for each candidate.
- `tools/racing/specialists/evaluate_sensor_only_slip_relaxation.py`
  — adds the causal steering-feedback-lag residual gate; it remains an offline
  evaluator, not production odometry.

### Remaining-error diagnosis and low/mid-speed unwind specialist (2026-10-09)

The next pass stayed on existing exact-two-packet caches and bags; no simulator
was launched. A proposed 3–4 m/s unwind expert was selected only after both
whole-run r01↔r03 folds improved. Its causal gate is 3–4 m/s rear-wheel mean
speed, 0.10–0.20 rad absolute steering feedback, unwind, and 0–300 ms after
the command edge. On the independent midpoint capture it covers 566 rows:

| Midpoint metric inside this gate | Global age-residual correction | Local residual expert |
|---|---:|---:|
| RMSE | 0.04806 rad/s | **0.03769 rad/s** |
| p95 absolute error | 0.10027 rad/s | **0.07766 rad/s** |
| samples over 0.1 rad/s | 30 | **16** |
| maximum absolute error | **0.28171 rad/s** | 0.29336 rad/s |

This is a targeted, cross-run-supported improvement, not a complete fix: the
maximum in the cell rose slightly and remains far above the 0.1 requirement.
Across the complete r04 run the selected-regional candidate moves from RMSE
0.02053, p95 0.03712, 654 samples >0.1, max 0.93095 to RMSE 0.02044, p95
0.03710, 640 samples >0.1, max 0.93095. The absolute worst case is unchanged.
The separate direct-yaw expert for this gate improves both training folds but
does worse on r04 (34 >0.1 vs 16), so it is not selected. The residual expert
is in the offline evaluator only; no odometry/MPC runtime model changed.

The remaining r04 maximum is an off-grid interpolation/state-conditioning
failure, not explained by the slip proxy alone. It is the 10.00 m/s, 0.425 rad
negative-turn unwind midpoint (measured speed 9.08 m/s, feedback −0.042 rad,
event age 226 ms). Its yaw rate changes from −0.784 to −1.020 rad/s, while the
selected model predicts approximately −0.089 rad/s (0.931 rad/s error). In
training, the neighboring 9.75/10.25 m/s and 0.40/0.45 rad trajectories at
similar turn phase continue yaw toward roughly −0.97 to −1.01 rad/s. Thus the
data support the *direction* of the held-out motion, but the present tree
specialist learned an early release at the between-grid state. One packet
later the held-out yaw collapses, so this is a response-phase boundary that a
smooth/state-conditioned local response model must represent.

A different r02 extreme is a one-packet actuator staircase phase difference:
at the 8.25 m/s, 0.45 rad unwind, r01/r03 drop yaw between the approximately
250- and 275-ms samples, while r02 holds the same measured steering through
the 275-ms sample and drops in the following interval.
At the 250-ms prediction instant, current/previous steering, yaw, GT body
velocity, and reconstructed wheel slip are nearly identical. The phase-level
exact-two quality rule measures command-to-first-feedback onset; it does not
guarantee that every later step of the feedback ramp advances on the same
packet. Consequently, a deterministic next-step predictor from the current
sample cannot know this one-packet-later update from the available history.
Keep this as actuator timing/observability uncertainty, separate from the
off-grid interpolation error. Do not claim either is fixed by the new expert.

The selected residual candidate still has 640/112,276 r04 rows above 0.1
rad/s and a 0.931 maximum. Current work is therefore not complete. Next, test
one targeted continuous/local response model on the 8–10 m/s unwind midpoints,
with whole-run r01↔r03 folds and the frozen r04 capture, and separately test
whether a sensor-only rear-longitudinal-slip estimate adds information in
high-speed near-center unwind. Keep GT slip diagnostic/training-only; do not
feed future truth to either predictor. No new data capture is justified yet.

Machine-readable result:

- `live_runs/racing_model_diagnostics_20261009/continued_error_reduction_20261009/r04_low_mid_unwind_probe/report.json`
  — held-out r04 scores and whole-run expert selection.

### Existing high-speed response data: transfer check (2026-10-09)

The remaining high-speed response errors were checked against the already
captured 10.5/11.1 m/s steering-onset/unwind family (two train runs and one
whole-run validation). All three manifests pass the clean-run gate: complete
schedule, no timing faults/collisions, and at least 99.99% odometry packet
alignment. Only exact-two contiguous response windows were admitted. No new
simulation was run.

A model trained only on the two high-speed family runs looks excellent on its
independent third run in part of that same scheduled family:

| Held-out region | Broad full-spectrum model | High-speed-only model |
|---|---:|---:|
| All 8–12 m/s, `abs(steering)<0.25`, 1,237 rows: RMSE / p95 / >0.1 / max | 0.0703 / 0.1301 / 78 / 0.4717 | 0.0262 / 0.0264 / 14 / 0.2715 |
| Unwind, 8–12 m/s, `0.10≤abs(steering)<0.25`, 195 rows | 0.1175 / 0.4199 / 14 / 0.4717 | **0.0115 / 0.0276 / 0 / 0.0512** |
| Unwind, 8–12 m/s, `abs(steering)<0.10`, 410 rows | 0.0385 / 0.0936 / 21 / 0.2430 | 0.0173 / 0.0325 / 1 / 0.2715 |

This does **not** transfer to the independently captured full-spectrum runs.
For the same observable 8–12 m/s, `abs(steering)<0.25` gate, the high-speed-only
model increases r02 errors >0.1 from 314/10,650 to 1,689/10,650 and r04 errors
from 947/29,553 to 4,650/29,553. Its RMSE rises from 0.0390 to 0.1428 on r02
and 0.0412 to 0.1374 on r04; maxima rise to 1.075 rad/s. Mixing those runs
into broad training also fails: r02 threshold errors rise 314→392 and r04
947→1,207, with the maximum worsening on both. Thus these captures reveal a
highly repeatable *schedule-family response*, not a safe global or deployable
speed/steering expert. Neither candidate is promoted.

This is important for the outstanding large errors: they are not solved by
adding another small set of high-speed training rows or by selecting an expert
using only speed and steering. The r04 0.931 rad/s late-unwind maximum remains.
Even pooling both high-speed runs into the broad training set regresses the
full-spectrum gates: r02 RMSE/threshold-count/max changes `0.0390/314/0.932`
to `0.0424/392/0.940`, and r04 changes `0.0412/947/0.823` to
`0.0444/1,207/0.856`. These are yaw-rate residual metrics in rad/s and counts.
The reproducible comparison and >0.1 error rows are in
`tools/racing/specialists/evaluate_existing_highspeed_yaw_transfer.py` and
`live_runs/racing_model_diagnostics_20261009/continued_error_reduction_20261009/highspeed_pooled_transfer/`.
The existing per-wheel `WheelCollider` capture is present at
`live_runs/openplane_source_player_equivalence_retry2_20260927/`, but that
player's high-angle response failed equivalence against the pinned player; its
internal data are therefore not valid quantitative training labels for this
model. The workspace has the recorder source but no Unity project/controller
source or installed Unity editor with which to build the matching player. The
next model-identification step requires restoring that exact build path (or
providing an equivalent source/player pair); otherwise use the current
sensor-only measurements and keep the per-wheel explanation explicitly
unconfirmed. No production odometry or MPC changes have been made.

### Broad-data high-speed regional yaw candidate (2026-10-09)

The schedule-only specialist was rejected, but a different local architecture
was then tested on the full-spectrum captures: a left/right-symmetric
`HistGradientBoostingRegressor`, trained only on exact-two rows in the measured
8–12 m/s and `abs(steering)<0.25 rad` region. The gate uses runtime-measurable
rear-wheel mean speed and steering; scheduled event labels are used only for
report stratification, not inference. Training used the two whole-run grid
training captures (20,702 regional rows). Before opening r02/r04, the model
passed both leave-one-training-run-out folds: RMSE changed `0.04170→0.02537`
and >0.1 errors `336→76` on the first fold, and `0.03731→0.01891` and
`294→47` on the second. The maximum increased slightly in each fold.

On the independent full-spectrum validations, within the same 8–12 m/s,
`abs(steering)<0.25` gate:

| Capture | Broad model RMSE / p95 / >0.1 / max | Regional boosted model RMSE / p95 / >0.1 / max |
|---|---:|---:|
| r02, 10,650 rows | 0.0390 / 0.0782 / 314 / 0.932 | **0.0223 / 0.0372 / 38 / 1.001** |
| r04, 29,553 rows | 0.0412 / 0.0846 / 947 / 0.823 | **0.0300 / 0.0696 / 381 / 0.621** |

In the 8–12 m/s unwind slice at `0.10≤abs(steering)<0.25`, errors over 0.1
fall from 40 to 1 on r02 and 53 to 2 on r04; respective maxima are 0.113 and
0.151 rad/s. The result is a genuine, whole-run-transfer gain in this supported
high-speed response region, unlike the schedule-only fit. It is retained as an
offline regional candidate.

It does **not** meet the requested maximum-error criterion. On r02 it worsens
the single worst 8–12 m/s gate error from 0.932 to 1.001 rad/s; on r04 it
reduces but leaves a 0.621 rad/s maximum. The r02 maximum is a 249.8-ms unwind
sample at 8.448 m/s and 0.045-rad feedback: GT yaw-rate residual is +0.023
rad/s while the model predicts −0.977 rad/s. The r04 maximum is a 226-ms
unwind sample at 9.016 m/s and −0.042-rad feedback: target −0.236 versus
prediction +0.385 rad/s. These are opposite response-branch errors near
center steering, not a remaining broad speed/steering coverage gap. The model
is a piecewise tree ensemble, not a smooth interpolator, and it is not yet
wired into odometry or MPC. Production odometry currently uses measured IMU
yaw and yaw rate directly; replacing that measurement with a next-step learned
mapping would be inappropriate. Recursive use for future plant prediction is
not yet validated.

Machine-readable cross-fold/held-out scores and all candidate >0.1 errors:
`live_runs/racing_model_diagnostics_20261009/continued_error_reduction_20261009/highspeed_regional_surface_transfer/report.json` and
`.../highspeed_specialist_errors_over_0p1.csv`. The reproducible fitter and
evaluator is `tools/racing/specialists/evaluate_existing_highspeed_yaw_transfer.py`.
Next, isolate why this local model still chooses the wrong yaw branch on the
rare unwind states; do not promote it to runtime solely from its aggregate
gain. No simulator run, physics change, odometry change, or MPC change occurred.

#### Does 500 ms of causal history resolve the release branch?

One targeted repeat added 200/300/500-ms sensor history to the same regional
boosting architecture. It does not resolve the maximum: the r02 249.8-ms
unwind sample at 8.448 m/s remains the worst case and changes from 1.001 to
1.014 rad/s error. In the complete 8–12 m/s gate, >0.1 counts are 38→38 on
r02 and 381→644 on r04; r04 maximum improves only 0.621→0.592 rad/s while
the error count rises. At 0.10–0.25 rad steering during unwind, the 100-ms
model has 1/1,750 and 2/4,328 errors on r02/r04; the 500-ms version still has
1 and 2, with maxima 0.129 and 0.157 rad/s. Retaining more causal history is
therefore not the missing explanation and the 500-ms variant is rejected for
this regime. Results are in
`live_runs/racing_model_diagnostics_20261009/continued_error_reduction_20261009/highspeed_regional_surface_history500ms/report.json`.

This narrows the remaining branch ambiguity: it persists with 500 ms of
measured steering, wheel-speed, IMU, roll and actuator history, and with a
model trained on 20,702 relevant rows. Existing body-level truth/slip-oracle
features also did not remove the maximum. A deterministic causal predictor
cannot select the correct next-step branch from these available signals in
every sample. The outstanding causal candidates are internal wheel/contact
state or a genuinely unobserved per-step actuator update. The only existing
WheelCollider telemetry was captured on a source-built player that failed
high-angle equivalence, so it cannot identify the pinned-player branch. More
sensor-only fitting or repeating the same excitation is not justified as the
next step; exact-build internal telemetry is needed to distinguish those
causes. This does not invalidate the regional 100-ms candidate's substantial
cross-run error reduction, but it prevents a claim that the yaw model now
meets the all-samples 0.1 rad/s requirement.

#### Further checks on the remaining high-speed near-center unwind errors

The worst r02/r04 samples were decomposed as
`GT_yaw_rate[k+1] - IMU_yaw_rate[k] = (GT_yaw_rate[k+1] - GT_yaw_rate[k]) + (GT_yaw_rate[k] - IMU_yaw_rate[k])`.
On the selected high-speed near-center unwind rows, the current GT/IMU yaw-rate
offset is exactly zero (including each of the five largest candidate errors).
The misses are therefore not caused by a current gyro bias or that label
offset; they are errors in predicting the next physical yaw response.

I also trained the same regional tree architecture against absolute next-step
GT yaw rate, converting its output back to the required residual at inference.
This is a targeted loss/target formulation comparison, not a runtime change.
It helped the *count* in the center-unwind subregion (r02 `23→10`, r04
`249→194` errors above 0.1), but worsened the regional held-out results:

| Capture/gate | Residual-target regional model | Absolute-next-yaw target |
|---|---:|---:|
| r02 full 8–12 m/s, `abs(steering)<0.25` | RMSE 0.02229, >0.1 `38`, max 1.001 | RMSE 0.02400, >0.1 `41`, max 1.019 |
| r04 full 8–12 m/s, `abs(steering)<0.25` | RMSE 0.03001, >0.1 `381`, max 0.621 | RMSE 0.03167, >0.1 `420`, max 0.884 |
| r02 center unwind | RMSE 0.02878, >0.1 `23`, max 1.001 | RMSE 0.03239, >0.1 `10`, max 1.019 |
| r04 center unwind | RMSE 0.03343, >0.1 `249`, max 0.621 | RMSE 0.03780, >0.1 `194`, max 0.884 |

The center-unwind count reduction is not a reliable full-region or worst-case
improvement, so this target change is rejected. It cannot be used to claim the
large errors are fixed.

Two physical-feature checks were also made. First, the reset-initialized
sensor-only integral of `v_dot = IMU_ay - IMU_yaw_rate * mean_rear_wheel_speed`
has about 0.10 m/s whole-run RMSE against GT lateral velocity and only ~0.28
correlation. In the high-speed, low-steering validation region, RMSE is
0.11–0.12 m/s while GT lateral velocity RMS is only ~0.04 m/s, and correlation
is negative. The integral is dominated by drift there; it is rejected as a
runtime feature.

Second, the non-integrated interaction `IMU_ay - r*u_wheel` was added at each
existing 0/25/50/100-ms history lag and tested on the same run splits. It
slightly improved both training-run folds, but did not transfer consistently:
the full-gate r02 count changes `38→36` while r04 changes `381→396`; the r04
0.10–0.25-rad unwind count changes `2→4`. It is not promoted. Scores are in
`live_runs/racing_model_diagnostics_20261009/continued_error_reduction_20261009/highspeed_regional_surface_lateral_accel/report.json`.
The experiment is implemented in
`tools/racing/specialists/evaluate_existing_highspeed_yaw_transfer.py`.

Current result: no production odometry or MPC changes. The regional HGB
surface remains a useful *offline* improvement in most of its supported
high-speed/steering distribution, but neither added history, target
reparameterization, nor the explicit lateral-acceleration interaction
resolves the rare wrong-sign tail. The remaining large cases have matching
current GT/IMU yaw and adequate measured-state coverage. A useful next
measurement must discriminate the hidden response state—particularly
per-wheel tire/contact response or the simulator's internal actuator update—on
the exact player build. Existing source-build WheelCollider samples are not a
valid substitute because that player failed high-angle equivalence. Until
that measurement is available, more sensor-only fitting of these same inputs
cannot honestly be called a fix. No simulator was launched and no production
runtime behavior was changed in this continuation.
