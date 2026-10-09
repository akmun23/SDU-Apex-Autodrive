# Full-range yaw-model audit under the exact-two-packet rule — 2026-10-08

## Direct answer

The requested accurate model for every speed from 0–12 m/s and full steering
has **not** been delivered. Earlier work concentrated on a narrow high-steering
patch, and our broad-corpus audit failed to include several existing capture
families. We also conflated exact IMU source-stamp alignment with the separate
requirement that command-to-actuator feedback take exactly two simulator
packets. That was an analysis/process failure; it was not evidence that the
existing dataset contained no broader information.

This audit now fits a broad **offline one-step yaw-rate candidate** from
existing admitted captures, using only exactly-two-packet command-response
transitions. It is meaningful progress over the prior narrow-only audit, but
it is not a complete per-regime model, does not meet the requested error
bound, and is not integrated into odometry or MPC.

The saved exact-two local bank has 136 event experts, but they occupy only
82/132 signed 1 m/s × 0.1 rad selector cells: 50 cells have no local expert.
This is the direct reason the request for full-domain per-regime models has
not been met. Coverage is complete at 2–4 m/s, but only 2/11 steering cells
are supported at 7–8 m/s and 4/11 at 10–12 m/s. The full matrix is in the
[exact-two cell atlas](YAW_EXACT_TWO_CELL_ATLAS_20261008.md).

## Data and method

- 61 captures had identifiable yaw-response phases. Training and validation
  splits only were used; test/final-test remained sealed.
- The response-count audit found 5,030 exactly-two-packet events; 648 three-
  packet, 184 one-packet, and all other counts were excluded. The established
  packet counter was checked against the known high-steering r03 capture.
- Current fit/scoring rows: 88,685 training and 31,086 validation rows from
  3,879 and 1,387 exact-two response events, respectively. Validation spans
  19 held-out captures, including the new 9–11.5 m/s high-steering capture.
- Prediction target is next 25-ms simulator-GT yaw rate minus current
  exact-source-time IMU yaw rate. GT is only a training label/scoring target;
  future truth/sensors are not inputs.
- Model inputs are causal sensor/actuator history. An event-specialist router
  uses causal command intent, not the recorded phase label.
- This is one-step yaw-rate prediction in rad/s, not yaw-angle error, recursive
  trajectory prediction, odometry drift, or lap simulation.

## Held-out aggregate result

| Model | RMSE (rad/s) | p95 abs. error | Max abs. error | Samples >0.1 rad/s |
|---|---:|---:|---:|---:|
| IMU persistence | 0.19144 | 0.47090 | 1.74520 | 7,046 / 31,086 |
| Global ExtraTrees | 0.05089 | 0.08395 | 1.63355 | 1,195 / 31,086 (3.84%) |
| Causal event specialists | 0.04151 | 0.05314 | 1.63522 | 745 / 31,086 (2.40%) |

Global model run-macro RMSE is 0.04554 rad/s across 19 held-out runs; the
event-specialist run-macro RMSE is 0.03527 rad/s. The event specialist
improves the aggregate but still fails the requested ≤0.1 bound on 745
samples. Its large maximum is not hidden by the favorable p95.

The separate speed/steering/event local experts cover only 70.2% of validation
rows (21,816/31,086). Their supported-row RMSE is 0.05531 rad/s and maximum
error is 1.00344 rad/s; the bank is not a superior replacement. This does not
prove that regime-specific models are ineffective; the current bin/support
design did not establish a better one.

### Why “models for all regimes” is still not true

There are useful prior full-band prototypes, so saying there are no broader
models at all is inaccurate. The existing GT-conditioned atlas has 478
supported cell/phase experts over 628 occupied speed/steering cells; the
requested grid contains 1,032 cells. It covers 92.17% of its validation
transitions, but its validation was used during model-family development, it
was not filtered to the exact-two packet subset, and its worst one-step yaw-rate
error is 1.221 rad/s. It is a research comparator, not the requested
complete, packet-qualified model.

I also tested a run-held-out support frontier on the exact-two subset, using
only causal wheel speed, measured steering, and maneuver event to select local
experts. Four-fold grouped CV within 42 training captures and whole-capture
validation on 18 separate captures gave:

| Candidate | Grouped-CV coverage | CV RMSE / p95 / max (rad/s) | Held-out coverage | Held-out RMSE / p95 / max (rad/s) | Held-out samples >0.1 |
|---|---:|---:|---:|---:|---:|
| Global event comparator | 100% | 0.04577 / 0.07805 / 1.09577 | 100% | 0.04126 / 0.05377 / 1.61920 | 754 / 31,086 |
| Coarse: 1.0 m/s × 0.10 rad | 55.2% | 0.04330 / 0.07327 / 0.93770 | 70.2% | 0.05565 / 0.06185 / 1.00035 | 728 / 21,816 |
| Medium: 0.5 m/s × 0.05 rad | 40.6% | 0.03982 / 0.06831 / 0.83724 | 57.0% | 0.04350 / 0.04719 / 1.02917 | 416 / 17,713 |
| Fine: 0.5 m/s × 0.025 rad | 45.0% | 0.04744 / 0.07976 / 0.88110 | 62.0% | 0.04832 / 0.04993 / 1.04695 | 496 / 19,275 |

Coverage is the fraction of rows for which that local bank had a directly
supported expert; no global fallback was used. Thus finer binning did not
solve the gap: it reduced support and did not beat the medium bank in grouped
CV. The medium bank has the lowest RMSE among local banks, but leaves 36.6% of
validation rows without a model and still has 416 errors above 0.1 rad/s and a
1.029-rad/s maximum. These metrics do not license extrapolation into empty
cells. Also, the local-bank scores are on different supported subsets; the
frontier is a support/accuracy diagnostic, not a paired promotion test.
Left/right pooling does not justify a universal mirrored selector: on coarse
common support it improves RMSE from 0.05569 to 0.04914 rad/s, but still loses
to the full-coverage event model (0.04126); on medium common support it worsens
RMSE from 0.04350 to 0.04948. Keep signed steering in the selector rather
than mirroring as a general gap-filling shortcut.

The distinction is therefore: broad *prototype models exist*; a validated
model for every physically observed speed × signed-steering × maneuver regime
does not. In the current exact-two held-out set, measured speed spans
0.495–11.256 m/s and steering spans −0.50–+0.50 rad. There are samples in
120/132 signed 1-m/s × 0.1-rad cells, 118 cells have at least ten rows, and
115 have at least twenty. This is coverage, not accuracy certification; the
training-only local model bank still lacks experts in 50 selector cells, and
the high-error transition cases are not solved by filling static cells.
Some combinations in the requested 0–12 m/s × ±0.5-rad rectangle may be
physically unreachable. The new gap-fill captures target the empty empirical
cells first; achieved GT state, not the command schedule, determines whether
each gap was actually filled.

### Error depends strongly on maneuver

| Causal event | Rows | Event-model RMSE | Event-model p95 | Max abs. error | Event-model >0.1 |
|---|---:|---:|---:|---:|---:|
| Hold | 18,057 | 0.02170 | 0.02073 | 0.4073 | 129 |
| Turn-in | 7,404 | 0.03199 | 0.05829 | 0.3954 | 153 |
| Unwind | 4,479 | 0.05906 | 0.11415 | 0.8349 | 277 |
| Reversal | 1,146 | 0.13814 | 0.34725 | 1.6352 | 186 |

Reversal is the worst residual regime; unwind is next. The largest event-model
error is 1.635 rad/s in a reversal at 3.463 m/s and 0.263 rad measured
steering. The rear-wheel/body-speed mismatch there is about 2.17 m/s. That is
a diagnostic association, not yet a proven causal correction.

## How much of the requested domain is represented?

Observed training rows span 0–11.290 m/s and 0–0.50 rad absolute steering.
The new high-speed frontier capture reached 11.483 m/s and ±0.50 rad before
the exact-two filter; the admitted held-out yaw rows span 0.495–11.256 m/s.
Exact standstill and the final 0.744 m/s below 12 remain unobserved in this
exact-two set. A Cartesian grid also includes combinations that may be
physically unreachable; those should be marked from measurements, not
invented by interpolation.

Held-out validation currently occupies 120/132 signed 0.1-rad × 1-m/s cells;
118 have at least 10 rows and 115 at least 20. Steering coverage in the
high-speed bands is materially better than the pre-frontier audit: measured
absolute steering reaches 0.50 rad in 9–10, 10–11, and 11–12 m/s bins. This
does not certify all transitions or accurate predictions in those cells; the
per-cell held-out errors remain listed below and in the separate cell atlas.

| Actual GT speed | Held-out exact-two rows | Steering support in held-out data | Status |
|---|---:|---|---|
| 0–1 m/s | 389 | through ±0.50 rad | No exact standstill samples; 65 rows at ≥0.40 rad |
| 1–2 m/s | 3,501 | through ±0.50 rad | Broad coverage; 878 rows at ≥0.40 rad |
| 2–4 m/s | 11,799 | through ±0.50 rad | Densest range; reversal/unwind errors remain |
| 4–9 m/s | 8,583 | through ±0.42 rad | Sparse high-angle coverage; no measured ±0.50 in these bins |
| 9–10 m/s | 2,096 | through ±0.50 rad | Frontier capture supplies high-angle samples |
| 10–11 m/s | 3,844 | through ±0.50 rad | Frontier capture supplies high-angle samples |
| 11–12 m/s | 874 | through ±0.50 rad; exact-two max 11.256 m/s | No admitted rows near 12 m/s |

Steering and speed coverage counts are observational support, not a guarantee of
accuracy. The held-out per-cell metrics are in the machine-readable report.

### Held-out grid, by measured speed and absolute steering

The following uses the causal event-specialist model (the best broad candidate
in this audit). Each cell is `validation rows / RMSE rad/s [rows with absolute
error >0.1 rad/s]`. Speed is floored to 1 m/s bands; measured steering is
rounded to the nearest 0.1 rad. Left and right are separated. These are
held-out rows, not independent run counts; sparse cells are especially
uncertain.

| GT speed | −0.5 | −0.4 | −0.3 | −0.2 | −0.1 | 0.0 rad |
|---|---:|---:|---:|---:|---:|---:|
| 0–1 | 31/.078[2] | 4/.019[0] | 47/.089[8] | 53/.031[2] | 37/.021[1] | 47/.025[0] |
| 1–2 | 341/.056[19] | 117/.027[1] | 398/.036[13] | 346/.029[8] | 264/.026[4] | 494/.016[2] |
| 2–3 | 444/.046[10] | 525/.039[10] | 369/.036[7] | 251/.059[15] | 269/.028[3] | 1,031/.024[12] |
| 3–4 | 390/.050[7] | 809/.046[12] | 605/.053[16] | 344/.084[19] | 397/.035[13] | 1,665/.022[23] |
| 4–5 | — | 209/.026[4] | 220/.017[2] | 181/.017[2] | 255/.018[0] | 684/.027[8] |
| 5–6 | — | 77/.012[0] | 75/.010[0] | — | 100/.017[0] | 291/.006[0] |
| 6–7 | — | 227/.035[4] | 110/.021[2] | 227/.052[3] | 330/.030[8] | 419/.028[6] |
| 7–8 | — | 169/.013[1] | 132/.004[0] | 99/.011[0] | 178/.041[2] | 413/.016[2] |
| 8–9 | — | 134/.036[2] | 17/.012[0] | 129/.051[1] | 346/.044[3] | 415/.014[1] |
| 9–10 | — | 8/.002[0] | — | — | 185/.018[1] | 248/.007[0] |
| 10–11 | — | — | — | 72/.003[0] | 182/.014[1] | 293/.026[4] |
| 11–12 | — | — | — | 72/.003[0] | 105/.008[0] | 266/.014[1] |

| GT speed | +0.1 | +0.2 | +0.3 | +0.4 | +0.5 |
|---|---:|---:|---:|---:|---:|
| 0–1 | 40/.012[0] | 49/.033[1] | 45/.055[5] | 4/.042[0] | 32/.084[6] |
| 1–2 | 259/.021[2] | 396/.037[16] | 419/.039[16] | 121/.074[11] | 346/.066[26] |
| 2–3 | 271/.035[9] | 254/.058[16] | 398/.054[9] | 521/.031[11] | 548/.037[11] |
| 3–4 | 391/.044[16] | 375/.063[15] | 675/.083[17] | 824/.055[14] | 443/.042[8] |
| 4–5 | 280/.026[4] | 203/.046[3] | 201/.018[2] | 184/.028[5] | — |
| 5–6 | 67/.017[0] | — | 69/.021[1] | 66/.013[0] | — |
| 6–7 | 345/.031[4] | 232/.032[2] | 103/.021[1] | 231/.024[3] | — |
| 7–8 | 166/.026[1] | 92/.009[0] | 109/.004[0] | 164/.028[2] | — |
| 8–9 | 345/.035[3] | 134/.037[2] | 12/.008[0] | 143/.018[2] | — |
| 9–10 | 188/.012[0] | — | — | 8/.012[0] | — |
| 10–11 | 191/.022[2] | 65/.003[0] | — | — | — |
| 11–12 | 100/.007[0] | 53/.003[0] | — | — | — |

This makes the missing surface explicit: validation has no samples in several
high-speed/high-steering cells, and the few 9–10 m/s, 0.4-rad rows are only 16
samples from one throttle-up capture—not a steering transient. At 10–12 m/s,
held-out steering reaches only about 0.2–0.3 rad; the 11–12 m/s rows only
reach about 11.20 m/s. At the other end, 0–1 m/s, |steering|≈0.4 rad has only
8 rows and high-steering cells mostly come from one run. Positive and negative
steering also differ in several occupied cells, so mirroring one side is not
validated. Even where a cell has many samples, reversals can produce large
isolated errors: the worst event-expert miss is 1.617 rad/s in the 3–4 m/s,
roughly +0.3-rad cell. This is evidence of where the broad candidate works and
fails, not evidence that each cell has its own validated model.

The scored target is **next-step yaw-rate residual in rad/s**, not yaw-angle
error in radians. Integrating a rate residual for one 25-ms step gives an
angle increment, but does not establish low yaw-angle drift over a lap or
recursive rollout. The exploratory phase-reset angle integration is not an
acceptance test for the requested full-range observer.

## Work now in progress — 2026-10-08

- The new reset-isolated Explore high-speed validation capture completed all
  192 phases with no collision, tilt abort, or quality failure. It recorded
  17,161 samples over 64 sequences at 39.975 Hz; packet IDs were contiguous,
  and GT speed/steering reached 11.483 m/s and ±0.50 rad. Peak measured roll
  was 4.62°. It remains held out from fitting.
- After adding its 4,778 exact-two rows, the exact-two audit contains 88,685
  training and 31,086 validation rows from 42 and 19 whole runs. On that new
  held-out corpus, causal event specialists score RMSE 0.04151 rad/s, p95
  0.05314 rad/s, max 1.63522 rad/s, with 745/31,086 rows above 0.1 rad/s.
  This improves over the global comparator (0.05089 RMSE, 0.08395 p95), but
  does not satisfy the requested error bound.
- The new frontier run exposes two different failure mechanisms. First, its
  largest event-model miss is 0.625 rad/s during unwind: measured steering is
  already near zero (−0.095 rad), while yaw rate is still −0.604 rad/s. This
  is a delayed/history-dependent yaw response, not a missing static steering
  cell. Second, among its 248/4,778 rows above 0.1 rad/s, absolute prediction
  error correlates with observable throttle command/feedback mismatch (r=0.50)
  and longitudinal acceleration magnitude (r=0.47). In the severe examples,
  body speed is about 10.55 m/s while rear-wheel surface speed is about
  3.1 m/s, throttle feedback is zero against a 0.49 command, and longitudinal
  acceleration is about −9.5 m/s². The model already receives the causal
  throttle, encoder, and IMU histories; this rare coupled regime is poorly
  learned, not unobserved at inference.
- Recomputed the signed cell atlas from the enlarged held-out set. The frozen
  train-only local bank has 50 unsupported cells out of 132. A grouped,
  run-held-out comparison including the frontier capture is complete: none of
  the tested coarse/medium/fine local banks beats the broad causal event model
  overall, and left/right pooling is not consistently safe. The paired
  common-support metrics are in `regime_frontier_sign_shared_with_r02.json`.
- Added `yaw_exact_two_domain_gapfill`: randomized reset-isolated transitions
  at the speed-bin centers for all 50 unsupported signed selector cells (31
  unique speed/steering-magnitude points, both turn directions, two repeats).
  This is a finite, bin-targeted measured-domain gap fill, not a claim that
  every point in the continuous 0–12 m/s × ±0.5-rad rectangle is reachable.
  The first training capture (`openplane_yaw_exact_two_domain_gapfill_train_20261008_r01`)
  completed all 372/372 phases in the established batch-mode Explore
  simulator. It had zero collisions, zero invalid phases, zero quality
  failures, and zero timing faults; the direct command stream averaged
  39.71 Hz. The closed bag exported 29,051 samples over 124 reset-separated
  sequences; packet-ID gap count is zero and recorded sensor streams average
  39.985 Hz (p95 receipt gaps 25.85–25.87 ms). GT speed reached 11.520 m/s and
  steering reached ±0.50 rad. Its manifest and dataset are stored at
  `live_runs/racing_model_diagnostics_20261008/yaw_exact_two_domain_gapfill_train_r01_dataset/`.
  Broad measured occupancy now intersects all 50 frozen unsupported signed
  selector cells. This is not yet an exact-two response-support claim: the
  yaw phase audit must verify that every cell has eligible two-packet response
  windows.
- The independent second training capture
  (`openplane_yaw_exact_two_domain_gapfill_train_20261008_r02`, seed 202610082)
  completed all 372 phases, with zero collisions, zero invalid phases, zero
  quality failures, zero packet-ID gaps, and 39.987-Hz sensor streams. It
  exported 29,044 samples over 124 reset-separated sequences; GT speed reached
  11.523 m/s, steering ±0.50 rad, and GT yaw rate ±2.47 rad/s. Its effective
  split is train. The two independent training captures now provide broad
  occupancy across every frozen unsupported cell; eligibility under the
  strict exact-two response rule remains to be counted by the yaw audit.
- The separate validation capture
  (`openplane_yaw_exact_two_domain_gapfill_validation_20261008_r03`, seed
  202610083) is running in the same batch setup. It is not used for fitting;
  its bag will remain unread until recording finishes.
- Recorder output includes IMU, both encoders, actuator feedback/commands,
  reset edges, simulator packet timing/truth, and phase labels. Exact-two
  audit/cache readers admit newly added clean train/validation runs while
  preserving cached measurements. Test/final-test partitions remain sealed.
- No candidate has been promoted to odometry/MPC. No runtime physics or
  control behavior has been changed.

## Why this did not happen earlier

1. Previous fitting focused on the high-steering response patch and then
   treated those narrow results as the active yaw-model effort instead of
   building the requested full-domain inventory.
2. A prior broad-corpus pass recognized only some `probe_yawerr` labels and
   missed existing `atlas`, `lowyaw`, `yawgap`, and `subnet` capture families.
3. We used “exact packet” ambiguously: exact source-time IMU alignment does
   **not** establish exactly-two-packet command-to-feedback response.
4. Once the exact-two rule was correctly enforced across the added families,
   the broad model still showed a serious reversal/unwind error tail and clear
   high-speed/high-steering coverage gaps. More fitting alone cannot validate
   those absent regions.

The current candidate is one global response model plus four causal maneuver
experts (hold, turn-in, unwind, reversal), not independently identified
responses at every speed/steering point. A separate speed/steering local-expert
attempt fitted 136 experts, but covered only 76.4% of held-out rows and was
slightly worse than the global model on rows it supported; it did not justify
promoting that model bank. The exact-two training rows reach 11.29 m/s and
0.50 rad steering, but those marginal maxima do not mean
every combination was tested. Some Cartesian combinations may be infeasible;
they need evidence-based feasibility labels, not interpolation.

One attempted follow-up diagnostic of later feedback-update “gaps” was
rejected: it measured packets since any prior feedback change, including
ordinary command holds, rather than command-to-feedback latency. Its output
was removed and is not used here. The exact-two admission rule above remains
the only packet-latency result reported.

## Artifacts and status

- [Analysis script](../../tools/racing/specialists/audit_yaw_full_domain_exact_two.py)
- [Machine-readable report](../../live_runs/racing_model_diagnostics_20261008/yaw_full_domain_exact_two_frontier_r02/report.json)
- [Human-readable report](../../live_runs/racing_model_diagnostics_20261008/yaw_full_domain_exact_two_frontier_r02/report.md)
- [Run-held-out local-bin support frontier](../../live_runs/racing_model_diagnostics_20261008/yaw_full_domain_exact_two_frontier_r02/regime_frontier_sign_shared_with_r02.json)
- [Held-out cell-by-cell validation matrix](YAW_EXACT_TWO_CELL_ATLAS_20261008.md)
- [Saved offline candidate and held-out predictions](../../live_runs/racing_model_diagnostics_20261008/yaw_full_domain_exact_two_v2/)

The pinned Explore simulator was launched in Xvfb batch mode for the targeted
frontier capture above. No production odometry/MPC code or runtime physics was
changed. The offline candidate must not be described as a full-range,
production-ready yaw model.

## Next work

1. Finish the current gap-fill run, close/quality-check its bag, then repeat
   with a second independent training seed and one held-out whole-run seed.
   Export on the fixed 25-ms packet grid. Use achieved measured speed/steering
   and exact-two packet counts to decide which selector cells were actually
   filled; record unreachable conditions explicitly.
2. Refit the event and local candidates on training runs only, with grouped
   run-held-out selection. Also compare causal longer steering/actuator
   history (out through 500 ms) against the current 100-ms baseline, since the
   held-out unwind/reversal residuals show memory beyond static speed/angle.
   Report cell-wise coverage, RMSE, p95, maximum, and count above 0.1—not only
   aggregate scores.
3. For every remaining >0.1-rad/s region, distinguish absent support from
   supported-but-mis-modelled response using the held-out per-sample atlas and
   causal input histories. Only collect a targeted repeat when the diagnostic
   identifies a specific missing condition; otherwise revise the response
   model and test on untouched runs.
4. Continue until every empirically reachable speed/steering/event cell has
   held-out support. Mark combinations that cannot be reached under the
   simulator's dynamics as such; do not fill them by unverified interpolation.
   The requested <0.1-rad/s bound remains unmet and no model is promoted to
   odometry/MPC yet.
5. Keep test/final-test partitions sealed throughout model selection.
