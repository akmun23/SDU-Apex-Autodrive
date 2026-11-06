# Progress-aware corridor merit: implemented and strict Unity tested

## Confirmed gap
The QP already propagates progress-deviation state through the full 30-stage source-frame linearization, but the full-footprint soft wall-reserve objective originally used only lateral and heading error. The exact nonlinear merit evaluates left/right corridor bounds at the candidate's **actual** progress, so a faster trajectory can encounter a narrower wall than the nominal QP assumes.

## Change
- New opt-in `source_progress_aware_reserve` (default false) applies a GN derivative to the **existing** near-wall reserve merit using the loaded reference bounds sampled at `s +/- 0.002m` and the original footprint margin routine.
- The progress deviation variable is zero at the source nominal; the additional QP gradient is `2*w*deficit*dslope` and its Hessian is PSD (`2*w*dslope^2`). This adds no plant integrations, no hard wall constraints and no fallback path.
- `MpcSqpClearanceMerit.ProgressAwareReserveDerivativeMatchesExactMargin` checks seven stages against physical margin finite differences on a nonconstant-width track. It initially caught the left/right lower/upper slack mapping and passes after the correction.
- Full ROS MPC package `259 tests, 0 failures`; Release build succeeded, `git diff --check` clean. Canonical practice/competition configurations unchanged.

## Strict Unity competition experiment
Config: `live_runs/mpc_source_phase1_trial_20261102/comp_progress_aware_20k.yaml`, based on the source-informed Phase-I + soft-corridor-reserve 20k experimental overlay. Output: `live_runs/comp_progress_aware_20k_20261102_10lap_01/`.

| Configuration | Completed laps | Max station | P95 / max control end-to-end | Outcome |
|---|---:|---:|---:|---|
| Progress-aware QP (`...comp_progress_aware_20k...`) | 1 | **105.646m** | 14.664 / 19.290ms | `corridor@0` rejection |
| Same 20k reserve, no progress correction, run 1 | 2 | 148.419m | 15.167 / 18.454ms | `corridor@0` rejection |
| Same 20k reserve, no progress correction, run 2 | 1 | 123.195m | 14.345 / 19.235ms | `corridor@0` rejection |

Progress-aware run: 450 full-horizon accepted cycles (242 `accepted_optimal`, 208 `accepted_line_search`), four previously certified plan publications, one strict `rejected_nonlinear_rollout` at stage 0; no collision recorded. The last accepted source horizon slack fell from 9.3mm to 0.9mm. This is **not** a racing improvement or a repeatable full-run qualification.

## Disposition
**Retain opt-in experimental mathematics and numerical test, disabled in production.** Do not increase its weight or remove recovery behaviours based on this result. The derivative is locally valid, but it does not replace full-horizon source rollout acceptance or provide recursive feasibility. Next identify why the normal SQP chooses actions that reduce next-state feasible control options, using delayed-command causality, source/Unity synchronization, and a source-validated viability/backup trajectory that the normal QP optimizes *before* approaching the wall. The 20ms publication deadline still needs worst-case margin, not just mean performance.
