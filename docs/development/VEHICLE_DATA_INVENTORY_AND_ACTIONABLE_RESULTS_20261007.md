# Vehicle data inventory and actionable results — 2026-10-07

## Bottom line

There is substantial existing data, and some of it now supports specific
measured conclusions. It does **not** yet support a general-purpose vehicle
model, a full speed-by-steering lateral-acceleration limit, or a faster
production raceline. No new simulator capture was started for this analysis.

The largest actionable result is a narrowly valid throttle transient model:
on held-out whole captures it predicts paired step-versus-ramp wheel/body
mismatch and a wheel-slip proxy much better than a coarse speed/throttle model.
It does not predict absolute motion, and its integrated acceleration result is
inconclusive. A broad throttle-ramp controller candidate did not improve lap
times, so it remains disabled by default.

The fixed speed × steering table is only a coarse coverage audit, not the
modeling strategy and not an optimizer envelope. Its edges are not discovered
physical regimes. I should not have described its per-cell heuristic as a
capability limit. The table only demonstrates that this archive has weak
support in many joint operating conditions.

## Where the data is and how to browse it

`live_runs/` is intentionally local/ignored. The complete bag index is
[`live_runs/INDEX_20261007.csv`](../../live_runs/INDEX_20261007.csv). It lists
all 495 `.db3` files, their category, run directory, size, path, and nearby
metadata/log files. Rebuild it with:

```sh
python3 tools/racing/inventory_live_run_bags.py
```

The index distinguishes 453 original `run_0.db3` capture files from 42
`replayed_0.db3`/diagnostic bags produced by odometry replay or diagnostics;
the replay bags are not additional independent simulator runs. There are 55
standard `run_manifest.json` files and 7,356 total files under `live_runs/`
(including this generated index).

| Data group | Bag files | Bag size | What is there |
|---|---:|---:|---|
| Open-plane dynamics/throttle | 260 | 10.92 GiB | Excitation, throttle transitions, steering/swerve, braking, and their derived replay outputs |
| Practice/racing | 114 | 3.21 GiB | Lap attempts, reference/candidate runs, and raceline/controller evaluations |
| Odometry/localization | 60 | 2.71 GiB | Odom, AMCL, localization and observer runs/replays |
| MPC/controller | 39 | 1.52 GiB | MPC/controller diagnostics and candidate runs |
| Mapping/SLAM | 2 | 0.53 GiB | Practice-map captures |
| Competition/bridge | 7 | 0.42 GiB | Bridge/competition diagnostics |
| Other | 13 | 0.46 GiB | Unclassified or mixed-purpose captures |
| **All bag files** | **495** | **19.78 GiB** | 453 original capture bags + 42 replay/diagnostic bags |

`live_runs/` occupies about 26 GiB total; the derived dynamics tree occupies
about 5.7 GiB. These totals include non-bag artifacts and are rounded by the
filesystem. The two fine throttle-sweep source bags alone occupy about 3.1
GiB. Large files are not automatically redundant: those two bags contain the
reset-isolated 1,508-condition throttle sweep.

## Existing datasets and findings

### 1. Fine throttle-step sweep: retain; do not mistake it for a plant model

Source bags:

- `live_runs/openplane_throttle_5pct_5deg_20260930_r04/run/run_0.db3`
- `live_runs/openplane_throttle_5pct_5deg_20260930_r05_resume/run/run_0.db3`

These contain 1,508 reset-isolated throttle transitions spanning 754
steering/throttle conditions with two repeats per key. The fixed-steering
surface is useful evidence about throttle response, but it does not by itself
represent a racing swerve or a recursively usable plant. The prior
held-out-steering endpoint-speed fit was poor (RMSE 2.829 m/s, R² −0.023).
That is a failed generalization result, not a reason to discard the raw
measurements. Its historical analysis is
[`OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md`](OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md).

### 2. Paired throttle slew/step data: useful local transient prediction

Frozen confirmatory artifact:

`live_runs/derived_dynamics_learning_20260928/swerve_throttle_slew_20261007/empirical_throttle_response_confirmatory_20261007.json`

It fits 333 valid pairs from 13 independent captures and scores 96 pairs from
four held-out 6.5/7.5 m/s captures; test/final-test splits were not loaded.
The mirror-pooled exact regime-cell lookup predicts **step-minus-ramp
effects**, not the vehicle's full response:

| Held-out target/window | Exact-cell RMSE | Coarse speed/throttle RMSE | Interpretation |
|---|---:|---:|---|
| Wheel/body speed mismatch, 0.10–1.00 s | 0.01375 m/s | 0.13327 m/s | Exact local response substantially better; run-cluster CI favors it |
| Early mismatch, 0.10–0.35 s | 0.04174 m/s | 0.20101 m/s | Improvement repeats across all four held-out captures |
| Early wheel-slip proxy, 0.10–0.35 s | 0.00603 | 0.02542 | Improvement repeats across all four held-out captures |
| Integrated longitudinal acceleration | 0.06394 m/s² | 0.44399 m/s² | CI includes zero; no demonstrated acceleration-model gain |

The strongest specific control observation is at 6.5 m/s, 0.08 rad steering,
and a +0.08 throttle change: a 0.15 s ramp beat a step on mismatch, slip
proxy, and integrated acceleration in two training and two held-out captures.
It is a narrow supported condition, not a general throttle law. A different
7.5 m/s/high-steering recommendation failed its held-out direction check.
Full methods and results are in
[`THROTTLE_RESPONSE_DATA_REUSE_AND_RACE_DOMAIN_TEST_20261007.md`](THROTTLE_RESPONSE_DATA_REUSE_AND_RACE_DOMAIN_TEST_20261007.md).

### 3. Coarse lateral-data coverage audit (not a vehicle model)

The corrected descriptive audit is implemented in
[`analyze_lateral_speed_steering_surface.py`](../../tools/racing/analyze_lateral_speed_steering_surface.py).
It uses the hashed open-plane dynamics dataset below, only its whole-run
train/validation splits, and simulator-truth body velocity offline. It
estimates `a_y = dv/dt + u*r` on a fixed 25 ms packet grid using a 100 ms
central difference. The estimate is an observed body-acceleration response,
not contact-patch tire force.

Inputs and output:

- Dataset: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/replacement_teacher_dataset_v2_20261004/openplane_dynamics.npz`
- Dataset SHA-256: `2b9623f8ba33f602746e2a29a0fdc9c53e91e349db21cb4e8b1950f2ddce6ed5`
- Results: `live_runs/racing_model_diagnostics_20261007/lateral_speed_steering/summary.json` and `lateral_acceleration_by_speed_steering_demand.csv`

The audit uses 16 train runs (532,612 eligible intervals) and 7 validation
runs (108,318 eligible intervals), summarized into 324 fixed reporting cells.
Those edges were not learned from changes in vehicle behavior; they are not
used as regime definitions or as optimizer constraints. Sealed test splits
were not loaded.

Only 152/324 cells meet the audit's training-support count, 114/324 are
exercised sufficiently to score the validation gate, and two clear that
particular gate. These counts are a warning about sparse *joint* coverage,
not evidence for two physical limits. Above 0.4 rad steering, only 18/108
speed/turn/demand cells have training support and 7/108 are validation
exercised. The CSV retains the descriptive rows for traceability, but its
fixed ranges and derived thresholds must not drive the planner.

The first version mistakenly used signed rather than absolute training
acceleration; this was caught and corrected. That fixed a calculation bug but
does not make the fixed-cell thresholds physically meaningful. The next
analysis must retain continuous measurements and exact capture/phase identity,
then locate where prediction residuals or response mechanisms change; no
hand-set speed/steering rectangles should be promoted as behavior regimes.

### 4. Practice transfer and lap-time status

The frozen reference practice run scored mean 5.7633 s and best 5.7018 s.
The throttle-slew candidate runs were collision-free, but their reported means
were 5.7678–5.7780 s and best laps did not beat 5.7018 s. The best current
record in these runs remains 5.7018 s; no sub-5 s result has been demonstrated.
The broad candidate also increased rejected MPC cycles (six versus one in the
baseline). Its measured benefit was a smaller tail of large wheel/body
mismatch in selected samples, not faster laps or generally improved tracking.

### 5. MPC/odom candidates are not production improvements yet

There are development implementations in the worktree: regime-gated throttle
rise-rate overlays and an MPC yaw-residual candidate. Defaults remain off.
The yaw-residual candidate has promising open-plane one-step/rollout evidence,
but did not establish a successful practice-lap improvement; a related active
optimizer/model-parity gate failed. The empirical curvature candidate also
failed convergence and is not a valid raceline. No candidate has been promoted
to production, and the reference trajectory/physics were not changed here.

That distinction matters: there is **no proven improvement to production MPC
or odometry in this result set**. The validated throttle artifact is an
offline, short-horizon response component that could inform a bounded policy;
it is not safe to insert as a general MPC rollout model. The two lateral cells
are not enough to change the raceline optimizer's full feasibility envelope.

## Cleanup performed and remaining storage

- The two 1,508-condition source captures and other unique raw bags were
  retained; no raw data was deleted based only on size or age.
- Exact byte-identical derived NPZ/checkpoint copies were verified and
  hard-linked under their existing names, saving about 218 MiB while keeping
  every path usable.
- Rebuildable compiler/test artifacts in the yaw-residual integration
  directory were removed (about 12 MiB); analysis/parity results were kept.
- The bag index is now the navigation entry point. `live_runs/` still occupies
  about 26 GiB, so this is an inventory and safe cleanup—not a claim that every
  historical run has been individually proven useful. Further deletion should
  be limited to files proven byte-identical or artifacts whose source,
  consumers, and reproducibility are all recorded; the remaining 42 replay
  bags and historical controller captures have not been declared redundant.

## Decisions

1. Reuse the existing throttle response artifact; do not repeat its grid.
2. Do not enable the broad throttle cap or install the paired lookup as a
   full plant model; the practice laps did not improve.
3. Do not integrate the two lateral caps into the raceline optimizer: their
   narrow open-plane validation is not practice transfer and does not define
   a continuous speed/steering envelope.
4. Keep raw and derived sources indexed. Do not collect another broad sweep.
5. Do not repeat a global nearest-neighbor or generic multi-expert plant fit:
   the earlier WP18 run-balanced neighbor study found no supported history
   plateau, the grey-box candidates failed recursive validation, and the
   multi-expert family did not pass the plant gate. See
   [`BLACK_BOX_OFFLINE_PLANT_PROGRESS_20261003.md`](BLACK_BOX_OFFLINE_PLANT_PROGRESS_20261003.md)
   and [`OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md`](OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md).
6. Build on behavior-specific evidence already supported by interventions.
   Preserve exact condition/phase/run provenance; use continuous measured
   state, command and rate, wheel/body mismatch, and relevant history. Fit and
   tune each response mechanism separately (for example throttle pickup and
   wheelspin, steering turn-in/unwind, braking/release, and combined lateral
   plus longitudinal demand). Let held-out capture residuals determine whether
   a specialist or a transition between specialists is warranted. Unknown
   regions stay unknown; do not fill them with a hand-chosen cap. Score the
   resulting pieces on untouched whole runs before any optimizer or runtime
   integration.
