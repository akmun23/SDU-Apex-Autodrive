# Vehicle-dynamics data catalog — 2026-09-30

This is the entry point for the current offline vehicle-modeling data. Raw
captures live under the ignored `live_runs/` directory; the tracked dataset
builder is `tools/vehicle_dynamics_learning/prepare_dataset.py`. Do not train
from an arbitrary bag or NPZ: use the split and quality metadata listed here.

## Current usable data

| Question | Authoritative artifact | What it supports / limitation |
|---|---|---|
| Plant teacher over mixed open-plane and racing motion | `live_runs/derived_dynamics_learning_20260928/plant_teacher_mixed_dataset_20260930/` | Schema 3, 16-frame history, 32-step recursive target. 93 clean whole-run records: 63 train, 3 validation, 25 test, 2 final-test. The NPZ is 28.5 MiB; source bag paths and gates are in `manifest.json`. |
| Phase-level salvage from whole-run rejects | `live_runs/derived_dynamics_learning_20260928/boundary_phase_salvage_dataset_20260930/` | Three separate archives: verified phases (1,804 sequences; 908 train, 896 held-out), speed-target-mismatch training-only (45), and unverified/unscored intervals (18; 13 train, 5 held-out). All have local timing, collision/fault, finite-feature, length, and exact-duplicate gates. Do not combine the latter two into the baseline. |
| Continuous-sequence plant alternative | `live_runs/derived_dynamics_learning_20260928/plant_teacher_continuous_dataset_20260930/` | Separate 26.0 MiB derived view with its own manifest; retain for sequence/continuity comparisons, not as an independent test set. |
| Practice transfer data | `live_runs/derived_dynamics_learning_20260928/practice_transfer_dataset_20260930/` | One practice run only; useful for a fixed transfer check, not run-level statistical evidence. |
| Paired throttle ramp/step study | `live_runs/throttle_slew_pair_study_20260928/confirmatory_analysis.json` and the `openplane_throttle_slew_holdout_{45,65}_rNN` bags | Ten clean runs and matched randomized ramp/step pairs at 4.5 and 6.5 m/s. Supports a narrow transient comparison; its encoder/kinematic residual is a proxy, not tire force or direct slip. |
| Fine throttle-transition surface | `live_runs/openplane_throttle_5pct_5deg_20260930_r04/` and `..._r05_resume/` | Closed and analyzed. Combined fit-usable coverage is 1,508/1,508 replicate keys: 754 steering/throttle conditions with two valid repeats each. r05 passed reset, command/feedback, collision/fault, and approximately 40 Hz stream gates. See [`OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md`](OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md). |

The mixed-dataset manifest contains 118 discovered records: 93 clean records
assigned to the four whole-run splits, one clean record excluded for a
different simulator player, and 24 records rejected by the whole-capture
quality gate. Two additional captures failed import because command topics
were absent. The 24 rejected captures were audited independently over 2,095
phase/segment intervals: 1,867 passed for salvage; 228 intervals were
rejected. The accepted intervals yielded 1,867 recursive sequences. Twenty-one
captures yielded at least one sequence.
Three zero-usable captures were removed after the audit. The salvage archives
remain separate from the canonical NPZ and preserve source-run splits.

The current teacher **training split** has maximum speed 8.41 m/s, maximum
throttle feedback 0.50, and no training samples in its 9–10 m/s or 10+ m/s
bins. Its coverage report finds 517/660 coarse speed/steering/throttle cells
observed, but only 239 meet the threshold of 100 samples from at least two
runs. The coarse grid is not a feasibility map. The new throttle sweep is
separate from this archive: r04 observed up to 21.76 m/s and spans full
throttle, but that does not by itself establish a valid recursively usable
plant model. At the measured salvage support, the verified archive peaks at
8.34 m/s—below the canonical 8.41 m/s training maximum. The speed-mismatch and
unverified archives peak at 8.42 and 8.47 m/s, respectively; neither closes
the 9–10 m/s support gap. The throttle sweep observed up to 21.76 m/s in r04
and 20.80 m/s in r05, but its endpoint response fit is not a recursive plant
model. Export reset-delimited 40 Hz trajectories and resolve `/ips` versus
`/odom` before teacher training.

The exporter’s state label is bridge `/odom`; it does not currently use the
recorded `/ips` stream as a separate target. Treat model accuracy as accuracy
against that exported state until `/ips` semantics, frame, and timestamp
alignment are explicitly checked. IMU orientation/rates were found to match
odometry orientation/rates exactly in the earlier audit, so these are not an
independent roll measurement. No per-wheel loads or tire-force labels are
present in the current identified dataset.

## Still pending after phase-level salvage

- The verified salvage contains only phases whose recorded experiment marked
  valid and whose *local* streams, collision/fault state, aligned data, and
  recursive length passed. A replay-named capture was confirmed by its log to
  be a fresh seeded simulator rerun, so its nonduplicate valid phases are
  training data, not a held-out replay.
- `speed_target_mismatch` is not clean matched-speed evidence. Its command and
  state traces may aid plant training, but remain a separate training-only
  archive. `unverified` includes completed unscored phases and bounded
  no-phase-marker segments; its run-level train/test labels are preserved and
  it is not part of the baseline.
- `live_runs/openplane_throttle_5pct_5deg_20260930_r03/` is a closed but
  unanalysed capture. Its bag records a very prolonged first condition; keep
  it pending condition-level review.
- `live_runs/openplane_throttle_transitions_straight_20260930_r01/` stopped
  at 237/5,050 pairs. It may contain useful unique straight-line transitions;
  compare its completed condition keys with r04/r05 before deciding whether
  it belongs in a separate training view. Do not rerun the same sweep until
  this comparison is done.

## Next-step plan

The full handoff-derived implementation order is in
[`OFFLINE_PLANT_NEXT_STEPS_20260930.md`](OFFLINE_PLANT_NEXT_STEPS_20260930.md).
In brief: ingest the completed throttle surface as reset-isolated trajectories;
resolve `/ips` versus `/odom`; compare the frozen GRU against the narrow
physics-structured/latent/residual candidates using whole-run splits; then
score untouched continuous trajectories. No new broad capture is justified
until those results identify a specific missing feasible region.

## Cleanup decisions

Removed as non-dataset clutter or demonstrably failed zero-response attempts:

- `live_runs/vehicle_model_training_env/` and
  `live_runs/derived_dynamics_learning_20260928/model_eval_venv/` (about
  12.2 GiB combined). They were local Python environments, not data, unused by
  any running job, and absent from the dataset pipeline. The two directories
  had drifted versions (Python 3.12.3, NumPy 1.26.4; Torch 2.10.0+cu128 and
  2.14.0+cu130 respectively). Recreate an environment using the setup section
  of `tools/vehicle_dynamics_learning/README.md` when training is needed; do
  not treat these ignored environments as reproducibility artifacts.
- The failed initial throttle-transition invocations `r01` and `r02`, which
  produced no completed fit-usable conditions (bad CLI invocation; then
  reset-recovery abort).
- The two `throttle_surface` attempts `r01` and `r02`, which aborted before a
  usable throttle probe (0 and 2/103 phases, respectively).
- The one-off reset smoke captures `openplane_reset_smoke_20260930_r01` and
  `openplane_throttle_reset_smoke_20260930_full_r01`; their reset check is
  superseded by r04’s 771/771 verified resets and full-rate stream gates.
- Empty/log-only `openplane_high_angle_boundary_20260926_162713`,
  `openplane_high_angle_boundary_20260928_201924`, and the `_clean`, `_live`,
  and `_retry` folders from the 2026-09-26 attempts. They had no bag or
  non-empty diagnostic output. The successful capture
  `openplane_high_angle_boundary_capture_20260926_1636` is retained.
- After the phase-salvage audit, removed the complete run directories for
  `openplane_isolated_boundary_20260926_171623`,
  `openplane_isolated_boundary_smooth_20260926_172247`, and
  `openplane_model_holdout_45_r06`. None contained an interval long enough
  for the recursive dataset after the local gates; their reasons and source
  paths remain in the salvage manifest. The other 21 source captures remain
  because one or more curated sequences were exported from them.

At the latest inventory check, `live_runs/` occupied about 13 GB and contained
At the latest inventory, `live_runs/` occupied about 15 GB and contained 242
`.db3` files. The 1.61 GB r05 bag is closed. The earlier 245-file/12.67-GiB
inventory preceded removal of three unusable captures and completion of the
throttle sweep. The 12.2-GiB environment cleanup and earlier failed/reset-
smoke capture removals remain as recorded above; these are point-in-time
figures.

Historical MPC, localization, map, and raceline runs were not deleted by this
model-data cleanup: they have separate controller/regression uses and are
outside this catalog’s plant-training inclusion rule. Their retention is not
a claim that every historical tuning run is needed; removing them requires a
separate best-run/regression audit.

## 2026-10-01 schema-7 audit view

The source schema-6 archive remains unchanged. A separate reset-aware analysis
view is available at:

```text
live_runs/derived_dynamics_learning_20260928/
  plant_teacher_mixed_dataset_full3d_reset_safe_20261001/
```

It contains the same 850,576 fixed-25-ms rows and 7,797 sequences, with
explicit sequence/run family, condition, replicate, reset-index, and run
quality metadata. Simulator pose, rigid-body state, and acceleration labels
remain separate from production-odometry and sensor-estimator channels.

Reset-topic inventory across the 116 run records found three bags: throttle
surface r04 (771 reset-command epochs; rejected by the whole-run quality
gate), r05 (739; accepted), and straight throttle-transition r01 (246; rejected
by the whole-run gate). For clean r05, all reset edges already fell between
archived sequences; the 741 archived r05 sequences have zero within-sequence
reset crossings and zero internal packet gaps. The initial suspicion that the
archive merged r05 conditions across resets was tested and disproved.

The future dataset exporter now treats a recorded reset-command rising edge
as a hard boundary even if packet IDs happen to be consecutive. Three focused
unit tests cover reset splitting, continuous joining without reset, and packet
gap splitting.

The four exporter-exception bags were re-audited after the fix. None produced
clean eligible aligned sequences, so none were added. See
[`FULL_MODELING_RESET_PROGRESS_20261001.md`](FULL_MODELING_RESET_PROGRESS_20261001.md)
for per-bag findings and the independently counted continuous-horizon run
coverage.
