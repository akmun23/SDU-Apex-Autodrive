# Packed training/validation archive access audit — 2026-10-12

## Finding

The benchmark archive
`live_runs/derived_dynamics_learning_20260928/practice_transfer_validation_20261001_r03/openplane_dynamics.npz`
contains 21 runs: 12 `train`, 5 `validation`, 2 `unseen_practice`, 1 `test`,
and 1 `final_test`. NPZ stores each numeric array as a compressed member for
the entire archive. A consumer that calls `_load_dataset()` therefore
decompresses the archive-wide `frames`, pose, sensor, and truth arrays before
it can filter by run ID.

Several 2026-10-12 experiments did exactly that. Their code filtered rows by
the intended train/development/practice run IDs before fitting or scoring, so
there is no indication that test/final-test rows entered fitting, feature
selection, or checkpoint choice. However, the test/final-test numeric arrays
were read into process memory. Thus the old report flags such as
`test_or_final_test_samples_used_for_fit_or_selection: false` are narrowly
true but do **not** prove the sealed arrays were untouched. These runs are
exploratory diagnostics, not protocol-clean held-out results. Do not cite
their practice metrics as independent sealed-set evidence.

The issue affects outputs derived from the mixed archive, including the
2026-10-12 small yaw-response model family (stable-AR, throttle-history,
IMU-acceleration ablation, fixed/adaptive recursive MLP, and absolute-next
state comparison), the low/mid-steering transition screens, and the paired
MPC/lateral-velocity benchmark analyses that load this archive directly.
Production odometry replays sourced directly from individual ROS bags and
full-spectrum yaw evaluations sourced from separately admitted run-level
captures are not implicated by this specific packed-archive issue.

## Preventive change and verification

`tools/vehicle_dynamics_learning/train_nssm.py::_load_dataset()` now checks
only the small `run_ids`/`run_splits` metadata first and refuses any archive
containing `test` or `final_test` before loading numeric members. The
yaw-specific row collector has the same preflight. A targeted test replaces
the numeric loader with a function that fails if invoked; the mixed archive
is rejected first, and the numeric loader is never called. The paired
throttle archive was also checked by metadata only; it contains train and
validation runs, with no sealed split.

Future model work must use a clean archive assembled from explicitly
admitted per-run sources, or a run-level loader that opens only approved
captures. Merely filtering `run_splits` after loading a mixed NPZ is no longer
accepted. The existing full-spectrum corpus has separately admitted
train/validation capture groups and remains the source for further yaw work.

## Required interpretation of old artifacts

- Keep the historical model weights and score JSON for reproducibility, but
  label their results exploratory; do not delete source data or artifacts as
  though they were corrupt.
- Do not fit, select, or promote models using the mixed benchmark NPZ again.
- The old score outputs are not evidence that sealed labels influenced the
  model. The demonstrated failure is that sealed numeric arrays were read,
  which invalidates the claim of untouched evaluation and warrants this
  conservative downgrade.
- Do not overwrite their original results. This audit and the correction in
  `MPC_YAW_RESPONSE_AND_ODOM_REPLAY_PROGRESS_20261012.md` supersede their
  archive-access interpretation.
