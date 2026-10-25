# Restore the best repeated MPC profile — 2026-10-26

## Target run

The fastest completed five-scored-lap mean currently recorded in `live_runs`
is `mpc_throttle_slew75_20261025_01`. It used the practice map and the
unchanged `ipopt_v1_baseline_raceline.csv` trajectory. Its first lap was the
warmup; the five scored lap times were:

`5.6998, 5.7258, 5.7928, 5.8118, 5.8648 s`

Mean: **5.7790 s**. The run completed six laps total, recorded no collision
or unpublished MPC rejection, and had a 17.055 ms maximum publication-gate
time (no cycle at or above 20 ms). The only run-local MPC overlay was
`weight_throttle_slew: 75.0`.

The next-best comparable five-lap profile in the available reports averaged
5.7846 s (`mpc_iterations150_repeat_20261025_01`). Its throttle-slew weight
was 100.0. Therefore the restored production value is 75.0, not 100.0.

## Restoration made

- Set production `weight_throttle_slew` to the best run's value, 75.0.
- Removed the later, not-live-validated residual-quality rejection on the
  full QP/line-search candidate path for iteration-limited solves. Exact
  nonlinear rollout/corridor checks remain in place. The separately existing
  residual-aware comparison used only when choosing recovery prefixes remains
  unchanged.
- Kept the recorded practice trajectory, 150-iteration setting, and existing
  controller configuration. No IPOPT process or output was changed.

## Provenance limitation

The run manifest records commit `c2099fe...` plus a dirty-worktree hash list.
The exact source snapshots for several MPC files from that run are not present
in Git history or the retained run directory, so a byte-for-byte source
checkout cannot be reconstructed from the available artifacts. This change
restores the measured profile setting and removes the clearly later residual
gate; it does not claim that every source line is identical to the historical
container.

## Verification

`f1tenth_mpc` rebuilt successfully in the project's ROS container. All nine
applicable package test targets passed (127 individual tests). The excluded
`test_mpc_qp_oracle_refinement_selection` target could not import SciPy in the
dev image; it is unrelated to the changed production path. The historical
five-lap result remains the performance evidence; this source restoration is
not itself a new simulator run or a new lap-time claim.
