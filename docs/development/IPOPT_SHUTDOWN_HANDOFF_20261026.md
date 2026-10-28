# IPOPT shutdown handoff — 2026-10-26

## Current state at shutdown request

Two minimum-time IPOPT jobs were active. Both were using approximately one CPU
core and 3.5 GB RSS. No candidate raceline or optimizer checkpoint had been
written by either active run; their output directories contained only
`provenance.json` and `optimizer_reference_track_mesh.csv`.

### v336 — outer-wall seed continuation

- PID at inspection: `1381634` (ephemeral; do not reuse after reboot).
- Input seed: `SDU_Apex_IPOPT_PhysX_MinTime_v2/optimized_practice_v334_outer_wall_feasible_seed_gate_fix_20261025/ipopt_candidate_pass_00/optimizer_seed.npz`.
- Trust-region time stage 128 completed at the 1,800-iteration limit, not
  converged. Reported objective: `6.299811 s`; max constraint violation:
  `1.20998e-4`; dual infeasibility: `2.24774e-2`. Its in-memory restart point
  was updated, but that restart point was not persisted to disk.
- Stage 256 had started. The console log is
  `SDU_Apex_IPOPT_PhysX_MinTime_v2/optimized_practice_v336_outer_wall_small_trust_continue_20261025_console.log`.
- Invocation: `run_ipopt.py`, 0.06 m mesh, 16 m/s max speed, 500 feasibility
  iterations, 1,800 iterations per time stage, trust factors `64,128,256,512`,
  limited-memory Hessian, source seed backend, 5 cm wall margin.

### v337 — v324 fast-seed continuation

- PID at inspection: `1435495` (ephemeral).
- Input seed: `SDU_Apex_IPOPT_PhysX_MinTime_v2/optimized_practice_v324_tight_trust_from_v323_20261025/ipopt_candidate_pass_00/optimizer_seed.npz`.
- It was still consuming approximately one CPU core after about 58 minutes.
  Its live iteration output was not available in a durable log, and no result
  checkpoint/candidate had been written.
- Invocation: same mesh/speed/iteration/model setup as v336, with trust factors
  `16,64,256,1024`.

## What is and is not saved

The original v334 and v324 seed checkpoints, both provenance files, the v336
console log, and the source/configuration are on disk. These are sufficient to
relaunch both experiments from their original seeds. They do **not** contain
the latest in-memory IPOPT iterates. The current stage result/restart point is
only held in process memory until the continuation completes; stopping or
powering off now loses that incremental solver progress. Resuming from the
saved seeds will repeat computation.

## Resume guidance

After reboot, first check whether the output directories gained a completed
`ipopt_candidate_pass_*` directory or report before relaunching. If they did
not, rerun the same v336/v337 commands recorded in the shell/process history or
in this handoff, using the preserved seed paths above. Do not treat the v336
6.30 s intermediate iterate as a finished/validated raceline: it did not
converge and has not been checked by the native witness or MPC.
