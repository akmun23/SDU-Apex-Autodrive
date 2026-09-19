# MPC replay on the earlier PP simulator retry

This replay uses the real simulator event stream from
`../11_shadow/pp_mpc_shadow_candidate_progress_retry_20260918/events.csv` and
the same legal runtime inputs as production: `/odom`, `/current_map_pose`,
`/cmd/speed`, and the accepted raceline.

It is retained as a negative/diagnostic holdout, not as a claim of MPC
readiness:

- 4,690 synchronized/projected simulator states were available.
- 3,674 reached the solver; 3,180 were accepted and 494 rejected.
- 1,016 samples failed the startup path gate before solving, so this capture
  is not a clean closed-track MPC holdout.
- Solver-side rejects were 73 input and 50 residual; nonlinear rejects were
  371. The residual-recovery path was triggered 56 times and cannot repair the
  larger path/corridor failures.

The clean 16-lap simulator capture is therefore the relevant replay evidence
for command feasibility. This retry remains useful for the next investigation:
separate map-pose/path projection failures from controller infeasibility
before any authority run.
