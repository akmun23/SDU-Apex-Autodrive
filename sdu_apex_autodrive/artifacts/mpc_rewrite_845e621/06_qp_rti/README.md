# Phase 8 — 9-state RTI QP construction and nonlinear acceptance

`mpc_rti_build_ltv_qp` builds the absolute-state/input affine problem for the
seven-state exact plant plus two previous-rate states. It adds tracking costs,
hard state/input bounds, and the exact previous-input penalty expansion
`Q_pp += 2w`, `R_uu += 2w`, `N_pu = -2w`. There is no direct target-speed-state
tracking cost, and the initial lateral-velocity weight is zero.

`mpc_rti_build_nominal` provides cold feedforward seeding or one-stage-shifted
controls, then performs exactly two nonlinear progress/reference passes using
each stage's `delta_s`. `mpc_rti_rollout_candidate` recursively evaluates QP
controls with the same nonlinear model and rejects invalid model stages,
command/state-limit violations, or hard-corridor violations.

The synthetic steady-circle test constructs and solves an N=4 QP, verifies its
nonlinear candidate rollout, and checks N+1 state/reference indexing. This is
not actual-raceline replay or live controller acceptance. The production ROS
node still calls the old 10-state path; source-to-command-time prediction,
degraded-solve policy, N=30 timing, and ROS integration remain pending.
