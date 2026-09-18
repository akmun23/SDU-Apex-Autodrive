# MPC live rejection analysis

- Diagnostic cycles: 4696
- Accepted: 3039
- Rejected: 1657
- Malformed diagnostic events: 0
- Rejection classification coverage: 100.0% (meets the 95% gate)

Missing-feature bins are reported as coverage gaps, not physical operating regimes.

## Rejection categories

- `solver_residual`: 1400 (84.5% of rejected)
- `nonlinear_unknown`: 164 (9.9% of rejected)
- `invalid_input_or_state`: 85 (5.1% of rejected)
- `missing_synchronized_state`: 6 (0.4% of rejected)
- `input_gate:mpc_command-time_state_prediction_failed`: 2 (0.1% of rejected)

## Interpretation

Rates are associations within recorded diagnostic cycles, not causal claims. Source/command timing is reported as data and is never used to reject or discard a cycle. Simulator truth is not read.
