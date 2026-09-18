# MPC live rejection analysis

- Diagnostic cycles: 13036
- Accepted: 11742
- Rejected: 1294
- Malformed diagnostic events: 0
- Rejection classification coverage: 100.0% (meets the 95% gate)

Missing-feature bins are reported as coverage gaps, not physical operating regimes.

## Rejection categories

- `solver_residual`: 1288 (99.5% of rejected)
- `missing_synchronized_state`: 6 (0.5% of rejected)

## Interpretation

Rates are associations within recorded diagnostic cycles, not causal claims. Source/command timing is reported as data and is never used to reject or discard a cycle. Simulator truth is not read.
