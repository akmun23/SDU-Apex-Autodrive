# Source-SQP nonlinear prediction audit — 2026-11-03

## Status

The source-exact two-pass condensed SQP is **not qualified for ten robust, fast Unity practice laps**. The dominant issue is the accuracy of the finite-step QP prediction compared with the exact nonlinear physical source trajectory. Its local first derivatives are substantially more accurate than its extrapolated QP full-step predictions.

The 30-stage, 60-command optimizer uses the complete 26-coordinate source plant, physical steering rack dynamics, wheel/contact state, current command queue, full-horizon footprint constraints, and strict independent source certification. Steering and throttle retain their physical ranges. No arbitrary input trust region, fixed lower-throttle seed, fallback driving policy or zero-throttle braking behavior was introduced.

## Measured evidence

- At saved Unity failure 16 (practice progress approximately 22.12 m, 7.461 m/s), the QP predicted essentially zero corridor violation, whereas the exact source full-step rollout produced approximately 0.05961 square meters of summed squared margin violation. The largest individual clearance-model discrepancy was about 0.335 m on the first pass.
- At the same state, independent control finite differences (normalized epsilon 0.002) versus the full-source local wall Jacobian gave a P95 absolute entry error of 0.000171 and maximum 0.0204 across 3,600 wall-J entries. The local J is useful, but not globally predictive across nonlinear steering and slip transitions.
- Across 30 sequential causal source-model states, 58 source-feasible full-step racing-cost QPs had 25/58 WRONG-SIGN cost predictions. Median absolute true-source cost model error was 4.565 weighted units, P95 1586.936 units. All 58 observed full-step cost model errors were positive, i.e. the local Gauss–Newton quadratic systematically underestimated the cost change in this particular corpus.
- A severe prior Unity failure (15) was independently proved source-feasible by an offline 18,800-candidate physical command sweep: 28 valid corridor plans, one passing the production source validator with 14.097 mm minimum clearance. The normal high-speed MPC warm start still could not reach that narrow feasible region within 22 ms.
- Previous live runs also showed nine consecutive missed control publications preceding an SQP feasibility rejection. An offline solver fitting within 22 ms does not establish realtime ROS/Unity reliability.

## Changes implemented and retained

The full-step line search ALREADY evaluates the true source plant. This turn instrumented the same evaluation, with no additional physics evaluations, to compare:

- First and second SQP pass: QP predicted squared wall violation versus exact nonlinear source violation, largest individual clearance defect, accepted line-search alpha and physical-feasibility slack.
- First and second racing-cost SQP pass: QP-predicted quadratic racing-cost change versus the actual exact-source cost change, along with model error and an explicit validity flag.
- Crucial correctness condition: the Phase-I identity-Hessian minimum-correction QP is NOT a racing-cost QP. Those first/second pass cost fields are invalid/null in that regime; never mislabel its curvature as a source racing-cost prediction.

The per-epoch ROS diagnostics are under native_source_sqp_full_step_model_agreement. The run-level analyzer tools/analyze_mpc_qp_agreement.py now aggregates racing-cost and footprint model errors per pass, avoids double-counting the existing accepted-alpha cost metric, and records false-safe QP predictions and nonpublication streaks. Its Python tests were extended.

Validation: 22/22 C++ CTest suites and 3/3 Python analyzer tests pass. A source-exact pre-experiment replay versus the final diagnostic-enabled replay produced 42/42 EXACT matching command sequences, source feasibility, final objective, physical violations and QP iteration counts (saved Unity failures and sequential causal source states, one/two iterations). Median wall clock was 8.652 ms before versus 8.691 ms after, indicating negligible diagnostic overhead in that frozen test.

## Rejected experiments (removed from source)

Each candidate retained exact source scoring and source output certification, but none robustly improved real convergence:

1. Source-measured quadratic line-search ordering: negligible changes, no additional feasible corner recoveries.
2. Single unified source-Gauss–Newton cost/soft-wall-penalty QP: some physical violations and high-slip timing worsened; not superior to the hard-corridor baseline.
3. Alternative hard-corridor Phase-I GN and wall-J curvature: one mode repaired corner 16 in 8.84 ms from its true high-throttle warm start but lost corner 14 and exceeded 22 ms in challenging cases. No uniform win.
4. Powell-damped source-gradient BFGS update: worse source cost prediction and longer median time across 67 snapshots.
5. Nonmonotone filter trial centers: no increase in the 20-state feasible count and worse margin at the hard fast corner.
6. Exact source full-step rank-one curvature correction: unchanged 33/36 feasible count and 25 to 26 wrong-sign full-step cost predictions.

A temporary incorrect BFGS/GN branch introduced during experiment cleanup was detected by snapshot replay parity, corrected, retested and did not remain in the working tree.

## Necessary next mathematical implementation

1. Add discrete-source costate-weighted second derivatives to the SQP Lagrangian, not arbitrary diagonal/rank-one Hessian patches. In a true SQP, the curvature comprises both objective residual Gauss–Newton terms AND the second derivatives of source dynamics and active footprint constraints. Evaluate independent finite-step true-cost and physical-clearance prediction accuracy, not merely local Jacobian parity.
2. Compare sparse multiple-shooting source-SQP (states include all 26 physical source variables) against an exact curvature-aware condensing implementation. Multiple shooting makes nonlinear dynamic defects explicit instead of relying entirely on a frozen 60-control condensed map across steering/slip branch transitions. Maintain the SAME source physics and full physical control authority.
3. Introduce principled nonlinear globalization and feasibility restoration inside the sole MPC. A curvature regularizer or accurate nonlinear merit/filter is mathematically legitimate; an arbitrary hard steering/throttle trust-region bound or alternate fallback driving policy is not desired.
4. Include future viability, physical rack-slew authority, combined slip and a source-derived terminal continuation. The existing 30*25ms horizon does not by itself guarantee recursive feasibility at the next control epoch.
5. Independently close Unity/source one-step high-slip residuals, source state and actuator-command timestamp/queue alignment. Exact source optimization does not automatically equal accurate Unity prediction.
6. Certify the full 40 Hz callback including state synchronization, source derivatives, QP factorization, nonlinear scoring and publication: normally below 20 ms, hard cutoff 22 ms, no consecutive deferrals. Only then run repeated full 10-lap practice qualification (target sub-49 seconds) and competition on separate settings.

## Reproduce diagnostics

Export the run's diagnostics JSONL with tools/export_mpc_run_report.py, then run:

PYTHONPATH=tools python3 tools/analyze_mpc_qp_agreement.py /path/to/diagnostics.jsonl --out /path/to/model_agreement.json

Saved evidence on the connected machine includes:

- /tmp/mpc_source_cost_full_step_accuracy_30epoch_20261103.json
- /tmp/mpc_source_cost_live_telemetry_exact_parity_20261103.json
- /tmp/mpc_source_fullstep_exact_curvature_ab_20261103.json
- /tmp/mpc_source_phase1_curvature_shape_grid_20261103.json
- /tmp/mpc_live_failure_replay_20261103/16/source_context.mctx

These /tmp diagnostics should be archived to a durable run directory before cleaning the machine.

**Outcome: precision failure is quantified, production-cost/source-truth comparison is live-instrumented and output-neutral, but a correct source second-order SQP still requires a substantial mathematical implementation. No successful ten-lap claim.**
