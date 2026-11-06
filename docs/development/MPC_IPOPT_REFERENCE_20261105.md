# Original-source IPOPT reference

The MPC remains unqualified. Further customized SQP tuning is suspended while
the same recorded nonlinear problems are tested with an independent mature NLP
method. No runtime solver replacement is justified until that comparison passes.
Earlier SQP edits remain uncommitted development work, not a qualified release.

`tools/reference_mpc_source_ipopt.py` uses IPOPT 3.14.11 / MUMPS 5.4.1 shipped
with CasADi 3.7.2. IPOPT owns its filter, restoration, limited-memory Hessian,
regularization and termination. The adapter supplies source values and analytic
first derivatives; it contains no penalty tuning or alternate control policy.

The original 30 steering/throttle pairs are the only 60 variables. Coordinates
are steering divided by 0.15 rad and throttle divided by 0.25. The native oracle
integrates the exact condensed source dynamics, including the recorded initial
state and actuator queues. It evaluates the original racing cost and retains any
reserve term belonging to that original cost. The unintended additional 3 cm
penalty is disabled in every oracle request.

All 240 footprint and 120 actuator-rate inequalities remain explicit. The 120
command-box inequalities appear once as IPOPT variable bounds, with zero bound
relaxation. This removes duplicate representations of the same bounds without
changing the feasible set. The native model receives its original float32
commands and states; the adapter does not substitute smoothed or double-precision
vehicle dynamics. IPOPT's initial interior push is an internal solver operation;
its supplied `x0` is always the recorded controls, never a witness or an improved
SQP trajectory.

CasADi multipliers use `L = J + lam_g*g + lam_x*z`. The audit maps these back to
the original `L = J - lambda*g`, including control-coordinate factors for bound
duals. A fresh oracle process independently evaluates the final float commands.
Qualification requires both `Solve_Succeeded` and the original independent
primal/stationarity/complementarity/dual checks. Any negative physical wall
clearance fails, including penetration below IPOPT's numerical tolerance.

The tool reconstructs cases directly from the immutable competition corpus;
it never calls the customized SQP. It copies the oracle and fixture executables
into the output directory before solving, protecting a running comparison from
concurrent builds. Each case stores input hashes, exact controls, accepted
iteration records, IPOPT residuals, status and final original-model KKT metrics.
Two repeats must agree numerically. An iteration limit or failed solve proves
neither physical unrecoverability nor that the reference method cannot recover
under any settings.

## Reproduction

Build the native source oracle and fixture separately from any active runtime
experiments, or use an already validated build directory. No production
dependency on CasADi/IPOPT is introduced.

```bash
python3 -m pip install -r tools/requirements-source-reference.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest \
  tools.test_reference_mpc_source_ipopt -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m tools.reference_mpc_source_ipopt \
  --tools build/f1tenth_mpc --output /tmp/apex_ipopt_reference \
  --iterations 2000 --repeat 2
```

The output directory must be new. `reference_report.json` records partial
progress after each completed case and sets `passed` only for all 22 qualified,
deterministic problems. `runtime_replacement_justified` uses the same gate.
Per-repeat `iterations.jsonl` is flushed while solving, and `result.json.gz`
preserves the complete result. The ten perturbations remain a later qualification
gate after primary recorded-start convergence is established.

## Preliminary comparison

The initial 500-iteration smoke comparison failed original KKT in all three
focused cases:

| Epoch | Final physical clearance | Normalized stationarity | IPOPT status |
| --- | ---: | ---: | --- |
| 765 | -0.0000001024 m | 15.99 | Maximum iterations |
| 769 | -0.000153676 m | 2276.46 | Maximum iterations |
| 775 | -0.0472065 m | 646623673 | Maximum iterations |

These are failures, including the submicron penetration in 765. They do not
establish a qualified replacement. Independent earlier curvature audits also
found source branch crossings and inconsistent finite-difference HVPs around
765/769. The full recorded-corpus comparison must distinguish numerical-method
limits from source-derivative, nonsmoothness and formulation issues; successful
adapter tests alone do not validate the physical derivatives.

## Full recorded-corpus result

The 2,000-iteration, two-repeat run completed: **22/22 deterministic,
16/22 physically feasible, 0/22 original KKT qualified**. The archived
[report](MPC_IPOPT_REFERENCE_QUALIFICATION_20261105.json) and
[dependency/source provenance](MPC_IPOPT_REFERENCE_PROVENANCE_20261105.json)
record the failed gate. Full per-iteration artifacts are in
`/tmp/apex_ipopt_corpus_parallel/`. These results do not justify replacing
the runtime solver, and do not prove any recorded state unrecoverable.
