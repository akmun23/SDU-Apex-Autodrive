# Source-exact MPC: local model validity failure at the competition corner

Status: **reproduced, not repaired**. November 5, 2026.

## What was tested

Recorded original-source SQP primal trace:
`/tmp/apex_phase1_full_compact/epoch_775/replay.json.gz`, iteration 0.

The test independently reconstructs each candidate by replaying its recorded
30 steering/throttle commands through `mpc_source_nlp_oracle`, with the
*extra* preferred-clearance penalty disabled. It compares the exact original
hard inequalities `g(U + d)` to the condensed source-AD affine model
`g(U) + J(U)d`. No controller parameters, source forward equations,
state trajectories, control commands, constraints or numerical solver steps
are modified.

Reproduction on this machine:

```bash
python3 -m tools.audit_mpc_source_linearization_domain \
  --case /tmp/apex_phase1_full_compact/epoch_775 \
  --iteration 0 \
  --output /tmp/mpc_linearization_domain_epoch775.json
```

The `/tmp` fixture is a locally generated version of the immutable recorded
competition source fixture. The audit tool accepts any valid source fixture
directory containing both `source_context.mctx` and a `replay.json.gz`
with a complete original primal trace.

## Direct finding

The initial 60-control normalized SQP trial has maximum absolute control
coordinate step **2.00879** (physical scaling: 0.15 rad steering and 0.25
throttle). Its original hard constraints show:

| Fraction of recorded trial | Maximum source constraint linearization error | Minimum affine clearance | Minimum exact source clearance | Count of compared substep-schedule mismatched *events* |
| ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.00000760 m | -0.904982 m | -0.904974 m | 0 |
| 0.003 | 0.00025190 m | -0.903056 m | -0.903017 m | 0 |
| 0.01 | 0.00207414 m | -0.896316 m | -0.895802 m | 20 |
| 0.1 | 0.02700185 m | -0.809656 m | -0.794859 m | 40 |
| 0.4 | 0.15619569 m | -0.520790 m | -0.546999 m | 140 |
| 0.8 | 0.69224464 m | -0.138300 m | -0.462519 m | 340 |
| 1.0 | **1.13934028 m** | **+0.010360 m** | **-0.553683 m** | **820** |

The maximum error at full step is one future body-corner inequality (hard-row
239). At full step, the source transition comparison also reports 2,882
combined-slip saturation region changes, 6,781 longitudinal/lateral tire-curve
region events, and nine reference interpolation changes. These values count
**internal compared events, not independent physical discontinuities**.

The source forward-model implementation contains a deliberate hard switch:
`unity_source_wheel_substep_count` selects **five** wheel substeps for
`abs(forward_speed)<5.0m/s` and **one** at/above 5.0m/s, based on a recovered
PhysX configuration. This is a discrete integration/scheduling rule, not
a smooth tire-force model. Other tire, contact, slip and interpolation
branches also change as the proposed controls alter the 30-stage trajectory.

## Replication across earlier competition epochs

The same original-source oracle audit was run on the first recorded full
SQP trial for epochs 765 and 769, not only the terminal 775 failure:

| Epoch | Largest scaled control change | QP minimum affine footprint clearance | Exact source footprint clearance | Largest hard-constraint model error |
| ---: | ---: | ---: | ---: | ---: |
| 765 | 0.492 | approximately 0.0 cm | -70.12 cm | 82.57 cm |
| 769 | 1.228 | +0.56 cm | -42.17 cm | 66.06 cm |
| 775 | 2.009 | +1.04 cm | -55.37 cm | 113.93 cm |

These are **local QP predictions compared with exact reintegration of the
recorded *full candidate***, not published controller trajectories. They
demonstrate that the source linearization domain problem affects several
competition corner epochs. All still require physically feasible controls
from their recorded warm starts and actual original-NLP KKT convergence.

## Independent differential check

At recorded epoch 775, body-corner row 179, control coordinate 3:

- Condensed source-AD Jacobian: **1.56276**.
- Symmetric finite difference at normalized perturbation 0.002:
  about **0.881**, with approximately 20 discrete schedule mismatch events
  in each direction, as well as tire branch changes.
- Symmetric finite difference at normalized perturbation 0.0002:
  **1.56163**, with zero schedule/reference mismatches.
- At normalized perturbation 0.00002: **1.55486**, with float32 source
  precision and a tire-force spline region change contributing noise.

The objective-gradient spot checks at the same state agree within roughly
0.1--0.2% for the tested early controls. The mismatch in *large* control
changes is therefore **not proof of an incorrect local source adjoint**.

## Engineering interpretation and limits

The controller currently differentiates the source plant and hard track
constraints at **one nominal trajectory**, then solves a local affine-constraint
QP. It starts SQP with `step_radius=0` (no local model-validity bound).
Consequently the QP can propose a large, apparently collision-free step that
crosses many nonlinear/hybrid regimes before exact rollout rejects its
predicted feasibility. This is an **unjustified global use of local
linearization**, not evidence that the nonlinear source forward model is
inaccurate.

The event counts and exact source mismatches are correlated; the audit does
**not** establish that the 5 m/s substep switch alone causes the error.
Smooth second-order state/constraint curvature, tire saturation, dynamic
coupling and reference interpolation can all contribute. The current source
Lagrangian Hessian and source-branch audits therefore still matter.

## What must be fixed, rather than disguised

1. **Original nonlinear constraint curvature and hybrid branch semantics:**
   distinguish true source discontinuities (including 5-versus-1 substeps),
   continuous but nonsmooth contact/tire conditions, and smooth high-curvature
   dynamics. Verify derivatives/HVPs *within each valid source branch* and
   one-sided/generalized derivatives at real switching surfaces. Do not
   smooth the real PhysX source step or replace it with bicycle physics.
2. **SQP model validity:** the inner optimization must not treat a one-point
   first-order footprint model as globally accurate across arbitrary changes
   of steering and throttle. Use a mathematically defined nonlinear
   globalization/model-agreement mechanism or a source-consistent hybrid
   nonlinear formulation. Do not equate arbitrary old input trust restrictions
   with the mathematical locality needed for an accurate SQP model.
3. **Predictive versus exact feasibility:** compare each QP's predicted
   footprint margins to original-source recomputation and report the failed
   corner rows, Jacobian branch IDs and constraint residuals. Never publish
   an infeasible candidate regardless of QP status.
4. **Convergence from original recorded controls:** only count a repair when
   epochs 773/775 and then all 22 known recoverable states are feasible,
   with genuine original-NLP KKT convergence. Recovery from a separately
   prepared feasible witness is not adequate.
5. **Unity fidelity is separate:** locally correct source derivatives and
   correct SQP convergence still do not prove the source forward model matches
   Unity under high slip. Revalidate source-to-Unity state prediction later.

The related temporary audit artifacts are
`/tmp/source_chain_fd_audit_20261105.log`,
`/tmp/clearance_derivative_branch_20261105.log`,
`/tmp/phase1_fullstep_switch_audit.log`, and
`/tmp/mpc_linearization_domain_epoch775.json`.

**No production solver/plant implementation was changed by this audit.**
