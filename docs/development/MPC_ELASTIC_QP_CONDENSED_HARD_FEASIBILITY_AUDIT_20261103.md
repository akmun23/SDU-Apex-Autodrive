# MPC QP correctness: elastic walls, independent oracle and condensed hard feasibility — 2026-11-03

**Release status: not qualified; do NOT enable on race profiles.**

This is a record of changes and exact evidence from a synchronized offline
replay of the complete source-plant/actuator-FIFO MPC. The objective is correct
mathematical QP solutions and accurate *finite-step nonlinear* source
prediction before more Unity races. No simulator race was run as part of this
audit. No code was committed; the checkout has other, earlier, uncommitted
changes which were preserved.

## Code implemented and validated

1. Retain the physical source corridor and original 2D/3D wall planes.
   Introduce convex, explicit nonnegative QP slack via
   `(w/2) sum(max(0, a*x-b)^2)`. The native Riccati ADMM eliminates slack
   analytically in the per-stage 2D/3D proximity operator; the independent
   OSQP oracle instead creates one slack decision and `a*x-b<=slack` per face.
   **Neither approach certifies a physically legal vehicle state.**
   Production source rollout retains the strict original zero-slack gate.
   Main files:
   `include/mpc_qp_elastic_polygon.hpp`,
   `src/riccati_solver.cpp`, `src/riccati_scaling.cpp`,
   `src/mpc_rti.cpp`, `src/mpc_source_adjoint.cpp`.

2. Deterministically reproduce the complete frozen QP via V7 `.fqp`
   capture (elastic penalty plus progress-coupled normals), supported by the
   C++ reader/writer and independent Python parser/OSQP oracle.
   The new R1 300-iteration/penalty-preconditioned records exposed a real
   previously inconsistent schema limit (C++ reader accepted <=200,
   Python <=300). Both now bound values to 1..10000 and the real V7
   300-iteration records round-trip successfully.
   Main files: `include/mpc_qp_fixture_format.hpp`,
   `tools/mpc_qp_fixture.py`, `tools/mpc_qp_oracle.py`.

3. Add optional `source_elastic_wall_weight` (default 500 only when
   `source_violation_filter_sqp` is explicitly enabled). Apply a
   well-conditioned `rho_x=rho_u=127` and `tolerance<=0.002` only to
   the active elastic QP, at most 300 initial ADMM iterations plus the
   controller's existing strict deadline. Rescale ADMM dual multipliers when
   changing rho: preserve physical lambda=rho*y; do not silently reuse the
   old multiplier scale. Existing hard-QP and racing configs are unchanged.
   Targeted warm-state transition tests pass.

4. Add an opt-in RAW QP certificate on the original physical-coordinate
   Riccati x/u: check all 30 affine state-transition equalities using the
   raw unprojected primal controls, physical-unit x-vs-z ADMM split agreement,
   all original *hard* stage/control/terminal bounds and hard wall polygons.
   Report elastic wall violation separately. A projected z inside the
   polygon, an ADMM 'optimal' status or a small numerical residual alone is
   NOT sufficient for certification.
   Targeted tests cover false projected feasibility and artificially
   inconsistent raw successor dynamics; tests pass.

5. Add repeatable offline tools:
   - `tools/verify_mpc_elastic_qp_full_horizon.py`
   - `tools/audit_mpc_elastic_wall_feasibility.py`
   - `tools/audit_mpc_condensed_qp.py`
   - `tools/audit_mpc_condensed_source_rollout.py`
   - `tools/mpc_live_failure_replay.cpp` external selected frozen QP
     source-fidelity comparison with source-reachable physical rack
     command-increment convention.
   Rebuild and run `build/f1tenth_mpc/test_mpc_qp_fixture_roundtrip` for
   V7, `test_mpc_sqp_convergence` for raw certificate, and the full CTest
   package before promoting any config.

## Independent numerical parity

A test-generated 30-stage, 23-state, 2-control QP with mutually
contradictory original walls and separately penalized elastic slacks:

- Native Riccati ADMM, tolerance 1e-5: 154 iterations, raw primal
  5.14e-7 and dual 9.65e-6.
- Independently constructed explicit-slack OSQP 1.0.4:
  `solved`, primal 1.49e-13, dual 3.19e-14.
- Over the ENTIRE native horizon:
  **maximum state difference 2.604226e-5 physical units**;
  **maximum control difference 2.935221e-5 command units**.
  These are cross-channel maxima, not all metres.
- Per-stage elastic 2D/3D proximal checked on 160 randomized
  OSQP problems: 160 match. Expanded 1200-case sweep reported 1198
  matches and 2 OSQP maximum-iteration cases, NOT 2 confirmed native errors.

Reproduction:

```bash
build/f1tenth_mpc/test_mpc_qp_fixture_roundtrip \
  --gtest_filter=MpcQpFixture.ElasticFullHorizonRiccatiMatchesExplicitSlackOracleFixture
python3 tools/verify_mpc_elastic_qp_full_horizon.py \
  --osqp-vendor /tmp/sdu-mpc-osqp-v104
```

Independent OSQP evaluated the **actual captured** native elastic QP at
`epoch_1793641756087085412`. With `w=500` the optimum original-wall
violation reached **0.1590m**; with `w=8000` it was still **0.1005m**.
The optimizer was mathematically solving the *elastic* problem; those
solutions were NOT physically safe. Simply increasing slack weights is
not a proof of hard feasibility and often hurts ADMM conditioning.

## Decisive 59-case frozen-corpus feasibility classification

Corpus: `live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures`.
New opt-in QP captures from mode26, frozen local source-model epoch:

- **59** replayable snapshots have captured QPs.
- **8** were already hard-polygon (no elastic wall penalty).
- **51** were locally elastic. An independent linear program, retaining
  ALL captured affine state dynamics, input boxes, other state/terminal
  constraints and original wall geometry, minimized one uniform physical
  wall allowance over the ENTIRE horizon.
- **45 / 51** elastic snapshots have **zero required uniform hard-wall
  allowance** — the original frozen affine QP is mathematically feasible.
- **6 / 51** are *affine locally infeasible*, with minimum required wall
  allowance (in metres) **0.00493, 0.00513, 0.02461, 0.03359, 0.07295,
  0.08774**. At least some are stage-1 constraints; a new command may
  not be able to undo the already queued physical successor. The
  particular cause for each needs exact source/Unity evaluation.
- A previous LP reporting routine incorrectly inferred "active stages"
  directly from under-determined slack allocations. Fixed: active stages
  are now computed from actual `a*x-b` of the LP solution.

Audit:
`/tmp/mpc_elastic_hard_qp_feasibility_corpus_20261103.jsonl`.
Use `python3 tools/audit_mpc_elastic_wall_feasibility.py FILE.fqp`
for any saved elastic QP.

**Do not conclude that 45/51 will finish the race.** This certificate
concerns frozen **linearized** dynamics, not nonlinear source rollout or
recursive feasibility. It PROVES that at least 45 hard-QP failures should
not simply be attributed to unavoidable local mathematical infeasibility.

## Condensed hard QP: promising structural optimization

Eliminate affine dynamics exactly via
`x[k]=c[k]+G[k]*u[0:N-1]`:
`x[0]=x0`,
`c[k+1]=A[k]*c[k]+d[k]`,
`G[k+1]=A[k]*G[k]; G[k+1][:,2k:2k+2]+=B[k]`.

For a frozen source-corner QP:

- Original hard QP has **773** variables and 23x30 dynamics equalities.
  OSQP took **7350 iterations / 373.2ms solve** in the reference machine.
- After condensing, it has **60 steering/throttle decision variables**,
  466 nonredundant inequalities, and no state-dynamics equality.
  OSQP took **700 iterations / 24.6ms solve** (plus 2.9ms setup).
- Original and condensed solutions agree on the hard-QP decision to
  **9.6e-6** max control difference; identical cost to about
  **2.6e-7**. An early experimental implementation mistakenly
  used OSQP's upper-triangle P in `P@T`; this was corrected to a fully
  symmetrized Hessian, `Pfull=P+P.T-diag(P)`, before parity was claimed.
- A 23-case stratified sample: **17 hard-feasible condensed QPs solved**,
  **6 LP-proven hard-infeasible QPs correctly reported primal infeasible**.
  For the 17 solved: median solve 24.08ms, worst 76.72ms (Python OSQP,
  not optimized C++ implementation). This is a proof of improved
  conditioning and exactness, not a deployable sub-20ms solver yet.

Scripts:
```bash
python3 tools/audit_mpc_condensed_qp.py \
  /tmp/mpc_frozen_elastic_qps_20261103/epoch_1793641756087085412_mode_14_r1.fqp \
  --osqp-vendor /tmp/sdu-mpc-osqp-v104
```

A native hard Riccati ADMM solve with rho127 still had about **10mm
original-wall violation after 1200 iterations** on the same QP. Native
"optimal" with a loose 0.02 numerical tolerance is not acceptable as
2cm of raw wall ambiguity. An optimized condensed solver or active-set
strategy is now a principal solver-development target.

## Hardest remaining blocker: finite-step NONLINEAR source mismatch

Feed the same independently solved zero-slack condensed control sequence
back into the ORIGINAL exact nonlinear C++ source plant (complete hidden
wheel/suspension state and actuator FIFO).

For `epoch_1793641756087085412_case_6`:
- independent condensed hard QP: **original linear wall violation
  ~1.9e-15m**, solved;
- nonlinear source rollout: **fails the exact original wall at stage 8**;
- full-horizon maximum lateral-prediction error: **~0.3946m**;
  heading error **~0.1584rad**, path-progress deviation error
  **~2.509m**, max wheel-rate error across channels ~16.135rad/s,
  maximum footprint prediction error **~0.4231m**.

The source error was measured from the **same captured experimental V7
QP** via `MPC_EXTERNAL_REPLAY_QP_FIXTURE_PATH`, with rack-relative
steering and issued-throttle relative command transform; not the old
different QP's predictions. One of four tested hard-QP candidate
sequences passed original full-horizon source validation; three failed
at stages 16, 17, 22. Thus hard linearized feasibility does NOT imply
source feasibility. Some starting nonlinear nominals were already invalid.

Reproduction:

```bash
python3 tools/audit_mpc_condensed_source_rollout.py \
  /tmp/mpc_elastic_default_hardfeasibility_corpus_20261103/epoch_1793641756087085412_mode_26_r1.fqp \
  live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793641756087085412_case_6
export MPC_EXTERNAL_REPLAY_QP_FIXTURE_PATH=/tmp/mpc_elastic_default_hardfeasibility_corpus_20261103/epoch_1793641756087085412_mode_26_r1.fqp
build/f1tenth_mpc/mpc_live_failure_replay \
  live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793641756087085412_case_6 \
  --external-controls=/tmp/mpc_condensed_hard_source_controls.txt
```

The source simulator is not yet verified to match Unity sufficiently
near sub-centimetre wall margins. No claim of Unity feasibility is made.

## Release gates still FAILED

1. **Phase 0**: source-to-OSQP QP formulation parity for V7 is strong,
   but **26 historical R1 QP captures still fail exact bitwise replay**
   on current build. Do not change the pass criterion or silently exclude
   them from release qualification. They may involve changed program
   source or historic numeric compilation; independently classify.
2. **Phase 1**: QP first-order derivatives near nominal plausible;
   finite-step 30-stage source model prediction is provably wrong by
   many centimetres in corners. Need source-derived nonlinear dynamics
   defect correction / true multiple nonlinear SQP stages.
3. **Phase 2**: opt-in exact violation-decreasing R1→R2 filter often
   reduces exact corridor violation (~85% median on selected tested
   steps), but **still fails to obtain exact fully feasible horizons
   on many snapshots**. Quadratic elastic slack is not exact penalty.
4. **Phase 3**: directional second-order cost Hessian is experimentally
   useful for SMALL steps but has poor large-step outliers; off by
   default. A true nonlinear source dynamics/footprint correction, not
   more blind curvature damping, is needed.
5. **Phase 4**: source/Unity matching and recursive controlled-successor
   feasibility remain unqualified. Stage-0 future frontier failures
   require earlier anticipatory source-based control.
6. **Phase 5**: no live race or 5×10 qualification on these changes. Hard
   20ms budget cannot be met with the current Python reference condensed
   hard QP on difficult cases.

## Next implementation work (in order)

- Develop a **native condensed hard-wall QP** or efficient small active-set
  solver using the exact linear dynamics and original cost. Get reliable
  hard-QP feasibility and primal/KKT certificates inside the deadline,
  without elastic misclassification.
- Make R1/R2 *nonlinear* constraints meaningful at the **actual QP
  direction**: frozen source trial, per-stage dynamics and footprint
  defect/residual, source-exact relinearization, acceptance by physical
  feasibility reduction followed by exact-cost descent. Recover *before*
  entering irrecoverable stage-1 states. Distinguish true infeasible-start
  SQP from accepted output.
- Use multi-iteration offline same-epoch source SQP to determine minimum
  required nonlinear relinearizations and identify slip/contact regime
  transitions, then find an efficient deadline-qualified formulation.
- Extend per-channel source-versus-Unity synchronization to the first
  infeasible corner epoch and its queued command history. Do not use the
  approximate Python geometry checker as a physical feasibility oracle.
- Only then repeat strict Unity practice/competition race and 5x10
  qualification. Keep all experimental switches off in current production.

No user-visible command output or new steering target is authorized by
this audit. The primary achievement is a sharply isolated **numerical
hard-QP solver conditioning issue** and a separate, larger
**nonlinear finite-step source prediction issue**.
