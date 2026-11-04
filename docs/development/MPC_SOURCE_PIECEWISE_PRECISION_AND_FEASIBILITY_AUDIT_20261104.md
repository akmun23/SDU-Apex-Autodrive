# Piecewise-source SQP precision and feasibility audit — 2026-11-04

## Decision

The correct approach is a **single nonlinear SQP with the original 26-state Unity-derived source plant, branch-correct derivatives and exact-source feasibility acceptance**, not a separate controller for each tire/geometry regime. A local QP is not globally exact across finite control corrections or nondifferentiable transitions.

**Production status:** No branch-aware multicut QP has been enabled in any live launch profile. The original source physics, full physical steering range, positive drive throttle [0.001,1], exact source certificate, and 22ms live deadline remain unchanged. No artificial command trust region, zero-throttle fallback, or substitute vehicle model has been introduced.

## Actual source-model and constraint switching surfaces

| Original physical or mathematical threshold | Code source | Regularity and treatment |
| --- | --- | --- |
| Longitudinal tire slip magnitude 0.15 and 0.25 | unity_source_plant.cpp curve_base_response | C1 but generally not C2. Use source selected-branch curvature; crossing demands exact new evaluation |
| Lateral tire slip magnitude 0.01rad and 0.1rad | same tire spline | C1 but not generally C2; especially severe second-derivative jump at 0.01rad |
| Combined-slip squared normalized force demand crosses 1 | source tire-force utilization sqrt(max(1,z)) | Continuous, generally not C1. One-sided or generalized first derivative required |
| Ray travel equals zero or suspension jounce crosses negative max droop | wheel_contact_kinematics, grounded check | Contact active-set switch, possibly nonsmooth force jump. Source rollout and stage Jacobians must use actual active set |
| Jounce reaches compression/droop limits | suspension clamp | Active saturation and derivative switch |
| Raw load/rest load crosses 0 or 3 | tire_load | Source piecewise-linear load filtering, derivative switch |
| Magnitudes of wheel-ground and wheel-surface forward speed become equal | longitudinal slip denominator | max/abs gradient switch; exact slip denominator required |
| Steering actuator reaches target, changes sign/rate, or snaps | source steering controller | Real physical slew/snap must be part of dynamics, never a command trust band |
| Absolute forward speed crosses 5m/s | unity_source_wheel_substep_count | **Discrete source numerical integration switch**: 5 internal wheel substeps per 1ms below 5m/s, 1 above. Not a globally smooth physical map |
| Zero throttle, sticky-tire speed/time thresholds | drive torque/brake torque, sticky constraints | The source switches to brake at exactly zero and changes sticky constraints at low speed. Racing domain positive drive avoids this mode |
| Physical footprint heading selects another corner or changes cos sign | mpc_rti.cpp corridor_margin_for_state | max/abs active geometry changes. Need source-active vertex Jacobians |
| Frenet geometry approaches singular denominators or heading wrap | vehicle_model.cpp tracking integration | Nonlinear coordinate singularity/representation branch; cannot use one global quadratic |
| Raceline reference interpolation changes segments | reference-at-progress | Piecewise reference/width/progress derivatives; source objective schedule may freeze speed reference but must never freeze actual wall geometry |
| Speed weighting, clearance and slip reserve hinge | nonlinear source racing cost | Piecewise C1/non-C2 objective terms; GN is PSD but not exact Lagrangian curvature |

The trace now records original contact, jounce saturation, longitudinal-slip denominator selection and speed signs, steering target capture, brake mode, substep counts, existing tire/load/combined-slip and sticky decisions. It reuses original source force evaluations; no alternative plant is calculated. Tests verify trace output neutrality and the original 4-versus-20 wheel events per 1ms across 5m/s.

Actual native-QP high-slip probes at normalized epsilon 0.00005:

- Failure 15, stage 15: combined-slip demand 0.9999907539 crosses 1.0 at ~15.0ms; lateral tire slip 0.009995702rad crosses 0.01rad at ~16.4ms, hidden in original 1ms-only tracing.
- Failure 16, stage 29: two lateral-curve branch crossings around 17–18ms.
- No additional switches detected at this epsilon in these two specific stages. Other states may differ.

Paths: live_runs/mpc_source_second_order_20261104_01/case_15/full_branch_stage15_20261104.json and case_16/full_branch_stage29_20261104.json. Reproduce with mpc_source_stage_second_order_probe and MPC_SOURCE_INNER_TRACE_STAGE=15/29.

## Corrected exact-source two-branch QP comparison

From 270 frozen real Unity capture/state/iteration combinations, original 60-control physical domain, full 30-stage source rollout, two SQP passes, six pinned cores:

| Variant | Strictly source-feasible | Newly recovered | Previously feasible lost | Median offline time | P95 offline time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original hard Phase-I QP | 103/270 | — | — | 9.47ms | 21.38ms |
| **Two truly distinct source Jacobian cuts** (experimental OFF) | **118/270** | **17** | **2** | 12.94ms | 24.61ms |
| Earlier accidentally blended-second-Jacobian variant, REJECTED | 123/270 | 21 | 1 | 12.81ms | 25.27ms |

The earlier claim of 123 feasible results **does not belong to the mathematically correct distinct-branch implementation**; the actual distinct branch relinearization was retested and gives 118. A later full-regression rerun with independent post-QP affine certification, under concurrent build/test machine load, still gave 118 source-feasible results, with ~14.3ms median and ~26.9ms P95. Timings are CPU-load dependent, not hard realtime bounds. The correct implementation still fails the key case_live_fast_17 within two SQP iterations. It loses two previously feasible cases and exceeds the 22ms time budget at P95. It remains OFF. A first/second-branch independent physical affine constraint and vehicle command-bound check was added after candidate QP solving; no QP surrogate certificate is accepted as nonlinear source feasibility.

An optional exact physical-box presolve removes affine rows proved redundant across **all** legal steering and throttle values, not only a trust region around previous controls. Median 23 two-branch wall rows were removable in a 270-case corpus; feasibility stayed 118, but median solve speed improved negligibly. The option stays OFF live.

Rejected and removed from source: single elastic Phase I, one-column source secant, trapezoidal blended Jacobian, optimistic wall-margin compensation, two-source GN wall residual QP, and one/two-branch maximin clearance QPs. These either failed on critical cases or traded feasibility for worse compute and lost previously feasible outcomes.

## Validated computational reductions

- Source stage-local AD: positive-drive passive low-speed sticky histories have identically zero influence on the racing physical transition. Eight inactive sensitivity seeds were omitted, **all 26 physical state values continue**. Source AD interval count halves from 120 to 60 per two-pass solve; 536/536 comparisons had identical published-command candidates and source feasibility.
- Inexact physical hard Phase-I QP: stop when original affine constraints, complementarity and bounded dual residual pass independent checks, then retain full nonlinear source certification. No newly infeasible outcomes in repeated 268-distinct-case comparisons; median speed saving small, severe previous 40-iteration failures much better.
- Phase-I corridor-only differentiation: the infeasible feasibility QP does not use racing cost derivatives. It now optionally differentiates exact source footprint/progress alone using original physical geometry. 270/270 selected commands unchanged; all 60x60 wall and 120x60 rate Jacobian elements match full-source scorer on three critical captures. Median saved ~0.03ms, not a breakthrough.

## Actual mathematical route to a high-precision feasible MPC

1. Preserve the complete original source state and exact original internal integration. Enumerating all wheel/substep branch combinations as separate control modes would be combinatorial. Instead evaluate active branches during each exact rollout and use correct selected-branch first derivatives and source-derived one-sided/generalized derivatives at nonsmooth surfaces. At discrete 5m/s integration switches do not pretend a smooth derivative exists.
2. Derive the **discrete full-source Lagrangian second-order information**, including costate-weighted f_xx/f_xu/f_uu, physical footprint curvature, terminal continuation and source objective. Compare each Hessian-vector product against independent frozen-dual finite differences away from and on both sides of switching surfaces. GN remains a positive-semidefinite approximate Hessian, not a globally exact SQP quadratic.
3. Replace the Phase-I zero-affine-wall-violation obsession with a consistent **nonlinear physical feasibility restoration and filter**, not arbitrary clearance reserves or a fallback throttle. Exact objective/violation acceptance must update the **same** reference/progress source state after every accepted iteration; reject tiny ineffective steps as convergence. Do not publish a merely QP-feasible but physically wall-violating plan.
4. Benchmark structured **full-state multiple-shooting SQP** as an alternative to the 60-command condensed single-shooting QP. Retain 26 states at each stage and source dynamics equality constraints; use sparse/block-banded factorization/condensing to meet realtime. Do not assume a naive 800-variable dense QP can fit 22ms.
5. Check viability at each successive horizon/terminal: strict source feasibility at the current horizon does not guarantee recovery from all possible later states; do not claim an always-feasible solution when the vehicle is already physically unrecoverable. The captured case_live_fast_17 **does** have a strict source-feasible offline witness with about 0.138m minimum wall clearance, so its live miss is a genuine current solver problem.
6. Time the entire source rollout, Jacobians, Hessian, QP, nonlinear line search, strict source and native Unity transfer, and ROS publication. Gate on <=22ms worst case with a practical 15–17ms typical goal; qualify 5+ repeated full 10-lap practice races and full competition races before any default replacement.

**Result:** original source tire switching and several additional physical/geometry switching surfaces are now observable and testable. Accurate *local* derivatives and a faster source plant have improved, but the current QP is NOT yet globally accurate across nonlinear branch changes and cannot guarantee recoverable-corner feasibility at 22ms. Production remains unchanged. The next major solver implementation is a source-derived, feasibility-aware semismooth/multiple-shooting SQP rather than another hard-coded band or branch-specific fallback.
