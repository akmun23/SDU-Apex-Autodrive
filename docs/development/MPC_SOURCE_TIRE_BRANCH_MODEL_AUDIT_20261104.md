# Exact-source tire-force branch audit — 2026-11-04

## Purpose and outcome

This audit diagnoses large local-derivative and finite-step QP prediction discrepancies using the ORIGINAL 26-state source plant, saved Unity physical states, and the actual first SQP 60-command QP directions. No bicycle surrogate, force smoothing, nonphysical input restriction or fallback command policy was introduced.

**Primary finding:** the inspected high-slip nonlinear mismatches are strongly associated with real tire-force regime changes. The largest previously unexplained second-derivative discrepancy at case 15, prediction stage 15 was resolved by tracing all internal 0.2 ms solver substeps, rather than only 1 ms outputs.

The physical source dynamics, tire forces and actual command choices are unchanged.

## Original source tire equations

From f1tenth_mpc/src/unity_source_plant.cpp:

- Forward friction spline: peak slip 0.15, peak response 0.9, asymptote slip 0.25, asymptote response 0.58, stiffness 0.8.
- Lateral friction spline: peak slip 0.01 rad, peak response 1.0, asymptote slip 0.1 rad, asymptote response 0.5, stiffness 1.0.
- Normalized force demands: longitudinal response / 0.9 and lateral response / 1.0.
- Combined-slip utilization: sqrt(max(1, longitudinal_demand² + lateral_demand²)).
- Tire forces use these piecewise spline responses, source filtered wheel load, and division by combined-slip utilization.

The two-piece cubic spline is continuous and C1 at each extremum/asymptote, but is NOT generally C2. At lateral slip 0.01 rad (positive side), its response second derivative jumps from -40,000 to approximately -370.37 per rad² (before the contact load and combined-slip terms): a more than 100-fold curvature change. At longitudinal slip 0.15, it changes from -160 to -192 per squared unit slip.

At combined-slip demand² = 1, the sqrt(max(1,z)) term has a **first-derivative discontinuity** when the derivative of z is nonzero. On one side the derivative is zero and on the other it is (dz/dq)/2. It is mathematically incorrect to treat a finite update crossing this active set as accurately predicted by one ordinary smooth Hessian.

Smoothing or removing these exact source branches just to help a QP would make the MPC physics less accurate relative to Unity. The correct next direction is a piecewise/semismooth source SQP and exact nonlinear relinearization, while retaining physical steering/throttle bounds.

## Measured first switching events in original frozen QP directions

All times below are within a single 25 ms source-prediction interval, not lap times.

| Recorded Unity failure | Source prediction stage | Earliest important switching event at normalized epsilon 0.00005 | Source evidence |
|---|---:|---|---|
| Practice failure 15 | 4 | Longitudinal tire-curve switch ~4.4 ms | Front-right longitudinal slip about -0.15028 around the source peak 0.15 |
| Practice failure 15 | 15 | Combined-slip saturation at 15.0 ms | Nominal utilization exactly around 1.0 |
| Same failure, stage 15 | 15 | Lateral curve switch ~16.4 ms | Front-left lateral slip 0.0099957 rad, immediately below the 0.01 rad extremum |
| Practice failure 16 | 29 | Lateral curve switch 17.0 ms | Rear-left lateral slip about 0.009987 rad near the 0.01 rad extremum |

Case 15's wheel solver is below 5 m/s and executes five 0.2 ms internal substeps per millisecond: **500 wheel force evaluations per 25 ms**. Case 16's fast stage 29 executes one substep per millisecond: 100 wheel force evaluations.

The 25 ms source interval was numerically identical to 25 separately run 1 ms intervals across the tested probes (all 26 state channels). The earlier 1 ms output-only trace incorrectly indicated no regime switch at case 15 stage 15 for epsilon 0.00005. The new authoritative internal 0.2 ms trace resolved that mystery. It records the source quantities already used for each original force evaluation, not a reimplemented force curve.

No 5 m/s solver substep-count change, grounded-state change or tire-load-filter transition was detected in these critical comparisons. This does not prove other branch effects never occur elsewhere; Unity/source prediction transfer remains a separate qualification problem.

## Local first- and second-order prediction

The new SourceJet2 direction propagates a nominal-branch 26-state physical second directional derivative through the SAME native source plant. It uses a double-precision first/second directional scalar and all original source branch decisions. Existing source 20-lane first-derivative AD agrees with SourceJet2 to roughly 1e-6 relative scaled difference on the recorded QP directions.

SourceJet2 is **only the Hessian of the selected smooth branch**. Independent three-source-step finite differences cross nearby friction/saturation boundaries and may not converge to the same scalar curvature. At epsilon 0.00005 the maximum scaled componentwise SourceJet2 vs central-FD second-derivative discrepancy is approximately:

- Case 15, stage 4: 2.94, with source longitudinal/combined-slip switches.
- Case 15, stage 15: 2.21, with genuine hidden 0.2 ms lateral/combined-slip switches.
- Case 16, stage 29: 0.017, with two lateral curve region changes.

At epsilon 0.005, 12/30 case-16 stages have one or more tracked source force switches; the other 18 stages have maximum scaled first-derivative error about 0.0001 for that ONE tested QP direction. This does not imply universal model accuracy.

## CPU performance and validation

Six-core parallel second derivative evaluation, 30 physical source stages and the direction already available:

| Saved source state | Serial median | Six-core wall-time median |
|---|---:|---:|
| Failure 15 | 7.38 ms | 1.49 ms |
| Failure 16 | 2.19 ms | 0.47 ms |

Three runs per case showed exact bitwise serial/parallel agreement. This is a stage curvature kernel benchmark, **NOT** full SQP time. It omits source tangent generation, state/progress tracking second derivatives, Lagrangian/Hessian assembly, QP factorization, nonlinear validation and ROS callback/publication.

Regression tests include: source second directional versus source AD and independent FD, physical steering/positive throttle bound rejection, symmetry, exact 25x1 ms source state identity, and complete output neutrality of a 500-event inner wheel trace. The frozen source-SQP before/after numerical comparison shows **24/24 identical selected control sequences, feasibility, source violation, objective, and QP iterations**. No meaningful median timing regression was observed. All 22 CTest suites passed.

Important remaining timing failure: two-pass offline high-slip solve medians are about 47 ms for failure 15 and 26 ms for failure 16, already above the 22 ms complete live callback limit before Unity/ROS overhead.

## Reproduction on ExternalDrive

Source fixture root: live_runs/mpc_source_second_order_20261104_01/

Build the offline tools with:

    cmake --build build/f1tenth_mpc --target mpc_source_stage_second_order_probe mpc_native_source_sqp_replay -j6

For case 15 and its hidden inner-substep switch:

    P=live_runs/mpc_source_second_order_20261104_01/case_15
    MPC_SOURCE_TRACE_MICROBRANCHES=1 MPC_SOURCE_INNER_TRACE_STAGE=15 OMP_NUM_THREADS=6 OMP_PROC_BIND=spread OMP_PLACES=threads taskset -c 0,2,4,6,8,10 build/f1tenth_mpc/mpc_source_stage_second_order_probe "$P" "$P/first_qp_qp_direction.txt" 0.00005

For case 16 stage 29:

    P=live_runs/mpc_source_second_order_20261104_01/case_16
    MPC_SOURCE_INNER_TRACE_STAGE=29 OMP_NUM_THREADS=6 OMP_PROC_BIND=spread OMP_PLACES=threads taskset -c 0,2,4,6,8,10 build/f1tenth_mpc/mpc_source_stage_second_order_probe "$P" "$P/first_qp_qp_direction.txt" 0.00005

Saved JSON includes inner_physical_branch, microbranch_comparison, branch_local_analytic_state, parallel_analytic_wall_us and exact serial/parallel parity fields. The full actual-QP-direction nonlinear cost/corridor sweep is independently available in tools/audit_source_sqp_real_direction_second_order.py.

## Next mathematical implementation and qualification gates

1. **Piecewise source Lagrangian Hessian:** build state-control, terminal and active footprint costate-weighted curvature from source stage derivatives. Use selected-branch curvature only in regimes where a meaningful smooth local Hessian exists. At original combined-slip/contact transitions, use correct piecewise/generalized derivative treatments rather than fitting an inaccurate global C2 quadratic.
2. **Unified nonlinear feasibility convergence:** compare 26-physical-state multiple-shooting SQP to the existing 60-command condensed SQP, with the SAME physical wall/actuator constraints and exact source acceptance. Relinearize around each accepted exact source trajectory, with explicit nonlinear feasibility progress and no tiny ineffective step counted as solver success.
3. **Computational reuse:** share stage A/B, forward tangents, costate/second-order products and exact scoring among both SQP passes. Complete callback must remain under 22 ms worst case, preferably 15–17 ms, including actual ROS publication.
4. **Source/Unity model transfer and terminal viability:** independently test contact and actuator synchronization; ensure current states are physically recoverable and previously published trajectories have viable successors.
5. **Release gate:** restore each recoverable captured high-speed corner without fallback driving, retain all actual vehicle steering/throttle authority and positive drive throttle, then qualify repeated complete fast practice and competition races.

**Current status:** nonlinearity is located and measurable at the original PhysX internal substep; the full source branch curvature is implemented/tested offline. The QP, high-speed nonlinear feasibility and real-time 10-lap qualification are NOT yet fixed or passed.
