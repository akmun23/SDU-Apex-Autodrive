# Native source-SQP runtime optimization — 2026-11-03

Status: implemented locally and source-model tested; NOT qualified for the live 25 ms publication deadline. No Unity/races were driven. Do not change the established production MPC default until qualified.

## Changes

1. Full-source finite-difference prefix reuse: for a command perturbation at stage k, restart exact source propagation from the cached actual stage-k state (including the 22 hidden source channels, wheel/suspension dynamics and actuator history), rather than resimulating all previous stages. The accepted baseline is reconstructed before the next derivative batch, so a rejected SOC/line-search trial never pollutes sensitivities. The 180-row physical wall/actuator Jacobian matched the original finite-difference method exactly on the corner reference, and six frozen cases had identical exact objectives, wall violations and source-feasibility status. Measured two-pass speedup on these cases: 1.35–1.82x.

2. True 20-stage decision horizon while retaining 30-stage physical prediction: reduce the QP from 60 to 40 command variables. The 10-stage tail is kept from the current valid warm start; the nonlinear source plant still propagates and validates all 30 stages, all actuator/rate limits and all full-vehicle wall footprints. Finite-difference source perturbations decrease from 120 to 80 per SQP pass. The full 30-command alternative remains selectable. Shorter decision horizons restrict recovery in some severe frozen states; this is deliberately NOT described as proven safe in every condition.

3. Dense QP: reuse one KKT LDLT factorization for both predictor and corrector in each primal-dual IPM iteration, retaining independent primal and dual certificates. The IPM checks the actual absolute deadline every iteration and stops without falsely claiming success if exhausted.

4. Adaptive one/two SQP passes: optional source-exact early stop after the first accepted step, only when full 30-stage strict nonlinear source validation passes, the measured racing objective improved by >=2% and the minimum physical footprint margin is >=2 mm. Otherwise allow a second iteration. Do not infer feasibility from the affine QP alone.

5. Unified exact source scoring: expose the left/right physical footprint margins already computed inside the fixed-objective nonlinear scorer and reuse them as SQP constraint rows. Remove the second duplicated source geometry pass. A footprint-calculation failure now invalidates the candidate score, instead of being represented as zero-filled geometry. This passed the impossible-track regression and numerical cost/constraint parity checks.

6. The experimental 3-stage terminal continuation remains disabled by default. No wall geometry was relaxed and no unqualified fallback was promoted.

## Normal source-only closed-loop results

All configurations applied exactly 20 successive native controller commands, retained original full 30-stage source feasibility 20/20, and maintained source stage-1 physical state plus all hidden actuator/plant state channels within about 5e-12.

| Control configuration | Median over 20 epochs | Maximum | Final source racing objective |
| --- | ---: | ---: | ---: |
| 20 decision stages, 2 SQP passes, full 30-stage prediction | 78.93 ms | 127.24 ms | 147.05 |
| 20 decision stages, 1 SQP pass, full 30-stage prediction | 45.13 ms | 65.53 ms | 191.32 |
| 20 decision stages, adaptive 1–2 passes (1.5 average), full 30-stage prediction | 63.42 ms | 121.13 ms | 160.21 |

These are source-model simulation measurements, not Unity/lap times or guaranteed worst-case latency. The current median is still greater than the 25 ms publication period. The source-only 20-epoch closed loop is an encouraging necessary but insufficient test.

The same source-model SQP can still fail frozen high-slip/terminal cases. In the six-case 2-pass comparison, the reduced 40-variable QP sometimes decreased physical wall violation substantially without achieving strict source feasibility; it never published these incomplete proposals. These cases remain important to terminal continuation and sensitivity correctness, even if normal-case compute is prioritized.

## Reproduction

- Build: cmake --build build/f1tenth_mpc -j8
- Tests: (cd build/f1tenth_mpc && ctest -j8 --output-on-failure)
- Reference corner QP: build/f1tenth_mpc/mpc_native_source_sqp_replay live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692917333601071_case_6 - 2
- Optimized source QP: build/f1tenth_mpc/mpc_native_source_sqp_replay live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692917333601071_case_6 - 2 --suffix-fd --opt20 --adaptive
- Closed loop: python3 tools/mpc_native_source_closed_loop.py live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692917233236533_case_2 --cycles 20 --iterations 2 --opt20 --suffix-fd --adaptive --out /tmp/mpc_source_adaptive_check

## Remaining major optimization

Replace the 80 exact full-source control finite-difference perturbations per SQP pass with independently validated all-state source-model sensitivity propagation. The existing 23-state QP Jacobian is NOT equivalent to the full source plant: it misses 22 hidden source channels and progress/map nonlinearities. Preserving that high-slip accuracy is the condition for any faster derivative engine to replace the proven oracle. Then refine QP stateful warm starts, dense linear algebra scaling and deadline compliance. The ROS source mode and all performance features stay disabled by default until live timing and source/Unity correctness are demonstrated.

Existing uncommitted source files and prior experiments were not reset, committed or deleted.
