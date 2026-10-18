# v2.2.3 — IPOPT Phase-I dynamic-defect continuation (in-place)

## Failure diagnosed
The v2.2.2 practice OCP (143 nodes, zero brake phases) ran 1200 L-BFGS Phase-I
iterations. It ended at infeasibility 33.8048, including 10.0707 m/s² in
lateral-force equilibrium and a numerical captured-motion seed-distance cap
violation of 33.8048. IPOPT never reached the minimum-time solve.

## What changed

- Recovered v1.3 IPOPT code remains immutable in `_restored_git_history/`.
- The existing `_working_solver_limited_memory/` cache is safely upgraded in
  place only if its previous contents match the known v2.2.x working copy.
  Unknown local modifications cause an error rather than being overwritten.
- On the synthetic-seed Phase-I solve only, deactivate the historical
  measured-capture seed-distance inequality; it was calibrated for a measured
  full-lap run that this package does not have. The Phase-II min-time trust
  regions are untouched.
- Replace the single Phase-I solve with sequential dynamic equality tolerance
  stages: `0.5, 0.2, 0.07, 0.015, 0.0` (in each defect's ORIGINAL normalized
  units; longitudinal/lateral acceleration force balance stays in m/s²).
  Iteration budgets are partitioned 10% / 14% / 18% / 22% / 36%.
- The only temporarily relaxed constraints are the spatial collocation
  equations, wheel angular-velocity update, longitudinal/lateral force
  balances, and algebraic spring/damper normal loads. Lane/vehicle footprint,
  steering, speed, actuator, tire-load and all other inequalities stay hard.
- Stage-I **MUST end with zero permitted equality defect**, and the full
  nonlinear problem's existing 1e-3 constraint acceptance is applied against
  the current strict bounds. No native PhysX or controller export is allowed
  from an intermediate relaxed stage.
- Output records each homotopy stage's tolerance, iterations, status,
  residual and whether it supplied a usable finite warm start.

## Limitations
This is a numerical feasibility strategy, not additional measured physics.
Even a converged OCP remains a collocation approximation of native PhysX;
only collision-free native replay and Unity validation can establish lap
performance. Sub-4.9s performance has NOT been demonstrated.

## Setup / run

The ZIP extracts over the SAME existing folder,
`SDU_Apex_IPOPT_PhysX_MinTime_v2/`, preserving `.venv/`, the native PhysX
binary, old result directories and historical recovered sources.

```
cd ~/Documents/GitHub/SDU-Apex-Autodrive
unzip -o ~/Downloads/SDU_Apex_IPOPT_PhysX_MinTime_v2_2_3_IN_PLACE_UPDATE.zip -d .
cd SDU_Apex_IPOPT_PhysX_MinTime_v2
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/
python3 run_ipopt.py --repo-root .. --setup-check
python3 run_ipopt.py --repo-root .. --mesh-spacing 0.20 \
  --brake-zones 0 --phase1-hessian limited-memory \
  --feasibility-iterations 1200 --iterations 1200 \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_2_3 --allow-history-fetch
```
