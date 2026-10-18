# v2.2.5 — Physical residual merit function for IPOPT phase I

Based on user run: stage-1 tolerance=0.5 initial feasibility accepted at
violation=0, stage-2 tolerance=0.35 began infeasible at 0.1165433 and ended
at 0.1630731 after 150 IPOPT iterations; minimum-time and native PhysX
never ran. Reaching a relaxed bound is not enough: phase I must REDUCE
physical defects at every step.

- Replace captured-motion distance minimization with `sum(normalized_physics_defects²)
  + 0.001 * captured_motion_deviation` *during Phase I only*.
- Continue using the recovered original lap-time objective in Phase II.
- Dynamic equality bounds relaxed only temporarily; track and safety limits remain hard.
- Optimize even when a stage is already relaxed-feasible but contains nonzero defects.
- Preserve last relaxed-feasible state if the solver returns a worse or infeasible iterate.
- Print and save per-stage dynamics-defect energy, start/end violations, selected iterate.
- Accept both verified v2.2.4 and earlier known solver caches during in-place update.
- Legacy IPOPT source remains unchanged; no new PhysX or Unity results claimed.
