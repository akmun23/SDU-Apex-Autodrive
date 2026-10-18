# v2.2 — Phase-I Hessian continuation diagnostics

**Evidence for change**: user-provided practice-map IPOPT report, 237 nodes,
1000 feasibility iterations, max violation 1.10059 dominated by yaw-rate and
lateral-velocity collocation, 7671.9 final dual infeasibility. The initial
NLP was closer to feasible (max violation 0.39884). The numerical trust
bound was not the active constraint. No native simulation was reached.

- `--phase1-hessian limited-memory` (default): a numerical strategy for the
  hard feasibility NLP that avoids exact second derivatives across the stiff
  wheel/brake force model. This makes *no changes* to the physical equations,
  wall limits, wheel parameters, brake contract, or acceptance tolerance.
- `--phase1-hessian exact` reproduces the historical Phase-I Hessian strategy.
- Historical optimizer source remains exactly restored in `_restored_git_history/`.
  A separate `_working_solver_limited_memory/` contains the one numerical IPOPT
  option overlay. SHA256 values of both files are recorded in `provenance.json`.
- Recommendation for initial diagnostic: test drive-only topology with
  `--brake-zones 0 --mesh-spacing 0.20`, then add discrete brake phases after
  obtaining a dynamically consistent periodic lap; use independent runs for
  each topology. Full braking at exactly zero throttle is discontinuous.
- Three regression tests for source preservation, one-site option modification,
  and fail-closed behavior if the historical solver's source layout changes.

**No claim**: This does not establish feasibility, a completed native-PhysX
lap, sub-4.9 seconds, or parity with Unity. The solver may still fail, and
further source-aware modelling changes may be required.
