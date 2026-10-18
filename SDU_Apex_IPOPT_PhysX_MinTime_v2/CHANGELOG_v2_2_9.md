# v2.2.9 - improved stability in the elastic IPOPT OCP

**Observed v2.2.8 failure:** IPOPT reported NaNs for constraint functions and
returned no feasible elastic iterate after 1200 iterations; the strict dynamics
residual remained the synthetic-seed value 0.46654. This is NOT solved yet.

- Added `detect_simple_bounds` to **Phase-I** CasADi/IPOPT configuration.
  In the previous run IPOPT listed 0 bounded variables despite thousands of
  simple steering, speed, throttle, load and elastic-slack inequalities.
  Moving recognized constraints to IPOPT variable bounds may improve
  scaling and line-search stability without changing the admissible set.
- Protected Frenet spatial progress divisions from 0 or negative denominators
  **outside the original physically feasible region**. On all feasible points
  (`den>=0.25`, `sdot>=0.30`) the exact original equations and derivatives are
  unchanged. The hard physical constraints are still applied to the unguarded
  quantities. This reduces NaN risk during infeasible IPOPT trial steps.
- Warn when `--brake-zones 0` disables braking completely. The original car's
  throttle law applies full CAWB braking only at **exactly zero throttle**;
  drive-phase throttle is constrained to `>=0.015`. A 0-zone topology cannot
  exploit braking and may be dynamically infeasible for a tight 28.4m track.
- Cache upgrade recognizes the exact v2.2.8 working solver without changing
  `_restored_git_history`. No `.venv`, PhysX executable, or run outputs
  included in the update ZIP.

**Not verified:** convergence of the full 143-node IPOPT problem or any native
PhysX/Unity complete lap. Do not interpret package tests as optimizer results.

## Recommended next test

```sh
python3 run_ipopt.py --repo-root .. --mesh-spacing 0.20 \
  --brake-zones 3 --phase1-hessian limited-memory \
  --feasibility-iterations 800 --iterations 1200 \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_2_9 --allow-history-fetch
```

If the optimizer still returns its initial elastic iterate, **do not keep
raising iteration limits**: the next necessary development is a physically
consistent motion seed from a real Unity/native PhysX rollout, not more
feasibility-parameter overlays.
