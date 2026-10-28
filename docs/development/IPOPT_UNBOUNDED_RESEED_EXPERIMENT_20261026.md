# Prepared IPOPT wide-search → tight-search experiment

Status: **started 2026-10-27.** Two runs use the saved v337 source-model
candidate: factor 1e12 explores with essentially no effective time-stage trust
cap, and a separate factor-1024 continuation gives the fast but infeasible
v337 region 5,000 iterations to satisfy its dynamics. Both passed source-seed
recertification (max violation 0.000104968344 <= 0.001) and reached IPOPT.
The current logs show 8,533 variables, 7,585 equalities, and 28,917
inequalities per run. IPOPT has printed its solver banner; iteration rows have
not appeared yet, while both processes remain CPU-active.

Live logs:

- Wide search: SDU_Apex_IPOPT_PhysX_MinTime_v2/v338_wide_console.log
- Factor-1024 refinement: SDU_Apex_IPOPT_PhysX_MinTime_v2/v338_tf1024_console.log
- Wide raw-stage capture: optimized_practice_v338_wide_from_v337_20261026/time_stage_iterates/
- Factor-1024 raw-stage capture: optimized_practice_v338_tf1024_refine_from_v337_20261027/time_stage_iterates/

## Why the two phases are necessary

The v337 factor-1024 iterate had a 4.9583299769 s objective but a
0.0387703988 NLP constraint violation (limit 0.001), dominated by periodic
roll-rate collocation residuals, so it was not a valid raceline. The
present runner exports only its best feasible iterate. This setup also saves
the raw wide-stage iterate as an **exploratory** archive, even if the solver
misses feasibility. The raw archive must never be sent to MPC or treated as a
source-feasible checkpoint.

The existing componentwise bound has radius
`sqrt(max(Phase-I seed-distance, minimum_reference) * factor / component_count)`
in normalized units. Factor `1000000000000` is an effectively unbounded search
over the physically admitted state and control ranges; it is **not** a literal
removal of the trust constraint. Inspect its result to confirm that the trust
constraint was inactive. Every physical corridor, vehicle, and dynamic
constraint remains in the NLP.

## Stage A: wide exploration

The wide-search command below is running with the indicated raw-stage capture.

```bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  SDU_IPOPT_STAGE_CAPTURE_DIR="$PWD/optimized_practice_v338_wide_from_v337_20261026/time_stage_iterates" \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python -u \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/run_ipopt.py \
  --repo-root "$PWD" \
  --map f1tenth_planning/maps/autodrive_practice_20260924_b.pgm \
  --centerline f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv \
  --output optimized_practice_v338_wide_from_v337_20261026 \
  --optimizer-seed-file optimized_practice_v337_v324_fast_seed_continue_20261025/ipopt_candidate_pass_00/optimizer_seed.npz \
  --mesh-spacing 0.06 --max-speed 16 --wall-margin 0.05 \
  --phase1-formulation source-equalities --phase1-hessian limited-memory \
  --seed-backend source --skip-native \
  --feasibility-iterations 500 --iterations 5000 \
  --time-trust-region-factors 1000000000000 --print-level 4 \
  > SDU_Apex_IPOPT_PhysX_MinTime_v2/v338_wide_console.log 2>&1
```

Expected raw archive: `optimized_practice_v338_wide_from_v337_20261026/time_stage_iterates/factor_1e+12_raw_iterate.npz`.
The sidecar JSON holds the objective, violation, and solver status. A completed
wide solve may also export an ordinary candidate checkpoint, but only when
its source-model feasibility gate passes.

## Stage B: re-centre and refine

If the wide stage itself becomes the selected feasible candidate, use its
`ipopt_candidate_pass_00/optimizer_seed.npz` directly as the next
`--optimizer-seed-file`. Its Phase-I state will be the centre of the new trust
region.

If the wide stage is **infeasible**, build a modified initializer by blending
its path offset, heading error, and speed with the v337 certified checkpoint:

```bash
SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/make_exploratory_ipopt_seed.py \
  --raw-stage optimized_practice_v338_wide_from_v337_20261026/time_stage_iterates/factor_1e+12_raw_iterate.npz \
  --base-seed optimized_practice_v337_v324_fast_seed_continue_20261025/ipopt_candidate_pass_00/optimizer_seed.npz \
  --fraction 0.25 \
  --output optimized_practice_v338_wide_from_v337_20261026/exploratory_geometry_seed_alpha_0p25.npz
```

This file is explicitly marked as **not source feasible**. `run_ipopt.py`
reprojects the altered state and must pass strict Phase I before doing any
minimum-time optimization. If Phase I cannot restore feasibility, use a
smaller blend fraction; do not promote the invalid wide trajectory.

Once a suitable seed exists, use the same map, centerline, mesh, and vehicle
settings as Stage A. For the modified initializer produced above, the complete
Stage B command is:

```bash
cd /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python -u \
  SDU_Apex_IPOPT_PhysX_MinTime_v2/run_ipopt.py \
  --repo-root "$PWD" \
  --map f1tenth_planning/maps/autodrive_practice_20260924_b.pgm \
  --centerline f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv \
  --output optimized_practice_v339_recentered_tight_20261026 \
  --optimizer-seed-file optimized_practice_v338_wide_from_v337_20261026/exploratory_geometry_seed_alpha_0p25.npz \
  --reproject-optimizer-seed \
  --mesh-spacing 0.06 --max-speed 16 --wall-margin 0.05 \
  --phase1-formulation source-equalities --phase1-hessian limited-memory \
  --seed-backend source --skip-native \
  --feasibility-iterations 500 --iterations 1800 \
  --time-trust-region-factors 8,16,32,64 --print-level 4 \
  > SDU_Apex_IPOPT_PhysX_MinTime_v2/v339_tight_console.log 2>&1
```

If Stage A produced a **certified** candidate instead, replace the
`--optimizer-seed-file` path with its `ipopt_candidate_pass_00/optimizer_seed.npz`
and omit `--reproject-optimizer-seed`. Never substitute the raw iterate archive
for `--optimizer-seed-file`.

Record the Stage B Phase-I feasibility result, all time-stage residuals, and
whether its best **feasible** predicted lap improves on v337's 5.1465 s.
Only then consider independent source-native and Unity/MPC validation.

## Parallel factor-1024 feasibility refinement

This run directly tests whether more IPOPT iterations can reduce the v337
factor-1024 roll-rate residual while retaining its fast basin. It is seeded
from the certified v337 factor-16 checkpoint, uses the same source-equalities
physics and 474-node mesh, and allows 5,000 iterations at factor 1024. Its raw
stage is captured separately. A predicted time below 5.0 s only counts if
every non-trust constraint violation is at most 0.001.

Output directory:
optimized_practice_v338_tf1024_refine_from_v337_20261027

The 1024 run uses the same command inputs as Stage A, with:

- output: optimized_practice_v338_tf1024_refine_from_v337_20261027
- capture directory: optimized_practice_v338_tf1024_refine_from_v337_20261027/time_stage_iterates
- iteration budget: 5,000
- trust factor: 1024
- log: SDU_Apex_IPOPT_PhysX_MinTime_v2/v338_tf1024_console.log
