# Full-source MPC: parallel sensitivity and QP scheduling audit — 2026-11-03

## Purpose and limits

The ongoing overhaul has one objective: a single high-fidelity, source-derived 60-control SQP/MPC that converges in a 40 Hz closed loop and fits the **complete 25 ms command publication deadline**, without an alternate controller or fallback policy.

This patch optimizes **how the identical source derivatives are evaluated**. It does not change the 26-coordinate source dynamics, physical footprint inequalities, 120 actuator-rate rows, QP formulation, frozen racing objective, actual nonlinear acceptance, 30-stage physical horizon, or independent strict source certificate. It does not claim that the high-slip source model matches Unity, or that the live ROS callback meets the deadline yet.

The new build flag `SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES` is **OFF by default**. The standalone native source mode and the full-source-forward differentiation remain explicit opt-ins.

## Implementation

- `f1tenth_mpc/src/mpc_source_sensitivity.cpp`: Each of the three causal batches of 20 command derivatives now has dedicated thread-local exact-source trace and perturbed nominal scratch. The three batches have independent source integration and write **disjoint columns** in the same final gradient, footprint Jacobian and rate Jacobian. Their timing counters accumulate after all threads complete. The sum of `full_forward_dual_plant_us` across parallel workers represents **summed worker time**, while `derivative_us` represents elapsed wall time.
- Any non-null world-sweep callback keeps differentiation serial until that callback is separately qualified as thread-safe; the default dev map-authority option is disabled.
- The scheduler uses the actual allowed CPU set and, for explicitly bound OpenMP workers, the available configured places. With fewer than three usable CPUs it **does not launch an OpenMP region**. This avoids extra threads and first-use runtime overhead on a one-core control executor.
- `f1tenth_mpc/src/mpc_source_reduced_sqp.cpp`: `EIGEN_DONT_PARALLELIZE` is local to the 60-variable dense QP translation unit. Running an additional Eigen matrix worker team over already occupied control cores previously generated severe QP stalls. Single-thread QP arithmetic retains the same strict optimality/KKT and independent nonlinear checks.
- `f1tenth_mpc/CMakeLists.txt`: Opt-in flag and dependency check. No nonportable ISA flags, no fast-math and no altered physics.
- `docker/development_entrypoint.sh` and `tools/start_dev.sh`: Pass the opt-in CMake setting into **mounted-source** dev builds; production default remains unchanged.
- `tools/mpc_native_source_closed_loop.py`: `--deadline-us 25000` forwards an actual absolute C++ solver deadline to each causal source epoch, not a Python-only timer.

## Quantitative evidence

Machine: Intel i5-13400F, Release -O3, 3 separate P-core CPUs 2, 4 and 6, OpenMP placement spread across those threads. All comparisons here use the **actually rebuilt** benchmark executable; the normal CMake build excludes that diagnostic tool, so it must be explicitly relinked.

| Test | Serial reference | Parallel optimized |
|---|---:|---:|
| 122 matched frozen source-SQP solves, median total | 14.596 ms | **10.479 ms** |
| Same 122, median exact-source derivative wall time | 9.859 ms | **4.650 ms** |
| Same 122, observed worst total | 24.486 ms | **16.771 ms** |
| Same 122, solver statuses/accepted iterates/objective/clearance | Reference | **122/122 identical** |
| Six difficult high-slip corner captures, gradient and entire physical Jacobian files | Reference | **Six/six byte-identical** |

The six difficult cases (before final serial-Eigen scheduling guard, same physical derivatives) each improved in total time, including a 27.442 -> 17.798 ms case. This is a performance comparison, **not** evidence that pre-existing high-slip finite-difference vs dual-source derivative disagreements are corrected.

Additional **source-only 25 ms absolute-deadline** closed-loop tests, each 60 consecutive 25 ms physical epochs, with actual controller memory shifts and strict source validation:

| Run | Exact-source valid | Median controller solve | P95 | Maximum |
|---|---:|---:|---:|---:|
| Repeated run 1, final scheduling | 60/60 | 11.242 ms | 17.232 ms | 18.376 ms |
| Repeated run 2, final scheduling | 60/60 | 11.102 ms | 17.417 ms | 19.439 ms |

No 25 ms solver deadline expired in either run; no secondary controls were introduced. The tests start from one recorded competition pre-corner condition, not from independent Unity trajectories. The **ROS callback, synchronization and publication overhead are not included** in the reported solver time.

C++ CTest: **22/22 passed**, both with the CMake option ON and with its default OFF. `bash -n` validation of changed launch scripts and `git diff --check` also passed.

Reproduction data:
- `/tmp/mpc_parallel_serial_fair_122_20261103.json`
- `/tmp/mpc_parallel_eigen_serial_122_20261103.json`
- `/tmp/mpc_parallel_hardsix_parity_20261103.json`
- `/tmp/mpc_parallel_eigen_strict_1_20261103/results.json`
- `/tmp/mpc_parallel_eigen_strict_2_20261103/results.json`

## How to reproduce / enable locally

```bash
cmake -S f1tenth_mpc -B build/f1tenth_mpc \
  -DSDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES=ON
cmake --build build/f1tenth_mpc -j6
cmake --build build/f1tenth_mpc \
  --target mpc_native_source_sqp_replay mpc_native_source_sqp_controller_epoch -j6
(cd build/f1tenth_mpc && ctest --output-on-failure)
```

To test the CPU-pinned standalone source MPC:

```bash
taskset -c 2,4,6 env OMP_NUM_THREADS=3 OMP_PROC_BIND=spread \
  OMP_PLACES=threads \
  python3 tools/mpc_native_source_closed_loop.py \
  live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692917233236533_case_2 \
  --cycles 60 --iterations 1 --deadline-us 25000 --forward-source \
  --out /tmp/mpc_parallel_reproduction
```

The development Docker opt-in is `SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES=true`, and `AUTODRIVE_REBUILD=0` makes the mounted-source entrypoint compile it. `SDU_APEX_MPC_RUNTIME_AFFINITY_CPUS=2,4,6` places the whole ROS process on the tested CPUs; this **also constrains other ROS threads**, so it is not a guaranteed production scheduling configuration. Do not simultaneously configure `control_thread_cpu_affinity=2` if multiple P-core source batches are expected: a single-core pin can make the optimizer run serially. Rebuild and verify the actual container before drawing live timing conclusions.

## Outstanding work / exit gates

1. **High-slip fidelity remains unqualified.** Previous six-corner FD-vs-dual tests reported significant cost-gradient discrepancies at source contact/slip branch boundaries (worst scaled mismatch in the prior audit 1.697). Parallelizing the same derivative preserves those values, it does not fix them. Extend model-branch derivative tests and compare exact nonlinear step agreement before replacing the FD reference.
2. **Full ROS/Unity timing not yet qualified.** Measure callback-to-command publication at 40 Hz, across startup and both tracks, with actuator-state synchronization and realistic CPU contention. The 25 ms deadline is stricter than simply a good median solve time.
3. **Actual live convergence remains untested by this patch.** Begin with a monitored short Unity live test; capture each SQP accepted step, full source/Unity model error, strict feasibility, and deadlines. Do not interpret an offline sequence of 60 exact-source steps as a live race.
4. **Single primary controller.** Do not introduce or rely on competing fallback controllers, arbitrary artificial trust regions, unpublished stale-plan output or full-brake-on-solver-error policies to disguise numerical problems. After live performance is established, consolidate and remove superseded implementations in a separately verified cleanup.
5. **Additional deadline margin.** Repeat under system load on the actual deployment CPU and Docker config. If online 25 ms remains marginal, profile world-sweep geometry, QP tail iterations and synchronous state/ROS publication without reducing physical fidelity.

This is an experimental optimized implementation, **not yet a qualified racing controller**.
