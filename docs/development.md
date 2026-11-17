# Development and diagnostics

Unity trials test driving behavior; frozen vehicle-model problems test solver accuracy and compute cost.

## Build options

[f1tenth_mpc/CMakeLists.txt](../f1tenth_mpc/CMakeLists.txt) selects C++20,
`-O3 -march=x86-64-v3 -mtune=native -ffp-contract=off` on GCC/Clang x86-64.
The default requires x86-64-v3 CPU support. `SDU_APEX_MPC_NATIVE_TUNING=ON`
selects `-march=native` for the build host. `SDU_APEX_MPC_LTO` controls link-time
optimization and defaults to the native-tuning setting.
`SDU_APEX_MPC_HORIZON` sets compile-time arrays: default 30 stages, range 10–80,
each 25 ms. The trial runner exposes these as environment settings.

The reported build comparison uses four pinned E-cores, 30 frozen cases and
117 iterations. Each SQP iteration takes 11.54 ms with SSE2, 9.58 ms with
x86-64-v3 and 9.34 ms with native tuning, about a 17% gain over SSE2.
In 2,700 cycles arranged as 300-cycle replays with a 22 ms deadline, SSE2
defers 44 cycles and the native-class build defers three.

The reported comparisons find no gain from LTO, PGO, unrolling or `-O2`.
Enabling floating-point contraction changes QP fields at `6e-12` and changes
borderline replay statuses, so the build disables it. The reported portable
and native plant streams are byte-identical. 

[f1tenth_localization/CMakeLists.txt](../f1tenth_localization/CMakeLists.txt)
uses C++17 and builds CUDA AMCL by default. `BUILD_GPU_AMCL=OFF` retains
odometry, replay and EKF for CPU-only checks. In a ROS environment with
dependencies installed, build the CPU subset with:

```bash
colcon build --merge-install \
  --packages-up-to f1tenth_localization f1tenth_mpc \
  --cmake-args -DBUILD_GPU_AMCL=OFF
```

For a controller build tuned to the host, use `-DSDU_APEX_MPC_NATIVE_TUNING=ON`
and optionally `-DSDU_APEX_MPC_LTO=ON` in the CMake arguments. CUDA AMCL is
required for the full [runtime](architecture.md); the CPU subset cannot run it.

## Closed-loop development

[Trials](trials.md) describes the runner, lap evidence and CPU isolation.
From the repository root, set `run_dir` to the printed trial output directory
and `sweep_spec` to a prepared sweep JSON, then use:

```bash
tools/sim/run_trial.sh practice RUN_ID
python3 tools/analysis/summarize_trial.py "$run_dir" --keep-export "$run_dir/analysis"
python3 tools/analysis/evaluate_mpc_bag.py "$run_dir/run"
python3 tools/analysis/sweep_mpc.py run "$sweep_spec"
```

Sweep candidates can supply an `env`, including trajectory overrides.
`interleave: true` runs one trial per candidate per round. Pool identical control
settings and fix a stopping rule before running trials. Record rear slip, exit
position, heading wobble and laps before failure at each hairpin alongside
failure counts. At the reported 1–2% failed-pass rate, counts alone poorly
separate settings. The reported frozen-model replay with recorded localization
noise does not fail at baseline, so use Unity trials to assess tuning and
controller failure rates.

## Frozen problems and diagnostics

The [utility README](../f1tenth_mpc/util/README.md) documents capture/export,
`mpc_export_case`, `mpc_sqp_bench`, `mpc_qp_check` and their full command syntax.
These executables are excluded from the default controller build. Set `mpc_build`
to the configured MPC CMake build directory before building targets:

```bash
cmake --build "$mpc_build" \
  --target mpc_export_case mpc_sqp_bench mpc_qp_check plant_derivative_bench -j4
```

Run a trial with overlay `capture_source_benchmark: true`; capture only changes
diagnostics. Decode with `summarize_trial.py RUN --keep-export RUN/analysis`, then use
[export_cases.py](../f1tenth_mpc/util/export_cases.py) to select 30 cases per track,
including standing starts, infeasible starts, hairpins and the observed progress
bins. The manifest records source hashes and record indices. Each case freezes
the first iteration's problem and the last attempted radius; it cannot reconstruct
every live-cycle trial. The `MPCCASE6` format with `source-pod/v4` captures
checks the file magic and size. Regenerate cases after a POD layout change. Keep captures outside the source package.

Set `case_file` to an exported case. The benchmark argument order/defaults are:

```bash
"$mpc_build/mpc_sqp_bench" solve "$case_file" 4 0.5
"$mpc_build/mpc_sqp_bench" drive "$case_file" 300 4 22 0.5 0 0 0
```

`solve`: case, iterations, radius. `drive`: case, cycles, iterations, deadline in ms,
radius, lateral-noise SD, heading-noise SD, seed. JSON reports include
model/exact reductions, curvature corrections, trial attempts and accuracy flags.
`drive` applies the first issued command for 25 ms; deferrals hold the preceding
command. It ends at first wall contact with zero reserve and nonlinear tolerance,
logs the configured reserve separately and publishes only horizons that pass
the configured checks, including any far-stage wall tolerance.
It counts rejected horizons separately from contact; the live supervisor aborts
on rejection. Fresh source state arrives each cycle, so it does not model ROS
gaps or epoch-gated continuation. Pin/isolate workers for timing comparisons.

| Variable | Benchmark control |
|---|---|
| `MPC_OVERRIDE="name=value,..."` | Captured weights/margins |
| `MPC_THREADS` | Jacobian workers; use 1 with fixed iterations and a large deadline for determinism |
| `SQP_PENALTY`, `SQP_ACCURACY` | Violation penalty and accuracy tolerance |
| `NOISE_FILE` | Recorded `e_y_error e_psi_error` lines, blank lines between recordings |

[bench_cases.py](../f1tenth_mpc/util/bench_cases.py) repeats cases, times phases
and compares result fingerprints or quality. `plant_derivative_bench` measures
the derivative kernel; [MPC](mpc.md#derivative-cost-and-convergence) lists its
substep results. The reported 5 ms derivative-tick comparison runs 4.8× faster but has
20–25% derivative error from stiff wheel/tire dynamics. Fused multiply-add
does not shorten the kernel in that comparison.
The runtime retains 1 ms ticks.

The production C API covers interval rollout, sensitivities, five-input Hessians,
instantaneous lateral slip/Jacobian and suspension-jounce reconciliation.
The reported portable and native/LTO builds match byte-for-byte over 100
ten-stage rollouts, 28-lane sensitivities, Hessians and reconciliation
(SHA-256 prefix/suffix `8213fc71...efb6`). This is numerical parity, not a Unity driving certificate.

For state-gap forensics, first export a trial summary, then run:

```bash
python3 tools/analysis/analyze_mpc_state_gaps.py "$run_dir" --output "$run_dir/gaps.json"
```

The tool aligns recorder receipt, source stamp, controller snapshot and first
collision/rejection. It separates post-abort waits and reports callback/sync timing.
Recorder receipt and controller callback entry are different events; callback
analysis needs recorded per-message callback-entry timestamps.

## Tests and CI

After building and sourcing the ROS workspace, run C++ suites and the actuator test:

```bash
colcon test --packages-select f1tenth_localization f1tenth_mpc
colcon test-result --verbose
python3 -m pytest sdu_apex_autodrive/test
python3 tools/verify_runtime_topic_policy.py
```

The C++ suites cover packet/epoch assembly, odometry and command histories,
observer/plant behavior, MPC state synchronization, trajectory parsing,
control timing, QP/SQP and publication bounds. The Python actuator test requires
`rclpy` and built `sdu_apex_msgs`. For host tools and optimizer suites, use a
Python environment with the optimizer dependencies and pytest installed:

```bash
python3 -m pytest tools/tests
PYTHONPATH=raceline_optimizer python3 -m pytest raceline_optimizer/tests
python3 f1tenth_mpc/util/check_qp_reference.py --solver "$mpc_build/mpc_qp_check"
```

The last command checks convex QP properties against OSQP (`pip install osqp`).
Optimizer orchestration tests and frozen replays do not certify Unity performance.

[CI](../.github/workflows/ci.yml) runs on pushes and pull requests. Its CPU
validation job uses a ROS 2 Humble container on Ubuntu 22.04. It builds
localization and MPC with CUDA AMCL off, runs both packages' CTest suites,
checks `git diff --check` and compiles integration-package Python for syntax.
A second job runs the optimizer suite with Python 3.12, checks Python syntax
in the tools and optimizer, and runs the documentation-link test. GPU AMCL
and Unity driving require separate checks.
