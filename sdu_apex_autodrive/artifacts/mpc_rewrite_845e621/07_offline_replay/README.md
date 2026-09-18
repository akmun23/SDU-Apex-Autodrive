# Phase 9 — 9-state RTI offline replay

Status: passed for numerical replay only. Static ROS-node integration now
exists, but this is not a closed-loop lap test, live shadow acceptance, or
live MPC acceptance.

The C++ replay executable consumed only the recorded runtime `/odom`,
`/current_map_pose`, `/cmd/speed`, and the raceline from the two existing
Pure Pursuit traces. Commands are replayed causally in event order. The legal
pose/odometry synchronizer aligns the map anchor to latest odometry; the
continuous projector builds the Frenet state. No simulator-truth topic,
simulator packet, throttle, contact, or hidden simulator state was read by this
replay. The optimizer's alternative commands were not applied to the recorded
vehicle states, so this does not measure closed-loop prediction accuracy.

With ADMM capped at 50 iterations, residual-gated solver rejection occurred
on 152/1,845 development and 150/785 holdout calls. Raising only the iteration
cap to 100 resolved all such failures. On the final profile all 2,630 calls
were accepted optimal, with zero degraded solves, solver rejections, path
rejections, source-order faults, or source-gap faults. No model equations,
weights, or trajectory values were changed to obtain this result.

The original replay above used a relative stopping term based on the largest
raw state/dual magnitude. That criterion could mix meters, radians, m/s, and
rate units, and is superseded by the absolute-residual audit below. Keep the
original measurements as historical evidence; use the audit measurements for
the current solver behavior.

## Development trace — `pp_amcl_recovery_24ms_20260917`

- 1,851 odometry packets; 1,845 synchronized/projected solves.
- Source dt p50/p95/max: 25.000 / 27.000 / 35.000 ms.
- State age p50/p95/max: 38.424 / 42.860 / 83.648 ms.
- Solve time p50/p95/p99/max: 0.525 / 0.984 / 1.053 / 2.148 ms.
- ADMM iterations p50/p95/p99: 10 / 30 / 32; degraded 0%; regularization 0.
- Minimum predicted corridor clearance: 4.120 m.
- Maximum |q_delta| / |q_v| / |delta|: 1.805 rad/s / 8.000 m/s² / 0.519 rad.
- Predicted target-speed range: 0.000–9.919 m/s.

## Untouched holdout — `pp_timing_repeat_20260917`

- 791 odometry packets; 785 synchronized/projected solves.
- Source dt p50/p95/max: 25.000 / 26.002 / 33.002 ms.
- State age p50/p95/max: 36.668 / 43.633 / 52.018 ms.
- Solve time p50/p95/p99/max: 0.530 / 0.891 / 0.969 / 1.949 ms.
- ADMM iterations p50/p95/p99: 10 / 31 / 34; degraded 0%; regularization 0.
- Minimum predicted corridor clearance: 4.112 m.
- Maximum |q_delta| / |q_v| / |delta|: 1.747 rad/s / 8.000 m/s² / 0.520 rad.
- Predicted target-speed range: 0.000–9.899 m/s.

## N=30 raceline core timing

Five hundred accepted solves on the 2,586-point, 51.718 m raceline at its
2.232 m/s start-point speed: p50/p95/p99/max 0.496 / 0.613 / 0.783 / 0.997
ms, with 25 iterations maximum. This microbenchmark is core-only and does not
include ROS callbacks, state synchronization, DDS, or shadow logging.

Reproduce inside the Humble workspace container with:

```bash
/workspace/build/f1tenth_mpc/mpc_rti_offline_replay \
  /workspace/src/sdu_apex_autodrive/artifacts/simulator_trace/pp_amcl_recovery_24ms_20260917/events.csv \
  100 0.01
/workspace/build/f1tenth_mpc/mpc_rti_offline_replay \
  /workspace/src/sdu_apex_autodrive/artifacts/simulator_trace/pp_timing_repeat_20260917/events.csv \
  100 0.01
/workspace/build/f1tenth_mpc/mpc_rti_benchmark
```

Next gate: run the integrated non-commanding shadow alongside Pure Pursuit in
batchmode, record `/mpc_shadow/diagnostics` together with legal state and
command topics, and compare predicted versus later observed N1/N5/N10/N20/N30
states. Keep command authority disabled until shadow and staged live gates are
separately passed.

## 2026-09-18 residual-stopping audit

The old stopping rule used `eps = tolerance + 0.02 * max(raw channel magnitude)`.
On the saved N30 replay, this allowed p95 raw residuals of 0.170 primal and
1.967 dual to be labelled optimal. Because the channels have different units,
the largest state or multiplier could loosen convergence for every other
channel. The solver now treats configured `solver_tolerance` as an absolute
maximum for both raw primal and dual residuals. A regression test constructs
a 0.1 terminal-bound violation next to an unrelated state of magnitude 10 and
verifies that tolerance 0.01 does not report convergence.

Rebuilt from the current source with ROS Jazzy/GCC 13.3 in a fresh temporary
build directory; all seven package tests passed. Strict replay results:

| Trace | Accepted / solves | Iterations p50/p95/p99 | Residual p95 primal/dual | Solve ms p50/p95/p99/max |
|---|---:|---:|---:|---:|
| `pp_amcl_recovery_24ms_20260917` | 1845 / 1845 | 23 / 56 / 57 | 0.009904 / 0.009588 | 0.878 / 1.665 / 1.772 / 2.448 |
| `pp_timing_repeat_20260917` | 785 / 785 | 25 / 57 / 57 | 0.009910 / 0.009656 | 0.902 / 1.672 / 1.704 / 2.744 |

No replay call was rejected or used regularization. Core solve p99 remains well
below the handoff's 10 ms target, but this is still exogenous-state offline
replay—not closed-loop prediction or live acceptance. The original run's lower
iteration counts and residuals must not be compared as if they used the same
stopping contract.

The lower 50-iteration cap is not adequate with the strict criterion:
development accepted 1,683, degraded 2, rejected 162; holdout accepted 624,
degraded 3, rejected 161. Keep the configured 100-iteration cap until a
separate solver improvement demonstrates equivalent residual quality at a
lower cap. A 500-cycle 50-cap raceline-core benchmark produced 499 optimal,
one degraded, zero rejected; it does not replace the broader trace results.
