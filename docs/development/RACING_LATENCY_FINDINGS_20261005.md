# Racing cadence, localization, and MPC latency findings — 2026-10-05

These are offline analyses of saved simulator bags. No truth/debug-only input
is consumed by the controller, odometry, EKF, or AMCL. The two current
practice baseline bags and the one failed current final-track bag have full
run manifests; the older complete final bag is historical and has incomplete
configuration provenance.

## Cadence and control timing

| Run | LiDAR header rate | Header interval p95 / max | Bridge response p50 / p95 | MPC source age p95 | Pose/odom skew p95 | MPC solve p95 / max | Whole callback p95 | Prediction fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fresh practice R0-1 | 39.964 Hz | 25.77 / 44.89 ms | 28.64 / 29.88 ms | 2.46 ms | 25.92 ms | 1.89 / 4.63 ms | 2.12 ms | 0 |
| Fresh final R0-1, aborted | 40.008 Hz | 25.77 / 44.89 ms | 2.85 / 27.60 ms | 1.86 ms | 25.76 ms | 6.67 / 10.69 ms | 6.81 ms | 0 |
| Historical complete final | 39.960 Hz | 25.90 / 75.27 ms | 28.36 / 29.32 ms | 2.25 ms | 25.90 ms | 5.32 / 8.94 ms | 5.50 ms | 0 |

The two fresh runs deliver LiDAR at about 40 Hz, not 20 Hz. Each has one
approximately 45 ms maximum interval; the historical complete run has a
75 ms maximum interval. Their p95 intervals remain close to the nominal 25 ms.
The fresh final bag has one packet-sequence gap; the practice bag also has one.

The pose/odom skew is about one nominal sensor interval. The synchronizer
interpolates odometry at the older map-pose timestamp, propagates that map
anchor using the newer odometry sample, and reports a fused state at the
newest source timestamp. The MPC control-time predictor then predicts from
that fused state to control time. Both stages ran without command/time
fallbacks in these bags. State age p95 is below 2.5 ms, predictor compute p95
is below 61 microseconds, and the longest measured MPC callback is 10.9 ms,
well inside the 25 ms control interval. The evidence does not support changing
state extrapolation or increasing solver time budgets as a first response.

The final simulator's bridge response-latency distribution differs from the
practice image, but its sensor cadence remains 40 Hz, and fresh source age is
low. Arrival cadence and request/response latency are reported separately;
the latter is not itself proof of lost sensor packets.

## Frenet localization comparison

| Run | Estimate | Normal p95 | Tangential p95 | Yaw p95 |
|---|---|---:|---:|---:|
| Fresh practice R0-1, ten laps | AMCL map input | 4.4 cm | 16.7 cm | 0.0001 rad |
| Fresh practice R0-1, ten laps | Odom, one initial alignment | 1.58 m | 1.93 m | 0.0000 rad |
| Fresh final R0-1, first ~4.5 m only | AMCL map input | 11.4 cm | 2.61 m | 0.0002 rad |
| Fresh final R0-1, first ~4.5 m only | Odom, one initial alignment | 0.8 cm | 2.5 cm | 0.0000 rad |
| Historical complete final | AMCL map input | 3.4 cm | 10.7 cm | 0.0000 rad |
| Historical complete final | Odom, one initial alignment | 3.67 m | 3.20 m | 0.0000 rad |

The short failed final capture is not evidence that AMCL is universally poor:
its error becomes predominantly tangential after a small-distance startup
segment. The complete historical final and complete fresh practice recordings
both show AMCL constraining the multi-lap pose drift that raw odom accumulates.
Thus a blanket removal of AMCL's along-track correction is not justified.

In the failed final recording, 105 accepted AMCL scan updates could be joined
causally to the health diagnostics and truth path. Their applied corrections
sum to `-2.601 m` along the path, while the final controller-facing pose has
`-2.613 m` tangential error. On the same captured segment, odom remains within
2.5 cm tangential error. This is strong evidence that scan corrections create
the short-run bias in that straight, low-speed regime. It is not yet evidence
that the correction should be disabled in corners or over a complete lap,
where those corrections counter substantial odom drift.

## Current diagnosis and next handoff steps

The measured failure is not a 20 Hz bridge or MPC compute overrun. The current
final run stops after about 4.5 m, with 920 nonlinear corridor rejections. The
first saved failure is at stage 0 and exceeds the configured corridor tolerance
by approximately 0.7 mm. In parallel, the low-speed straight scan corrections
accumulate about 2.6 m of negative progress error. These are distinct
quantitative findings; both must be addressed before a full final baseline can
be claimed.

The handoff's next stages are the post-run sector report and empirical
lateral/longitudinal/combined envelopes. The AMCL bias is retained as a
specific hypothesis for the later localization work package; do not globally
remove along-track correction, because complete runs show it counteracts
multi-lap odom drift. Do not alter the controller's state extrapolation or the
vehicle model based on cadence alone. Revisit the stage-0 corridor failure
after its input pose and Frenet bounds have been joined to truth at the failure
timestamp.

Artifacts:

- `tools/racing/analyze_localization_frenet_error.py`
- `tools/racing/analyze_control_state_latency.py`
- Per-run CSV/JSON outputs under each run's ignored `analysis/` directory.
