# Offline plant signal semantics — 2026-10-03

This document is the WP16 signal contract for offline plant work. It describes the existing data paths and does not authorize changes to simulator physics, runtime odometry, or MPC. The machine-readable definitions and unit-tested rigid-body/rate conversions live in [`signal_semantics.py`](../../tools/vehicle_dynamics_learning/signal_semantics.py).

## Timing and frame conventions

- Model datasets use a fixed simulator sample step of 25 ms (40 Hz). Packet/source timestamps are used to associate messages and reject invalid intervals; network/bridge stamp jitter is not treated as a change to the physical integration step.
- The fixed-cadence adjacent encoder rate is `(angle[k] - angle[k-1]) * 0.059 / 0.025`. The source interval is accepted only in `[15, 35] ms`, and the query uses the newest encoder sample at or before the query time. It never uses a future sample.
- The stamp-divided rate `(delta angle * radius / source_dt)` is retained as a diagnostic alternate definition only. It must not be confused with the fixed-40-Hz rate or silently substituted for it.
- The earlier `openplane_dynamics_raw_wheels.npz` sidecar is the source-stamp-divided branch. The later `*_fixed40hz.npz` sidecar is the fixed-25-ms branch. The latter retains the original `frames[:,5:7]` filtered feature unchanged. They are different labels and must remain separately named in comparisons.
- A 50 ms or 100 ms fixed-cadence estimate uses the cumulative angle difference across exactly 2 or 4 valid encoder periods and divides by `n * 25 ms`. A separately reconstructed stored 100 ms signal uses the stored view's documented four-packet causal window and actual selected endpoint elapsed time (accepted range 75–125 ms); it is an audit of the old feature, not a replacement definition for fixed cadence.
- Positive body yaw follows ROS planar convention. With rear-axle body speed `u` and track `0.236 m`, rigid-body rear contact-speed proxies are `left = u - yaw_rate * track/2` and `right = u + yaw_rate * track/2`. These are kinematic contact-speed estimates, not tire-force or tire-slip truth.
- Simulator rigid-state velocity/acceleration and simulator pose are carried as offline labels in `/autodrive/roboracer_1/bridge_packet_timing` JSON, keyed to bridge receive source stamp and packet sequence. They are not deployment sensor inputs and are never valid future rollout features.
- For a COM longitudinal offset of `0.15532 m`, `u_rear = u_com` and `v_rear = v_com - yaw_rate * 0.15532`. Pose conversion additionally rotates that body offset by world yaw. Yaw errors/differences are wrapped.
- Sequence computations must preserve dataset run, reset epoch, packet identity, and encoder validity. No finite difference or rolling window may cross a run/reset boundary or an invalid packet gap.

## Signal inventory

The exact per-signal entries—including units, topic/array source, frame/reference point, observation/derivation, causal window, nominal-delay qualification, validity, training use, rollout use, and semantic role—are defined in `SIGNAL_DICTIONARY` in the Python module. Important distinctions are summarized here:

| Signal | Meaning and use |
| --- | --- |
| `frames[:,0:3]` | Rear-axle body `u`, `v`, yaw rate from bridge odometry; `v` includes the existing COM-to-rear point shift. Supervised state/transition target, not a sensor-only rollout truth input. |
| `odom_pose_xyyaw` | Rear-axle/base-link world pose from bridge odometry; offline trajectory target/diagnostic. `simulator_pose_xyyaw` is a separate truth label. |
| `simulator_rigid_state` | Position/quaternion plus COM body linear velocity and angular velocity from packet debug JSON; offline truth labels only. |
| `frames[:,7:9]` | Steering and throttle commands; known future command inputs are allowed in command-driven rollout. |
| `frames[:,3:5]` | Steering and throttle actuator feedback; feedback history is measured, but future measured feedback is allowed only in the explicitly labeled oracle intervention. Normal free-run predicts actuator state. |
| `left_encoder.position[0]`, `right_encoder.position[0]` | Cumulative rear wheel angles. The timestamps associate samples; the physical data contract is 40 Hz. Cumulative angles do not themselves represent speed. |
| `encoder_raw_surface_mps` in fixed40 sidecar | Fixed 25 ms left/right encoder rate. Causal target; not future feedback in rollout. |
| old raw sidecar rate | Source-stamp-divided diagnostic rate. Not the canonical fixed-cadence definition. |
| `frames[:,5:7]` / `sensor_frames[:,2:4]` | Existing causal approximately-100 ms rear encoder-speed proxy. It is filtered wheel rotation, not body speed. Its timing must be stated whenever called a wheel signal/state. |
| Production `wheel_raw_mps`, `wheel_mapped_mps`, `wheel_packet_mps` | Exact local production observer replay: rolling 100 ms source-time wheel magnitude, calibrated rolling magnitude, and current-packet magnitude. Diagnostic measurement outputs, including reset/burst flags; they are not future plant inputs. |
| IMU `linear_acceleration.{x,y}`, `angular_velocity.z` | Direct sensor channels. Acceleration includes sensor reference/gravity convention; it is not interchangeable with simulator COM acceleration labels. |
| IMU quaternion roll and `angular_velocity.x` | Measured roll and roll rate. Prior tests did not promote roll; any plant that needs roll must predict it internally rather than read future measured roll. |
| `simulator_linear_acceleration` | Packet-aligned simulator COM acceleration labels, not IMU measurements and not rollout inputs. |

No fixed numeric actuator group delay is asserted by this inventory: the recorded channels are packet-aligned within the existing 30 ms association limit, while command-to-feedback response/lag is a model quantity to estimate. Likewise, the approximately 100 ms proxy's exact group delay is not claimed beyond its causal trailing-window definition.

## Production odometry replay semantics

For WP16's comparison, production outputs must be generated from the repository's current C++ `SensorPacketAssembler` and `OdometryObserver`, with the deployment `sensor_odometry.yaml`. The assembler only processes a packet when left encoder, right encoder, and IMU share the same source stamp. `wheel_raw_mps` is an absolute left/right mean over the configured rolling source-time window; `wheel_mapped_mps` applies the configured speed table; `wheel_packet_mps` is the absolute mean over the current source interval. Keep `wheel_update_used`, `wheel_burst_rejected`, `reset_epoch`, `timing_degraded`, and packet validity with the signal. A pre-existing shared library without a source/config hash is not sufficient evidence that it matches current production code.

## Operating-region names

Use the handoff's exact speed bins S0–S5, absolute steering bins D0–D4, steering-rate bins R0–R3, throttle-slew bins T0–T3, and mismatch bins M0–M6. Record steering sign separately. For throttle amplitude, retain the established race-domain edges `[-1.001,-0.05,0.05,0.20,0.40,0.60,0.80,1.001]`; name negative command values as braking/negative command, not as a fabricated coast regime. Required combined labels include high-speed/near-straight, high-speed/moderate-steering, 7–9 m/s/high-steering, low-speed/high-steering, braking/release, throttle pickup, steering turn-in/unwind, simultaneous steering+throttle transition, large wheel/body mismatch, and ordinary practice-track region. Always report independent run counts; fewer than three runs is weak evidence.

