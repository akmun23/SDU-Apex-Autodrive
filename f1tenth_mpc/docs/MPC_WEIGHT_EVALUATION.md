# MPC objective weights and offline screening

This note covers the MPC objective only. The raceline is treated as frozen
input data; this work does not modify, regenerate, or evaluate raceline
geometry.

## Weight sets

The BachelorProject `MPC` implementation defines the following default
objective:

| Term | BachelorProject weight | Meaning |
|---|---:|---|
| lateral error `e_y` | 1500 | lateral tracking error |
| heading error `e_psi` | 50 | tangent-heading error |
| longitudinal velocity `u` | 200 | speed-profile error |
| lateral velocity `v` | 5 | lateral-motion error |
| yaw rate `r` | 1.5 | yaw-rate reference error |
| steering effort | 2 | steering input effort |
| acceleration effort | 0.5 | longitudinal input effort |
| steering-rate change | 5 | steering smoothness |
| acceleration-rate change | 5 | longitudinal smoothness |
| effective steering | 1 | steering feed-forward bias |

The current 10-state RTI controller has additional command states. Its
production YAML is currently:

| Term | Production weight |
|---|---:|
| `e_y` | 150 |
| `e_psi` | 10 |
| actual-speed tracking `u` | 50 |
| overspeed `u` | 150 |
| target-speed command state | 20 |
| `v` | 0 |
| `r` | 1.5 |
| steering-command state | 5 |
| steering-rate input | 5 |
| target-speed-rate input | 0.5 |
| steering-rate change | 10 |
| target-speed-rate change | 5 |
| terminal multiplier | 3 |

`u` and `target_speed_state` are different terms. Increasing the first makes
the optimizer value actual speed-profile tracking. Increasing the second makes
it value maintaining the internal target-speed command. They must not be
collapsed into one number when investigating a corner failure.

The node parameter fallbacks are now identical to the production YAML. A
missing or renamed YAML therefore cannot silently switch back to the old
1500/50/200 BachelorProject defaults.

## Offline evaluator

`tools/evaluate_mpc_weight_grid.py` runs the same RTI core against recorded
MPC-authority event streams. It consumes only the runtime-legal event topics
(`/odom`, `/current_map_pose`, and `/cmd/speed`) and exposes every objective
term. `tools/mpc_weight_screen_candidates.csv` contains an isolation set and
user-requested moderate-heading/high-velocity candidates.

Example inside the Humble development container:

```bash
python3 tools/evaluate_mpc_weight_grid.py \
  --binary /workspace/build/f1tenth_mpc/mpc_rti_offline_replay \
  --events /path/to/mpc_authority/events.csv \
  --candidates tools/mpc_weight_screen_candidates.csv \
  --output /tmp/mpc_weight_screen.csv --jobs 2
```

The evaluator deliberately calls this an **offline screen**, not a lap-time
oracle. The recorded state trajectory is fixed, so it can compare solver
feasibility, nonlinear rollout acceptance, predicted corridor slack, and
first-action changes. It cannot predict closed-loop lap time for a new weight
set. The leading candidates must be run with MPC as the live authority and a
collision must terminate that run immediately.

The screen is most useful when the `--events` argument contains several
independent MPC-authority traces, for example one clean run at startup and
one or more runs with different lap phases. A candidate is only retained for
live testing when it remains feasible across the traces and does not create a
large first-action discontinuity. This is a model/solver compatibility gate,
not a replacement for closed-loop speed measurement.

The added user-profile candidates deliberately keep `e_y` moderate, reduce
heading weight, and increase speed tracking. They are test profiles only;
they must not be promoted from fixed-trace ranking because changing weights
changes the future state trajectory that the fixed trace cannot reproduce.

`tools/analyze_mpc_model_residuals.py` is the companion model check. It uses
accepted live-MPC diagnostics plus the offline simulator truth file and reports
signed yaw-rate residuals by N-step horizon, current speed, and absolute
curvature. On the production authority run, the N10 low-speed/high-curvature
group had a negative mean residual and approximately `0.93 rad/s` p95 absolute
error, while low-curvature groups were substantially smaller. This is evidence
of a corner-response/model mismatch, not evidence that a heading weight should
be set near zero.

## Multi-trace weight screen

The 16 candidates were screened on three independent MPC-authority event
streams. The aggregate fixed-trace results were:

| Candidate | Mean accepted | Mean nonlinear rejects | Minimum slack seen | Screen score |
|---|---:|---:|---:|---:|
| `only_high_lateral` (`e_y=1500`) | 94.98% | 4.70% | +0.000895 m | 0.9255 |
| `mapped_bachelor_moderate` (`e_y=500,e_psi=5,u=200`) | 94.62% | 5.08% | +0.000480 m | 0.9201 |
| `only_bachelor_control_terms` | 93.79% | 6.09% | -0.000334 m | 0.9070 |
| `only_high_heading` (`e_psi=50`) | 93.76% | 6.07% | -0.000110 m | 0.9068 |
| `production` (`150,10,50`) | 93.53% | 6.35% | -0.000661 m | 0.9025 |
| `only_high_velocity` (`u=200`) | 93.41% | 6.34% | -0.000650 m | 0.9013 |
| `user_low_heading_high_speed` | 93.35% | 6.47% | +0.000420 m | 0.9008 |
| `user_moderate_low_heading_speed` | 93.29% | 6.42% | +0.000420 m | 0.9001 |
| `user_moderate_speed_v5` (`u=250`) | 93.34% | 6.40% | -0.000777 m | 0.9000 |
| `user_moderate_balanced` (`125,2,200`) | 93.21% | 6.59% | +0.000002 m | 0.8987 |
| `user_moderate_balanced_steer5` | 93.20% | 6.58% | -0.000046 m | 0.8985 |
| `user_balanced_high_speed` | 92.93% | 6.81% | -0.000317 m | 0.8944 |
| `user_moderate_low_heading` | 92.93% | 6.86% | -0.000319 m | 0.8942 |
| `moderate_user_profile` | 92.92% | 6.92% | -0.000310 m | 0.8940 |
| `very_low_heading` (`e_psi=0.25`) | 92.88% | 6.90% | -0.000323 m | 0.8936 |
| `user_low_lateral_low_heading_speed` | 92.60% | 7.23% | -0.000461 m | 0.8888 |

These numbers are compatibility evidence only. They do not rank closed-loop
lap time because every candidate is evaluated on the same recorded state
trajectory. In particular, the user-requested speed-priority candidate with
`u=250` did not earn a live promotion merely because it was close to the
moderate candidate in this table.

## Live authority weight validation

The current production profile completed 12 post-ramp intervals with a mean
of `11.090 s`; only three of those intervals were at or below `11 s` and the
slowest was `11.614 s`. The separate moderate candidate
`mpc_weight_test_moderate_balanced_authority.yaml` used:

```text
e_y=125, e_psi=2, u=200, u_overspeed=200, target_speed_state=20,
v=5, r=1.5, steering_command=3, steering_rate=5,
target_speed_rate=0.5, steering_rate_change=10,
target_speed_rate_change=5, terminal_multiplier=3
```

It was run with MPC as the sole authority in batchmode. It completed the
first ramp lap plus 10 further intervals without collision:

```text
11.015, 10.770, 10.928, 10.836, 10.922,
10.953, 10.971, 10.929, 10.927, 11.008 s
```

The post-ramp mean was `10.926 s`; eight of ten intervals were at or below
`11 s`. This is the best validated weight profile so far, but it is not yet
claimed as a 10/10 sub-11-second result. The `u=250` candidate collided in
the first corner before a lap was completed, so it is rejected as a speed
push despite the offline screen being similar. Collision is terminal and no
reset/recovery was used.

The moderate profile is therefore the current live candidate, not yet a
production promotion. The next useful weight experiment must change one
objective group at a time around this profile and be validated by another
collision-terminal MPC-authority run.

## First results

The first screen used 5,813 cycles from an existing long MPC-authority run,
with the production corridor and 100 ADMM iterations. Results are ordered by
the evaluator's fixed-trace compatibility score; this score is not a speed
claim.

| Candidate | Accepted | Nonlinear rejects | Residual rejects | Minimum predicted slack |
|---|---:|---:|---:|---:|
| BachelorProject weights | 96.53% | 187 | 15 | +0.000765 m |
| high lateral only (`e_y=1500`) | 96.49% | 189 | 15 | +0.000895 m |
| high heading only (`e_psi=50`) | 95.18% | 275 | 5 | -0.000110 m |
| current production | 94.86% | 295 | 4 | -0.000428 m |
| high velocity only (`u=200`) | 94.72% | 299 | 8 | +0.000471 m |
| moderate user profile (`e_y=100,e_psi=2,u=200`) | 94.50% | 313 | 7 | -0.000310 m |
| very low heading (`e_psi=0.25`) | 94.39% | 317 | 9 | -0.000323 m |

This does **not** justify promoting the BachelorProject or high-lateral
profile: both violate the requested preference for moderate lateral weight,
and the fixed-trace screen cannot measure their closed-loop speed. It does
show that simply reducing heading weight and increasing velocity weight does
not currently improve nonlinear feasibility. Those candidates need live MPC
authority validation after the model issue below is addressed.

## Iteration and RTI evidence

On the same production profile and holdout:

| Configuration | Nonlinear rejects | Residual rejects |
|---|---:|---:|
| 50 ADMM iterations, adaptive RTI | 288 | 18 |
| 100 ADMM iterations, adaptive RTI | 295 | 4 |
| 200 ADMM iterations, adaptive RTI | 297 | 3 |
| 100 iterations, R1 only | 275 | 426 |
| 100 iterations, R2 only | 298 | 1 |
| 100 iterations, adaptive R1/R2 | 295 | 4 |

The second RTI pass is useful for residual convergence, but additional ADMM
iterations do not remove the nonlinear failures. The next work therefore
belongs in the model/rollout mismatch and in weight validation, not in blindly
raising the iteration cap.

## Model warning exposed by authority data

The live authority trace shows the model turning faster than the Unity object
in the tight high-curvature transition: the predicted yaw rate becomes roughly
`-2.5` to `-3 rad/s` while legal `/odom` remains roughly `-1.1` to `-1.4
rad/s`. The predicted lateral error then recovers too early and the optimizer
continues increasing target speed. This is a model optimism problem; changing
weights can mask it but cannot make the horizon prediction correct.

The next MPC-only investigation should therefore compare, on the same legal
state/action samples, the identified yaw response and steering-state delay
against the accepted rollout at stages 1, 5, 10, and 30. Only after that
comparison should a moderate-heading/high-velocity profile be promoted to a
live authority run.

## Identified yaw-response A/B

The model now exposes an optional curvature-dependent response correction and
the replay tool accepts `--yaw-rate-tau`, `--yaw-rate-gain`, and
`--yaw-curvature-reduction` (with start/end curvature thresholds). The
production YAML leaves the correction disabled and retains the previously
validated constants until a live authority run proves otherwise.

On the 5,813-cycle authority holdout, with the production weights and the
same 100-iteration adaptive-Rho/adaptive-RTI settings:

| Model variant | Accepted | Nonlinear rejects | Residual rejects |
|---|---:|---:|---:|
| Current `tau=0.087735`, gain `3.011897` | 5514 | 295 | 4 |
| `tau=0.030`, gain `3.011897` | 5526 | 286 | 1 |
| Curvature reduction `0.8 / m` | 5511 | 296 | 6 |
| Curvature reduction `1.3 / m` | 5508 | 298 | 7 |

This is a fixed-trace solver/model screen, not live proof. The small
improvement from the shorter response time is worth one MPC-only batchmode
authority A/B. The curvature reduction is not promoted: it worsened the
replay result and remains an investigation parameter only. The run must still
be collision-terminal and judged by actual post-ramp lap times.
