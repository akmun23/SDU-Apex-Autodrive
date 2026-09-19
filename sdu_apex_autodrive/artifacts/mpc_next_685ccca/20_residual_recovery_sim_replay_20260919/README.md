# MPC replay on PP simulator data: residual recovery

This is an offline replay of the real PP-driven simulator capture:

`../11_shadow/pp_mpc_shadow_icra_20260918/events.csv.gz`

The replay consumes only the same values that the production MPC is allowed
to use: recorded `/odom`, `/current_map_pose`, `/cmd/speed`, and the accepted
raceline. Simulator packet position, collision count, and ground truth are not
controller inputs; they were used only to diagnose the one rejected sample.

## Result

- 13,030 synchronized simulator cycles replayed.
- 13,030 accepted (`100%`); no projection, path, timing, or corridor rejects.
- The holdout covers about 16.27 raceline laps and reaches 8.59 m/s odometry
  speed.
- Adaptive rho remained enabled, with a 100-iteration cap and N30 at 40 Hz
  (`dt=0.025 s`).
- The MPC output stayed within the configured simulator envelope: target speed
  max 8.85 m/s, steering max 0.4705 rad, target-speed rate in `[-8, 3]`
  m/s², and steering-rate in `[-3.2, 3.2]` rad/s.
- Predicted corridor slack was always positive; minimum was 0.2173 m.

## Specific recovered sample

At event `210722`, PP had issued a stop command while `/odom` still reported
`2.447 m/s`. R1 reached the 100-iteration cap with residual `0.1998`, so it
was not published. Its finite projected candidate passed the exact nonlinear
rollout and was used only as the same-cycle R2 seed. R2 converged in 53
iterations and produced full braking (`-8 m/s²`) with 0.2611 m minimum
predicted corridor slack. This is recorded by the action CSV with trigger mask
`4096` (`MPC_RTI2_TRIGGER_R1_RESIDUAL_RECOVERY`).

The recovery ceiling is explicitly bounded at `0.25`, while the normal
degraded gate remains `0.05`. R1-only mode remains strict; only a converged,
nonlinear-feasible R2 candidate can be selected.

## What this proves and does not prove

It proves that the current MPC can produce finite, bounded, corridor-safe
commands at the states actually visited by a fast PP simulator run, including
the observed stop transition. It does not yet prove closed-loop MPC authority:
the replay does not apply MPC commands back to Unity. A live authority run is
the next acceptance test after this replay milestone.

Command used inside the Humble container:

```text
build/f1tenth_mpc/mpc_rti_offline_replay \
  <(gzip -dc src/sdu_apex_autodrive/artifacts/mpc_next_685ccca/11_shadow/pp_mpc_shadow_icra_20260918/events.csv.gz) \
  100 0.01 --prefactorized --adaptive-rho --rti-mode adaptive \
  --actions src/sdu_apex_autodrive/artifacts/mpc_next_685ccca/20_residual_recovery_sim_replay_20260919/adaptive_recovery_actions.csv
```
