# Source/Unity command-epoch accuracy: 2026-11-01

## Root cause: command timing, not necessarily tire physics

The earlier practice run had 16 overlapping 750ms position-rollout outliers at unwrapped progress 160–161.5m, worst 9.35cm. One 25ms interval at ROS stamp 1793543092414731421 (225ms before the worst window) was decisive:
- Odometry source command throttle: 0.1213041
- Actual Bridge request sent throttle: 0.079, matching Unity feedback 0.079
- One-step new source forward-speed error: +0.21331m/s; yaw error -0.30873rad/s
- Counterfactual correct-previous-command one-step forward-speed error: -0.01406m/s

The odometry previously guessed the input associated with a Unity request by sampling HEADERLESS actuator-command ROS callback receipt times. This can select a newer command near a request boundary. The source-derived delay queue was correct only if given the correct request command.

## Implemented

1. Bridge now publishes the EXACT rounded outgoing request payload as a team-generated sdu_apex_msgs/msg/DriveCommand on /odom/bridge_request_command, with its actual sensor request header timestamp, BEFORE Socket.IO emit.
2. Odometry uses a bounded, exact-integer-nanosecond request history to apply the matching stamped Bridge command rather than an inferred receipt-time command. An explicit logged compatibility fallback remains only for absent matching records; any use disqualifies a run from strict accuracy certification.
3. The stamped topic is included in both recording profiles; a new tools/audit_bridge_request_command_epochs.py verifier detects missing/mismatched steering/throttle inputs. C++ request-history and Python bridge/epoch tests cover the critical ordering and alignment contracts.
4. No MPC weights, source tire parameters, fallbacks or race strategies were changed.

## Live validation, original Unity simulator and unchanged practice MPC

Run phase13_bridge_request_epoch_accuracy_20261101_01: 3 laps, no collisions, **646/646** exact Bridge-command timestamps and values, zero missing/mismatched.

Run phase13_bridge_request_epoch_accuracy_20261101_02: 8 laps, no collisions, 213.82m odometry travel, **1627/1627** exact stamped commands and IMU epochs, zero mismatches.

The original problematic progress 159–162m, 38 rolling windows each:
- Before: worst 9.353cm, 16 windows >5cm.
- After: worst 2.036cm, zero windows >5cm.

These are different closed-loop races; the difference alone is NOT a controlled causal estimate. The exact old command mismatch and new epoch parity provide direct evidence.

One-step source/Unity RMSE, previous versus new race:
- Longitudinal speed 0.01359 -> 0.009835m/s
- COM lateral speed 0.001972 -> 0.001070m/s
- Yaw rate 0.011852 -> 0.005897rad/s

Original full run had 2191 rolling 750ms windows, 16 >5cm, no truth self-inconsistency. New 8-lap run has 1597 windows, 3 >5cm among truth-consistent windows. BUT 30 overlapping windows failed the 5cm per-25ms truth pose-versus-twist consistency test: one Unity telemetry interval mismatched by **5.427cm**. The raw new count was 10 >5cm, of which scoring excludes self-inconsistent windows. The case index now stores exact rejected timestamps and discrepancies as invalid_truth_closure_windows.

Three otherwise valid >5cm raw-pose errors at progress 71.44, 75.91 and 152.70m have source-versus-Unity-velocity-integrated endpoint errors of only ~1.5–2.6cm, while reported Unity pose and velocity disagree by 3.2–3.7cm in one 25ms interval. These data do NOT establish a wrong tire model. Preserve independent raw-pose and velocity-integrated comparisons.

## Remaining accuracy gates

- Reverse-engineer and resolve the Unity pose/twist sampling phase at the single 5.43cm self-inconsistent interval; never fit source dynamics to internally inconsistent truth.
- Independently repeat the live 8-lap capture and verify strict stamped-command parity over all packets.
- Continue source-adjoint exact cost-derivative parity improvements. After fixing its frozen-target mismatch the initial 60-component relative errors were 0.50%,0.70%,1.24% on three difficult samples, but later slip regimes remain nonzero.
- The physically nonlinear MPC optimizer is not yet qualified or enabled in production. 20ms deadline and full competition race certification remain.

## Repeated live test and accuracy/optimization separation

Run phase13_bridge_request_epoch_accuracy_20261101_03 repeated the same unchanged practice configuration. Before a controller SQP rejection after seven completed laps, its bag contains **1590/1590** exactly matched stamped Bridge/IMU/odometry epochs, zero steering and throttle discrepancies. The source/Unity audit has **one of 1559 truth-consistent 750ms windows over5cm** (5.17cm max), no truth pose/twist invalid exclusions. The strong 160m steering reversal no longer showed the earlier concentrated 16-window failure.

This third run was NOT a completed eight-lap success. At progress ~227.063m, the **normal QP** proposed highly nonlinear steering/throttle changes and exact source validation rejected the 750ms corridor at stage15; the strict watcher stopped. That failure is a distinct optimization/feasibility defect, not a failure of stamped command synchronization.

Exact logged failure diagnostic and recreated source cold-state replay:
live_runs/mpc_phase13_prediction_fidelity_20261101/lap7_rejected_source_state

Nominal controls themselves failed the strict 750ms corridor at stage15. An OFFLINE source-integrated 60-control adjoint optimizer, initialized with the same logged 22-channel internal plant state and the SAME-EPOCH nominal controls, improved exact source cost **177160 -> 57463** over eight iterations. The first hard-source-feasible 750ms horizon was found on **iteration7** (minimum clearance 0.69cm); iteration8 increased clearance to **8.93cm**. This proves a source-feasible trajectory exists, while the normal QP could not find it. Intermediate iterates were evaluated with widened diagnostic corridor and final candidates independently validated on ORIGINAL hard geometry.

IMPORTANT reproducibility correction: the first version of the offline fixture had accidentally inserted the next-epoch warm-start duplication TWICE, leading to a different control seed (cost 175481 -> 57042, feasible on iteration6, final clearance 10.16cm). The corrected fixture in tools/extract_mpc_rejection_source_case.py contains the exact 30-stage command sequence; the replay itself performs the required next-epoch duplication once. Discard the misaligned older benchmark. Both trials used the exact same physical source state, but not identical source control seeds.

Measured offline nonlinear iteration cost with the previous seed: 7.08–8.70ms per iteration, source Jacobian+adjoint 2.58–3.27ms. Seven iterations cannot fit unchanged into the 20ms live MPC budget. Accuracy and per-iteration source curvature/viability must improve before any runtime promotion. No new fallback or trust region has been added.
