# Simulation-only MPC continuation on predicted corridor failure (2026-11-01)

A future MPC-predicted corridor violation is not an observed Unity collision. Previously, the MPC latched on rejection and the watcher stopped the whole test before observing the actual effect of later 40Hz control cycles.

## Implementation

- Set SDU_APEX_IROS_CONTINUE_ON_MPC_FAULT=1 on tools/run_iros_trial.sh to opt in. Default remains the strict, qualifying behavior.
- The simulator controller container receives SDU_APEX_SIM_DIAGNOSTIC_ONLY=1. Only in this explicitly tagged simulation-only environment, a normal solver candidate rejected solely for a predicted corridor violation can issue its existing first QP command, provided it is finite, within physical steering/throttle limits, and can be published in less than 20ms. The diagnostic still reports the original REJECTED horizon and its failing stage; the controller continues solving.
- Other controller faults do not latch in the simulator diagnostic environment. No arbitrary new command is fabricated when the QP lacks an admissible proposal.
- The watcher receives --continue-after-mpc-rejection, records counts and samples but does not abort for rejected MPC horizons, localization delays, or odometry stalls. The experiment ends on ACTUAL Unity collision, target lap completion, or a bounded wall-clock timeout.
- The runner writes a supervision_policy.txt record to differentiate unsafe simulation diagnostics from qualified ten-lap autonomous races.
- The physical-car controller, the normal strict simulator, the source plant model, command safety envelopes and original wall margin are unchanged. This is not a production-safety promotion.

## Measured Unity competition results

1. comp_sim_nonfatal_delta_20261101_01: full 6.925s geometry-only IPOPT path, command-increment normal-MPC QP. Over 100 observed predicted corridor rejections while the car kept driving, 172 simulation-only first-command proposals were issued in the controller logs. Actual collision at 103.538s from bag start, 0 laps, 0 missed publication deadlines, worst measured gate 19.540ms. The watcher ended with SCREEN_ABORT: collision.
2. comp_sim_nonfatal_canonical_20261101_01: same full-speed geometry-only path and simulator continuation policy using the canonical MPC. Actual collision at 3.829s from bag start, 0 laps, 0 missed publication deadlines, worst gate 18.548ms. The watcher ended with SCREEN_ABORT: collision.

Both run folders include watcher.log, controller.log, mpc_run_report.json, mpc_cycle_detail.csv, full diagnostics JSONL, raceline and parameter overlays. Neither run is safe/fully-qualified. The result confirms the previous simulator halt was premature for diagnostics, but actual physical corner-control failures remain.

## Tests and use

21/21 MPC CTest suite and 9/9 Python watcher unit tests pass. Command: SDU_APEX_IROS_CONTINUE_ON_MPC_FAULT=1 SDU_APEX_IROS_TOTAL_LAPS=11 bash tools/run_iros_trial.sh <new-unique-run-id>. For strict qualification, omit the environment flag.
