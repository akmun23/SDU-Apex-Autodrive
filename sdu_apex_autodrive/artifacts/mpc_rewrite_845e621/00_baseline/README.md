# MPC rewrite Phase 0 baseline

Captured 2026-09-18 before beginning the handoff-driven rewrite. The checkout
HEAD matches the handoff's pinned source SHA, but the worktree was already
dirty with an in-progress 10-state Riccati/weight change. Therefore these
results describe that dirty worktree, not a pristine build of the pinned
commit. No simulator or Unity process was started for this baseline.

The package was built and tested in the running ROS 2 Humble workspace
container (`sdu_apex_autodrive`) with `BUILD_TESTING=ON`. All three current MPC
package tests passed. The standalone N30 core benchmark is a no-ROS microbench
only; it does not establish callback-to-command timing or driving acceptance.

Accepted yaw and longitudinal holdout numbers below are copied from the
existing model-identification report in
`f1tenth_mpc/docs/SIMULATOR_NATIVE_MODEL_WORKLIST.md`. They are replay evidence,
not MPC closed-loop results. The failed 4 m/s state-age run is retained at
`sdu_apex_autodrive/artifacts/simulator_trace/mpc_target_response_candidate_4mps_20260918`.

## Phase 0 gate

- Pinned HEAD confirmed: `845e621989daaf1e9fbe041723f7e6066a6f01d2`.
- Current dirty worktree captured in `git_status.txt`.
- Current package tests: 3 passed, 0 failed.
- Accepted model replay and trajectory baseline recorded in `metrics.json`.
- No model, localization, actuator, simulator, or physics behavior was changed
  as part of this evidence capture.

The 10-state worktree under test is not the target architecture in the new
handoff. The target is 7 nonlinear plant states plus two previous-control
states (9 QP states); Phase 1 regression tests must be established before that
rewrite proceeds.
