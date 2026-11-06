# Competition: normal-MPC solver experiments, 2026-11-01

## Target and immutable safety conditions

- Full-speed track: latest native IPOPT source witness, ipopt_best_6p925_source_witness.csv. Its 6.925s timing is OFFLINE, not an achieved Unity lap.
- Original Unity/PhysX plant/collision authority and actual footprint.
- 50mm vehicle-body clearance target, 20ms end-to-end command publication deadline.
- No speed reference reduction in the experiments below (unless otherwise marked).
- Race qualification requires 10 scored laps and multiple independent trials, not a warmup lap.

## Implemented, tested and removed

1. **Nonlinear SQP clearance-filter acceptance**: Accepted a fully source-validated candidate with larger physical wall reserve despite slightly higher tracking cost. Focused unit tests passed. Full-speed Unity: comp_filter_raw06_20261101_01, comp_filter_raw12_20261101_01, comp_filter_contour10_20261101_01 all aborted before lap 1, corridor stages 6, 5, 22. Entire filter code/flags/tests removed.
2. **Progress-dependent 3D wall polygon** in normal Riccati-ADMM QP: Initial Unity runs failed before the first iteration. Two exact bugs were temporarily fixed: the 2D preflight demanded the old finite e_y box even when 3D halfspaces replaced it, and the third progress-deviation polygon normal was not scaled by the state-scale transform. After fixes, direct and scaled ADMM regression tests and all 21 MPC test targets passed. Full-speed Unity rebuilds comp_progress3d_raw_fixedsolver_20261101_01, comp_progress3d_contour_fixedsolver_20261101_01 and comp_progress3d_raw_zero_preview_fixedsolver_20261101_01 still failed before lap 1: corridor stages 9, 16, 19. All 3D-specific code/tests/flags were removed; normal yaw-aware 2D footprint QP retained.
3. **Source-command throttle prior**, **stage-dependent line-search damping**, **terminal heading funnel**, and **source command seed retained after altered speed** tested at full speed and removed on failure. Run folders retain evidence.
4. **Clearance buffer diagnostic:** 30mm, 15mm and 5mm footprint margins with zero numerical tolerance did not yield a full race. Never promoted; 50mm remains the standard.
5. **One-pass RTI and nonlinear Riccati feedback:** comp_r1_native_20261101_01 failed before lap 1 (corridor stage 8). comp_r1_feedback075_20261101_01 reached 1 lap boundary then failed (invalid model). comp_r1_feedback100_20261101_01 reached 2 lap boundaries then failed at corridor stage 21. Its first 400 cycles included 107 viability recovery actions, not near-zero fallback.
6. **Normal MPC only, no recovery or fallback:** comp_pure_r1_feedback100_20261101_01 failed corridor stage 2 before any lap; comp_pure_r2_feedback100_20261101_01 failed corridor stage 1 before any lap. Increasing RTI passes alone did not confer feasibility.

## Root-cause evidence and next engineering target

The full-speed IPOPT source raceline had only approximately 1.85mm sampled EXTRA clearance beyond its nominal 50mm body buffer. One 5ms ROS callback is not a 25ms bridge command epoch, and prior warm-start shifting by a full model stage per callback can misalign the candidate with true queued-actuator timing. Prior fractional progress-alignment and naive command-epoch reset experiments failed in Unity; do not resurrect without a causal queue/epoch model.

**The issue remains normal nonlinear MPC recursive feasibility at high-speed/high-curvature on a wall-skimming source trajectory.** Next: coherent queue/epoch-aware state, warm start and feasible-successor model; and/or speed-preserving spatial objective that leaves physically achievable tracking reserve through corners. Fallback-only and weight-only changes are not primary fixes.

## Qualification

- No full-speed competition ten-lap completion or repeatability qualification.
- No new experimental algorithm promoted.
- Failed C++ branches removed; raw live_runs evidence retained.
- IPOPT optimization intentionally stopped.

## Follow-up: spatial anticipation, startup alignment, full-feedback RTI

- Smoothed inboard contour preview tested with otherwise identical full native IPOPT speed/commands, footprint and publication gate. Full-speed runs comp_contourpreview075_20261101_01, comp_contourpreview120_20261101_01, comp_contourpreview150_20261101_01 and comp_contourwide140_20261101_01 all aborted before lap 1 at nonlinear corridor stages 13, 23, 19 and 18. No contour parameter promotion.
- First accepted Unity sample on existing full-speed contour trial was at s=38.946m, with lateral tracking error +0.255m and heading error +0.362rad; the car was at 0.0025m/s while source desired 4.135m/s. Therefore the mismatch at startup is measurable and material, but not necessarily the only cause of failures on later laps.
- Tested a startup-only progressive speed ceiling, reaching full speed before scoring: comp_ramp050_20261101_01, comp_ramp075_20261101_01 and comp_ramp100_20261101_01 (half, three-quarters, one warmup lap). All failed prior to a lap, corridor stages 14, 19 and 13. Neither a start ramp nor more spatial preview alone resolves the core normal-MPC feasibility issue.
- Original controller production configs are unchanged. The experimental overlays remain in live_runs for comparison but not installed. No QP/solver experimental code remains from the failed 3D, throttle-prior, clearance-filter, terminal-heading or stagewise-globalization paths.

## Geometry-only IPOPT controller interface (user-requested separation)

The loaded IPOPT CSV remains untouched and contains its offline PhysX witness for separate optimizer analysis, but the competition MPC now sets `geometry_only_raceline_reference: true` in `f1tenth_mpc/config/mpc_competition.yaml`. Immediately after path parsing, speed envelope and optional spatial contour processing, the controller invalidates each in-memory `source_witness_valid` flag. Consequently `reference_at_progress`, cold- and warm-tail control seeds, QP steering target, exact cost, and source-witness recovery no longer use historical IPOPT steering/throttle or physical slip/yaw/heading states. The MPC uses spatial path curvature for steering feedforward and the desired speed, heading, lateral position and wall constraints as the reference. Source model and physical body checks are still used to validate predicted MPC commands.

The source-command steering-variation experiment was rolled back completely after trials without any scored competition lap: `comp_nativeincrement30_20261101_01` corridor stage 14; `comp_nativeincrement100_20261101_01` invalid model. All four zero-trust source-profile pseudo-Huber tests also failed before lap 1 (stages 17,15, invalid model,18). In the first geometry-only zero-trust high-variation trial (`comp_geometry_free80_20261101_01`) MPC reached approximately 4.4 m/s and failed with `corridor@0`; this is not qualified. The canonical geometry-only competition test `comp_geometry_only_canonical_20261101_01` compiled/initialized correctly but the Unity watcher aborted **before controller operation**, citing lack of first localized AMCL map pose. Thus no closed-loop MPC claim can be drawn from that canonical run. Its AMCL reported `travel=0/0.100 m` and the controller waited for localization. The IROS runner readiness condition was updated to recognize both old and geometry-only load log strings. The production practice profile was not modified.
