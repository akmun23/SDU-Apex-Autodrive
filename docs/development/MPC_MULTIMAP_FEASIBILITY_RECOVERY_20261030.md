# MPC multi-map feasibility recovery — 2026-10-30

## Scope and requirements

Goal: restore a repeatable, fast practice-track MPC while improving controller robustness across the Autodrive practice and IROS maps. This is a single generic MPC with source-plant validation, **not** steering rules indexed by a map or a specific corner. The optimized raceline is a target, not an absolute corridor centerline command. Its nominal 5 cm planning clearance must not be treated as an additional hard 5 cm MPC constraint. The hard collision envelope must represent the whole vehicle and a deliberate physical/model/localization uncertainty allowance. Never accept a solver candidate merely because a QP converged or a short prefix is safe.

## Implemented

- `f1tenth_mpc/include/mpc_viability_recovery.hpp`: track-independent terminal slip/heading checks, deterministic tapered steering/throttle trial commands, and an *experimental* steady-actuation bridge generator.
- `f1tenth_mpc/src/mpc_controller_node.cpp`: opt-in, exact 30-stage/750 ms source-plant viability repairs on infeasible or unstable MPC candidates, plus a proactive minimum-predicted-clearance monitor. A candidate must pass full nonlinear source-plant and body-corridor validation before actual publication, including the existing 20 ms publication deadline. The accepted plan and warm start are committed only after publication.
- `f1tenth_mpc/include/mpc_rti.h`: new `accepted_viability_recovery` status. Updated supervisor, diagnostic reporting and trajectory audit so this certified action does not count as a fatal fault.
- `f1tenth_mpc/config/mpc_competition.yaml`: **base competition recovery is OFF** until a runtime configuration is explicitly chosen; the conservative original competition corridor remains unchanged. The generic tapered mode can be enabled with `viability_recovery_enabled`. Independent `viability_steady_bridge_enabled` remains OFF by default, isolating insufficiently validated wider-search research.
- `f1tenth_mpc/config/mpc_practice_fastline.yaml`: **restored the practice fastline's default runtime** (not a separate manual overlay) to use the tested 2 cm physical corridor, 2 cm early-warning threshold and full-horizon tapered recovery. The steady bridge is disabled. The normal practice launch inherits these parameters automatically.
- `f1tenth_mpc/test/test_mpc_viability_recovery.cpp`: bounded controls, horizon tail preservation, terminal screen regression. C++/ROS suite: 176 tests passing on 2026-10-30.

### Correct interpretation of clearance

`corridor_margin_m` / `first_prediction_corridor_margin_m` represent a physical-model clearance allowance applied to the source-derived corridor, **not the raceline's planning buffer**. Experimental practice runs have used a 2 cm allowance. Experimental IROS runs have used 1 cm where 2 cm excluded a genuine narrow section. At an IROS rejection near station 37.76 m, the map's sampled physical footprint clearance was about 1.12 cm: 2 cm was not feasible but 1 cm could pass. This is **not** a proof that 1 cm will always suffice under uncertainty; it is a simulation experiment, and the final production solution should ideally validate world-frame body footprint against a distance field instead of relying solely on lateral Frenet corridor bounds.

## Completed simulator evidence

One warmup plus ten scored laps corresponds to `11` in the supervisor. All times are Unity `/lap_time`-derived scored averages, omitting the warmup.

| Run (under `live_runs/`) | Parameters | Scored laps | Mean lap | Source-certified rescue actions | Publication deadline violations | Outcome |
|---|---|---:|---:|---:|---:|---|
| `practice_viability_proactive_2cm_20261030_01` | practice 2 cm physical, tapered, 2 cm lookahead | 10 | 4.9971 s | 25 | 2 | Complete, no reported collision |
| `practice_viability_proactive_2cm_20261030_02` | identical independent repeat | 10 | 5.0116 s | 25 | 0 | Complete, no reported collision |
| `practice_viability_tapered_2cm_20261030_01` | latest default tapered-only, 2 cm physical, 2 cm lookahead | 10 | 5.0638 s | 30 | 0 | Complete, no reported collision; fastest 4.8818 s |
| `practice_viability_2cm_latest_20261030_01` | aggressive 5 cm lookahead, steady bridge | 10 | 5.3127 s | 190 | 0 | Complete but excessive recovery overhead and slower |
| `practice_viability_2cm_restore_20261030_01` | 2 cm lookahead with steady bridge | 3 | — | — | 1 | Unpublished MPC fault after 4 total laps: cannot promote |
| `practice_viability_hierarchical_20261030_01` | 2 cm lookahead, tapered first then experimental steady fallback | 1 | — | 5 | 0 | Unpublished corridor fault; hierarchical bridge is NOT promoted |
| `practice_default_mpc_recovered_20261030_01` | **Normal practice launch, no external overlay** | 10 | **5.0471 s** | 17 | 0 | Complete, no reported collision or scored fault; fastest 4.9018 s |
| `iros_viability_1cm_5mps_20261030_01` | IROS 5 m/s, 1 cm physical, rescue enabled | 10 | 9.6849 s | 5 | 0 | Complete, no reported collision (prior code variant) |
| `iros_viability_1cm_6p5mps_steady_20261030_01` | IROS 6.5 m/s, steady bridge | 10 | 8.6219 s | 19 | 0 | Complete, no reported collision |
| `iros_viability_1cm_6p5mps_steady_20261030_02` | independent repeat at 6.5 m/s | 0 | — | 4 | 0 | MPC source-plant/corridor fault near station 37.84 m |
| `iros_viability_1cm_6p5mps_early_lowdrive_20261030_01` | 6.5 m/s, 5 cm lookahead and 1%-2.5% bridge options | 10 | 8.8305 s | 94 | 0 | Complete but slower and overactive |
| `iros_viability_1cm_6p5mps_early_lowdrive_20261030_02` | same independent repeat | 3 | — | — | 0 | MPC corridor fault near station 37.72 m |

### Map/body geometry discrepancy confirmed, not yet fixed in runtime

At the IROS 5 m/s tapered-only stop (`iros_viability_tapered_1cm_5mps_20261030_01`, station 37.808 m), the nonlinear Frenet **corridor validator** rejected the candidate at stage zero even though independently reconstructed full-body map clearance was **0.013147 m** (i.e. +0.003147 m above the configured 1 cm physical margin). The immediately measured state had clearance 0.010091 m, barely +0.000091 m above the physical margin. This is a **false rejection relative to the sampled map geometry**, but the tiny numerical reserve means that neither localization uncertainty nor intervening swept motion have been certified. Do not simply force acceptance based on this one-point observation.

Reproduce generically with `tools/audit_mpc_wall_clearance.py` (3 offline unit tests, all passing), which reconstructs the rear-axle world pose from recorded MPC Frenet progress and state, samples the original PGM with unknown occupancy treated as unsafe, and checks the 0.510 × 0.273 m body envelope through `MapGrid.footprint_clearance`. Audited JSON evidence: `live_runs/mpc_collision_margin_experiment_20261030/iros_tapered_5mps_map_vs_corridor.json`. The next engineering change is a **time-bounded full-horizon map-coordinate swept-body certificate** that is authoritative for physical collision checks, with a separate uncertainty reserve, while keeping the conservative corridor for solver approximation. It must preserve the validated practice default until independently tested on both maps.

**Do not claim robust high-speed IROS 6.5 m/s performance yet.** Successful single runs are not proof of repeatability, and the 5 cm proactive reserve search consumed substantial racing time. At the second 6.5 m/s failure, the source command-time state and 25 ms prediction were within millimeters of/over the map footprint, so relaxing hard margins further is not an acceptable solution. Offline exact-plant replay exposed feasible constant-control candidates earlier in the failure sequence; the online search must become more anticipatory and preserve successor feasibility rather than merely minimizing terminal slip.

## Runtime configuration and reproducibility

The default practice launch now reads `f1tenth_mpc/config/mpc_practice_fastline.yaml` directly; **no external overlay is needed**. For reproducible experimental comparisons the equivalent overlay is `live_runs/mpc_collision_margin_experiment_20261030/viability_2cm_speed_restore.yaml`. The pre-edit profile is saved at `.../mpc_practice_fastline_pre_recovery.yaml`. General experimental original: `.../viability_2cm.yaml`. IROS at a proven lower speed: `.../viability_iros_1cm_5mps.yaml`. Higher-speed research only: `.../viability_iros_1cm_6p5mps_guarded.yaml` (the guarded profile explicitly requests the 5 cm *proactive* warning, not 5 cm hard physical clearance).

Build/test with the repository checkout (do not wipe the other developer's uncommitted optimizer/GUI work):

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select f1tenth_mpc --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select f1tenth_mpc
colcon test-result --test-result-base build/f1tenth_mpc
```

Repeat with existing trial scripts and an 11-lap supervisor, keeping the concurrent optimizer paused via `tools.optimizer_mpc_validation.yield_optimizer` for each live Unity experiment. The exact run artifacts contain replayable ROS bags, manifests, controller logs, watcher reports, JSON analyses, and per-cycle diagnostics. Never overwrite them. No Git commit was made because this checkout includes substantial unrelated staged/uncommitted optimizer and dashboard changes. The steady bridge is still present behind its independent opt-in control for offline research; practice runtime explicitly disables it, and a single IROS success must not be promoted as repeatable.

## Open engineering work

Current state at handoff: the automatic no-overlay practice launcher has passed ten scored Unity laps (5.0471 s mean; 0 deadline violations). The cross-map IROS 5 m/s tapered-only repeat failed at a narrow corner; 6.5 m/s runs were fast but not repeatable. The optional steady-bridge/hierarchical approach is **unvalidated** and not enabled by normal practice. Do not revert the default practice profile based on single-run IROS gains. The simulator/optimizer validation lease was released and the background optimizer resumed after each completed experiment.


1. Run independent multi-map repeats with the same **default tapered-only** policy and practical physical reserves. Separate speed-profile feasibility from controller robustness: a speed limit is not a guarantee of a feasible next state.
2. Add a map-independent terminal *recursive viability* metric. A 750 ms exact collision-free horizon ending at 1–2 mm clearance is insufficient if no safe continuation exists. Prefer a terminal recoverable set, emergency braking viability, and minimum clearance in following horizons, not only nominal cost.
3. Make source-model footprint collision constraints directly world-map-aware (signed-distance/occupancy grid, sensor/pose uncertainty bound). Allow deliberate use of raceline planning padding while continuing to reject physical collisions and unknown occupancy.
4. Improve solver/RTI speed without relaxing hard physical constraints. Keep 200 Hz control and 20 ms publication gates observable, retain cached source sensitivities, and avoid expensive emergency search when the nominal is safely feasible.
5. Revisit track-independent reference-speed anticipation and how objective weights trade speed against corridor viability. Never 'solve' the tight IROS corner by inserting hardcoded map-point steering corrections.

