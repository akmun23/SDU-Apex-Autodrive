# Full-speed MPC cross-map robustness and practice qualification

Updated November 1, 2026. All controller architectural changes are allowed
provided practice speed remains competitive and the hard native source-plant
footprint, actuator, and 20 ms publication-safety gates are retained.

## Stop and freeze IPOPT

Stopped systemd user service sdu-track-optimizer-5c036cfcb327.service.
Verified no track_optimizer.py process remains active.
Best fully source-certified 6.925 s competition command witness:
live_runs/optimizer_iros_adaptive_regions_v6_20261030/exports/9a7d502bbd2e0aca/source_command_raceline_with_witness.csv
CSV SHA256: 59e39773f4b44b7768cf7e9130258f812ebe1984ab6b094462932dc8fe64b692.
Minimum sampled additional body-wall clearance 1.8518 mm above 50 mm.
This is NOT yet a closed-loop race. The optimizer's best metadata is
archived under paused_20261101/best_at_stop.json.

## Protected five-race practice baseline

Practice reference:
f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv.
Original five independent fully clean races at lateral 4500, heading 1450,
feedback gain 0.75: 49.162, 49.091, 49.175, 49.291, 49.092 s.
Mean 49.1622 s, zero hard deadline misses. Config:
f1tenth_mpc/config/mpc_practice_feedback_qualified_20261031.yaml.
Do NOT replace it without five independent clean faster races, each below
49.5 s, zero collisions/fatal faults/20 ms publication misses.
Lateral 4000, proactive recovery enabled, was 5/5 clean but mean 49.275 s
and had a 49.575 s outlier. Lateral 4250 without proactive recovery was
49.096 s in its first race and aborted with corridor@1 in its second.
Lateral 4250 with proactive recovery finished a first race in 49.062 s,
max callback 18.288 ms; more independent qualification is necessary.

## New experimental, opt-in C++ MPC core algorithm

Flag prefer_clearance_when_cost_close, default FALSE, exposed to ROS overlay.
At an SQP R1/R2 full-horizon choice, if BOTH candidates are already
source-validated for all 30 stages, prefer >=5mm additional minimum body
clearance when the other plan has <30mm clearance reserve and the
exact objective disadvantage is <=2%. Other cases preserve the existing
exact-cost choice. No changes to hard physical clearance, optimization
feasibility or short-prefix validation. Separate existing
prefer_terminal_continuable_refinement tests two exact valid 75ms
continuations when time permits.

Modified source: mpc_rti.cpp, mpc_rti.h, mpc_controller_node_impl.hpp,
test_vehicle_model_source_rollout.cpp. Focused C++ tests and controller
build passed. New settings remain OFF for qualified production configs.

## Live regression matrix

Practice full independent qualifying runs:
 /tmp/sdu_mpc_practice_protected_gate_20261101.py
 /tmp/sdu_mpc_practice_protected_gate_20261101.log
 live_runs/mpc_collision_margin_experiment_20261030/practice_protected_gate_results_20261101.json

Competition after practice validation, no concurrent Unity instances:
 /tmp/sdu_mpc_competition_clearance_20261101.py
 /tmp/sdu_mpc_competition_clearance_20261101.log
 live_runs/mpc_collision_margin_experiment_20261030/ios_6925_selector_results_20261101.json
Use full native speed from the same verified 6.925s IPOPT export, plus
a 7.075s cross-map regression. Report real ten-lap race completion
separately from source optimizer duration and partial scored laps.

## Architecture next

A source-certified 750ms plan is not an invariant safe terminal set,
particularly with two commands already queued in the actuator model.
A 25-75ms exact-valid recovery prefix is not a certificate of a
following 750ms safe horizon. The 6.925s source witness's additional
1.85mm sampled clearance also offers practically no localization margin:
a small error can cause future-horizon infeasibility.
Investigate early geometry and actuator-delay-aware feasible control
selection, a longer certified successor/terminal safety set, and
reducing repeated fallback trajectories. Accept no speed claims unless
the new controller passes actual full ten-lap closed-loop Unity races
with both reliability and published-command timing preserved.
