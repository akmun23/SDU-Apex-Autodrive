# First-corner failure of the 4000/1450 practice MPC

31 October 2026. The 4000/1450 configuration, source-validated feedback
gain 0.75, is NOT five-race qualified. A full ten-scored-lap Unity run
finished in 49.071s with no deadline miss. The next independent run was
aborted before any scored lap by rejected_nonlinear_rollout, corridor@0.
Its worst callback was 17.355ms, so the cause was physical feasibility,
not a CPU deadline miss. Retain the 4500/1450 five-race-qualified rollback.

## Read-only recorded-state reconstruction

At 26.894m the failed run had e_y=-0.05775m, heading error=+0.03813rad,
speed=3.38m/s, yaw=-2.00rad/s, steering=-0.2223rad. At a comparable
26.87m in the successful 4000 run, e_y=+0.018m, heading error=+0.011rad,
speed=3.59m/s and steering=-0.203rad. The vehicle was already on
a materially different trajectory before recovery; do not claim that
the weight setting alone determines this difference.

The full 750ms source-validated minimum corridor slack collapsed:
26.403m: 20.28mm; 26.509m: 17.67mm; 26.612m: 15.10mm;
26.710m: 11.11mm; 26.804m: 6.89mm; 26.894m: 0.93mm.
All were accepted SQP cycles. The physical hard corridor was retained,
but there was virtually no additional forward clearance by 26.894m.

At 26.981m, the full horizon failed corridor@3 with primal residual
~0.0559 and 302 combined iterations across the two passes. The
source-valid viability recovery immediately released steering from
-0.2223 to -0.06375rad (+0.15855rad discontinuity). Next at 27.064m
the full horizon failed corridor@2 and a source-valid 2-step prefix
was published (-0.14155rad). At 27.142m the full horizon failed
corridor@1 and a source-valid 1-step prefix was published (-0.27105rad).
At 27.216m the full horizon failed corridor@0; with two prior short
actions already used, the controller published no command. Heading error
increased 0.0449 -> 0.0739 -> 0.1067 -> 0.1415rad; yaw momentarily
became much less negative. The two in-flight actuator commands mean the
new control cannot save a first-stage imminent corridor violation.

## Identified policy contributor

The practice overlay enables 20mm viability lookahead reserve but also
sets viability_proactive_after_fallback_only: true. The
try_publish_viability_recovery() implementation in
f1tenth_mpc/src/mpc_controller_recovery.cpp returns immediately for an
accepted SQP result when no short fallback occurred in the episode.
Therefore the 20mm reserve never triggers the viable proactive search
on the failed drive's 26.509--26.894m approach; the first search
happens after the full-horizon optimizer has already been rejected.

A one-parameter experimental overlay was prepared:
live_runs/mpc_collision_margin_experiment_20261030/fixed_heading1450_lateral4000_proactive_reserve_trial_20261031.yaml

It sets viability_proactive_after_fallback_only: false; all weights,
native rollout, hard map/corridor/footprint and budget checks remain.
That changes the eligibility of five accepted states on the failing
approach, first at 26.509m (~0.47m before the initial rejection).
This is NOT proof a different physical candidate exists or will be
selected without exceeding the time budget. On a successful complete
4000 race, 17/2208 accepted states had <20mm reserve; new candidate
testing might increase callback cost or change behavior. In a
successful 4500 race only 10 such states occurred.

Serialized follow-up policy trial script:
 /tmp/sdu_mpc_lateral4000_proactive_qualify_20261031.py
It waits for the active 17-case bidirectional sweep to stop owning
Unity, then performs independent complete races. Stop on first
rejection, collision, deadline miss, or incomplete run; never
promote the variant without five clean races.

## Further work if enabling proactive reserve fails

1. Feed the actual two-slot actuator queue into a cheap
   first-transition viability guard BEFORE exact recovery candidates.
2. Compare safe R1/R2 full-horizon terminal continuations using the
   existing source model when enough deadline margin remains.
3. Evaluate the recovery bank's abrupt +0.15855rad steering release
   by imminent yaw/heading dynamics, not just terminal cost and slack.
4. Avoid chaining short valid prefixes into a physically dead horizon;
   prefix validation is not full-horizon certification.
5. Keep existing qualified 4500/1450 as rollback; use repeatability,
   20ms publication, native safety and tracking, not fastest single
   lap, as gates to a better configuration.

Evidence: live_runs/practice_fixed_heading1450_lateral4000_20261031_01
and _02, their mpc_cycle_detail.csv, mpc_cycle_detail_diagnostics.jsonl
and mpc_run_report.json. Machine-readable threshold audit is
lateral4000_proactive_corner_counterfactual_20261031.json in the
mpc_collision_margin_experiment_20261030 directory.
