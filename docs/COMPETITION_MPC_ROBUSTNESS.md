# Competition MPC: observed failure and required controller improvements

Verified on November 1, 2026. Treat practice/competition weight optimization
as independent experiments. The same source-plant and safety-critical C++
controller can be shared ONLY after passing practice regression testing.

## Current production profiles

Practice: f1tenth_mpc/config/mpc_practice.yaml (locked and five-race-qualified).
Competition: f1tenth_mpc/config/mpc_competition.yaml (not qualified).
Both are standalone and selected exclusively by map identity.
The simplest entrypoint is tools/run_mpc_trial.sh show / practice / competition.

Fastest source-validated competition IPOPT witness: 6.925 s, file
f1tenth_planning/trajectories/autodrive_iros_2026_compete/ipopt_best_6p925_source_witness.csv
SHA256: 59e39773f4b44b7768cf7e9130258f812ebe1984ab6b094462932dc8fe64b692.
Sampled extra corridor reserve above the mandatory 5 cm hard margin: 1.85mm.
It is NOT a closed-loop-validated 6.925s physical lap.

## Latest exact Unity baseline / forensic evidence

Run: live_runs/competition_canonical_profile_20261101_02
The canonical track-selection launcher loaded the new, separately
configured competition YAML, with no MPC overlay. This run completed no
lap and stopped source-safely with rejected_nonlinear_rollout corridor@0
near 66.018m. No wall collision and no >=20ms publication deadline miss;
max measured callback 18.979ms. The failed prior run suffixed _01
was a launcher argument error (optional empty overlay), fixed and
retested; do not count it as a controller race.

Near station 64.468m the full-horizon extra predicted corridor clearance
fell to 15mm. From 64.605m through 65.850m, successful individual cycles
frequently had just 1-3mm additional clearance. The vehicle was at about
6.55m/s, heading error +0.103rad, when a retained plan was accepted at
65.850m. The next 66.018m cycle failed exact corridor@0; no action
could be newly published to prevent the first imminent invalid
transition. This cycle had e_y=-0.036m, heading error +0.106rad,
u=6.726m/s, reference u=7.134m/s. Summaries and per-cycle evidence:
mpc_run_report.json, mpc_cycle_detail.csv and
mpc_cycle_detail_diagnostics.jsonl in the run folder.

The episode used 38 accepted retained-plan cycles and 34 candidate-prefix
cycles, but never regained a robust full source-valid 750ms sequence.
The active competition base deliberately has viability_recovery_enabled
false. Experiments with it true, geometric candidates, feedback rollout
and experimental R1/R2 clearance/terminal selectors also failed complete
races; do NOT assume simply toggling it restores robustness.

## Not only weights

Historically tested competition configurations vary enormously in cost
scales (e.g. lateral 250 vs 5000, heading 30 vs 1700,
steering slew 1 vs 2000), yet all failed to qualify on the current
full-speed native IPOPT line. The binding issue is the controller's loss
of recursively feasible full-horizon continuation near a corner and
its use of individually source-valid 25-75ms stopgaps without a
certified next long horizon. The supplied source witness has only
1.85mm of theoretical spare clearance at its worst point, so any
localization, delay, tire-model or geometry error has very little
room. Weight tuning can improve the approach, but it cannot create a
feasible continuation after the physical first transition is already
unavoidable.

## Engineering priorities and validation gates

1. Improve steering and braking anticipation from native source
   witness control and curvature ahead. Include the actual actuator
   queue and command publication lead. Make reference commands
   dynamically reachable before the high-curvature transition.
2. Enforce/control terminal viability: prefer states with an exact
   validated full-horizon successor or certifiable recovery tail, not
   only a current horizon with near-zero margin. If the 20ms budget
   cannot contain a full extra SQP, use a bounded safety-verified
   continuation over a coarse preview grid or a maintained feasible
   safe plan. Never silently relax 5cm hard body clearance.
3. Replace repeated prefix/reused-plan chains with a plan whose
   full continuation is source validated from the actual measured
   state. Trigger recovery BEFORE the end-to-end 750ms margin
   collapses; track a signed future hazard and command-delay horizon.
4. Optimize competition cost weights and trust regions independently
   after baseline feasibility improves. Preserve practice's 4500/1450
   weights and existing 0.75 feedback rollout; do not bleed them
   into the competition map.
5. Every controller implementation is shared and must pass current
   practice: 10 scored laps, no collisions/rejected publications,
   no hard 20ms misses, competitive time near the qualified five-run
   49.162s mean. Promote only after five independent clean practice
   races. Every claimed competition gain likewise requires complete
   scored laps and repeatability; a 6.925s source or 1 partial lap
   is not validation.

The 2026-11-01 experiment that selected R1/R2 by 5mm extra wall
clearance for up to 2% objective penalty did not help either map.
Its code/ROS flag and old tuning utility were removed instead of
leaving parallel unqualified production paths. Raw independent race
experiments remain preserved in live_runs.

## 1 November: IPOPT witness rescue and reduced-speed full race

Added opt-in viability_source_witness_recovery_enabled (default false).
It samples actual held steering and throttle from the verified source
witness, applies bounded lateral/heading feedback and drive reduction,
and runs complete exact 750ms source-plant/corridor validation before
ever publishing a candidate. The reserve-aware ranking applies only
to this opt-in branch; the qualified practice scoring remains unchanged.

Experiments on the SAME latest 6.925s IPOPT track geometry and unchanged
mandatory 50mm physical car-to-wall margin:
- comp_source_witness_6925_20261101_01: warmup only; later rejected.
- comp_witness_reserve_6925_20261101_02: warmup + one scored lap in
  7.6606s, then corridor@0 at a later hairpin, no deadline misses.
- comp_witness_interior_6925_20261101_03: 40mm soft target offset
  was less reliable and failed early; not promoted.
- comp_witness_latacc8_6925_20261101_04: warmup + one 8.3197s scored
  lap; future-horizon corridor failure, no deadline misses.
- comp_witness_earlyhazard_6925_20261101_05: warmup + one scored
  lap, later rejected.
- comp_witness_tail300_6925_20261101_06: experimental 300ms terminal
  probe failed before warmup, was REMOVED from active C++ source;
  original 75ms terminal probe remains.
- comp_witness_reducedspeed_6925_20261101_07: maximum 6.5m/s, lateral
  acceleration envelope 6m/s2, five scored laps 8.951/8.927/9.060/
  9.042/8.913s, then rejected corridor@0 at only ~2.2m/s.
- comp_witness_5p0_margin010_20261101_09: maximum 5.0m/s, lateral
  acceleration envelope 4m/s2, proactive threshold 100mm extra above
  the unchanged 50mm hard margin; ALL TEN SCORED LAPS FINISHED in
  112.2661s, no collision, abort or hard deadline misses; max
  19.226ms callback, but 4157 full-horizon recovery cycles. This
  proves baseline closed-loop physical feasibility on the fast IPOPT
  geometry only when driving far more conservatively; it does NOT
  qualify as a competitive/full-speed outcome.
- Further speed qualification overlays with 5.5/6m/s speed caps and
  staged acceleration caps are retained with individual run results
  under live_runs/mpc_competition_recovery_20261101.

The source 6.925s time is OFFLINE PhysX, not a driven 6.925s lap.
Source sampled extra clearance at its tightest point: only 1.852mm
beyond mandatory 50mm. Even the full ten-lap 5m/s closed-loop run
had 52mm lateral RMS error and ~2.4mm minimum predicted spare
physical corridor. Near-zero track clearance and real state-estimation
error are distinct constraints, not resolvable by lowering speed
or QP cost weights alone. The remaining engineering challenge is
maintaining recursive feasibility through future corner transitions
with the actual actuator queue while tracking a source line requiring
near-millimetre accuracy, or using a safer dynamically optimized
reference. IPOPT was stopped at the user's request.

## Shared-controller promotion policy (November 1 follow-up)

The MPC algorithm is shared. There must be no permanent map-specific
branch in the solver, model, state estimator, physical validator or
recovery logic. Tracks retain independently tuned weights and references.

Temporary opt-in algorithm flags are acceptable only during evaluation:
- Demonstrate a measurable physical benefit on the failure being addressed:
  full-horizon convergence, actual clearance, tracking or race completion.
- Run unchanged practice regression, then qualify the contender with
  five independent complete ten-scored-lap races: no collision/fatal fault,
  no >=20ms publication deadline miss, competitive 49.1622 s mean.
- Once verified on both maps, promote the change to a data-driven shared
  MPC implementation. Remove the experimental toggle and replaced logic.
  Source-witness presence, native physics and forecast state may determine
  behavior; the track name must not.
- Remove unsuccessful variants rather than leaving dormant code paths.

The experimental input-slew change that aligned both steering and throttle
slew penalties with native source-witness command increments was rejected
after the actual full-native-speed IPOPT run failed before one scored lap
and a 6m/s comparison also failed early. The flag and unproven algorithm
were removed; source experiment YAML/logs were retained in live_runs.

Shared-controller practice regression: 10/10 scored laps, 49.158 s,
zero collision/fatal MPC rejection, zero publication misses, callback
maximum 18.062ms. One run does not replace five-run qualification.
