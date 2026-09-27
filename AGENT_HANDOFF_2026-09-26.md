# SDU Apex performance-engineering reference attachment

Updated: 2026-09-26

Workspace: `/home/akselmo/Documents/GitHub/SDU-Apex-Autodrive`

This file is a **project-state and runbook reference** for the next coding agent.
It is not a standalone prompt and does not override `AGENTS.md` or the user's
current instructions.

The purpose of this handoff is to let the next agent **continue from the
existing evidence instead of rediscovering it**.

---

# 1. Mission

Build a repeatable performance-development pipeline that can move the practice
car from the current verified ~6.0 s/lap range toward the user-reported
4.94 s/lap benchmark while preserving zero-collision robustness.

The main development areas are:

1. raceline / speed-profile generation;
2. vehicle, actuator, odometry, and MPC model fidelity;
3. MPC weight and controller optimization;
4. whole-system closed-loop lap-time optimization.

The agent should make forward progress. Investigation is useful only when it
answers a concrete engineering question needed for the next implementation
decision.

---

# 2. Root-agent operating mode

Follow `AGENTS.md`.

The root GPT-6 Sol agent owns:

- architecture;
- prioritization;
- acceptance/rejection decisions;
- integration;
- final review.

Use Luna explorers for bounded read-only investigation, Luna workers for
bounded implementation, and Luna testers for validation.

Worker completion is not acceptance.

The root must review the actual diff and test evidence before accepting a
change.

---

# 3. Progress policy

## 3.1 Existing evidence is authoritative unless inputs changed

Do not automatically reproduce every prior result.

Use existing bags, reports, manifests, and recorded results as valid project
history unless one of the following changed materially:

- simulator image/build;
- controller code;
- planner/raceline;
- vehicle model;
- actuator behavior;
- localization/odometry implementation;
- relevant runtime configuration;
- host/network condition being tested.

If none of those changed, do not rerun an unchanged baseline merely for more
confidence.

## 3.2 Every work package must end

Each work package must end as one of:

- **IMPLEMENTED**
- **REJECTED**
- **BLOCKED**

Do not leave a work package indefinitely in `INVESTIGATING`.

## 3.3 Investigation budget

For one work package:

- up to 3 parallel explorer tasks for the initial investigation;
- at most 1 follow-up investigation round if a concrete contradiction remains;
- at most 1 additional live experiment if existing data cannot answer the
  defined engineering question.

A second repeat run is justified when confirming a new behavioral change.
A third equivalent run requires a specific reason.

## 3.4 Define acceptance before testing

Before running a new experiment, state:

1. engineering question;
2. hypothesis;
3. expected measurable effect;
4. safety bound;
5. acceptance/rejection criterion.

After the run, apply that criterion.

Do not invent stricter criteria after seeing the result.

## 3.5 Bias toward reversible implementation

If:

- the responsible subsystem is known;
- the intended change is bounded;
- rollback is easy;
- acceptance criteria are defined;

implement and test it.

Do not launch another broad investigation merely because more data could be
collected.

---

# 4. Hard constraints

These remain strict.

- Do not modify simulator physics, vehicle setup, or track behavior to make a
  controller/model appear faster.
- Improve planning, estimation, modeling, and control against the existing
  simulated car.
- A collision is an immediate abort. Use only pre-impact samples for clean
  model conclusions.
- Use the repository simulator launch path in batchmode with Xvfb.
- Do not use `-no-graphics`.
- Competition runtime nodes must not subscribe to, remap, proxy, or consume
  restricted simulator-truth inputs.
- Development mapping, debug bags, and offline analysis may use restricted
  truth for identification and validation, but that data must remain outside
  the compliant runtime.
- Keep the competition image free of recording/debug dependencies.
- Practice map and raceline must remain a matched pair.
- Do not change multiple coupled high-risk model/tuning assumptions in one
  experiment.
- Protect the dirty worktree. Never reset, checkout, clean, delete, or
  overwrite unrelated user work.
- Keep storage bounded. Avoid unnecessary image pulls, unbounded Unity logs,
  and large duplicate recordings.

---

# 5. Development run convention

The standard qualification-style development run is:

```text
1 warmup lap
10 scored laps
1 extra lap
= 12 completed laps
```

with:

```text
zero collision-count increase
saved debug bag
```

Use the full 12-lap run for:

- final acceptance of a meaningful controller/raceline change;
- comparison of elite candidates;
- confirmation of record-level robustness.

Do **not** require a full 12-lap run for every early screening step.

Use a staged live funnel:

```text
offline feasibility
→ first-corner / short-screen validation where appropriate
→ 1 lap
→ 3 laps
→ 12 laps for promoted candidates
```

A candidate that collides is rejected immediately.

---

# 6. Current practice runbook

## 6.1 Controller stack

```bash
source tools/docker_env.sh

AUTODRIVE_REBUILD=0 \
SDU_APEX_IMAGE=sdu-apex-autodrive:practice-dev \
SDU_APEX_MAP_YAML="$PWD/f1tenth_planning/maps/autodrive_practice_20260924_b.yaml" \
SDU_APEX_TRAJECTORY_FILE="$PWD/f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv" \
SDU_APEX_MPC_PUBLISH_DIAGNOSTICS=true \
SDU_APEX_BUILD_MPC=1 \
./tools/start_dev.sh
```

Run:

```bash
python3 tools/verify_runtime_topic_policy.py
```

before relying on a competition-style runtime graph.

## 6.2 Recorder

```bash
SDU_APEX_IMAGE=sdu-apex-autodrive:practice-dev \
SDU_APEX_RUN_ID=descriptive_unique_run_id \
./tools/record_debug_bag.sh
```

## 6.3 Watcher

```bash
./tools/watch_sim_run.sh \
  sdu_apex_autodrive_dev \
  sdu_apex_sim_practice \
  rec_descriptive_unique_run_id \
  12 \
  360
```

## 6.4 Simulator

```bash
SDU_APEX_SIM_TRACK=practice \
SDU_APEX_SIM_MODE=batchmode \
./tools/start_simulator.sh
```

Start order:

```text
controller
→ recorder
→ watcher
→ simulator
```

The watcher aborts on collision.

Keep run IDs/container names consistent.

---

# 7. Current map / trajectory pair

Practice map:

```text
f1tenth_planning/maps/autodrive_practice_20260924_b.yaml
```

Current practice trajectory family:

```text
f1tenth_planning/trajectories/autodrive_practice_20260924_b/
```

Primary trajectory currently referenced by the runbook:

```text
f1tenth_planning/trajectories/autodrive_practice_20260924_b/
autodrive_practice_20260924_b_mintime_raceline.csv
```

Do not substitute an older-course trajectory.

---

# 8. Established project evidence

Treat the following as project history.

Do not rerun them simply to rediscover the same conclusion.

## 8.1 Safe baseline

Bag:

```text
live_runs/practice_ntumethod_eval_12lap_20260926/run/run_0.db3
```

Result:

```text
12 laps
0 collisions

10 scored-lap mean: ~6.0212 s
best:               ~6.0027 s
worst:              ~6.0407 s

LiDAR:               ~39.93 Hz
AMCL position error:
  p50                 ~5.8 cm
  p95                 ~17.5 cm
```

This establishes a safe ~6.0 s baseline.

It does not establish a sub-5 capability.

## 8.2 Faster yaw-response trial

Bag:

```text
live_runs/practice_yaw_tau015_12lap_20260926/run/run_0.db3
```

Result:

```text
mean: ~6.1004 s
0 collisions
CTE p95: ~9.3 cm
```

Conclusion:

```text
REJECTED as an overall performance improvement.
```

It improved one tracking metric but made lap time slower.

Do not repeat this exact hypothesis unless another model change alters its
meaning.

## 8.3 Calibrated moderate-steering yaw trial

Bag:

```text
live_runs/practice_yaw_model_calibrated_12lap_20260926/run/run_0.db3
```

Result:

```text
mean: ~6.0606 s
0 collisions
CTE p95: ~7.1 cm
```

The tested safe laps only exercised approximately:

```text
|steering| <= 0.281 rad
```

Conclusion:

- moderate-steering fit has useful evidence;
- the combined change was still slower than the safe baseline;
- it does not validate high-steering behavior.

Do not treat it as the final yaw model.

## 8.4 Failed fast-line candidate

Bag:

```text
live_runs/practice_yawcal_line_12lap_20260926/run/run_0.db3
```

Result:

```text
collision before one completed lap
```

Pre-impact evidence:

- steering reached approximately 0.42 rad;
- yaw authority fell materially relative to moderate-steering behavior;
- timing cadence was normal enough that transport was not the primary
  explanation for this specific crash;
- true CTE grew strongly while requested turn rate was not achieved;
- target speed was already rising through corner exit.

Conclusion:

```text
REJECTED trajectory/model combination.
```

Do not promote or retest the same candidate unchanged.

The useful engineering conclusion is:

> the current optimizer/controller model is too optimistic in the
> high-steering / high-demand region.

## 8.5 Failed optimizer prediction

Candidate:

```text
live_runs/raceline_candidates/practice_yawcal_mintime_20260926/
practice_yawcal_mintime_0p110_wallchecked.csv
```

Optimizer prediction:

```text
~5.203 s
length:         ~27.85 m
max speed:      ~8.565 m/s
max curvature:  ~1.272 1/m
steering demand ~0.405 rad
wall clearance: ~0.15 m
```

Live result:

```text
collision before one valid lap
```

Conclusion:

```text
The current nominal optimizer result is not a valid closed-loop lap-time
prediction in the high-demand regime.
```

Do not call nominal optimizer time achievable until a closed-loop model
validates it.

---

# 9. High-steering model status

Known moderate-steering behavior:

```text
steady yaw gain r/(u*tan(delta)) ≈ 2.9
```

around:

```text
|delta| ≈ 0.25–0.30 rad
```

The failed high-demand run showed strong authority loss near:

```text
|delta| ≈ 0.35–0.42 rad
```

That is a real modeling gap.

However:

- the original failed-line samples are correlated;
- that run alone is not a sufficient high-angle model fit.

The correct next step is **not repeated broad model investigation**.

Use controlled open-plane data and/or already-collected open-plane sweeps to
fit a model that is explicitly validated on a separate recording.

Once one model generalizes better than the current model, implement it and
move on.

If a simple angle-only taper repeatedly fails held-out validation, mark that
model structure `REJECTED` and test a different structure rather than fitting
the same equation repeatedly.

---

# 10. Open-plane use

The wall-free open-plane simulator is authorized for development excitation.

Use it for:

- steering response;
- acceleration;
- braking;
- wheel-slip;
- combined steering/throttle experiments.

Before transferring fitted parameters to the practice controller, validate the
result on independent practice data.

Do not require repeated image-parity audits once parity has already been
established for the relevant vehicle/actuator properties and no simulator
image changed.

Record the image digest in the run ledger.

---

# 11. Target model-development direction

Prefer a compact grey-box model:

```text
known simulator geometry/physics
+
identified effective actuator/vehicle behavior
```

rather than either:

```text
fully generic Pacejka model
```

or:

```text
unstructured black-box fit
```

The important predictions are:

- steering response;
- yaw response;
- longitudinal acceleration;
- braking;
- wheel slip / body-speed mismatch;
- actuator delay;
- relevant sensor/transport timing.

Evaluate prediction over the MPC horizon, not only one sample.

Suggested horizon checks:

```text
25 ms
125 ms
250 ms
500 ms
750 ms
```

A simpler model with better held-out rollout accuracy is preferred to a more
complicated model.

---

# 12. Offline evaluator

The long-term tuner must use a **closed-loop** evaluator.

It should combine:

```text
production MPC/controller
→ independent empirical plant
→ actuator/timing model
→ state/odometry approximation
→ controller again
```

Do not rank MPC weights using only the MPC's own internal objective.

Do not use a frozen-state replay as the final lap-time tuner.

Frozen replay remains useful for:

- solver checks;
- numerical screening;
- identifying obviously bad candidates.

Closed-loop candidate ranking must allow different weights to generate
different future trajectories.

---

# 13. Planner direction

The planner should ultimately optimize the fastest **trackable** trajectory.

Required physical checks should include:

- exact vehicle footprint;
- wall clearance;
- acceleration envelope versus speed;
- braking envelope versus speed;
- reachable curvature/yaw authority;
- steering-rate feasibility;
- actuator delay;
- wheel-slip-limited acceleration;
- controller trackability.

Do not rely on nominal optimizer lap time alone.

The planner may continue generating candidate geometry with the existing TUM
tooling, but promoted candidates should be scored by the validated closed-loop
model.

---

# 14. MPC-weight optimization

Do not start by manually changing one weight after another.

Once the closed-loop evaluator is credible:

1. group related weights;
2. sample a broad range cheaply offline;
3. reject unsafe/unstable candidates;
4. use Bayesian optimization or another model-based search for refinement;
5. test only elite candidates live.

Initial grouped dimensions may include:

```text
lateral tracking
heading tracking
speed tracking
overspeed penalty
yaw/lateral state tracking
steering-command tracking
steering smoothness
longitudinal smoothness
terminal weighting
```

The external objective is robust race time, not the MPC QP cost.

---

# 15. Acceptance rules

## 15.1 Model changes

A model change is accepted when it:

- improves held-out prediction in the operating region it targets;
- does not materially degrade previously validated regions;
- improves or preserves closed-loop performance;
- has targeted tests.

Do not require perfect global identification before using a locally better,
well-bounded model.

## 15.2 Raceline / speed-profile changes

Use a staged acceptance process.

Early screening:

```text
physical feasibility
closed-loop offline feasibility
short live screen
```

Promoted candidate:

```text
1–3 clean laps
```

Final performance acceptance:

```text
12 completed laps
0 collisions
saved bag
paired statistics
```

A reproducible, safe improvement should be accepted as the new development
baseline even if the improvement is modest.

Do not demand publication-grade statistical certainty for every incremental
engineering gain.

## 15.3 Rejected changes

Record a rejected hypothesis and move on.

Do not repeat it without:

- new evidence;
- a different model structure;
- or changed system assumptions.

---

# 16. Run ledger

Maintain one concise project ledger, preferably:

```text
docs/development/ENGINEERING_STATE.md
```

For every meaningful run/change record:

```text
status
git SHA
image digest
trajectory/config hashes
run command
bag path
lap times
collision status
sensor cadence
MPC fallbacks/rejections
main conclusion
next action
```

Use statuses:

```text
ACCEPTED
REJECTED
BLOCKED
ACTIVE
```

This file is the first place the next agent should look before scheduling a
new experiment.

---

# 17. Current dirty worktree — preserve, then classify

The handoff snapshot observed edits in:

```text
f1tenth_mpc/config/mpc_competition.yaml
f1tenth_mpc/include/mpc_types.h
f1tenth_planning/config/autodrive_sim_vehicle.yaml
f1tenth_planning/global_racetrajectory_optimization/params/racecar.ini
f1tenth_planning/global_racetrajectory_optimization/opt_mintime_traj/src/opt_mintime.py
f1tenth_planning/scripts/optimize_trajectory.py
f1tenth_planning/trajectories/autodrive_practice_20260924_b/...
tools/start_dev.sh
SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/.../model.py
```

There are also untracked trajectory/optimizer artifacts.

Do not reset or clean them.

At task start:

```text
git status --short
git diff --stat
```

Then classify each relevant edit:

```text
accepted current work
candidate under evaluation
historical experiment
unrelated user work
```

Do this once.

Do not repeatedly rediscover the same worktree state unless it changes.

---

# 18. Current recommended work sequence

The exact next task may change with the current worktree, but the default
dependency order is:

## WP1 — engineering state / run ledger

Create or update `docs/development/ENGINEERING_STATE.md`.

Consolidate:

- safe baseline;
- rejected yaw trials;
- failed fast-line evidence;
- open-plane results;
- accepted/rejected speed-profile changes;
- current active hypothesis.

This is a small task and should be completed quickly.

## WP2 — close the high-steering model gap

Use existing controlled data first.

If additional data is needed, run one bounded experiment with predefined
acceptance criteria.

Fit candidate model structures.

Validate on held-out open-plane data and at least one practice recording.

End as:

```text
IMPLEMENTED
or
REJECTED
```

Do not keep refitting the same failing structure.

## WP3 — unify vehicle/actuator limits used by planner and controller

Reconcile:

```text
geometry
steering limits
steering rate
acceleration vs speed
braking vs speed
slip constraints
```

Create one explicit vehicle contract.

## WP4 — planner candidate generation

Generate faster geometry/speed candidates with the unified limits.

Reject physically infeasible candidates before live testing.

## WP5 — closed-loop offline evaluator

Use production controller code against an independent plant.

Validate against held-out real runs.

## WP6 — MPC weight optimization

Use grouped offline search and model-based optimization.

## WP7 — integrated system optimization

Jointly optimize selected:

```text
raceline
speed-profile margins
MPC groups
slip/actuator limits
```

Only after the earlier pieces are trustworthy.

---

# 19. What not to do next

Do not:

- rerun the same safe baseline without a changed comparison condition;
- repeat the same rejected yaw-taper fit;
- launch another repository-wide investigation of already-understood
  subsystems;
- call a nominal 5.2 s optimizer output a realistic race prediction;
- change MPC weights before a trustworthy closed-loop evaluator exists;
- change several physical/model assumptions together;
- loosen collision/corridor safety to manufacture a faster result;
- treat one failed high-speed lap as enough to fit an entire global vehicle
  model;
- wait for perfect certainty before making a reversible, bounded engineering
  improvement.

---

# 20. Source / artifact locations

Core orchestration:

```text
AGENTS.md
.codex/config.toml
```

Run tools:

```text
tools/start_dev.sh
tools/start_simulator.sh
tools/record_debug_bag.sh
tools/watch_sim_run.sh
tools/verify_runtime_topic_policy.py
```

Safe baseline:

```text
live_runs/practice_ntumethod_eval_12lap_20260926/
```

Yaw experiments:

```text
live_runs/practice_yaw_tau015_12lap_20260926/
live_runs/practice_yaw_model_calibrated_12lap_20260926/
```

Failed fast-line run:

```text
live_runs/practice_yawcal_line_12lap_20260926/
```

Failed fast optimizer candidate:

```text
live_runs/raceline_candidates/practice_yawcal_mintime_20260926/
```

Practice map:

```text
f1tenth_planning/maps/autodrive_practice_20260924_b.yaml
```

Practice trajectory family:

```text
f1tenth_planning/trajectories/autodrive_practice_20260924_b/
```

---

# 21. Handoff principle

This handoff is a snapshot of project evidence.

At task start:

1. read `AGENTS.md`;
2. read `ENGINEERING_STATE.md` if present;
3. inspect current `git status`;
4. confirm the exact active map/trajectory/image;
5. continue from the next unfinished work package.

Do **not** re-prove every historical result.

Recheck a prior conclusion only when a changed input makes it invalid.

The goal is disciplined forward progress:

```text
measure enough
→ decide
→ implement
→ validate
→ record
→ move on
```