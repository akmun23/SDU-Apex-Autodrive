# Optimizer search and dashboard — 2026-10-30

## Scope and result

Improve search allocation and retain useful feasible alternatives, not merely
increase iteration counts. No vehicle equations, collision checks, required
planning clearance, MPC weights, or runtime control code were changed here.

Current competition-map source-model results:

| Choice | Periodic model lap | Minimum planned body-to-wall gap |
| --- | ---: | ---: |
| Fastest exported | 7.050 s | approximately 5.00 cm |
| Roomiest exported | 7.075 s | 7.980 cm |

The roomier line already existed in the equal-time archive. This work recovered
it into the exported portfolio and gives its shortening search priority; it is
not a newly measured Unity improvement. The additional gap costs 25 ms in the
model. Both have independent source-model certificates, not ten-lap Unity
drivability certificates. The 5 cm is nominal planning clearance, which tracking
may partly consume; this work does not add another MPC wall buffer.

Workspace: `live_runs/optimizer_iros_adaptive_regions_v6_20261030`.
Roomier witness ID: `e75d1f49f552ec98`.

## Diagnosed behavior

1. Equal-time alternatives were searched in insertion order, with each child
   exhaustively searching before the next child. The first tiny steering/geometry
   change occupied hours while a 7.5 cm RMS geometry change with nearly 3 cm extra
   clearance waited. That is allocation bias, not evidence of a global minimum.
2. `publish_best()` returned before exporting any slower candidate once a faster
   line existed. A useful speed/clearance tradeoff could disappear from the
   available immutable exports and automated driving-test inputs.
3. Parent campaign progress updated only when a child returned. A child could
   produce a faster certificate while the parent still reported its old best.
4. Nondefault campaign margins overwrote larger requested polish margins.
   Equal-time polishing also hardcoded 5 cm in several paths.
5. Dashboard labels implied all exports were speed gains; its fixed 1e-4 guide
   did not match the exact-grid 1e-5 feasibility gate. Large raw IPOPT dual
   residuals and an active infeasible trial were not explained plainly.
6. The continuous-time predictor is a 24/25 ms transition-chord initializer,
   not an exact free-time physics solve. Its latest completed 7.050 s search
   returned `no_time_proposal`: the retained feasible point stayed at the
   incumbent duration and shorter proposed seeds did not beat plain resampling
   on the existing exact-map initialization test. Solver activity there was
   not a certified lap-time improvement.

## Implemented

- `command_campaign.py`: roomiest alternative first, then a cooperative
  round-robin queue. Each visit completes at most two duration starts; a source
  repair and its independent certificate are never interrupted merely to yield.
  Solves retain their configured 1200-iteration / 300-stagnation budgets. This
  is a scheduling limit, not a reduced IPOPT iteration limit or hard wall-clock
  bound. Saved child primals resume; yielded starts are not rerun.
- Immediate durable parent-incumbent updates from certified child callbacks.
- `track_optimizer.py`: export slower certified alternatives too; maintain
  `portfolio.json` with non-dominated time/clearance/steering-smoothness/throttle-
  smoothness choices. `best.json` remains the fastest pointer. Duplicate
  publication does not rewrite immutable manifests or fake a recent gain.
- Reject a published lap time that disagrees with its independent certificate.
- `elite_refinement.py`: propagate the selected base margin through polishing
  and restoration; preserve requests for greater clearance; independently
  check that margin trials attained the extra requested gap. Keep certificate
  baseline margin separate from the trial's requested margin.
- Dashboard: plain-language current intent, uncertified-trial warning, speed
  gain age, total planned wall gap in centimetres, roomiest-versus-fastest
  comparison, queue/visit counts, separate predictor/exact-grid thresholds,
  raw dual-error explanation, and a roomiest-line map overlay. Existing MPC
  test controls and reports remain intact.

## Verification

- 98 optimizer regressions passed across campaign, elite refinement, map
  optimizer, command OCP, sector timing and adaptive regions.
- 49 dashboard/catalog/automatic-validation tests passed, including a
  slower-but-roomier portfolio summary. JavaScript passed `node --check`.
- New regressions cover fair branch rotation, yield/resume without repeated
  starts, unchanged per-solve budgets, slower alternative retention, immutable
  manifest timestamps, and certificate/time consistency.
- A child improvement is persisted before its parent callback returns, verified
  by interrupting that callback and checking the parent's saved best witness.
- Live HTTP snapshot: no warnings; 7 exported certified choices, 4 Pareto
  choices, roomiest gap 7.980007 cm, fastest gap approximately 5 cm.
- Browser rendered and inspected at 1600×1500. Screenshot is a temporary
  verification artifact: `/tmp/apex-optimizer-dashboard-20261030.png`.

## Running and next evidence

Exactly one optimizer service is used:
`sdu-iros-adaptive-regions-v6-20261030.service`. It was stopped gracefully,
checkpoints preserved, and resumed with explicit audited search-code migration.
The new queue first selected the roomier witness and started its 7.050 s
shortening attempt. A separate supervised MPC test may pause this service;
the dashboard reports that pause, and the existing owner resumes it afterward.

Dashboard: <http://127.0.0.1:8765/>.

Still to establish: whether the roomier basin shortens better, and whether its
extra planned clearance improves closed-loop Unity tracking. No new faster
Unity lap or global-optimality claim follows from these code changes. The
predictor's physics approximation and repair limitations have not been removed.
Retain both speed and quality choices while gathering that evidence; do not
equate a flat lap-time counter with absence of useful clearance progress.

## Follow-up: explain why a below-threshold checkpoint is not accepted

The live page showed a saved iteration-90 constraint residual of 7.40545e-6
against a 1e-5 gate without showing the actual replay blocker. At inspection,
the latest early source replay was iteration 120: closure passed, recursive XY
drift was 0.2221 mm (limit 50 mm), but minimum source-model body-to-wall gap was
48.9122 mm (required 50 mm). The 1.0878 mm planning-buffer shortfall failed
that replay. This is not a wall intersection and does not prove that the
different saved iteration-90 point fails; the two iterates must not be conflated.

The dashboard now presents separate model-constraint, periodic-closure,
recursive-drift, 1 ms clearance and independent-certification/archive gates.
Each has passed/failed/pending status, its measured value and threshold when
available, plus its evidence provenance. Missing results remain pending. A
completed solve report supersedes an earlier replay; an incumbent's existing
certificate is never reused as evidence for the active trial. Predictor-only
results remain explicitly uncertifiable until exact-grid repair.

Early replay state is scoped to the active solve and reset on a new solve or
log rotation, preventing a stale previous failure from being shown as current.
Only dashboard code changed in this follow-up; the active optimizer was not
restarted and none of its acceptance thresholds were relaxed.

## Follow-up: feed missing clearance samples back into IPOPT

The prior loop could replay a nearly feasible iterate, observe the same
millimetre-scale margin miss, then continue optimizing an NLP without that
within-packet wall constraint until its iteration budget ended. This was a
real gap in the optimization loop, not a reason to loosen the verifier.

Implemented in `command_ocp.py`:

1. On an eligible low-residual iterate, replay the production plant at 1 ms.
2. Collect violating samples and select the worst *new* tick per interval,
   excluding the already modeled 6/12/18/24/25 ms prefixes and existing cuts.
3. Stop the stale NLP immediately, checkpoint precisely the replayed state and
   control sequence, and save a durable `clearance_refinement.json` request.
4. Campaign/bootstrap/elite orchestration adds constraints for those literal
   production-plant prefixes and restarts from the same checked primal.
5. Repeat within the configured refinement budget; retain the independent
   final full-lap and full-perimeter certificate. No planning margin is relaxed.

This applies to clearance/smoothness objectives too, without prematurely
stopping their useful objective optimization once source checks pass. A replay
whose recursive drift exceeds 5 cm cannot propose these local cuts. Already
constrained violations are not disguised as useful duplicate constraints.
Durable requests are recovered only for matching source, map image, packet
count and required margin. The dashboard reports added source-tick constraints,
repair passes and the exact clearance-measurement provenance.

### What the check does and does not establish

- All 25 literal within-packet prefixes agree with independently advanced 1 ms
  production-plant steps to absolute/relative tolerance 1e-10 in regression
  tests at 3 and 8 m/s. This is production-model parity, not Unity native-solver
  parity.
- The CasADi wall-distance/footprint calculation agrees with the independent
  checker within 1e-12 on rotated and untranslated/translated map fixtures.
- This competition map has 2.5 cm cells. The perimeter checker subtracts a
  conservative 22.75 mm correction for unsampled points between perimeter
  probes; optimizer constraints use the same correction. A reported 1.09 mm
  shortfall is precise within that representation, not proof of a physical
  Unity wall-distance error measured to 1.09 mm.
- Neither continuous motion between 1 ms samples nor exact Unity contact
  geometry is certified by this check.

Validation: 162 targeted optimizer/dashboard/automatic-validation regressions
passed. Real IPOPT callback fixtures verify immediate early stopping, exact
checked-primal retention and pending-cut recovery, for both repair and smooth
objectives. These fixtures test orchestration, not simulated car performance.
The live optimizer was restarted with audited algorithm-only migration. A new
restart guard respects any still-live supervised-driving pause lease before
starting CPU-heavy work; PID reuse and cancellation are covered by tests.

Live confirmation: iteration 0 requested six missing ticks; the rebuilt NLP
exposed the previous 1.0887 mm replay miss as a 0.0010887 m constraint error.
After 15 iterations its replay shortfall fell to 0.1719 mm, triggering two more
tick constraints. The next pass reduced the replay miss to 0.000536 mm at
iteration 10 and approximately 0.00000013 mm at iteration 50. This is a measured
source-model feasibility gain, not yet a new certified lap or Unity result.

The GUI now increases numeric precision for near-boundary failures, rather than
showing equal rounded values such as `50.00 mm < 50.00 mm`. The summary uses
smaller units for tiny shortfalls. A replay-scheduling line shows the earliest
next eligible iteration (40 iterations after the prior replay), live-iterate
feasibility requirements, and any supervised-test pause. A regression links
the displayed cadence to the current source callback policy. The expanded
dashboard/catalog/automatic-validation suite passed 57 tests.

## Follow-up: explicit numerical clearance tolerance

Requiring an exactly nonnegative floating-point slack was needlessly rejecting
sub-nanometre boundary differences. The acceptance policy now allows a **1 µm
(0.001 mm) absolute numerical tolerance**. The OCP still targets 50 mm nominal
clearance; verification accepts a conservative measured gap of at least
49.999 mm. This is an explicit numerical acceptance band, not exact ≥50 mm and
not rounding the vehicle state, controls, dynamics or stored measurements.

The canonical value lives in `config/racing/raceline_clearance_policy.json`.
Its contents are included in the optimizer implementation digest and audited
resume migration. It is shared by model verification and lightweight readers.
Early replay, final independent certification, missing-tick refinement, elite
selection, historical-certificate indexing and the dashboard use the same
declared tolerance. Certificates preserve raw slack and record the tolerance;
old unannotated certificates remain strict. Invalid or larger-than-policy
certificate tolerances are rejected by readers.

The GUI marks `Passed (numerical tolerance)` when the nominal target is missed
only within this band and displays `tolerance 0.001 mm` beside the target. Real
shortfalls such as 0.172 mm or 1.09 mm still fail. Regression cases cover both
ends of the 1 µm boundary, unchanged raw values, non-finite input, excessive
declared tolerances, final certification and downstream catalog/GUI agreement.
173 targeted tests passed; JavaScript syntax and diff whitespace checks passed.
The background optimizer was restarted with checkpoints retained; its restart
guard continues to honor any live supervised-driving pause lease.

Separately, before this tolerance change was deployed, the preceding adaptive
constraint repair had already certified its repaired 7.050 s source-model line
under the old strict check and moved on to a 7.025 s trial. That is not evidence
of a new fastest time or successful Unity driving.

## Follow-up: explain feasible seeds and ongoing quality optimization

The dashboard previously called every active solve a new unvalidated time
trial. A clearance solve starting from an already feasible 7.050 s witness
therefore showed all source checks passed at iteration 0 while continuing to
iterate, without explaining why. No solver behavior needed changing here:
feasibility and optimality are different goals.

The dashboard now puts the current objective first: clearance, command
smoothness, time prediction or feasibility repair. Quality searches explicitly
say the lap duration is fixed and explain why feasibility does not end the
optimization. Saved certified results are separated from verification evidence.
The evidence section lists live, saved and replay iterations independently;
an iteration-0 verified seed is not presented as verification of the current
iterate or as a newly improved result. The card also says "Fixed lap time ·
quality search", rather than suggesting another shorter-time attempt.

Unscaled callback objective values are retained for start/latest/saved
comparisons, alongside the latest callback's constraint residual. They are
explicitly labeled combined solver costs, not wall distances or certified
gains. Raw IPOPT objective values (which can be scaled) are not mixed into this
comparison. Missing headers/objectives remain unknown; new-solve boundaries
discard the prior goal and evidence. The current strategy is also preserved
when the active attempt is a polishing job rather than a duration trial.

Validation: 64 dashboard/catalog/automatic-validation tests passed, including
the exact iteration-0-pass / live-iteration-158 scenario with an infeasible
lower-cost live candidate, objective reset, different objectives, mismatched
saved/replay points and completed-report provenance. JavaScript syntax and
diff whitespace checks passed. Only the dashboard service needs restarting;
the background optimizer and its checkpoints are not modified or restarted.
Deployed by restarting only the dashboard. A headless-browser render confirmed
the live clearance objective, fixed 7.050 s duration, iteration-229 vs saved/replay
iteration-0 distinction and "no improved feasible checkpoint" message. The
existing optimizer PID (79387) was retained; its supervised-simulator pause was
left intact.

## Follow-up: identify refinement/restoration solves correctly

The next adaptive corner attempt exposed an actual identity bug. Its manifest
moved from `polish_02` to `polish_02_cut_01`, but that direct solver call emitted
no outer `solve` event. The observer retained the old output path, rejected the
new header as unassociated, and displayed an empty solver/iteration despite
live IPOPT output. Proposal restoration had the same missing-event risk.

Production-command and time-predictor headers now carry their own absolute
output paths. A header updates identity and clears previous replay/iteration
data independently of campaign events. This also works after the observer
starts from a bounded log tail without the original campaign event.

For the worker already running before these self-identifying headers, the
dashboard can prove a refinement handoff using the same polishing family,
source digest, lap duration, margin and exact parent-reported union of new
and existing wall cuts. Proposal restoration requires its explicitly named
quality proposal. Equal duration alone is never enough. Unproven associations
remain hidden with a visible warning rather than silently looking like setup.
The objective description now names adaptive corner repair or proposal
restoration instead of treating every repair as an unspecified time trial.

Validation: 125 optimizer/dashboard/catalog/automatic-validation tests passed.
Coverage includes self-identifying headers without campaign events, old replay
reset, in-flight refinement recovery, named proposal handoff, and rejection of
wrong source/duration/margin/cuts/branch. Real tiny IPOPT callback fixtures also
check the emitted header paths. Replaying the actual recorded missing-event
handoff recovered the repair identity and iteration 725. A live headless browser
then confirmed the subsequent smoothness solve and its iteration were visible.
Only the dashboard service was restarted; optimizer PID 79387 was unchanged.
