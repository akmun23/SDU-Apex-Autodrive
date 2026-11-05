# Competition source failure benchmark — 2026-11-05

The recorded 143 m competition episode is reproducible offline with the same
source dynamics, competition weights, frozen objective and command limits.
No optimizer mathematics, race profile or throttle domain was changed for this
benchmark. The pre-existing terminal-speed experiment remains separate.

The bundle and commands are documented in
`f1tenth_mpc/test/data/competition_corner_20261105/README.md`.

Measured findings:

- All 22 initial source costs pass relative parity tolerance 1e-7. Published
  final primals at records 754–756 independently reproduce final cost,
  violation and minimum footprint clearance. Other final primals were omitted
  by the historical recorder; their offline reconstructions closely match the
  recorded summaries but cannot replace independent final-primal evidence.
- Two repeated native replays give identical numerical results after excluding
  timing fields. All 31 reference samples, terminal targets and complete model
  settings are archived and reused unchanged on subsequent optimizer runs.
- Last publication is record 756, at 142.841567 m. First infeasible result is
  record 757, at 142.992145 m, before the reported failure at 143.847508 m.
- The selected sequence contains 19 consecutive recorded solver epochs without
  publication. By its final source epoch the last publication header is
  430.992 ms old. The host bridge has held the same quantized input for
  425.768 ms. At the final future candidate epoch the header age is 480.992 ms.
  These are different clocks/events; none proves Unity FixedUpdate application.
- Twelve rate-limited older-epoch warnings are preserved from the whole run.
  Their largest logged backwards delta is 0.595814 ms. The early-return guard
  runs before cycle diagnostics, so this is not a count of all skipped callbacks.

Admissible existence witnesses use only throttle in [0.001, 1], the recorded
steering/command-rate limits and original source body-corner constraints.
The preferred 3 cm reserve does not become a hard collision boundary.

| Epoch | Best 0.75 s excess clearance, m | Best 1.5 s excess clearance, m |
| --- | ---: | ---: |
| Last feasible, 142.841567 m | 0.088852 | 0.088852 |
| First infeasible, 142.992145 m | 0.041486 | 0.041486 |
| Terminal failure, 143.847508 m | 0.182555 | 0.175318 |

Every listed witness's 30-stage prefix also passes the independent original
NLP evaluator. The extension uses the identical source transition and ordinary
body-corner inequality beyond stage 30. The witness slows to roughly 0.02 m/s;
it demonstrates recovery existence, not racing performance, optimality,
recursive invariance or Unity fidelity. These states do not require intervention
before their recorded command epoch to have a source-feasible continuation.
No unsuccessful finite search is treated as an infeasibility proof.

The expanded 30-stage search finds witnesses for 21 of the 22 recorded epochs.
Record 775 was initially unresolved: the finite control-family search reaches
-0.021400 m clearance; refinement with the same source NLP improves this to
-0.010141 m but still finds no feasible continuation. Its initial body
clearance is positive (+0.276800 m), so an already-invalid initial footprint
does not explain that result. This is not a reachability impossibility proof.
The original Phase 1 replay independently rechecked 24 stored feasible witnesses
(21 ordinary horizons and three longer continuations), and preserves the
unsuccessful final-epoch search as initially unresolved evidence.

Phase 2 follow-up resolves source reachability for record 775: independent
SLSQP/BFGS on the same frozen NLP finds a trajectory with +0.000085503 m
clearance, independently revalidated by the original source evaluator. It uses
only throttle >=0.001. The search required 300 iterations and 2007 oracle calls;
it did not satisfy the source KKT criterion. The bundle retains the failed
earlier searches and adds this witness, bringing regression checks to 25 and
source-recoverable recorded states to 22/22. This tiny margin is not Unity or
deadline qualification. This independent optimizer is offline research only.

Running the unchanged SQP for up to 20 iterations recovers the first infeasible
epoch on iteration 4; observed iteration work through that point is 15.95 ms on
this host. This excludes nominal preparation, publication and integrated
contention. Several later epochs remain infeasible despite the longer solve;
the terminal epoch reaches only -0.084787 m clearance. None reports KKT
convergence. More iterations alone are not an established 22 ms solution.

Benchmark defects fixed:

- Removed inherited hidden flags and model settings from unrelated fixtures.
- Applied the actual recorded curvature/braking envelope to the correct
  competition reference before freezing its objective.
- Stopped comparing initial cost against final-iterate clearance/violation.
- Added independent scoring of recorded final controls, including rejected
  finals when future recordings contain them.
- Corrected witness feasibility to use hard physical clearance and the actual
  positive-throttle/controller command bounds.
- Fixed continuation checks beyond the API's 30-stage selector by using its
  ordinary-stage margin, preserving the same source equation.
- Preserved partial-iteration warm starts when a capture ends after derivative
  work at a deadline. Added opt-in live SQP trace/frozen-POD capture and retained
  final source controls independently of publication.
- Increased diagnostic precision to 17 significant digits. Native checks show
  capture on/off gives identical unbudgeted commands and source costs at the
  last feasible, first infeasible and terminal epochs.

Phase 1 remains limited by historical evidence that was never recorded: exact
live per-iteration primals, rejected final primals and Unity apply-tick history
cannot be recovered retroactively. The bundle labels those gaps explicitly.
Fresh captures can close the primal/schedule gaps. Current tests do not qualify
Unity safety, recursive feasibility, racing performance or deadline reliability.

Phase 2 should use this fixed corpus to correct original-problem KKT accounting,
separate hard collision constraints from elastic preferred clearance in the
subproblem, and improve full-step convergence. The existing QP's convergence
and small correction must not be confused with source-NLP convergence.
