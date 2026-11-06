# Convergence, useful partial gains and readable stopping — 2026-10-31

## Measured findings

- Fastest certified source-model duration remains 6.925 s. The existing 7.050 s
  alternative has 84.4069 mm minimum conservative clearance.
- On the fastest original witness, 22.686% of model time has less than 60 mm
  body-to-wall clearance. Tight points span several sectors, not one anomaly.
  The lowest points include 5.886 s at (0.678, 0.896), 6.645 s at
  (-0.487, -1.751), 4.763 s at (3.576, -2.931), and 4.275 s at
  (2.217, -5.881). These are source-model positions, not new Unity telemetry.
- A failed 52 mm request had actually produced **51.701397 mm** at the same
  6.925 s. Independent source certification at the unchanged nominal 50 mm
  policy passed: local dynamics defect 5.889659e-6, periodic defect 0,
  maximum recursive XY mismatch 0.410345 mm. The old all-or-nothing search
  policy discarded this useful result because it did not meet the stronger
  52 mm search goal.
- Report: `live_runs/source_partial_clearance_certification_20261031/report.json`.
  This establishes a better nominal source witness, not Unity drivability or
  a proof of the globally fastest/roomiest line.

## Changes

### Normal solver stops, not manufactured exceptions

`optimizer_solver.py` installs a native CasADi solver callback that returns a
stop flag. Opti's ordinary Python callback returns void, which is why the old
implementation threw exceptions and printed tracebacks during normal search.

Stop reasons are explicit: feasibility/objective/time stagnation, requested
source checkpoint accepted, new wall constraints, or interruption signal.
The standard IPOPT footer is translated **only when our callback has explicitly
requested a known stop**. Unexpected callback exceptions and unexplained stops
remain visible. The native `User_Requested_Stop` code is retained in reports
alongside the actual controlled reason and callback iteration. Expected stops
have `error: null`; they do not imply convergence or certificate acceptance.

The reader converts scaled solver coordinates back to physical variables,
including offsets, and caches graph readers for multiple Opti variables. In a
real 277-command problem, the old and new callback paths produced exactly the
same first 11 primal iterates (maximum scaled difference 0). A full vehicle
solve accepted the independently verified partial witness at callback iteration
0 with the new footer and no deliberate exception. Artifact:
`live_runs/ipopt_clean_stop_acceptance_20261031/report.json`.
The final native-bound implementation was also exercised on the same complete
problem: `live_runs/ipopt_clean_stop_acceptance_20261031_001/report.json`.
It stopped normally at iteration 0 with `early_source_checkpoint`, no error,
and the translated automatic-stop footer, retaining the 51.701397 mm result.

### Exact native box bounds

CasADi now extracts throttle, steering, timing and simple state limits into
native variable bounds. In the real problem this moves 1,944 constraints from
general inequalities to native bounds: inequalities decrease from 63,822 to
61,878, with the same mathematical restrictions. Bound relaxation remains zero;
small bound-push/fraction settings avoid materially disturbing boundary seeds.
The physics equations and all independent certificate gates are unchanged.

Three equal-60-iteration probes compared general inequalities, extracted bounds,
and extracted bounds with close initialization. None found a certified 52 mm
point in that budget. Wall times were 73.11, 73.91 and 69.32 s under concurrent
load; one run each does not establish a reliable timing gain. Native extraction
is retained for the exact smaller representation and in-domain trials, not as
a claimed convergence breakthrough. An earlier adaptive-versus-monotone probe
hit its default CPU limit before equal iteration budgets and is inconclusive;
the production barrier strategy remains adaptive.

### Backtrack clearance goals and retain useful partial improvements

After a failed stronger request, the next request bisects the remaining bracket
above the best certified margin instead of blindly asking for a larger gap.
On resume, archives that exhausted their coarse steps get one bounded retry
round. Existing attempt indices, pending requests and checkpoints are preserved;
sub-0.1 mm bracket work is skipped because the quality ranking uses 0.1 mm bands.
This is a local search policy, not a physical clearance limit.

An explicitly failed stronger-margin solve may now be independently certified
at the original nominal policy and retained if it improves the measured frontier
by more than 0.1 mm. Its archive record says `requested_target_met: false` and
`partial_clearance_improvement: true`; the 52 mm goal is not marked passed.
Purportedly successful goals still require independent confirmation of their
entire requested margin. Bad dynamics, closure, duration or nominal clearance
are not accepted. Existing failed reports can be recovered without rerunning
IPOPT, and the verified gain is exported immediately for subsequent search.

The dashboard shows programmed stop reasons, measured partial gains versus
requested goals, and the fastest line's clearance separately from its time.

## Validation and remaining work

199 targeted optimizer/dashboard/catalog/automatic-validation tests passed.
Coverage includes normal and unexpected callback termination, multi-variable
scaling, bound-domain preservation, independent partial-result rejection,
recovery without another solve, bounded/idempotent clearance retries and
unchanged stronger-goal verification. JavaScript syntax and diff checks passed.

The single existing campaign was restarted with checkpoints preserved and
verified active (PID 146577 at deployment). Its existing simulator-test pause
lease was respected; restart waited for that test owner to finish. On startup,
the campaign independently recovered/exported the 51.701397 mm partial result.
The live dashboard API now reports the fastest 6.925 s line's minimum gap as
5.1701397 cm. Three bounded retry indices were persisted, and the preexisting
pending restoration attempt was resumed before new clearance retries.

Observe subsequent independent certificates and timing before claiming further
convergence or speed improvements. New source witnesses still need separate
Unity/MPC multi-lap tests. No global floor has been established at 6.925 s.
