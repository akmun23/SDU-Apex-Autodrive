# Fast-line clearance and convergence improvements — 2026-10-31

## Observed progress

The competition campaign improved its independently source-verified minimum
time from 6.975 s to 6.925 s. The latter's conservative minimum gap is
49.999687 mm (within the declared 0.001 mm numerical tolerance of 50 mm).
A separate 7.050 s alternative has 84.406939 mm minimum clearance. These are
production-model witnesses, not new Unity/MPC racing results.

Inspection of completed reports found 54 repair solves ending to add missing
1 ms wall constraints. Only three of 111 repair reports reached IPOPT success
or acceptable optimality; feasible witnesses were often retained earlier via
source checks or objective stagnation. Increasing iteration counts alone
would not address the repeated constraint discovery or redundant margin work.

## Implemented

1. **Reuse known exact wall checks.** `command_ocp.inherited_clearance_cuts`
   follows hash-linked witness/certificate provenance and retains validated
   interval/tick indices only for matching source, map image and command count.
   Phase-seed files and count changes do not blindly reuse indices. New
   independent certificates retain map identity and the known-cut metadata.
   A new solve still performs all independent recursive/source checks; no
   certificate is transferred or physical constraint relaxed.
2. **Measure the clearance frontier.** A margin trial now requests an increase
   above the best certified clearance already obtained, not merely above the
   original 50 mm baseline. Previously a seed with an 84 mm gap could be asked
   to meet 50.1 mm again. Each request is persisted before solving. Interrupted
   and already-completed pending trials retain their exact original target.
3. **Continuation before unrestricted exploration.** New archives try 2, 5 and
   10 mm clearance increments, subject to the configured refinement count,
   before the less-conditioned global quality/geometry trials. Each successful
   step warm-starts the next from the independently certified better seed.
   Existing archives retain their saved input increments. Stable attempt
   indices are preserved, and an in-flight attempt takes priority on resume.
4. **Finish useful feasibility work promptly.** Margin-only steps and their
   missing-tick repairs stop once the stronger requested margin passes full
   source replay, rather than waiting for unnecessary objective/dual optimality.
   Independent certification at the baseline margin must additionally prove
   the requested extra clearance before archiving. Clearance/smoothness
   objective solves still seek improvements rather than stopping on their
   feasible starting points.
5. **Expose the actual task.** The dashboard names incremental clearance
   refinement and its requested wall gap, separately from the fixed lap time.

The source vehicle equations, steering/throttle bounds, nominal footprint,
1 ms verification, 25 ms commands and acceptance tolerances are unchanged.
These changes improve search construction/allocation, not model fidelity.

## Verification and deployment

190 targeted optimizer/dashboard/catalog/automatic-validation tests passed.
Tests cover source/map/count mismatch, hash-linked certificate provenance,
bad/duplicate cut indices, true incremental targets, target persistence,
independent rejection of insufficient clearance, stable reordering and early
feasibility completion.

On the actual 6.925 s seed, all 14 known source cuts were recovered. Adding them
increased constraints from 69,045 to 69,661 (0.89%), left the initial residual
unchanged at 1.388700e-6, and changed measured graph construction from 0.172 s to
0.179 s in one comparison. This is a consistency/setup-cost check, not proof
of a convergence or Unity driving improvement.

Deployed with a checkpoint-preserving restart of
`sdu-track-optimizer-5c036cfcb327.service`. Its first new job is margin attempt
15 at fixed 6.925 s, requesting 52 mm clearance with all 14 inherited cuts.
The initial constraint residual is 0.002 m, correctly exposing the new 2 mm
request. The live service and dashboard agree on identity, objective and target.
The interrupted 6.900 s trial and prior predictor candidates remain on disk;
none was promoted or deleted. The campaign still keeps only one optimizer
running and retains the existing per-solve 1,200/300 iteration budgets.

## Remaining evidence to collect

Observe whether the 52 mm continuation step passes full replay and independent
certification, then whether the further increments remain feasible without
increasing lap duration. A failure is evidence about this local basin/budget,
not proof of a physical global limit. New certified roomier lines still need
the separate Unity/MPC multi-lap validation before being called reliable race
lines. No reduction in solve time, new clearance gain, global minimum or online
reliability is claimed until measured.
