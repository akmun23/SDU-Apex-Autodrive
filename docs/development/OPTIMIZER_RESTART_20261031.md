# Competition optimizer restart — 2026-10-31

## Failure and correction

The previous campaign failed in the sector-time predictor at iteration 850:
`predictor_timing()` rejected an out-of-domain sector fraction. That exception
escaped both the IPOPT callback and terminal-iterate observation, aborting the
campaign rather than retaining its earlier valid predictor checkpoints.

`command_ocp.py` now skips invalid trial/terminal timing observations, records
their raw fractions and reason, and retains the valid checkpoints. It does not
clip fractions, change the strict timing domain, relax the vehicle model or
certify a fractional-time proposal. An invalid initial/resumed timing contract
still fails explicitly. Real tiny IPOPT fixtures force both below-zero and
above-one terminal fractions and verify that the report is written, the valid
checkpoint survives, and no invalid proposal is exported.

The failed predictor's saved checkpoints were usable: `best_time_candidate.npz`
was saved at iteration 835 with approximately 6.943081 s relaxed predicted time
and 1.414e-7 constraint error. This is an uncertified initializer, not a new
raceline. The independent best source-model raceline remains 6.975 s.

## Resumed process

- Same workspace: `live_runs/optimizer_iros_adaptive_regions_v6_20261030`.
- Started through `tools/run_raceline_campaign.sh` with the original map,
  command seed, 12 workers, 1,200 iterations, 300-iteration stagnation patience,
  12 time sectors, 4 phases, 6 refinements, 3 elite refinements, 8 adaptive regions,
  hybrid time search and audited optimizer-only code migration.
- Detached service: `sdu-track-optimizer-5c036cfcb327.service`; PID 11170 at
  verification. No duplicate optimizer was running.
- Existing exports and checkpoints are preserved. On resumption the campaign
  scheduled clearance polishing of its newer 6.975 s incumbent; pending child
  predictor work remains available for later checkpoint resumption.
- Observed live IPOPT advancing from iteration 5 to 74. Source-model digest
  remains `8f4e47544d5e5205f31ec83a4e067f8c980c9cb1f588b127a46cfd559ae77747`.

The dashboard initially selected the suspended pre-failure child predictor
instead of the newly active parent polishing job. Parent polishing now takes
precedence while it owns the active attempt. A regression recreates the
simultaneously retained child predictor and parent polishing records. The
dashboard service was restarted and its live API confirmed the clearance
objective, 6.975 s duration, explicit solver identity and no warnings.

## Validation and monitoring

100 predictor/command/campaign/track tests and 83 dashboard/catalog/automatic-
validation/elite tests passed. These are software regression tests, not new
Unity racing results.

Dashboard: <http://127.0.0.1:8765/>.

```bash
tail -F /home/akselmo/Documents/GitHub/SDU-Apex-Autodrive/live_runs/optimizer_iros_adaptive_regions_v6_20261030/optimizer.log
```

To stop this campaign gracefully:

```bash
systemctl --user stop sdu-track-optimizer-5c036cfcb327.service
```
