# IROS 2026 source-model IPOPT: multi-incumbent plateau escape (29 Oct 2026)

## Purpose

A single minimum-time incumbent can trap both the 25 ms packet-shortening
OCP and the continuous-time relaxation in one local geometry/actuator basin.
At identical lap duration, multiple source-feasible trajectories can have
different full-car clearance, control history, geometry and local descent
directions. Preserve them as **independent seeds**, not as competing claims
about an achieved Unity lap.

## Implementation

- `SDU_Apex_IPOPT_PhysX_MinTime_v2/elite_refinement.py` implements a durable,
  source-certified, equal-duration elite archive.
- **Same-time clearance targets:** +0.1, +0.5, +2 mm clearance beyond the
  baseline 5 cm **full-vehicle** constraint. These are *target constraints*, not
  soft violations or proof of an optimum. IPOPT only minimizes distance to the
  initial state/control guess and command roughness; it does **not** directly
  maximize clearance. The *independent* 1 ms source replay supplies the ranking.
- **Other basins:** smoothly displaced starts with +/-4 cm (one harmonic),
  +/-9 cm (two harmonics) and +14 cm (three harmonics). These are explicitly
  **uncertified initial guesses**. All production vehicle states and controls
  are reoptimized, then *exact-grid* source checks and 1 ms independent full
  footprint verification are mandatory. Only passing, geometrically distinct
  solutions or improved-clearance solutions enter the archive.
- Archive retains up to five additional same-time candidates per incumbent,
  recording immutable witnesses, certificates, hashes, clearance and positional
  RMS diversity. Incumbent time remains unchanged until a strictly faster
  witness passes all gates.
- `command_campaign.py` explores a selected elite with a **new independent
  duration campaign** including both the 25 ms fixed-duration packet/drop
  optimizer and the existing *hybrid-time IPOPT* (24/25 ms source-map
  interpolation). The hybrid continuous-time proposal is NEVER a certificate:
  only exact 25 ms source replay plus independent full-lap 1 ms validation can
  promote a lower duration.
- `elite_first` enabled by default in `track_optimizer.py` starts
  diversification *before* repeating an already-plateaued incumbent's shorter
  trials. `--no-elite-first` defers until the next plateau.
- Only one IPOPT solver is active at any moment: the archive-polishing solve or
  one selected child's time/packet solve, **sequentially**, not parallel jobs.
- Every attempt and active branch are checkpointed; separate child campaigns
  retain their own contracts and resume files.

## Safety and status

The old `optimizer_iros_conditioned_20261029` campaign ended with an
independently certified **7.125 s** source witness. Its separate 7.100 s
near-miss was dynamically accurate but had a tiny **negative** source-tick
footprint slack (~1.14e-9 m), and therefore was **not certified**. A
positive-margin restart using the saved iterate stopped at the CPU limit and
was also NOT source-certified. It was not promoted, and no hard safety rule
was weakened.

A model-feasible lap is NOT the achieved Unity/MPC lap. In particular
the controller has not yet achieved repeatable competition-map ten-lap
performance at these aggressive references. This overhaul is exclusively
to improve the search diversity and physical source certification.

## Live dashboard

`tools/optimizer_dashboard.py` and `tools/optimizer_dashboard.html`
show a separate "Search diversification" section: number of same-time
certified alternatives, explored branches, clearance improvements,
active mode and branch. Polishing and child searches expose their own
solver checkpoints; residual progress is not mislabeled as a faster lap.
The existing certified lap history and all historical/Unity records remain.

## Tests

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=SDU_Apex_IPOPT_PhysX_MinTime_v2 SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python -m pytest -q SDU_Apex_IPOPT_PhysX_MinTime_v2/tests/test_command_campaign.py SDU_Apex_IPOPT_PhysX_MinTime_v2/tests/test_elite_refinement.py SDU_Apex_IPOPT_PhysX_MinTime_v2/tests/test_track_optimizer.py`
- `python3 -m pytest -q tools/test_optimizer_dashboard.py`
- JavaScript syntax: `node --check /tmp/sdu_dashboard_latest.js` extracted
  from the dashboard's script tag.

## Known remaining limitations

- The objective remains **25 ms packet-quantized** after certification.
  The continuous-time predictor creates smaller-than-one-packet proposals
  and potential better seeds, but cannot directly certify a fractional-tick
  lap. The command-acceptance test is still strict.
- Geometry perturbations only explore a finite set of nearby basins; they
  cannot prove a global minimum.
- Archive scoring currently uses minimum full-lap wall margin and geometry
  novelty, not the closed-loop MPC recoverability metric.
- A child full campaign can consume substantial time before the next elite
  receives its turn. A future priority scheduler could interleave bounded
  duration attempts across elites.

## Budget increase and expected iteration resets

- Equal-duration source-witness polishing now receives the **full configured
  IPOPT iteration and stagnation budgets**; previously it was silently capped
  at 450 and 100, even when the top-level campaign permitted more.
- The active IROS run is resumed at **1200 max IPOPT iterations / 300
  stagnation iterations** rather than the previous 900/180 top-level and
  450/100 equal-time split.
- IPOPT iteration numbers start at **0 for each new solve**: different
  margin targets, different geometry starts, new child campaigns and time
  predictors are independent NLPs. A reset is not by itself evidence that
  the daemon restarted or repeated the same candidate.
- Previously completed rejected attempts are checkpointed and not silently
  retried; an interrupted current attempt resumes from its durable saved
  primal in a distinct solve directory.
- Resuming after an optimizer-only code upgrade requires the *explicit*
  `--allow-search-code-update` command-line argument. The metadata/map,
  vehicle source, seed, driving direction and all other settings must match
  the prior contract, and budgets may only increase. The migration is logged
  in `optimizer.json`; the original witnesses are left intact.
- As always, a source-witness candidate only improves the best lap after
  independent 1 ms full-vehicle footprint checks pass. No shortfall can
  be hidden by relaxing the verifier or reporting an IPOPT convergence alone.
