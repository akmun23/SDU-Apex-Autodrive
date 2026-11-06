# IROS v6 — autonomous source-IPOPT corner exploration and quality portfolio
Date: 2026-10-30

## Prior problem

The older optimizer perturbed fixed global harmonics, repeated similar
unrealizable seeds and hid independently certified same-duration improvements
when the active workspace switched. Those two failure modes are separate:
**algorithmic search diversity** and **historical result visibility**.

## Adaptive exploration

- `adaptive_regions.py` scores the eight sectors of a certified source
  trajectory from curvature times speed squared and node vehicle-footprint
  slack beyond 50 mm. This is a *prioritization heuristic*, not a proof that
  this is the true optimum or a controller stability result.
- A compact periodic, mean-zero steering-and-countersteering adjustment
  localizes the perturbation near a selected corner and its exit.
- Every seed is propagated through the production **21-state, 25 ms source
  map**, including hidden vehicle states. The periodic projection remains
  explicitly uncertified. A >10 mm initial track footprint deficit is
  rejected before launching an expensive IPOPT solve.
- The deterministic, persisted scheduler uses each attempt's certificate
  result. After an infeasible seed it shrinks amplitude; after a certified
  and accepted successor it explores a changed corner center (+/- 0.25
  sector) with a somewhat greater amplitude; after failures it explores
  other ranked sectors.
- Eight feedback-driven adaptive starts are interleaved with clearance
  and smoothness objectives. Existing physical source and independent full
  1 ms perimeter certification remain mandatory.
- The fixed-duration repair solver now checks an already feasible
  checkpoint against exact source replay before wasting hundreds of
  iterations on dual convergence. The normal final independent
  certification and elite diversity/quality checks must still pass.
- Exactly one heavy source IPOPT process runs at a time. Local source
  propagation for candidate generation is synchronous, not concurrent.

## Experimental checks

The first three sector-local initializations on the IROS competition map,
from the 7.100 s +16.89 mm source witness, had positive minimum node
footprint slack and scaled initial source defects approximately
0.000607, 0.0000837 and 0.000351. Those were *initializers*, not certified
alternative laps. A prior interrupted faster-duration IPOPT solve was
subsequently independently certified as **7.075 s** (at approximately zero
additional wall slack beyond the 50 mm minimum). The 7.075 s witness and
7.100 s +16.89 mm witness have different tradeoffs, both retained.

## Read-only historical portfolio in GUI

`tools/source_witness_catalog.py` reads previously certified source reports
in bounded known workspace directories. It requires matching plant SHA,
required 50 mm margin, full 1 ms perimeter validation and recursive
dynamics check. For sibling optimizer workspaces, it also requires the
same bundled occupancy-map image SHA. Standalone source certification
directories lacking a comparable map fingerprint are explicitly marked
as such. It never promotes anything or launches physics.

Witnesses are deduplicated by their file contents. The UI shows separate
current-run alternatives, all distinct historical source witnesses and the
**non-dominated time/clearance/steering RMS/throttle RMS portfolio**.
This deliberately retains slower trajectories with substantially more
physical margin, rather than ranking exclusively by shortest theoretical
time.

A model-certified trajectory does not imply ten completed Unity/MPC laps.
No claim of a global minimum, global convergence or guaranteed controller
robustness is made.

## Tests and reproducibility

- 95 optimizer tests passed on Oct 30.
- 31 dashboard and certified-catalog tests passed on Oct 30.
- The adaptive campaign has a distinct optimizer SHA and fresh workspace,
  avoiding silent in-place mutation of older source/seed/search contracts.
- Previous workspaces and their certified exports remain immutable.
