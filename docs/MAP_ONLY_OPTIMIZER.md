# Map-only raceline optimizer

## Run, observe, stop, resume

From the repository root:

```bash
tools/run_raceline_campaign.sh /path/to/track.pgm /path/to/track.yaml
```

The YAML must supply the image, metric resolution and map origin. An adjacent
same-name YAML can be omitted from the command; supplying the YAML alone also
works. No raceline, bag, controller, requested lap time or track-specific
script is required.

Use exactly the same map arguments and optional settings for these commands:

```bash
# Human-readable, updating stage / solver-checkpoint / memory status
tools/run_raceline_campaign.sh /path/to/track.pgm /path/to/track.yaml --watch

# One-shot status, without starting a solver
tools/run_raceline_campaign.sh /path/to/track.pgm /path/to/track.yaml --status

# Graceful stop; retain saved iterates and every certified result
tools/run_raceline_campaign.sh /path/to/track.pgm /path/to/track.yaml --stop

# Same start command resumes rather than overwriting
tools/run_raceline_campaign.sh /path/to/track.pgm /path/to/track.yaml
```

`Ctrl+C` in the viewer stops only the viewer. `--foreground` runs the optimizer
in the current terminal instead of a user systemd service. The launcher prints
the output directory and a `tail -F` command for the complete IPOPT log.
`--prepare-only` checks centerline/geometry generation without a dynamics solve;
`--bootstrap-only` stops after the first independently certified periodic lap.
Use `--output DIR` to select a durable workspace.

### Live graphical dashboard

```bash
# Auto-select the workspace when exactly one map-only optimizer workspace exists.
tools/watch_raceline_gui.sh

# Or select one explicitly (needed when several tracks have been optimized).
tools/watch_raceline_gui.sh live_runs/optimizer_autodrive_iros_2026_compete_dc3c93097d
```

This opens **http://127.0.0.1:8765/** in your browser. It refreshes every two
seconds and runs as a separate, low-priority local observer service. Closing
the browser or its terminal does not stop IPOPT. Relaunching the same viewer
reuses the existing service. Use `--no-open` to print the address without
opening a browser, `--port 8766` for another workspace, or `--serve` to keep
the viewer in the current terminal. The launcher prints the exact command
to stop **only the viewer service**.

The dashboard shows:

- Best independently source-validated lap versus the current unvalidated trial.
- Live IPOPT iteration and per-solve budget, not a made-up overall completion percentage.
- Actual constraint residuals, sampled by the existing solver callback every
  five iterations, with best-observed and saved-checkpoint values distinguished.
- Historical validated lap-time improvements, the occupancy map, extracted
  centerline, and latest saved raceline in the map's metric frame.
- Geometry basin, cyclic phase, clearance refinement, checkpoint age,
  worker count, process memory, recent attempts and raw solver output.
- Downloads of the current validated reference, command-enriched CSV and report.

The displayed `1e-4` line is only the solver's candidate feasibility gate.
Independent certification also requires local/periodic defects at most
`1e-5`, whole-lap recursive XY error at most 5 cm and nonnegative 1 ms
footprint slack. Residuals are scaled constraint violations, **not metres**.
Raw IPOPT `inf_pr` may use internal scaling and is deliberately not mixed
into the actual-residual plot. Neither set of model checks certifies Unity
driving or MPC performance.

The viewer does not import optimizer modules, compile physics, change settings,
write into the workspace, or start/stop a solver. It binds only to localhost,
has no control/write endpoints and serves no arbitrary filesystem paths.
An inactive process or stale log is labeled explicitly; no reliable overall
ETA or global-minimum claim is shown. The dashboard is outside the optimizer's
implementation-hash contract, so installing it does not invalidate a running
campaign or require restarting one.

There is deliberately **no 6 s, 7 s, practice-time or other target** in this
entrypoint. Nor is there a ten-minute campaign timeout. Per-solve iteration
chunks and stagnation checks decide when to continue an improving primal or
try another initialization. They do not imply a physical lap-time floor.

## Runtime and stopping budgets

The first IROS clockwise bootstrap used five IPOPT iterations, 36.36 seconds
in IPOPT, and 57.37 seconds from saved initializer through independent
certification. This is an observation for this run, not a universal timeout
or a promise for other maps. More difficult initializations can take longer.

There is no campaign-wide timer or fixed total iteration count. Each solve
has a default 1,500-iteration chunk; a local stagnation check triggers after
250 iterations without a 2% improvement in the tracked feasibility residual.
An iteration-limited solve may continue from its primal when its residual has
fallen below half of its starting residual. Otherwise the campaign tries
other durations, phases and geometry starts. Defaults are three geometry
starts, four cyclic phases, up to six clearance refinements, and duration
reductions ranging from approximately 10% down to one 25 ms packet.

The former defaults were 300/50. The 50-iteration gate stopped an 18.8 s
candidate at a 2.63e-4 residual despite passing recursive-position and wall
checks; it was not accepted, because dynamics feasibility still failed.
No completed solve at that point had exhausted 300 iterations. The larger
iteration ceiling provides headroom, while the longer stagnation window
addresses the observed stopping reason directly.

For still more exploration, use the same map/output with, for example,
`--iterations 2500 --stagnation-iterations 500`. Raising only these budgets
is supported on resume: completed bootstrap data and the incumbent remain,
earlier budget-limited failures are reopened from their saved primals, and
the active interrupted phase is preserved. Old attempt reports remain in
the history. Lower budgets or other model/map/settings changes still fail
the contract check; raising budgets never relaxes validation thresholds.

Every newly certified faster incumbent starts another shortening search.
The campaign finishes when its configured tries cannot produce another
validated improvement. This can take hours or longer; current measurements
do not support a reliable completion estimate. It is a finite-search plateau,
not proof that an unlimited solve or another geometry basin could not improve.
The separate 30-packet numerical lower domain is reported as
`search_domain_reached`, never mislabeled as a plateau or a 6 s goal.

## Automatic stages

1. Validate the supplied map pair and snapshot pixels plus metric metadata.
2. Extract a closed map centerline and raw left/right wall distances. Preserve
   it as `geometry/<direction>/centerline.csv`.
3. Generate footprint-checked geometry starts: curvature refinement and two
   curvature/length compromises. These weights select starting basins, not
   tire/friction limits. The centerline is an initialization, not a path the
   optimized vehicle must exactly follow.
4. Create explicitly uncertified 25 ms periodic state/control guesses from
   each geometry. The guesses use kinematics and the production source's
   parameters/suspension initializer. They are **not measurements**, and are
   never exported as dynamically feasible racing lines.
5. Repair each guess with IPOPT using the compiled production transition and
   its derivatives. The code keeps the symbolic graph unexpanded rather than
   expanding the complete 1 kHz plant into the large spatial NLP.
6. Independently recheck local dynamics, periodicity, no-reset whole-lap
   recursive prediction and the full footprint at each 1 ms source tick.
   Only passing results can become an incumbent/export.
7. Reuse the existing command campaign: try approximately 10% duration
   reductions, halve unsuccessful reductions down to one 25 ms packet, try
   cyclic starting phases, discover missing wall cuts, and feed saved primals
   back to IPOPT. Try the other geometry basins as well.
8. Publish each independently checked improvement immediately. Stop with
   `plateau` after the configured starts cannot improve the incumbent, or
   `no_validated_raceline` if no initialization passes. Neither means a global
   optimum or physical impossibility has been proven.

At most one sub-agent may be active during development. The unattended
optimizer itself uses native worker threads, not AI sub-agents.

## Outputs and validity

Default workspaces are `live_runs/optimizer_<map-name>_<map-hash>/`.

- `optimizer.json`: orchestration stage, completed/active basins, saved best.
- `optimizer.log`: full progress, IPOPT iterations and residuals.
- `input/`: immutable working map pair, including resolution and origin.
- `geometry/`: extracted centerlines, geometry seeds and plots.
- `starts/`: bootstrap iterations, reports and resumable campaigns.
- `best.json`: atomic pointer to the fastest independently checked export.
- `exports/<witness-hash>/`: witness, ordinary reference CSV, command-enriched
  reference CSV and validation report. Exports are immutable; slower or
  unvalidated attempts cannot overwrite the best pointer.

The ordinary reference remains nine columns. Held steering/throttle and the
complete shooting state are retained separately in the witness. No generated
file overwrites the production reference automatically.

Source-validated means agreement with our production model, **not** proof of
Unity native-solver identity or successful MPC driving. Online competition
validation remains a separate gate. A plateau is the best found within this
multistart search, vehicle/domain constraints and 25 ms command grid—not a
global minimum-time certificate.

Racing is always **clockwise**, as specified by the user. The entrypoint no
longer offers bidirectional or counterclockwise search. Only clockwise starts
are generated, and export checks both negative yaw winding and negative signed
path area before publishing an incumbent. Maps must describe an extractable closed circuit;
ambiguous/open maps fail explicitly rather than silently selecting another
track or inventing a feasible path.

Map pixels, metric metadata, production source, optimizer implementation and
search settings are hashed. Restart refuses mismatched workspaces; use a new
`--output` after changing model/code/settings. Saved paths are durable within
the workspace; do not move it during a campaign.

## Development record — 2026-10-29

Implemented:

- Added the read-only browser dashboard (`tools/optimizer_dashboard.py` and
  `.html`, launched by `tools/watch_raceline_gui.sh`). Eleven observer tests
  cover real NPZ metadata decoding, scaled-vs-actual residual separation,
  new-solve/reset handling, map frame preservation, workspace read-only
  behavior, PID identity checks and local HTTP/path confinement. Desktop
  rendering was checked with headless Chrome while the original optimizer
  process continued unchanged. Its latest state is always read live rather
  than hardcoded in the viewer.
- Replaced the practice-only launcher with PGM/YAML input resolution.
- Added map-only centerline, geometry-start and command-guess bootstrap.
- Reused/extended the periodic source-command campaign for coarse-to-fine,
  target-free searches; saved attempt keys identify their starting incumbent.
- Added independent-validation-only best exports, atomic manifests, locks,
  source/map/implementation guards, stage/status/watch/stop/resume commands.
- Preserved the current production model, odometry, MPC and practice reference.
- Stopped `sdu-apex-iros-ipopt.service`, the memory-heavy standalone bootstrap.
  It had no IPOPT iterate/checkpoint; its setup logs remain in
  `live_runs/iros_ipopt_20261029_02/`.

Verification results:

- All 38 campaign, map-only and source-command tests pass, covering warm
  restart, power-loss-style interruption, coarse-to-fine trial selection,
  metadata/source guards and rejection of unvalidated exports. Test doubles
  exercise orchestration, not vehicle-accuracy certification.
- A real map-only IROS preparation generated the counterclockwise centerline
  and all three geometry starts. The first attempt to refine the already
  wall-hugging curvature result failed its two-sided-corridor check; geometry
  variants now correctly start from the preserved extracted centerline.
- The canonical PGM/YAML-only service generated starts for both directions,
  then converged its first clockwise periodic bootstrap in five IPOPT
  iterations (36.36 s solver time), with residual 1.92e-7. Independent source
  replay/footprint checks passed and the 39.175 s initial witness was exported.
  This deliberately slow initializer is not the requested racing performance.
- The automatic campaign proceeded to a 35.275 s trial, saving intermediate
  iterates. During this solve memory was about 2.5 GB with no process swap,
  rather than the standalone bootstrap's 27 GB resident plus heavy swapping.
- After the user's direction clarification, the map-only entrypoint was
  restricted to clockwise geometry and exports, with a regression test that
  rejects a counterclockwise witness even if it carries a source certificate.
  The already-clockwise incumbent and active solver primal were retained;
  counterclockwise development geometry remains only in the historical run
  folder and is no longer selected for optimization.
- Subsequent automatic improvements reached 35.275, 31.750, 28.575, 25.725,
  23.175 and 20.875 s, all independently source-validated, not online racing
  records. The defaults were then raised to 1,500/250 using the observed
  18.8 s stagnation case. Regression tests cover reopening old failed primals
  after a budget increase, preserving an interrupted later phase, and rejecting
  disguised source changes or smaller budgets.
- Live resume with the larger settings preserved the active clockwise phase,
  reopened earlier budget-limited reports, then certified 18.800 s (four
  iterations, local defect 2.62e-10) and started a 16.925 s trial. This verifies
  operation of the new resume path; it does not prove the larger ceiling caused
  that improvement, because the accepted solve itself needed only four steps.
- The production-model initializer and derivatives were exercised numerically;
  the artificial circular geometry fixture is labeled as a module test, not
  as captured vehicle data.
- An actual service stop returned `interrupted`, retained the best validated
  export and saved primal, and the identical start command resumed into
  `solve_001` from `solve_000/best_iterate.npz`. The resumed initial residual
  exactly matched 0.05215741284790565. No bootstrap or map extraction was
  repeated, and the existing best pointer remained unchanged.
- One independent reviewer identified stale orphaned input bundles, incomplete
  helper-module hashing and loss of search-domain terminal status. All three
  were fixed and covered by regression tests. Both IROS and practice maps
  generated their geometry starts without any previous raceline or bag.
- After those non-physics audit fixes, the IROS input bundle and unchanged
  source digest were checked again and the incumbent independently revalidated
  (`audit_revalidation/report.json`, local defect 1.92e-7, periodic defect 0).
  The development run's implementation hash was explicitly updated with a
  recorded audit migration, preserving its original saved primal. This was a
  pre-handoff maintenance action, not a bypass of validation or an automatic
  acceptance of arbitrary future code changes.

Run: `live_runs/optimizer_autodrive_iros_2026_compete_dc3c93097d/`.
Its current best is a model-certified slow starting lap, not an online racing
record. No new 7 s or 6 s competition raceline is claimed. Subsequent search
results are recorded automatically in its manifests, reports and log rather
than copied into this static document after each iteration.

The BachelorProject pipeline was inspected as the workflow reference:
`f1tenth_planning/scripts/optimize_trajectory.py` there extracts a GVD/thinned
centerline, measures widths, invokes its optimizer, then exports the reference.
Its real-car vehicle model, heading conversions, speed clamping and INI-file
patching were not copied into this simulator-source optimizer.
