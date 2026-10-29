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

There is deliberately **no 6 s, 7 s, practice-time or other target** in this
entrypoint. Nor is there a ten-minute campaign timeout. Per-solve iteration
chunks and stagnation checks decide when to continue an improving primal or
try another initialization. They do not imply a physical lap-time floor.

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

PGM/YAML do not encode the race's permitted direction. By default both winding
directions are searched. If the event specifies one, append
`--direction counterclockwise` or `--direction clockwise`; do not use a faster
wrong-direction result. Maps must describe an extractable closed circuit;
ambiguous/open maps fail explicitly rather than silently selecting another
track or inventing a feasible path.

Map pixels, metric metadata, production source, optimizer implementation and
search settings are hashed. Restart refuses mismatched workspaces; use a new
`--output` after changing model/code/settings. Saved paths are durable within
the workspace; do not move it during a campaign.

## Development record — 2026-10-29

Implemented:

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

- All 33 campaign, map-only and source-command tests pass, covering warm
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
