# Raceline optimization

The offline optimizer searches for a fast periodic lap from an occupancy map, then checks the commands against the production vehicle model.

## Inputs and entrypoints

The map YAML supplies image, metric resolution and origin. No bag, prior raceline,
controller, requested lap time or track-specific script is needed for the search.
An adjacent same-name YAML can accompany an image implicitly; YAML alone also works.
Travel direction is a separate constraint: IROS uses a Unity-driven reference's
yaw winding and signed area, not an orientation inferred from occupancy pixels.

| Source in [raceline_optimizer](../raceline_optimizer/) | Purpose |
|---|---|
| `track_optimizer.py` | Map-only orchestration and resumable workspace |
| `min_curvature_seed.py` | Geometry initialization |
| `command_ocp.py` | Periodic source-plant command optimization and witness validation |
| `command_campaign.py` | Checkpointed duration and initialization search |

The launcher requires the optimizer's package-local Python virtual environment
named `.venv`; it is generated locally and is absent from a source-only checkout.
Source and focused tests live in the package. Generated solver workspaces and
caches are disposable; keep validated runtime references under
[f1tenth_planning/trajectories](../f1tenth_planning/trajectories/).
The practice reference is
[raceline.csv](../f1tenth_planning/trajectories/practice/raceline.csv),
with a [source witness](../f1tenth_planning/trajectories/practice/witness.npz)
beside it. Three supervised ten-lap trials were run with that practice configuration. The saved validation report
records a source-model check and sets `unity_validated` and `mpc_validated`
to `false`; it does not certify Unity driving. [Trials](trials.md) gives the
checked-in lap extract and the other reported comparisons.

## Run, observe, stop and resume

Run from the repository root. These examples use checked-in inputs:

```bash
map=f1tenth_planning/maps/practice.yaml
tools/raceline/run_raceline_campaign.sh "$map"
tools/raceline/run_raceline_campaign.sh "$map" --watch
tools/raceline/run_raceline_campaign.sh "$map" --status
tools/raceline/run_raceline_campaign.sh "$map" --stop
tools/raceline/run_raceline_campaign.sh "$map"
```

Repeat exactly the same map and settings to resume. `--watch` updates stage,
checkpoint and memory status; `--status` prints once without starting a solver.
`--stop` retains saved iterates and certified exports. `Ctrl+C` in the viewer
stops only that viewer. `--foreground` runs in the terminal instead of a user
systemd service. The launcher requires a user systemd session for its preflight
even with `--foreground`. The launcher prints its workspace and a `tail -F` log command.
`--output DIR` selects a durable workspace; do not move it during a campaign.

For IROS, pass the checked-in runtime direction reference explicitly. The
optimizer uses a local recorded-lap default only when that file exists;
it is absent from this checkout:

```bash
map=f1tenth_planning/maps/competition.yaml
direction=f1tenth_planning/trajectories/competition/raceline.csv
tools/raceline/run_raceline_campaign.sh "$map" --driving-reference "$direction"
tools/raceline/run_raceline_campaign.sh "$map" --driving-reference "$direction" --watch
tools/raceline/run_raceline_campaign.sh "$map" --driving-reference "$direction" --stop
```

The IROS reference has positive yaw winding and signed area in ROS map XY.
Both signs must match at certification; reversing a witness invalidates
its dynamics certificate. Without a verified direction reference, map-only search
uses its clockwise convention. Unity trials remain the direction/startup check.

`--prepare-only` checks geometry without a dynamics solve. `--bootstrap-only`
stops after the first independently certified periodic lap. To continue a witness,
set `new_workspace` to a fresh output location, then run:

```bash
tools/raceline/run_raceline_campaign.sh \
  f1tenth_planning/maps/practice.yaml \
  --command-seed f1tenth_planning/trajectories/practice/witness.npz \
  --output "$new_workspace"
```

The copied seed is independently recertified against the supplied map, current
source and required direction; invalid seeds cannot enter the best manifest.

## Clearance and rear-slip reserve

`--margin M` adds wall clearance in metres beyond the rear-axle perimeter probes
and their unsampled-edge correction. The default is 0.050 m in
`MapGrid.footprint_clearance`, the command OCP and campaign. Bootstrap, search and independent 1 ms certification
share that margin; use it again with `command_ocp.py --validate-only`.
Changing reserve options requires a new output workspace.

Choose clearance at station `s` at least `controller_reserve + 3*sigma_y(s)`.
The practice controller reserve is 0.020 m. With the reported total
apex-scatter SD of 0.024 m, this rule gives `--margin 0.092`. The reported
localization SD of 0.013 m is included in the total; do not add it again or
divide correlated scatter by the square root of sample count.

One margin must cover the worst relevant station. Geometric starting lines
are initial guesses, not validated command trajectories.

`--rear-slip-limit X` optionally bounds absolute dimensionless
`WheelHit.sidewaysSlip` for each rear wheel at every 1 ms tick in each 25 ms
packet; default `None` leaves it off. The trace uses the final vehicle substep's
slip, matching `SourcePlant.step(..., duration_s=.001)`, rather than a maximum
over finer wheel substeps or continuous time. The OCP imposes signed bounds
`-1 <= slip/X <= 1` with branch-local forward AD, without a new absolute-value kink.
Certification checks shooting packets and independent recursive 1 ms replay;
nonfinite or over-limit slip fails with no slip tolerance. Reports retain maxima
and the limit. Pass the same option to `--validate-only`.

Choose `X <= 0.01 - 3*sigma_slip` from the worst rear wheel/tick, estimating
`sigma_slip² = J Sigma J^T` with disturbance covariance and correlated
localization/tracking errors. Source cubic response of 90–95% corresponds to
slip 0.00762–0.00835, conditional on measured uncertainty. There is no automatic
sigma estimate or fixed grip reserve. Both options are accepted by
`track_optimizer.py`, `command_campaign.py` and `command_ocp.py`.
To drive an export, set `SDU_APEX_TRIAL_TRAJECTORY_FILE` to its repository-local
CSV and use [the trial runner](trials.md#run-a-closed-loop-trial).

## Search and stopping

The campaign snapshots map pixels and metadata, extracts a closed centerline
and raw wall distances, then generates footprint-checked curvature/length starts.
It uses kinematics, source parameters and suspension initialization to create
uncertified periodic guesses for states and controls at 25 ms intervals.
IPOPT repairs them using compiled production transitions and derivatives; the symbolic graph stays unexpanded.
Independent checks test local dynamics, periodicity and whole-lap motion
without state resets. They also check footprint slack at every 1 ms source tick.

When a candidate certifies, the campaign tries to shorten its duration by about
10%. It halves the reduction down to one 25 ms packet and tries cyclic phases,
additional wall cuts and saved primal solutions. The defaults are three
geometric starting lines, four phases and six clearance refinements. Each solve has a 1,500-iteration chunk; stagnation triggers after
250 iterations without 2% improvement in feasibility residual. An iteration-limited
primal can continue when its residual falls below half its starting value.
A positive factor scales the fixed-duration objective without changing its
minimizers or relative penalties. Before globally resampling for a one-packet
reduction, the search tries removing a single gap selected using the source model.

Raising only budgets, for example `--iterations 2500 --stagnation-iterations 500`,
is supported on resume. It preserves bootstrap/incumbent data, interrupted phases
and reports, and reopens budget-limited failures from saved primals. Lower budgets
or model/map/settings mismatches fail the workspace contract. Map pixels,
metadata, source, implementation and settings are hashed. Use a fresh output
for such changes. An explicit `--allow-search-code-update` supports audited
optimizer-only migrations with unchanged physics and nondecreasing budgets.

`--hybrid-time-search` enables an uncertified local time predictor after a failed
shortening attempt. It interpolates literal 24/25 ms source maps, proposes joint
time/geometry/control changes and rounds up to full packets. It conservatively
bounds a complete slip trace when requested. Saved incomplete predictor primals
can resume and compete with ordinary seeds by exact defect; their raw schema is
rejected by witness certification. Exact 25 ms repair and full-lap 1 ms certification
remain mandatory; the predictor adds no free-duration derivative to the runtime ABI.

There is no campaign-wide timer, fixed iteration total or requested 6 s/7 s lap
target. One reported IROS bootstrap uses five IPOPT iterations, spends
36.36 s in IPOPT
and takes 57.37 s through certification. Other maps can take longer.

Search has no reliable ETA and may take hours or longer. `plateau` means configured tries cannot
improve the incumbent; `no_validated_raceline` means no initialization passes.
The 30-packet numerical lower domain reports `search_domain_reached`.
None proves a global minimum or physical impossibility.

## Generated outputs and validity

The launcher creates an ignored workspace whose name includes map/source hashes.
These names describe generated artifacts, not checked-in files:

| Artifact | Meaning |
|---|---|
| `optimizer.json`, `optimizer.log` | Stage, basins, saved best and complete IPOPT progress |
| `input`, `geometry`, `starts` directories | Immutable map, directional centerlines/starts, bootstrap reports and resumable campaigns |
| `best.json` | Atomic pointer to fastest independently source-certified export |
| Witness-hash export directories | Witness, nine-column ordinary reference, command-enriched CSV and validation report |

Exports are immutable. Slower/unvalidated attempts cannot replace the best pointer.
Held steering/throttle and full shooting state remain in the witness separately.
No generated file automatically replaces a production reference. Certification
means agreement with the production model, not private Unity PhysX identity or
successful MPC driving; [Unity validation](trials.md#automatic-export-validation)
is a separate gate.

## Dashboard

Set `workspace` to the launcher's printed output directory when selecting explicitly:

```bash
tools/raceline/watch_raceline_gui.sh
tools/raceline/watch_raceline_gui.sh "$workspace"
```

The first command auto-selects only when exactly one workspace exists.
The viewer serves http://127.0.0.1:8765/ and polls every 2 s as a separate,
low-priority local observer. Closing it does not stop IPOPT; reopening reuses
the service. `--no-open` prints the URL, `--port 8766` selects another port and
`--serve` stays foreground. The launcher prints the viewer-only stop command.

The dashboard separates the certified best result from the uncertified trial.
It shows IPOPT iterations and chunk budgets, residuals from five-iteration
callbacks, saved checkpoints and lap-time improvements. The map, centerline
and raceline use metric coordinates. Other panels show the basin, phase and
refinement, checkpoint age, worker memory, attempts, raw logs and certified downloads.
Its `1e-4` line is the solver candidate gate, not metres: certification also needs
local/periodic defects ≤`1e-5`, recursive XY error ≤5 cm and nonnegative footprint
slack at 1 ms. IPOPT `inf_pr` can use different scaling and is plotted separately.

The observer does not import optimizer modules, compile physics, alter settings
or solver state, or start/stop a solver. It binds to localhost and serves no
arbitrary filesystem paths. A same-origin JSON retry endpoint queues failed
certified Unity trials for the supervisor. Stale/inactive processes are labeled;
there is no global-minimum claim or overall completion percentage. The viewer
lies outside the implementation hash contract. Expanded panels and selected
text survive polling; Pause live display stops polling only.
