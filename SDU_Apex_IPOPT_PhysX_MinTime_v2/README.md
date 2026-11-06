# Raceline optimization

The production-practice reference is
`f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv`.
It has a source-plant witness beside it. Three supervised ten-lap trials
validated the practice configuration; measured scores are recorded in
`docs/PRODUCTION_BASELINE.md`.

## Maintained optimizer entrypoints

- `run_ipopt.py`: spatial path and speed search.
- `command_ocp.py`: periodic, command-feasible source-plant optimization.
- `command_campaign.py` and `tools/run_raceline_campaign.sh`: resumable,
  checkpointed search over configured command durations and initializations.
- `native_validate.py` and `compare_truth.py`: independent result checks.

Run the unattended map-only campaign from the repository root:

```bash
tools/run_raceline_campaign.sh /path/to/track.pgm /path/to/track.yaml
```

Generated solver workspaces and run outputs are disposable. Validated runtime
trajectories are kept under `f1tenth_planning/trajectories/`; optimizer output
must not replace them until independently validated.

The map-only entrypoint is `track_optimizer.py`. It extracts its own
centerline, generates curvature/length geometry starts, constructs uncertified
command initializers, repairs periodic production-model feasibility, and runs
coarse-to-fine duration searches with IPOPT. Neither a prior raceline, a bag,
nor a target lap time is required. Race direction is always clockwise;
counterclockwise candidates cannot be selected or exported.

Repeat the same invocation to resume. Add `--watch` for live status or `--stop`
for graceful interruption. `best.json` points to the fastest independently
source-validated export bundle found so far. See `docs/MAP_ONLY_OPTIMIZER.md`
for output semantics, limits and current verification results.

The optimizer environment is `SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv`. Source
code, focused tests and the practice-map inputs are kept in this directory.
