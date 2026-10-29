# Build and run

## Practice trial

The maintained practice pair is the `autodrive_practice_20260924_b` map and
`source_command_4p800.csv` trajectory. Its MPC profile is
`mpc_practice_fastline.yaml`; production AMCL parameters are used directly.
Run a warmup lap plus ten scored laps with:

```bash
SDU_APEX_TRIAL_TOTAL_LAPS=11 \
  tools/run_mpc_practice_trial.sh practice_fastline_01
```

The runner starts the controller, recorder, simulator, and collision/lap
watcher, and stores the complete run under `live_runs/`. Use a new run ID each
time. The current baseline and recorded scores are in
[`docs/PRODUCTION_BASELINE.md`](docs/PRODUCTION_BASELINE.md).

## Competition image

Build and launch the fixed competition stack:

```bash
tools/build_competition.sh
tools/run_competition.sh
```

The competition launch uses the installed competition map/trajectory pair and
only steering and normalized throttle actuator outputs. Check permitted ROS
topics with `python3 tools/verify_runtime_topic_policy.py` before launch.

## Development launch

For development without the supervised trial wrapper:

```bash
AUTODRIVE_REBUILD=0 SDU_APEX_IMAGE=sdu-apex-autodrive:dev tools/start_dev.sh
SDU_APEX_SIM_TRACK=practice SDU_APEX_SIM_MODE=batchmode tools/start_simulator.sh
```

Use `tools/stop_autodrive.sh` or Ctrl-C to stop the relevant process. Do not
reuse a run ID; `live_runs/` is ignored by Git and contains only local captures.
