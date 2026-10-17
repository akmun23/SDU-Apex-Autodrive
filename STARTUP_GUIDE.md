# Startup guide

The maintained stack is bridge + PhysX odometry + localization + direct
throttle MPC + actuator interface. Start the ROS stack before connecting the
simulator. The simulator helper uses Unity batch mode with a graphics display
through Xvfb; do not add `-no-graphics`.

## Build and run the development stack

```bash
./tools/start_dev.sh
```

This rebuilds the development image by default, checks the competition topic
policy, and starts the single MPC/localization launch. To reuse an existing
image without rebuilding the image layers (the mounted current workspace is
still compiled before launch):

```bash
AUTODRIVE_REBUILD=0 SDU_APEX_IMAGE=sdu-apex-autodrive:dev \
  ./tools/start_dev.sh
```

For the practice map/raceline pair:

```bash
AUTODRIVE_REBUILD=0 \
SDU_APEX_IMAGE=sdu-apex-autodrive:dev \
SDU_APEX_MAP_YAML="$PWD/f1tenth_planning/maps/autodrive_practice_20260924_b.yaml" \
SDU_APEX_TRAJECTORY_FILE="$PWD/f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv" \
./tools/start_dev.sh
```

Map and trajectory overrides must be supplied together. Optional development
diagnostics and parameter overlays are available through `SDU_APEX_WITH_RVIZ`,
`SDU_APEX_MPC_PUBLISH_DIAGNOSTICS`, and the `SDU_APEX_*_PARAMETER_OVERLAY`
environment variables.

## Start the simulator

In another terminal, after the ROS stack is listening:

```bash
SDU_APEX_SIM_TRACK=practice SDU_APEX_SIM_MODE=batchmode \
  ./tools/start_simulator.sh
```

Use `SDU_APEX_SIM_TRACK=compete` to select the other installed simulator
image. Keep the simulator terminal open; stop it with `Ctrl-C`.

## Run monitoring and recording

Start the collision/lap watcher and optional debug recorder before starting
the simulator:

```bash
./tools/watch_sim_run.sh <controller-container> <simulator-container> \
  <recorder-container> 12
```

```bash
SDU_APEX_RUN_ID=run_name ./tools/record_debug_bag.sh
```

The watcher stops the simulator immediately on collision and finalizes the
recorder. Preserve only captures intended for a specific analysis in the
workspace; run data belongs under ignored `live_runs/`.

## Competition image

Build the self-contained image with its installed map/raceline pair:

```bash
./tools/build_competition.sh
```

With the simulator already running, launch the fixed competition stack:

```bash
./tools/run_competition.sh
```

The competition launch has no recorder, RViz, mapping, alternate controller,
or debug parameter overlay. Its only actuator command is steering angle plus
normalized throttle.

Stop the development container with `Ctrl-C`, or run
`./tools/stop_autodrive.sh` from another terminal.
