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

## Freeze and execute one MPC horizon

For a development-only rollout check, the MPC can latch its next accepted
30-command plan once odometry reports at least 7 m/s, publish all 30 outputs
without another solve, and emit predicted-versus-current state samples on
`/mpc/open_loop_test`. The normal controller profile leaves this disabled.
Use the focused recorder profile to avoid recording the high-volume lidar
stream:

```bash
SDU_APEX_MPC_PARAMETER_OVERLAY="$PWD/f1tenth_mpc/config/mpc_open_loop_horizon_test.yaml" \
SDU_APEX_MPC_PUBLISH_DIAGNOSTICS=true \
AUTODRIVE_REBUILD=0 SDU_APEX_IMAGE=sdu-apex-autodrive:practice-physx-odom-20261016 \
SDU_APEX_MAP_YAML="$PWD/f1tenth_planning/maps/autodrive_practice_20260924_b.yaml" \
SDU_APEX_TRAJECTORY_FILE="$PWD/f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv" \
./tools/start_dev.sh
```

In separate terminals, start the focused recorder, the completion/collision
watcher, and then the practice simulator in batch mode:

```bash
SDU_APEX_IMAGE=sdu-apex-autodrive:practice-physx-odom-20261016 \
SDU_APEX_RUN_ID=mpc_horizon_run SDU_APEX_RECORD_PROFILE=mpc_open_loop_test \
  ./tools/record_debug_bag.sh
./tools/watch_sim_run.sh sdu_apex_autodrive_dev sdu_apex_sim_practice \
  rec_mpc_horizon_run mpc-horizon 60
SDU_APEX_SIM_TRACK=practice SDU_APEX_SIM_MODE=batchmode \
  ./tools/start_simulator.sh
```

The watcher aborts on any collision or execution-state fault, and stops the
simulator shortly after the 750 ms horizon has been sampled. Analyze a
completed capture with:

```bash
python3 tools/evaluate_mpc_bag.py live_runs/mpc_horizon_run/run
```

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
