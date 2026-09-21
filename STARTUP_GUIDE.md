# AutoDRIVE startup

This repository has one runtime raceline and one production MPC profile:

```text
f1tenth_planning/trajectories/autodrive_mintime_sim_5p0_dense/
  autodrive_mintime_raceline.csv
f1tenth_mpc/config/mpc_iros_2026_competition.yaml
```

FTG and Pure Pursuit remain available as the mapping and baseline controllers.
They are selected explicitly; the default controller is MPC.

## Competition image

Build and run the pinned competition image without a source mount or Compose:

```bash
docker build -t sdu-apex-autodrive:iros-2026 .
docker run --name autodrive_roboracer_api --rm -it \
  --network=host --ipc=host --privileged --gpus all \
  sdu-apex-autodrive:iros-2026
```

The fixed entrypoint starts the official API bridge, legal sensor odometry,
EKF, AMCL, MPC, and actuator chain. MPC is delayed while AMCL initializes.
There is no shadow controller, simulator-truth recorder, RViz, or parameter
overlay in the competition launch.

## Local simulator development

Build/run the ROS container from this repository while the externally managed
AutoDRIVE simulator is running:

```bash
./tools/start_dev.sh
```

The default is live MPC. The maintained comparison modes are:

```bash
SDU_APEX_CONTROLLER=pure_pursuit ./tools/start_dev.sh
SDU_APEX_CONTROLLER=ftg ./tools/start_dev.sh
```

Stop the development container with:

```bash
./tools/stop_autodrive.sh
```

`AUTODRIVE_REBUILD=0` reuses the already built image. The simulator is not
part of this repository and this cleanup does not modify its physics, timing,
scene, or assets.

## Mapping

FTG mapping remains a separate maintained launch and uses only the legal
LiDAR/encoder/IMU-derived data path:

```bash
ros2 launch sdu_apex_autodrive mapping.launch.py
```

The mapping launch writes a new map under `f1tenth_planning/maps/`; mapping
outputs are not used as racing racelines.

## Runtime boundary

The autonomous stack consumes the allowed LiDAR, IMU, and encoder topics plus
team-derived `/odom`, `/ekf_odom`, and `/current_map_pose`. It does not
subscribe to simulator pose/IPS, simulator odometry, collision/reset, lap,
result, or absolute `/tf` topics. Team TF is isolated on `/sdu/tf` and
`/sdu/tf_static`.

Run the static compliance check before starting a stack:

```bash
python3 tools/verify_runtime_topic_policy.py
```

The exact minimum-time raceline generator is retained as offline source under
`SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3/`. Its default generated
checkpoints are ignored under `artifacts/raceline_generation/`; it does not
add another runtime trajectory.
