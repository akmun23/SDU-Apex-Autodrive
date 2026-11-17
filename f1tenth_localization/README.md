# f1tenth_localization

Three ROS 2 nodes estimate the car state; an offline tool replays sensor recordings.

| Executable | Role |
|---|---|
| `sensor_odometry_node` | Builds odometry from the IMU, wheel encoders and the steering and throttle feedback, and propagates the source-derived vehicle model between sensor packets. Publishes `/odom` and the typed `sdu_apex_msgs/UnityPlantState` on `/odom/unity_plant_state`. |
| `ekf_localization_node` | Extended Kalman filter over the odometry. |
| `gpu_amcl_cpp_node` | CUDA particle filter that corrects the pose against the LiDAR occupancy map and publishes `/current_map_pose`. Built only when `BUILD_GPU_AMCL=ON` (the default). |
| `odometry_replay` | Replays recorded sensor data through the odometry pipeline offline. |

Source layout: `src/` and `include/f1tenth_localization/` hold the odometry pipeline
(`odometry_pipeline`, `odometry_observer`, packet assembly, bridge request command history,
wheel-encoder model, `plant_odometry_runtime` and `plant_actuation`); `gpu_amcl_cpp/` holds the AMCL and EKF nodes with their CUDA
kernels; `config/` holds the parameter files; `test/` holds the C++ unit tests.

None of the nodes subscribe to simulator ground truth, collision or lap topics.

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-up-to f1tenth_localization \
  --cmake-args -DBUILD_GPU_AMCL=OFF      # drop the flag to build the CUDA AMCL
ctest --test-dir build/f1tenth_localization --output-on-failure
```

Design notes: [../docs/localization.md](../docs/localization.md).
