# Localization and state estimation

Localization estimates body motion from sensors, filters the odometry with an EKF and corrects the map pose with a CUDA particle filter.

## Source-derived odometry

[`sensor_odometry_node`](../f1tenth_localization/src/sensor_odometry_node.cpp)
assembles encoder/IMU packets with bridge request commands and associated
steering/throttle feedback. `OdometryPipeline` advances the production source
plant by 25 ticks of 1 ms, corrects it with the encoders and IMU, and reseeds
the plant for the next packet. It publishes body motion on `/odom` and the
typed `sdu_apex_msgs/UnityPlantState` on `/odom/unity_plant_state`.
The latter includes the observer's internal vehicle state and delayed commands.
Both outputs are estimates derived from sensors and the vehicle model.
The model propagation lives in `plant_odometry_runtime`; `plant_actuation`
handles the delayed actuator inputs.

[sensor_odometry.yaml](../f1tenth_localization/config/sensor_odometry.yaml)
defines the sensor topics and geometry. The wheel radius is 0.059 m with
1,920 encoder ticks per revolution; the LiDAR is 0.2733 m ahead of the rear axle.
The competition launch disables TF publication; mapping supplies its own
`odom → base_link → lidar` tree.

`odometry_replay` runs recorded packets through the same pipeline and compares
body velocity with Unity odometry offline.
[export_odometry_replay.py](../f1tenth_localization/util/export_odometry_replay.py)
extracts input from a full-profile bag. The reported replay reproduces the
logged estimate to `1e-13 m/s`. Configuration values are checked in.

Across the reported 16 recorded runs, normal-driving forward-speed RMS error
is 0.007–0.03 m/s.
Impulsive wall contact produces errors of 0.1–1.2 m/s: the IMU's final-sample
acceleration sees the impulse but cannot integrate it. Encoder innovation has
correlation `r = 0.02` with the 25 ms speed-prediction error over 11,851
pre-collision packets. Encoder-derived speed `r*omega / (1 + kappa_model)` has
RMS error 0.22–0.56 m/s, versus 0.014–0.075 m/s for the observer on that comparison.
The observer therefore does not apply that slip-based wheel-speed correction.
Planning grip reserve must absorb disturbances that the sensors cannot observe.

## EKF and GPU AMCL

`ekf_localization_node` propagates each `/odom` sample in the local odometry
frame and publishes `/ekf_odom` and `/ekf_pose`. Its
[configuration](../f1tenth_localization/config/ekf.yaml) supplies process-noise
densities. Raw covariance sets the initial uncertainty and its floor; the filter
does not add it again on every callback. The XY time density is `0.0003 m²/s`, fitted
from a 2 m/s recording (`2.95e-4 m²/s`, 99.7% chronological-holdout ellipse coverage).
Odometry jumps above 3 m are rejected as an epoch change or teleport.

`gpu_amcl_cpp_node` combines `/ekf_odom`, LiDAR and the occupancy map and publishes
`/current_map_pose`. CUDA kernels perform particle-filter work. The
[base parameters](../f1tenth_localization/config/gpu_amcl_cpp_params.yaml) cap
KLD sampling at 5,000 particles and use 270 beams for normal updates. The
practice configuration sets scan acquisition delay to 17.5 ms; changing the
simulator player, sensor cadence or bridge
timing requires checking that calibration. The
[IROS overlay](../f1tenth_localization/config/amcl_iros.yaml) supplies track-specific settings.

AMCL propagates accepted map poses with odometry between scan corrections.
At startup, AMCL checks scan support, saved map-start priors and raceline
heading. Cluster, covariance and pose-jump checks limit the corrections it
publishes. Source-time
odometry supplies heading. Competition nodes do not consume TF or ground truth.
`BUILD_GPU_AMCL=OFF` omits the CUDA node while retaining odometry, EKF and replay;
see [development](development.md).

## MPC state consumption

`MpcStateSynchronizer` combines raw `/odom`, `/odom/unity_plant_state` and
`/current_map_pose` at compatible stamps. The MPC consumes the EKF indirectly
through AMCL, not as a replacement for the raw body-motion/plant state.
Bridge request epochs and command history support prediction to the command
publication window; [MPC](mpc.md) describes continuation during state gaps.

`MpcStateSynchronizer::set_map_pose` blends a new pose into the prediction at
its stamp with `map_pose_gain = 0.2`. A displacement above
`map_pose_reset_distance_m = 0.3` is treated as relocalization.
Against Unity truth offline, lateral map-pose SD is 1.3 cm, p99 4.0 cm and
maximum 9.2 cm; blending lowers p99 by about 20–25%. First-hairpin pose error
differs by about 0.1 cm between lap 1 and laps 3+, so it does not explain
the concentration of practice failures on the first lap.

At the first hairpin, lateral error is 0.5–0.85 cm and failed passes agree
with truth within 1–2 cm until impact. Measured corner heading error against
the reference has median 25 mrad and p99 115 mrad, versus 9 mrad planned.
Along-track pose error of 2.7 cm, 3.4 times the lateral error, shifts the
projected path heading by curvature times that error; the resulting exit-window
wobble has median 11 mrad. A separate along-track pose gain does not reduce it.

The reported competition scoring alignment translates Unity world XY by
`(+0.783, +3.158) m` into map XY, with equal yaw. A raw map/world position
offset is not
a localization error score. Ground truth remains an offline evaluation input
under the [topic policy](architecture.md#topic-policy).
