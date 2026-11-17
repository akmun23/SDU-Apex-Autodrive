# f1tenth_control

The LiDAR follow-the-gap controller drives a slow mapping lap while SLAM
builds the occupancy grid.

The node subscribes to `/autodrive/roboracer_1/lidar` and publishes
`sdu_apex_msgs/DriveCommand` on `/mpc/drive_command`, where the actuator interface picks
it up. Steering comes from the connected LiDAR gap, checked against the rear-axle vehicle
footprint. Throttle is scaled from the requested speed and capped by
`mapping_throttle_normalized`. A stop request produces zero throttle, the
simulator's brake command. The shared actuator interface rejects values below
0.001, so that stop command is not forwarded to the bridge. The mapping
supervisor stops the run on collision, stall, timeout or lap completion.

Run the mapping stack with `tools/mapping/run_iros_mapping.sh` and start the simulator
with `SDU_APEX_SIM_TRACK=iros tools/sim/start_simulator.sh`. The saved grid should be
inspected before a raceline is built from it; see [../docs/mapping.md](../docs/mapping.md).
