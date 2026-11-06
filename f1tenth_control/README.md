# Follow-The-Gap mapping controller

This package contains the restored LiDAR Follow-The-Gap controller used to
explore an unknown simulator track while a map is being built. It is not the
production racing controller. It subscribes to
`/autodrive/roboracer_1/lidar` and publishes `sdu_apex_msgs/msg/DriveCommand`
to `/mpc/drive_command`, which the existing actuator interface forwards.

FTG steering comes from the historical gap-selection algorithm. Mapping
throttle is scheduled proportionally from FTG's requested speed, capped by
`mapping_throttle_normalized`; no legacy speed-controller node is used. A
zero-speed emergency request maps to zero throttle (Unity's hard-brake mode).

Use `tools/run_iros_mapping.sh` to launch the mapping stack and
`SDU_APEX_SIM_TRACK=iros tools/start_simulator.sh` for the IROS competition
simulator. The mapping supervisor records the run and saves the grid
automatically; inspect it before creating a raceline, as described in
`docs/IROS_TRACK_MAPPING.md`. The final competition launch remains separate
and does not run FTG or use simulator ground truth.
