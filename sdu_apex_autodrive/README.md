# sdu_apex_autodrive

Integration with the AutoDRIVE simulator.

| Executable | Role |
|---|---|
| `autodrive_bridge_40hz` | Wraps the official AutoDRIVE ROS bridge, paces requests at 40 Hz and publishes each associated sensor response. |
| `actuator_interface` | Converts `sdu_apex_msgs/DriveCommand` into the simulator's steering and throttle inputs. |
| `csv_command_replay` | Development only. Follows a raceline CSV with its recorded steering and throttle, using simulator ground truth. Not part of the competition graph. |
| `mapping_supervisor` | Records a mapping run and saves the occupancy grid. |

Launch files:

* `competition.launch.py` starts the fixed competition graph: bridge, odometry, EKF, AMCL,
  MPC and actuator interface. Map and raceline come from `SDU_APEX_MAP_REL` and
  `SDU_APEX_TRAJECTORY_REL` (competition track by default). The MPC always
  uses `mpc_competition.yaml`.
* `controller.launch.py` starts the same graph for development and selects the MPC profile
  from the map name.
* `mapping.launch.py` starts follow-the-gap driving with SLAM.

Run `tools/verify_runtime_topic_policy.py` before starting a stack; it checks that the
autonomous launch files and nodes avoid restricted inputs (see
[../docs/architecture.md](../docs/architecture.md)).
