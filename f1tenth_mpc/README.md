# AutoDRIVE MPC

This package contains the single production MPC controller and its compact
Riccati/ADMM core. It uses only the team-derived `/odom` and
`/current_map_pose` inputs plus the canonical installed raceline. Simulator
truth, hidden Unity state, and source timestamps are not controller inputs.

The controller runs a 30-stage horizon at 40 Hz with the competition model
and weights in `config/mpc_iros_2026_competition.yaml`. There is no shadow
controller or alternate runtime model in this repository.

Build and unit-test it in the ROS 2 Humble image:

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-select f1tenth_mpc
colcon test --merge-install --packages-select f1tenth_mpc \
  --event-handlers console_direct+
```
