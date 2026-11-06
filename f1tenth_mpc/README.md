# f1tenth_mpc

The controller: a 30-stage, 40 Hz sequential quadratic program over the
source-derived Unity vehicle model; see `docs/MPC.md` for the formulation, the
measured model accuracy and the results.

* `src/` and `include/` - everything that runs live: `mpc_source_sqp`,
  `mpc_source_qp`, the Unity source plant and vehicle model, reference and state
  handling, and the ROS node.
* `util/` - development only: the diagnostics publisher, the frozen-problem
  export and the solver benchmark. Nothing in the control path depends on the
  benchmark tools.
* `config/` - the practice and competition profiles.
* `test/` - unit and end-to-end tests (`colcon test`).

Output: `sdu_apex_msgs/DriveCommand` (steering angle in radians, normalized
throttle in [0.001, 1]). Build:

    source /opt/ros/<distro>/setup.bash
    colcon build --merge-install --packages-up-to f1tenth_mpc
