# f1tenth_mpc

The racing controller solves a 30-stage problem at 40 Hz with sequential
quadratic programming over a vehicle model derived from Unity source. The
formulation, solver details, measured model accuracy
and limits are in [../docs/mpc.md](../docs/mpc.md).

* `src/`, `include/`: everything that runs live. `mpc_source_sqp` and `mpc_source_qp` are the
  solver. `unity_source_plant` and `vehicle_model` implement the plant.
  `mpc_reference` and
  `mpc_state_synchronizer` handle the reference and state. `mpc_cycle.{h,cpp}`
  drives each solver cycle. The ROS node class is declared in
  `src/mpc_controller_node.hpp`.
* `config/`: the practice and competition parameter profiles.
* `test/`: unit and end-to-end tests (`ctest`).
* `util/`: development tools only, such as the frozen-problem export and the solver benchmark.
  Nothing in the control path depends on them; see [util/README.md](util/README.md).

Output is `sdu_apex_msgs/DriveCommand`: steering angle in radians and normalized throttle in
[0.001, 1].

```bash
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-up-to f1tenth_mpc
ctest --test-dir build/f1tenth_mpc --output-on-failure
```
