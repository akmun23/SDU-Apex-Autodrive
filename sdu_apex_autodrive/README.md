# AutoDRIVE integration

This package owns the official bridge wrapper, legal sensor-derived odometry
interface, actuator interface, and the three maintained local launch modes:

- `competition.launch.py`: fixed MPC-only competition path;
- `controller.launch.py`: local MPC/PP/FTG selection;
- `mapping.launch.py`: development-only FTG mapping from simulator pose and
  LiDAR; team odometry still runs separately for comparison and offline analysis.

The package contains no simulator-truth consumers, shadow controller, model
identification recorder, or generated analysis data. Use
`tools/verify_runtime_topic_policy.py` before starting a stack.
