# AutoDRIVE runtime integration

This package contains the 40 Hz bridge wrapper and the direct actuator
interface. The maintained launch is `controller.launch.py` for development
and `competition.launch.py` for the fixed competition image. Both run the
same PhysX odometry, EKF/AMCL localization, and throttle-output MPC.

The MPC actuator contract is `sdu_apex_msgs/DriveCommand`: steering angle in
radians and normalized throttle in `[0, 1]`. There is no speed-controller or
Ackermann target-speed path. Use `tools/verify_runtime_topic_policy.py` before
starting either stack.
