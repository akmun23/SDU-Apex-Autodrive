# sdu_apex_msgs

This ROS 2 package defines the drive command and estimated plant-state messages.

* `DriveCommand`: physical front-wheel steering target in radians and normalized
  throttle in [0, 1]. Throttle 0 is the simulator's active brake branch, not a coast
  command; the MPC publishes at least 0.001.
* `UnityPlantState`: the plant state estimated by the odometry (speeds, wheel speeds,
  body height, roll, pitch, suspension jounce, steering angle), a mask of which entries
  the sensors corrected at that epoch, and the command history the MPC needs to
  continue the plant forward.
