std_msgs/Header header

# Explicit direct actuator semantics.  This is deliberately not encoded in
# AckermannDrive.acceleration, whose meaning remains physical acceleration on
# the existing Pure Pursuit speed-command path.
float64 steering_angle_rad
float64 throttle_normalized
bool enable
