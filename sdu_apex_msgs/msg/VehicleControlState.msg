std_msgs/Header header

# Map-frame pose and body-frame dynamic state.  The header stamp is the
# source-time stamp of the odometry epoch that produced this message.
float64 x_map_m
float64 y_map_m
float64 yaw_map_rad
float64 body_u_mps
float64 body_v_mps
float64 yaw_rate_radps
float64 steering_angle_rad
float64 ax_mps2
float64 ay_mps2

float64 pose_xy_variance
float64 pose_yaw_variance
float64 body_u_variance
float64 body_v_variance
float64 yaw_rate_variance

bool localization_valid
bool dynamics_state_valid
bool steering_valid
uint64 reset_epoch
