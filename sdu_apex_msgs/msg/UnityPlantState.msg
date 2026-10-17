std_msgs/Header header
uint32 state_schema_version
# Version 2 names the body-motion and actuator channels below explicitly.
# 1 = propagated by the source-derived local PhysX plant in odometry; this is
# an estimate, not Unity ground truth or an independently observed state.
uint8 state_origin
# Bit i marks state[i] as corrected by an available runtime sensor at this
# epoch. Unset bits remain source-model estimates.
uint32 sensor_corrected_state_mask
bool suspension_history_initialized
# Named body-motion channels at this exact odometry epoch. The lateral
# velocities are both exposed because the source plant uses COM coordinates
# while the odometry body frame is referenced at the rear axle.
float64 longitudinal_velocity_mps
float64 lateral_velocity_com_mps
float64 lateral_velocity_rear_axle_mps
float64 yaw_rate_radps
float64 gyro_yaw_rate_radps
# Latest normalized actuator values observed by odometry at this epoch.
float64 steering_command_normalized
float64 throttle_command_normalized
bool actuator_feedback_valid
float64 steering_feedback_rad
float64 throttle_feedback_normalized
# Source-plant order: u, v_COM, yaw_rate, wheel omega FL/FR/RL/RR,
# body height, world vertical speed, roll, roll rate, pitch, pitch rate,
# physical steering, suspension jounce FL/FR/RL/RR.
float64[18] state
float64[4] longitudinal_slip
float64[4] lateral_slip
float64[4] normal_load_n
float64[4] suspension_force_n
float64[4] suspension_jounce_rate_mps
float64[4] tire_longitudinal_force_n
float64[4] tire_lateral_force_n
float64 longitudinal_acceleration_mps2
float64 lateral_acceleration_mps2
float64 yaw_acceleration_radps2
