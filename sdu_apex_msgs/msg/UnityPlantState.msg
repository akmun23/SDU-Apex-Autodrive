std_msgs/Header header
uint32 state_schema_version
# Version 10 reports PhysX sticky-tire timers as hidden causal plant state.
# Unity WheelCollider's private native handling is not exposed by this source.
# 1 = propagated by the source-derived local PhysX plant in odometry; this is
# an estimate, not Unity ground truth or an independently observed state.
uint8 state_origin
# Bit order: longitudinal u, COM lateral v, yaw rate, wheel omega FL/FR/RL/RR,
# body height/rate, roll/rate, pitch/rate, physical steering, then suspension
# jounce FL/FR/RL/RR. Set bits are sensor-corrected at this epoch; unset bits
# remain source-model estimates.
uint32 sensor_corrected_state_mask
bool suspension_history_initialized
float64[4] low_forward_speed_time_s
float64[4] low_side_speed_time_s
# Named body-motion channels at this exact odometry epoch. The lateral
# velocities are both exposed because the source plant uses COM coordinates
# while the odometry body frame is referenced at the rear axle.
float64 longitudinal_velocity_mps
float64 lateral_velocity_com_mps
float64 lateral_velocity_rear_axle_mps
float64 yaw_rate_radps
float64 gyro_yaw_rate_radps
# Rear-axle position predicted by the 1 ms public-PhysX source model, relative
# to its first odometry epoch. Diagnostic only; /odom pose uses the
# sensor-corrected body-velocity integration because this pose model has shown
# greater drift on the held-out practice capture.
float64 physx_model_rear_axle_x_m
float64 physx_model_rear_axle_y_m
# Latest normalized actuator values observed by odometry at this epoch.
float64 steering_command_normalized
float64 throttle_command_normalized
# Two command samples currently in the vehicle's transport/actuation pipeline.
# Slot 1 is consumed by the next 25 ms plant step; slot 2 follows it.
float64 delayed_steering_command_1_normalized
float64 delayed_steering_command_2_normalized
float64 delayed_throttle_command_1_normalized
float64 delayed_throttle_command_2_normalized
bool actuator_feedback_valid
float64 steering_feedback_rad
float64 throttle_feedback_normalized
# Complete causal plant state at this epoch. Wheel order is FL, FR, RL, RR.
float64[4] wheel_omega_radps
float64[4] wheel_rotation_angle_rad
float64 body_height_m
# Unity Rigidbody world-Y velocity; not body-frame heave velocity.
float64 body_height_rate_mps
float64 roll_rad
float64 roll_rate_radps
float64 pitch_rad
float64 pitch_rate_radps
float64 physical_steering_rad
float64[4] suspension_jounce_m
# Per-wheel slip/load/force/rate fields are the latest completed 1 ms PhysX
# solver outputs in this packet. The observer corrects selected state channels
# after that solve; these diagnostic outputs are not recomputed without another
# physics step. The MPC rollout consumes the corrected mechanical state above.
float64[4] longitudinal_slip
# Unity WheelHit.sidewaysSlip sign convention; opposite the internal
# body-left-positive contact-velocity coordinate used to calculate tire force.
float64[4] lateral_slip
float64[4] normal_load_n
float64[4] suspension_force_n
float64[4] suspension_jounce_rate_mps
float64[4] tire_longitudinal_force_n
float64[4] tire_lateral_force_n
# Gravity-compensated IMU inertial acceleration in the vehicle body frame.
# Values are the latest 1 ms Unity FixedUpdate sample, not a 25 ms average.
float64 longitudinal_acceleration_mps2
float64 lateral_acceleration_mps2
# Derived yaw-rate change from the same 1 ms source-plant sample interval.
float64 yaw_acceleration_radps2
