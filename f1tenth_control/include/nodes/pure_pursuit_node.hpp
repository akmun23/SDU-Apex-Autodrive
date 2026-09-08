#ifndef F1TENTH_CONTROL_PURE_PURSUIT_NODE_HPP_
#define F1TENTH_CONTROL_PURE_PURSUIT_NODE_HPP_

/**
 * @file pure_pursuit_node.hpp
 * @brief ROS2 node wrapper for the Pure Pursuit path-following controller.
 * @details Fuses encoder/IMU odometry (velocity) with the propagated CUDA
 *          AMCL map pose.
 *          Applies command-side rate limiting on steering and acceleration.
 *          Follows the explicitly selected planning trajectory for the race.
 *          Commands are shaped only by configured actuator-rate limits.
 * @dependencies pure_pursuit.hpp, rclcpp, nav_msgs, ackermann_msgs, geometry_msgs
 */

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "algorithms/pure_pursuit.hpp"
#include <memory>
#include <mutex>
#include <algorithm>
#include <cmath>

namespace f1tenth_control {

/**
 * @brief ROS2 Node for Pure Pursuit path following
 * 
 * This node:
 * - Loads a pre-computed racing line trajectory from CSV
 * - Subscribes to current map-frame /current_map_pose and local /odom state
 * - Publishes physical-unit speed commands to /cmd/speed
 * - Supports dynamic parameter reconfiguration
 * 
 * Topics:
 *   Subscriptions:
 *     - /current_map_pose (geometry_msgs/PoseWithCovarianceStamped): Current map pose
 *     - /odom (nav_msgs/Odometry): Encoder/IMU velocity and odometry
 *     - /odom (nav_msgs/Odometry): Control event and longitudinal state
 *   
 *   Publications:
 *     - /cmd/speed (ackermann_msgs/AckermannDriveStamped): Speed commands
 * 
 * @param trajectory_file Path to CSV trajectory file
 * @param min_lookahead Minimum lookahead distance [m]
 * @param max_lookahead Maximum lookahead distance [m]
 * @param lookahead_gain Velocity-proportional lookahead gain
 * 
 */
class PurePursuitNode : public rclcpp::Node {
public:
    /**
     * @brief Construct a Pure Pursuit node instance with specified options.
     * @param options ROS2 node options controlling parameters and execution behavior.
     * @return None.
     */
    explicit PurePursuitNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    // Controller
    std::unique_ptr<PurePursuit> controller_;   // Pure Pursuit controller instance
    PurePursuitConfig config_;                  // Active controller configuration parameters
    std::mutex controller_mutex_;               // Protects access to controller and config for thread safety
    
    // State
    VehicleState current_state_;    // Current vehicle state (pose, velocity, etc.)
    std::mutex state_mutex_;        // Protects access to current_state_ for thread safety
    bool trajectory_loaded_{false}; // Whether a trajectory has been successfully loaded into the controller
    bool trajectory_aligned_{false}; // Whether the cyclic seam matches the startup pose
    bool startup_alignment_validated_{false}; // Whether pose/path heading passed the startup gate
    
    // ROS2 Communication
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;                         // Subscription for odometry messages
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;   // Subscription for pose estimate messages
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steering_feedback_sub_;
    
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;    // Publisher for drive commands
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diagnostics_pub_;
    std::mutex control_mutex_;             // Prevent overlapping sensor-triggered updates
    
    double last_cmd_steering_{0.0};         // Last commanded steering angle for rate limiting
    double last_cmd_speed_{0.0};            // Last commanded speed for rate limiting
    bool cmd_history_initialized_{false};   // Whether the command history has been initialized for rate limiting
    rclcpp::Time last_cmd_time_;            // Timestamp of the last published command for rate limiting
    
    // Parameters
    std::string trajectory_file_;           // Path to trajectory CSV file
    std::string odom_topic_{"/odom"};
    std::string pose_topic_{"/current_map_pose"};
    std::string command_topic_{"/cmd/speed"};
    std::string path_frame_{"map"};
    std::string command_frame_{"base_link"};
    bool pose_received_{false};             // Whether a valid pose estimate has been received
    bool odom_received_{false};             // Whether a valid odometry message has been received
    int localization_good_updates_{0};      // Consecutive covariance-qualified poses
    rclcpp::Time last_pose_time_;           // Timestamp of the last received pose message
    rclcpp::Time last_odom_time_;           // Timestamp of the last received odometry message
    rclcpp::Time last_pose_stamp_;          // Sensor timestamp of the last accepted pose
    rclcpp::Time last_odom_stamp_;           // Sensor timestamp of the last odometry message
    rclcpp::Time last_steering_feedback_time_;
    bool steering_feedback_received_{false};
    double steering_feedback_angle_{0.0};
    double pose_timeout_s_{0.1};            // Timeout for considering pose data stale [s]
    double odom_timeout_s_{0.2};            // Timeout for considering odometry data stale [s]
    double state_extrapolation_max_s_{0.12}; // Allowed-sensor pose-to-command latency [s]
    double control_rate_hz_{20.0};           // Nominal odometry cadence; actual rate follows /odom
    double localization_covariance_xy_max_{0.25};   // Maximum AMCL x/y variance [m^2]
    double localization_covariance_yaw_max_{0.12};  // Maximum AMCL yaw variance [rad^2]
    int localization_required_updates_{5};          // Consecutive qualified poses before drive
    double max_speed_{22.88};               // [m/s] Documented simulator envelope
    double max_steering_rate_{3.2};         // [rad/s] AutoDRIVE documented limit
    double max_accel_cmd_{3.0};             // [m/s^2] Maximum acceleration command for speed ramping
    double max_decel_cmd_{8.0};             // [m/s^2] Maximum deceleration command for speed ramping
    std::string steering_feedback_topic_{"/autodrive/roboracer_1/steering"};
    double steering_feedback_timeout_s_{0.25};
    double steering_feedback_lead_gain_{0.25};
    double startup_path_max_distance_m_{0.80};
    double startup_path_heading_tolerance_rad_{0.75};
    
    // Parameter handling
    /**
     * @brief Declare ROS parameters for the Pure Pursuit node.
     * @return None.
     */
    void declareParameters();

    /**
     * @brief Load parameters from the ROS parameter server into internal config.
     * @return None.
     */
    void loadParameters();

    /**
    * @brief Callback for dynamic parameter updates.
    * @param parameters Vector of parameters that were updated.
    * @return Result indicating whether the parameter update was successful.
    */
    rcl_interfaces::msg::SetParametersResult parametersCallback(
        const std::vector<rclcpp::Parameter>& parameters
    );
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;  // Handle for parameter callback registration
    
    // Callbacks

    /**
     * @brief Callback for odometry messages.
     * @param msg Shared pointer to the received Odometry message.
     * @return None.
     */
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

    /**
     * @brief Callback for pose estimate messages.
     * @param msg Shared pointer to the received PoseWithCovarianceStamped message.
     * @return None.
     */
    void poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

    /** Store the official raw-radian steering actuator feedback. */
    void steeringFeedbackCallback(const std_msgs::msg::Float32::ConstSharedPtr msg);

    /**
     * @brief Main control loop callback for one odometry event.
     * Executes one cycle of Pure Pursuit control using the newest allowed
     * localization and odometry state extrapolated to the odometry timestamp.
     * @return None.
     */
    void controlLoop(const rclcpp::Time & event_stamp);
    
    // Publishing
    /**
     * @brief Publish drive command based on computed steering and speed.
     * This function constructs and publishes an AckermannDriveStamped command.
     * @param steering Desired steering angle in radians.
     * @param speed Desired speed in meters per second.
     * @return None.
     */
    void publishDriveCommand(double steering, double speed, double acceleration = 0.0);

    /** Publish one diagnostic record for the exact control event. */
    void publishDiagnostics(const rclcpp::Time& event_stamp,
                            const VehicleState& state,
                            const PurePursuitOutput& output,
                            double command_steering,
                            double command_speed);

    // Helpers

    /**
     * @brief Load trajectory from CSV file into the Pure Pursuit controller.
     * This function reads a trajectory from the specified CSV file, parses it into the expected format, and loads it into the Pure Pursuit controller instance. It also updates the trajectory_loaded_ flag based on whether the loading was successful.
     * @return True if the trajectory was successfully loaded and parsed, false otherwise.
     */
    bool loadTrajectory();
};

}  // namespace f1tenth_control

#endif  // F1TENTH_CONTROL_PURE_PURSUIT_NODE_HPP_
