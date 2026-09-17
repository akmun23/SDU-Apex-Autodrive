// One ROS adapter around the BachelorProject MPC core.  It deliberately uses
// the same legal pose/odometry and Ackermann target-speed boundary as the
// established Pure Pursuit controller.  Unity truth/contact data is absent.

#include "mpc.h"

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace f1tenth_mpc {
namespace {

constexpr double kPi = 3.14159265358979323846;

double normalize_angle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

double clamp(double value, double lower, double upper)
{
    return std::min(std::max(value, lower), upper);
}

struct Waypoint {
    double s{};
    double x{};
    double y{};
    double heading{};
    double curvature{};
    double speed{};
    double left_bound{1.0};
    double right_bound{1.0};
};

struct PoseState {
    double x{};
    double y{};
    double yaw{};
    rclcpp::Time received{0, 0, RCL_ROS_TIME};
    bool valid{false};
};

struct MotionState {
    double u{};
    double v{};
    double yaw_rate{};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    bool valid{false};
};

}  // namespace

class MpcControllerNode final : public rclcpp::Node {
public:
    explicit MpcControllerNode(const rclcpp::NodeOptions & options)
        : Node("mpc_controller_node", options)
    {
        enabled_ = declare_parameter<bool>("enabled", false);
        odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
        pose_topic_ = declare_parameter<std::string>(
            "pose_topic", "/current_map_pose");
        command_topic_ = declare_parameter<std::string>("command_topic", "/cmd/speed");
        path_frame_ = declare_parameter<std::string>("path_frame", "map");
        command_frame_ = declare_parameter<std::string>("command_frame", "base_link");
        trajectory_file_ = declare_parameter<std::string>("trajectory_file", "");
        max_speed_mps_ = clamp(declare_parameter<double>("max_speed_mps", 16.0), 0.0, 16.0);
        startup_speed_mps_ = clamp(
            declare_parameter<double>("startup_speed_mps", 1.5), 0.0, max_speed_mps_);
        startup_ramp_laps_ = std::max(
            0.0, declare_parameter<double>("startup_ramp_laps", 1.0));
        pose_timeout_s_ = std::max(
            0.01, declare_parameter<double>("pose_timeout_s", 0.30));
        source_dt_min_s_ = std::max(
            0.001, declare_parameter<double>("source_dt_min_s", 0.015));
        source_dt_max_s_ = std::max(
            source_dt_min_s_, declare_parameter<double>("source_dt_max_s", 0.035));
        startup_path_max_distance_m_ = std::max(
            0.0, declare_parameter<double>("startup_path_max_distance_m", 0.80));
        startup_path_heading_tolerance_rad_ = std::max(
            0.0, declare_parameter<double>("startup_path_heading_tolerance_rad", 0.75));

        command_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            command_topic_, rclcpp::QoS(10));
        pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            pose_topic_, rclcpp::QoS(10),
            std::bind(&MpcControllerNode::pose_callback, this, std::placeholders::_1));
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, rclcpp::QoS(20),
            std::bind(&MpcControllerNode::odom_callback, this, std::placeholders::_1));

        mpc_initialize();
        trajectory_loaded_ = load_trajectory(trajectory_file_);
        if (!trajectory_loaded_) {
            RCLCPP_ERROR(get_logger(),
                "MPC has no valid trajectory; it will publish only safe stops");
        }
        if (!enabled_) {
            RCLCPP_WARN(get_logger(),
                "MPC command output is inhibited until the source-command stage-map gate is approved");
        }
    }

private:
    bool load_trajectory(const std::string & path)
    {
        if (path.empty()) {
            return false;
        }
        std::ifstream input(path);
        if (!input.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open MPC trajectory: %s", path.c_str());
            return false;
        }

        std::vector<Waypoint> parsed;
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') {
                continue;
            }
            std::stringstream stream(line);
            std::string token;
            std::vector<double> values;
            bool valid = true;
            while (std::getline(stream, token, ',')) {
                try {
                    values.push_back(std::stod(token));
                } catch (...) {
                    valid = false;
                    break;
                }
            }
            if (!valid || values.size() < 6) {
                continue;
            }
            Waypoint point;
            point.s = values[0];
            point.x = values[1];
            point.y = values[2];
            point.heading = values[3];
            point.curvature = values[4];
            point.speed = values[5];
            if (values.size() >= 9) {
                point.left_bound = values[7];
                point.right_bound = values[8];
            }
            if (!std::isfinite(point.s) || !std::isfinite(point.x) ||
                !std::isfinite(point.y) || !std::isfinite(point.heading) ||
                !std::isfinite(point.curvature) || !std::isfinite(point.speed)) {
                continue;
            }
            if (!std::isfinite(point.left_bound) || point.left_bound <= 0.0) {
                point.left_bound = 1.0;
            }
            if (!std::isfinite(point.right_bound) || point.right_bound <= 0.0) {
                point.right_bound = 1.0;
            }
            parsed.push_back(point);
        }
        if (parsed.size() > 2) {
            const auto & first = parsed.front();
            const auto & last = parsed.back();
            if (std::hypot(first.x - last.x, first.y - last.y) < 1.0e-4) {
                parsed.pop_back();
            }
        }
        if (parsed.size() < 3) {
            return false;
        }
        trajectory_ = std::move(parsed);
        track_length_m_ = trajectory_.back().s;
        if (!(track_length_m_ > 1.0e-3)) {
            track_length_m_ = 0.0;
            for (std::size_t index = 0; index < trajectory_.size(); ++index) {
                const auto & a = trajectory_[index];
                const auto & b = trajectory_[(index + 1) % trajectory_.size()];
                track_length_m_ += std::hypot(a.x - b.x, a.y - b.y);
            }
        }
        mean_waypoint_spacing_m_ = track_length_m_ /
            static_cast<double>(trajectory_.size());
        if (!(mean_waypoint_spacing_m_ > 1.0e-5)) {
            return false;
        }
        RCLCPP_INFO(get_logger(), "Loaded MPC trajectory with %zu points (%.2f m)",
                    trajectory_.size(), track_length_m_);
        return true;
    }

    std::size_t nearest_index(const PoseState & pose)
    {
        const std::size_t count = trajectory_.size();
        const std::size_t search = progress_initialized_ ?
            std::min(count, std::size_t{160}) : count;
        const std::size_t start = progress_initialized_ ?
            (last_closest_index_ + count - std::min(count - 1, std::size_t{24})) % count : 0;
        double best_distance = std::numeric_limits<double>::infinity();
        std::size_t best = last_closest_index_;
        for (std::size_t offset = 0; offset < search; ++offset) {
            const std::size_t index = (start + offset) % count;
            const auto & point = trajectory_[index];
            const double heading_error = std::abs(normalize_angle(point.heading - pose.yaw));
            if (heading_error > 0.5 * kPi) {
                continue;
            }
            const double distance = std::hypot(point.x - pose.x, point.y - pose.y);
            if (distance < best_distance) {
                best_distance = distance;
                best = index;
            }
        }
        if (!std::isfinite(best_distance)) {
            for (std::size_t index = 0; index < count; ++index) {
                const auto & point = trajectory_[index];
                const double distance = std::hypot(point.x - pose.x, point.y - pose.y);
                if (distance < best_distance) {
                    best_distance = distance;
                    best = index;
                }
            }
        }
        closest_distance_m_ = best_distance;
        return best;
    }

    void update_progress(std::size_t closest)
    {
        if (!progress_initialized_) {
            progress_initialized_ = true;
            last_closest_index_ = closest;
            return;
        }
        double delta = trajectory_[closest].s - trajectory_[last_closest_index_].s;
        if (delta < -0.5 * track_length_m_) {
            delta += track_length_m_;
        }
        if (delta > 0.0 && delta < 0.5 * track_length_m_) {
            startup_progress_m_ += delta;
        }
        last_closest_index_ = closest;
    }

    double active_speed_ceiling() const
    {
        if (startup_ramp_laps_ <= 1.0e-6 || track_length_m_ <= 1.0e-6) {
            return max_speed_mps_;
        }
        const double fraction = clamp(
            startup_progress_m_ / (startup_ramp_laps_ * track_length_m_), 0.0, 1.0);
        return startup_speed_mps_ + (max_speed_mps_ - startup_speed_mps_) * fraction;
    }

    void make_reference(std::size_t closest, double speed_ceiling,
                        TrajectoryReferencePoint_t reference[PREDICTION_HORIZON]) const
    {
        std::size_t index = closest;
        for (int stage = 0; stage < PREDICTION_HORIZON; ++stage) {
            const auto & point = trajectory_[index];
            const double speed = clamp(point.speed, 0.0, speed_ceiling);
            TrajectoryReferencePoint_t stage_reference{};
            stage_reference.reference_lateral_error = 0.0f;
            stage_reference.reference_heading_error = 0.0f;
            stage_reference.reference_velocity = static_cast<float>(speed);
            stage_reference.reference_lateral_velocity = 0.0f;
            stage_reference.reference_yaw_rate =
                static_cast<float>(point.curvature * speed);
            stage_reference.path_curvature = static_cast<float>(point.curvature);
            stage_reference.left_wall_bound = static_cast<float>(point.left_bound);
            stage_reference.right_wall_bound = static_cast<float>(point.right_bound);
            reference[stage] = stage_reference;
            const auto advance_m = std::max(0.25, speed * TIME_STEP_SECONDS);
            const auto advance_count = static_cast<std::size_t>(std::max(
                1.0, std::round(advance_m / mean_waypoint_spacing_m_)));
            index = (index + advance_count) % trajectory_.size();
        }
    }

    void publish_command(double steering, double speed)
    {
        auto command = ackermann_msgs::msg::AckermannDriveStamped();
        command.header.stamp = now();
        command.header.frame_id = command_frame_;
        command.drive.steering_angle = static_cast<float>(steering);
        command.drive.speed = static_cast<float>(speed);
        // The actuator interface owns the safe speed-to-throttle conversion.
        command.drive.acceleration = 0.0f;
        command_pub_->publish(command);
    }

    void publish_stop(const char * reason)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", reason);
        publish_command(0.0, 0.0);
    }

    void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr message)
    {
        if (!message->header.frame_id.empty() && message->header.frame_id != path_frame_) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                "Rejected MPC pose frame '%s'; expected '%s'",
                message->header.frame_id.c_str(), path_frame_.c_str());
            return;
        }
        const auto & q = message->pose.pose.orientation;
        const double yaw = std::atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        if (!std::isfinite(message->pose.pose.position.x) ||
            !std::isfinite(message->pose.pose.position.y) || !std::isfinite(yaw)) {
            return;
        }
        std::lock_guard<std::mutex> lock(state_mutex_);
        pose_.x = message->pose.pose.position.x;
        pose_.y = message->pose.pose.position.y;
        pose_.yaw = yaw;
        pose_.received = now();
        pose_.valid = true;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr message)
    {
        PoseState pose;
        MotionState motion;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            motion_.u = message->twist.twist.linear.x;
            motion_.v = message->twist.twist.linear.y;
            motion_.yaw_rate = message->twist.twist.angular.z;
            motion_.stamp = rclcpp::Time(message->header.stamp, get_clock()->get_clock_type());
            if (motion_.stamp.nanoseconds() == 0) {
                motion_.stamp = now();
            }
            motion_.valid = std::isfinite(motion_.u) && std::isfinite(motion_.v) &&
                std::isfinite(motion_.yaw_rate);
            pose = pose_;
            motion = motion_;
        }

        if (!enabled_) {
            publish_stop("MPC is command-inhibited pending stage-map acceptance");
            return;
        }
        if (!trajectory_loaded_ || !pose.valid || !motion.valid) {
            publish_stop("MPC requires valid trajectory, map pose and odometry");
            return;
        }
        if ((now() - pose.received).seconds() > pose_timeout_s_) {
            publish_stop("MPC map pose is stale");
            return;
        }
        if (last_source_stamp_.nanoseconds() != 0) {
            const double source_dt = (motion.stamp - last_source_stamp_).seconds();
            if (source_dt < source_dt_min_s_ || source_dt > source_dt_max_s_) {
                mpc_reset();
                target_speed_initialized_ = false;
                last_source_stamp_ = motion.stamp;
                publish_stop("MPC source timing fault; state was reset");
                return;
            }
        }
        const double source_dt = last_source_stamp_.nanoseconds() == 0 ?
            TIME_STEP_SECONDS : (motion.stamp - last_source_stamp_).seconds();
        last_source_stamp_ = motion.stamp;

        const std::size_t closest = nearest_index(pose);
        const auto & point = trajectory_[closest];
        const double heading_error = normalize_angle(pose.yaw - point.heading);
        if (!startup_path_validated_) {
            if (closest_distance_m_ > startup_path_max_distance_m_ ||
                std::abs(heading_error) > startup_path_heading_tolerance_rad_) {
                publish_stop("MPC startup path gate rejected pose");
                return;
            }
            startup_path_validated_ = true;
        }
        update_progress(closest);
        const double speed_ceiling = active_speed_ceiling();
        TrajectoryReferencePoint_t reference[PREDICTION_HORIZON];
        make_reference(closest, speed_ceiling, reference);

        const double dx = pose.x - point.x;
        const double dy = pose.y - point.y;
        FrenetState_t state{};
        state.flat_error = static_cast<float>(
            -std::sin(point.heading) * dx + std::cos(point.heading) * dy);
        state.fhead_error = static_cast<float>(heading_error);
        state.flong_vel = static_cast<float>(std::max(0.0, motion.u));
        state.flat_vel = static_cast<float>(motion.v);
        state.fyaw_rate = static_cast<float>(motion.yaw_rate);
        mpc_set_previous_command_with_dt(
            &last_control_, static_cast<float>(source_dt), 0.0f, 0);
        MpcSolverResult_t result{};
        const MpcSolverStatus_t status = mpc_compute_optimal_control(
            &state, reference, &result);
        if (status != MPC_STATUS_SUCCESS && status != MPC_STATUS_MAXIMUM_ITERATIONS_REACHED) {
            publish_stop("MPC solver did not produce a usable command");
            return;
        }

        const double race_speed = clamp(point.speed, 0.0, speed_ceiling);
        if (!target_speed_initialized_) {
            target_speed_mps_ = clamp(std::max(0.0, motion.u), 0.0, race_speed);
            target_speed_initialized_ = true;
        }
        target_speed_mps_ = clamp(
            target_speed_mps_ + result.optimal_control.target_speed_rate * source_dt,
            0.0, race_speed);
        last_control_ = result.optimal_control;
        publish_command(last_control_.steer_ang, target_speed_mps_);
    }

    bool enabled_{};
    bool trajectory_loaded_{};
    bool startup_path_validated_{};
    bool progress_initialized_{};
    bool target_speed_initialized_{};
    std::string odom_topic_;
    std::string pose_topic_;
    std::string command_topic_;
    std::string path_frame_;
    std::string command_frame_;
    std::string trajectory_file_;
    double max_speed_mps_{};
    double startup_speed_mps_{};
    double startup_ramp_laps_{};
    double pose_timeout_s_{};
    double source_dt_min_s_{};
    double source_dt_max_s_{};
    double startup_path_max_distance_m_{};
    double startup_path_heading_tolerance_rad_{};
    double track_length_m_{};
    double mean_waypoint_spacing_m_{};
    double startup_progress_m_{};
    double closest_distance_m_{};
    double target_speed_mps_{};
    std::size_t last_closest_index_{};
    rclcpp::Time last_source_stamp_{0, 0, RCL_ROS_TIME};
    ControlInput_t last_control_{};
    std::vector<Waypoint> trajectory_;
    PoseState pose_;
    MotionState motion_;
    std::mutex state_mutex_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr command_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
};

}  // namespace f1tenth_mpc

RCLCPP_COMPONENTS_REGISTER_NODE(f1tenth_mpc::MpcControllerNode)
