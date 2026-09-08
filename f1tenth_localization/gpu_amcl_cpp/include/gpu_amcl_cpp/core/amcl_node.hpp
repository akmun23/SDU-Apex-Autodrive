#pragma once

#include "gpu_amcl_cpp/core/particle_filter.hpp"
#include "gpu_amcl_cpp/helpers/map_utils.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <mutex>
#include <atomic>
#include <deque>
#include <Eigen/Core>

namespace gpu_amcl_cpp {

/**
 * @brief AMCL ROS 2 node — GPU-accelerated particle-filter localisation.
 *
 * Subscribes to laser scans and odometry, publishes pose estimates
 * and a particle cloud. AMCL publishes the independent map->odom correction
 * computed from its map pose and the timestamp-aligned local odometry pose.
 * The local EKF never subscribes to this AMCL output.
 */
class AmclNode : public rclcpp::Node {
public:
    explicit AmclNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    // ── Callbacks ──────────────────────────────────────────────────
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void publish_particle_cloud(const rclcpp::Time& stamp);
    void publish_pre_resample_weighted_cloud(const rclcpp::Time& stamp);
    void publish_current_map_pose(const rclcpp::Time& stamp,
                                  double odom_x,
                                  double odom_y,
                                  double odom_theta);
    void publish_scan_alignment_diagnostic(const rclcpp::Time& scan_stamp,
                                           const rclcpp::Time& matched_odom_stamp,
                                           const rclcpp::Time& bracket_before_stamp,
                                           const rclcpp::Time& bracket_after_stamp,
                                           double matched_odom_error_ms,
                                           bool accepted);
    void retry_pending_scans();

    // ── Helpers ────────────────────────────────────────────────────
    void declare_all_parameters();
    void load_parameters();
    void publish_pose(const PoseEstimate& est, const rclcpp::Time& stamp);
    std::string resolve_global_heading_trajectory_file() const;
    std::vector<ParticleFilter::TrackHeadingPoint> load_global_heading_points() const;
    double raceline_heading_near_pose(
        double x,
        double y,
        double fallback_yaw,
        const std::vector<ParticleFilter::TrackHeadingPoint>& heading_points) const;
    bool should_publish_pose_estimate(const PoseEstimate& est);
    void reset_pose_jump_gate();
    void push_odom_sample(const rclcpp::Time& stamp,
                          double x,
                          double y,
                          double theta,
                          const nav_msgs::msg::Odometry& msg);
    bool interpolate_odom_pose(const rclcpp::Time& stamp,
                               double& x,
                               double& y,
                               double& theta,
                               rclcpp::Time* matched_stamp = nullptr,
                               rclcpp::Time* bracket_before_stamp = nullptr,
                               rclcpp::Time* bracket_after_stamp = nullptr,
                               Eigen::Matrix3d* covariance = nullptr) const;
    double raceline_distance_to_pose(double x, double y) const;
    double raceline_heading_error_to_pose(double x, double y, double theta) const;

    // ── ROS I/O ────────────────────────────────────────────────────
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;

    // Publishers    
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;  // /amcl_pose
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr current_map_pose_pub_;  // /current_map_pose
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr cloud_pub_;                 // /particlecloud
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pre_resample_cloud_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr timing_pub_;                       // /amcl_timing
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr particle_count_pub_;                  // /amcl_particle_count
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr kld_diag_pub_;            // /amcl_kld_diagnostics
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gpu_timing_pub_;          // /amcl_gpu_timing
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr scan_alignment_pub_;     // /amcl_scan_alignment
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr localization_health_pub_; // /amcl_localization_health
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr scan_retry_timer_;
    
    // ── Core ───────────────────────────────────────────────────────
    ParticleFilter pf_;     // The particle filter
    MapProcessor   map_;    // Map + distance field

    // Motion tracking
    bool   odom_received_ = false;
    double prev_x_ = 0;
    double prev_y_ = 0;
    double prev_theta_ = 0;
    double update_min_d_ = 0.001;
    double update_min_a_ = 0.001;
    double max_scan_age_ = 0.05;
    double cloud_publish_rate_ = 2.0;
    rclcpp::Time last_cloud_publish_time_;
    bool debug_pre_resample_particles_ = false;
    bool initial_heading_from_raceline_ = true;
    bool initial_pose_from_raceline_ = true;
    double initial_pose_heading_offset_rad_ = 0.0;

    // Slip-aware noise scaling
    double slip_angular_threshold_ = 1.0;  // rad/s
    double slip_noise_multiplier_ = 2.0;
    rclcpp::Time last_scan_time_;          // For dt calculation

    // Prediction baseline (reset on reinit)
    bool prediction_baseline_ready_ = false;
    bool global_pose_published_ = false;
    bool global_localization_locked_ = false;
    bool local_odom_reference_ready_ = false;
    bool initial_scan_update_pending_ = true;
    bool localization_start_time_set_ = false;
    bool startup_scan_refinement_attempted_ = false;
    double pred_last_x_ = 0;
    double pred_last_y_ = 0;
    double pred_last_theta_ = 0;
    double local_odom_reference_x_ = 0.0;
    double local_odom_reference_y_ = 0.0;
    double local_odom_reference_theta_ = 0.0;
    PoseEstimate local_pose_reference_;
    double force_max_particles_initial_sec_ = 0.0;
    rclcpp::Time localization_start_time_;
    rclcpp::Time last_processed_scan_stamp_;
    rclcpp::Time last_map_correction_stamp_;
    rclcpp::Time last_current_map_pose_stamp_;
    bool map_odom_valid_ = false;
    Eigen::Vector3d map_odom_pose_{0.0, 0.0, 0.0};
    Eigen::Matrix3d current_map_pose_covariance_ = Eigen::Matrix3d::Identity();
    double current_map_pose_process_xy_m2_per_s_ = 0.002;
    double current_map_pose_process_yaw2_per_s_ = 0.0005;
    double localization_degraded_after_s_ = 0.30;
    uint64_t consecutive_rejected_scans_ = 0;
    bool last_scan_correction_accepted_ = false;
    double global_pose_covariance_xy_max_ = 0.25;
    double global_pose_covariance_yaw_max_ = 0.12;
    double global_pose_max_track_distance_m_ = 0.45;
    double global_pose_max_track_heading_error_rad_ = 0.45;
    bool global_start_anchor_enabled_ = true;
    double global_start_anchor_radius_m_ = 0.90;
    std::vector<ParticleFilter::TrackHeadingPoint> global_heading_points_;

    // A likelihood-field match can select a visually similar section of the
    // circuit. Once globally locked, reject scan corrections that are
    // implausibly large relative to the accepted pose plus this scan's odom
    // delta, then restart the local cloud around the odometry prediction.
    double local_scan_correction_max_distance_m_ = 0.35;
    double local_scan_correction_max_yaw_rad_ = 0.45;
    double local_cluster_association_max_distance_m_ = 0.80;
    double local_cluster_min_weight_ = 0.75;
    // A scan match is an absolute map-pose measurement.  Keep odometry as
    // the short-term prediction and apply only a bounded fraction of the
    // scan correction so a small systematic likelihood-field bias cannot
    // accumulate into a large along-track error at the native scan rate.
    double local_scan_correction_gain_ = 0.35;
    bool local_tracking_reinitialize_cloud_ = true;
    double local_tracking_cloud_covariance_xy_ = 0.01;
    double local_tracking_cloud_covariance_yaw_ = 0.01;

    struct OdomSample {
        rclcpp::Time stamp;
        double x;
        double y;
        double theta;
        double covariance_xx;
        double covariance_xy;
        double covariance_yy;
        double covariance_xyaw;
        double covariance_yyaw;
        double covariance_yawyaw;
    };

    std::deque<OdomSample> odom_history_;
    double odom_history_duration_s_ = 0.2;
    double odom_reset_distance_m_ = 2.0;
    double odom_reset_yaw_rad_ = 1.5;

    // A scan can be delivered before the odometry sample carrying the same
    // source-time interval. Keep it briefly and retry once newer odometry is
    // available; never process it against the previous sample.
    std::deque<sensor_msgs::msg::LaserScan::SharedPtr> pending_scans_;
    std::mutex pending_scan_mutex_;
    std::atomic<size_t> pending_scan_drop_count_{0};
    std::atomic<size_t> scan_processing_drop_count_{0};

    // Pose-jump guard: prevents one-frame false global relocalization from
    // teleporting the controller.
    bool pose_jump_gate_enabled_ = true;
    double pose_jump_max_distance_m_ = 1.0;
    double pose_jump_max_yaw_rad_ = 1.2;
    double pose_jump_confirm_distance_m_ = 0.75;
    double pose_jump_confirm_yaw_rad_ = 0.6;
    int pose_jump_confirm_scans_ = 5;
    bool have_last_published_pose_ = false;
    PoseEstimate last_published_pose_;
    bool have_pending_jump_pose_ = false;
    PoseEstimate pending_jump_pose_;
    int pending_jump_pose_count_ = 0;

    // Thread safety
    std::mutex pf_mutex_;                       // Protects pf_ during GPU ops
    std::atomic<bool> processing_scan_{false};  // Drop scans during processing

    // Cached estimate for decoupled publishing
    PoseEstimate cached_estimate_;
    std::mutex   estimate_mutex_;
    uint64_t last_published_kld_diag_seq_ = 0;

    // Frame IDs and topic names
    std::string base_frame_;
    std::string odom_frame_;
    std::string global_frame_;
    std::string scan_topic_;
    std::string odom_topic_;

    // Callback groups for parallel execution
    rclcpp::CallbackGroup::SharedPtr scan_cb_group_;
    rclcpp::CallbackGroup::SharedPtr odom_cb_group_;
};

}  // namespace gpu_amcl_cpp
