// Controller diagnostics implementation, separated from actuator decisions.
// Keep exact JSON schema and callbacks unchanged while establishing the boundary.
#include "mpc_controller_node_impl.hpp"

namespace f1tenth_mpc {

void MpcControllerNode::publish_cycle_diagnostics(const MpcRtiState_t &state,
    const MpcSynchronizedState &synchronized,
    const MpcControlTimePrediction &control_prediction,
    double control_prediction_us, double progress,
    double source_dt_s, const MpcRtiCycleResult_t &result,
    double solve_us, int64_t control_ros_stamp_ns,
    int64_t candidate_command_stamp_ns,
    int64_t callback_steady_ns, int64_t synchronize_steady_ns)
{
    if (!diagnostics_pub_) return;
    const DiagnosticTimingSample previous_diagnostics =
        last_diagnostic_timing_;
    const DiagnosticPublisherWorker::Stats async_publisher_stats =
        diagnostics_worker_ ? diagnostics_worker_->stats()
                            : DiagnosticPublisherWorker::Stats{};
    DiagnosticTimingRecorder diagnostic_timing(last_diagnostic_timing_);
    diagnostic_timing.begin_json_build();
    std::ostringstream json;
    const auto json_number = [&json](double value) {
        if (std::isfinite(value)) json << value;
        else json << "null";
    };
    const auto append_rollout_failure =
        [&json, &json_number](const char *name,
            const MpcRtiRolloutFailure_t &failure) {
        json << ",\"" << name << "\":{\"valid\":"
             << (failure.valid ? "true" : "false")
             << ",\"stage\":" << failure.stage
             << ",\"progress_m\":";
        json_number(failure.progress);
        json << ",\"state\":[";
        json_number(failure.state.plant.e_y);
        json << ',';
        json_number(failure.state.plant.e_psi);
        json << ',';
        json_number(failure.state.plant.u);
        json << ',';
        json_number(failure.state.plant.v);
        json << ',';
        json_number(failure.state.plant.r);
        json << ',';
        json_number(failure.state.plant.delayed_steering_command_2);
        json << ',';
        json_number(failure.state.plant.actual_steering_angle);
        json << "],\"reference\":[";
        json_number(failure.reference.e_y);
        json << ',';
        json_number(failure.reference.e_psi);
        json << ',';
        json_number(failure.reference.path_curvature);
        json << ',';
        json_number(failure.reference.left_bound);
        json << ',';
        json_number(failure.reference.right_bound);
        json << "],\"margin_m\":";
        json_number(failure.margin_m);
        json << ",\"lower_bound_m\":";
        json_number(failure.lower_bound_m);
        json << ",\"upper_bound_m\":";
        json_number(failure.upper_bound_m);
        json << '}';
    };
    MpcTrajectorySample_t current_path_sample{};
    const bool current_path_sample_valid = mpc_trajectory_sample(
        trajectory_.data(), trajectory_.size(), track_length_m_, progress,
        &current_path_sample);
    const bool gate_budget_violation =
        last_publication_gate_elapsed_ms_ >= end_to_end_cycle_budget_ms_;
    json << std::setprecision(9)
        << "{\"diagnostics_schema_version\":4"
        << ",\"control_callback_interval_ms\":";
    json_number(control_callback_interval_ms_);
    json << ",\"previous_diagnostics_timing\":{";
    json << "\"valid\":" << (previous_diagnostics.valid ? "true" : "false")
        << ",\"wall_us\":";
    json_number(previous_diagnostics.wall_us);
    json << ",\"thread_cpu_us\":";
    json_number(previous_diagnostics.thread_cpu_us);
    json << ",\"queue_enqueue_wall_us\":";
    json_number(previous_diagnostics.queue_enqueue_wall_us);
    json << ",\"queue_enqueue_thread_cpu_us\":";
    json_number(previous_diagnostics.queue_enqueue_thread_cpu_us);
    json << ",\"reconstructed_plant_steps\":"
        << previous_diagnostics.reconstructed_plant_steps
        << ",\"json_build_wall_us\":";
    json_number(previous_diagnostics.json_build_wall_us);
    json << ",\"json_build_thread_cpu_us\":";
    json_number(previous_diagnostics.json_build_thread_cpu_us);
    json << "},\"diagnostic_async_publisher\":{"
         << "\"published_frames\":"
         << async_publisher_stats.published_frames
         << ",\"replaced_frames\":"
         << async_publisher_stats.replaced_frames
         << ",\"publish_failures\":"
         << async_publisher_stats.publish_failures
         << ",\"lower_priority\":"
         << (async_publisher_stats.lower_priority ? "true" : "false")
         << ",\"last_queue_wait_us\":";
    json_number(async_publisher_stats.last_queue_wait_us);
    json << ",\"last_serialization_wall_us\":";
    json_number(async_publisher_stats.last_serialization_wall_us);
    json << ",\"last_serialization_thread_cpu_us\":";
    json_number(async_publisher_stats.last_serialization_thread_cpu_us);
    json << ",\"last_publish_wall_us\":";
    json_number(async_publisher_stats.last_publish_wall_us);
    json << ",\"last_publish_thread_cpu_us\":";
    json_number(async_publisher_stats.last_publish_thread_cpu_us);
    json << "},\"publication_gate_budget_violation\":";
    if (last_publication_gate_elapsed_ms_ < 0.0) json << "null";
    else json << (gate_budget_violation ? "true" : "false");
    json << ",\"prior_plan_published_while_over_budget\":";
    if (last_publication_gate_elapsed_ms_ < 0.0) json << "null";
    else json << ((command_published_this_cycle_ &&
            prior_plan_revalidation_attempted_ && gate_budget_violation)
                ? "true" : "false");
    json << ",\"steering_limit_rad\":"
         << rti_config_.model.max_steering_rad;
    json << ",\"issued_command_stamp_ns\":";
    if (last_published_command_stamp_ns_ > 0)
        json << last_published_command_stamp_ns_;
    else
        json << "null";
    json << ",\"status\":\"" << cycle_status_name(result.status)
        << "\",\"action_source\":\""
        << (!command_published_this_cycle_ ? "none_published" :
            (result.status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH
                ? "current_solver_line_search" :
            (result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX
                ? "revalidated_candidate_prefix" :
                (result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN
                    ? "revalidated_prior_plan"
                    : "current_solver_candidate"))))
        << "\",\"command_published\":"
        << (command_published_this_cycle_ ? "true" : "false")
        << ",\"issued_action\":";
    if (command_published_this_cycle_) {
        json << "{\"steering_rad\":";
        json_number(result.published_steering_command);
        json << ",\"throttle\":";
        json_number(result.published_throttle);
        json << ",\"command_stamp_ns\":";
        if (last_published_command_stamp_ns_ > 0)
            json << last_published_command_stamp_ns_;
        else
            json << "null";
        json << '}';
    } else {
        json << "null";
    }
    json << ",\"causal_command_queue\":{\"delayed_steering_rad\":[";
    json_number(state.plant.delayed_steering_command_1);
    json << ',';
    json_number(state.plant.delayed_steering_command_2);
    json << "],\"delayed_throttle\":[";
    json_number(state.plant.delayed_throttle_command_1);
    json << ',';
    json_number(state.plant.delayed_throttle_command_2);
    json << "]},\"decision_epoch_timestamps_ns\":{";
    json << "\"source\":" << synchronized.source_stamp_ns
         << ",\"control\":" << control_ros_stamp_ns
         << ",\"candidate_command\":"
         << candidate_command_stamp_ns
         << ",\"callback_steady\":" << callback_steady_ns << '}';
    const bool recovery_snapshot =
        !command_published_this_cycle_ ||
        result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX ||
        result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN ||
        consecutive_candidate_prefix_actions_ > 0 ||
        consecutive_prior_plan_actions_ > 0;
    if (recovery_snapshot) {
        json << ",\"certified_plan_snapshot\":{\"valid\":"
             << (certified_plan_.valid ? "true" : "false")
             << ",\"horizon\":" << certified_plan_.horizon
             << ",\"cursor\":" << prior_plan_control_index_
             << ",\"actions_remaining\":"
             << (certified_plan_.valid
                    ? std::max(0, certified_plan_.horizon -
                        prior_plan_control_index_)
                    : 0)
             << ",\"controls\":[";
        if (certified_plan_.valid) {
            for (int k = 0; k < certified_plan_.horizon; ++k) {
                if (k > 0) json << ',';
                json << '[';
                json_number(
                    certified_plan_.controls[k].steering_target_rad);
                json << ',';
                json_number(
                    certified_plan_.controls[k].throttle_command);
                json << ']';
            }
        }
        json << "],\"states\":[";
        if (certified_plan_.valid) {
            for (int k = 0; k <= certified_plan_.horizon; ++k) {
                if (k > 0) json << ',';
                const MpcRtiState_t &plan_state =
                    certified_plan_.states[k];
                json << "{\"stage\":" << k << ",\"progress_m\":";
                json_number(certified_plan_.progress[k]);
                json << ",\"plant\":[";
                json_number(plan_state.plant.e_y);
                json << ',';
                json_number(plan_state.plant.e_psi);
                json << ',';
                json_number(plan_state.plant.u);
                json << ',';
                json_number(plan_state.plant.v);
                json << ',';
                json_number(plan_state.plant.r);
                json << ',';
                json_number(
                    plan_state.plant.delayed_steering_command_1);
                json << ',';
                json_number(
                    plan_state.plant.delayed_steering_command_2);
                json << ',';
                json_number(plan_state.plant.actual_steering_angle);
                json << ',';
                json_number(
                    plan_state.plant.delayed_throttle_command_1);
                json << ',';
                json_number(
                    plan_state.plant.delayed_throttle_command_2);
                json << "],\"source_plant_internal_state\":[";
                for (int i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i) {
                    if (i > 0) json << ',';
                    json_number(
                        plan_state.plant.source_plant_internal_state[i]);
                }
                json << "],\"plant_state_flags\":["
                     << plan_state.plant.source_plant_initialized << ','
                     << plan_state.plant.source_suspension_history_initialized
                     << ','
                     << plan_state.plant.source_imu_state_initialized << ']';
                json << ",\"corridor_slack_m\":[";
                MpcTrajectorySample_t plan_reference{};
                float lower_slack = NAN;
                float upper_slack = NAN;
                const bool slack_valid = k > 0 &&
                    mpc_trajectory_sample(trajectory_.data(),
                        trajectory_.size(), track_length_m_,
                        certified_plan_.progress[k], &plan_reference) &&
                    mpc_rti_compute_corridor_slack(
                        &plan_state.plant, plan_reference.left_bound,
                        plan_reference.right_bound, &rti_config_.model, k,
                        &lower_slack, &upper_slack);
                if (slack_valid) {
                    json_number(lower_slack);
                    json << ',';
                    json_number(upper_slack);
                } else {
                    json << "null,null";
                }
                json << "]}";
            }
        }
        json << "]}";
    }
    json << ",\"validated_recovery_prefix\":{\"valid\":"
         << (has_validated_recovery_prefix() ? "true" : "false")
         << ",\"cursor\":" << validated_recovery_prefix_cursor_
         << ",\"controls\":[";
    if (has_validated_recovery_prefix()) {
        for (int i = 0; i < validated_recovery_prefix_count_; ++i) {
            if (i > 0) json << ',';
            json << '[';
            json_number(
                validated_recovery_prefix_controls_[i].steering_target_rad);
            json << ',';
            json_number(
                validated_recovery_prefix_controls_[i].throttle_command);
            json << ']';
        }
    }
    json << "]}";
    json << ",\"solver_skipped_for_deadline\":"
        << (result.solver_skipped_for_deadline ? "true" : "false")
        << ",\"prefer_full_validated_prior_plan_on_rejection\":"
        << (prefer_full_validated_prior_plan_on_rejection_
                ? "true" : "false")
        << ",\"quality_aware_prefix_selection_on_rejection\":"
        << (quality_aware_prefix_selection_on_rejection_
                ? "true" : "false")
        << ",\"quality_aware_prefix_selection\":{\"attempted\":"
        << (quality_prefix_attempted_ ? "true" : "false")
        << ",\"choice\":\"" << quality_prefix_choice_ << "\""
        << ",\"clearance_override\":"
        << (quality_prefix_clearance_override_ ? "true" : "false")
        << ",\"stale_prior_refresh\":"
        << (quality_prefix_stale_prior_refresh_ ? "true" : "false")
        << ",\"yaw_direction_guard\":"
        << (quality_prefix_yaw_direction_guard_ ? "true" : "false")
        << ",\"clearance_reserve_m\":"
        << quality_prefix_clearance_reserve_m_
        << ",\"minimum_clearance_gain_m\":"
        << quality_prefix_minimum_clearance_gain_m_
        << ",\"stale_prior_refresh_after_actions\":"
        << quality_prefix_stale_prior_refresh_after_actions_
        << ",\"stale_prior_max_clearance_loss_m\":"
        << quality_prefix_stale_prior_max_clearance_loss_m_
        << ",\"stale_prior_objective_loss_fraction\":"
        << quality_prefix_stale_prior_objective_loss_fraction_
        << ",\"near_boundary_minimum_clearance_gain_m\":"
        << quality_prefix_near_boundary_minimum_gain_m_
        << ",\"near_boundary_max_objective_ratio\":"
        << quality_prefix_near_boundary_max_objective_ratio_
        << ",\"near_boundary_control_changes_bounded\":"
        << (quality_prefix_near_boundary_controls_bounded_ ? "true" : "false")
        << ",\"near_boundary_override\":"
        << (quality_prefix_near_boundary_override_ ? "true" : "false")
        << ",\"iteration_limited_relative_gain\":"
        << quality_prefix_iteration_limited_relative_gain_
        << ",\"prior_source\":\""
        << (quality_prefix_choice_ == "candidate"
                ? "not_selected" :
            (quality_prefix_prior_is_recovery_
                ? "validated_recovery_prefix" : "certified_plan"))
        << "\""
        << ",\"candidate_valid\":"
        << (quality_candidate_prefix_valid_ ? "true" : "false")
        << ",\"prior_valid\":"
        << (quality_prior_prefix_valid_ ? "true" : "false")
        << ",\"candidate_objective\":";
    json_number(quality_candidate_prefix_objective_);
    json << ",\"prior_objective\":";
    json_number(quality_prior_prefix_objective_);
    json << ",\"candidate_minimum_corridor_slack_m\":";
    json_number(quality_candidate_prefix_slack_m_);
    json << ",\"prior_minimum_corridor_slack_m\":";
    json_number(quality_prior_prefix_slack_m_);
    json << ",\"validation_us\":";
    json_number(quality_prefix_validation_us_);
    json << ",\"candidate_prefix_controls\":[";
    for (int i = 0; i < quality_candidate_control_count_; ++i) {
        if (i > 0) json << ',';
        json << '[' << quality_candidate_controls_[i].steering_target_rad
             << ',' << quality_candidate_controls_[i].throttle_command
             << ']';
    }
    json << "],\"prior_prefix_controls\":[";
    for (int i = 0; i < quality_prior_control_count_; ++i) {
        if (i > 0) json << ',';
        json << '[' << quality_prior_controls_[i].steering_target_rad
             << ',' << quality_prior_controls_[i].throttle_command
             << ']';
    }
    json << "]}"
        << ",\"prior_plan_revalidation\":{\"attempted\":"
        << (prior_plan_revalidation_attempted_ ? "true" : "false")
        << ",\"exact_valid\":"
        << (prior_plan_exact_valid_ ? "true" : "false")
        << ",\"timing_rejected\":"
        << (prior_plan_timing_rejected_ ? "true" : "false")
        << ",\"validation_horizon\":"
        << prior_plan_validated_horizon_
        << ",\"control_index\":"
        << prior_plan_validated_control_index_
        << ",\"actions_remaining\":"
        << prior_plan_actions_remaining_
        << ",\"validation_status\":"
        << prior_plan_validation_status_
        << ",\"failure_stage\":"
        << prior_plan_validation_failure_stage_
        << ",\"validation_us\":";
    json_number(prior_plan_validation_us_);
    json << "},\"full_prior_plan_suffix_revalidation\":{\"attempted\":"
        << (full_prior_suffix_revalidation_attempted_ ? "true" : "false")
        << ",\"exact_valid\":"
        << (full_prior_suffix_exact_valid_ ? "true" : "false")
        << ",\"timing_rejected\":"
        << (full_prior_suffix_timing_rejected_ ? "true" : "false")
        << ",\"validation_horizon\":"
        << full_prior_suffix_validation_horizon_
        << ",\"validation_status\":"
        << full_prior_suffix_validation_status_
        << ",\"failure_stage\":"
        << full_prior_suffix_failure_stage_
        << ",\"validation_us\":";
    json_number(full_prior_suffix_validation_us_);
    json << "},\"candidate_prefix_revalidation\":{\"attempted\":"
        << (candidate_prefix_revalidation_attempted_ ? "true" : "false")
        << ",\"exact_valid\":"
        << (candidate_prefix_exact_valid_ ? "true" : "false")
        << ",\"residual_quality_acceptable\":"
        << (candidate_prefix_residual_quality_ ? "true" : "false")
        << ",\"timing_rejected\":"
        << (candidate_prefix_timing_rejected_ ? "true" : "false")
        << ",\"solver_converged\":"
        << (candidate_prefix_solver_converged_ ? "true" : "false")
        << ",\"solver_iteration_limited\":"
        << (candidate_prefix_iteration_limited_ ? "true" : "false")
        << ",\"validation_horizon\":"
        << candidate_prefix_validation_horizon_
        << ",\"validation_status\":"
        << candidate_prefix_validation_status_
        << ",\"failure_stage\":"
        << candidate_prefix_validation_failure_stage_
        << ",\"validation_us\":";
    json_number(candidate_prefix_validation_us_);
    json << "},\"consecutive_prior_plan_actions\":"
        << consecutive_prior_plan_actions_
        << ",\"prior_plan_actions_remaining\":"
        << prior_plan_actions_remaining_
        << ",\"feasibility_restoration\":{\"attempted\":"
        << result.feasibility_restoration_attempted
        << ",\"accepted\":"
        << result.feasibility_restoration_accepted
        << ",\"iterations\":"
        << result.feasibility_restoration_iterations
        << ",\"primal_residual\":";
    json_number(result.feasibility_restoration_primal_residual);
    json << ",\"elapsed_us\":";
    json_number(result.feasibility_restoration_us);
    json << '}'
        << ",\"consecutive_candidate_prefix_actions\":"
        << consecutive_candidate_prefix_actions_
        << ",\"input_failure_reason\":\""
        << input_failure_reason_name(result.input_failure_reason)
        << "\",\"nominal_build_failure_stage\":\""
        << nominal_build_failure_stage_name(
            result.nominal_build_failure_stage)
        << "\",\"nominal_build_failure_constraint\":\""
        << command_envelope_failure_name(
            result.nominal_build_failure_constraint)
        << "\",\"nominal_build_failure_index\":"
        << result.nominal_build_failure_index
        << ",\"nonlinear_failure_stage\":"
        << result.nonlinear_failure_stage
        << ",\"nonlinear_failure_reason\":\""
        << nonlinear_failure_reason_name(result.nonlinear_failure_reason)
        << '"';
    append_rollout_failure("nonlinear_failure", result.nonlinear_failure);
    json
        << ",\"source_stamp_ns\":" << synchronized.source_stamp_ns
        << ",\"odom_source_stamp_ns\":"
        << synchronized.odom_source_stamp_ns
        << ",\"map_pose_source_stamp_ns\":"
        << synchronized.map_pose_source_stamp_ns
        << ",\"control_ros_stamp_ns\":" << control_ros_stamp_ns
        << ",\"candidate_command_stamp_ns\":"
        << candidate_command_stamp_ns
        << ",\"candidate_command_lead_s\":"
        << static_cast<double>(candidate_command_stamp_ns -
            control_ros_stamp_ns) * 1.0e-9
        << ",\"candidate_command_lateness_ms\":"
        << std::max(0.0, static_cast<double>(now().nanoseconds() -
            candidate_command_stamp_ns) * 1.0e-6)
        << ",\"max_command_start_lateness_ms\":"
        << max_command_start_lateness_ms_
        << ",\"end_to_end_control_ms\":"
        << std::max(0.0, static_cast<double>(steady_time_ns() -
            callback_steady_ns) * 1.0e-6)
        << ",\"publication_gate_elapsed_ms\":";
    if (last_publication_gate_elapsed_ms_ >= 0.0)
        json_number(last_publication_gate_elapsed_ms_);
    else
        json << "null";
    json
        << ",\"middleware_publish_us\":";
    json_number(last_middleware_publish_us_);
    json << ",\"command_history_us\":";
    json_number(last_command_history_us_);
    json
        << ",\"end_to_end_cycle_budget_ms\":"
        << end_to_end_cycle_budget_ms_
        << ",\"source_age_s\":" << synchronized.source_age_s
        << ",\"source_dt_s\":" << source_dt_s
        << ",\"pose_odom_skew_s\":" << synchronized.pose_odom_skew_s
        << ",\"control_time_prediction\":{\"mode\":\"unity_plant\","
        << "\"age_s\":"
        << control_prediction.age_s << ",\"elapsed_us\":"
        << control_prediction_us << ",\"command_changes_used\":"
        << control_prediction.command_changes_used
        << ",\"plant_source_stamp_ns\":"
        << control_prediction.plant_source_stamp_ns
        << ",\"plant_handoff_stamp_ns\":"
        << control_prediction.plant_handoff_stamp_ns
        << ",\"plant_handoff_progress_m\":"
        << control_prediction.plant_handoff_progress_m
        << ",\"plant_to_sensor_s\":"
        << control_prediction.plant_to_sensor_s
        << ",\"sensor_state_fused\":"
        << (control_prediction.sensor_state_fused ? "true" : "false")
        << ",\"command_actuation_delay_s\":"
        << control_time_predictor_config_.command_actuation_delay_s
        << ",\"command_event_stamps_ns\":[";
    for (std::size_t i = 0;
         i < control_prediction.command_event_stamp_count; ++i) {
        if (i > 0) json << ',';
        json << control_prediction.command_event_stamps_ns[i];
    }
    json << "]"
        << ",\"plant_state_propagated\":"
        << (control_prediction.plant_state_valid ? "true" : "false")
        << ",\"odom_hidden_state_used\":"
        << (unity_plant_state_used_ ? "true" : "false")
        << ",\"source_map_pose\":[" << synchronized.map_x << ','
        << synchronized.map_y << ',' << synchronized.map_yaw
        << "],\"predicted_map_pose\":["
        << control_prediction.state.map_x << ','
        << control_prediction.state.map_y << ','
        << control_prediction.state.map_yaw << ']'
        << ",\"predicted_speed_state\":["
        << control_prediction.state.u << ',' << control_prediction.state.v
        << ',' << control_prediction.state.yaw_rate << ']'
        << "}"
        << ",\"host_timing_ns\":{" << "\"callback\":"
        << callback_steady_ns << ",\"synchronize\":"
        << synchronize_steady_ns << '}'
        << ",\"progress_m\":" << progress
        << ",\"active_speed_ceiling_mps\":"
        << rti_config_.model.active_speed_ceiling_mps
        << ",\"path_curvature_per_m\":";
    json_number(current_path_sample_valid ? current_path_sample.curvature :
        std::numeric_limits<double>::quiet_NaN());
    json << ",\"reference\":{\"trajectory_acceleration_mps2\":";
    json_number(current_path_sample_valid ? current_path_sample.acceleration :
        std::numeric_limits<double>::quiet_NaN());
    const double current_reference_speed = current_path_sample_valid
        ? std::min(current_path_sample.speed,
            static_cast<double>(rti_config_.model.active_speed_ceiling_mps))
        : std::numeric_limits<double>::quiet_NaN();
    json << ",\"trajectory_speed_mps\":";
    json_number(current_path_sample_valid ? current_path_sample.speed :
        std::numeric_limits<double>::quiet_NaN());
    json << ",\"speed_reference_mps\":";
    json_number(current_reference_speed);
    json << "}";
    json << ",\"state\":[" << state.plant.e_y << ','
        << state.plant.e_psi << ',' << state.plant.u << ','
        << state.plant.v << ',' << state.plant.r << ','
        << state.plant.delayed_steering_command_1
        << ',' << state.plant.delayed_steering_command_2
        << ',' << state.plant.actual_steering_angle
        << ',' << state.plant.delayed_throttle_command_1
        << ',' << state.plant.delayed_throttle_command_2 << ']'
        << ",\"plant_internal_state\":[";
    for (int i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i) {
        if (i > 0) json << ',';
        json_number(state.plant.source_plant_internal_state[i]);
    }
    json << "],\"plant_state_flags\":["
        << state.plant.source_plant_initialized << ','
        << state.plant.source_suspension_history_initialized << ','
        << state.plant.source_imu_state_initialized << ']'
        << ",\"plant_state_source\":\""
        << (unity_plant_state_used_
                ? "odom_unity_plant_state"
                : "source_plant_from_odom_or_handoff")
        << "\"";
    json << ",\"objective_weights\":{\"underspeed\":"
        << rti_config_.model.weight_u
        << ",\"overspeed\":"
        << rti_config_.model.weight_u_overspeed
        << ",\"overspeed_curvature_gain_m\":"
        << rti_config_.model.overspeed_curvature_gain_m
        << ",\"underspeed_curvature_gain_m\":"
        << rti_config_.model.underspeed_curvature_gain_m
        << ",\"wheel_slip\":"
        << rti_config_.model.weight_longitudinal_slip
        << ",\"wheel_slip_curvature_gain_m\":"
        << rti_config_.model.longitudinal_slip_curvature_gain_m
        << ",\"steering_slew\":"
        << rti_config_.model.weight_steering_slew
        << ",\"throttle_slew\":"
        << rti_config_.model.weight_throttle_slew << '}';
    json << ",\"solver\":{\"iterations\":" << result.solver_iterations
        << ",\"max_iterations_requested\":"
        << requested_max_solver_iterations_
        << ",\"max_iterations_configured\":"
        << rti_config_.solver.max_iterations
        << ",\"time_budget_exhausted\":"
        << (result.solver_time_budget_exhausted ? "true" : "false")
        << ",\"source_informed_sqp_enabled\":"
        << (rti_config_.source_informed_sqp ? "true" : "false")
        << ",\"terminal_viability_enabled\":" << (rti_config_.terminal_viability ? "true":"false")
        << ",\"first_pass_objective\":";
    json_number(result.first_pass_objective);
    json << ",\"second_pass_objective\":";
    json_number(result.second_pass_objective);
    json << ",\"exact_objective_improvement\":" << result.exact_objective_improvement
        << ",\"source_directional_derivative\":" << result.source_directional_derivative
        << ",\"source_curvature_steering\":" << result.source_curvature_regularization[0]
        << ",\"source_curvature_throttle\":" << result.source_curvature_regularization[1]
        << ",\"source_refinement_selected\":"
        << (result.experimental_source_nonlinear_selected ? "true" : "false")
        << ",\"source_nonlinear_exact_evaluations\":"
        << result.source_nonlinear_exact_evaluations
        << ",\"source_nonlinear_wall_us\":"
        << result.source_nonlinear_wall_us
        << ",\"source_nonlinear_objective_improvement\":"
        << result.source_nonlinear_objective_improvement
        << ",\"source_nonlinear_terminal_probe_count\":"
        << result.source_nonlinear_terminal_probe_count
        << ",\"source_nonlinear_terminal_rejected\":"
        << result.source_nonlinear_terminal_rejected
        << ",\"source_nonlinear_timed_out\":"
        << (result.source_nonlinear_timed_out ? "true" : "false")
        << ",\"sqp_iterations_attempted\":"
        << result.sqp_iterations_attempted
        << ",\"sqp_iterations_completed\":"
        << result.sqp_iterations_completed
        << ",\"sqp_iterations_accepted\":"
        << result.sqp_iterations_accepted
        << ",\"refinement_prefix_selected_pass\":"
        << result.refinement_prefix_selected_pass
        << ",\"refinement_full_horizon_selected_pass\":"
        << result.refinement_full_horizon_selected_pass
        << ",\"refinement_terminal_probe_count\":"
        << result.refinement_terminal_probe_count
        << ",\"refinement_terminal_selected_pass\":"
        << result.refinement_terminal_selected_pass
        << ",\"refinement_terminal_probe_us\":"
        << result.refinement_terminal_probe_us
        << ",\"refinement_nominal_build_us\":"
        << result.refinement_nominal_build_us
        << ",\"refinement_prefix_validation_us\":"
        << result.refinement_prefix_validation_us;
    json << ",\"refinement_first_prefix_objective\":";
    json_number(result.refinement_first_prefix_objective);
    json << ",\"refinement_second_prefix_objective\":";
    json_number(result.refinement_second_prefix_objective);
    json
        << ",\"admm_status\":" << result.admm_status
        << ",\"admm_state_reset_for_recovery\":"
        << (result.admm_state_reset_for_recovery ? "true" : "false")
        << ",\"admm_stop_reason\":\""
        << (result.solver_skipped_for_deadline
                ? "deadline_before_admm"
            : result.solver_time_budget_exhausted
                ? "time_budget"
            : result.admm_status == 0
                ? "converged"
            : result.admm_status == 1
                ? "iteration_cap"
            : result.admm_status < 0
                ? "not_run"
                : "numerical_or_solver_error")
        << "\",\"primal_residual\":";
    json_number(result.primal_residual);
    json << ",\"dual_residual\":";
    json_number(result.dual_residual);
    json << ",\"residual_hotspots\":{";
    json << "\"state_primal\":{" << "\"value\":";
    json_number(result.state_primal_residual);
    json << ",\"stage\":" << result.state_primal_stage
        << ",\"coordinate\":" << result.state_primal_index << "},"
        << "\"state_dual\":{" << "\"value\":";
    json_number(result.state_dual_residual);
    json << ",\"stage\":" << result.state_dual_stage
        << ",\"coordinate\":" << result.state_dual_index << "},"
        << "\"input_primal\":{" << "\"value\":";
    json_number(result.input_primal_residual);
    json << ",\"stage\":" << result.input_primal_stage
        << ",\"coordinate\":" << result.input_primal_index << "},"
        << "\"input_dual\":{" << "\"value\":";
    json_number(result.input_dual_residual);
    json << ",\"stage\":" << result.input_dual_stage
        << ",\"coordinate\":" << result.input_dual_index << "}}";
    json << ",\"max_regularization\":";
    json_number(result.maximum_regularization);
    json << ",\"regularization_count\":"
         << result.regularization_count
         << ",\"nonsmooth_columns\":"
         << result.nonsmooth_jacobian_columns
         << ",\"rho_start\":" << result.rho_start
         << ",\"rho_u_start\":" << result.rho_u_start
         << ",\"rho_final\":" << result.rho_final
         << ",\"rho_u_final\":" << result.rho_u_final
         << ",\"rho_change_count\":" << result.rho_change_count
         << ",\"factorization_count\":" << result.factorization_count
         << ",\"factorization_time_ns\":"
         << result.factorization_time_ns
         << ",\"cold_start_initialization_time_ns\":"
         << result.solver_cold_start_initialization_time_ns
         << ",\"riccati_rhs_time_ns\":"
         << result.riccati_rhs_time_ns
         << ",\"admm_projection_time_ns\":"
         << result.admm_projection_time_ns
         << ",\"rho_update_time_ns\":" << result.rho_update_time_ns
         << ",\"scaling_transform_time_ns\":"
         << result.solver_scaling_time_ns
         << ",\"output_unscale_time_ns\":"
         << result.solver_output_unscale_time_ns
         << ",\"qp_build_us\":" << result.qp_build_us
         << ",\"jacobian_wall_us\":" << result.jacobian_wall_us
         << ",\"jacobian_thread_cpu_us\":"
         << result.jacobian_thread_cpu_us
         << ",\"jacobian_slowest_stage_us\":"
         << result.jacobian_slowest_stage_us
         << ",\"jacobian_stages_completed\":"
         << result.jacobian_stages_completed
         << ",\"jacobian_threads_used\":"
         << result.jacobian_threads_used
         << ",\"jacobian_source_sensitivity_calls\":"
         << result.jacobian_source_sensitivity_calls
         << ",\"jacobian_finite_difference_columns\":"
         << result.jacobian_finite_difference_columns
         << ",\"jacobian_worker_cpu_us\":[";
    for (int worker = 0; worker < MPC_RTI_MAX_JACOBIAN_THREADS; ++worker) {
        if (worker > 0) json << ',';
        json_number(result.jacobian_worker_cpu_us[worker]);
    }
    json << "],\"jacobian_worker_stage_wall_us\":[";
    for (int worker = 0; worker < MPC_RTI_MAX_JACOBIAN_THREADS; ++worker) {
        if (worker > 0) json << ',';
        json_number(result.jacobian_worker_stage_wall_us[worker]);
    }
    json << "],\"jacobian_worker_stage_count\":[";
    for (int worker = 0; worker < MPC_RTI_MAX_JACOBIAN_THREADS; ++worker) {
        if (worker > 0) json << ',';
        json << result.jacobian_worker_stage_count[worker];
    }
    json << "],\"jacobian_worker_os_tid\":[";
    for (int worker = 0; worker < MPC_RTI_MAX_JACOBIAN_THREADS; ++worker) {
        if (worker > 0) json << ',';
        json << result.jacobian_worker_os_tid[worker];
    }
    json << ']'
         << ",\"nominal_build_us\":" << result.nominal_build_us
         << ",\"reference_seed_us\":" << result.reference_seed_us
         << ",\"cold_start_queue_seed_us\":"
         << result.cold_start_queue_seed_us
         << ",\"nominal_rollout_a_us\":"
         << result.nominal_rollout_a_us
         << ",\"progress_reference_resampling_us\":"
         << result.progress_reference_resampling_us
         << ",\"nominal_control_seed_source\":"
         << result.nominal_control_seed_source
         << ",\"command_epoch_shift_stages\":"
         << result.command_epoch_shift_stages
         << ",\"admm_us\":" << result.admm_us
         << ",\"admm_thread_cpu_us\":" << result.admm_thread_cpu_us
         << ",\"exact_rollout_us\":" << result.exact_rollout_us
         << ",\"nonlinear_validation_us\":"
         << result.nonlinear_validation_us
         << ",\"line_search_attempted\":"
         << (result.line_search_attempted ? "true" : "false")
         << ",\"line_search_trials\":" << result.line_search_trials
         << ",\"line_search_accepted\":"
         << (result.line_search_accepted ? "true" : "false")
         << ",\"line_search_merit_triggered\":"
         << (result.line_search_merit_triggered ? "true" : "false")
         << ",\"line_search_nominal_exact_valid\":"
         << (result.line_search_nominal_exact_valid ? "true" : "false")
         << ",\"line_search_nominal_objective\":";
    json_number(result.line_search_nominal_objective);
    json << ",\"line_search_skipped_budget\":"
         << (result.line_search_skipped_budget ? "true" : "false")
         << ",\"line_search_skipped_command_epoch\":"
         << (result.line_search_skipped_command_epoch ? "true" : "false")
         << ",\"line_search_full_step_failure_reason\":"
         << result.line_search_full_step_failure_reason
         << ",\"line_search_full_step_failure_stage\":"
         << result.line_search_full_step_failure_stage
         << ",\"line_search_last_failure_reason\":"
         << result.line_search_last_failure_reason
         << ",\"line_search_last_failure_stage\":"
         << result.line_search_last_failure_stage
         << ",\"line_search_selected_alpha\":"
         << result.line_search_selected_alpha
         << ",\"line_search_us\":" << result.line_search_us
         << ",\"solve_us\":" << result.total_solve_us
         << ",\"node_solve_us\":" << solve_us
         << ",\"cycle_us\":" << result.total_cycle_us
         << ",\"objective\":";
    json_number(result.nonlinear_objective);
    json << '}'
         << ",\"first_action\":["
         << result.published_steering_command << ','
         << result.published_throttle << ']'
         << ",\"nominal_first_action\":["
         << result.nominal_first_control.steering_target_rad << ','
         << result.nominal_first_control.throttle_command << ']'
         << ",\"linearization_failure\":["
         << result.linearization_failure_stage << ','
         << result.linearization_failure_reason << ','
         << result.linearization_failure_index << ']'
         << ",\"nominal_vs_candidate_progress_error_max_m\":";
    json_number(result.max_candidate_progress_error_m);
    json << ",\"nominal_vs_candidate_curvature_error_max_per_m\":";
    json_number(result.max_candidate_curvature_error_per_m);
    json << ",\"nominal_vs_candidate_left_bound_error_max_m\":";
    json_number(result.max_candidate_left_bound_error_m);
    json << ",\"nominal_vs_candidate_right_bound_error_max_m\":";
    json_number(result.max_candidate_right_bound_error_m);
    json << ",\"candidate_corridor_slack_by_stage_m\":[";
    for (int stage = 1;
         stage < result.candidate_path_delta.sample_count &&
             stage <= PREDICTION_HORIZON;
         ++stage) {
        if (stage > 1) json << ',';
        json << "{\"stage\":" << stage << ",\"lower\":";
        json_number(result.candidate_path_delta
            .lower_corridor_slack_m[stage]);
        json << ",\"upper\":";
        json_number(result.candidate_path_delta
            .upper_corridor_slack_m[stage]);
        json << '}';
    }
    json << ']';
    json << ",\"minimum_predicted_corridor_slack_m\":";
    json_number(result.minimum_predicted_corridor_slack_m);
    json << ",\"lateral_accel_proxy_mps2\":";
    json_number(result.lateral_accel_proxy_mps2);
    json << ",\"lateral_accel_proxy_stage\":"
         << result.lateral_accel_proxy_stage
         << ",\"lateral_accel_proxy_by_stage_mps2\":[";
    for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
        if (k > 0) json << ',';
        json_number(result.lateral_accel_proxy_by_stage_mps2[k]);
    }
    json << ']';
    json << ",\"nominal_controls\":[";
    for (int k = 0; k < result.nominal_control_count; ++k) {
        if (k > 0) json << ',';
        json << '['
             << result.nominal_controls[k].steering_target_rad << ','
             << result.nominal_controls[k].throttle_command << ']';
    }
    json << ']';
    if (result.candidate_control_count > 0) {
        const bool accepted_controls =
            result.status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
            result.status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
            result.status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH ||
            result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX ||
            result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN;
        json << (accepted_controls ? ",\"candidate_controls\":["
                                  : ",\"rejected_candidate_controls\":[");
        for (int k = 0; k < result.candidate_control_count; ++k) {
            if (k > 0) json << ',';
            json << '['
                 << result.candidate_controls[k].steering_target_rad
                 << ',' << result.candidate_controls[k].throttle_command
                 << ']';
        }
        json << ']';
    }
    json << ",\"predictions\":[";

    const bool accepted =
        result.status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
        result.status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
        result.status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH ||
        result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX ||
        result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN;
    if (!accepted) {
        json << "],\"diagnostic_recomputed_plant_steps\":0"
             << ",\"diagnostic_publish_steady_ns\":"
             << steady_time_ns() << '}';
        diagnostic_timing.end_json_build();
        diagnostic_timing.begin_enqueue();
        enqueue_diagnostic_stream(std::move(json));
        diagnostic_timing.end_enqueue();
        return;
    }

    bool first = true;
    const MpcRtiNominal_t &diagnostic_nominal =
        (result.status == MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX ||
         result.status == MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN)
            ? recovered_action_revalidated_nominal_
            : rti_memory_.nominal;
    for (int output_index = 0;
         output_index < result.diagnostic_output_count &&
             output_index < MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT;
         ++output_index) {
        const MpcRtiDiagnosticStageOutput_t &cached_output =
            result.diagnostic_outputs[output_index];
        const int step = cached_output.stage;
        if (!cached_output.valid || step < 1 ||
            step > diagnostic_nominal.horizon) continue;
        if (!first) json << ',';
        first = false;
        const MpcRtiState_t &predicted = diagnostic_nominal.states[step];
        MpcTrajectorySample_t sample{};
        const double predicted_s = diagnostic_nominal.progress[step];
        if (!mpc_trajectory_sample(trajectory_.data(), trajectory_.size(),
                track_length_m_, predicted_s, &sample)) {
            diagnostic_timing.end_json_build();
            publish_cycle_failure("prediction_reference_unavailable");
            return;
        }
        const double reference_speed = std::min(
            sample.speed,
            static_cast<double>(
                rti_config_.model.active_speed_ceiling_mps));
        const double feedforward = clamp(
            vehicle_model_steering_for_curvature(
                static_cast<float>(sample.curvature)),
            -rti_config_.model.max_steering_rad,
            rti_config_.model.max_steering_rad);
        const MpcModelControl_t &control =
            diagnostic_nominal.controls[step - 1];
        const MpcModelOutput_t &plant_output = cached_output.output;
        json << "{\"n\":" << step << ",\"s_m\":" << predicted_s
            << ",\"state\":[" << predicted.plant.e_y << ','
            << predicted.plant.e_psi << ',' << predicted.plant.u << ','
            << predicted.plant.v << ',' << predicted.plant.r << ','
            << predicted.plant.delayed_steering_command_1 << ','
            << predicted.plant.delayed_steering_command_2 << ','
            << predicted.plant.actual_steering_angle << ']'
            << ",\"source_wheel_omega_radps\":["
            << predicted.plant.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_FL]
            << ',' << predicted.plant.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_FR]
            << ',' << predicted.plant.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_RL]
            << ',' << predicted.plant.source_plant_internal_state[
                MPC_PLANT_STATE_WHEEL_OMEGA_RR]
            << "]"
            << ",\"source_longitudinal_slip\":["
            << plant_output.longitudinal_slip[0] << ','
            << plant_output.longitudinal_slip[1] << ','
            << plant_output.longitudinal_slip[2] << ','
            << plant_output.longitudinal_slip[3]
            << "]"
            << ",\"source_lateral_slip\":["
            << plant_output.lateral_slip[0] << ','
            << plant_output.lateral_slip[1] << ','
            << plant_output.lateral_slip[2] << ','
            << plant_output.lateral_slip[3]
            << "]"
            << ",\"source_normal_load_n\":["
            << plant_output.normal_load_n[0] << ','
            << plant_output.normal_load_n[1] << ','
            << plant_output.normal_load_n[2] << ','
            << plant_output.normal_load_n[3]
            << "]"
            << ",\"reference\":[0,0," << reference_speed
            << ",0," << sample.curvature * reference_speed << ','
            << feedforward << ']'
            << ",\"input\":[" << control.steering_target_rad << ','
            << control.throttle_command << ']'
            << ",\"corridor\":[" << sample.left_bound << ','
            << sample.right_bound << "]}";
    }
    json << "],\"diagnostic_recomputed_plant_steps\":0"
         << ",\"diagnostic_cached_source_output_count\":"
         << result.diagnostic_output_count;
    json << ",\"diagnostic_publish_steady_ns\":"
         << steady_time_ns() << '}';
    diagnostic_timing.end_json_build();
    diagnostic_timing.begin_enqueue();
    enqueue_diagnostic_stream(std::move(json));
    diagnostic_timing.end_enqueue();
}


} // namespace f1tenth_mpc
