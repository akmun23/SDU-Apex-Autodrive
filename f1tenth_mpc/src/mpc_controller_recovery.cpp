// Command-time recovery and retained-prefix selection.
// All acceptance requires the same exact source-plant/corridor gates.
#include "mpc_controller_node_impl.hpp"

namespace f1tenth_mpc {

bool MpcControllerNode::try_publish_prior_plan(
    const MpcRtiState_t &state,
    double progress,
    int64_t callback_steady_ns,
    int64_t control_stamp_ns,
    MpcRtiCycleResult_t *result,
    const char **fault_reason,
    bool validate_full_remaining_suffix)
{
    if (fault_reason) *fault_reason = nullptr;
    if (fallback_episode_.requires_full_horizon(
            fallback_max_short_prefix_actions_) &&
        !validate_full_remaining_suffix) {
        if (fault_reason) *fault_reason =
            "retained short prefix exhausted; require full-horizon "
            "reacquisition or newly solved MPC plan";
        return false;
    }
    prior_plan_revalidation_attempted_ = true;
    if (validate_full_remaining_suffix) {
        full_prior_suffix_revalidation_attempted_ = true;
        full_prior_suffix_exact_valid_ = false;
        full_prior_suffix_timing_rejected_ = false;
        full_prior_suffix_validation_horizon_ = 0;
        full_prior_suffix_validation_status_ = -1;
        full_prior_suffix_failure_stage_ = -1;
        full_prior_suffix_validation_us_ = 0.0;
    }
    const MpcRtiNominal_t &previous = certified_plan_;
    if (!result || !previous.valid || previous.horizon < 2) {
        if (fault_reason)
            *fault_reason = "no unconsumed accepted MPC plan remains";
        return false;
    }

    /* Always validate through the first interval where the command being
     * published affects the plant (two queued packets plus one interval).
     * The opt-in rejection policy asks for the stronger certificate over
     * every remaining action in the retained full-horizon plan. */
    const int next_control_index = prior_plan_control_index_ + 1;
    const int remaining_controls = previous.horizon - next_control_index;
    if (remaining_controls < kCommandEffectValidationHorizon) {
        if (fault_reason)
            *fault_reason = "accepted MPC plan has insufficient actions "
                "to validate through the delayed command effect";
        return false;
    }
    if (fallback_episode_.requires_full_horizon(
            fallback_max_short_prefix_actions_) &&
        remaining_controls < PREDICTION_HORIZON) {
        if (fault_reason) *fault_reason =
            "expired retained plan cannot supply a full 750ms certificate";
        return false;
    }
    const int validation_horizon = validate_full_remaining_suffix
        ? remaining_controls : kCommandEffectValidationHorizon;
    if (validate_full_remaining_suffix)
        full_prior_suffix_validation_horizon_ = validation_horizon;

    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    for (int i = 0; i < validation_horizon; ++i) {
        controls[i] = previous.controls[next_control_index + i];
        if (!std::isfinite(controls[i].steering_target_rad) ||
            !std::isfinite(controls[i].throttle_command) ||
            controls[i].throttle_command <
                MPC_THROTTLE_MIN_DRIVE - 1.0e-6f ||
            controls[i].throttle_command > 1.0f + 1.0e-6f ||
            std::abs(controls[i].steering_target_rad) >
                actuator_steering_limit_rad_ + 1.0e-6) {
            if (fault_reason)
                *fault_reason = "accepted-plan prefix violates the "
                    "production drive-command envelope";
            return false;
        }
        /* Match publish_command(): clamp only float-rounding overrun;
         * validate the exact bounded sequence that will be commanded. */
        controls[i].steering_target_rad = static_cast<float>(clamp(
            controls[i].steering_target_rad,
            -actuator_steering_limit_rad_, actuator_steering_limit_rad_));
        controls[i].throttle_command = static_cast<float>(clamp(
            controls[i].throttle_command, MPC_THROTTLE_MIN_DRIVE, 1.0));
    }

    MpcRtiNominal_t revalidated{};
    int failure_stage = -1;
    prior_plan_validated_horizon_ = validation_horizon;
    const auto validation_start = std::chrono::steady_clock::now();
    const MpcRtiRolloutStatus_t rollout_status =
        mpc_rti_validate_candidate_plan_with_diagnostics(
            &state, progress, controls,
            trajectory_.data(), trajectory_.size(), track_length_m_,
            validation_horizon, TIME_STEP_SECONDS,
            &rti_config_, &revalidated,
            result->diagnostic_outputs,
            &result->diagnostic_output_count,
            &failure_stage);
    prior_plan_validation_status_ = static_cast<int>(rollout_status);
    prior_plan_validation_failure_stage_ = failure_stage;
    prior_plan_validation_us_ = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - validation_start).count();
    if (validate_full_remaining_suffix) {
        full_prior_suffix_validation_status_ =
            static_cast<int>(rollout_status);
        full_prior_suffix_failure_stage_ = failure_stage;
        full_prior_suffix_validation_us_ = prior_plan_validation_us_;
        full_prior_suffix_exact_valid_ =
            rollout_status == MPC_RTI_ROLLOUT_OK;
    }
    if (rollout_status != MPC_RTI_ROLLOUT_OK) {
        if (fault_reason)
            *fault_reason = validate_full_remaining_suffix
                ? "retained MPC plan suffix failed exact current-state "
                    "rollout/corridor validation"
                : "accepted-plan prefix failed exact delayed-command "
                    "rollout/corridor validation";
        RCLCPP_WARN(get_logger(),
            "Prior MPC plan not reused: exact %d ms rollout "
            "status=%d stage=%d",
            validation_horizon * 25,
            static_cast<int>(rollout_status), failure_stage);
        return false;
    }
    prior_plan_exact_valid_ = true;
    recovered_action_revalidated_nominal_ = revalidated;
    result->nonlinear_objective = revalidated.validation_objective;
    result->minimum_predicted_corridor_slack_m =
        revalidated.validation_minimum_corridor_slack_m;
    prior_plan_validated_control_index_ = next_control_index;
    prior_plan_actions_remaining_ = previous.horizon -
        next_control_index - 1;

    const char * timing_fault_reason = nullptr;
    if (!cycle_time_is_valid(callback_steady_ns, &timing_fault_reason)) {
        prior_plan_timing_rejected_ = true;
        if (validate_full_remaining_suffix)
            full_prior_suffix_timing_rejected_ = true;
        if (fault_reason) *fault_reason = timing_fault_reason;
        return false;
    }
    const MpcModelControl_t &first = revalidated.controls[0];
    MpcModelControl_t next_warm_start[PREDICTION_HORIZON]{};
    const int warm_start_count =
        f1tenth_mpc::copy_retained_plan_warm_start(previous.controls,
            previous.horizon, next_control_index, PREDICTION_HORIZON,
            next_warm_start);
    next_warm_start[0] = first;
    const char *publication_fault_reason = nullptr;
    const int64_t command_stamp_ns = publish_command(
        first.steering_target_rad, first.throttle_command,
        callback_steady_ns,
        &publication_fault_reason);
    if (command_stamp_ns == 0) {
        if (last_publication_gate_elapsed_ms_ >=
                end_to_end_cycle_budget_ms_)
            prior_plan_timing_rejected_ = true;
        if (fault_reason) *fault_reason = publication_fault_reason ?
            publication_fault_reason :
            "revalidated prior-plan action violates the drive envelope";
        return false;
    }
    /* Commit the issued-sequence warm start only after publish_command()
     * confirms that this action entered the actuator command history. */
    if (!mpc_rti_memory_set_warm_start_controls_at_epoch(&rti_memory_,
            next_warm_start, warm_start_count, active_command_epoch_ns_)) {
        if (fault_reason)
            *fault_reason = "published prior action could not seed the "
                "next MPC nominal sequence";
        return false;
    }
    /* The issued recovery sequence now differs from the rejected solve's
     * nominal. Its ADMM primal/dual iterates can oppose the shifted command
     * seed, so cold-start only the solver history and retain both plans. */
    mpc_rti_memory_reset_solver_state(&rti_memory_);
    result->admm_state_reset_for_recovery = 1;

    /* Keep the original converged sequence and advance its cursor only
     * after publishing this exactly validated action. This preserves the
     * remaining plan if another isolated solve rejection follows. */
    prior_plan_control_index_ = next_control_index;
    consecutive_candidate_prefix_actions_ = 0;
    ++consecutive_prior_plan_actions_;
    last_command_publication_lead_ns_ = command_stamp_ns >= control_stamp_ns
        ? command_stamp_ns - control_stamp_ns : 0;
    result->status = MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN;
    result->first_control = first;
    result->published_steering_command = first.steering_target_rad;
    result->published_throttle = first.throttle_command;
    result->candidate_control_count = 0;
    return true;
}


bool MpcControllerNode::try_publish_source_reacquisition(
    const MpcRtiState_t &state, double progress,
    int64_t callback_steady_ns, int64_t control_stamp_ns,
    MpcRtiCycleResult_t *result)
{
    if (!result || fallback_max_short_prefix_actions_ == 0 ||
        !fallback_episode_.requires_full_horizon(
            fallback_max_short_prefix_actions_)) return false;
    const int64_t budget_ns = callback_steady_ns +
        static_cast<int64_t>(end_to_end_cycle_budget_ms_ * 1.e6);
    if (steady_time_ns() >= budget_ns - 3'000'000) return false;
    MpcRtiNominal_t seed{};
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    if (!mpc_rti_build_nominal(&state, progress, nullptr,
            trajectory_.data(), trajectory_.size(), track_length_m_,
            TIME_STEP_SECONDS, PREDICTION_HORIZON, &rti_config_.model,
            &seed, refs)) return false;
    // Native witness commands alone may be unsuitable when the vehicle
    // has accumulated lateral/yaw deviation. Search causal steering and
    // torque perturbations decaying back to the witness over 300-500ms;
    // every option is source-plant checked for all 30 stages. Use the
    // exact physical terminal state and keep the least-disruptive viable
    // candidate, not the fastest local short prefix.
    MpcRtiNominal_t validated{};
    double best_risk = INFINITY;
    bool found = false;
    const float ey_direction = state.plant.e_y >= 0.0f ? -1.0f : 1.0f;
    const float steering_offsets[] = {0.f, ey_direction*0.08f,
        ey_direction*0.16f, ey_direction*0.24f,
        -ey_direction*0.08f, -ey_direction*0.16f};
    const float throttle_offsets[] = {0.f, -0.12f, -0.24f};
    int trial_count = 0;
    // Reacquisition is triggered by an episode of short certified actions.
    // First locate the SOURCE witness's exact future wall violation from
    // the current physical state and queued actuators. The old search
    // changed only the beginning of the command sequence; a wall failure
    // at a late stage may require turning earlier in that future section.
    // Do not change source collision constraints or publish this witness
    // unless a complete replacement passes exact 30-stage validation.
    if (viability_stage_targeted_recovery_enabled_) {
        MpcRtiNominal_t seed_checked{};
        int source_failure_stage = -1;
        const MpcRtiRolloutStatus_t seed_status =
            mpc_rti_validate_candidate_plan(
                &state, progress, seed.controls, trajectory_.data(),
                trajectory_.size(), track_length_m_, PREDICTION_HORIZON,
                TIME_STEP_SECONDS, &rti_config_, &seed_checked,
                &source_failure_stage);
        ++trial_count;
        if (seed_status == MPC_RTI_ROLLOUT_OK &&
            f1tenth_mpc::terminal_recovered(
                seed_checked.states[PREDICTION_HORIZON].plant)) {
            validated = seed_checked;
            best_risk = f1tenth_mpc::viability_score(
                seed_checked, seed.speed_reference[PREDICTION_HORIZON]);
            found = true;
        } else if (seed_status == MPC_RTI_ROLLOUT_CORRIDOR &&
                   source_failure_stage >= 8) {
            const int start = std::max(0, source_failure_stage - 16);
            const int stop = std::max(start + 2, source_failure_stage - 2);
            for (const float steering : {0.12f, -0.12f, 0.20f, -0.20f}) {
                if (trial_count >= 5 ||
                    steady_time_ns() >= budget_ns - 15'000'000) break;
                MpcModelControl_t controls[PREDICTION_HORIZON]{};
                f1tenth_mpc::targeted_viability_controls(
                    seed.controls, controls, PREDICTION_HORIZON,
                    start, stop, steering, -0.10f,
                    static_cast<float>(actuator_steering_limit_rad_));
                MpcRtiNominal_t candidate{};
                int failed_stage = -1;
                ++trial_count;
                if (mpc_rti_validate_candidate_plan(
                        &state, progress, controls,
                        trajectory_.data(), trajectory_.size(),
                        track_length_m_, PREDICTION_HORIZON,
                        TIME_STEP_SECONDS, &rti_config_, &candidate,
                        &failed_stage) != MPC_RTI_ROLLOUT_OK ||
                    !f1tenth_mpc::terminal_recovered(
                        candidate.states[PREDICTION_HORIZON].plant))
                    continue;
                const double risk = f1tenth_mpc::viability_score(
                    candidate, seed.speed_reference[PREDICTION_HORIZON]);
                if (risk < best_risk) {
                    best_risk = risk;
                    validated = candidate;
                    found = true;
                }
            }
        }
    }
    // A longer correction matters when the unsafe physical state lies
    // near stage 22-30 rather than in the first three command intervals.
    for (const int taper_steps : {20, 12}) {
        for (const float steer : steering_offsets) {
            for (const float torque : throttle_offsets) {
                // Do not steal the entire callback from a fresh QP solve.
                // Protect the core 30-stage QP solve. Earlier 24-trial
                // emergency search left only 5 ADMM iterations and a
                // 3.44 primal residual. Never consume the solver reserve
                // solely attempting to recreate a reference trajectory.
                if (trial_count >= 12 ||
                    steady_time_ns() >= budget_ns - 15'000'000)
                    break;
                MpcModelControl_t controls[PREDICTION_HORIZON]{};
                f1tenth_mpc::taper_viability_controls(seed.controls,
                    controls, PREDICTION_HORIZON, taper_steps,
                    steer, torque,
                    static_cast<float>(actuator_steering_limit_rad_));
                ++trial_count;
                MpcRtiNominal_t candidate{};
                int failed_stage = -1;
                if (mpc_rti_validate_candidate_plan(&state, progress,
                        controls, trajectory_.data(), trajectory_.size(),
                        track_length_m_, PREDICTION_HORIZON,
                        TIME_STEP_SECONDS, &rti_config_, &candidate,
                        &failed_stage) != MPC_RTI_ROLLOUT_OK ||
                    !f1tenth_mpc::terminal_recovered(
                        candidate.states[PREDICTION_HORIZON].plant))
                    continue;
                const float target_speed =
                    seed.speed_reference[PREDICTION_HORIZON];
                const double risk = f1tenth_mpc::viability_score(
                    candidate, target_speed) +
                    0.10 * std::abs(steer) + 0.10 * std::abs(torque);
                if (risk < best_risk) {
                    best_risk = risk;
                    validated = candidate;
                    found = true;
                }
            }
        }
    }
    if (!found) return false;
    const char *fault_reason=nullptr;
    if (!cycle_time_is_valid(callback_steady_ns, &fault_reason)) return false;
    const auto first=validated.controls[0];
    const int64_t issued_ns=publish_command(
        first.steering_target_rad,first.throttle_command,
        callback_steady_ns,&fault_reason);
    if (!issued_ns) return false;
    rti_memory_.nominal=validated;
    mpc_rti_memory_set_warm_start_controls_at_epoch(&rti_memory_,
        validated.controls,PREDICTION_HORIZON, active_command_epoch_ns_);
    mpc_rti_memory_reset_solver_state(&rti_memory_);
    certified_plan_=validated;
    certified_plan_epoch_ns_=active_command_epoch_ns_;
    prior_plan_control_index_=0;
    consecutive_prior_plan_actions_=0;
    consecutive_candidate_prefix_actions_=0;
    clear_validated_recovery_prefix();
    result->status=MPC_RTI_CYCLE_ACCEPTED_VIABILITY_RECOVERY;
    result->admm_state_reset_for_recovery=1;
    result->first_control=first;
    result->published_steering_command=first.steering_target_rad;
    result->published_throttle=first.throttle_command;
    result->minimum_predicted_corridor_slack_m=
        validated.validation_minimum_corridor_slack_m;
    result->nonlinear_objective=validated.validation_objective;
    result->candidate_control_count=PREDICTION_HORIZON;
    std::copy_n(validated.controls,PREDICTION_HORIZON,
                result->candidate_controls);
    last_command_publication_lead_ns_=issued_ns>=control_stamp_ns
        ? issued_ns-control_stamp_ns : 0;
    RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,
        "MPC fallback episode reacquired a full 750ms source-plant plan "
        "after %d short prefix actions (slack %.4fm)",
        fallback_episode_.consecutive_short_actions(),
        validated.validation_minimum_corridor_slack_m);
    return true;
}


bool MpcControllerNode::try_publish_viability_recovery(
    const MpcRtiState_t &state, double progress,
    int64_t callback_steady_ns, int64_t control_stamp_ns,
    MpcRtiCycleResult_t *result)
{
    if (!viability_recovery_enabled_ || !result ||
        result->nominal_control_count != PREDICTION_HORIZON) return false;
    const bool solver_accepted =
        result->status == MPC_RTI_CYCLE_ACCEPTED_OPTIMAL ||
        result->status == MPC_RTI_CYCLE_ACCEPTED_FEASIBLE ||
        result->status == MPC_RTI_CYCLE_ACCEPTED_LINE_SEARCH;
    if (solver_accepted &&
        (viability_recovery_on_rejection_only_ ||
         (viability_proactive_after_fallback_only_ &&
          fallback_episode_.recent_short_actions() == 0)))
        return false;
    if (!solver_accepted &&
        result->status != MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT &&
        !(viability_steady_bridge_enabled_ &&
          result->status == MPC_RTI_CYCLE_REJECTED_SOLVER))
        return false;
    constexpr int horizon = PREDICTION_HORIZON;
    // A rare strongly damped corridor-invalid full step is a useful
    // predictor of next-cycle full-horizon infeasibility. Require both
    // a small certified reserve and a physically accepted short step.
    const bool line_search_wall_warning =
        viability_line_search_wall_warning_enabled_ &&
        f1tenth_mpc::line_search_wall_early_warning(
            result->status,
            result->line_search_full_step_failure_reason,
            result->line_search_selected_alpha,
            result->minimum_predicted_corridor_slack_m,
            0.25f,0.025f);
    const bool geometric_anticipatory_warning =
        viability_geometric_anticipatory_enabled_ &&
        f1tenth_mpc::line_search_wall_early_warning(
            result->status,
            result->line_search_full_step_failure_reason,
            result->line_search_selected_alpha,
            result->minimum_predicted_corridor_slack_m,
            0.25f,0.025f);
    const bool extension_probe =
        viability_terminal_extension_select_enabled_ &&
        solver_accepted && rti_memory_.nominal.valid &&
        rti_memory_.nominal.horizon==horizon &&
        std::isfinite(result->minimum_predicted_corridor_slack_m) &&
        result->minimum_predicted_corridor_slack_m < 0.025f;
    const bool terminal_extension_warning=extension_probe &&
        !source_terminal_extension_safe(rti_memory_.nominal);
    // When a complete 750+75ms continuation is already source-valid,
    // preserve that high-speed accepted plan rather than overriding it
    // just to gain a few millimetres of approximate corridor slack.
    if(extension_probe && !terminal_extension_warning &&
       !f1tenth_mpc::terminal_instability_predicted(
            rti_memory_.nominal.states[horizon].plant))
        return false;
    // The accepted RTI candidate was ALREADY checked against all 30 exact
    // source-plant/corridor stages. When it has both terminal stability and
    // enough wall clearance there is no reason to repeat the expensive
    // identical nonlinear rollout just to decide that rescue is unnecessary.
    // On small clearance or terminal instability retain the independent
    // command-time revalidation and full-horizon rescue logic below.
    if (solver_accepted && rti_memory_.nominal.valid &&
        rti_memory_.nominal.horizon == horizon &&
        std::isfinite(result->minimum_predicted_corridor_slack_m) &&
        result->minimum_predicted_corridor_slack_m >=
            viability_lookahead_reserve_m_ &&
        !f1tenth_mpc::terminal_instability_predicted(
            rti_memory_.nominal.states[horizon].plant) &&
        !line_search_wall_warning && !terminal_extension_warning &&
        !geometric_anticipatory_warning)
        return false;
    // Leave time for the publication gate, actuator history and diagnostics.
    const int64_t deadline_ns = callback_steady_ns +
        static_cast<int64_t>(end_to_end_cycle_budget_ms_ * 1.0e6);
    constexpr int64_t kPublicationReserveNs = 2'500'000;
    if (steady_time_ns() >= deadline_ns - kPublicationReserveNs)
        return false;

    MpcRtiCycleConfiguration_t trial_configuration = rti_config_;
    if (result->comparison_objective.valid)
        trial_configuration.objective_schedule = &result->comparison_objective;
    MpcModelControl_t original[horizon]{};
    if (solver_accepted && rti_memory_.nominal.valid &&
        rti_memory_.nominal.horizon == horizon) {
        std::copy_n(rti_memory_.nominal.controls, horizon, original);
    } else {
        std::copy_n(result->nominal_controls, horizon, original);
    }
    // Accepted SQP passes have already performed the exact full 750 ms
    // physical rollout. Reuse that certified plan and its minimum slack.
    // A rejected solve still requires an independent source validation.
    MpcRtiNominal_t original_plan;
    const MpcRtiNominal_t *const accepted_plan =
        solver_accepted && rti_memory_.nominal.valid &&
        rti_memory_.nominal.horizon == horizon &&
        std::isfinite(rti_memory_.nominal.validation_minimum_corridor_slack_m)
            ? &rti_memory_.nominal : nullptr;
    int stage = -1;
    const bool base_valid = accepted_plan != nullptr ||
        mpc_rti_validate_candidate_plan(
            &state, progress, original, trajectory_.data(),
            trajectory_.size(), track_length_m_, horizon,
            TIME_STEP_SECONDS, &trial_configuration,
            &original_plan, &stage) == MPC_RTI_ROLLOUT_OK;
    const MpcRtiNominal_t &base_plan = accepted_plan
        ? *accepted_plan : original_plan;
    // A nominal can have stable terminal dynamics but skim the wall so
    // closely that normal sensor/model uncertainty destroys feasibility at
    // the NEXT control tick. Monitor the *whole* 750 ms rollout's minimum
    // slack and search for a higher-reserve successor before this happens.
    // The nominal is a speed target, not a safe terminal state: increase
    // body-wall reserve while the full source-plant horizon is still viable.
    const float lookahead_reserve_m =
        static_cast<float>(viability_lookahead_reserve_m_);
    constexpr float kUsefulClearanceGainM = 0.005f;
    const bool nominal_instability = !base_valid ||
        f1tenth_mpc::terminal_instability_predicted(
            base_plan.states[horizon].plant);
    const bool near_boundary = base_valid &&
        (base_plan.validation_minimum_corridor_slack_m <
            lookahead_reserve_m || line_search_wall_warning ||
            terminal_extension_warning ||
            geometric_anticipatory_warning);
    if (!nominal_instability && !near_boundary) return false;

    MpcTrajectorySample_t target{};
    const float speed_reference = mpc_trajectory_sample(
            trajectory_.data(), trajectory_.size(), track_length_m_,
            base_valid ? base_plan.progress[horizon] : progress,
            &target)
        ? static_cast<float>(std::min(target.speed, static_cast<double>(
            rti_config_.model.active_speed_ceiling_mps)))
        : state.plant.u;
    // On accepted SQP passes original_plan was not rolled out: the
    // exact-validated accepted_plan is base_plan. Never score an
    // uninitialized candidate when deciding whether recovery is better.
    // Source-aware recovery is a generic capability of the loaded raceline,
    // never selected by track name. If witness data is unavailable, preserve
    // the historical physical-recovery ranking without any approximation.
    const bool witness_recovery_available =
        viability_source_witness_recovery_enabled_ &&
        !trajectory_.empty() && trajectory_.front().source_witness_valid;
    const auto risk_of = [&](const MpcRtiNominal_t &plan) {
        return f1tenth_mpc::viability_score_with_reserve(
            plan, speed_reference, witness_recovery_available,
            lookahead_reserve_m);
    };
    const double baseline_risk = base_valid ? risk_of(base_plan) : INFINITY;
    double selected_risk = INFINITY;
    float selected_clearance_m = -INFINITY;
    MpcRtiNominal_t selected{};
    int tested = 0;
    // Leave accepted racing dynamics alone unless the 750ms physical
    // envelope is about to become unsafe. Experimental mild early-warning
    // recovery uses small steering-only changes; it does not add artificial
    // speed clipping or a blanket stronger tracking cost. The historical
    // recovery search remains exactly unchanged by default.
    const bool mild_near_wall =
        viability_minimal_nearwall_recovery_enabled_ &&
        near_boundary && !nominal_instability;
    constexpr float steer_offsets[] = {0.12f, -0.12f, 0.18f, -0.18f};
    constexpr float mild_offsets[] = {
        0.04f, -0.04f, 0.08f, -0.08f, 0.12f, -0.12f};
    constexpr int taper_durations[] = {12, 16};
    constexpr float torque_offsets[] = {-0.08f, 0.0f};
    bool selected_blended_rescue=false;
    bool selected_geometric_rescue=false;
    // Distinguish a witness-derived *published* plan from trials which
    // were evaluated but lost to another source-validated recovery seed.
    bool selected_witness_rescue=false;
    // When a rejected SQP step has no exact-valid 750ms continuation,
    // try a small geometric-feasibility seed bank. This recovers the
    // solver's search outside a locally invalid command sequence without
    // changing normal accepted racing solutions. No precomputed map
    // control or source future truth is used: only path curvature, the
    // current physical speed, and the two-slot actuator queue in the
    // exact source-plant validator.
    if (!selected_blended_rescue &&
        ((viability_geometric_feasibility_seed_enabled_ &&
         !solver_accepted && !base_valid &&
         result->status==MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT) ||
        (viability_geometric_anticipatory_enabled_ &&
         geometric_anticipatory_warning && base_valid))) {
        float feedforward[2][10]{};
        bool sampled=true;
        const double u=std::clamp(
            static_cast<double>(state.plant.u),0.0,
            static_cast<double>(rti_config_.model.active_speed_ceiling_mps));
        // The native reference is indexed by travelled distance, not
        // elapsed time. With 6-10% drive torque the source vehicle
        // rapidly slows; extrapolating its current speed for the entire
        // 750ms creates an early steering reversal and a terminal
        // heading error >.6rad in the IROS hairpin. Trial both the
        // original fast progress and a conservative expected progression.
        const int feed_count=
            viability_braking_aware_geometric_enabled_ ? 2 : 1;
        for(int feed=0;feed<feed_count && sampled;++feed) {
            const double progress_fraction=feed==0 ? 1.0 : .55;
            for(int knot=0;knot<10;++knot) {
                MpcTrajectorySample_t sample{};
                const double predicted_progress=progress+
                    progress_fraction*u*TIME_STEP_SECONDS*
                    (PREDICTION_HORIZON-1)*double(knot)/9.0;
                if (!mpc_trajectory_sample(
                        trajectory_.data(),trajectory_.size(),
                        track_length_m_,predicted_progress,&sample)) {
                    sampled=false;break;
                }
                feedforward[feed][knot]=vehicle_model_steering_for_curvature(
                    static_cast<float>(sample.curvature));
                if(!std::isfinite(feedforward[feed][knot])) {
                    sampled=false;break;
                }
            }
        }
        if(sampled) {
            struct GeometricArm {float gain,drive;int feed;};
            constexpr GeometricArm normal[]={
                {1.20f,.12f,0},{1.20f,.06f,0},
                {1.00f,.06f,0},{1.45f,.12f,0}};
            // Empirically viable from the saved 7.075s first-hairpin
            // source state. The pure source trajectory speed remains the
            // normal fast target; this set is searched ONLY on rejection.
            constexpr GeometricArm braking[]={
                {1.00f,.08f,1},{1.20f,.08f,1},
                {1.00f,.06f,1},{1.20f,.06f,1},
                {1.20f,.12f,0},{1.20f,.06f,0},
                {1.45f,.12f,0}};
            const GeometricArm *const arms=
                viability_braking_aware_geometric_enabled_ ?
                braking : normal;
            const int arm_count=viability_braking_aware_geometric_enabled_
                ? int(sizeof(braking)/sizeof(braking[0]))
                : int(sizeof(normal)/sizeof(normal[0]));
            for(int option=0;option<arm_count;++option) {
                if(tested>=viability_recovery_max_trials_ ||
                   steady_time_ns()>=deadline_ns-
                       (geometric_anticipatory_warning ?
                        5'000'000 : kPublicationReserveNs))
                    break;
                MpcModelControl_t candidate[horizon]{};
                const auto &arm=arms[option];
                if(!f1tenth_mpc::geometric_feasibility_controls(
                       feedforward[arm.feed],arm.gain,arm.drive,
                       static_cast<float>(actuator_steering_limit_rad_),
                       candidate))
                    break;
                MpcRtiNominal_t validated{};
                int failure=-1;
                ++tested;
                if(mpc_rti_validate_candidate_plan(
                      &state,progress,candidate,trajectory_.data(),
                      trajectory_.size(),track_length_m_,horizon,
                      TIME_STEP_SECONDS,&trial_configuration,
                      &validated,&failure)!=MPC_RTI_ROLLOUT_OK ||
                    !f1tenth_mpc::terminal_recovered(
                        validated.states[horizon].plant))
                    continue;
                const double risk=risk_of(validated);
                if(geometric_anticipatory_warning &&
                   (validated.validation_minimum_corridor_slack_m <
                        std::max(.025f,base_plan.validation_minimum_corridor_slack_m+
                            .010f) ||
                    validated.states[horizon].plant.u <
                        base_plan.states[horizon].plant.u-4.f))
                    continue;
                if(risk<selected_risk) {
                    selected_risk=risk;
                    selected_clearance_m=
                        validated.validation_minimum_corridor_slack_m;
                    selected=validated;
                    selected_geometric_rescue=true;
                }
                // Emergency rescue is not another optimizer. Stop once
                // a source-certified terminal-stable horizon has >20mm
                // extra clearance, preserving the publication budget.
                if(validated.validation_minimum_corridor_slack_m >=
                    std::max(.020f,
                      static_cast<float>(viability_lookahead_reserve_m_)))
                    break;
            }
        }
    }

    // A rejected SQP may lie only a few control adjustments outside
    // the nonlinear feasible set. Independent saved-state source-plant
    // search showed that exact-valid convex blends between the rejected
    // nominal and a cheap geometric seed retain much more speed than
    // a pure low-drive geometric rescue. This bank makes no additional
    // QP solve, uses only present-state/track curvature, and NEVER
    // relaxes the physical wall, command queue, or actuator limits.
    if(viability_blended_geometric_recovery_enabled_ &&
       !solver_accepted &&
       result->status==MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT &&
       result->nominal_control_count==horizon) {
        float feedforward[10]{};
        bool sampled=true;
        const double speed=std::clamp(
            static_cast<double>(state.plant.u),0.0,
            static_cast<double>(rti_config_.model.active_speed_ceiling_mps));
        for(int knot=0;knot<10;++knot) {
            MpcTrajectorySample_t ref{};
            const double station=progress+
                speed*TIME_STEP_SECONDS*(horizon-1)*
                double(knot)/9.0;
            if(!mpc_trajectory_sample(trajectory_.data(),
                    trajectory_.size(),track_length_m_,station,&ref)) {
                sampled=false;break;
            }
            feedforward[knot]=vehicle_model_steering_for_curvature(
                static_cast<float>(ref.curvature));
            if(!std::isfinite(feedforward[knot])) {
                sampled=false;break;
            }
        }
        if(sampled) {
            struct BlendArm {
                float gain,drive,original_fraction;
            };
            constexpr BlendArm arms[]={
                {1.20f,.12f,.75f},
                {1.20f,.12f,.50f},
                {1.20f,.12f,.25f},
                {1.20f,.06f,.75f},
                {1.20f,.06f,.50f},
                {1.20f,.06f,.25f},
                {1.00f,.06f,.50f},
                {1.45f,.12f,.50f},
            };
            double best_objective=selected.valid &&
                std::isfinite(selected.validation_objective)
                ? selected.validation_objective : INFINITY;
            const int blend_trial_limit=selected.valid
                ? viability_recovery_max_trials_
                : std::max(0,viability_recovery_max_trials_-2);
            for(const auto &arm:arms) {
                if(tested>=blend_trial_limit ||
                   steady_time_ns()>=deadline_ns-kPublicationReserveNs)
                    break;
                MpcModelControl_t geometry[horizon]{};
                MpcModelControl_t controls[horizon]{};
                if(!f1tenth_mpc::geometric_feasibility_controls(
                       feedforward,arm.gain,arm.drive,
                       static_cast<float>(actuator_steering_limit_rad_),
                       geometry) ||
                   !f1tenth_mpc::blended_viability_controls(
                       geometry,original,horizon,arm.original_fraction,
                       static_cast<float>(actuator_steering_limit_rad_),
                       controls))
                    continue;
                MpcRtiNominal_t validated{};
                int failure=-1;
                ++tested;
                if(mpc_rti_validate_candidate_plan(
                       &state,progress,controls,trajectory_.data(),
                       trajectory_.size(),track_length_m_,horizon,
                       TIME_STEP_SECONDS,&trial_configuration,
                       &validated,&failure)!=MPC_RTI_ROLLOUT_OK ||
                   !f1tenth_mpc::terminal_recovered(
                       validated.states[horizon].plant) ||
                   validated.validation_minimum_corridor_slack_m<
                       (selected.valid ? .010f : .005f) ||
                   !std::isfinite(validated.validation_objective))
                    continue;
                if(validated.validation_objective<best_objective) {
                    best_objective=validated.validation_objective;
                    selected_risk=risk_of(validated);
                    selected_clearance_m=
                        validated.validation_minimum_corridor_slack_m;
                    selected=validated;
                    selected_blended_rescue=true;
                }
                // Prefer the first physically certified, margin-bearing
                // fast alternative over spending the remaining callback
                // on tiny objective improvements.
                if(validated.validation_minimum_corridor_slack_m>=.015f)
                    break;
            }
        }
    }

    // At tight full-speed corners, curvature-only recovery can turn the
    // wheel too late relative to the actual source vehicle. The IPOPT
    // witness carries *causal held steering and throttle*, and is the exact
    // source-feasible control sequence at zero tracking error. Reconstruct
    // its command-time sequence along the current rollout's progress and
    // perturb it with the current lateral/heading errors. Trial an early
    // steering correction plus moderate command-time drive reductions.
    // This is NOT an open-loop override: only a complete independently
    // validated 750ms physical horizon can replace the current controller.
    if (witness_recovery_available &&
        (!solver_accepted || near_boundary)) {
        struct WitnessArm {
            float progress_lead_m, lateral_gain, heading_gain, drive_scale;
        };
        constexpr WitnessArm arms[] = {
            {0.00f, 0.40f, 0.60f, 0.80f},
            {0.10f, 0.60f, 0.80f, 0.70f},
            {0.20f, 0.85f, 1.10f, 0.50f},
            {0.00f, 0.90f, 1.30f, 0.35f},
        };
        MpcModelControl_t witness[horizon]{};
        for (const WitnessArm &arm : arms) {
            if (tested >= viability_recovery_max_trials_ ||
                steady_time_ns() >= deadline_ns - kPublicationReserveNs)
                break;
            bool available = true;
            const float estimated_speed = std::clamp(
                state.plant.u, 0.f,
                rti_config_.model.active_speed_ceiling_mps);
            for (int k = 0; k < horizon; ++k) {
                MpcTrajectorySample_t reference{};
                // Only reuse nominal progress if it came from an actual
                // accepted full-horizon source rollout. Rejected nominal
                // progress is not a certificate of physically reachable
                // future time and must not drive the recovery witness.
                const double station =
                    (base_valid ? base_plan.progress[k] :
                     progress + estimated_speed * TIME_STEP_SECONDS * k) +
                    arm.progress_lead_m;
                if (!mpc_trajectory_sample(
                      trajectory_.data(), trajectory_.size(), track_length_m_,
                      station, &reference) ||
                    !reference.source_witness_valid) {
                    available = false;
                    break;
                }
                witness[k].steering_target_rad =
                    static_cast<float>(reference.source_steering_target);
                witness[k].throttle_command =
                    static_cast<float>(reference.source_throttle);
            }
            if (!available) break;
            MpcModelControl_t candidate[horizon]{};
            if (!f1tenth_mpc::witness_recovery_controls(
                    witness, candidate, horizon,
                    state.plant.e_y, state.plant.e_psi,
                    arm.lateral_gain, arm.heading_gain, arm.drive_scale,
                    static_cast<float>(actuator_steering_limit_rad_)))
                continue;
            MpcRtiNominal_t validated{};
            int failed_stage = -1;
            ++tested;
            if (mpc_rti_validate_candidate_plan(
                    &state, progress, candidate, trajectory_.data(),
                    trajectory_.size(), track_length_m_, horizon,
                    TIME_STEP_SECONDS, &trial_configuration,
                    &validated, &failed_stage) != MPC_RTI_ROLLOUT_OK ||
                !f1tenth_mpc::terminal_recovered(
                    validated.states[horizon].plant))
                continue;
            if (base_valid && !nominal_instability &&
                (validated.validation_minimum_corridor_slack_m <
                    base_plan.validation_minimum_corridor_slack_m + .005f ||
                 validated.states[horizon].plant.u <
                    base_plan.states[horizon].plant.u - 1.0f))
                continue;
            const double risk = risk_of(validated);
            if (risk < selected_risk) {
                selected_risk = risk;
                selected_clearance_m =
                    validated.validation_minimum_corridor_slack_m;
                selected = validated;
                selected_witness_rescue = true;
            }
        }
    }

    // If the exact SQP rollout failed far in the horizon, try to repair
    // the future maneuver rather than moving only today's steering.
    // Full-horizon physical validation remains mandatory. Budget and
    // candidate count are shared with the existing tapered search.
    if (viability_early_hazard_recovery_enabled_ &&
        !solver_accepted && result->nonlinear_failure_reason ==
            MPC_RTI_ROLLOUT_CORRIDOR &&
        result->nonlinear_failure.valid &&
        result->nonlinear_failure_stage >= 8) {
        const auto &hazard=result->nonlinear_failure;
        const float direction=f1tenth_mpc::corridor_corrective_sign(
            hazard.state.plant.e_y,hazard.lower_bound_m,
            hazard.upper_bound_m);
        if(direction!=0.0f) {
            // First two controls reflect the existing in-flight queue.
            // Trial controls begin at the fresh command. A correction
            // that cannot arrest the physically unavoidable queue is
            // rejected by the same full 750ms exact source rollout.
            for(const float magnitude : {0.12f,0.22f,0.32f,0.42f}) {
                if (tested >= viability_recovery_max_trials_ ||
                    steady_time_ns() >= deadline_ns-kPublicationReserveNs)
                    break;
                MpcModelControl_t candidate[horizon]{};
                f1tenth_mpc::early_hazard_viability_controls(
                    original,candidate,horizon,
                    result->nonlinear_failure_stage,
                    direction*magnitude,-0.12f,
                    static_cast<float>(actuator_steering_limit_rad_));
                MpcRtiNominal_t validated{};
                int failure=-1;
                ++tested;
                if(mpc_rti_validate_candidate_plan(
                       &state,progress,candidate,trajectory_.data(),
                       trajectory_.size(),track_length_m_,horizon,
                       TIME_STEP_SECONDS,&trial_configuration,
                       &validated,&failure)!=MPC_RTI_ROLLOUT_OK ||
                   !f1tenth_mpc::terminal_recovered(
                       validated.states[horizon].plant)) continue;
                const double risk=risk_of(validated);
                if(risk<selected_risk) {
                    selected_risk=risk;
                    selected = validated;
                    selected_witness_rescue = false;
                }
            }
        }
    }
    if (viability_stage_targeted_recovery_enabled_ &&
        !selected_blended_rescue && !selected_geometric_rescue &&
        !solver_accepted && result->nonlinear_failure_reason ==
            MPC_RTI_ROLLOUT_CORRIDOR &&
        result->nonlinear_failure_stage >= 8) {
        const int failed_stage = std::clamp(
            result->nonlinear_failure_stage, 8, horizon - 1);
        const float directions[] = {0.12f, -0.12f, 0.20f, -0.20f};
        for (const float correction : directions) {
            if (tested >= viability_recovery_max_trials_ ||
                steady_time_ns() >= deadline_ns - kPublicationReserveNs)
                break;
            MpcModelControl_t candidate[horizon]{};
            f1tenth_mpc::targeted_viability_controls(
                original, candidate, horizon,
                std::max(0, failed_stage - 16), failed_stage - 2,
                correction, -0.10f,
                static_cast<float>(actuator_steering_limit_rad_));
            MpcRtiNominal_t validated{};
            int failure = -1;
            ++tested;
            if (mpc_rti_validate_candidate_plan(
                    &state, progress, candidate, trajectory_.data(),
                    trajectory_.size(), track_length_m_, horizon,
                    TIME_STEP_SECONDS, &trial_configuration,
                    &validated, &failure) != MPC_RTI_ROLLOUT_OK ||
                !f1tenth_mpc::terminal_recovered(
                    validated.states[horizon].plant))
                continue;
            const double risk = risk_of(validated);
            if (risk < selected_risk) {
                selected_risk = risk;
                selected = validated;
                selected_witness_rescue = false;
            }
        }
    }

    const int steer_count = mild_near_wall ? 6 : 4;
    const int torque_count = mild_near_wall ? 1 : 2;
    for (const int steps : taper_durations) {
        // A source-validated proactive candidate already increased the
        // physical reserve and passed terminal checks. There is no reason
        // to spend another bounded solver interval replacing it with a
        // less predictable local steering taper.
        if(selected_blended_rescue || selected_geometric_rescue ||
           (geometric_anticipatory_warning && selected.valid)) break;
        for (int angle_index = 0; angle_index < steer_count; ++angle_index) {
            const float correction = mild_near_wall
                ? mild_offsets[angle_index] : steer_offsets[angle_index];
            for (int torque_index = 0; torque_index < torque_count;
                 ++torque_index) {
                const float torque = mild_near_wall
                    ? 0.0f : torque_offsets[torque_index];
                if (tested >= viability_recovery_max_trials_ ||
                    steady_time_ns() >= deadline_ns - kPublicationReserveNs)
                    break;
                MpcModelControl_t candidate[horizon]{};
                f1tenth_mpc::taper_viability_controls(
                    original, candidate, horizon, steps, correction, torque,
                    static_cast<float>(actuator_steering_limit_rad_));
                MpcRtiNominal_t validated{};
                int failure = -1;
                ++tested;
                if (mpc_rti_validate_candidate_plan(
                        &state, progress, candidate, trajectory_.data(),
                        trajectory_.size(), track_length_m_, horizon,
                        TIME_STEP_SECONDS, &trial_configuration,
                        &validated, &failure) != MPC_RTI_ROLLOUT_OK)
                    continue;
                if (!f1tenth_mpc::terminal_recovered(
                        validated.states[horizon].plant)) continue;
                if (terminal_extension_warning &&
                    !source_terminal_extension_safe(validated)) continue;
                const double risk = risk_of(validated);
                if (near_boundary && !nominal_instability) {
                    // For a nominal whose endpoint is already well behaved,
                    // retain its speed while prioritizing extra body-wall
                    // reserve. A tiny numerical improvement is not reason
                    // to override the high-speed racing plan.
                    const float slack =
                        validated.validation_minimum_corridor_slack_m;
                    if (slack < base_plan.validation_minimum_corridor_slack_m +
                            kUsefulClearanceGainM ||
                        validated.states[horizon].plant.u <
                            base_plan.states[horizon].plant.u - 0.5f)
                        continue;
                    if (mild_near_wall) {
                        // A useful physical margin gain is mandatory, but
                        // among safe alternatives prefer a smaller error
                        // in terminal motion and a smaller actuator change
                        // over maximizing clearance at any lap-time cost.
                        const double balanced_risk = risk +
                            0.5 * std::abs(candidate[0].steering_target_rad -
                                            original[0].steering_target_rad);
                        if (balanced_risk >= selected_risk) continue;
                        selected_risk = balanced_risk;
                    } else {
                        if (slack < selected_clearance_m + 1.e-4f &&
                            !(fabsf(slack - selected_clearance_m) <= 1.e-4f &&
                              risk < selected_risk))
                            continue;
                        selected_clearance_m = slack;
                        selected_risk = risk;
                    }
                } else {
                    if (risk >= selected_risk) continue;
                    selected_risk = risk;
                }
                selected = validated;
                selected_witness_rescue = false;
            }
        }
    }
    // Only if the tapered controls failed (or failed the required material
    // improvement), try a small number of steady actuator bridges. Never
    // displace a valid tapered racing solution with an unnecessarily
    // constant/slow bridge. This is map-independent and still requires the
    // same exact 750ms nonlinear body-corridor validation and deadline.
    const bool tapered_acceptable = selected.valid &&
        std::isfinite(selected_risk) &&
        (!base_valid || !nominal_instability ||
         selected_risk < 0.75 * baseline_risk);
    if (nominal_instability && viability_steady_bridge_enabled_ &&
        !tapered_acceptable) {
        int steady_tested = 0;
        MpcTrajectorySample_t local_reference{};
        const bool have_reference = mpc_trajectory_sample(
            trajectory_.data(), trajectory_.size(), track_length_m_,
            progress, &local_reference);
        const float feedforward = have_reference
            ? vehicle_model_steering_for_curvature(
                  static_cast<float>(local_reference.curvature))
            : original[0].steering_target_rad;
        const float steering_seeds[] = {
            state.plant.actual_steering_angle,
            feedforward,
            original[0].steering_target_rad};
        // At narrow high-curvature corners, 6%/10% drive alone can
        // overshoot the full-horizon body corridor. Search smaller positive
        // drive torques before consuming the remaining deadline on local
        // SQP variations. This is dimensionless control, not a map rule.
        const float throttle_seeds[] = {0.06f, 0.025f, 0.10f, 0.01f};
        for (const float steering : steering_seeds) {
            for (const float throttle : throttle_seeds) {
                if (steady_tested >= 8 ||
                    steady_time_ns() >= deadline_ns - kPublicationReserveNs)
                    break;
                MpcModelControl_t candidate[horizon]{};
                f1tenth_mpc::steady_viability_controls(
                    candidate, horizon, steering, throttle,
                    static_cast<float>(actuator_steering_limit_rad_));
                MpcRtiNominal_t validated{};
                int failed = -1;
                ++steady_tested;
                ++tested;
                if (mpc_rti_validate_candidate_plan(
                        &state, progress, candidate, trajectory_.data(),
                        trajectory_.size(), track_length_m_, horizon,
                        TIME_STEP_SECONDS, &trial_configuration,
                        &validated, &failed) != MPC_RTI_ROLLOUT_OK ||
                    !f1tenth_mpc::terminal_recovered(
                        validated.states[horizon].plant))
                    continue;
                const double risk = risk_of(validated);
                if (risk < selected_risk) {
                    selected_risk = risk;
                    selected = validated;
                    selected_witness_rescue = false;
                }
            }
        }
    }
    if (!selected.valid || !std::isfinite(selected_risk) ||
        (base_valid && nominal_instability &&
         selected_risk >= 0.75 * baseline_risk))
        return false;
    const char *fault_reason = nullptr;
    if (!cycle_time_is_valid(callback_steady_ns, &fault_reason))
        return false;
    const MpcModelControl_t first = selected.controls[0];
    const int64_t issued_ns = publish_command(
        first.steering_target_rad, first.throttle_command,
        callback_steady_ns, &fault_reason);
    if (!issued_ns) return false;
    // Preserve original clearance before overwriting the accepted
    // nominal: base_plan may alias rti_memory_.nominal.
    const float previous_slack_m = base_valid
        ? base_plan.validation_minimum_corridor_slack_m : NAN;
    // Only commit the newly validated plan AFTER the actual command was
    // published, so a rejected alternate cannot poison the next solve.
    rti_memory_.nominal = selected;
    mpc_rti_memory_set_warm_start_controls_at_epoch(
        &rti_memory_, selected.controls, selected.horizon, active_command_epoch_ns_);
    mpc_rti_memory_reset_solver_state(&rti_memory_);
    certified_plan_ = selected;
    certified_plan_epoch_ns_ = active_command_epoch_ns_;
    prior_plan_control_index_ = 0;
    clear_validated_recovery_prefix();
    consecutive_prior_plan_actions_ = 0;
    consecutive_candidate_prefix_actions_ = 0;
    result->status = MPC_RTI_CYCLE_ACCEPTED_VIABILITY_RECOVERY;
    result->admm_state_reset_for_recovery = 1;
    result->first_control = first;
    result->published_steering_command = first.steering_target_rad;
    result->published_throttle = first.throttle_command;
    result->minimum_predicted_corridor_slack_m =
        selected.validation_minimum_corridor_slack_m;
    result->nonlinear_objective = selected.validation_objective;
    result->candidate_control_count = horizon;
    std::copy_n(selected.controls, horizon, result->candidate_controls);
    result->diagnostic_output_count = 0;
    last_command_publication_lead_ns_ =
        issued_ns >= control_stamp_ns ? issued_ns - control_stamp_ns : 0;
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "MPC full-horizon viability repair: trial_count=%d "
        "trigger=%s terminal_v=%.3f terminal_epsi=%.3f "
        "slack=%.4f (previous=%.4f) selected_seed=%s",
        tested, nominal_instability ? "instability" : "near-wall",
        selected.states[horizon].plant.v,
        selected.states[horizon].plant.e_psi,
        selected.validation_minimum_corridor_slack_m,
        previous_slack_m,
        selected_witness_rescue ? "source_witness" : "other_validated");
    return true;
}


bool MpcControllerNode::try_publish_quality_aware_prefix(
    const MpcRtiState_t &state,
    double progress,
    int64_t callback_steady_ns,
    int64_t control_stamp_ns,
    MpcRtiCycleResult_t *result,
    bool *attempted,
    const char **fault_reason)
{
    if (attempted) *attempted = false;
    if (fault_reason) *fault_reason = nullptr;
    if (fallback_episode_.requires_full_horizon(
            fallback_max_short_prefix_actions_)) {
        if (fault_reason) *fault_reason =
            "short prefix episode exceeded limit; full 750ms physical "
            "solution required (no more stale plan replay)";
        return false;
    }
    if (!quality_aware_prefix_selection_on_rejection_ || !result ||
        result->status != MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT) {
        return false;
    }

    const int prior_start = prior_plan_control_index_ + 1;
    const int prior_remaining = certified_plan_.valid
        ? certified_plan_.horizon - prior_start : 0;
    const bool use_validated_recovery_prefix =
        prior_remaining < 1 && has_validated_recovery_prefix();

    const bool solver_converged =
        result->admm_status == RICCATI_STATUS_OPTIMAL;
    const bool solver_iteration_limited =
        result->admm_status == RICCATI_STATUS_MAX_ITERATIONS;
    const bool candidate_solver_status_usable =
        solver_converged || solver_iteration_limited;
    const bool candidate_controls_available =
        candidate_solver_status_usable &&
        result->candidate_control_count > 0;
    if (prior_remaining < 1 && !use_validated_recovery_prefix &&
        !candidate_controls_available)
        return false;
    const bool residual_quality_acceptable = candidate_solver_status_usable &&
        (solver_converged ||
        iteration_limited_prefix_meets_residual_quality(
            result->primal_residual, result->dual_residual,
            rti_config_.solver.tolerance, 2.0));
    candidate_prefix_residual_quality_ = residual_quality_acceptable;

    /* Candidate convergence and retained-plan safety are independent.
     * A poor/failed ADMM iterate must not suppress exact revalidation of
     * an already certified plan: that plan may still provide a safe
     * command-effect prefix even when its full remaining suffix is no
     * longer valid from the current measured state. */
    if (attempted) *attempted = true;
    quality_prefix_attempted_ = true;
    candidate_prefix_revalidation_attempted_ = true;
    candidate_prefix_solver_converged_ = solver_converged;
    candidate_prefix_iteration_limited_ = solver_iteration_limited;
    prior_plan_revalidation_attempted_ = true;

    MpcModelControl_t candidate_controls[
        kCommandEffectValidationHorizon]{};
    MpcModelControl_t prior_controls[
        kCommandEffectValidationHorizon]{};
    for (int i = 0; i < kCommandEffectValidationHorizon; ++i) {
        if (candidate_solver_status_usable &&
            i < result->candidate_control_count)
            candidate_controls[i] = result->candidate_controls[i];
        /* A published actuator command is held until its replacement
         * takes effect. If the retained plan has fewer than three queued
         * actions left, use its last command as a zero-order hold for the
         * remainder of this equal-length 75 ms comparison. This makes
         * the candidate compete against the action sequence that would
         * actually remain on the actuator, instead of bypassing the
         * retained plan merely because its horizon is nearly exhausted. */
        if (use_validated_recovery_prefix) {
            const int prior_index = std::min(
                validated_recovery_prefix_cursor_ + i,
                validated_recovery_prefix_count_ - 1);
            prior_controls[i] =
                validated_recovery_prefix_controls_[prior_index];
        } else if (prior_remaining > 0) {
            const int prior_index = std::min(
                prior_start + i, certified_plan_.horizon - 1);
            prior_controls[i] = certified_plan_.controls[prior_index];
        } else {
            prior_controls[i] = candidate_controls[i];
        }
    }
    quality_candidate_control_count_ = std::min(
        result->candidate_control_count, PREDICTION_HORIZON);
    for (int i = 0; i < quality_candidate_control_count_; ++i)
        quality_candidate_controls_[i] = result->candidate_controls[i];
    const auto controls_in_envelope = [this](
        MpcModelControl_t controls[
            kCommandEffectValidationHorizon], int horizon) {
        for (int i = 0; i < horizon; ++i) {
            if (!std::isfinite(controls[i].steering_target_rad) ||
                !std::isfinite(controls[i].throttle_command) ||
                controls[i].throttle_command <
                    MPC_THROTTLE_MIN_DRIVE - 1.0e-6f ||
                controls[i].throttle_command > 1.0f + 1.0e-6f ||
                std::abs(controls[i].steering_target_rad) >
                    actuator_steering_limit_rad_ + 1.0e-6) {
                return false;
            }
            controls[i].steering_target_rad = static_cast<float>(clamp(
                controls[i].steering_target_rad,
                -actuator_steering_limit_rad_,
                actuator_steering_limit_rad_));
            controls[i].throttle_command = static_cast<float>(clamp(
                controls[i].throttle_command,
                MPC_THROTTLE_MIN_DRIVE, 1.0));
        }
        return true;
    };

    const bool prior_controls_available = prior_remaining > 0 ||
        use_validated_recovery_prefix;
    MpcRtiNominal_t candidate_revalidated{};
    MpcRtiNominal_t prior_revalidated{};
    MpcRtiDiagnosticStageOutput_t candidate_outputs[
        MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT]{};
    MpcRtiDiagnosticStageOutput_t prior_outputs[
        MPC_RTI_DIAGNOSTIC_SAMPLE_COUNT]{};
    int candidate_output_count = 0;
    int prior_output_count = 0;
    int candidate_failure_stage = -1;
    int prior_failure_stage = -1;
    MpcRtiRolloutStatus_t candidate_status =
        MPC_RTI_ROLLOUT_INVALID_INPUT;
    MpcRtiRolloutStatus_t prior_status =
        MPC_RTI_ROLLOUT_INVALID_INPUT;

    MpcRtiCycleConfiguration_t comparison_configuration = rti_config_;
    if (result->comparison_objective.valid)
        comparison_configuration.objective_schedule = &result->comparison_objective;
    auto validate = [&](
        const MpcModelControl_t *controls,
        MpcRtiNominal_t *validated,
        MpcRtiDiagnosticStageOutput_t *outputs,
        int *output_count,
        int *failure_stage,
        int horizon) {
        const auto start = std::chrono::steady_clock::now();
        const MpcRtiRolloutStatus_t status =
            mpc_rti_validate_candidate_plan_with_diagnostics(
                &state, progress, controls,
                trajectory_.data(), trajectory_.size(), track_length_m_,
                horizon, TIME_STEP_SECONDS,
                &comparison_configuration, validated, outputs, output_count,
                failure_stage);
        return std::pair<MpcRtiRolloutStatus_t, double>{
            status, std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count()};
    };

    double candidate_validation_us = 0.0;
    double prior_validation_us = 0.0;
    int selected_validation_horizon = 0;
    /* Prefer the usual 75 ms action prefix. If both the new candidate and
     * retained sequence fail it, retry at equal 50 ms then 25 ms horizons.
     * Each published action remains exactly source-plant validated
     * against the same hard corridor; only an unissued tail is omitted,
     * and the MPC replans on the next control tick. */
    for (int horizon = kCommandEffectValidationHorizon;
         horizon >= 1; --horizon) {
        candidate_status = MPC_RTI_ROLLOUT_INVALID_INPUT;
        prior_status = MPC_RTI_ROLLOUT_INVALID_INPUT;
        if (candidate_controls_available &&
            result->candidate_control_count >= horizon &&
            controls_in_envelope(candidate_controls, horizon)) {
            const auto validation = validate(
                candidate_controls, &candidate_revalidated,
                candidate_outputs, &candidate_output_count,
                &candidate_failure_stage, horizon);
            candidate_status = validation.first;
            candidate_validation_us += validation.second;
        }
        if (prior_controls_available &&
            controls_in_envelope(prior_controls, horizon)) {
            // If both sequences issue the EXACT same physical commands,
            // the already-computed source model and map validation are
            // identical. Reuse the result; never run the 25ms vehicle step
            // twice for the same state/control/trajectory/epoch.
            bool identical = candidate_status !=
                MPC_RTI_ROLLOUT_INVALID_INPUT && candidate_controls_available;
            for (int i = 0; identical && i < horizon; ++i)
                identical =
                    candidate_controls[i].steering_target_rad ==
                        prior_controls[i].steering_target_rad &&
                    candidate_controls[i].throttle_command ==
                        prior_controls[i].throttle_command;
            if (identical) {
                prior_status = candidate_status;
                prior_failure_stage = candidate_failure_stage;
                prior_revalidated = candidate_revalidated;
                prior_output_count = candidate_output_count;
                std::copy_n(candidate_outputs, candidate_output_count,
                            prior_outputs);
            } else {
                const auto validation = validate(
                    prior_controls, &prior_revalidated, prior_outputs,
                    &prior_output_count, &prior_failure_stage, horizon);
                prior_status = validation.first;
                prior_validation_us += validation.second;
            }
        }
        if (candidate_status == MPC_RTI_ROLLOUT_OK ||
            prior_status == MPC_RTI_ROLLOUT_OK) {
            selected_validation_horizon = horizon;
            break;
        }
    }
    candidate_prefix_validation_horizon_ = selected_validation_horizon;
    prior_plan_validated_horizon_ = selected_validation_horizon;
    quality_prior_control_count_ = selected_validation_horizon;
    for (int i = 0; i < quality_prior_control_count_; ++i)
        quality_prior_controls_[i] = prior_controls[i];
    quality_prefix_validation_us_ =
        candidate_validation_us + prior_validation_us;
    candidate_prefix_validation_us_ = candidate_validation_us;
    candidate_prefix_validation_status_ =
        static_cast<int>(candidate_status);
    candidate_prefix_validation_failure_stage_ = candidate_failure_stage;
    prior_plan_validation_us_ = prior_validation_us;
    prior_plan_validation_status_ = static_cast<int>(prior_status);
    prior_plan_validation_failure_stage_ = prior_failure_stage;
    candidate_prefix_exact_valid_ =
        candidate_status == MPC_RTI_ROLLOUT_OK;
    prior_plan_exact_valid_ = prior_status == MPC_RTI_ROLLOUT_OK;
    quality_candidate_prefix_valid_ = candidate_prefix_exact_valid_;
    quality_prior_prefix_valid_ = prior_plan_exact_valid_;

    if (candidate_prefix_exact_valid_) {
        quality_candidate_prefix_objective_ =
            candidate_revalidated.validation_objective;
        quality_candidate_prefix_slack_m_ =
            candidate_revalidated.validation_minimum_corridor_slack_m;
    }
    if (prior_plan_exact_valid_) {
        quality_prior_prefix_objective_ =
            prior_revalidated.validation_objective;
        quality_prior_prefix_slack_m_ =
            prior_revalidated.validation_minimum_corridor_slack_m;
    }

    if (!candidate_prefix_exact_valid_ && !prior_plan_exact_valid_) {
        quality_prefix_choice_ = "none_exact_valid";
        if (fault_reason)
            *fault_reason = "neither candidate nor retained-plan prefix "
                "passes exact source-plant/corridor validation";
        return false;
    }

    const int shared_prefix_count = std::min(
        candidate_revalidated.horizon, prior_revalidated.horizon);
    quality_prefix_near_boundary_controls_bounded_ =
        f1tenth_mpc::prefix_control_changes_bounded(
            candidate_revalidated.controls,
            prior_revalidated.controls, shared_prefix_count,
            quality_prefix_near_boundary_max_steering_delta_rad_,
            quality_prefix_near_boundary_max_throttle_delta_);
    const auto prefix_selection =
        f1tenth_mpc::select_validated_recovery_prefix(
        candidate_prefix_exact_valid_,
        candidate_revalidated.validation_objective,
        candidate_revalidated.validation_minimum_corridor_slack_m,
        solver_converged,
        residual_quality_acceptable,
        prior_plan_exact_valid_,
        prior_revalidated.validation_objective,
        prior_revalidated.validation_minimum_corridor_slack_m,
        quality_prefix_iteration_limited_relative_gain_,
        quality_prefix_clearance_reserve_m_,
        quality_prefix_minimum_clearance_gain_m_,
        consecutive_prior_plan_actions_,
        quality_prefix_stale_prior_refresh_after_actions_,
        quality_prefix_stale_prior_max_clearance_loss_m_,
        quality_prefix_stale_prior_objective_loss_fraction_,
        quality_prefix_near_boundary_minimum_gain_m_,
        quality_prefix_near_boundary_max_objective_ratio_,
        quality_prefix_near_boundary_controls_bounded_);
    bool candidate_yaw_direction_conflict = false;
    if (prefix_selection.choose_candidate &&
        candidate_prefix_exact_valid_ && prior_plan_exact_valid_ &&
        candidate_revalidated.horizon >= 1 &&
        prior_revalidated.horizon >= 1) {
        const int candidate_terminal = candidate_revalidated.horizon;
        const int prior_terminal = prior_revalidated.horizon;
        MpcTrajectorySample_t terminal_reference{};
        if (mpc_trajectory_sample(trajectory_.data(), trajectory_.size(),
                track_length_m_,
                candidate_revalidated.progress[candidate_terminal],
                &terminal_reference)) {
            const double terminal_speed = std::min(
                terminal_reference.speed,
                static_cast<double>(
                    rti_config_.model.active_speed_ceiling_mps));
            const double reference_yaw_rate =
                terminal_reference.curvature * terminal_speed;
            candidate_yaw_direction_conflict =
                f1tenth_mpc::candidate_prefix_yaw_direction_conflicts(
                    reference_yaw_rate,
                    candidate_revalidated.states[candidate_terminal]
                        .plant.r,
                    prior_revalidated.states[prior_terminal].plant.r);
        }
    }
    const bool choose_candidate = prefix_selection.choose_candidate &&
        !candidate_yaw_direction_conflict;
    quality_prefix_yaw_direction_guard_ =
        candidate_yaw_direction_conflict;
    if (!choose_candidate && !prior_plan_exact_valid_) {
        quality_prefix_choice_ = "no_quality_eligible_prefix";
        if (fault_reason)
            *fault_reason = "iteration-limited candidate did not meet the "
                "residual/quality gate and no exact-valid prior prefix remains";
        return false;
    }
    quality_prefix_clearance_override_ =
        prefix_selection.clearance_override;
    quality_prefix_near_boundary_override_ =
        prefix_selection.near_boundary_override;
    quality_prefix_stale_prior_refresh_ =
        prefix_selection.stale_prior_refresh;
    quality_prefix_choice_ = choose_candidate ? "candidate" :
        (quality_prefix_yaw_direction_guard_
            ? (use_validated_recovery_prefix
                ? "retained_recovery_prefix_yaw_direction_guard"
                : "retained_plan_yaw_direction_guard")
            : (use_validated_recovery_prefix
                ? "retained_recovery_prefix" : "retained_plan"));
    quality_prefix_prior_is_recovery_ =
        !choose_candidate && use_validated_recovery_prefix;

    const auto &selected_nominal = choose_candidate
        ? candidate_revalidated : prior_revalidated;
    const MpcModelControl_t first = selected_nominal.controls[0];
    result->diagnostic_output_count = choose_candidate
        ? candidate_output_count : prior_output_count;
    const auto *selected_outputs = choose_candidate
        ? candidate_outputs : prior_outputs;
    for (int i = 0; i < result->diagnostic_output_count; ++i)
        result->diagnostic_outputs[i] = selected_outputs[i];
    result->nonlinear_objective = selected_nominal.validation_objective;
    result->minimum_predicted_corridor_slack_m =
        selected_nominal.validation_minimum_corridor_slack_m;
    recovered_action_revalidated_nominal_ = selected_nominal;

    const char *timing_fault_reason = nullptr;
    if (!cycle_time_is_valid(callback_steady_ns, &timing_fault_reason)) {
        quality_prefix_choice_ = choose_candidate
            ? "candidate_rejected_deadline" : "prior_rejected_deadline";
        if (choose_candidate)
            candidate_prefix_timing_rejected_ = true;
        else
            prior_plan_timing_rejected_ = true;
        if (fault_reason) *fault_reason = timing_fault_reason;
        return false;
    }

    MpcModelControl_t next_warm_start[PREDICTION_HORIZON]{};
    int warm_start_count = result->candidate_control_count;
    if (choose_candidate) {
        warm_start_count = f1tenth_mpc::compose_candidate_prefix_warm_start(
            candidate_revalidated.controls,
            candidate_revalidated.horizon,
            certified_plan_.valid ? certified_plan_.controls : nullptr,
            certified_plan_.valid ? certified_plan_.horizon : 0,
            prior_start,
            PREDICTION_HORIZON,
            next_warm_start);
        if (warm_start_count < 1) {
            if (fault_reason)
                *fault_reason = "validated candidate prefix could not seed "
                    "the next MPC nominal sequence";
            return false;
        }
        next_warm_start[0] = first;
    } else if (use_validated_recovery_prefix) {
        warm_start_count = selected_validation_horizon;
        for (int k = 0; k < warm_start_count; ++k)
            next_warm_start[k] = prior_controls[k];
        next_warm_start[0] = first;
    } else {
        const MpcRtiNominal_t &previous = certified_plan_;
        warm_start_count = f1tenth_mpc::copy_retained_plan_warm_start(
            previous.controls, previous.horizon, prior_start,
            PREDICTION_HORIZON, next_warm_start);
        next_warm_start[0] = first;
    }

    const char *publication_fault_reason = nullptr;
    const int64_t command_stamp_ns = publish_command(
        first.steering_target_rad, first.throttle_command,
        callback_steady_ns,
        &publication_fault_reason);
    if (command_stamp_ns == 0) {
        if (last_publication_gate_elapsed_ms_ >=
                end_to_end_cycle_budget_ms_) {
            if (choose_candidate)
                candidate_prefix_timing_rejected_ = true;
            else
                prior_plan_timing_rejected_ = true;
        }
        quality_prefix_choice_ = choose_candidate
            ? "candidate_publish_rejected" : "prior_publish_rejected";
        if (fault_reason) *fault_reason = publication_fault_reason
            ? publication_fault_reason
            : "quality-selected exact-valid prefix violates drive envelope";
        return false;
    }
    if (!mpc_rti_memory_set_warm_start_controls_at_epoch(
            &rti_memory_, next_warm_start, warm_start_count, active_command_epoch_ns_)) {
        if (fault_reason)
            *fault_reason = "published quality-selected prefix could not "
                "seed the next MPC nominal sequence";
        return false;
    }
    /* Prefix publication intentionally does not commit a full candidate
     * nominal. Discard solver iterates from that rejected full-horizon
     * problem, regardless of whether candidate or retained prefix won. */
    mpc_rti_memory_reset_solver_state(&rti_memory_);
    result->admm_state_reset_for_recovery = 1;

    if (choose_candidate) {
        if (solver_converged)
            remember_validated_recovery_prefix(candidate_revalidated);
        consecutive_prior_plan_actions_ = 0;
        ++consecutive_candidate_prefix_actions_;
        result->status = MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX;
    } else {
        if (use_validated_recovery_prefix) {
            prior_plan_actions_remaining_ = 0;
        } else {
            prior_plan_control_index_ = prior_start;
            prior_plan_validated_control_index_ = prior_start;
            prior_plan_actions_remaining_ = certified_plan_.horizon -
                prior_start - 1;
        }
        consecutive_candidate_prefix_actions_ = 0;
        ++consecutive_prior_plan_actions_;
        result->candidate_control_count = 0;
        result->status = MPC_RTI_CYCLE_ACCEPTED_PRIOR_PLAN;
    }
    result->first_control = first;
    result->published_steering_command = first.steering_target_rad;
    result->published_throttle = first.throttle_command;
    last_command_publication_lead_ns_ =
        command_stamp_ns >= control_stamp_ns
            ? command_stamp_ns - control_stamp_ns : 0;
    return true;
}


bool MpcControllerNode::try_publish_candidate_prefix(
    const MpcRtiState_t &state,
    double progress,
    int64_t callback_steady_ns,
    int64_t control_stamp_ns,
    MpcRtiCycleResult_t *result,
    const char **fault_reason)
{
    if (fault_reason) *fault_reason = nullptr;
    if (fallback_episode_.requires_full_horizon(
            fallback_max_short_prefix_actions_)) {
        if (fault_reason) *fault_reason =
            "candidate short-prefix budget exhausted; must obtain "
            "complete source-plant feasible plan";
        return false;
    }
    if (!result || result->status !=
            MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT ||
        result->candidate_control_count < 1 ||
        result->candidate_control_count > PREDICTION_HORIZON) return false;

    candidate_prefix_revalidation_attempted_ = true;
    const bool solver_converged =
        result->admm_status == RICCATI_STATUS_OPTIMAL;
    const bool solver_iteration_limited =
        result->admm_status == RICCATI_STATUS_MAX_ITERATIONS;
    if (!solver_converged && !solver_iteration_limited) {
        if (fault_reason)
            *fault_reason = "full-horizon-invalid candidate did not "
                "produce a finite ADMM iterate";
        return false;
    }
    candidate_prefix_solver_converged_ = solver_converged;
    candidate_prefix_iteration_limited_ = solver_iteration_limited;
    candidate_prefix_residual_quality_ = solver_converged ||
        iteration_limited_prefix_meets_residual_quality(
            result->primal_residual, result->dual_residual,
            rti_config_.solver.tolerance, 2.0);
    if (!candidate_prefix_residual_quality_) {
        if (fault_reason)
            *fault_reason = "iteration-limited candidate exceeds twice "
                "the configured ADMM residual tolerance and no "
                "revalidated prior prefix was selected";
        return false;
    }
    if (result->candidate_control_count <
            kCommandEffectValidationHorizon) {
        if (fault_reason)
            *fault_reason = "rejected candidate is too short to validate "
                "through the delayed command effect";
        return false;
    }
    MpcModelControl_t controls[kCommandEffectValidationHorizon]{};
    for (int i = 0; i < kCommandEffectValidationHorizon; ++i) {
        controls[i] = result->candidate_controls[i];
        if (!std::isfinite(controls[i].steering_target_rad) ||
            !std::isfinite(controls[i].throttle_command) ||
            controls[i].throttle_command <
                MPC_THROTTLE_MIN_DRIVE - 1.0e-6f ||
            controls[i].throttle_command > 1.0f + 1.0e-6f ||
            std::abs(controls[i].steering_target_rad) >
                actuator_steering_limit_rad_ + 1.0e-6) {
            if (fault_reason)
                *fault_reason = "rejected candidate prefix violates "
                    "the production drive-command envelope";
            return false;
        }
        controls[i].steering_target_rad = static_cast<float>(clamp(
            controls[i].steering_target_rad,
            -actuator_steering_limit_rad_, actuator_steering_limit_rad_));
        controls[i].throttle_command = static_cast<float>(clamp(
            controls[i].throttle_command, MPC_THROTTLE_MIN_DRIVE, 1.0));
    }

    MpcRtiNominal_t revalidated{};
    int failure_stage = -1;
    candidate_prefix_validation_horizon_ =
        kCommandEffectValidationHorizon;
    const auto validation_start = std::chrono::steady_clock::now();
    const MpcRtiRolloutStatus_t rollout_status =
        mpc_rti_validate_candidate_plan_with_diagnostics(
            &state, progress,
            controls, trajectory_.data(), trajectory_.size(),
            track_length_m_, kCommandEffectValidationHorizon,
            TIME_STEP_SECONDS, &rti_config_, &revalidated,
            result->diagnostic_outputs,
            &result->diagnostic_output_count,
            &failure_stage);
    candidate_prefix_validation_status_ = static_cast<int>(rollout_status);
    candidate_prefix_validation_failure_stage_ = failure_stage;
    candidate_prefix_validation_us_ =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - validation_start).count();
    if (rollout_status != MPC_RTI_ROLLOUT_OK) {
        if (fault_reason)
            *fault_reason = "rejected candidate prefix failed exact "
                "delayed-command rollout/corridor validation";
        RCLCPP_WARN(get_logger(),
            "Rejected horizon prefix not reused: exact 75 ms "
            "rollout status=%d stage=%d",
            static_cast<int>(rollout_status), failure_stage);
        return false;
    }
    candidate_prefix_exact_valid_ = true;
    recovered_action_revalidated_nominal_ = revalidated;
    result->nonlinear_objective = revalidated.validation_objective;
    result->minimum_predicted_corridor_slack_m =
        revalidated.validation_minimum_corridor_slack_m;
    MpcModelControl_t next_warm_start[PREDICTION_HORIZON]{};
    const int warm_start_count =
        f1tenth_mpc::compose_candidate_prefix_warm_start(
            revalidated.controls, revalidated.horizon,
            certified_plan_.valid ? certified_plan_.controls : nullptr,
            certified_plan_.valid ? certified_plan_.horizon : 0,
            prior_plan_control_index_ + 1,
            PREDICTION_HORIZON,
            next_warm_start);
    if (warm_start_count < kCommandEffectValidationHorizon) {
        if (fault_reason)
            *fault_reason = "validated candidate prefix could not seed "
                "the next MPC nominal sequence";
        return false;
    }
    next_warm_start[0] = revalidated.controls[0];

    const char * timing_fault_reason = nullptr;
    if (!cycle_time_is_valid(callback_steady_ns, &timing_fault_reason)) {
        candidate_prefix_timing_rejected_ = true;
        if (fault_reason) *fault_reason = timing_fault_reason;
        return false;
    }
    const MpcModelControl_t &first = revalidated.controls[0];
    const char *publication_fault_reason = nullptr;
    const int64_t command_stamp_ns = publish_command(
        first.steering_target_rad, first.throttle_command,
        callback_steady_ns,
        &publication_fault_reason);
    if (command_stamp_ns == 0) {
        if (last_publication_gate_elapsed_ms_ >=
                end_to_end_cycle_budget_ms_)
            candidate_prefix_timing_rejected_ = true;
        if (fault_reason) *fault_reason = publication_fault_reason ?
            publication_fault_reason :
            "exact-validated candidate prefix violates the drive envelope";
        return false;
    }
    /* Do not let an unpublished candidate overwrite the next nominal.
     * The command is now in the actuator history, so its shifted suffix
     * is the correct next-cycle seed. */
    if (!mpc_rti_memory_set_warm_start_controls_at_epoch(&rti_memory_,
            next_warm_start, warm_start_count, active_command_epoch_ns_)) {
        if (fault_reason)
            *fault_reason = "published candidate prefix could not seed "
                "the next MPC nominal sequence";
        return false;
    }
    /* The QP candidate was rejected as a full horizon and only a short
     * exact-valid prefix was issued. Its saved ADMM iterates are tied to
     * the rejected full plan; preserve the issued control seed, not those
     * stale primal/dual variables. */
    mpc_rti_memory_reset_solver_state(&rti_memory_);
    result->admm_state_reset_for_recovery = 1;

    if (solver_converged)
        remember_validated_recovery_prefix(revalidated);

    /* This is a newly solved control, not stale-plan replay. Even when
     * ADMM hit its iteration cap, publish only if the exact three-step
     * delayed-action prefix is corridor-safe and meets the deadline.
     * Re-solve at the next tick; never repeat a stale candidate blindly. */
    consecutive_prior_plan_actions_ = 0;
    ++consecutive_candidate_prefix_actions_;
    last_command_publication_lead_ns_ = command_stamp_ns >= control_stamp_ns
        ? command_stamp_ns - control_stamp_ns : 0;
    result->status = MPC_RTI_CYCLE_ACCEPTED_CANDIDATE_PREFIX;
    result->first_control = first;
    result->published_steering_command = first.steering_target_rad;
    result->published_throttle = first.throttle_command;
    return true;
}


} // namespace f1tenth_mpc
