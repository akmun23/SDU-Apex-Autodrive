// Single-source nonlinear SQP diagnostics.  The asynchronous recorder is
// deliberately outside the control deadline; no recovery/fallback fields.
#include "mpc_controller_node_impl.hpp"

namespace f1tenth_mpc {
void MpcControllerNode::publish_cycle_diagnostics(
    const MpcRtiState_t &state,
    const MpcSynchronizedState &synchronized,
    const MpcControlTimePrediction &control_prediction,
    double control_prediction_us,double progress,double source_dt_s,
    const MpcRtiCycleResult_t &result,double solve_us,
    int64_t control_ros_stamp_ns,int64_t candidate_command_stamp_ns,
    int64_t callback_steady_ns,int64_t synchronize_steady_ns)
{
    if(!diagnostics_pub_)return;
    const auto finite_json=[](std::ostringstream &os,double x){
        if(std::isfinite(x))os<<x;else os<<"null";
    };
    const auto &plant=state.plant;
    std::ostringstream json;
    json<<std::setprecision(17)
        <<"{\"status\":\""<<cycle_status_name(result.status)
        <<"\",\"solver\":{"
        <<"\"backend\":\"source_exact_full_step_sqp\"";
    const char *termination="invalid_model";
    switch(result.source_solver_status) {
    case MPC_SOURCE_DEADLINE_EXCEEDED:termination="deadline_exceeded";break;
    case MPC_SOURCE_PRIMAL_INFEASIBLE:termination="primal_infeasible";break;
    case MPC_SOURCE_NONLINEAR_STEP_REJECTED:termination="nonlinear_step_rejected";break;
    case MPC_SOURCE_KKT_NOT_CONVERGED:termination="KKT_not_converged";break;
    case MPC_SOURCE_INVALID_MODEL:break;
    case MPC_SOURCE_FEASIBLE_SOLUTION_AVAILABLE:termination="feasible_solution_available";break;
    }
    json<<",\"termination\":\""<<termination<<'"'
        <<",\"feasible_solution_available\":"<<(result.source_exact_nlp_final_source_feasible?"true":"false")
        <<",\"linearization_failed\":"<<(result.source_linearization_failed?"true":"false")
        <<",\"qp_failed\":"<<(result.source_qp_failed?"true":"false")
        <<",\"iteration_admission_stopped\":"<<(result.source_iteration_admission_stopped?"true":"false")
        <<",\"sqp_iterations_attempted\":"
        <<result.sqp_iterations_attempted
        <<",\"sqp_iterations_accepted\":"
        <<result.source_exact_nlp_accepted_iterations
        <<",\"source_exact_nlp_source_trials\":"
        <<result.source_exact_nlp_source_trial_evaluations
        <<",\"source_exact_nlp_deadline_exhausted\":"
        <<(result.source_exact_nlp_deadline_exhausted?"true":"false")
        <<",\"source_exact_nlp_initial_source_feasible\":"
        <<(result.source_exact_nlp_initial_source_feasible?"true":"false")
        <<",\"source_exact_nlp_final_source_feasible\":"
        <<(result.source_exact_nlp_final_source_feasible?"true":"false")
        <<",\"source_exact_nlp_kkt_converged\":"
        <<(result.source_exact_nlp_kkt_converged?"true":"false")
        <<",\"source_exact_nlp_full_step_stalled\":"
        <<(result.source_exact_nlp_full_step_stalled?"true":"false")
        <<",\"source_exact_nlp_original_cost_initial\":";
    finite_json(json,result.source_exact_nlp_initial_source_cost);
    json<<",\"source_exact_nlp_original_cost_final\":";
    finite_json(json,result.source_exact_nlp_final_source_cost);
    json<<",\"source_exact_nlp_original_minimum_corner_margin_m\":";
    finite_json(json,result.source_exact_nlp_final_original_margin_m);
    json<<",\"source_exact_nlp_original_constraint_violation\":";
    finite_json(json,result.source_exact_nlp_final_original_violation);
    json<<",\"source_iterations\":[";
    for(int i=0;i<result.source_iteration_metrics_count;++i) {
        if(i)json<<',';
        const auto &in=result.source_iteration_metrics[i];
        json<<"{\"iteration\":"<<i
            <<",\"accepted\":"<<(in.accepted?"true":"false")
            <<",\"full_step_evaluated\":"<<(in.full_step_evaluated?"true":"false")
            <<",\"full_step_accepted\":"<<(in.full_step_accepted?"true":"false")
            <<",\"qp_converged\":"<<(in.qp_converged?"true":"false")
            <<",\"original_duals_trustworthy\":"<<(in.original_duals_trustworthy?"true":"false")
            <<",\"qp_iterations\":"<<in.qp_iterations;
        const auto field=[&](const char *name,double value) {
            json<<",\""<<name<<"\":";finite_json(json,value);
        };
        field("linearization_ms",in.linearization_ms);
        field("qp_ms",in.qp_ms);
        field("source_rollout_ms",in.source_rollout_ms);
        field("total_ms",in.total_ms);
        field("original_cost_before",in.cost_before);
        field("original_cost_after",in.cost_after);
        field("line_search_alpha",in.accepted?in.accepted_step_length:NAN);
        field("qp_primal_residual",in.qp_converged?in.qp_primal_residual:NAN);
        field("qp_dual_residual",in.qp_converged?in.qp_dual_residual:NAN);
        field("initial_nlp_stationarity_inf",in.original_duals_trustworthy?in.stationarity_inf:NAN);
        field("initial_nlp_complementarity_inf",in.original_duals_trustworthy?in.complementarity_inf:NAN);
        field("qp_predicted_cost_change",in.full_step_evaluated?in.predicted_cost_change:NAN);
        field("full_step_original_cost",in.full_step_evaluated?in.full_step_cost:NAN);
        field("full_step_cost_prediction_error",in.full_step_evaluated?
            in.full_step_cost-in.cost_before-in.predicted_cost_change:NAN);
        field("maximum_footprint_prediction_error_m",in.full_step_evaluated?
            in.maximum_footprint_prediction_error_m:NAN);
        json<<'}';
    }
    json<<']';
    json<<"},\"source_stamp_ns\":"<<synchronized.source_stamp_ns
        <<",\"odom_source_stamp_ns\":"<<synchronized.odom_source_stamp_ns
        <<",\"map_pose_source_stamp_ns\":"
        <<synchronized.map_pose_source_stamp_ns
        <<",\"control_ros_stamp_ns\":"<<control_ros_stamp_ns
        <<",\"candidate_command_stamp_ns\":"
        <<candidate_command_stamp_ns
        <<",\"issued_command_stamp_ns\":"
        <<(command_published_this_cycle_?
               last_published_command_stamp_ns_:0)
        <<",\"command_published\":"
        <<(command_published_this_cycle_?"true":"false")
        <<",\"consecutive_no_publication_cycles\":"<<consecutive_no_publication_cycles_
        <<",\"maximum_no_publication_cycles\":"<<maximum_no_publication_cycles_
        <<",\"older_command_epoch_rejections\":"<<older_command_epoch_rejections_
        <<",\"same_stage_epoch_revisions\":"<<same_stage_epoch_revisions_
        <<",\"last_issued_command_age_ms\":";
    finite_json(json,last_published_command_stamp_ns_>0?
        std::max(0.,double(control_ros_stamp_ns-last_published_command_stamp_ns_)*1.e-6):NAN);
    json<<",\"bridge_request_unchanged_command_age_ms\":";
    finite_json(json,bridge_request_unchanged_command_age_ms_);
    json
        <<",\"source_dt_s\":";
    finite_json(json,source_dt_s);
    json<<",\"progress_m\":";
    finite_json(json,progress);
    json<<",\"state\":[";
    const double state_values[]={
        plant.e_y,plant.e_psi,plant.u,plant.v,plant.r,
        plant.delayed_steering_command_1,plant.delayed_steering_command_2,
        plant.actual_steering_angle,
        plant.delayed_throttle_command_1,plant.delayed_throttle_command_2};
    for(size_t k=0;k<std::size(state_values);++k) {
        if(k)json<<',';
        finite_json(json,state_values[k]);
    }
    json<<"],\"plant_state_flags\":["<<plant.source_plant_initialized<<','
        <<plant.source_suspension_history_initialized<<','<<plant.source_imu_state_initialized
        <<"],\"plant_internal_state\":[";
    for(int k=0;k<MPC_PLANT_INTERNAL_STATE_COUNT;++k){
        if(k)json<<',';
        finite_json(json,plant.source_plant_internal_state[k]);
    }
    json<<"],\"first_action\":[";
    finite_json(json,result.published_steering_command);
    json<<',';
    finite_json(json,result.published_throttle);
    json<<"],\"nominal_first_action\":[";
    finite_json(json,result.nominal_first_control.steering_target_rad);
    json<<',';
    finite_json(json,result.nominal_first_control.throttle_command);
    json<<"],\"nominal_controls\":[";
    for(int k=0;k<result.nominal_control_count;++k){
        if(k)json<<',';
        json<<'[';
        finite_json(json,result.nominal_controls[k].steering_target_rad);
        json<<',';
        finite_json(json,result.nominal_controls[k].throttle_command);
        json<<']';
    }
    json<<"],\"candidate_controls\":[";
    for(int k=0;k<result.candidate_control_count;++k){
        if(k)json<<',';
        json<<'[';
        finite_json(json,result.candidate_controls[k].steering_target_rad);
        json<<',';
        finite_json(json,result.candidate_controls[k].throttle_command);
        json<<']';
    }
    json<<"],\"source_optimized_controls\":[";
    for(int k=0;k<result.source_optimized_control_count;++k) {
        if(k)json<<',';
        json<<'[';finite_json(json,result.source_optimized_controls[k].steering_target_rad);
        json<<',';finite_json(json,result.source_optimized_controls[k].throttle_command);json<<']';
    }
    json<<"],\"source_primal_trace_truncated\":"
        <<(result.source_primal_trace_truncated?"true":"false")
        <<",\"source_primal_trace\":[";
    for(int i=0;i<result.source_primal_trace_count;++i) {
        if(i)json<<',';
        const auto &trace=result.source_primal_trace[i];
        json<<"{\"iteration\":"<<trace.iteration
            <<",\"accepted\":"<<(trace.accepted?"true":"false")
            <<",\"trial_evaluated\":"<<(trace.trial_evaluated?"true":"false");
        const auto commands=[&](const char *name,const MpcModelControl_t *u) {
            json<<",\""<<name<<"\":[";
            for(int k=0;k<PREDICTION_HORIZON;++k) {
                if(k)json<<',';
                json<<'[';finite_json(json,u[k].steering_target_rad);
                json<<',';finite_json(json,u[k].throttle_command);json<<']';
            }
            json<<']';
        };
        commands("warm_start_controls",trace.warm_start);
        if(trace.trial_evaluated)commands("trial_controls",trace.trial);
        json<<",\"correction_evaluated\":"<<(trace.correction_evaluated?"true":"false")
            <<",\"correction_accepted\":"<<(trace.correction_accepted?"true":"false");
        if(trace.correction_evaluated)commands("correction_controls",trace.correction);
        json<<",\"normal_accepted\":"<<(trace.normal_accepted?"true":"false")
            <<",\"filter_accepted\":"<<(trace.filter_accepted?"true":"false");
        if(trace.accepted)commands("accepted_controls",trace.accepted_controls);
        json<<'}';
    }
    json<<']';
    if(rti_config_.capture_source_benchmark) {
        // Same-build POD capture retains every frozen reference/weight and
        // constraint setting bit exactly. Pointer callbacks cannot be replayed.
        auto model=rti_config_.model;
        const bool world=model.world_sweep_check!=nullptr;
        model.world_sweep_check=nullptr;model.world_sweep_context=nullptr;
        const auto hex=[&](const void *object,size_t size) {
            constexpr char digits[]="0123456789abcdef";
            const auto *bytes=static_cast<const unsigned char *>(object);
            json<<'"';
            for(size_t i=0;i<size;++i)json<<digits[bytes[i]>>4]<<digits[bytes[i]&15];
            json<<'"';
        };
        json<<",\"frozen_source_problem\":{\"schema\":\"source-pod/v1\",\"world_sweep_was_configured\":"
            <<(world?"true":"false")<<",\"model_size\":"<<sizeof(model)
            <<",\"objective_size\":"<<sizeof(result.comparison_objective)
            <<",\"model_pod_hex\":";
        hex(&model,sizeof(model));
        json<<",\"objective_pod_hex\":";
        hex(&result.comparison_objective,sizeof(result.comparison_objective));
        json<<'}';
    }
    json<<",\"control_time_prediction\":{"
        <<"\"mode\":\"unity_plant\",\"age_s\":";
    finite_json(json,control_prediction.age_s);
    json<<",\"elapsed_us\":";
    finite_json(json,control_prediction_us);
    json<<",\"plant_source_stamp_ns\":"
        <<control_prediction.plant_source_stamp_ns
        <<",\"plant_state_propagated\":"
        <<(control_prediction.plant_state_valid?"true":"false")
        <<"},\"host_timing_ns\":{"
        <<"\"callback\":"<<callback_steady_ns
        <<",\"synchronize\":"<<synchronize_steady_ns<<"}"
        <<",\"solve_us\":";
    finite_json(json,solve_us);
    json<<",\"end_to_end_control_ms\":";
    finite_json(json,
        std::max(0.,double(steady_time_ns()-callback_steady_ns)*1.e-6));
    json<<",\"publication_gate_elapsed_ms\":";
    finite_json(json,last_publication_gate_elapsed_ms_);
    json<<",\"end_to_end_cycle_budget_ms\":";
    finite_json(json,end_to_end_cycle_budget_ms_);
    json<<",\"reference\":{\"speed_reference_mps\":";
    MpcTrajectorySample_t sample{};
    if(mpc_trajectory_sample(
        trajectory_.data(),trajectory_.size(),track_length_m_,progress,
        &sample))finite_json(json,sample.speed);
    else json<<"null";
    json<<"},\"reason\":\""
        <<(result.status==MPC_RTI_CYCLE_ACCEPTED_FEASIBLE?
          "source_feasible":"source_sqp_unqualified")
        <<"\"}";
    enqueue_diagnostic_stream(std::move(json));
}
} // namespace f1tenth_mpc
