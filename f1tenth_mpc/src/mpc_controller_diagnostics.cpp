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
    json<<std::setprecision(12)
        <<"{\"status\":\""<<cycle_status_name(result.status)
        <<"\",\"solver\":{"
        <<"\"backend\":\"source_exact_full_step_sqp\""
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
    json<<"],\"plant_internal_state\":[";
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
    json<<"],\"control_time_prediction\":{"
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
