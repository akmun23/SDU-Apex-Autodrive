// Long-lived offline oracle of the SAME recorded nonlinear problem. Used for
// independent mathematical solver comparisons, never linked into ROS control.
#include "mpc_qp_failure_context.hpp"
#include "mpc_source_exact_compact.hpp"
#include "mpc_source_exact_qp.hpp"
#include "vehicle_model.h"
#include <algorithm>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <iostream>
#include <memory>

using namespace sdu_apex::source_nlp;
using json=nlohmann::json;
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (std::filesystem::path(argv[1])/"source_context.mctx").string(),ctx.get()))return 3;
    MpcRtiCycleConfiguration_t cfg{};cfg.model=ctx->model;cfg.objective_schedule=&ctx->objective;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto decision=std::make_unique<Decision>();
    auto score=std::make_unique<Evaluation>();
    auto derivative=std::make_unique<ReducedSourceGradient>();
    auto program=std::make_unique<SparseNonlinearProgram>();
    auto qp=std::make_unique<SourceConstrainedQpResult>();
    const auto evaluate=[&](const json &request)->json {
        if(request.value("preferred_clearance_weight",0.)!=0.)
            return {{"valid",false},{"reason","extra_reserve_is_not_original_objective"}};
        if(request.value("metadata",false)) {
            json refs=json::array(),speed=json::array();
            for(int k=0;k<=kStages;++k) {
                const auto &r=ctx->objective.references[k];
                refs.push_back({r.e_y,r.e_psi,r.u,r.v,r.r,r.steering_command,
                    r.path_curvature,r.left_bound,r.right_bound,r.source_witness_valid,
                    r.source_steering_target,r.source_throttle});
                speed.push_back(ctx->objective.speed_weights[k]);
            }
            return {{"valid",true},{"references",refs},{"speed_weights",speed},
                {"terminal_targets_valid",ctx->objective.terminal_targets_valid},
                {"successor_recourse_valid",ctx->objective.successor_recourse_valid}};
        }
        const auto &u=request.at("controls");
        if(u.size()!=kStages)return {{"valid",false},{"reason","control_shape"}};
        const bool derivatives=request.value("derivatives",true);
        nominal->states[0]=ctx->initial_state;nominal->progress[0]=ctx->progress_m;
        nominal->horizon=kStages;
        for(int k=0;k<kStages;++k) {
            nominal->controls[k]={u[k][0].get<float>(),u[k][1].get<float>()};
            if(!mpc_rti_source_exact_stage(&nominal->states[k],nominal->progress[k],
                &nominal->controls[k],ctx->trajectory.data(),ctx->trajectory.size(),ctx->lap_length_m,
                ctx->prediction_dt,&nominal->states[k+1],&nominal->progress[k+1]))
                return {{"valid",false},{"reason","source_stage"}};
        }
        nominal->valid=1;
        if(!decision_from_nominal(*nominal,decision.get()) || !evaluate_existing_source_nominal(
            ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),ctx->trajectory.size(),ctx->lap_length_m,
            ctx->prediction_dt,cfg,*nominal,*decision,score.get()))
            return {{"valid",false},{"reason","source_score"}};
        if(derivatives) {
            const ConstraintMeritWeights weights{0.,0.,.03};
            if(!source_exact_compact_qp_linearization(ctx->initial_state,ctx->progress_m,
                ctx->trajectory.data(),ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,cfg,
                *decision,*score,*nominal,weights,derivative.get(),program.get()))
                return {{"valid",false},{"reason","source_derivative"}};
            SourceConstrainedQpOptions opts;
            opts.source_configuration=&cfg;opts.source_nominal=nominal.get();
            opts.capture_curvature_model=true;opts.linearization_audit_only=!request.value("qp_diagnostics",false);
            if(!source_exact_constrained_qp_direction(*program,derivative->original_source_cost_control,
                nominal->controls,opts,qp.get()))return {{"valid",false},{"reason","condensation"}};
        }
        json result={{"valid",true},{"original_cost",score->objective},
            {"minimum_clearance_m",score->minimum_corridor_slack_m},
            {"minimum_rate_margin",score->minimum_command_rate_margin},
            {"minimum_box_margin",score->minimum_command_box_margin},
            {"gradient_scaled",json::array()},{"hard_margins",json::array()},
            {"hard_jacobian_scaled",json::array()},{"row_types",json::array()},
            {"max_steering_rad",cfg.model.max_steering_rad},
            {"rate_constraints_redundant",
                cfg.model.max_steering_command_step_rad>=2.f*cfg.model.max_steering_rad &&
                cfg.model.max_steering_rad>=SOURCE_MAX_STEERING_RAD &&
                cfg.model.max_throttle_command_step>=MPC_THROTTLE_MAX-MPC_THROTTLE_MIN_DRIVE &&
                ctx->initial_state.plant.delayed_throttle_command_2>=MPC_THROTTLE_MIN_DRIVE}};
        if(derivatives && request.value("qp_diagnostics",false)) {
            result["qp_original_duals_trustworthy"]=qp->original_duals_trustworthy;
            result["physical_multipliers"]=json::array();
            for(int row:qp->constraint_row_indices)
                result["physical_multipliers"].push_back(qp->inequality_multipliers[row]);
        }
        double gradient[2*kStages]{};
        for(int j=0;j<2*kStages;++j)
            if(derivatives)gradient[j]=derivative->original_source_cost_control[j/2][j%2]*(j%2?.25:.15);
        if(derivatives) {
            for(size_t r=0;r<qp->constraint_row_indices.size();++r) {
                const auto &row=program->rows[qp->constraint_row_indices[r]];
                if(row.type==NonlinearRowType::TerminalSpeedViability)continue;
                result["hard_margins"].push_back(qp->constraint_margins[r]);
                result["row_types"].push_back(row.type==NonlinearRowType::BodyCorner?"body":
                    row.type==NonlinearRowType::PhysicalCommandRate?"rate":"box");
                json jac=json::array();
                for(int j=0;j<2*kStages;++j) {
                    const double a=qp->constraint_jacobian_scaled[r*2*kStages+j];
                    jac.push_back(a);
                }
                result["hard_jacobian_scaled"].push_back(jac);
            }
            for(double g:gradient)result["gradient_scaled"].push_back(g);
        }
        if(request.contains("lagrangian_hessian_multipliers")) {
            const auto &requested=request.at("lagrangian_hessian_multipliers");
            if(!requested.is_array() || requested.size()!=16*kStages)
                return {{"valid",false},{"reason","hessian_multiplier_shape"}};
            std::array<double,16*kStages> multipliers{};
            for(size_t j=0;j<multipliers.size();++j) {
                multipliers[j]=requested[j].get<double>();
                if(!std::isfinite(multipliers[j]) || multipliers[j]<0.)
                    return {{"valid",false},{"reason","invalid_hessian_multiplier"}};
            }
            SourceLagrangianHessian hessian{};
            const bool valid=source_lagrangian_hessian(*nominal,*score,ctx->trajectory.data(),
                ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,cfg,multipliers,&hessian);
            result["lagrangian_hessian"]={{"valid",valid},
                {"source_directional_evaluations",hessian.source_directional_evaluations},
                {"quadratic_consistency_relative",hessian.quadratic_consistency_relative}};
            if(valid)result["lagrangian_hessian"]["scaled_row_major"]=hessian.scaled;
        }
        if(request.contains("constraint_direction")) {
            const auto &requested=request.at("constraint_direction");
            if(requested.size()!=kStages)return {{"valid",false},{"reason","direction_shape"}};
            double direction[kStages][2]{};
            for(int k=0;k<kStages;++k) {
                if(requested[k].size()!=2)return {{"valid",false},{"reason","direction_shape"}};
                for(int j=0;j<2;++j)direction[k][j]=requested[k][j].get<double>();
            }
            SourceConstraintCurvature curvature{};
            const bool valid=source_constraint_path_curvature(*nominal,*score,ctx->trajectory.data(),
                ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,cfg,direction,&curvature);
            result["constraint_path_model"]={{"valid",valid},{"first_invalid_stage",curvature.first_invalid_stage},
                {"physical_switch_ties",curvature.physical_switch_ties},
                {"selected_switch_ties",curvature.selected_switch_ties},
                {"reference_knots",curvature.reference_knots},
                {"contact_switches",curvature.contact_switches},
                {"substep_switches",curvature.substep_switches},{"brake_switches",curvature.brake_switches}};
            if(valid) {
                result["constraint_path_model"]["first"]=curvature.first;
                result["constraint_path_model"]["second"]=curvature.second;
                result["constraint_path_model"]["original_objective_first"]=curvature.original_objective_first;
                result["constraint_path_model"]["original_objective_second"]=curvature.original_objective_second;
            }
        }
        if(request.contains("reference_controls")) {
            auto reference=std::make_unique<MpcRtiNominal_t>();
            reference->states[0]=ctx->initial_state;reference->progress[0]=ctx->progress_m;
            const auto &old=request.at("reference_controls");
            if(old.size()!=kStages)return {{"valid",false},{"reason","reference_control_shape"}};
            for(int k=0;k<kStages;++k) {
                reference->controls[k]={old[k][0].get<float>(),old[k][1].get<float>()};
                if(!mpc_rti_source_exact_stage(&reference->states[k],reference->progress[k],
                    &reference->controls[k],ctx->trajectory.data(),ctx->trajectory.size(),ctx->lap_length_m,
                    ctx->prediction_dt,&reference->states[k+1],&reference->progress[k+1]))
                    return {{"valid",false},{"reason","reference_source_stage"}};
            }
            reference->horizon=kStages;reference->valid=1;
            auto reference_decision=std::make_unique<Decision>();
            auto reference_score=std::make_unique<Evaluation>();
            if(!decision_from_nominal(*reference,reference_decision.get()) ||
               !evaluate_existing_source_nominal(ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
                ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,cfg,*reference,
                *reference_decision,reference_score.get()))return {{"valid",false}};
            SourceCurvatureBranches changes{};
            if(!compare_source_curvature_branches(*reference,*reference_score,*nominal,*score,
                ctx->trajectory.data(),ctx->trajectory.size(),ctx->lap_length_m,
                ctx->prediction_dt,&changes))return {{"valid",false}};
            json physical=json::object();
#define RECORD_BRANCH(name) physical[#name]=changes.physical.name
            RECORD_BRANCH(mismatched_substep_schedule_events);
            RECORD_BRANCH(contact_grounded_changes);
            RECORD_BRANCH(sticky_forward_changes);
            RECORD_BRANCH(sticky_side_changes);
            RECORD_BRANCH(longitudinal_spline_region_changes);
            RECORD_BRANCH(lateral_spline_region_changes);
            RECORD_BRANCH(tire_load_filter_changes);
            RECORD_BRANCH(combined_slip_saturation_changes);
            RECORD_BRANCH(suspension_jounce_region_changes);
            RECORD_BRANCH(slip_denominator_region_changes);
            RECORD_BRANCH(steering_target_reached_changes);
            RECORD_BRANCH(throttle_brake_mode_changes);
#undef RECORD_BRANCH
            result["physical_branch_changes"]=physical;
            result["peak_branch_changes"]=changes.peak_changes;
            result["geometry_branch_changes"]=changes.reference_changes;
            result["tracking_reference_segment_changes"]=changes.tracking_reference_changes;
        }
        result["objective"]=score->objective;
        return result;
    };
    for(std::string line;std::getline(std::cin,line);) {
        try {std::cout<<evaluate(json::parse(line)).dump()<<'\n'<<std::flush;}
        catch(const std::exception &e){std::cout<<json({{"valid",false},{"reason",e.what()}}).dump()<<'\n'<<std::flush;}
    }
    return 0;
}
