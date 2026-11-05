// Long-lived offline oracle of the SAME recorded nonlinear problem. Used for
// independent mathematical solver comparisons, never linked into ROS control.
#include "mpc_qp_failure_context.hpp"
#include "mpc_source_exact_compact.hpp"
#include "mpc_source_exact_qp.hpp"
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
            opts.capture_curvature_model=true;opts.linearization_audit_only=true;
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
        double merit=score->objective;
        double gradient[2*kStages]{};
        for(int j=0;j<2*kStages;++j)
            if(derivatives)gradient[j]=derivative->original_source_cost_control[j/2][j%2]*(j%2?.25:.15);
        const double reserve=request.value("preferred_clearance_m",.03);
        const double weight=request.value("preferred_clearance_weight",1.e6);
        for(int k=0;k<kStages;++k)
            for(int side=0;side<8;++side) {
                const auto &c=score->corridor_corners[k];
                const double deficit=std::min(0.,(side<4?c.left[side]:c.right[side-4])-reserve);
                merit+=.5*weight*deficit*deficit;
            }
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
                    if(row.type==NonlinearRowType::BodyCorner)
                        gradient[j]+=weight*std::min(0.,qp->constraint_margins[r]-reserve)*a;
                }
                result["hard_jacobian_scaled"].push_back(jac);
            }
            for(double g:gradient)result["gradient_scaled"].push_back(g);
        }
        result["objective"]=merit;
        return result;
    };
    for(std::string line;std::getline(std::cin,line);) {
        try {std::cout<<evaluate(json::parse(line)).dump()<<'\n'<<std::flush;}
        catch(const std::exception &e){std::cout<<json({{"valid",false},{"reason",e.what()}}).dump()<<'\n'<<std::flush;}
    }
    return 0;
}
