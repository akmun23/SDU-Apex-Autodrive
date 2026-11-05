// Source-exact nonlinear multiple-shooting ORACLE and state necessity audit.
// No QP, no fitted/smoothed model, no approximate racing cost.
#include "mpc_source_exact_nlp.hpp"
#include "mpc_qp_failure_context.hpp"
#include "vehicle_model.h"

#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>

using json=nlohmann::json;
namespace fs=std::filesystem;
using namespace sdu_apex::source_nlp;
namespace {
constexpr int N=kStages;
constexpr int P=kPhysical;
const char *names[P]={
    "body_u","body_com_v","yaw_rate",
    "wheel_omega_FL","wheel_omega_FR","wheel_omega_RL","wheel_omega_RR",
    "body_height","body_height_rate","roll","roll_rate","pitch","pitch_rate",
    "physical_steering",
    "suspension_jounce_FL","suspension_jounce_FR",
    "suspension_jounce_RL","suspension_jounce_RR",
    "sticky_forward_FL","sticky_forward_FR",
    "sticky_forward_RL","sticky_forward_RR",
    "sticky_side_FL","sticky_side_FR","sticky_side_RL","sticky_side_RR"};
double perturbation(int i) {
    if(i==7||(i>=14&&i<=17))return .001;
    if(i==9||i==11||i==13)return .005;
    if(i>=18)return .06;
    return .02;
}
double physical_response(const State &baseline,const State &perturbed,
                         int exclude) {
    double max_error=0.;
    for(int j=0;j<18;++j) {
        if(j==exclude)continue;
        max_error=std::max(max_error,
            std::abs(perturbed.z[j]-baseline.z[j]));
    }
    for(int j=26;j<29;++j)
        max_error=std::max(max_error,
            std::abs(perturbed.z[j]-baseline.z[j]));
    return max_error;
}
json audit_sticky_timers_at_low_speed(
    const MpcRtiNominal_t &original,
    const sdu_apex::qp_fixture::FailureReplayContext &ctx)
{
    // This is NOT a fabricated tire model; it is the original physical
    // source evaluator driven into the documented <0.2 m/s sticky region,
    // with the original VehicleController zero-throttle brake branch.
    // Use timer values straddling the ORIGINAL one-second activation
    // threshold. The slow physical regime is absent from fast corner logs.
    MpcRtiState_t slow=original.states[0];
    State seed{};
    pack_state(slow,original.progress[0],&seed);
    seed.z[UNITY_SOURCE_U]=.12;
    seed.z[UNITY_SOURCE_V]=.03;
    seed.z[UNITY_SOURCE_YAW_RATE]=.05;
    seed.z[UNITY_SOURCE_PHYSICAL_STEERING]=0.;
    // Nonzero wheel RPM avoids the source's separate *immediate* sticky
    // forward shortcut, so the timer itself can be tested independently.
    for(int i=UNITY_SOURCE_WHEEL_OMEGA_FL;i<=UNITY_SOURCE_WHEEL_OMEGA_RR;++i)
        seed.z[i]=1.;
    for(int i=UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL;i<P;++i)
        seed.z[i]=i<UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL ? .5 : .5;
    double progress=original.progress[0];
    if(!inject_state(seed,&slow,&progress))return json::object();
    MpcModelControl_t braking{};
    braking.steering_target_rad=0.;
    braking.throttle_command=0.;
    json result=json::array();
    for(int index=UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL;index<P;++index) {
        MpcRtiState_t base=slow,changed=slow;
        State s0=seed,s1=seed;
        // Side timer activation depends on forward sticky history.
        if(index>=UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL) {
            for(int j=UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL;
                    j<UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL;++j) {
                s0.z[j]=1.2;
                s1.z[j]=1.2;
            }
        }
        s1.z[index]=1.2;
        double p0=progress,p1=progress;
        if(!inject_state(s0,&base,&p0)||
           !inject_state(s1,&changed,&p1))continue;
        MpcRtiState_t next0{},next1{};
        double next_p0=0.,next_p1=0.;
        if(!mpc_rti_source_exact_stage(
              &base,p0,&braking,ctx.trajectory.data(),
              ctx.trajectory.size(),ctx.lap_length_m,
              ctx.prediction_dt,&next0,&next_p0) ||
           !mpc_rti_source_exact_stage(
              &changed,p1,&braking,ctx.trajectory.data(),
              ctx.trajectory.size(),ctx.lap_length_m,
              ctx.prediction_dt,&next1,&next_p1))continue;
        State out0{},out1{};
        pack_state(next0,next_p0,&out0);
        pack_state(next1,next_p1,&out1);
        double peak_difference=0.;
        for(int w=0;w<4;++w)
            peak_difference=std::max(peak_difference,std::abs(
                next0.incoming_wheel_slip_excess_mps[w]-
                next1.incoming_wheel_slip_excess_mps[w]));
        result.push_back({
            {"state",names[index]},
            {"source_index",index},
            {"base_timer_s",s0.z[index]},
            {"test_timer_s",s1.z[index]},
            {"source_other_physical_effect",
                physical_response(out0,out1,index)},
            {"source_peak_slip_effect",peak_difference},
            {"source_evaluations_both_valid",true}
        });
    }
    return result;
}

json audit_states(const MpcRtiNominal_t &original,
                  const sdu_apex::qp_fixture::FailureReplayContext &ctx) {
    json out=json::array();
    for(int coordinate=0;coordinate<P;++coordinate) {
        double maximum_other_effect=0.,max_peak_slip_change=0.;
        int max_stage=-1;
        int changed_stages=0;
        for(int k : {0,4,10,15,20,25,29}) {
            const auto &base=original.states[k];
            State source{};
            pack_state(base,original.progress[k],&source);
            MpcRtiState_t modified=base;
            State perturbed=source;
            perturbed.z[coordinate]+=perturbation(coordinate);
            double progress=original.progress[k];
            if(!inject_state(perturbed,&modified,&progress))continue;
            MpcRtiState_t next0{},next1{};
            double s0=0.,s1=0.;
            if(!mpc_rti_source_exact_stage(&base,original.progress[k],
                  &original.controls[k],ctx.trajectory.data(),
                  ctx.trajectory.size(),ctx.lap_length_m,
                  ctx.prediction_dt,&next0,&s0))continue;
            if(!mpc_rti_source_exact_stage(&modified,progress,
                  &original.controls[k],ctx.trajectory.data(),
                  ctx.trajectory.size(),ctx.lap_length_m,
                  ctx.prediction_dt,&next1,&s1))continue;
            State z0{},z1{};
            pack_state(next0,s0,&z0);
            pack_state(next1,s1,&z1);
            const double effect=physical_response(z0,z1,coordinate);
            double slip_change=0.;
            for(int w=0;w<4;++w)
                slip_change=std::max(slip_change,std::abs(
                    next1.incoming_wheel_slip_excess_mps[w]-
                    next0.incoming_wheel_slip_excess_mps[w]));
            if(effect>1.e-8||slip_change>1.e-8)++changed_stages;
            if(effect>maximum_other_effect) {
                maximum_other_effect=effect;max_stage=k;
            }
            max_peak_slip_change=std::max(max_peak_slip_change,slip_change);
        }
        out.push_back({
            {"state",names[coordinate]},
            {"source_index",coordinate},
            {"perturbation",perturbation(coordinate)},
            {"max_other_physical_tracking_effect",maximum_other_effect},
            {"max_incoming_peak_slip_effect",max_peak_slip_change},
            {"stages_with_nontrivial_effect",changed_stages},
            {"worst_stage",max_stage},
            {"conditional_only",coordinate>=18}
        });
    }
    return out;
}
}
int main(int argc,char **argv) {
    if(argc<2||argc>5) {
        std::cerr<<"usage: mpc_source_exact_nlp_audit CASE_DIR "
            "[CONTROLS.txt] [--fd-scale=VALUE] "
            "[--control-fd-step=VALUE]\n";
        return 2;
    }
    double fd_scale=1.;
    float control_fd_step=.0002f;
    std::string controls_file;
    for(int i=2;i<argc;++i) {
        const std::string arg(argv[i]);
        if(arg.rfind("--fd-scale=",0)==0)
            fd_scale=std::stod(arg.substr(11));
        else if(arg.rfind("--control-fd-step=",0)==0)
            control_fd_step=std::stof(arg.substr(18));
        else if(controls_file.empty()) controls_file=arg;
        else return 2;
    }
    if(!(fd_scale>0.)||!std::isfinite(fd_scale)||
       !(control_fd_step>0.)||!std::isfinite(control_fd_step))return 2;
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (fs::path(argv[1])/"source_context.mctx").string(),ctx.get()))
        return 3;
    MpcRtiCycleConfiguration_t config{};
    config.model=ctx->model;
    config.model.world_sweep_check=nullptr;
    config.model.world_sweep_context=nullptr;
    config.model.world_map_rejection_only=0;
    config.objective_schedule=&ctx->objective;
    MpcModelControl_t controls[N]{};
    std::copy_n(ctx->r1_nominal,N,controls);
    if(!controls_file.empty()) {
        std::ifstream stream(controls_file);
        if(!stream)return 4;
        for(auto &c:controls)
            if(!(stream>>c.steering_target_rad>>c.throttle_command))
                return 4;
    }
    auto original=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[N+1]{};
    if(!mpc_rti_build_nominal_same_epoch(&ctx->initial_state,ctx->progress_m,
          nullptr,controls,N,ctx->trajectory.data(),ctx->trajectory.size(),
          ctx->lap_length_m,ctx->prediction_dt,N,&config.model,
          original.get(),refs))return 5;
    auto decisions=std::make_unique<Decision>();
    auto evaluation=std::make_unique<Evaluation>();
    if(!decision_from_nominal(*original,decisions.get()))return 6;
    if(!evaluate(ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
         ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
         config,*decisions,evaluation.get()))return 7;
    float source_margin=0.,source_margins[N+1][2]{};
    double source_cost=0.;
    if(!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
         original.get(),ctx->trajectory.data(),ctx->trajectory.size(),
         ctx->lap_length_m,&config,&source_cost,&source_margin,
         source_margins))return 8;
    const double objective_error=std::abs(source_cost-evaluation->objective);
    double margin_error=0.;
    for(int k=1;k<=N;++k)
        for(int side=0;side<2;++side)
            margin_error=std::max(margin_error,std::abs(double(
                evaluation->corridor[k][side]-source_margins[k][side])));
    const double original_max=evaluation->max_dynamics_abs;
    const double original_cost=evaluation->objective;

    // Disturb ONE independent multiple-shooting node, not controls.
    // Its incoming equality and next outgoing equality must respond,
    // while the exact original source objective changes without any
    // quadratic approximation. Do not silently project x onto a rollout.
    constexpr int node=12;
    decisions->nodes[node].z[UNITY_SOURCE_U]+=.01;
    const bool perturbed_ok=evaluate(
        ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        config,*decisions,evaluation.get());
    const double disturbed_incoming=perturbed_ok
        ?std::abs(evaluation->dynamics[node][UNITY_SOURCE_U]):NAN;
    double disturbed_outgoing=0.;
    if(perturbed_ok)
        for(int j=0;j<P;++j)
            disturbed_outgoing=std::max(disturbed_outgoing,
                std::abs(evaluation->dynamics[node+1][j]));
    const double disturbed_cost=perturbed_ok?evaluation->objective:NAN;

    // Exact nonlinear epigraph for the original max-over-internal-source-
    // substeps longitudinal slip penalty. No quadratic cost model.
    auto epi=std::make_unique<SlipEpigraphDecision>();
    auto epi_result=std::make_unique<SlipEpigraphEvaluation>();
    if(!decision_from_nominal(*original,&epi->shooting))return 21;
    for(int k=0;k<N;++k)
        for(int wheel=0;wheel<4;++wheel)
            epi->peak[k][wheel]=
                original->states[k+1].incoming_wheel_slip_excess_mps[wheel];
    if(!evaluate_slip_epigraph(
        ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        config,*epi,epi_result.get()))return 22;
    const double epi_cost_parity=
        std::abs(epi_result->objective-source_cost);
    const double epi_min_margin=epi_result->min_epigraph_margin;
    const double epi_nonnegative_margin=
        epi_result->min_peak_nonnegative_margin;
    const int epi_rows=epi_result->physical_source_rows;
    const int epi_constraint_count=
        epi_rows+4*N;
    auto epi_jac=std::make_unique<SlipEpigraphJacobian>();
    if(!linearize_slip_epigraph(ctx->initial_state,ctx->progress_m,
         ctx->trajectory.data(),ctx->trajectory.size(),
         ctx->lap_length_m,ctx->prediction_dt,epi->shooting,
         epi_jac.get()))return 24;
    if(epi_jac->rows.size()!=epi_result->rows.size())return 25;
    double source_substep_jac_value_parity=0.;
    for(size_t i=0;i<epi_jac->rows.size();++i) {
        const auto &a=epi_result->rows[i];
        const auto &b=epi_jac->rows[i];
        if(a.stage!=b.stage||a.wheel!=b.wheel||
           a.fixed_tick!=b.fixed_tick||
           a.vehicle_substep!=b.vehicle_substep)return 26;
        source_substep_jac_value_parity=std::max(
            source_substep_jac_value_parity,
            std::abs(a.source_excess-b.original_source_excess));
    }
    json source_substep_derivative_examples=json::array();
    for(const auto &v:epi_jac->rows)
        if(v.stage==20&&v.wheel==1&&v.vehicle_substep==0&&
           v.fixed_tick<=3)
            source_substep_derivative_examples.push_back({
                {"tick",v.fixed_tick},
                {"original_source_excess",v.original_source_excess},
                {"source_ad_d_throttle",v.control[1]},
                {"source_ad_d_steering",v.control[0]}});
    // Independent original primal-source perturbations validate that
    // the newly exposed 15,000 original source substep rows contain true
    // source-AD sensitivities, not a surrogate physical force curve.
    json epigraph_ad_independent_checks=json::array();
    double epigraph_ad_worst_scaled=0.;
    for(int stage : {5,10,16,20,29}) {
        for(int axis=0;axis<2;++axis) {
            const auto &base=original->controls[stage];
            double value=axis==0?base.steering_target_rad:
                                 base.throttle_command;
            const double limit=axis==0?ctx->model.max_steering_rad:
                                        MPC_THROTTLE_MAX;
            const double low=axis==0?-limit:MPC_THROTTLE_MIN_DRIVE;
            constexpr float h=.00001f;
            if(value-h<low||value+h>limit)continue;
            UnitySourceSlipSubstepC positive[8192]{},negative[8192]{};
            int npos=0,nneg=0;
            double positive_max[4]{},negative_max[4]{};
            MpcModelControl_t u_plus=base,u_minus=base;
            if(axis==0) {
                u_plus.steering_target_rad+=h;
                u_minus.steering_target_rad-=h;
            } else {
                u_plus.throttle_command+=h;
                u_minus.throttle_command-=h;
            }
            if(!vehicle_model_source_slip_substeps(
                    &original->states[stage].plant,&u_plus,
                    ctx->prediction_dt,positive,8192,&npos,
                    positive_max) ||
               !vehicle_model_source_slip_substeps(
                    &original->states[stage].plant,&u_minus,
                    ctx->prediction_dt,negative,8192,&nneg,
                    negative_max))return 27;
            if(npos!=nneg)return 28;
            double denom=axis==0?
                double(u_plus.steering_target_rad)-
                double(u_minus.steering_target_rad):
                double(u_plus.throttle_command)-
                double(u_minus.throttle_command);
            size_t first=0;
            while(first<epi_jac->rows.size() &&
                  epi_jac->rows[first].stage<stage)++first;
            for(int j=0;j<npos;++j) {
                const auto &row=epi_jac->rows.at(first+j);
                if(row.stage!=stage ||
                   row.wheel!=positive[j].wheel ||
                   row.wheel!=negative[j].wheel ||
                   row.fixed_tick!=positive[j].fixed_tick ||
                   row.fixed_tick!=negative[j].fixed_tick ||
                   row.vehicle_substep!=positive[j].vehicle_substep ||
                   row.vehicle_substep!=negative[j].vehicle_substep)
                    return 29;
                if(row.original_source_excess<.02)
                    continue; // away from the positive-part hinge
                const double finite=(positive[j].positive_excess_mps-
                                    negative[j].positive_excess_mps)/denom;
                const double analytic=row.control[axis];
                const double scaled=std::abs(analytic-finite)/
                    std::max(1.,std::abs(finite));
                epigraph_ad_worst_scaled=std::max(
                    epigraph_ad_worst_scaled,scaled);
                if((stage==20&&row.wheel==1 &&
                    row.vehicle_substep==0 &&
                    row.fixed_tick<=3) ||
                   (stage==5&&row.wheel==0&&j<4))
                    epigraph_ad_independent_checks.push_back({
                        {"stage",stage},{"input",axis},
                        {"tick",row.fixed_tick},
                        {"wheel",row.wheel},
                        {"original_source_AD",analytic},
                        {"original_source_independent_secant",finite},
                        {"relative_error",scaled}});
            }
        }
    }
    auto full_sparse=std::make_unique<SparseNonlinearProgram>();
    if(!assemble_sparse_nonlinear_program(
        ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        config,*epi,full_sparse.get())) {
        std::cerr<<"Sparse nonlinear assembly blocked at stage/reference: "
                 <<"invalid_reference_derivatives="
                 <<full_sparse->invalid_reference_derivatives
                 <<", knots="<<full_sparse->nonlinear_reference_knots
                 <<", rows="<<full_sparse->rows.size();
        const int stage=int(full_sparse->rows.size()-N*kState)/8+1;
        if(stage>=1 && stage<=N) {
            const double progress=original->progress[stage];
            const double half=config.model.corridor_preview_halfwidth_m;
            double near=INFINITY;
            for(const auto &p:ctx->trajectory)
                near=std::min(near,std::abs(std::remainder(
                    progress-p.s,ctx->lap_length_m)));
            std::cerr<<", stage="<<stage<<", progress="<<progress
                     <<", nearest_knot="<<near<<", halfwidth="<<half;
        }
        std::cerr<<"\n";
        return 30;
    }
    if(!full_sparse->valid ||
       full_sparse->num_variables!=kDecisionVariables+4*N ||
       full_sparse->equality_rows!=N*kState ||
       full_sparse->rows.size()!=
           static_cast<size_t>(N*(kState+8+4+4+4)+epi_rows))
        return 31;
    double worst_sparse_primal_error=0.;
    double max_sparse_dynamics=0.;
    for(int row=0;row<full_sparse->equality_rows;++row)
        max_sparse_dynamics=std::max(max_sparse_dynamics,
            std::abs(full_sparse->rows[row].value));
    // Sparse rows must contain precisely the original nonlinear
    // tire-substep inequalities in the SAME source order.
    for(int k=0;k<epi_rows;++k) {
        const auto &original_row=epi_result->rows[k];
        const auto &sparse=full_sparse->rows.at(
            full_sparse->rows.size()-epi_rows+k);
        if(sparse.type!=NonlinearRowType::OriginalSlipSubstep ||
           sparse.stage!=original_row.stage ||
           sparse.index!=original_row.wheel ||
           sparse.source_fixed_tick!=original_row.fixed_tick ||
           sparse.source_vehicle_substep!=
               original_row.vehicle_substep)return 32;
        worst_sparse_primal_error=std::max(
            worst_sparse_primal_error,
            std::abs(sparse.value-original_row.margin));
    }
    json sparse_switch_gradient{};
    for(size_t row=0;row<full_sparse->rows.size();++row) {
        const auto &r=full_sparse->rows[row];
        if(r.type!=NonlinearRowType::OriginalSlipSubstep||
           r.stage!=20||r.index!=1||
           r.source_fixed_tick!=1||
           r.source_vehicle_substep!=0)continue;
        double dthrottle=NAN,dpeak=NAN;
        for(const auto &a:full_sparse->jacobian) {
            if(a.row!=static_cast<int>(row))continue;
            if(a.column==N*kState+2*20+1)
                dthrottle=a.forward;
            if(a.column==kDecisionVariables+4*20+1)
                dpeak=a.forward;
        }
        sparse_switch_gradient={
            {"derivative_constraint_wrt_stage20_throttle",dthrottle},
            {"derivative_constraint_wrt_stage20_peak",dpeak},
            {"substep_source_margin",r.value}};
        break;
    }
    auto original_objective_gradient=
        std::make_unique<ExactSourceObjectiveGradient>();
    if(!original_source_objective_gradient(
        ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        config,*epi,original_objective_gradient.get()))return 33;
    const double source_cost_gradient_parity=std::abs(
        original_objective_gradient->original_source_objective-
        original_objective_gradient->epigraph_objective);
    const double full_epigraph_cost_gradient_parity=std::abs(
        original_objective_gradient->epigraph_objective-
        double(source_cost));
    json cost_gradient_checks=json::array();
    double worst_cost_gradient_error=0.;
    // Source objective perturbations act on truly independent shooting
    // variables; source physics + peak-slip rows are evaluated afresh,
    // while the epigraph peak is HELD as an independent variable.
    for(int stage : {4,12,20,28}) {
        for(int coord : {0,1,2,7,13,14,26,27,28}) {
            constexpr double h=.001;
            double observed[2]{};
            bool valid=true;
            for(int sign=0;sign<2;++sign) {
                auto changed=*epi;
                changed.shooting.nodes[stage].z[coord]+=
                    sign==0?h:-h;
                auto trial=std::make_unique<SlipEpigraphEvaluation>();
                if(!evaluate_slip_epigraph(
                    ctx->initial_state,ctx->progress_m,
                    ctx->trajectory.data(),ctx->trajectory.size(),
                    ctx->lap_length_m,ctx->prediction_dt,config,
                    changed,trial.get())) {
                    valid=false;break;
                }
                observed[sign]=trial->objective;
            }
            if(!valid)continue;
            const double fd=(observed[0]-observed[1])/(2.*h);
            const double ad=original_objective_gradient->state[stage][coord];
            const double err=std::abs(fd-ad)/
                std::max(1.,std::abs(fd));
            worst_cost_gradient_error=std::max(worst_cost_gradient_error,err);
            cost_gradient_checks.push_back({
                {"variable","state"},{"stage",stage},
                {"coordinate",coord},{"source_objective_fd",fd},
                {"derived_original_source_gradient",ad},
                {"relative_error",err}});
        }
    }
    for(int stage : {5,16,20,29}) {
        for(int input=0;input<2;++input) {
            constexpr double h=.0002;
            const float base=input==0?
                epi->shooting.controls[stage].steering_target_rad:
                epi->shooting.controls[stage].throttle_command;
            const double upper=input==0?
                double(ctx->model.max_steering_rad):MPC_THROTTLE_MAX;
            const double lower=input==0?
                -double(ctx->model.max_steering_rad):
                MPC_THROTTLE_MIN_DRIVE;
            if(double(base)-h<lower || double(base)+h>upper)
                continue;
            double observed[2]{};
            double perturbed_physical[2]{};
            bool valid=true;
            for(int sign=0;sign<2;++sign) {
                auto changed=*epi;
                auto &u=changed.shooting.controls[stage];
                if(input==0)u.steering_target_rad+=
                    sign==0?float(h):-float(h);
                else u.throttle_command+=
                    sign==0?float(h):-float(h);
                perturbed_physical[sign]=input==0?
                    double(u.steering_target_rad):
                    double(u.throttle_command);
                auto trial=std::make_unique<SlipEpigraphEvaluation>();
                if(!evaluate_slip_epigraph(
                    ctx->initial_state,ctx->progress_m,
                    ctx->trajectory.data(),ctx->trajectory.size(),
                    ctx->lap_length_m,ctx->prediction_dt,config,
                    changed,trial.get())) {
                    valid=false;break;
                }
                observed[sign]=trial->objective;
            }
            if(!valid)continue;
            const double fd=(observed[0]-observed[1])/
                (perturbed_physical[0]-perturbed_physical[1]);
            const double ad=original_objective_gradient->
                control[stage][input];
            const double err=std::abs(fd-ad)/
                std::max(1.,std::abs(fd));
            worst_cost_gradient_error=std::max(worst_cost_gradient_error,err);
            cost_gradient_checks.push_back({
                {"variable","control"},{"stage",stage},
                {"coordinate",input},{"source_objective_fd",fd},
                {"derived_original_source_gradient",ad},
                {"relative_error",err}});
        }
    }
    // Keep the original candidate peak, but change one physical control:
    // if another substep becomes the maximum, the original nonlinear
    // inequality MUST detect it, despite a frozen-peak gradient of zero.
    double branch_switch_margin=NAN;
    double branch_switch_source_change=NAN;
    int epigraph_branch_switch_stage=20;
    int epigraph_branch_switch_wheel=1;
    const auto saved_throttle=epi->shooting.controls[20].throttle_command;
    if(saved_throttle+0.0002f<=MPC_THROTTLE_MAX) {
        epi->shooting.controls[20].throttle_command+=.0002f;
        if(!evaluate_slip_epigraph(
            ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
            ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
            config,*epi,epi_result.get()))return 23;
        branch_switch_margin=epi_result->min_epigraph_margin;
        branch_switch_source_change=
            epi_result->source.source_peak_slip[20][1]-
            epi->peak[20][1];
        epi->shooting.controls[20].throttle_command=saved_throttle;
    }
    // Independent original-source finite response for sparse NLP
    // derivatives. These are validation secants ONLY; the optimizer's
    // source Jacobian uses the original C++ physical AD and tracking AD.
    auto jacobian=std::make_unique<DynamicsJacobian>();
    auto trials=std::make_unique<Decision>();
    auto original_decisions=std::make_unique<Decision>();
    auto trial_result=std::make_unique<Evaluation>();
    if(!decision_from_nominal(*original,original_decisions.get()) ||
       !linearize_dynamics(ctx->initial_state,ctx->progress_m,
           ctx->trajectory.data(),ctx->trajectory.size(),
           ctx->lap_length_m,ctx->prediction_dt,*original_decisions,
           jacobian.get()))return 11;
    auto original_peak_gradient=std::make_unique<PeakSlipObjectiveGradient>();
    if(!source_peak_slip_objective_gradient(
            config,*jacobian,original_peak_gradient.get()))return 13;
    auto base_shooting=std::make_unique<Evaluation>();
    if(!evaluate(ctx->initial_state,ctx->progress_m,
           ctx->trajectory.data(),ctx->trajectory.size(),
           ctx->lap_length_m,ctx->prediction_dt,config,
           *original_decisions,base_shooting.get()))return 14;
    const double peak_cost_parity=std::abs(
        original_peak_gradient->cost-
        base_shooting->source_peak_slip_cost);
    json original_peak_gradient_checks=json::array();
    for(int stage : {0,5,10,16,20,29}) {
        for(int axis=0;axis<2;++axis) {
            constexpr double h=.0002;
            const double v=axis==0
                ?original_decisions->controls[stage].steering_target_rad
                :original_decisions->controls[stage].throttle_command;
            const double upper=axis==0?
                double(ctx->model.max_steering_rad):
                double(MPC_THROTTLE_MAX);
            const double lower=axis==0?-upper:
                double(MPC_THROTTLE_MIN_DRIVE);
            if(v-h<lower || v+h>upper)continue;
            double costs[2]{};
            bool ok=true;
            for(int signIndex=0;signIndex<2;++signIndex) {
                *trials=*original_decisions;
                const double shift=(signIndex==0?h:-h);
                if(axis==0)
                    trials->controls[stage].steering_target_rad+=
                        static_cast<float>(shift);
                else
                    trials->controls[stage].throttle_command+=
                        static_cast<float>(shift);
                if(!evaluate(ctx->initial_state,ctx->progress_m,
                    ctx->trajectory.data(),ctx->trajectory.size(),
                    ctx->lap_length_m,ctx->prediction_dt,config,*trials,
                    trial_result.get())) {
                    ok=false;break;
                }
                costs[signIndex]=trial_result->source_peak_slip_cost;
            }
            if(!ok)continue;
            const double fd=(costs[0]-costs[1])/(2.*h);
            const double ad=original_peak_gradient->control[stage][axis];
            json active_source_branches=json::array();
            // The chosen max-substep is an ACTIVE SET in the original
            // source. The gradient at a strict local max is not the same
            // derivative as a finite step selecting a different substep.
            // Document both physical branch decisions, never "correct"
            // the original Jacobian with a fitted secant.
            if(stage==20 && axis==1) {
                for(const double delta : {-h,0.,h}) {
                    MpcModelControl_t u=original_decisions->
                        controls[stage];
                    u.throttle_command+=static_cast<float>(delta);
                    UnitySourcePeakSlipTraceC trace[4]{};
                    if(!vehicle_model_trace_peak_longitudinal_slip(
                        &original->states[stage].plant,&u,
                        ctx->prediction_dt,trace))return 15;
                    for(int wheel=0;wheel<4;++wheel) {
                        active_source_branches.push_back({
                            {"control_delta",delta},{"wheel",wheel},
                            {"fixed_tick",trace[wheel].active_fixed_tick},
                            {"vehicle_substep",
                             trace[wheel].active_vehicle_substep},
                            {"source_peak",trace[wheel].traced_max_excess_mps},
                            {"runner_up",trace[wheel].runner_up_excess_mps}});
                    }
                }
            }
            original_peak_gradient_checks.push_back({
                {"stage",stage},{"input",axis},
                {"original_source_active_branch_AD",ad},
                {"independent_source_central_difference",fd},
                {"relative_error",std::abs(fd-ad)/
                    std::max(1.,std::abs(fd))},
                {"h",h},
                {"original_source_active_substep_audit",active_source_branches}});
        }
    }
    json jacobian_checks=json::array();
    double worst_jacobian_error=0.;
    double worst_jacobian_scaled_error=0.;
    for(int stage : {0,5,10,16,22,29}) {
        for(int column : {0,1,2,3,7,9,13,14,18,22,26,27,28,29,30}) {
            if(stage==0 && column<29)continue;
            *trials=*original_decisions;
            double h=column==7||column==14?.0002:
                     column==29||column==30?.0005:.0005;
            h*=fd_scale;
            // At the original legal command box, derivatives are
            // ONE-SIDED. Moving outside the source command domain is
            // never a valid independent physics derivative check.
            if(column==29 || column==30) {
                const auto &u=original_decisions->controls[stage];
                double old=column==29?u.steering_target_rad:u.throttle_command;
                double limit=column==29?ctx->model.max_steering_rad:1.;
                double minimum=column==29?-limit:MPC_THROTTLE_MIN_DRIVE;
                if(old+h>limit)h=-h;
                if(old+h<minimum)continue;
                if(column==29)
                    trials->controls[stage].steering_target_rad+=float(h);
                else
                    trials->controls[stage].throttle_command+=float(h);
            } else {
                trials->nodes[stage-1].z[column]+=h;
            }
            if(!evaluate(ctx->initial_state,ctx->progress_m,
                ctx->trajectory.data(),ctx->trajectory.size(),
                ctx->lap_length_m,ctx->prediction_dt,config,*trials,
                trial_result.get()))continue;
            double max_abs=0.,max_scaled=0.,max_response=0.;
            int worst_row=-1;
            for(int row=0;row<kState;++row) {
                const double original_dynamics=0.; // source exact nominal
                const double measured=(trial_result->dynamics[stage][row]-
                                        original_dynamics)/h;
                const double predicted=column==29||column==30
                    ?-jacobian->stages[stage].B[row][column-29]
                    :-jacobian->stages[stage].A[row][column];
                const double error=std::abs(measured-predicted);
                const double scaled=error/std::max(1.,std::abs(measured));
                max_response=std::max(max_response,std::abs(measured));
                if(error>max_abs){max_abs=error;worst_row=row;}
                max_scaled=std::max(max_scaled,scaled);
            }
            worst_jacobian_error=std::max(worst_jacobian_error,max_abs);
            worst_jacobian_scaled_error=std::max(
                worst_jacobian_scaled_error,max_scaled);
            jacobian_checks.push_back({
                {"stage",stage},
                {"variable",column},
                {"h",h},
                {"max_original_source_AD_difference",max_abs},
                {"max_scaled_difference",max_scaled},
                {"max_source_response",max_response},
                {"worst_output_state",worst_row}});
        }
    }
    // Tracking/progress coordinates are not physical source memory, but
    // are genuine inputs to the ORIGINAL path projector/footprint model.
    // Test their causal effect on the NEXT exact nonlinear equality
    // without incorrectly counting a direct node's identity residual.
    json tracking_necessity=json::array();
    for(int col=kPhysical;col<kState;++col) {
        *trials=*original_decisions;
        constexpr double h=.005;
        trials->nodes[12].z[col]+=h;
        if(!evaluate(ctx->initial_state,ctx->progress_m,
             ctx->trajectory.data(),ctx->trajectory.size(),
             ctx->lap_length_m,ctx->prediction_dt,config,*trials,
             trial_result.get()))continue;
        double next_max=0.;
        for(int row=0;row<kState;++row)
            next_max=std::max(next_max,
                std::abs(trial_result->dynamics[13][row]));
        tracking_necessity.push_back({
            {"coordinate",col},
            {"name",col==kPhysical?"e_y":
                      col==kPhysical+1?"e_psi":"path_progress"},
            {"next_nonlinear_dynamics_effect",next_max},
            {"exact_source_objective_delta",
                trial_result->objective-original_cost}});
    }
    // The maximum longitudinal-slip penalty is a MAX of the original
    // 1ms/vehicle-substep tire expressions, NOT a stage-end slip proxy.
    // Observe the actual active time index and tied active substeps using
    // the original instrumented source (no modeled substitute).
    json peak_branches=json::array();
    double maximum_peak_slip_trace_error=0.;
    int exact_peak_ties=0;
    int peak_branch_samples=0;
    for(int stage=0;stage<N;++stage) {
        UnitySourcePeakSlipTraceC active[4]{};
        if(!vehicle_model_trace_peak_longitudinal_slip(
            &original->states[stage].plant,&original->controls[stage],
            ctx->prediction_dt,active))return 12;
        for(int wheel=0;wheel<4;++wheel) {
            const double captured=
                original->states[stage+1].incoming_wheel_slip_excess_mps[wheel];
            const double error=std::abs(
                captured-active[wheel].actual_source_max_excess_mps);
            maximum_peak_slip_trace_error=std::max(
                maximum_peak_slip_trace_error,error);
            if(active[wheel].equal_max_substeps>1 &&
               active[wheel].actual_source_max_excess_mps>0.)
                ++exact_peak_ties;
            ++peak_branch_samples;
            peak_branches.push_back({
                {"stage",stage+1},
                {"wheel",wheel},
                {"source_max_excess",captured},
                {"source_traced_max_excess",
                    active[wheel].traced_max_excess_mps},
                {"active_fixed_tick",active[wheel].active_fixed_tick},
                {"active_vehicle_substep",
                    active[wheel].active_vehicle_substep},
                {"equal_maximum_substeps",active[wheel].equal_max_substeps},
                {"runner_up_slip_excess",
                    active[wheel].runner_up_excess_mps},
                {"traced_original_substeps",
                    active[wheel].traced_substeps}});
        }
    }
    // Full original-source rollout gradient in the control-only feasible
    // manifold. This uses ALL original state dynamics equalities exactly:
    // no QP cost, no frozen/max-slippage surrogate, no finite-difference
    // model inside the optimizer.
    auto reduced_gradient=std::make_unique<ReducedSourceGradient>();
    if(!source_exact_reduced_gradient(
        ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),
        ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
        config,*original_decisions,reduced_gradient.get()))return 34;
    json reduced_gradient_checks=json::array();
    double worst_reduced_gradient_error=0.;
    for(int stage : {0,5,10,16,20,29}) {
        for(int axis=0;axis<2;++axis) {
            const double base=axis==0?
                original_decisions->controls[stage].steering_target_rad:
                original_decisions->controls[stage].throttle_command;
            const double upper=axis==0?
                double(ctx->model.max_steering_rad):MPC_THROTTLE_MAX;
            const double lower=axis==0?
                -double(ctx->model.max_steering_rad):MPC_THROTTLE_MIN_DRIVE;
            const float step=control_fd_step;
            if(base-step<lower||base+step>upper)continue;
            double costs[2]{}, commands_used[2]{};
            bool valid=true;
            for(int sign=0;sign<2;++sign) {
                MpcModelControl_t candidate[N]{};
                std::copy_n(controls,N,candidate);
                if(axis==0)
                    candidate[stage].steering_target_rad+=
                        sign==0?step:-step;
                else candidate[stage].throttle_command+=
                        sign==0?step:-step;
                commands_used[sign]=axis==0?
                    candidate[stage].steering_target_rad:
                    candidate[stage].throttle_command;
                auto fresh=std::make_unique<MpcRtiNominal_t>();
                MpcRtiReference_t references[N+1]{};
                if(!mpc_rti_build_nominal_same_epoch(
                    &ctx->initial_state,ctx->progress_m,nullptr,
                    candidate,N,ctx->trajectory.data(),ctx->trajectory.size(),
                    ctx->lap_length_m,ctx->prediction_dt,N,&config.model,
                    fresh.get(),references)) { valid=false;break; }
                double minimum=0.,objective=0.;
                float minimum_float=0.,margin[N+1][2]{};
                if(!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
                    fresh.get(),ctx->trajectory.data(),
                    ctx->trajectory.size(),ctx->lap_length_m,
                    &config,&objective,&minimum_float,margin)) {
                    valid=false;break;
                }
                costs[sign]=objective;
            }
            if(!valid)continue;
            const double finite=(costs[0]-costs[1])/
                (commands_used[0]-commands_used[1]);
            const double source_ad=reduced_gradient->control[stage][axis];
            const double scaled_error=std::abs(finite-source_ad)/
                std::max(1.,std::abs(finite));
            worst_reduced_gradient_error=std::max(
                worst_reduced_gradient_error,scaled_error);
            reduced_gradient_checks.push_back({
                {"stage",stage},{"input",axis},
                {"source_nonlin_full_rollout_gradient",finite},
                {"source_nlp_reduced_adjoint_gradient",source_ad},
                {"relative_error",scaled_error}});
        }
    }
    json report={
        {"schema","source-exact-nlp-evaluation/v6"},
        {"case",argv[1]},
        {"authority","mpc_rti_source_exact_stage + original source scorer"},
        {"optimizer","nonlinear multiple-shooting objective/constraints oracle; no solver"},
        {"states_per_stage",kState},
        {"physical_states_per_stage",P},
        {"tracking_states_per_stage",3},
        {"nonlinear_equality_rows",kDynamicsEquations},
        {"original_peak_slip_branch_samples",peak_branch_samples},
        {"original_peak_slip_trace_max_error",
            maximum_peak_slip_trace_error},
        {"original_peak_slip_positive_exact_ties",exact_peak_ties},
        {"original_peak_slip_active_substep_branches",peak_branches},
        {"decision_variables",kDecisionVariables},
        {"source_objective",source_cost},
        {"nlp_objective",original_cost},
        {"objective_parity_abs_error",objective_error},
        {"source_margin_abs_error",margin_error},
        {"nominal_dynamics_max_abs_residual",original_max},
        {"source_peak_slip_cost_parity_error",peak_cost_parity},
        {"slip_epigraph_source_objective_parity_error",epi_cost_parity},
        {"source_exact_reduced_gradient_source_cost",
         reduced_gradient->source_cost},
        {"source_exact_reduced_gradient_fd_step",control_fd_step},
        {"source_exact_reduced_gradient_active_epigraph_rows",
         reduced_gradient->active_peak_rows},
        {"source_exact_reduced_gradient_tied_peak_branches",
         reduced_gradient->source_branch_ties},
        {"source_exact_reduced_gradient_worst_scaled_error",
         worst_reduced_gradient_error},
        {"source_exact_reduced_gradient_independent_checks",
         reduced_gradient_checks},
        {"source_original_full_objective_gradient_cost_parity_error",
         full_epigraph_cost_gradient_parity},
        {"source_original_full_objective_gradient_peak_delta",
         source_cost_gradient_parity},
        {"source_original_full_objective_derivative_worst_scaled_error",
         worst_cost_gradient_error},
        {"source_original_full_objective_derivative_samples",
         cost_gradient_checks},
        {"source_original_full_objective_nonsmooth_wall_stages",
         original_objective_gradient->nonsmooth_wall_reserve_stages},
        {"source_original_full_objective_nonsmooth_reference_stages",
         original_objective_gradient->nonsmooth_reference_progress_stages},
        {"source_original_full_objective_lateral_slip_stages",
         original_objective_gradient->evaluated_source_lateral_slip_stages},
        {"sparse_source_nlp_variables",full_sparse->num_variables},
        {"sparse_source_nlp_constraint_rows",full_sparse->rows.size()},
        {"sparse_source_nlp_jacobian_entries",
         full_sparse->jacobian.size()},
        {"sparse_source_nlp_nonlinear_equality_max_residual",
         max_sparse_dynamics},
        {"sparse_source_nlp_substep_constraint_parity",
         worst_sparse_primal_error},
        {"sparse_source_nlp_original_reference_knots",
         full_sparse->nonlinear_reference_knots},
        {"sparse_source_nlp_reference_derivative_invalid",
         full_sparse->invalid_reference_derivatives},
        {"sparse_source_nlp_stage20_switch_gradient",
         sparse_switch_gradient},
        {"slip_epigraph_source_derivative_rows",
         epi_jac->rows.size()},
        {"slip_epigraph_derivative_source_value_parity",
         source_substep_jac_value_parity},
        {"slip_epigraph_source_AD_secant_worst_scaled",
         epigraph_ad_worst_scaled},
        {"slip_epigraph_source_AD_independent_examples",
         epigraph_ad_independent_checks},
        {"slip_epigraph_original_substep_AD_evaluations",
         epi_jac->original_source_ad_calls},
        {"slip_epigraph_stage20_wheel1_derivative_examples",
         source_substep_derivative_examples},
        {"slip_epigraph_original_substep_constraints",epi_rows},
        {"slip_epigraph_total_constraints_including_positivity",
         epi_constraint_count},
        {"slip_epigraph_nominal_minimum_constraint",epi_min_margin},
        {"slip_epigraph_nonnegative_peak_minimum",
         epi_nonnegative_margin},
        {"slip_epigraph_fixed_peak_trial_minimum_constraint",
         branch_switch_margin},
        {"slip_epigraph_fixed_peak_trial_source_peak_change",
         branch_switch_source_change},

        {"source_peak_slip_active_gradient_checks",
            original_peak_gradient_checks},
        {"sparse_nlp_source_ad_calls",jacobian->original_source_ad_calls},
        {"sparse_nlp_nonlinear_stage_calls",
         jacobian->source_nonlinear_stage_calls},
        {"sparse_nlp_jacobian_independent_checks",jacobian_checks},
        {"sparse_nlp_worst_source_AD_abs_difference",worst_jacobian_error},
        {"sparse_nlp_worst_source_AD_scaled_difference",
         worst_jacobian_scaled_error},
        {"perturbation_node",node+1},
        {"perturbation_delta_u_mps",.01},
        {"perturbed_evaluation_valid",perturbed_ok},
        {"perturbed_incoming_dynamics_defect",disturbed_incoming},
        {"perturbed_outgoing_dynamics_defect",disturbed_outgoing},
        {"perturbed_exact_source_objective",disturbed_cost},
        {"state_necessity",audit_states(*original,*ctx)},
        {"tracking_state_necessity",tracking_necessity},
        {"low_speed_braking_sticky_state_necessity",
         audit_sticky_timers_at_low_speed(*original,*ctx)}
    };
    std::cout<<std::setprecision(17)<<report.dump(2)<<'\n';
    if(original_max>5.e-5||objective_error>1.e-4||
       margin_error>5.e-5||!perturbed_ok||
       disturbed_incoming<.009||disturbed_outgoing<1.e-7||
       maximum_peak_slip_trace_error>1.e-9 ||
       peak_cost_parity>1.e-8 || epi_cost_parity>1.e-8 ||
       epi_min_margin < -1.e-9 || epi_nonnegative_margin < 0. ||
       source_substep_jac_value_parity>1.e-12 ||
       worst_sparse_primal_error>1.e-10 ||
       max_sparse_dynamics>1.e-9)
        return 10;
    return 0;
}
