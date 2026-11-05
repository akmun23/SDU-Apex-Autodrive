#include "mpc_source_exact_offline_solver.hpp"
#include "mpc_source_exact_qp.hpp"
#include "mpc_source_exact_compact.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include "vehicle_model.h"

namespace sdu_apex::source_nlp {
namespace {
struct Snapshot {
    std::unique_ptr<MpcRtiNominal_t> nominal{
        std::make_unique<MpcRtiNominal_t>()};
    std::unique_ptr<Evaluation> evaluation{std::make_unique<Evaluation>()};
    std::unique_ptr<Decision> decision{std::make_unique<Decision>()};
    double cost{};
    double corridor{};
    double rate{};
    double box{};
    double footprint_violation{};
    double rate_violation{};
    double total_violation{};
    double exact_constraint_merit{};
    bool valid{};
};

bool exact_source_rollout(const MpcRtiState_t &initial,
    double progress,const MpcTrajectorySample_t *track,size_t count,
    double lap,float dt,const MpcRtiCycleConfiguration_t &cfg,
    const MpcModelControl_t commands[kStages],
    const OfflineOptions &options,Snapshot &out,
    const MpcRtiNominal_t *same_epoch_prefix=nullptr,
    bool source_speed_ceiling_preconfigured=false)
{
    // Snapshot objects are reused after accepted source trials. Clear
    // ALL accumulated physical inequalities before every independent
    // original source rollout, including terminal continuation.
    out.cost=0.;
    out.corridor=0.;
    out.rate=0.;
    out.box=0.;
    out.footprint_violation=0.;
    out.rate_violation=0.;
    out.total_violation=0.;
    out.exact_constraint_merit=0.;
    out.valid=false;
    // IMPORTANT: the normal nominal *seed* builder projects inputs onto
    // actuator-rate bounds. A nonlinear solver must not silently optimize a
    // different (projected) command sequence: physical rate violations are
    // ORIGINAL inequality rows, and the trial's original racing cost must
    // correspond to precisely the controls supplied by the optimizer.
    // Integrate the unmodified commands with the authoritative source stage.
    // A changed suffix may reuse an identical, already source-integrated
    // prefix from the SAME initial state, track and control epoch.
    if(!source_speed_ceiling_preconfigured &&
       !vehicle_model_set_active_target_speed_ceiling(
           cfg.model.active_speed_ceiling_mps))
        return false;
    auto &nominal=*out.nominal;
    int changed_stage=0;
    const bool reusable=
        same_epoch_prefix && same_epoch_prefix->valid &&
        same_epoch_prefix->horizon==kStages &&
        same_epoch_prefix->progress[0]==progress &&
        std::memcmp(&same_epoch_prefix->states[0],&initial,
                    sizeof(initial))==0;
    if(reusable) {
        while(changed_stage<kStages &&
              same_epoch_prefix->controls[changed_stage].
                  steering_target_rad==
                  commands[changed_stage].steering_target_rad &&
              same_epoch_prefix->controls[changed_stage].
                  throttle_command==
                  commands[changed_stage].throttle_command)
            ++changed_stage;
        // Copy ONLY the source-integrated prefix whose original float
        // controls and 29-state initial epoch match. The changed suffix is
        // always reintegrated with original physics. Most QP candidates
        // differ already at stage zero, avoiding a deep 31-state copy.
        if(changed_stage>0) {
            std::copy_n(same_epoch_prefix->states,changed_stage+1,
                        nominal.states);
            std::copy_n(same_epoch_prefix->progress,changed_stage+1,
                        nominal.progress);
        } else {
            nominal.states[0]=initial;
            nominal.progress[0]=progress;
        }
    } else {
        nominal.states[0]=initial;
        nominal.progress[0]=progress;
    }
    nominal.horizon=kStages;
    nominal.valid=0;
    for(int k=0;k<kStages;++k)
        nominal.controls[k]=commands[k];
    for(int k=changed_stage;k<kStages;++k)
        if(!mpc_rti_source_exact_stage(
                &nominal.states[k],nominal.progress[k],&nominal.controls[k],
                track,count,lap,dt,&nominal.states[k+1],
                &nominal.progress[k+1]))
            return false;
    nominal.valid=1;
    MpcRtiReference_t refs[kStages+1]{};
    if(!mpc_rti_references_from_exact_nominal(
            &nominal,track,count,lap,&cfg.model,refs))
        return false;
    for(int k=0;k<=kStages;++k)
        nominal.speed_reference[k]=refs[k].u;
    nominal.speed_reference_valid=1;
    if(!decision_from_nominal(nominal,out.decision.get()) ||
       !evaluate_existing_source_nominal(
           initial,progress,track,count,lap,dt,cfg,
           *out.nominal,*out.decision,out.evaluation.get()))
        return false;
    const auto &e=*out.evaluation;
    if(!e.valid || e.max_dynamics_abs>1.e-8 ||
       !std::isfinite(e.objective) ||
       !std::isfinite(e.minimum_corridor_slack_m) ||
       !std::isfinite(e.minimum_command_rate_margin) ||
       !std::isfinite(e.minimum_command_box_margin))
        return false;
    out.cost=e.objective;
    out.corridor=e.minimum_corridor_slack_m;
    out.rate=e.minimum_command_rate_margin;
    out.box=e.minimum_command_box_margin;
    out.footprint_violation=0.;
    out.rate_violation=0.;
    // This is the numerical inequality merit evaluated by the ORIGINAL
    // source footprint and actuator equations at every actual trial.
    for(const auto &corners:e.corridor_corners)
        for(int j=0;j<4;++j) {
            const double left=std::min(0.,
                corners.left[j]-options.minimum_source_clearance_m);
            const double right=std::min(0.,
                corners.right[j]-options.minimum_source_clearance_m);
            out.footprint_violation+=.5*(left*left+right*right);
        }
    for(const auto &margins:e.command_rate)
        for(const double margin:margins) {
            const double v=std::min(0.,margin);
            out.rate_violation+=.5*v*v;
        }
    out.total_violation=out.footprint_violation+out.rate_violation;
    out.exact_constraint_merit=out.total_violation;
    if(!std::isfinite(out.exact_constraint_merit))return false;
    // The ORIGINAL wall/footprint corridor is always a hard physical
    // boundary. The separately requested *extra* clearance is a float
    // geometric threshold and can differ by submicrons from the exact
    // evaluated source margin. Only that reserve admits numerical
    // tolerance; never permit physical wall penetration.
    out.valid=
        out.corridor>=0. &&
        out.corridor>=options.minimum_source_clearance_m-
                         options.allowable_corridor_violation_m &&
        out.rate>=-options.allowable_command_rate_violation &&
        out.box>=-1.e-7;
    return true;
}


bool accept_step(const Snapshot &next,const Snapshot &current,
                 const OfflineOptions &options)
{
    if(next.valid!=current.valid)return next.valid;
    if(next.valid)return next.cost<=
        current.cost-options.minimum_source_cost_reduction;
    return next.total_violation<=
        current.total_violation-options.minimum_violation_reduction;
}
double wall_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
}
} // namespace

bool optimize_offline_exact_source(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const MpcModelControl_t seed[kStages],
    const OfflineOptions &options,OfflineResult *result)
{
    if(!result || !trajectory || trajectory_count==0 || !seed ||
       options.maximum_iterations<1 || options.maximum_iterations>100 ||
       options.source_qp_max_iterations<1 ||
       options.source_qp_max_iterations>5000 ||
       !std::isfinite(options.source_qp_curvature_regularization) ||
       options.source_qp_curvature_regularization<=0. ||
       !std::isfinite(options.source_qp_soft_physical_weight) ||
       options.source_qp_soft_physical_weight<=0. ||
       !std::isfinite(options.minimum_source_clearance_m) ||
       options.minimum_source_clearance_m<0. ||
       options.allowable_corridor_violation_m<0. ||
       options.allowable_command_rate_violation<0. ||
       options.absolute_deadline_monotonic_us<0. ||
       options.minimum_remaining_us_for_new_iteration<0.)
        return false;
    *result={};
    auto current=std::make_unique<Snapshot>();
    auto proposed=std::make_unique<Snapshot>();
    if(!exact_source_rollout(
        initial,initial_progress,trajectory,trajectory_count,
        lap_length,dt,configuration,seed,options,*current,
        options.initial_same_epoch_source_nominal))
        return false;
    std::vector<double> inequality_multipliers;
    std::vector<SourceNonlinearCurvatureSecant> source_secants;
    auto derivative=std::make_unique<ReducedSourceGradient>();
    auto program=std::make_unique<SparseNonlinearProgram>();
    bool have_current_exact_source_linearization=false;
    double qp_regularization=options.source_qp_curvature_regularization;
    constexpr double kMaxRegularization=100000.;
    constexpr double kRejectedFullStepDampingGrowth=3.;
    result->initial_source_feasible=current->valid;
    result->initial_source_cost=current->cost;
    result->initial_exact_constraint_violation=current->total_violation;
    result->initial_minimum_corridor_slack=current->corridor;
    const auto overdue=[&]() {
        return options.absolute_deadline_monotonic_us>0. &&
            mpc_rti_monotonic_time_us()>=
                options.absolute_deadline_monotonic_us;
    };
    for(int iteration=0;iteration<options.maximum_iterations;++iteration) {
        if(overdue() ||
           (options.absolute_deadline_monotonic_us>0. &&
            options.minimum_remaining_us_for_new_iteration>0. &&
            options.absolute_deadline_monotonic_us-
                mpc_rti_monotonic_time_us()<
                options.minimum_remaining_us_for_new_iteration)) {
            result->deadline_exhausted=true;
            break;
        }
        const auto iter_begin=std::chrono::steady_clock::now();
        OfflineIteration metrics{};
        metrics.iteration=iteration;
        metrics.cost_before=current->cost;
        metrics.cost_after=current->cost;
        metrics.violation_before=current->total_violation;
        metrics.violation_after=current->total_violation;
        metrics.corridor_slack_after=current->corridor;

        const auto lin_begin=std::chrono::steady_clock::now();
        // The convex subproblem is a LOCAL model of the SAME original
        // constrained NLP. Physical equality constraints are eliminated
        // using the authoritative source stage Jacobians; not a second
        // vehicle model, fallback QP or multiple-shooting optimization.
        // Rejected full steps do NOT invalidate the exact current
        // primal, source AD or physical Jacobian. Re-solve the SAME
        // constrained SQP quadratic model with stronger regularization,
        // rather than paying for identical source Jacobians again.
        metrics.qp_regularization_used=qp_regularization;
        metrics.reused_exact_source_linearization=
            have_current_exact_source_linearization;
        if(!have_current_exact_source_linearization) {
            const double rho=current->valid?0.:
                std::max(2.e6,3.*(1.+std::abs(current->cost))/
                         std::max(1.e-9,current->total_violation));
            const ConstraintMeritWeights merit_weights{
                rho,rho,options.minimum_source_clearance_m};
            const bool valid_derivatives=
                options.use_source_compact_qp_linearization?
                source_exact_compact_qp_linearization(
                    initial,initial_progress,trajectory,trajectory_count,
                    lap_length,dt,configuration,*current->decision,
                    *current->evaluation,*current->nominal,merit_weights,
                    derivative.get(),program.get()):
                source_exact_reduced_gradient(
                    initial,initial_progress,trajectory,trajectory_count,
                    lap_length,dt,configuration,*current->decision,
                    derivative.get(),merit_weights,program.get(),
                    current->evaluation.get(),current->nominal.get());
            if(!valid_derivatives || !derivative->valid || !program->valid)
                return false;
            if(std::abs(derivative->exact_source_violation_half_squared-
                        current->total_violation)>1.e-8*
                   std::max(1.,current->total_violation))
                return false;
            have_current_exact_source_linearization=true;
        }
        metrics.linearization_wall_ms=wall_ms(lin_begin);
        if(overdue()) {
            result->deadline_exhausted=true;
            break;
        }
        auto qp=std::make_unique<SourceConstrainedQpResult>();
        SourceConstrainedQpOptions qp_opts{};
        qp_opts.requested_corner_clearance_m=
            options.minimum_source_clearance_m;
        qp_opts.curvature_regularization=qp_regularization;
        qp_opts.inequality_penalty_weight=
            options.source_qp_soft_physical_weight;
        qp_opts.maximum_qp_iterations=options.source_qp_max_iterations;
        qp_opts.inequality_multipliers=
            inequality_multipliers.empty()?
                nullptr:&inequality_multipliers;
        // The original nonlinear objective is NEVER switched off while
        // satisfying physical constraints. All SQP steps solve the same
        // racing problem with coupled source-derived GN curvature.
        qp_opts.source_configuration=&configuration;
        qp_opts.source_nominal=current->nominal.get();
        qp_opts.source_curvature_secants=
            source_secants.empty()?nullptr:&source_secants;
        metrics.calibrated_source_secants=
            static_cast<int>(source_secants.size());
        if(!source_exact_constrained_qp_direction(
            *program,derivative->original_source_cost_control,
            current->decision->controls,qp_opts,qp.get()))
            return false;
        metrics.qp_iterations=qp->iterations;
        metrics.qp_converged=qp->converged;
        metrics.qp_dual_residual=qp->dual_residual;
        metrics.qp_primal_residual=qp->primal_residual;
        metrics.source_nlp_stationarity_inf=
            qp->original_nlp_stationarity_inf;
        metrics.source_nlp_inequality_residual_inf=
            qp->original_nlp_inequality_residual_inf;
        metrics.source_nlp_complementarity_inf=
            qp->original_nlp_complementarity_inf;
        metrics.qp_wall_ms=qp->wall_ms;
        metrics.predicted_cost_change=qp->predicted_original_objective_change;
        if(overdue()) {
            result->deadline_exhausted=true;
            break;
        }

        // Exactly ONE full QP step. No alpha sampling, alternate direction,
        // physical-control trust box or second optimizer is permitted.
        // Every proposed source state is independently reintegrated.
        MpcModelControl_t commands[kStages]{};
        bool changed=false;
        for(int k=0;k<kStages;++k) {
            const auto &u=current->decision->controls[k];
            auto &v=commands[k];
            v=u;
            v.steering_target_rad=static_cast<float>(std::clamp(
                double(u.steering_target_rad)+qp->direction[k][0],
                -double(configuration.model.max_steering_rad),
                double(configuration.model.max_steering_rad)));
            v.throttle_command=static_cast<float>(std::clamp(
                double(u.throttle_command)+qp->direction[k][1],
                double(MPC_THROTTLE_MIN_DRIVE),
                double(MPC_THROTTLE_MAX)));
            changed|=v.steering_target_rad!=u.steering_target_rad ||
                     v.throttle_command!=u.throttle_command;
        }
        if(!changed) {
            // A float-rounded zero control update alone proves nothing.
            // Require original nonlinear physical feasibility AND the
            // ORIGINAL constrained-NLP first-order KKT residuals.
            constexpr double kStationarityTolerance=1.e-2;
            constexpr double kComplementarityTolerance=1.e-4;
            result->source_nlp_kkt_converged=current->valid &&
                qp->original_nlp_stationarity_inf<=
                    kStationarityTolerance &&
                qp->original_nlp_inequality_residual_inf<=
                    options.allowable_corridor_violation_m+1.e-7 &&
                qp->original_nlp_complementarity_inf<=
                    kComplementarityTolerance &&
                qp->converged;
            if(!result->source_nlp_kkt_converged)
                result->full_step_sqp_stalled=true;
            metrics.iteration_wall_ms=wall_ms(iter_begin);
            result->history.push_back(metrics);
            break;
        }
        const auto rollout_begin=std::chrono::steady_clock::now();
        ++result->source_trial_evaluations;
        const bool evaluated=exact_source_rollout(
            initial,initial_progress,trajectory,trajectory_count,
            lap_length,dt,configuration,commands,options,*proposed,
            current->nominal.get());
        metrics.source_rollout_wall_ms=wall_ms(rollout_begin);
        metrics.nonlinear_evaluation_failed=!evaluated;
        if(evaluated) {
            metrics.cost_trial=proposed->cost;
            metrics.violation_trial=proposed->total_violation;
            metrics.actual_cost_change=proposed->cost-current->cost;
        }
        if(evaluated && accept_step(*proposed,*current,options)) {
            metrics.accepted=true;
            metrics.cost_after=proposed->cost;
            metrics.violation_after=proposed->total_violation;
            metrics.corridor_slack_after=proposed->corridor;
            // Update dual variables from the ACTUAL newly integrated
            // original nonlinear inequalities, not the QP's predicted
            // margins. This keeps outer SQP duals consistent across
            // reference and source-physics branch changes.
            std::vector<double> previous_multipliers=
                std::move(inequality_multipliers);
            inequality_multipliers=
                std::move(qp->inequality_multipliers);
            const auto &e=*proposed->evaluation;
            for(size_t r=0;r<program->rows.size();++r) {
                const auto &row=program->rows[r];
                double actual=0.;
                bool update=false;
                if(row.type==NonlinearRowType::BodyCorner) {
                    const auto &corners=e.corridor_corners[row.stage];
                    const int i=row.index;
                    actual=(i<4?corners.left[i]:corners.right[i-4])-
                        options.minimum_source_clearance_m;
                    update=true;
                } else if(row.type==
                          NonlinearRowType::PhysicalCommandRate) {
                    actual=e.command_rate[row.stage][row.index];
                    update=true;
                }
                if(update) {
                    const double old=previous_multipliers.empty()?
                        0.:previous_multipliers[r];
                    const double dual=std::max(0.,
                        old-options.source_qp_soft_physical_weight*actual);
                    if(!std::isfinite(dual))return false;
                    inequality_multipliers[r]=dual;
                }
            }
            std::swap(current,proposed);
            source_secants.clear();
            have_current_exact_source_linearization=false;
            result->full_step_sqp_stalled=false;
            ++result->accepted_iterations;
        } else {
            metrics.full_step_rejected=true;
            result->full_step_sqp_stalled=true;
            if(evaluated && source_secants.size()<4) {
                // Curvature calibration uses the ACTUAL nonlinear source
                // evaluation already performed in this SQP iteration.
                // It is a positive-semidefinite secant correction along
                // the rejected direction of the SAME nonlinear problem.
                constexpr double kScale[2]={.15,.25};
                std::array<double,2*kStages> unit{};
                double magnitude_squared=0.;
                for(int k=0;k<kStages;++k)
                    for(int axis=0;axis<2;++axis) {
                        const double component=
                            qp->direction[k][axis]/kScale[axis];
                        unit[2*k+axis]=component;
                        magnitude_squared+=component*component;
                    }
                if(magnitude_squared>1.e-12) {
                    const double predicted_merit_delta=
                        qp->predicted_original_objective_change+
                        options.source_qp_soft_physical_weight*
                        (qp->predicted_linearized_violation_after-
                         qp->predicted_linearized_violation_before);
                    const double actual_merit_delta=
                        (proposed->cost-current->cost)+
                        options.source_qp_soft_physical_weight*
                        (proposed->total_violation-
                         current->total_violation);
                    const double missing_curvature=
                        std::max(0.,2.*(actual_merit_delta-
                              predicted_merit_delta)/
                              magnitude_squared);
                    if(std::isfinite(missing_curvature) &&
                       missing_curvature>1.e-6) {
                        const double norm=std::sqrt(magnitude_squared);
                        for(double &value:unit)value/=norm;
                        const double curvature=std::min(
                            1.e9,missing_curvature);
                        source_secants.push_back({unit,curvature});
                        metrics.secant_curvature_added=curvature;
                    }
                }
            }
            // This is Hessian stabilization of ONE nonlinear SQP,
            // NOT a previous-iterate command trust region, alternate
            // optimizer, alpha grid or permission to publish infeasibility.
            // Every attempted full step uses the original source plant.
            qp_regularization=std::min(kMaxRegularization,
                qp_regularization*kRejectedFullStepDampingGrowth);
        }
        metrics.iteration_wall_ms=wall_ms(iter_begin);
        result->history.push_back(metrics);
        if(!metrics.accepted &&
           (qp_regularization<=metrics.qp_regularization_used ||
            iteration+1>=options.maximum_iterations))
            break;
    }
    result->optimized=*current->decision;
    result->final_source_feasible=current->valid;
    result->final_source_cost=current->cost;
    result->final_exact_constraint_violation=current->total_violation;
    result->final_minimum_corridor_slack=current->corridor;
    result->final_minimum_rate_margin=current->rate;
    return true;
}
} // namespace sdu_apex::source_nlp
