#include "mpc_source_exact_offline_solver.hpp"
#include "mpc_source_exact_qp.hpp"
#include "mpc_source_exact_compact.hpp"
#include "mpc_source_terminal_speed_viability.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include "vehicle_model.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>

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
    double hard_violation{};
    double hard_violation_l1{};
    double terminal_speed_margin_mps{};
    double terminal_speed_violation{};
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
    out.hard_violation=0.;
    out.hard_violation_l1=0.;
    out.terminal_speed_margin_mps=INFINITY;
    out.terminal_speed_violation=0.;
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
    if(options.terminal_speed_viability_weight>0.) {
        TerminalSpeedViability terminal{};
        if(!terminal_source_speed_viability(
             track,count,lap,*out.nominal,
             options.terminal_speed_allowance_mps,&terminal))
            return false;
        out.terminal_speed_margin_mps=terminal.margin_mps;
        const double deficit=std::min(0.,terminal.margin_mps);
        out.terminal_speed_violation=.5*deficit*deficit;
    }
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
            const double hard_left=std::min(0.,corners.left[j]);
            const double hard_right=std::min(0.,corners.right[j]);
            out.hard_violation+=.5*(hard_left*hard_left+hard_right*hard_right);
            out.hard_violation_l1-=hard_left+hard_right;
        }
    for(const auto &margins:e.command_rate)
        for(const double margin:margins) {
            const double v=std::min(0.,margin);
            out.rate_violation+=.5*v*v;
            out.hard_violation_l1-=v;
        }
    out.total_violation=out.footprint_violation+out.rate_violation;
    out.hard_violation+=out.rate_violation;
    for(const auto &margins:e.command_box)
        for(double margin:margins) {
            const double v=std::min(0.,margin);out.hard_violation+=.5*v*v;
            out.hard_violation_l1-=v;
        }
    out.exact_constraint_merit=out.total_violation;
    if(!std::isfinite(out.exact_constraint_merit))return false;
    // The physical wall remains HARD (zero penetration). The additional
    // nominal safety reserve is an ELASTIC inequality of the SAME
    // nonlinear program, not an independent collision boundary. An extra
    // 2mm reserve deficit must never suppress a command whose EXACT
    // source footprint remains inside the wall. Extra reserve is diagnostic;
    // genuine original-source reserve costs remain inside out.cost.
    out.valid=
        out.corridor>=0. &&
        out.rate>=-options.allowable_command_rate_violation &&
        out.box>=-1.e-7;
    return true;
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
       options.maximum_iterations<1 || options.maximum_iterations>2000 ||
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
       !std::isfinite(options.terminal_speed_viability_weight) ||
       options.terminal_speed_viability_weight<0. ||
       options.terminal_speed_viability_weight>1.e8 ||
       !std::isfinite(options.terminal_speed_allowance_mps) ||
       options.terminal_speed_allowance_mps<0. ||
       options.terminal_speed_allowance_mps>3. ||
       (options.terminal_speed_viability_weight>0. &&
        !options.use_source_compact_qp_linearization) ||
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
    result->exact_source_evaluations=1;
    // Monotone feasibility restoration within this SQP. No racing-cost
    // decrease can compensate for larger physical violation. Once the
    // exact source iterate is feasible, preserve it and optimize original J.
    const auto accept_step=[&](const Snapshot &next,const Snapshot &origin,
        const OfflineOptions &settings,double directional_slope=0.,double alpha=1.) {
        if(origin.valid && !next.valid)return false;
        if(next.valid && !origin.valid)return true;
        const double value=origin.valid?origin.cost:origin.hard_violation;
        const double candidate=origin.valid?next.cost:next.hard_violation;
        const double requested=origin.valid?settings.minimum_source_cost_reduction:
            settings.minimum_violation_reduction;
        const double sufficient=-1.e-4*alpha*std::min(0.,directional_slope);
        const double roundoff=32.*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(value));
        return candidate<value-std::max({requested,sufficient,roundoff});
    };
    int rejected_iterations=0;
    int source_precision_stagnation_iterations=0;
    std::vector<double> inequality_multipliers;
    // Constraint row identities guard carried hard duals and BFGS
    // multipliers across nonsmooth reference/source branch transitions.
    std::vector<SparseNonlinearRow> previous_row_identities;
    bool previous_row_identities_valid=false;
    auto lagrangian_memory=std::make_unique<SourceLagrangianMemory>();
    auto derivative=std::make_unique<ReducedSourceGradient>();
    auto program=std::make_unique<SparseNonlinearProgram>();
    bool have_current_exact_source_linearization=false;
    std::array<double,4*kStages*kStages> reference_lagrangian_hessian{};
    bool have_current_reference_hessian=false;
    bool current_reference_hessian_evaluated=false;
    double qp_regularization=options.source_qp_curvature_regularization;
    // Scaled source curvature can exceed 1e8 near tire saturation. A fixed
    // 1e5 ceiling stopped globalization before it could produce a local step.
    constexpr double kMaxRegularization=1.e12;
    result->initial_source_feasible=current->valid;
    result->initial_source_cost=current->cost;
    result->initial_terminal_speed_margin_mps=
        current->terminal_speed_margin_mps;
    result->initial_terminal_speed_violation=
        current->terminal_speed_violation;
    result->initial_exact_constraint_violation=current->total_violation;
    result->initial_minimum_corridor_slack=current->corridor;
    const auto overdue=[&]() {
        return options.absolute_deadline_monotonic_us>0. &&
            mpc_rti_monotonic_time_us()>=
                options.absolute_deadline_monotonic_us;
    };
    double measured_iteration_us=0.;
    for(int iteration=0;iteration<options.maximum_iterations;++iteration) {
        if(overdue() ||
           (options.absolute_deadline_monotonic_us>0. &&
            options.minimum_remaining_us_for_new_iteration>0. &&
            options.absolute_deadline_monotonic_us-
                mpc_rti_monotonic_time_us()<
            std::max(options.minimum_remaining_us_for_new_iteration,
                     measured_iteration_us))) {
            result->deadline_exhausted=true;
            result->iteration_admission_stopped=!overdue();
            break;
        }
        const auto iter_begin=std::chrono::steady_clock::now();
        OfflineIteration metrics{};
        metrics.iteration=iteration;
        metrics.physical_violation_before=current->hard_violation;
        metrics.physical_violation_after=current->hard_violation;
        metrics.maximum_hard_violation_before=std::max({0.,-current->corridor,-current->rate,-current->box});
        metrics.maximum_hard_violation_after=metrics.maximum_hard_violation_before;
        metrics.cost_before=current->cost;
        metrics.cost_after=current->cost;
        metrics.violation_before=current->total_violation;
        metrics.violation_after=current->total_violation;
        metrics.corridor_slack_after=current->corridor;
        std::unique_ptr<OfflinePrimalTrace> trace;
        if(options.record_primal_trace) {
            trace=std::make_unique<OfflinePrimalTrace>();
            trace->iteration=iteration;
            trace->warm_start=*current->decision;
        }

        const auto record_iteration=[&]() {
            metrics.iteration_wall_ms=wall_ms(iter_begin);
            // Optional refinement must fit the work actually observed in
            // this cycle, not only the nominal admission estimate. This
            // leaves the independently verified current NLP iterate intact.
            measured_iteration_us=std::max(measured_iteration_us,
                1000.*metrics.iteration_wall_ms);
            metrics.exact_source_evaluations=result->exact_source_evaluations;
            result->history.push_back(metrics);
            if(trace) {
                trace->accepted=metrics.accepted;
                result->primal_trace.push_back(std::move(*trace));
            }
        };

        const auto lin_begin=std::chrono::steady_clock::now();
        // The convex subproblem is a LOCAL model of the SAME original
        // constrained NLP. Physical equality constraints are eliminated
        // using the authoritative source stage Jacobians; not a second
        // vehicle model, fallback QP or multiple-shooting optimization.
        // Rejected full steps do NOT invalidate the exact current
        // primal, source AD or physical Jacobian. Re-solve the SAME
        // constrained SQP quadratic model with revised damping,
        // rather than paying for identical source Jacobians again.
        metrics.qp_regularization_used=qp_regularization;
        metrics.reused_exact_source_linearization=
            have_current_exact_source_linearization;
        if(!have_current_exact_source_linearization) {
            const double rho=0.;
            const ConstraintMeritWeights merit_weights{
                rho,rho,0.,
                options.terminal_speed_viability_weight,
                options.terminal_speed_allowance_mps,
                options.source_slip_epigraph};
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
            if(!valid_derivatives || !derivative->valid || !program->valid) {
                result->linearization_failed=true;
                metrics.linearization_failed=true;
                record_iteration();break;
            }
            if(std::abs(derivative->exact_source_violation_half_squared-
                        current->hard_violation)>1.e-8*
                   std::max(1.,current->hard_violation))
                return false;
            have_current_exact_source_linearization=true;
            if(previous_row_identities_valid &&
               !same_original_source_constraint_layout(
                    previous_row_identities,program->rows)) {
                // An active source constraint may change ID even when
                // the number of rows is unchanged. Never reuse a dual or
                // source Lagrangian curvature with a different physical
                // constraint. The current exact primal remains untouched.
                inequality_multipliers.clear();
                *lagrangian_memory={};
                metrics.source_constraint_layout_reset=true;
                ++result->source_constraint_layout_resets;
            }
            previous_row_identities=program->rows;
            previous_row_identities_valid=true;
        }
        metrics.linearization_wall_ms=wall_ms(lin_begin);
        if(overdue()) {
            result->deadline_exhausted=true;
            record_iteration();break;
        }
        auto qp=std::make_unique<SourceConstrainedQpResult>();
        const auto record_qp_failure=[&]() {
            const bool expired=qp->deadline_exceeded || overdue();
            result->deadline_exhausted=expired;
            result->qp_failed=!expired;metrics.qp_failed=!expired;
            result->numerical_failure=!expired;
            result->termination_reason=expired?"deadline_exhausted":qp->failure_reason;
            result->final_physical_multipliers.clear();
            metrics.normal_step_computed=qp->normal_step_computed;
            metrics.normal_primal_residual=qp->normal_primal_residual;
            metrics.normal_dual_residual=qp->normal_dual_residual;
            metrics.normal_linearized_violation=qp->normal_linearized_violation;
            metrics.qp_primal_residual=qp->primal_residual;
            metrics.qp_dual_residual=qp->dual_residual;
            metrics.qp_complementarity_residual=qp->complementarity_residual;
            metrics.qp_iterations=qp->iterations;
            record_iteration();
        };
        SourceConstrainedQpOptions qp_opts{};
        qp_opts.requested_corner_clearance_m=
            options.minimum_source_clearance_m;
        qp_opts.curvature_regularization=qp_regularization;
        qp_opts.inequality_penalty_weight=
            options.source_qp_soft_physical_weight;
        qp_opts.terminal_speed_viability_weight=
            options.terminal_speed_viability_weight;
        qp_opts.maximum_qp_iterations=options.source_qp_max_iterations;
        qp_opts.inequality_multipliers=
            inequality_multipliers.empty()?
                nullptr:&inequality_multipliers;
        // The original nonlinear objective is NEVER switched off while
        // satisfying physical constraints. All SQP steps solve the same
        // racing problem with coupled source-derived GN curvature.
        qp_opts.source_configuration=&configuration;
        qp_opts.source_nominal=current->nominal.get();
        qp_opts.lagrangian_memory=lagrangian_memory.get();
        // Source feature Jacobians change during restoration too. Refresh
        // their known curvature at every new source primal; retain only the
        // learned residual part of the previous Lagrangian approximation.
        qp_opts.refresh_known_gauss_newton=true;
        qp_opts.absolute_deadline_monotonic_us=options.absolute_deadline_monotonic_us;
        qp_opts.capture_curvature_model=
            options.record_primal_trace || options.exact_lagrangian_hessian;
        metrics.qp_attempted=true;
        if(!source_exact_constrained_qp_direction(
            *program,derivative->original_source_cost_control,
            current->decision->controls,qp_opts,qp.get())) {
            record_qp_failure();break;
        }
        if(options.exact_lagrangian_hessian && !options.source_slip_epigraph) {
            constexpr int dimension=2*kStages;
            if(!current_reference_hessian_evaluated) {
                current_reference_hessian_evaluated=true;
                const auto hessian_begin=std::chrono::steady_clock::now();
                std::array<double,16*kStages> multipliers{};
                // Restoration has no original-NLP dual certificate. Zero
                // multipliers still give the exact objective Hessian; do
                // not silently lose all source curvature there, or use
                // elastic/equality duals as physical multipliers.
                for(size_t r=0;qp->original_duals_trustworthy && r<program->rows.size();++r) {
                    const auto &row=program->rows[r];
                    int index=-1;
                    if(row.type==NonlinearRowType::BodyCorner)index=8*row.stage+row.index;
                    if(row.type==NonlinearRowType::PhysicalCommandRate)index=8*kStages+4*row.stage+row.index;
                    if(row.type==NonlinearRowType::PhysicalCommandBox)index=12*kStages+4*row.stage+row.index;
                    if(index>=0)multipliers[index]=qp->inequality_multipliers[r];
                }
                SourceLagrangianHessian exact{};
                if(source_lagrangian_hessian(*current->nominal,*current->evaluation,
                    trajectory,trajectory_count,lap_length,dt,configuration,multipliers,
                    &exact,options.absolute_deadline_monotonic_us) && !overdue()) {
                    Eigen::Map<const Eigen::Matrix<double,dimension,dimension,Eigen::RowMajor>> raw(exact.scaled.data());
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(raw);
                    if(eigen.info()==Eigen::Success) {
                        metrics.lagrangian_hessian_min_eigenvalue=eigen.eigenvalues().minCoeff();
                        const double floor=std::max(1.e-3,1.e-9*eigen.eigenvalues().cwiseAbs().maxCoeff());
                        const Eigen::MatrixXd stabilized=eigen.eigenvectors()*
                            eigen.eigenvalues().cwiseMax(floor).asDiagonal()*eigen.eigenvectors().transpose();
                        for(int i=0;i<dimension;++i)for(int j=0;j<dimension;++j)
                            reference_lagrangian_hessian[i*dimension+j]=stabilized(i,j);
                        have_current_reference_hessian=true;
                    }
                }
                result->exact_source_evaluations+=exact.source_directional_evaluations;
                metrics.hessian_source_directional_evaluations=exact.source_directional_evaluations;
                metrics.hessian_quadratic_consistency_relative=exact.quadratic_consistency_relative;
                metrics.lagrangian_hessian_wall_ms=wall_ms(hessian_begin);
            }
            metrics.lagrangian_hessian_valid=have_current_reference_hessian;
            if(have_current_reference_hessian && !overdue()) {
                qp_opts.lagrangian_hessian_scaled=&reference_lagrangian_hessian;
                qp_opts.lagrangian_memory=nullptr;
                if(!source_exact_constrained_qp_direction(*program,derivative->original_source_cost_control,
                    current->decision->controls,qp_opts,qp.get())) {
                    record_qp_failure();break;
                }
            }
        }
        metrics.normal_step_computed=qp->normal_step_computed;
        metrics.normal_only_step=qp->normal_only_step;
        metrics.normal_primal_residual=qp->normal_primal_residual;
        metrics.normal_dual_residual=qp->normal_dual_residual;
        metrics.normal_linearized_violation=qp->normal_linearized_violation;
        metrics.linearized_physical_violation_inf=qp->linearized_physical_violation_inf;
        metrics.original_duals_trustworthy=qp->original_duals_trustworthy;
        metrics.qp_complementarity_residual=qp->complementarity_residual;
        metrics.qp_hessian_min_eigenvalue=qp->hessian_min_eigenvalue;
        metrics.qp_hessian_max_eigenvalue=qp->hessian_max_eigenvalue;
        const auto physical_duals=[&]() {
            std::vector<double> values;
            for(size_t r=0;r<program->rows.size();++r) {
                const auto type=program->rows[r].type;
                if(type==NonlinearRowType::BodyCorner ||
                   type==NonlinearRowType::PhysicalCommandRate ||
                   type==NonlinearRowType::PhysicalCommandBox)
                    values.push_back(qp->inequality_multipliers[r]);
            }
            return values;
        };
        if(trace)trace->physical_multipliers=physical_duals();
        // A dual estimate is not a convergence claim. The qualification
        // oracle scores it at the final exact source controls independently.
        if(options.record_primal_trace && qp->original_duals_trustworthy)
            result->final_physical_multipliers=physical_duals();
        if(!qp->original_duals_trustworthy)result->final_physical_multipliers.clear();
        metrics.qp_iterations=qp->iterations;
        metrics.qp_converged=qp->converged;
        metrics.qp_dual_residual=qp->dual_residual;
        metrics.qp_primal_residual=qp->primal_residual;
        metrics.qp_tolerance_used=qp->tolerance_used;
        metrics.lagrangian_bfgs_updated=qp->lagrangian_bfgs_updated;
        metrics.lagrangian_bfgs_damping=qp->lagrangian_bfgs_damping;
        metrics.source_nlp_stationarity_inf=
            qp->original_nlp_stationarity_inf;
        metrics.source_nlp_inequality_residual_inf=
            qp->original_nlp_inequality_residual_inf;
        metrics.source_nlp_complementarity_inf=
            qp->original_nlp_complementarity_inf;
        metrics.source_nlp_dual_feasibility_inf=
            qp->original_nlp_dual_feasibility_inf;
        metrics.qp_wall_ms=qp->wall_ms;
        metrics.predicted_cost_change=qp->predicted_original_objective_change;
        if(overdue()) {
            result->deadline_exhausted=true;
            record_iteration();break;
        }

        // Try the full SQP direction first. Filter globalization below
        // evaluates this same nonlinear problem at each internal trial.
        MpcModelControl_t commands[kStages]{};
        double physical_direction[kStages][2]{};
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
            physical_direction[k][0]=double(v.steering_target_rad)-u.steering_target_rad;
            physical_direction[k][1]=double(v.throttle_command)-u.throttle_command;
        }
        const auto physical_row_index=[](const SparseNonlinearRow &row) {
            if(row.type==NonlinearRowType::BodyCorner)return 8*row.stage+row.index;
            if(row.type==NonlinearRowType::PhysicalCommandRate)return 8*kStages+4*row.stage+row.index;
            if(row.type==NonlinearRowType::PhysicalCommandBox)return 12*kStages+4*row.stage+row.index;
            return -1;
        };
        SourceConstraintCurvature path_model{};
        {
            metrics.source_constraint_path_model_valid=source_constraint_path_curvature(
                *current->nominal,*current->evaluation,trajectory,trajectory_count,
                lap_length,dt,configuration,physical_direction,&path_model);
            metrics.source_constraint_path_first_invalid_stage=path_model.first_invalid_stage;
            if(path_model.valid)
                for(size_t r=0;r<program->rows.size();++r) {
                    const auto &row=program->rows[r];
                    const int index=physical_row_index(row);
                    if(index<0)continue;
                    const double negative=std::min(0.,row.value);
                    const double first=path_model.first[index];
                    metrics.physical_violation_directional_first+=negative*first;
                    metrics.physical_violation_directional_second+=negative*path_model.second[index]+
                        ((row.value<0. || (row.value==0. && first<0.))?first*first:0.);
                    metrics.one_sided_constraint_direction_disagreement_inf=std::max(
                        metrics.one_sided_constraint_direction_disagreement_inf,
                        std::abs(first-(qp->predicted_constraint_margins[r]-row.value)));
                }
        }
        // Original nonlinear KKT conditions determine convergence. Primal
        // step size is not evidence either for or against stationarity.
        constexpr double kStationarityTolerance=1.e-2;
        constexpr double kComplementarityTolerance=1.e-4;
        const bool nonlinear_kkt_converged=current->valid && qp->original_duals_trustworthy &&
                qp->original_nlp_stationarity_inf<=
                    kStationarityTolerance &&
                qp->original_nlp_inequality_residual_inf<=
                    options.allowable_corridor_violation_m+1.e-7 &&
                qp->original_nlp_complementarity_inf<=
                    kComplementarityTolerance &&
                qp->original_nlp_dual_feasibility_inf<=1.e-8 &&
                qp->converged;
        if(nonlinear_kkt_converged || !changed) {
            result->source_nlp_kkt_converged=nonlinear_kkt_converged;
            if(!result->source_nlp_kkt_converged)
                result->full_step_sqp_stalled=true;
            record_iteration();
            if(result->source_nlp_kkt_converged) {result->termination_reason="original_nlp_kkt";break;}
            result->numerical_failure=true;
            result->termination_reason="rounded_zero_step_without_nlp_kkt";
            break;
        }
        if(!qp->converged && !qp->normal_only_step) {
            result->qp_failed=true;result->numerical_failure=true;
            result->termination_reason="uncertified_tangential_qp";
            metrics.qp_failed=true;metrics.invalid_qp_duals_discarded=true;
            record_iteration();break;
        }
        const auto rollout_begin=std::chrono::steady_clock::now();
        ++result->source_trial_evaluations;
        ++result->exact_source_evaluations;
        const bool evaluated=exact_source_rollout(
            initial,initial_progress,trajectory,trajectory_count,
            lap_length,dt,configuration,commands,options,*proposed,
            current->nominal.get());
        metrics.source_rollout_wall_ms=wall_ms(rollout_begin);
        metrics.full_step_evaluated=evaluated;
        metrics.nonlinear_evaluation_failed=!evaluated;
        if(trace) {
            trace->trial_evaluated=evaluated;
            if(evaluated)trace->trial=*proposed->decision;
            else std::copy_n(commands,kStages,trace->trial.controls);
        }
        if(evaluated) {
            metrics.physical_violation_trial=proposed->hard_violation;
            metrics.cost_trial=proposed->cost;
            metrics.violation_trial=proposed->total_violation;
            metrics.actual_cost_change=proposed->cost-current->cost;
        }
        const double source_directional_slope=path_model.valid?
            (current->valid?path_model.original_objective_first:
             metrics.physical_violation_directional_first):0.;
        bool accepted=evaluated && accept_step(*proposed,*current,options,source_directional_slope) &&
            (!qp->normal_only_step || proposed->hard_violation<current->hard_violation);
        metrics.full_step_accepted=accepted;
        if(evaluated) {
            // Audit the local QP against exact physical rollout. These
            // diagnostics never restrict the original legal control box.
            double predicted_hard=0.;
            double worst_error=0.;
            if(evaluated) {
                for(size_t row=0;row<program->rows.size();++row) {
                    const auto &r=program->rows[row];
                    double actual=INFINITY;
                    if(r.type==NonlinearRowType::BodyCorner) {
                        const auto &corners=
                            proposed->evaluation->corridor_corners[r.stage];
                        actual=r.index<4?corners.left[r.index]:
                            corners.right[r.index-4];
                    } else if(r.type==NonlinearRowType::PhysicalCommandRate)
                        actual=proposed->evaluation->command_rate[r.stage][r.index];
                    else if(r.type==NonlinearRowType::PhysicalCommandBox)
                        actual=proposed->evaluation->command_box[r.stage][r.index];
                    else continue;
                    const double prediction=
                        qp->predicted_constraint_margins[row];
                    const double deficit=std::min(0.,prediction);
                    predicted_hard+=.5*deficit*deficit;
                    if(r.type==NonlinearRowType::BodyCorner)
                        worst_error=std::max(worst_error,
                                             std::abs(actual-prediction));
                    if(r.type==NonlinearRowType::BodyCorner && path_model.valid) {
                        const int index=physical_row_index(r);
                        const double quadratic=r.value+path_model.first[index]+.5*path_model.second[index];
                        metrics.model_maximum_quadratic_footprint_error_m=std::max(
                            metrics.model_maximum_quadratic_footprint_error_m,std::abs(actual-quadratic));
                    }
                }
            }
            metrics.model_predicted_hard_violation=predicted_hard;
            metrics.model_maximum_hard_constraint_error=worst_error;
            const double predicted=current->valid?
                -qp->predicted_original_objective_change:
                current->hard_violation-predicted_hard;
            const double actual=evaluated?(current->valid?
                current->cost-proposed->cost:
                current->hard_violation-proposed->hard_violation):
                -INFINITY;
            const double ratio=predicted>1.e-12?
                actual/predicted:-INFINITY;
            metrics.model_actual_over_predicted_reduction=ratio;

        }
        if(accepted && qp->normal_only_step) {
            metrics.normal_step_accepted=true;
            if(trace) {
                trace->normal_evaluated=true;trace->normal_accepted=true;
                trace->normal=*proposed->decision;
            }
        }
        const double correction_budget_us=1200.+1.3*1000.*qp->wall_ms;
        if(!accepted && evaluated &&
           (qp->original_duals_trustworthy ||
            (options.source_infeasible_second_order_correction &&
             !current->valid)) &&
           (options.absolute_deadline_monotonic_us<=0. ||
            options.absolute_deadline_monotonic_us-mpc_rti_monotonic_time_us()>correction_budget_us)) {
            // One Maratos correction of this SQP step. The source defect is
            // c(U+d)-[c(U)+J(U)d], not another controller direction or alpha
            // candidate grid. Reuse the same original objective and Jacobian.
            std::vector<double> defects(program->rows.size(),0.);
            for(size_t r=0;r<program->rows.size();++r) {
                const auto &row=program->rows[r];
                if(row.type==NonlinearRowType::BodyCorner) {
                    const auto &c=proposed->evaluation->corridor_corners[row.stage];
                    const double actual=row.index<4?c.left[row.index]:c.right[row.index-4];
                    defects[r]=actual-qp->predicted_constraint_margins[r];
                } else if(row.type==NonlinearRowType::PhysicalCommandRate)
                    defects[r]=proposed->evaluation->command_rate[row.stage][row.index]-
                        qp->predicted_constraint_margins[r];
            }
            auto correction_options=qp_opts;
            correction_options.constraint_margin_corrections=&defects;
            auto correction_qp=std::make_unique<SourceConstrainedQpResult>();
            metrics.second_order_correction_attempted=true;
            if(source_exact_constrained_qp_direction(*program,
                derivative->original_source_cost_control,current->decision->controls,
                correction_options,correction_qp.get()) && correction_qp->converged) {
                MpcModelControl_t corrected_commands[kStages]{};
                double primary_squared=0.,correction_squared=0.;
                for(int k=0;k<kStages;++k) {
                    const auto &u=current->decision->controls[k];
                    corrected_commands[k].steering_target_rad=static_cast<float>(std::clamp(
                        double(u.steering_target_rad)+correction_qp->direction[k][0],
                        -double(configuration.model.max_steering_rad),
                        double(configuration.model.max_steering_rad)));
                    corrected_commands[k].throttle_command=static_cast<float>(std::clamp(
                        double(u.throttle_command)+correction_qp->direction[k][1],
                        double(MPC_THROTTLE_MIN_DRIVE),double(MPC_THROTTLE_MAX)));
                    for(int j=0;j<2;++j) {
                        const double scale=j==0?.15:.25;
                        primary_squared+=std::pow(qp->direction[k][j]/scale,2);
                        correction_squared+=std::pow((correction_qp->direction[k][j]-
                            qp->direction[k][j])/scale,2);
                    }
                }
                // A SOC must remain second order in the primary step.
                if(correction_squared<=primary_squared*primary_squared && !overdue()) {
                    auto corrected=std::make_unique<Snapshot>();
                    ++result->source_trial_evaluations;
                    ++result->exact_source_evaluations;
                    const auto start=std::chrono::steady_clock::now();
                    const bool valid=exact_source_rollout(initial,initial_progress,trajectory,
                        trajectory_count,lap_length,dt,configuration,corrected_commands,options,
                        *corrected,current->nominal.get());
                    metrics.source_rollout_wall_ms+=wall_ms(start);
                    if(trace) {
                        trace->correction_evaluated=valid;
                        if(valid)trace->correction=*corrected->decision;
                    }
                    if(valid) {
                        metrics.second_order_correction_cost=corrected->cost;
                        metrics.second_order_correction_clearance=corrected->corridor;
                    }
                    if(valid && accept_step(*corrected,*current,options,source_directional_slope)) {
                        metrics.second_order_correction_accepted=true;
                        if(trace)trace->correction_accepted=true;
                        std::swap(proposed,corrected);
                        qp=std::move(correction_qp);
                        accepted=true;
                    }
                }
            }
        }
        if(options.source_trial_relinearized_correction &&
           !accepted && evaluated && !overdue()) {
            // Mathematically distinct from shifting stale margins at U:
            // compute source F, constraint g, and exact source-AD J at
            // U+d itself, then solve ONE corrective local SQP QP there.
            // The physical correction is integrated as U+d+c and checked
            // against the ORIGINAL source constraints before acceptance.
            auto candidate_gradient=std::make_unique<ReducedSourceGradient>();
            auto candidate_program=std::make_unique<SparseNonlinearProgram>();
            const ConstraintMeritWeights weights{
                0.,0.,options.minimum_source_clearance_m,
                options.terminal_speed_viability_weight,
                options.terminal_speed_allowance_mps};
            const bool candidate_derivatives=
                source_exact_compact_qp_linearization(
                    initial,initial_progress,trajectory,trajectory_count,
                    lap_length,dt,configuration,*proposed->decision,
                    *proposed->evaluation,*proposed->nominal,weights,
                    candidate_gradient.get(),candidate_program.get());
            if(candidate_derivatives && !overdue()) {
                auto candidate_options=qp_opts;
                candidate_options.lagrangian_memory=nullptr;
                candidate_options.inequality_multipliers=nullptr;
                candidate_options.source_nominal=proposed->nominal.get();
                candidate_options.constraint_margin_corrections=nullptr;
                candidate_options.capture_curvature_model=false;
                auto correction=std::make_unique<SourceConstrainedQpResult>();
                metrics.second_order_correction_attempted=true;
                if(source_exact_constrained_qp_direction(
                    *candidate_program,candidate_gradient->original_source_cost_control,
                    proposed->decision->controls,candidate_options,
                    correction.get()) &&
                   (correction->converged || correction->normal_only_step)) {
                    MpcModelControl_t commands2[kStages]{};
                    double primary_norm2=0.,correction_norm2=0.;
                    for(int k=0;k<kStages;++k) {
                        const auto &u=proposed->decision->controls[k];
                        commands2[k].steering_target_rad=static_cast<float>(
                            std::clamp(double(u.steering_target_rad)+
                                          correction->direction[k][0],
                                       -double(configuration.model.max_steering_rad),
                                       double(configuration.model.max_steering_rad)));
                        commands2[k].throttle_command=static_cast<float>(
                            std::clamp(double(u.throttle_command)+
                                          correction->direction[k][1],
                                       double(MPC_THROTTLE_MIN_DRIVE),
                                       double(MPC_THROTTLE_MAX)));
                        for(int j=0;j<2;++j) {
                            const double scale=j==0?.15:.25;
                            primary_norm2+=std::pow(qp->direction[k][j]/scale,2);
                            correction_norm2+=std::pow(
                                correction->direction[k][j]/scale,2);
                        }
                    }
                    // Source curvature correction must be small compared
                    // with the first-order step, never an arbitrarily new
                    // warm start or another controller.
                    if(correction_norm2<=
                            std::max(1.e-7,2.*primary_norm2*primary_norm2) &&
                       !overdue()) {
                        auto corrected=std::make_unique<Snapshot>();
                        ++result->source_trial_evaluations;
                        ++result->exact_source_evaluations;
                        const auto start=std::chrono::steady_clock::now();
                        const bool valid=exact_source_rollout(
                            initial,initial_progress,trajectory,trajectory_count,
                            lap_length,dt,configuration,commands2,options,
                            *corrected,current->nominal.get());
                        metrics.source_rollout_wall_ms+=wall_ms(start);
                        if(trace) {
                            trace->correction_evaluated=valid;
                            if(valid)trace->correction=*corrected->decision;
                        }
                        if(valid) {
                            metrics.second_order_correction_cost=corrected->cost;
                            metrics.second_order_correction_clearance=
                                corrected->corridor;
                        }
                        if(valid && accept_step(*corrected,*current,options,source_directional_slope)) {
                            metrics.second_order_correction_accepted=true;
                            if(trace)trace->correction_accepted=true;
                            std::swap(proposed,corrected);
                            // The correction QP was linearized at the
                            // previously rejected U+d, NOT at current U.
                            // Neither set of QP duals may be used to claim
                            // original-NLP KKT at the new accepted primal.
                            qp->original_duals_trustworthy=false;
                            accepted=true;
                        }
                    }
                }
            }
        }
        if(!accepted && evaluated &&
           qp->converged) {
            // Adaptive filter backtracking of this one SQP direction. The
            // scalar interpolant is V while infeasible and J while feasible;
            // the filter always checks both original nonlinear quantities.
            // At a source/reference tie, the actual one-sided derivative
            // can differ from the nominal branch's J*d. Use the complete
            // condensed path derivative for the violation interpolant.
            // A genuinely discontinuous source direction has no derivative;
            // halve then evaluate exact physics rather than invent a slope.
            const double slope=source_directional_slope;
            const double origin=current->valid?current->cost:current->hard_violation;
            double previous_alpha=1.;
            double previous_value=current->valid?proposed->cost:proposed->hard_violation;
            for(int search=0;search<20 && !accepted && !overdue();++search) {
                const double denominator=2.*(previous_value-origin-previous_alpha*slope);
                const double interpolated=slope<0. && denominator>0.?
                    -slope*previous_alpha*previous_alpha/denominator:.5*previous_alpha;
                const double alpha=std::clamp(interpolated,.1*previous_alpha,.5*previous_alpha);
                MpcModelControl_t candidate[kStages]{};
                bool moved=false;
                for(int k=0;k<kStages;++k) {
                    const auto &u=current->decision->controls[k];
                    candidate[k].steering_target_rad=static_cast<float>(std::clamp(
                        double(u.steering_target_rad)+alpha*qp->direction[k][0],
                        -double(configuration.model.max_steering_rad),double(configuration.model.max_steering_rad)));
                    candidate[k].throttle_command=static_cast<float>(std::clamp(
                        double(u.throttle_command)+alpha*qp->direction[k][1],
                        double(MPC_THROTTLE_MIN_DRIVE),double(MPC_THROTTLE_MAX)));
                    moved|=candidate[k].steering_target_rad!=u.steering_target_rad ||
                        candidate[k].throttle_command!=u.throttle_command;
                }
                if(!moved)break;
                ++metrics.safeguarded_evaluations;++result->source_trial_evaluations;
                ++result->exact_source_evaluations;
                const auto start=std::chrono::steady_clock::now();
                const bool valid=exact_source_rollout(initial,initial_progress,trajectory,trajectory_count,
                    lap_length,dt,configuration,candidate,options,*proposed,current->nominal.get());
                metrics.source_rollout_wall_ms+=wall_ms(start);
                if(trace) {
                    trace->safeguarded_trials.push_back(*proposed->decision);
                    trace->safeguarded_step_lengths.push_back(alpha);
                    trace->safeguarded_valid.push_back(valid);
                }
                previous_alpha=alpha;
                previous_value=valid?(current->valid?proposed->cost:proposed->hard_violation):INFINITY;
                accepted=valid && accept_step(*proposed,*current,options,slope,alpha);
                if(accepted) {
                    metrics.safeguarded_step_length=alpha;
                    if(trace)trace->safeguarded_accepted=true;
                }
            }
        }
        if(!accepted &&
           qp->normal_step_computed && !current->valid) {
            SourceConstraintCurvature normal_model{};
            double normal_slope=0.;
            if(source_constraint_path_curvature(*current->nominal,*current->evaluation,
                trajectory,trajectory_count,lap_length,dt,configuration,
                qp->normal_direction,&normal_model))
                for(const auto &row:program->rows) {
                    const int index=physical_row_index(row);
                    if(index>=0)normal_slope+=std::min(0.,row.value)*normal_model.first[index];
                }
            double alpha=1.;
            for(int search=0;search<20 && !accepted && !overdue();++search) {
                MpcModelControl_t normal[kStages]{};
                for(int k=0;k<kStages;++k) {
                    const auto &u=current->decision->controls[k];
                    normal[k].steering_target_rad=static_cast<float>(std::clamp(
                        double(u.steering_target_rad)+alpha*qp->normal_direction[k][0],
                        -double(configuration.model.max_steering_rad),double(configuration.model.max_steering_rad)));
                    normal[k].throttle_command=static_cast<float>(std::clamp(
                        double(u.throttle_command)+alpha*qp->normal_direction[k][1],
                        double(MPC_THROTTLE_MIN_DRIVE),double(MPC_THROTTLE_MAX)));
                }
                ++result->source_trial_evaluations;++result->exact_source_evaluations;
                const bool valid=exact_source_rollout(initial,initial_progress,trajectory,trajectory_count,
                    lap_length,dt,configuration,normal,options,*proposed,current->nominal.get());
                if(trace) {
                    trace->normal_evaluated=valid;
                    if(valid)trace->normal=*proposed->decision;
                }
                accepted=valid && accept_step(*proposed,*current,options,normal_slope,alpha);
                if(accepted) {
                    metrics.normal_step_accepted=true;metrics.safeguarded_step_length=alpha;
                    qp->original_duals_trustworthy=false;metrics.original_duals_trustworthy=false;
                    if(trace)trace->normal_accepted=true;
                }
                alpha*=.5;
            }
        }
        if(accepted) {
            double nominal_coordinate_scale=1.;
            for(int k=0;k<kStages;++k) {
                const auto &u=current->decision->controls[k];
                const auto &v=proposed->decision->controls[k];
                nominal_coordinate_scale=std::max({nominal_coordinate_scale,
                    std::abs(double(u.steering_target_rad))/.15,
                    std::abs(double(u.throttle_command))/.25});
                metrics.accepted_step_inf_scaled=std::max({metrics.accepted_step_inf_scaled,
                    std::abs(double(v.steering_target_rad)-u.steering_target_rad)/.15,
                    std::abs(double(v.throttle_command)-u.throttle_command)/.25});
            }
            metrics.physical_violation_after=proposed->hard_violation;
            metrics.maximum_hard_violation_after=std::max({0.,-proposed->corridor,-proposed->rate,-proposed->box});
            metrics.accepted=true;
            metrics.actual_cost_change=proposed->cost-current->cost;
            metrics.predicted_cost_change=qp->predicted_original_objective_change;
            metrics.cost_after=proposed->cost;
            metrics.violation_after=proposed->total_violation;
            metrics.corridor_slack_after=proposed->corridor;
            // A representable change in one near-zero float command can
            // survive while every physically significant coordinate rounds
            // away. Do not spend thousands of iterations claiming progress
            // from roundoff-sized changes in the original scalar score.
            const double value_before=current->valid?current->cost:current->hard_violation;
            const double value_after=current->valid?proposed->cost:proposed->hard_violation;
            const double accumulation_roundoff=64.*kStages*std::numeric_limits<double>::epsilon()*
                std::max({1.,std::abs(value_before),std::abs(value_after)});
            const bool precision_limited=
                metrics.accepted_step_inf_scaled<=std::numeric_limits<float>::epsilon()*nominal_coordinate_scale &&
                std::abs(value_after-value_before)<=accumulation_roundoff;
            source_precision_stagnation_iterations=precision_limited?
                source_precision_stagnation_iterations+1:0;
            metrics.original_duals_trustworthy=qp->original_duals_trustworthy;
            if(options.record_primal_trace && qp->original_duals_trustworthy)
                result->final_physical_multipliers=physical_duals();
            else result->final_physical_multipliers.clear();
            // Source-contact, tire, suspension, peak-active and reference
            // switches change the *function being differentiated* even when
            // every original hard constraint retains the same row identity.
            // Reusing a cross-mode BFGS secant corrupts the next SQP Hessian.
            // Compare the two ALREADY integrated original-source trajectories;
            // do not create another candidate or change the plant.
            SourceCurvatureBranches accepted_branches{};
            const bool check_curvature_switches=
                options.strict_accepted_source_branch_curvature_guard ||
                options.source_discontinuous_secant_guard;
            const bool valid_branch_comparison=
                !check_curvature_switches ||
                compare_source_curvature_branches(
                    *current->nominal,*current->evaluation,
                    *proposed->nominal,*proposed->evaluation,trajectory,trajectory_count,lap_length,dt,
                    &accepted_branches);
            const bool same_source_curvature_regime=
                !options.strict_accepted_source_branch_curvature_guard ||
                (valid_branch_comparison && accepted_branches.same);
            // These events change the original source *integration schedule*
            // or the force/contact mode. In contrast, WheelFrictionCurve
            // spline knots are value/first-derivative continuous and should
            // not automatically suppress a BFGS secant.
            const auto &physics=accepted_branches.physical;
            const bool discontinuous_source_mode_switch=
                options.source_discontinuous_secant_guard &&
                (!valid_branch_comparison ||
                 physics.mismatched_substep_schedule_events>0 ||
                 physics.contact_grounded_changes>0 ||
                 physics.throttle_brake_mode_changes>0 ||
                 physics.sticky_forward_changes>0 ||
                 physics.sticky_side_changes>0);
            metrics.accepted_source_branch_curvature_reset=
                !same_source_curvature_regime;
            if(check_curvature_switches && valid_branch_comparison) {
                const auto &p=accepted_branches.physical;
                metrics.accepted_source_physical_branch_changes=
                    p.mismatched_substep_schedule_events+
                    p.contact_grounded_changes+
                    p.sticky_forward_changes+p.sticky_side_changes+
                    p.longitudinal_spline_region_changes+
                    p.lateral_spline_region_changes+
                    p.tire_load_filter_changes+
                    p.combined_slip_saturation_changes+
                    p.suspension_jounce_region_changes+
                    p.slip_denominator_region_changes+
                    p.steering_target_reached_changes+
                    p.throttle_brake_mode_changes;
                metrics.accepted_source_peak_branch_changes=
                    accepted_branches.peak_changes;
                metrics.accepted_source_reference_branch_changes=
                    accepted_branches.reference_changes;
            }
            if(!same_source_curvature_regime) {
                *lagrangian_memory={};
                ++result->accepted_source_branch_curvature_resets;
            } else {
                // Within one source regime, preserve the same original
                // Lagrangian BFGS update. During infeasible restoration
                // retain the original objective curvature with lambda=0.
                lagrangian_memory->anchor_valid=true;
                const bool use_constraint_curvature=
                    current->valid && qp->original_duals_trustworthy;
                lagrangian_memory->frozen_curvature_multipliers=
                    use_constraint_curvature?
                    qp->curvature_multipliers:
                    std::vector<double>(program->rows.size(),0.);
                lagrangian_memory->frozen_slip_multipliers=
                    use_constraint_curvature?
                    qp->slip_curvature_multipliers:
                    std::vector<SourceSlipCurvatureMultiplier>();
                for(int j=0;j<2*kStages;++j) {
                    const auto &u=current->decision->controls[j/2];
                    lagrangian_memory->anchor_controls[j]=(j%2?
                        double(u.throttle_command):
                        double(u.steering_target_rad))/(j%2?.25:.15);
                    lagrangian_memory->anchor_gradient[j]=
                        use_constraint_curvature?
                        qp->lagrangian_gradient_at_qp_duals[j]:
                        derivative->original_source_cost_control[j/2][j%2]*
                            (j%2?.25:.15);
                }
            }
            rejected_iterations=0;
            if(metrics.safeguarded_step_length>=.5)
                qp_regularization=std::max(.001,.3*qp_regularization);
            if(discontinuous_source_mode_switch) {
                // Keep the valid previous Hessian/regularization, but never
                // form y = grad L(new branch)-grad L(old branch) as if it
                // measured smooth curvature. Re-anchor only after the next
                // accepted step that stays in a compatible physical mode.
                lagrangian_memory->anchor_valid=false;
                metrics.accepted_source_discontinuous_secant_skipped=true;
                ++result->accepted_source_discontinuous_secant_skips;
            }
            // Standard SQP dual update: only hard original constraints have
            // carried multipliers. Preferred extra clearance is diagnostic
            // and has no original-source multiplier or curvature term.
            if(qp->original_duals_trustworthy &&
               same_source_curvature_regime)
                inequality_multipliers=std::move(qp->inequality_multipliers);
            else {
                // An infeasible/nonconverged QP may still give an improving
                // exact nonlinear primal, but its multipliers cannot be
                // treated as source KKT estimates or BFGS curvature memory.
                inequality_multipliers.clear();
                metrics.invalid_qp_duals_discarded=true;
            }
            // The QP duals were computed at the PRE-step source primal.
            // Once a different command sequence is accepted, they cannot
            // certify KKT stationarity of the new terminal iterate. A future
            // SQP linearization must produce fresh multipliers there.
            result->final_physical_multipliers.clear();
            std::swap(current,proposed);
            have_current_exact_source_linearization=false;
            have_current_reference_hessian=false;
            current_reference_hessian_evaluated=false;
            result->full_step_sqp_stalled=false;
            ++result->accepted_iterations;
            if(source_precision_stagnation_iterations>=8) {
                result->numerical_failure=true;
                result->full_step_sqp_stalled=true;
                result->termination_reason="source_precision_stagnation_without_nlp_kkt";
            }
        } else {
            metrics.full_step_rejected=true;
            result->full_step_sqp_stalled=true;
            // No multiplier from a rejected or relaxed QP enters source
            // curvature memory. A finite rejection budget bounds stagnation.
            ++rejected_iterations;
            if(qp_regularization>=kMaxRegularization || rejected_iterations>=12) {
                result->numerical_failure=true;
                result->termination_reason=qp_regularization>=kMaxRegularization?
                    "regularization_saturated_without_progress":"filter_rejected_without_progress";
            }
            qp_regularization=std::min(kMaxRegularization,10.*qp_regularization);
        }
        record_iteration();
        if(result->numerical_failure)break;
    }
    if(result->deadline_exhausted)result->termination_reason="deadline_exhausted";
    else if(result->linearization_failed)result->termination_reason="source_linearization_failed";
    result->solver_status=overdue()?MPC_SOURCE_DEADLINE_EXCEEDED:
        result->linearization_failed || result->qp_failed || result->numerical_failure?MPC_SOURCE_KKT_NOT_CONVERGED:
        result->full_step_sqp_stalled?MPC_SOURCE_NONLINEAR_STEP_REJECTED:
        result->source_nlp_kkt_converged?MPC_SOURCE_FEASIBLE_SOLUTION_AVAILABLE:
        current->valid?MPC_SOURCE_KKT_NOT_CONVERGED:MPC_SOURCE_PRIMAL_INFEASIBLE;
    result->optimized=*current->decision;
    result->final_source_feasible=current->valid;
    result->final_source_cost=current->cost;
    result->final_terminal_speed_margin_mps=
        current->terminal_speed_margin_mps;
    result->final_terminal_speed_violation=
        current->terminal_speed_violation;
    result->final_exact_constraint_violation=current->total_violation;
    result->final_minimum_corridor_slack=current->corridor;
    result->final_minimum_rate_margin=current->rate;
    return true;
}
} // namespace sdu_apex::source_nlp
