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
    // source footprint remains inside the wall. The reserve participates
    // in the merit function and constrained SQP direction.
    out.valid=
        out.corridor>=0. &&
        out.rate>=-options.allowable_command_rate_violation &&
        out.box>=-1.e-7;
    return true;
}


bool accept_step(const Snapshot &next,const Snapshot &current,
                 const OfflineOptions &options)
{
    if(next.valid!=current.valid)return next.valid;
    // This is one augmented nonlinear racing objective: original exact
    // source cost plus the explicitly SOFT extra-margin penalty used by
    // the constrained QP. Full nonlinearity is independently integrated.
    // A better source-feasible solution may trade a small racing cost
    // increase for significantly better 3cm reserve, never for a wall hit.
    const double current_merit=
        current.cost+options.source_qp_soft_physical_weight*
            current.total_violation+
        options.terminal_speed_viability_weight*
            current.terminal_speed_violation;
    const double next_merit=
        next.cost+options.source_qp_soft_physical_weight*
            next.total_violation+
        options.terminal_speed_viability_weight*
            next.terminal_speed_violation;
    if(next.valid)
        return next_merit<current_merit-std::max(options.minimum_source_cost_reduction,
            32.*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(current_merit)));
    if(options.elastic_sqp_globalization)
        return next_merit+1.e6*next.hard_violation_l1<
            current_merit+1.e6*current.hard_violation_l1-options.minimum_source_cost_reduction;
    return next.hard_violation<current.hard_violation-
        std::max(options.minimum_violation_reduction,
            32.*std::numeric_limits<double>::epsilon()*current.hard_violation);
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
    std::vector<double> inequality_multipliers;
    std::vector<SourceNonlinearCurvatureSecant> source_secants;
    auto lagrangian_memory=std::make_unique<SourceLagrangianMemory>();
    auto derivative=std::make_unique<ReducedSourceGradient>();
    auto program=std::make_unique<SparseNonlinearProgram>();
    bool have_current_exact_source_linearization=false;
    std::array<double,4*kStages*kStages> fd_lagrangian_hessian{};
    bool have_current_fd_hessian=false;
    double qp_regularization=options.source_qp_curvature_regularization;
    // Scaled source curvature can exceed 1e8 near tire saturation. A fixed
    // 1e5 ceiling stopped globalization before it could produce a local step.
    constexpr double kMaxRegularization=1.e12;
    constexpr double kRejectedFullStepDampingGrowth=3.;
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
    for(int iteration=0;iteration<options.maximum_iterations;++iteration) {
        if(overdue() ||
           (options.absolute_deadline_monotonic_us>0. &&
            options.minimum_remaining_us_for_new_iteration>0. &&
            options.absolute_deadline_monotonic_us-
                mpc_rti_monotonic_time_us()<
                options.minimum_remaining_us_for_new_iteration)) {
            result->deadline_exhausted=true;
            result->iteration_admission_stopped=!overdue();
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
        std::unique_ptr<OfflinePrimalTrace> trace;
        if(options.record_primal_trace) {
            trace=std::make_unique<OfflinePrimalTrace>();
            trace->iteration=iteration;
            trace->warm_start=*current->decision;
        }

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
                rho,rho,options.minimum_source_clearance_m,
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
                result->linearization_failed=true;break;
            }
            if(std::abs(derivative->exact_source_violation_half_squared-
                        current->total_violation)>1.e-8*
                   std::max(1.,current->total_violation))
                return false;
            have_current_exact_source_linearization=true;
            if(!inequality_multipliers.empty() && inequality_multipliers.size()!=program->rows.size()) {
                // A nonsmooth terminal reference can add/remove one cut.
                // Reset the dual metric instead of rejecting a valid primal
                // or assigning a carried multiplier to a different row.
                inequality_multipliers.clear();
                *lagrangian_memory={};
            }
        }
        metrics.linearization_wall_ms=wall_ms(lin_begin);
        if(overdue()) {
            result->deadline_exhausted=true;
            if(trace)result->primal_trace.push_back(std::move(*trace));
            break;
        }
        auto qp=std::make_unique<SourceConstrainedQpResult>();
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
        qp_opts.source_curvature_secants=
            source_secants.empty()?nullptr:&source_secants;
        qp_opts.lagrangian_memory=lagrangian_memory.get();
        qp_opts.refresh_known_gauss_newton=current->valid;
        qp_opts.absolute_deadline_monotonic_us=options.absolute_deadline_monotonic_us;
        qp_opts.capture_curvature_model=options.finite_difference_lagrangian_hessian;
        // Exact-L1 elastic SQP only regularizes an infeasible local model.
        // It is never a source-feasibility certificate or a publication policy.
        if(!current->valid && options.elastic_sqp_globalization) {
            qp_opts.infeasible_linearization_l1_weight=1.e6;
        }
        metrics.calibrated_source_secants=
            static_cast<int>(source_secants.size());
        if(!source_exact_constrained_qp_direction(
            *program,derivative->original_source_cost_control,
            current->decision->controls,qp_opts,qp.get())) {
            result->qp_failed=true;break;
        }
        if(options.finite_difference_lagrangian_hessian && !options.source_slip_epigraph) {
            constexpr int dimension=2*kStages;
            constexpr double scale[2]={.15,.25};
            if(!have_current_fd_hessian) {
                const auto hessian_begin=std::chrono::steady_clock::now();
                const auto lagrangian_gradient=[&](const ReducedSourceGradient &gradient,
                                                   const SourceConstrainedQpResult &linear) {
                    Eigen::VectorXd value(dimension);
                    for(int j=0;j<dimension;++j)value(j)=gradient.original_source_cost_control[j/2][j%2]*scale[j%2];
                    for(size_t r=0;r<linear.constraint_row_indices.size();++r) {
                        const size_t row=linear.constraint_row_indices[r];
                        if(row>=qp->curvature_multipliers.size())return Eigen::VectorXd();
                        for(int j=0;j<dimension;++j)value(j)-=qp->curvature_multipliers[row]*
                            linear.constraint_jacobian_scaled[r*dimension+j];
                    }
                    return value;
                };
                const Eigen::VectorXd origin=lagrangian_gradient(*derivative,*qp);
                const auto gradient_at=[&](const MpcModelControl_t controls[kStages]) {
                    auto point=std::make_unique<Snapshot>();
                    auto gradient=std::make_unique<ReducedSourceGradient>();
                    auto linear=std::make_unique<SparseNonlinearProgram>();
                    auto audit=std::make_unique<SourceConstrainedQpResult>();
                    const ConstraintMeritWeights weights{0.,0.,options.minimum_source_clearance_m,
                        options.terminal_speed_viability_weight,options.terminal_speed_allowance_mps};
                    if(!exact_source_rollout(initial,initial_progress,trajectory,trajectory_count,
                         lap_length,dt,configuration,controls,options,*point,nullptr,true) ||
                       !source_exact_compact_qp_linearization(initial,initial_progress,trajectory,
                         trajectory_count,lap_length,dt,configuration,*point->decision,*point->evaluation,
                         *point->nominal,weights,gradient.get(),linear.get()))return Eigen::VectorXd();
                    auto audit_options=qp_opts;
                    audit_options.lagrangian_memory=nullptr;
                    audit_options.source_nominal=point->nominal.get();
                    audit_options.linearization_audit_only=true;
                    audit_options.inequality_multipliers=nullptr;
                    if(!source_exact_constrained_qp_direction(*linear,gradient->original_source_cost_control,
                        point->decision->controls,audit_options,audit.get()))return Eigen::VectorXd();
                    return lagrangian_gradient(*gradient,*audit);
                };
                Eigen::MatrixXd hessian(dimension,dimension);
                bool valid=origin.size()==dimension;
                for(int j=0;j<dimension && valid && !overdue();++j) {
                    MpcModelControl_t plus[kStages],minus[kStages];
                    std::copy_n(current->decision->controls,kStages,plus);
                    std::copy_n(current->decision->controls,kStages,minus);
                    const int axis=j%2;
                    const double lower=axis?MPC_THROTTLE_MIN_DRIVE:-configuration.model.max_steering_rad;
                    const double upper=axis?MPC_THROTTLE_MAX:configuration.model.max_steering_rad;
                    const double value=axis?plus[j/2].throttle_command:plus[j/2].steering_target_rad;
                    const float a=static_cast<float>(std::min(upper,value+1.e-3*scale[axis]));
                    const float b=static_cast<float>(std::max(lower,value-1.e-3*scale[axis]));
                    (axis?plus[j/2].throttle_command:plus[j/2].steering_target_rad)=a;
                    (axis?minus[j/2].throttle_command:minus[j/2].steering_target_rad)=b;
                    const Eigen::VectorXd forward=a==value?origin:gradient_at(plus);
                    const Eigen::VectorXd backward=b==value?origin:gradient_at(minus);
                    valid=forward.size()==dimension && backward.size()==dimension && a>b;
                    if(valid)hessian.col(j)=(forward-backward)/((double(a)-double(b))/scale[axis]);
                }
                if(valid && !overdue()) {
                    hessian=(.5*(hessian+hessian.transpose())).eval();
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(hessian);
                    valid=eigen.info()==Eigen::Success;
                    if(valid) {
                        metrics.lagrangian_hessian_min_eigenvalue=eigen.eigenvalues().minCoeff();
                        const double floor=std::max(1.e-3,1.e-9*eigen.eigenvalues().cwiseAbs().maxCoeff());
                        const Eigen::VectorXd values=options.absolute_lagrangian_curvature?
                            Eigen::VectorXd(eigen.eigenvalues().cwiseAbs()):Eigen::VectorXd(eigen.eigenvalues());
                        hessian=eigen.eigenvectors()*values.cwiseMax(floor).asDiagonal()*
                            eigen.eigenvectors().transpose();
                        for(int i=0;i<dimension;++i)
                            for(int j=0;j<dimension;++j)fd_lagrangian_hessian[i*dimension+j]=hessian(i,j);
                        have_current_fd_hessian=true;
                    }
                }
                metrics.lagrangian_hessian_wall_ms=wall_ms(hessian_begin);
            }
            if(have_current_fd_hessian && !overdue()) {
                qp_opts.lagrangian_hessian_scaled=&fd_lagrangian_hessian;
                qp_opts.lagrangian_memory=nullptr;
                if(!source_exact_constrained_qp_direction(*program,derivative->original_source_cost_control,
                    current->decision->controls,qp_opts,qp.get())) {result->qp_failed=true;break;}
            }
        }
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
            if(trace)result->primal_trace.push_back(std::move(*trace));
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
        // Original nonlinear KKT conditions determine convergence. Primal
        // step size is not evidence either for or against stationarity.
        constexpr double kStationarityTolerance=1.e-2;
        constexpr double kComplementarityTolerance=1.e-4;
        const bool nonlinear_kkt_converged=current->valid &&
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
            metrics.iteration_wall_ms=wall_ms(iter_begin);
            result->history.push_back(metrics);
            if(trace)result->primal_trace.push_back(std::move(*trace));
            if(result->source_nlp_kkt_converged)break;
            // A rounded-zero primal correction can still improve the dual
            // estimate. Retain that estimate and tighten the SAME QP; do not
            // mistake command quantization for nonlinear convergence.
            inequality_multipliers=std::move(qp->inequality_multipliers);
            continue;
        }
        const auto rollout_begin=std::chrono::steady_clock::now();
        ++result->source_trial_evaluations;
        const bool evaluated=exact_source_rollout(
            initial,initial_progress,trajectory,trajectory_count,
            lap_length,dt,configuration,commands,options,*proposed,
            current->nominal.get());
        metrics.source_rollout_wall_ms=wall_ms(rollout_begin);
        metrics.nonlinear_evaluation_failed=!evaluated;
        if(trace) {
            trace->trial_evaluated=evaluated;
            if(evaluated)trace->trial=*proposed->decision;
            else std::copy_n(commands,kStages,trace->trial.controls);
        }
        if(evaluated) {
            metrics.cost_trial=proposed->cost;
            metrics.violation_trial=proposed->total_violation;
            metrics.actual_cost_change=proposed->cost-current->cost;
        }
        bool accepted=evaluated && accept_step(*proposed,*current,options);
        const double primary_terminal_violation=proposed->terminal_speed_violation;
        const double correction_budget_us=1200.+1.3*1000.*qp->wall_ms;
        if(!accepted && evaluated && qp->converged &&
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
                    if(valid && accept_step(*corrected,*current,options)) {
                        metrics.second_order_correction_accepted=true;
                        if(trace)trace->correction_accepted=true;
                        std::swap(proposed,corrected);
                        qp=std::move(correction_qp);
                        accepted=true;
                    }
                }
            }
        }
        if(!accepted && evaluated && qp->converged && options.safeguarded_merit_globalization) {
            double penalty=1.e6;
            for(double lambda:qp->inequality_multipliers)penalty=std::max(penalty,1.1*lambda);
            const auto merit=[&](const Snapshot &point) {
                return point.cost+options.source_qp_soft_physical_weight*point.total_violation+
                    options.terminal_speed_viability_weight*point.terminal_speed_violation+
                    penalty*point.hard_violation_l1;
            };
            double slope=qp->source_cost_linear_slope;
            for(size_t r=0;r<program->rows.size();++r) {
                const auto &row=program->rows[r];
                if(row.type!=NonlinearRowType::BodyCorner &&
                   row.type!=NonlinearRowType::PhysicalCommandRate &&
                   row.type!=NonlinearRowType::PhysicalCommandBox)continue;
                const double change=qp->predicted_constraint_margins[r]-row.value;
                if(row.value<0.)slope-=penalty*change;
                else if(row.value==0.)slope+=penalty*std::max(0.,-change);
                if(row.type==NonlinearRowType::BodyCorner)
                    slope+=options.source_qp_soft_physical_weight*
                        std::min(0.,row.value-options.minimum_source_clearance_m)*change;
                else if(row.type==NonlinearRowType::PhysicalCommandRate)
                    slope+=options.source_qp_soft_physical_weight*std::min(0.,row.value)*change;
            }
            const double origin=merit(*current);
            double previous_alpha=1.,previous_merit=merit(*proposed);
            for(int search=0;slope<0. && search<8 && !accepted && !overdue();++search) {
                const double denominator=2.*(previous_merit-origin-previous_alpha*slope);
                const double interpolated=denominator>0.?
                    -slope*previous_alpha*previous_alpha/denominator:.5*previous_alpha;
                const double alpha=std::clamp(interpolated,.1*previous_alpha,.5*previous_alpha);
                if(alpha<1.e-8)break;
                MpcModelControl_t safeguarded_commands[kStages]{};
                bool safeguarded_changed=false;
                for(int k=0;k<kStages;++k) {
                    const auto &u=current->decision->controls[k];
                    safeguarded_commands[k].steering_target_rad=static_cast<float>(std::clamp(
                        double(u.steering_target_rad)+alpha*qp->direction[k][0],
                        -double(configuration.model.max_steering_rad),double(configuration.model.max_steering_rad)));
                    safeguarded_commands[k].throttle_command=static_cast<float>(std::clamp(
                        double(u.throttle_command)+alpha*qp->direction[k][1],
                        double(MPC_THROTTLE_MIN_DRIVE),double(MPC_THROTTLE_MAX)));
                    safeguarded_changed|=safeguarded_commands[k].steering_target_rad!=u.steering_target_rad ||
                        safeguarded_commands[k].throttle_command!=u.throttle_command;
                }
                if(!safeguarded_changed)break;
                ++metrics.safeguarded_evaluations;
                ++result->source_trial_evaluations;
                const auto score_begin=std::chrono::steady_clock::now();
                const bool valid=exact_source_rollout(initial,initial_progress,trajectory,
                    trajectory_count,lap_length,dt,configuration,safeguarded_commands,
                    options,*proposed,current->nominal.get());
                metrics.source_rollout_wall_ms+=wall_ms(score_begin);
                if(trace) {
                    trace->safeguarded_trials.push_back(*proposed->decision);
                    trace->safeguarded_step_lengths.push_back(alpha);
                    trace->safeguarded_valid.push_back(valid);
                }
                previous_alpha=alpha;
                previous_merit=valid?merit(*proposed):INFINITY;
                accepted=valid && (!current->valid || proposed->valid) &&
                    ((proposed->valid && !current->valid) ||
                     (previous_merit<origin-32.*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(origin)) &&
                      previous_merit<=origin+1.e-4*alpha*slope));
                if(accepted) {
                    metrics.safeguarded_step_length=alpha;
                    if(trace)trace->safeguarded_accepted=true;
                }
            }
        }
        if(accepted) {
            metrics.accepted=true;
            metrics.actual_cost_change=proposed->cost-current->cost;
            metrics.predicted_cost_change=qp->predicted_original_objective_change;
            metrics.cost_after=proposed->cost;
            metrics.violation_after=proposed->total_violation;
            metrics.corridor_slack_after=proposed->corridor;
            lagrangian_memory->anchor_valid=true;
            lagrangian_memory->frozen_curvature_multipliers=qp->curvature_multipliers;
            lagrangian_memory->frozen_slip_multipliers=qp->slip_curvature_multipliers;
            for(int j=0;j<2*kStages;++j) {
                const auto &u=current->decision->controls[j/2];
                lagrangian_memory->anchor_controls[j]=(j%2?
                    double(u.throttle_command):double(u.steering_target_rad))/(j%2?.25:.15);
                lagrangian_memory->anchor_gradient[j]=qp->lagrangian_gradient_at_qp_duals[j];
            }
            // Curvature stabilization must decay when the original nonlinear
            // response agrees with the model. Permanently retaining the last
            // rejected-step damping prevents fast local KKT convergence.
            if(metrics.safeguarded_step_length<.75) {
                qp_regularization=std::min(kMaxRegularization,
                    qp_regularization/metrics.safeguarded_step_length);
            } else if(current->valid && proposed->valid) {
                const double predicted=-(qp->predicted_original_objective_change+
                    options.source_qp_soft_physical_weight*(
                        qp->predicted_linearized_violation_after-
                        qp->predicted_linearized_violation_before));
                const double actual=current->cost-proposed->cost+
                    options.source_qp_soft_physical_weight*(
                        current->total_violation-proposed->total_violation);
                if(predicted>1.e-9 && actual>.75*predicted)
                    qp_regularization=std::max(.001,qp_regularization*.3);
            } else if(current->hard_violation>1.e-12 &&
                      proposed->hard_violation<.5*current->hard_violation)
                qp_regularization=std::max(.001,qp_regularization*.5);
            // Standard SQP dual update: only hard original constraints have
            // carried multipliers. Preferred clearance is a finite elastic
            // objective term and must not accumulate an AL multiplier that
            // eventually turns its reserve into an unintended hard wall.
            inequality_multipliers=std::move(qp->inequality_multipliers);
            std::swap(current,proposed);
            source_secants.clear();
            have_current_exact_source_linearization=false;
            have_current_fd_hessian=false;
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
                         qp->predicted_linearized_violation_before)+
                        options.terminal_speed_viability_weight*
                        (qp->predicted_terminal_speed_violation_after-
                         qp->predicted_terminal_speed_violation_before);
                    const double actual_merit_delta=
                        (metrics.cost_trial-current->cost)+
                        options.source_qp_soft_physical_weight*
                        (metrics.violation_trial-
                         current->total_violation)+
                        options.terminal_speed_viability_weight*
                        (primary_terminal_violation-
                         current->terminal_speed_violation);
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
        if(trace) {
            trace->accepted=metrics.accepted;
            result->primal_trace.push_back(std::move(*trace));
        }
        if(!metrics.accepted &&
           (qp_regularization<=metrics.qp_regularization_used ||
            iteration+1>=options.maximum_iterations))
            break;
    }
    result->solver_status=overdue()?MPC_SOURCE_DEADLINE_EXCEEDED:
        result->linearization_failed || result->qp_failed?MPC_SOURCE_KKT_NOT_CONVERGED:
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
