#include "mpc_rti_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>

/*
 * Experimental nonlinear-source single shooting, completely isolated from
 * production cycle output selection and unrelated Riccati fallbacks.
 *
 * Source-integrated physical state and native source A/B are recomputed on
 * every accepted iterate. Exact cost and original hard corridor come from
 * the authoritative mpc_rti_validate_candidate_plan, never from the QP's
 * linear forecast. All iterates optimize ONE immutable cost schedule.
 */

// Progress is an explicit state in the Riccati QP. The original source
// feasibility merit also depends on progress through left/right track widths.
// This partial derivative holds the local source-frame pose fixed; the
// independent e_y/e_psi derivatives and QP A/B carry changes in that pose.
double mpc_rti_source_corridor_progress_derivative(
    const MpcModelState_t &plant, double station, int stage,
    const MpcTrajectorySample_t *trajectory, size_t count, double lap,
    const MpcRtiConfiguration_t *model)
{
    if(!trajectory || count<3 || !model || !std::isfinite(station) ||
       !std::isfinite(lap) || lap<=0. || stage<1 ||
       stage>PREDICTION_HORIZON)return NAN;
    constexpr double ds=.002;
    const auto at_progress=[&](double s) {
        MpcRtiReference_t shifted{};
        if(!mpc_rti_reference_at_progress(trajectory,count,lap,s,model,
                &shifted))return double(INFINITY);
        float left=0.f,right=0.f;
        if(!mpc_rti_compute_corridor_slack(&plant,
                shifted.left_bound,shifted.right_bound,model,stage,
                &left,&right))return double(INFINITY);
        const double vl=std::min(0.,double(left));
        const double vr=std::min(0.,double(right));
        return vl*vl+vr*vr;
    };
    const double before=at_progress(station-ds);
    const double after=at_progress(station+ds);
    if(!std::isfinite(before)||!std::isfinite(after))return NAN;
    return (after-before)/(2.*ds);
}

namespace {
using Clock = std::chrono::steady_clock;
constexpr int N = PREDICTION_HORIZON;

bool controls_finite(const MpcModelControl_t *controls)
{
    for (int k=0;k<N;++k)
        if (!std::isfinite(controls[k].steering_target_rad) ||
            !std::isfinite(controls[k].throttle_command))
            return false;
    return true;
}

bool valid_options(const MpcRtiSourceNonlinearOptions_t &p)
{
    return p.max_iterations>=1 && p.max_iterations<=8 &&
        p.max_exact_trials_per_iteration>=1 &&
        p.max_exact_trials_per_iteration<=16 &&
        std::isfinite(p.initial_step_scale) &&
        p.initial_step_scale>0.f && p.initial_step_scale<=4.f &&
        std::isfinite(p.minimum_step_scale) &&
        p.minimum_step_scale>0.f &&
        p.minimum_step_scale<=p.initial_step_scale &&
        std::isfinite(p.armijo_fraction) &&
        p.armijo_fraction>=0.f && p.armijo_fraction<=.5f &&
        std::isfinite(p.steering_direction_norm_rad) &&
        p.steering_direction_norm_rad>0.f &&
        p.steering_direction_norm_rad<=1.f &&
        std::isfinite(p.throttle_direction_norm) &&
        p.throttle_direction_norm>0.f &&
        p.throttle_direction_norm<=1.f &&
        std::isfinite(p.directional_check_step_scale) &&
        p.directional_check_step_scale>=0.f &&
        p.directional_check_step_scale<=.1f &&
        std::isfinite(p.maximum_wall_time_us) &&
        p.maximum_wall_time_us>=0. &&
        p.maximum_wall_time_us<=1.e8;
}

void build_candidate(
    const MpcModelControl_t current[N],
    const double direction[N][MPC_RTI_NU],
    double scale,
    const MpcRtiConfiguration_t &model,
    MpcModelControl_t out[N])
{
    for (int k=0;k<N;++k) {
        out[k].steering_target_rad=std::clamp(
            float(double(current[k].steering_target_rad) +
                scale*direction[k][0]),
            -model.max_steering_rad,model.max_steering_rad);
        out[k].throttle_command=std::clamp(
            float(double(current[k].throttle_command) +
                scale*direction[k][1]),
            MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
    }
}

double projected_directional_product(
    const MpcModelControl_t current[N],
    const MpcModelControl_t candidate[N],
    const double gradient[N][MPC_RTI_NU],
    double scale)
{
    double dot=0.;
    for(int k=0;k<N;++k) {
        dot+=gradient[k][0] *
            (double(candidate[k].steering_target_rad) -
             current[k].steering_target_rad)/scale;
        dot+=gradient[k][1] *
            (double(candidate[k].throttle_command) -
             current[k].throttle_command)/scale;
    }
    return dot;
}

/* Exact-source Phase-I feasibility objective. An infeasible 750ms nominal
 * must be optimized TOWARD corridor feasibility before optimizing its racing
 * objective. Never call this a valid drive plan merely because the relaxed
 * violation score decreased: ONLY original source validation can authorize
 * its output. This is a mathematical feasibility subproblem, not a separate
 * heuristic throttle/brake/steer fallback.
 *
 * At each physically integrated nominal, geometry margins are evaluated
 * at its actual reference progress. Source A/B linearizes at that SAME
 * physical trajectory; costate propagation differentiates the constraint
 * violation through all 30 source dynamics transitions.
 */
double corridor_violation(
    const MpcRtiNominal_t &plan,
    const MpcRtiReference_t refs[N+1],
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    const MpcRtiConfiguration_t &model,
    double dstate[N+1][RICCATI_MAX_NX])
{
    if(dstate)std::memset(dstate,0,
        sizeof(double)*(N+1)*RICCATI_MAX_NX);
    double merit=0.;
    for(int k=1;k<=N;++k) {
        const auto &p=plan.states[k].plant;
        const auto &ref=refs[k];
        const auto stage_cost=[&](const MpcModelState_t &state) {
            float left=0.f,right=0.f;
            if(!mpc_rti_compute_corridor_slack(
                    &state,ref.left_bound,ref.right_bound,&model,k,
                    &left,&right))return double(INFINITY);
            const double vl=std::min(0.,double(left));
            const double vr=std::min(0.,double(right));
            return vl*vl+vr*vr;
        };
        const double stage=stage_cost(p);
        if(!std::isfinite(stage))return INFINITY;
        merit+=stage;
        if(stage>0. && dstate) {
            /* Actual min/max footprint support is piecewise smooth. Use
             * precise central differences of just its 2D geometry; do NOT
             * rerun or finite-difference source tire dynamics. */
            constexpr float eps=1.e-4f;
            MpcModelState_t plus=p,minus=p;
            plus.e_y+=eps;minus.e_y-=eps;
            dstate[k][MPC_OPT_IDX_EY]=
                (stage_cost(plus)-stage_cost(minus))/(2.*eps);
            plus=p;minus=p;
            plus.e_psi+=eps;minus.e_psi-=eps;
            dstate[k][MPC_OPT_IDX_EPSI]=
                (stage_cost(plus)-stage_cost(minus))/(2.*eps);

            dstate[k][MPC_OPT_IDX_PROGRESS_DEVIATION]=
                mpc_rti_source_corridor_progress_derivative(p,
                    plan.progress[k],k,trajectory,trajectory_count,
                    lap_length,&model);
            if(!std::isfinite(
                    dstate[k][MPC_OPT_IDX_PROGRESS_DEVIATION]))
                return INFINITY;
        }
    }
    return merit;
}

bool source_corridor_phase_one(
    const MpcRtiState_t *initial,
    double progress,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float dt,
    const MpcRtiCycleConfiguration_t &config,
    const MpcRtiObjectiveSchedule_t &objective,
    const MpcRtiSourceNonlinearOptions_t &options,
    const std::chrono::steady_clock::time_point begin,
    MpcModelControl_t controls[N],
    MpcRtiSourceNonlinearResult_t *result,
    MpcRtiNominal_t *certified_nominal)
{
    const auto out_of_time=[&]() {
        return options.maximum_wall_time_us>0. &&
            std::chrono::duration<double,std::micro>(
                Clock::now()-begin).count()>=options.maximum_wall_time_us;
    };
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto problem=std::make_unique<MpcRtiProblem_t>();
    MpcRtiReference_t refs[N+1]{};
    MpcModelControl_t trial[N]{};
    MpcModelControl_t chosen[N]{};
    MpcRtiCycleConfiguration_t exact=config;
    exact.objective_schedule=&objective;
    exact.capture_diagnostic_outputs=0;
    exact.qp_capture=nullptr;
    int failure=-1;

    const auto evaluate=[&](const MpcModelControl_t plan[N],
        double deriv[N+1][RICCATI_MAX_NX]) {
        if(!mpc_rti_build_nominal_same_epoch(
                initial,progress,nullptr,plan,N,
                trajectory,trajectory_count,lap_length,dt,N,
                &config.model,nominal.get(),refs))
            return double(INFINITY);
        return corridor_violation(*nominal,refs,trajectory,
            trajectory_count,lap_length,config.model,deriv);
    };

    double source_gradient[N+1][RICCATI_MAX_NX]{};
    double current_violation=evaluate(controls,source_gradient);
    result->initial_corridor_violation_m2=current_violation;
    result->final_corridor_violation_m2=current_violation;
    if(!std::isfinite(current_violation))return false;
    for(int iteration=0;iteration<std::min(4,options.max_iterations) && current_violation>1.e-12;++iteration) {
        if(out_of_time()){result->budget_exhausted=1;break;}
        ++result->feasibility_iterations_attempted;
        if(!mpc_rti_build_ltv_qp_with_objective_schedule(
                nominal->states,nominal->controls,refs,N,dt,
                &config.model,&objective,problem.get()))
            break;
        double lambda[N+1][RICCATI_MAX_NX]{};
        std::copy_n(source_gradient[N],RICCATI_MAX_NX,lambda[N]);
        double decision_gradient[N][MPC_RTI_NU]{};
        for(int k=N-1;k>=0;--k) {
            const auto &stage=problem->steps[k];
            for(int a=0;a<MPC_RTI_NU;++a)
                for(int i=0;i<RICCATI_MAX_NX;++i)
                    decision_gradient[k][a]+=
                        stage.B[i][a]*lambda[k+1][i];
            for(int i=0;i<RICCATI_MAX_NX;++i) {
                lambda[k][i]=source_gradient[k][i];
                for(int j=0;j<RICCATI_MAX_NX;++j)
                    lambda[k][i]+=stage.A[j][i]*lambda[k+1][j];
            }
        }
        double gradient[N][MPC_RTI_NU]{},norm[MPC_RTI_NU]{};
        for(int k=0;k<N;++k)
            for(int a=0;a<MPC_RTI_NU;++a) {
                gradient[k][a]=decision_gradient[k][a]-
                    (config.model.command_increment_qp_enabled &&
                     k+1<N ? decision_gradient[k+1][a]:0.);
                norm[a]+=gradient[k][a]*gradient[k][a];
            }
        for(auto &n:norm)n=std::sqrt(n);
        if(norm[0]<1.e-9 && norm[1]<1.e-9)break;
        double direction[N][MPC_RTI_NU]{};
        for(int k=0;k<N;++k) {
            direction[k][0]=-options.steering_direction_norm_rad*gradient[k][0]/
                std::max(norm[0],1.e-9);
            direction[k][1]=-options.throttle_direction_norm*gradient[k][1]/
                std::max(norm[1],1.e-9);
        }
        const double start_merit=current_violation;
        double best=start_merit;
        constexpr double bracket[]={
            .1,.2,.4,.8,.05,.025,.0125,.00625};
        for(int trial_index=0;
            trial_index<std::min<int>(std::size(bracket),
                                     options.max_exact_trials_per_iteration);
            ++trial_index) {
            if(out_of_time()){result->budget_exhausted=1;break;}
            const double scale=bracket[trial_index];
            build_candidate(controls,direction,scale,config.model,trial);
            const double value=evaluate(trial,nullptr);
            if(value<best) {
                best=value;
                std::copy_n(trial,N,chosen);
            }
            if(value<=1.e-12)break;
        }
        if(best>=start_merit)break;
        std::copy_n(chosen,N,controls);
        current_violation=best;
        result->final_corridor_violation_m2=best;
        ++result->feasibility_iterations_accepted;
        // Only a full strict source-plant rollout with ORIGINAL footprint
        // constraints certifies a feasible initializer.
        ++result->exact_rollout_evaluations;
        if(mpc_rti_validate_candidate_plan(
                initial,progress,controls,trajectory,trajectory_count,
                lap_length,N,dt,&exact,nominal.get(),&failure)
                ==MPC_RTI_ROLLOUT_OK) {
            result->feasible_seed_recovered=1;
            // This complete strict-source rollout is already the desired
            // R1 linearization. Reuse it instead of integrating again.
            if (certified_nominal) *certified_nominal=*nominal;
            return true;
        }
        current_violation=evaluate(controls,source_gradient);
        if(!std::isfinite(current_violation))break;
    }
    return false;
}
} // namespace

extern "C" int mpc_rti_source_repair_infeasible_nominal(
    const MpcRtiState_t *initial, double progress,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration,
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    const MpcModelControl_t seed[PREDICTION_HORIZON],
    const MpcRtiSourceNonlinearOptions_t *options,
    MpcRtiSourceNonlinearResult_t *result,
    MpcRtiNominal_t *certified_nominal)
{
    if (!result) return 0;
    std::memset(result, 0, sizeof(*result));
    result->initial_failure_stage = -1;
    result->initial_failure_reason = MPC_RTI_ROLLOUT_INVALID_INPUT;
    if (!initial || !trajectory || trajectory_count < 3 || !configuration ||
        !frozen_objective || !frozen_objective->valid ||
        frozen_objective->horizon != N || !seed || !options ||
        !valid_options(*options) || !controls_finite(seed) ||
        !std::isfinite(progress) || !std::isfinite(lap_length) ||
        lap_length <= 0. || !std::isfinite(prediction_dt) ||
        prediction_dt <= 0.f)
        return 0;
    const auto begin = Clock::now();
    MpcRtiCycleConfiguration_t validation = *configuration;
    validation.objective_schedule = frozen_objective;
    validation.qp_capture = nullptr;
    validation.qp_capture_second_pass = nullptr;
    validation.capture_diagnostic_outputs = 0;
    auto plan = std::make_unique<MpcRtiNominal_t>();
    int failed_stage = -1;
    const auto initial_status = mpc_rti_validate_candidate_plan(
        initial, progress, seed, trajectory, trajectory_count, lap_length,
        N, prediction_dt, &validation, plan.get(), &failed_stage);
    result->initial_failure_stage = failed_stage;
    result->initial_failure_reason = initial_status;
    if (initial_status != MPC_RTI_ROLLOUT_CORRIDOR) return 0;
    MpcModelControl_t repaired[N]{};
    std::copy_n(seed, N, repaired);
    if (!source_corridor_phase_one(initial, progress, trajectory,
            trajectory_count, lap_length, prediction_dt, validation,
            *frozen_objective, *options, begin, repaired, result,
            plan.get())) {
        result->wall_time_ms = std::chrono::duration<double,std::milli>(
            Clock::now() - begin).count();
        return 0;
    }
    // Phase-I already certified this exact physical trajectory with the
    // original source/corridor/terminal validator. A second validation
    // would redundantly integrate the same 30 source steps.
    if (!plan->valid || plan->horizon != N) return 0;
    if (certified_nominal) *certified_nominal=*plan;
    std::copy_n(repaired, N, result->controls);
    result->final_valid = 1;
    result->minimum_corridor_clearance_m =
        plan->validation_minimum_corridor_slack_m;
    result->wall_time_ms = std::chrono::duration<double,std::milli>(
        Clock::now() - begin).count();
    return 1;
}

extern "C" int mpc_rti_source_nonlinear_optimize(
    const MpcRtiState_t *initial,
    double progress,
    const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,
    double lap_length,
    float prediction_dt,
    const MpcRtiCycleConfiguration_t *configuration,
    const MpcRtiObjectiveSchedule_t *frozen_objective,
    const MpcModelControl_t seed[PREDICTION_HORIZON],
    const MpcRtiSourceNonlinearOptions_t *options,
    MpcRtiSourceNonlinearResult_t *result,
    MpcRtiNominal_t *certified_nominal)
{
    if (!result) return 0;
    std::memset(result,0,sizeof(*result));
    result->initial_failure_stage=-1;
    result->initial_failure_reason=MPC_RTI_ROLLOUT_INVALID_INPUT;
    if (!initial || !trajectory || trajectory_count<3 ||
        !configuration || !seed || !options ||
        !frozen_objective || !frozen_objective->valid ||
        frozen_objective->horizon!=N ||
        !std::isfinite(progress) || !std::isfinite(lap_length) ||
        lap_length<=0 || !std::isfinite(prediction_dt) ||
        prediction_dt<=0 || !controls_finite(seed) ||
        !valid_options(*options))
        return 0;

    const auto begin=Clock::now();
    const auto out_of_time=[&]() {
        return options->maximum_wall_time_us>0. &&
            std::chrono::duration<double,std::micro>(
                Clock::now()-begin).count()>=options->maximum_wall_time_us;
    };
    MpcRtiCycleConfiguration_t exact_configuration=*configuration;
    exact_configuration.objective_schedule=frozen_objective;
    exact_configuration.capture_diagnostic_outputs=0;
    exact_configuration.qp_capture=nullptr;
    auto exact_plan=std::make_unique<MpcRtiNominal_t>();
    auto best_plan=std::make_unique<MpcRtiNominal_t>();
    MpcModelControl_t current[N]{};
    std::copy_n(seed,N,current);
    int failure=-1;
    const auto validate=[&](const MpcModelControl_t trial[N]) {
        ++result->exact_rollout_evaluations;
        const auto status=mpc_rti_validate_candidate_plan(
            initial,progress,trial,trajectory,trajectory_count,
            lap_length,N,prediction_dt,&exact_configuration,
            exact_plan.get(),&failure);
        if(status==MPC_RTI_ROLLOUT_CORRIDOR)
            ++result->trial_corridor_rejections;
        else if(status!=MPC_RTI_ROLLOUT_OK)
            ++result->trial_model_rejections;
        return status;
    };

    const auto initial_status=validate(current);
    result->initial_failure_reason=initial_status;
    result->initial_failure_stage=failure;
    if(initial_status!=MPC_RTI_ROLLOUT_OK) {
        /* An infeasible warm-start is not a source-exact control plan.
         * Use the constrained source dynamics to minimize actual footprint
         * penetration (Phase I) rather than restarting the old QP or
         * selecting fixed brake/steer commands. Success is declared ONLY
         * when the original nonlinear validator accepts the full horizon.
         * This is bounded by the same cycle time budget. */
        if(initial_status!=MPC_RTI_ROLLOUT_CORRIDOR ||
            !source_corridor_phase_one(
                initial,progress,trajectory,trajectory_count,lap_length,
                prediction_dt,*configuration,*frozen_objective,*options,
                begin,current,result,nullptr) ||
            validate(current)!=MPC_RTI_ROLLOUT_OK) {
            result->wall_time_ms=
                std::chrono::duration<double,std::milli>(
                    Clock::now()-begin).count();
            return 0;
        }
    }

    result->initial_valid=1;
    result->final_valid=1;
    result->initial_objective=exact_plan->validation_objective;
    result->final_objective=result->initial_objective;
    result->accepted_exact_objectives[0]=result->initial_objective;
    result->minimum_corridor_clearance_m=
        exact_plan->validation_minimum_corridor_slack_m;
    std::copy_n(current,N,result->controls);

    for(int iteration=0;iteration<options->max_iterations;++iteration) {
        if(out_of_time()) {
            result->budget_exhausted=1;
            break;
        }
        ++result->iterations_attempted;
        double derivative[N][MPC_RTI_NU]{};
        ++result->source_adjoint_evaluations;
        /* Every accepted trial leaves a full source-validated trajectory
         * in exact_plan. Do NOT integrate that physical trajectory a second
         * time merely to compute its LTV Jacobian/costate. The same-epoch
         * reference reconstruction has independent strict gradient parity
         * tests against the original source-integrating adjoint. */
        if(!mpc_rti_source_adjoint_gradient_from_exact_nominal(
                exact_plan.get(),trajectory,trajectory_count,
                lap_length,prediction_dt,&configuration->model,
                frozen_objective,derivative))
            break;

        double norms[MPC_RTI_NU]{};
        for(int k=0;k<N;++k)
            for(int c=0;c<MPC_RTI_NU;++c)
                norms[c]+=derivative[k][c]*derivative[k][c];
        for(auto &v:norms)v=std::sqrt(v);
        if(norms[0]<1.e-8 && norms[1]<1.e-8)break;

        // Scale each physical channel by its full-horizon gradient norm.
        // This is a direction preconditioner, not a hard trust region on the
        // QP and not a controller fallback. Source merit determines scale.
        double direction[N][MPC_RTI_NU]{};
        for(int k=0;k<N;++k) {
            direction[k][0]=-double(options->steering_direction_norm_rad)*
                derivative[k][0]/std::max(norms[0],1.e-8);
            direction[k][1]=-double(options->throttle_direction_norm)*
                derivative[k][1]/std::max(norms[1],1.e-8);
        }

        // Audit first-order consistency in the actually proposed direction.
        // Crossing tire/contact branches cannot be called a precise
        // derivative even if the local source AD is infinitesimally right.
        if (options->directional_check_step_scale>0.f) {
            const double h=options->directional_check_step_scale;
            MpcModelControl_t plus[N]{},minus[N]{};
            build_candidate(current,direction,h,configuration->model,plus);
            build_candidate(current,direction,-h,configuration->model,minus);
            const bool plus_valid=validate(plus)==MPC_RTI_ROLLOUT_OK;
            const double plus_cost=plus_valid
                ? exact_plan->validation_objective : NAN;
            const bool minus_valid=validate(minus)==MPC_RTI_ROLLOUT_OK;
            const double minus_cost=minus_valid
                ? exact_plan->validation_objective : NAN;
            if (plus_valid || minus_valid) {
                ++result->source_derivative_consistency_checks;
                // Central differences cancel the leading finite-step bias.
                // At active bounds use the actual projected physical
                // displacement, not an impossible unclipped direction.
                const double exact_slope=
                    plus_valid && minus_valid
                        ? (plus_cost-minus_cost)/(2.*h)
                        : plus_valid
                            ? (plus_cost-result->final_objective)/h
                            : (result->final_objective-minus_cost)/h;
                const auto &first=plus_valid?plus:current;
                const auto &second=minus_valid?minus:current;
                const double ad_slope=
                    projected_directional_product(
                        second,first,derivative,
                        plus_valid && minus_valid?2.*h:h);
                const double rel=std::abs(exact_slope-ad_slope)/
                    std::max({1.,std::abs(exact_slope),
                        std::abs(ad_slope)});
                result->max_directional_derivative_relative_error=
                    std::max(result->max_directional_derivative_relative_error,
                             rel);
                result->directional_relative_error_by_iteration[iteration]=rel;
                if(rel>.15) ++result->source_derivative_discrepancies;
            }
        }

        const double old_cost=result->final_objective;
        // Bracket a meaningful physical improvement in BOTH directions of
        // step size: the first valid descent is not necessarily the best.
        // Large finite changes may cross source tire/contact branches; they
        // are never judged by the frozen linearized QP merit. Conversely,
        // skipping larger options would turn this back into a tiny-alpha
        // pseudo-SQP. The total exact-source evaluation count is bounded.
        /* Interleave smaller and larger exact moves. A stiff corner can
         * require scale .5 while a high-speed correction benefits from
         * scale 2/4; trying six ever-larger steps and stopping before
         * evaluating .5 caused zero improvement at an otherwise viable
         * saved corner. Both directions are source-evaluated, not guessed
         * from the frozen QP. */
        constexpr double scale_multipliers[16] = {
            1.,.5,2.,.25,4.,.125,8.,.0625,16.,.03125,32.,
            .015625,.0078125,.00390625,.001953125,.0009765625};
        double best=old_cost;
        double best_scale=0.;
        double best_clearance=result->minimum_corridor_clearance_m;
        MpcModelControl_t best_controls[N]{};
        MpcModelControl_t trial[N]{};
        for(int step=0;step<options->max_exact_trials_per_iteration;++step) {
            if(out_of_time()) {
                result->budget_exhausted=1;
                break;
            }
            const double scale=std::min(
                4.,double(options->initial_step_scale)*
                    scale_multipliers[step]);
            if(scale<options->minimum_step_scale) continue;
            build_candidate(current,direction,scale,
                configuration->model,trial);
            const double slope=projected_directional_product(
                current,trial,derivative,1.);
            if(slope>=-1.e-10) continue;
            if(validate(trial)!=MPC_RTI_ROLLOUT_OK)continue;
            const double cost=exact_plan->validation_objective;
            if(std::isfinite(cost) && cost<best &&
                cost<=old_cost+options->armijo_fraction*slope) {
                best=cost;
                best_scale=scale;
                best_clearance=
                    exact_plan->validation_minimum_corridor_slack_m;
                std::copy_n(trial,N,best_controls);
                /* Candidate merit may evaluate more trials afterward.
                 * Preserve the PHYSICAL state corresponding to the best
                 * accepted commands for the next gradient, rather than
                 * accidentally differentiating the last tested trial. */
                *best_plan=*exact_plan;
            }
        }
        if(best_scale==0.)break;
        for(int k=0;k<N;++k) {
            result->largest_control_step_rad=std::max(
                result->largest_control_step_rad,
                std::abs(double(best_controls[k].steering_target_rad)-
                    current[k].steering_target_rad));
            result->largest_throttle_step=std::max(
                result->largest_throttle_step,
                std::abs(double(best_controls[k].throttle_command)-
                    current[k].throttle_command));
        }
        std::copy_n(best_controls,N,current);
        *exact_plan=*best_plan;
        std::copy_n(current,N,result->controls);
        result->final_objective=best;
        result->minimum_corridor_clearance_m=best_clearance;
        ++result->iterations_accepted;
        result->accepted_exact_objectives[result->iterations_accepted]=best;
        result->accepted_step_scales[result->iterations_accepted-1]=
            best_scale;
    }
    if (certified_nominal && result->final_valid && exact_plan->valid)
        *certified_nominal=*exact_plan;
    result->wall_time_ms=
        std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    return 1;
}
