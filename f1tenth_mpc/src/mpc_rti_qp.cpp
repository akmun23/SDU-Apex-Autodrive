// The only full-state LTV QP construction: source Jacobians, cost and corridor.
// Exact nonlinear rollout and acceptance remain in mpc_rti.cpp.
#include "mpc_rti_internal.hpp"
#include "mpc_steering_variation.hpp"
#include "mpc_tracking_tolerance.h"
#include "vehicle_model.h"
#include "mpc_qp_convexity.hpp"
#include <math.h>
#include <string.h>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

static void set_unconstrained_bounds(RiccatiStepData_t *stage)
{
    for (int i = 0; i < MPC_RTI_NX; ++i) {
        stage->x_lb[i] = -BIG_BOUND;
        stage->x_ub[i] = BIG_BOUND;
    }
    for (int i = 0; i < MPC_RTI_NU; ++i) {
        stage->u_lb[i] = -BIG_BOUND;
        stage->u_ub[i] = BIG_BOUND;
    }
}

/* Broad rear-axle boxes only. The full footprint belongs to the coupled
 * corner half-planes, never a heading-frozen e_y extent. */
static int set_state_bounds_for_corridor_polygon(
    float lower[MPC_RTI_NX],
    float upper[MPC_RTI_NX],
    float ey_lower,
    float ey_upper,
    const MpcRtiConfiguration_t *configuration)
{
    if (!isfinite(ey_lower) || !isfinite(ey_upper) || ey_lower > ey_upper)
        return 0;

    for (int i = 0; i < MPC_RTI_NX; ++i) {
        lower[i] = -BIG_BOUND;
        upper[i] = BIG_BOUND;
    }
    lower[MPC_STATE_IDX_EY] = ey_lower;
    upper[MPC_STATE_IDX_EY] = ey_upper;
    lower[MPC_STATE_IDX_EPSI] = -3.14159265358979323846f;
    upper[MPC_STATE_IDX_EPSI] = 3.14159265358979323846f;
    lower[MPC_STATE_IDX_U] = -MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS;
    upper[MPC_STATE_IDX_U] =
        vehicle_model_get_parameters().maximum_command_speed_mps;
    lower[5] =
        -configuration->max_steering_rad;
    upper[5] =
        configuration->max_steering_rad;
    // Wheel rates have no source/validator hard saturation. Keep the
    // standard +/-BIG_BOUND unconstrained sentinel; the old +/-350 values
    // also exceeded that sentinel and were never active physical bounds.
    return 1;
}

static void add_tracking_term(
    float *q_diag,
    float *q_linear,
    int index,
    float weight,
    float target,
    float multiplier)
{
    /* Zero-weight channels are common in the active profile (for example
     * lateral velocity). Avoid constructing/multiplying a term that cannot
     * affect the QP. Keeping this helper scalar also lets the compiler fold
     * the fixed factor without creating per-stage temporary arrays. */
    if (weight <= 0.0f) return;
    const float scaled_weight = multiplier * weight;
    const float doubled_weight = 2.0f * scaled_weight;
    q_diag[index] += doubled_weight;
    q_linear[index] -= doubled_weight * target;
}

float overspeed_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference)
{
    const float curvature_scale = 1.0f +
        configuration->overspeed_curvature_gain_m *
        fabsf(reference->path_curvature);
    return configuration->weight_u_overspeed * curvature_scale;
}

float underspeed_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference)
{
    const float curvature_scale = 1.0f +
        configuration->underspeed_curvature_gain_m *
        fabsf(reference->path_curvature);
    return configuration->weight_u / curvature_scale;
}

float longitudinal_slip_weight_for_reference(
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiReference_t *reference)
{
    const float curvature_scale = 1.0f +
        configuration->longitudinal_slip_curvature_gain_m *
        fabsf(reference->path_curvature);
    return configuration->weight_longitudinal_slip * curvature_scale;
}

/* Preserve the objective used by the successful v1 witness and geometry-only
 * controller: damp correction increments, not the steering change required by
 * the loaded path. This reference is physical rack/feedforward steering, not
 * an independently resampled future command history. Keep QP and exact merit
 * on the same reference schedule. No track-specific behavior belongs here. */
float steering_reference_increment(
    const MpcRtiReference_t *current, const MpcRtiReference_t *next)
{
    return next->steering_command-current->steering_command;
}

/* PSD Gauss-Newton model for the ACTIVE physical footprint corner's
 * squared reserve deficit. The exact source merit uses the minimum of all
 * corners at the actual nonlinear heading; the QP linearizes this single
 * currently limiting plane at the already-integrated source nominal.
 * No new hard wall box and no extra source rollout. At the terminal state
 * Riccati stores only a diagonal Hessian, so use a diagonal PSD majorizer
 * with the SAME exact first derivative. */
static void add_footprint_reserve_merit(
    float Q[MPC_RTI_NX],
    float q[MPC_RTI_NX],
    float (*Qcross)[MPC_RTI_NX],
    const RiccatiStatePolygon_t *polygon,
    const MpcModelState_t *nominal,
    const MpcRtiConfiguration_t *model)
{
    if(!polygon || !nominal || !model ||
       model->weight_source_corridor_reserve<=0.f ||
       model->near_wall_tracking_reserve_m<=0.f)return;
    float slack=INFINITY;
    int active=-1;
    for(int i=0;i<polygon->count;++i) {
        const float candidate=polygon->upper[i]-
            polygon->normal[i][0]*nominal->e_y-
            polygon->normal[i][1]*nominal->e_psi;
        if(candidate<slack) {slack=candidate;active=i;}
    }
    if(active<0 || !(slack<model->near_wall_tracking_reserve_m))return;
    const float a=polygon->normal[active][0];
    const float b=polygon->normal[active][1];
    const float w2=2.f*model->weight_source_corridor_reserve;
    const float residual=model->near_wall_tracking_reserve_m-slack;
    if(Qcross) {
        Q[MPC_STATE_IDX_EY]+=w2*a*a;
        Q[MPC_STATE_IDX_EPSI]+=w2*b*b;
        Qcross[MPC_STATE_IDX_EY][MPC_STATE_IDX_EPSI]+=w2*a*b;
        Qcross[MPC_STATE_IDX_EPSI][MPC_STATE_IDX_EY]+=w2*a*b;
        const float affine=model->near_wall_tracking_reserve_m-
            polygon->upper[active];
        q[MPC_STATE_IDX_EY]+=w2*affine*a;
        q[MPC_STATE_IDX_EPSI]+=w2*affine*b;
    } else {
        const float cross=fabsf(a*b);
        const float qa=w2*(a*a+cross);
        const float qb=w2*(b*b+cross);
        Q[MPC_STATE_IDX_EY]+=qa;
        Q[MPC_STATE_IDX_EPSI]+=qb;
        q[MPC_STATE_IDX_EY]+=w2*residual*a-qa*nominal->e_y;
        q[MPC_STATE_IDX_EPSI]+=w2*residual*b-qb*nominal->e_psi;
    }
}

// Source-consistent front tire saturation penalty. The instantaneous source
// wheel-contact kinematics are exactly the ones used by the 1ms tire callback.
// The full 20-physical-state derivative is evaluated at each already
// integrated nominal. This is a PSD GN Hessian and preserves the exact
// nominal gradient even at the Riccati terminal (diagonal majorizer).
static int add_source_front_lateral_slip_merit(
    float Q[MPC_RTI_NX],float q[MPC_RTI_NX],
    float (*Qcross)[MPC_RTI_NX],
    const MpcModelState_t *physical,
    const float nominal_x[MPC_RTI_NX],
    const MpcRtiConfiguration_t *model,float multiplier)
{
    if(!model || model->weight_source_lateral_slip<=0.f)return 1;
    if(!physical || !nominal_x)return 0;
    double slip[4]{};
    double J[4][MPC_OPT_PHYSICAL_NX]{};
    if(!vehicle_model_source_lateral_slip(physical,slip,J))return 0;
    const double w2=2.*model->weight_source_lateral_slip*multiplier;
    for(int front=0;front<2;++front) {
        const double magnitude=std::abs(slip[front]);
        if(magnitude<=kMpcSourceFrontLateralSlipReserveRad)continue;
        const double r=slip[front]-std::copysign(
            kMpcSourceFrontLateralSlipReserveRad,slip[front]);
        for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i) {
            if(J[front][i]==0.)continue;
            double affine=r;
            for(int j=0;j<MPC_OPT_PHYSICAL_NX;++j)
                affine-=J[front][j]*nominal_x[j];
            if(Qcross) {
                Q[i]+=(float)(w2*J[front][i]*J[front][i]);
                q[i]+=(float)(w2*J[front][i]*affine);
                for(int j=0;j<MPC_OPT_PHYSICAL_NX;++j)
                    if(i!=j)
                        Qcross[i][j]+=
                           (float)(w2*J[front][i]*J[front][j]);
            } else {
                double sum=0.;
                for(int j=0;j<MPC_OPT_PHYSICAL_NX;++j)
                    sum+=std::abs(J[front][j]);
                const double diagonal=w2*std::abs(J[front][i])*sum;
                Q[i]+=(float)diagonal;
                q[i]+=(float)(w2*J[front][i]*r-
                    diagonal*nominal_x[i]);
            }
        }
    }
    return 1;
}

static void add_tracking_cost(
    float Q[MPC_RTI_NX],
    float q[MPC_RTI_NX],
    const MpcRtiReference_t *reference,
    const MpcRtiConfiguration_t *configuration,
    float multiplier,
    const MpcModelState_t *nominal_plant,
    const float *frozen_speed_weight)
{
    const float nominal_u = nominal_plant->u;
    const float speed_weight = frozen_speed_weight ? *frozen_speed_weight :
        (nominal_u > reference->u
            ? overspeed_weight_for_reference(configuration, reference)
            : underspeed_weight_for_reference(configuration, reference));
    float target;
    if (mpc_tracking_local_target(nominal_plant->e_y, reference->e_y,
            configuration->tracking_lateral_tolerance_m, &target))
        add_tracking_term(Q, q, MPC_STATE_IDX_EY,
            configuration->weight_e_y, target, multiplier);
    if (mpc_tracking_local_target(nominal_plant->e_psi, reference->e_psi,
            configuration->tracking_heading_tolerance_rad, &target))
        add_tracking_term(Q, q, MPC_STATE_IDX_EPSI,
            configuration->weight_e_psi, target, multiplier);
    add_tracking_term(Q, q, MPC_STATE_IDX_U,
        speed_weight, reference->u, multiplier);
    add_tracking_term(Q, q, MPC_STATE_IDX_V,
        configuration->weight_v, reference->v, multiplier);
    add_tracking_term(Q, q, MPC_STATE_IDX_R,
        configuration->weight_r, reference->r, multiplier);
    add_tracking_term(Q, q, 5,
        configuration->weight_steering_command,
        reference->steering_command, multiplier);
}

/* The measured full-interval wheel slip residual is part of the EXACT merit.
 * Preserve its complete Gauss-Newton outer product in (state,steer,throttle).
 * The old diagonal approximation discarded steering/throttle and state/input
 * coupling even though the source sensitivity already supplies all columns.
 * H=2w J^T J is PSD. Affine terms preserve the exact nominal gradient. */
static void add_source_transition_slip_cost(
    RiccatiStepData_t *stage, const MpcStageLinearization_t *linearization,
    const float nominal_state[MPC_RTI_NX],
    const MpcModelControl_t *control, float weight, float multiplier)
{
    if (weight <= 0.0f) return;
    const double w2=2.0*weight*multiplier;
    const double ubar[2]={control->steering_target_rad,control->throttle_command};
    for(int wheel=0;wheel<4;++wheel) {
        const double residual=linearization->maximum_slip_excess_mps[wheel];
        if(residual<=0.)continue;
        double gx[MPC_RTI_NX]={};
        double gu[2]={
            linearization->maximum_slip_excess_B[wheel][0],
            linearization->maximum_slip_excess_B[wheel][1]};
        double nominal_derivative=0.;
        for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i) {
            const int col=i<5 ? i : i==5 ? 7 : i+4;
            gx[i]=linearization->maximum_slip_excess_A[wheel][col];
            nominal_derivative+=gx[i]*nominal_state[i];
        }
        for(int a=0;a<2;++a)nominal_derivative+=gu[a]*ubar[a];
        const double tangent=w2*(residual-nominal_derivative);
        for(int i=0;i<MPC_OPT_PHYSICAL_NX;++i) {
            stage->q[i]+=(float)(tangent*gx[i]);
            stage->Q_diag[i]+=(float)(w2*gx[i]*gx[i]);
            for(int j=0;j<MPC_OPT_PHYSICAL_NX;++j) {
                if(i!=j)
                    stage->Q_offdiag[i][j]+=(float)(w2*gx[i]*gx[j]);
            }
            for(int a=0;a<2;++a)
                stage->N[i][a]+=(float)(w2*gx[i]*gu[a]);
        }
        for(int a=0;a<2;++a) {
            stage->r[a]+=(float)(tangent*gu[a]);
            stage->R_diag[a]+=(float)(w2*gu[a]*gu[a]);
        }
        stage->R_cross+=(float)(w2*gu[0]*gu[1]);
    }
}

double source_slip_cost(
    const MpcRtiConfiguration_t *configuration,
    const MpcModelState_t *plant,
    const MpcRtiReference_t *reference)
{
    if (!configuration || !plant ||
        !reference || configuration->weight_longitudinal_slip <= 0.0f)
        return 0.0;
    const double weight = longitudinal_slip_weight_for_reference(
        configuration, reference);
    double excess[4] = {0}, derivative[4] = {};
    if (!vehicle_model_longitudinal_slip_excess(
            plant, excess, derivative)) return INFINITY;
    double cost = 0.0;
    for (int wheel = 0; wheel < 4; ++wheel)
        cost += weight * excess[wheel] * excess[wheel];
    return cost;
}

int mpc_rti_transform_qp_to_command_increments(
    MpcRtiProblem_t *problem, const MpcRtiConfiguration_t *config,
    float prediction_dt)
{
    if (!problem || !config || problem->horizon != RICCATI_HORIZON ||
        !config->command_increment_qp_enabled ||
        !(config->max_steering_command_step_rad > 0.f) ||
        !(config->max_throttle_command_step > 0.f)) return 0;

    const int previous[2] = {
        mpc_rti_steering_qp_state_index(config),
        MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND};
    const float step_limit[2] = {
        mpc_rti_steering_qp_step_limit(config,prediction_dt),
        config->max_throttle_command_step};
    const float absolute_min[2] = {
        -config->max_steering_rad, MPC_THROTTLE_MIN_DRIVE};
    const float absolute_max[2] = {
        config->max_steering_rad, MPC_THROTTLE_MAX};

    // Coordinate transformation for the COMPLETE symmetric stage Hessian.
    // Write u_abs = T*x + du, where T selects previous absolute commands.
    // Q'=Q+NT+(NT)^T+T^T R T, N'=N+T^T R, q'=q+T^T r.
    // Dynamics A'=A+B T. The earlier diagonal-only specialization rejected
    // the very state/input coupling introduced by full source-slip GN.
    for(int k=0;k<RICCATI_HORIZON;++k) {
        RiccatiStepData_t *stage=&problem->steps[k];
        const float R[2][2]={
            {stage->R_diag[0],stage->R_cross},
            {stage->R_cross,stage->R_diag[1]}};
        float original_N[MPC_RTI_NX][2];
        for(int i=0;i<MPC_RTI_NX;++i)
            for(int a=0;a<2;++a)original_N[i][a]=stage->N[i][a];
        for(int i=0;i<MPC_RTI_NX;++i) {
            for(int j=0;j<MPC_RTI_NX;++j) {
                double add=0.;
                for(int a=0;a<2;++a) {
                    if(j==previous[a])add+=original_N[i][a];
                    if(i==previous[a])add+=original_N[j][a];
                    for(int b=0;b<2;++b)
                        if(i==previous[a]&&j==previous[b])add+=R[a][b];
                }
                if(i==j)stage->Q_diag[i]+=(float)add;
                else stage->Q_offdiag[i][j]+=(float)add;
            }
            for(int a=0;a<2;++a)
                for(int b=0;b<2;++b)
                    if(i==previous[b])stage->N[i][a]+=R[b][a];
        }
        for(int a=0;a<2;++a) {
            const int p=previous[a];
            stage->q[p]+=stage->r[a];
            for(int i=0;i<MPC_RTI_NX;++i)
                stage->A[i][p]+=stage->B[i][a];
            if(k>0) {
                stage->x_lb[p]=absolute_min[a];
                stage->x_ub[p]=absolute_max[a];
            }
            stage->u_lb[a]=-step_limit[a];
            stage->u_ub[a]=step_limit[a];
        }
    }

    for(int j=0;j<2;++j) {
        problem->terminal_x_lb[previous[j]]=absolute_min[j];
        problem->terminal_x_ub[previous[j]]=absolute_max[j];
    }
    return 1;
}

/* Physical geometry is evaluated at the live nonlinear nominal; objective
 * targets and speed-weight branches can be frozen from an earlier SQP iterate.
 * Distinguishing these is essential: using stale geometry makes footprint
 * constraints wrong, while recomputing the merit's targets gives a QP gradient
 * of a DIFFERENT objective. The public legacy entrypoint below remains a
 * convenience for one-pass/unfrozen QP callers. */
/* The exact rollout evaluates track width at candidate progress. Construct
 * its locally consistent first derivative using the identical source-frame
 * path sampler, not stage-indexed width differences (those conflate speed).
 * The final exact rollout remains the nonlinear collision authority. */
static int source_corridor_progress_slopes(
    const MpcTrajectorySample_t *trajectory, size_t count, double lap,
    double progress, double previous_progress,
    const MpcRtiConfiguration_t *model,float *left,float *right)
{
    /* The preview-minimum wall width is only piecewise differentiable.
     * Local derivatives sampled at 2mm can be ~10 m/m while a physically
     * realized 20mm SQP change crosses a different active corridor knot.
     *
     * At subsequent SQP passes, use the ACTUAL finite-width secant between
     * two source-integrated same-epoch trajectories. This is exact for the
     * two observed stations, unlike extrapolating across a knot with an
     * infinitesimal derivative. The hard original-source validator still
     * handles every candidate; this local model is not a safety certificate.
     *
     * If the displacement is tiny or exceeds a local physical step, fall
     * back to the true local derivative, avoiding noisy/divergent secants. */
    constexpr double h=0.002;
    if(!isfinite(progress) || !left || !right)return 0;
    if(isfinite(previous_progress) &&
       fabs(progress-previous_progress)>=.004 &&
       fabs(progress-previous_progress)<=.5) {
        MpcRtiReference_t now{},previous{};
        if(!mpc_rti_reference_at_progress(trajectory,count,lap,progress,
                                         model,&now) ||
           !mpc_rti_reference_at_progress(trajectory,count,lap,
                                         previous_progress,model,&previous))
            return 0;
        const double ds=progress-previous_progress;
        *left=float((now.left_bound-previous.left_bound)/ds);
        *right=float((now.right_bound-previous.right_bound)/ds);
    } else {
        MpcRtiReference_t before{},after{};
        if(!mpc_rti_reference_at_progress(trajectory,count,lap,progress-h,
                                         model,&before) ||
           !mpc_rti_reference_at_progress(trajectory,count,lap,progress+h,
                                         model,&after))return 0;
        *left=float((after.left_bound-before.left_bound)/(2.*h));
        *right=float((after.right_bound-before.right_bound)/(2.*h));
    }
    return isfinite(*left) && isfinite(*right);
}

int mpc_rti_build_ltv_qp_with_progress_corridor(
    const MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1],
    const MpcModelControl_t nominal_controls[PREDICTION_HORIZON],
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiObjectiveSchedule_t *objective_schedule,
    MpcRtiProblem_t *problem,
    const double nominal_progress[PREDICTION_HORIZON+1],
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length,
    const double previous_progress[PREDICTION_HORIZON+1])
{
    if (!nominal_states || !nominal_controls || !references || !problem ||
        !valid_configuration(configuration) ||
        horizon != PREDICTION_HORIZON || !isfinite(prediction_dt) ||
        prediction_dt <= 0.0f) {
        return 0;
    }
    if (objective_schedule &&
        (!objective_schedule->valid || objective_schedule->horizon != horizon))
        return 0;
    if (nominal_progress &&
        (!trajectory || trajectory_count < 3 ||
         !isfinite(lap_length) || lap_length <= 0.)) return 0;
    if(previous_progress && !nominal_progress)return 0;
    const MpcRtiReference_t *objective_references = objective_schedule
        ? objective_schedule->references : references;
    for (int k = 0; k <= horizon; ++k) {
        if (!finite_reference(&objective_references[k]) ||
            (objective_schedule &&
             (!isfinite(objective_schedule->speed_weights[k]) ||
              objective_schedule->speed_weights[k] < 0.f)))
            return 0;
    }
    if (!vehicle_model_set_active_target_speed_ceiling(
            configuration->active_speed_ceiling_mps)) return 0;
    for (int k = 0; k <= horizon; ++k) {
        if (!finite_rti_state(&nominal_states[k]) ||
            !finite_reference(&references[k])) return 0;
    }
    for (int k = 0; k < horizon; ++k) {
        if (!isfinite(nominal_controls[k].steering_target_rad) ||
            !isfinite(nominal_controls[k].throttle_command)) return 0;
    }

    *problem = {};
    /* Command-time prediction has already advanced to the bridge epoch where
     * this new decision takes effect. All 30 subsequent plant transitions
     * therefore remain decision stages in the QP. */
    problem->horizon = RICCATI_HORIZON;
    problem->linearization_failure_stage = -1;
    problem->linearization_failure_index = -1;
    /* The current state is predicted to the new command's application epoch.
     * Start optimization at that state; the stage plant consumes its decision
     * directly while the queued fields remain actuator-history bookkeeping. */
    serialize_state(&nominal_states[MPC_COMMAND_DELAY_STEPS], problem->x0);
    /* Match evaluate_nonlinear_candidate(): the first optimized input's
     * slew cost is measured from the command already present at the incoming
     * state. Seeding these auxiliary coordinates from the post-delay nominal
     * state instead compares the decision against a newly generated nominal
     * command, weakening/misaligning the first-action continuity penalty. */
    problem->x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
        nominal_states[0].plant.delayed_steering_command_2;
    problem->x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
        nominal_states[0].plant.delayed_throttle_command_2;

    MpcStageLinearization_t plant_linearizations[RICCATI_HORIZON];
    int linearization_ok[RICCATI_HORIZON] = {};
    double jacobian_stage_wall_us[RICCATI_HORIZON] = {0.0};
    double jacobian_stage_cpu_us[RICCATI_HORIZON] = {0.0};
    int jacobian_stage_worker[RICCATI_HORIZON] = {};
    uint64_t jacobian_stage_worker_os_tid[RICCATI_HORIZON] = {};
    const int jacobian_threads = configuration->jacobian_threads > 0
        ? configuration->jacobian_threads : 1;
    const double jacobian_start_us = monotonic_microseconds();
#ifdef _OPENMP
/* Interleave model stages across physical workers. Adjacent stages can
 * have sharply different source-contact sensitivity costs at a corner;
 * assigning five consecutive stages to the same worker creates a long
 * straggler and stalls the entire 30-stage Jacobian. Static stride-one
 * balancing retains deterministic ownership and avoids dynamic-queue
 * synchronization. Each stage remains independently and exactly evaluated. */
#pragma omp parallel for schedule(static, 1) num_threads(jacobian_threads) \
    if(horizon >= 8 && jacobian_threads > 1)
#endif
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        const double stage_start_us = monotonic_microseconds();
        const double stage_cpu_start_us = thread_cpu_microseconds();
#ifdef _OPENMP
        jacobian_stage_worker[k] = omp_get_thread_num();
#endif
#ifdef __linux__
        const long worker_os_tid = syscall(SYS_gettid);
        if (worker_os_tid > 0) {
            jacobian_stage_worker_os_tid[k] = (uint64_t)worker_os_tid;
        }
#endif
        const int model_stage = k + MPC_COMMAND_DELAY_STEPS;
        /* Physical sensitivities are path-independent. Exact loaded-frame
         * derivatives below replace the two tracking rows. */
        const float linearization_curvature =
            nominal_states[model_stage + 1].incoming_tracking_jacobian_valid
                ? 0.0f : references[model_stage].path_curvature;
        MpcStageLinearization_t *plant_linearization =
            &plant_linearizations[k];
        MpcStageResult_t known_step = {};
        known_step.next = nominal_states[model_stage + 1].plant;
        known_step.branch_flags =
            nominal_states[model_stage + 1].incoming_branch_flags;
        known_step.body_forward_delta_m =
            nominal_states[model_stage + 1].incoming_body_forward_delta_m;
        known_step.body_lateral_delta_m =
            nominal_states[model_stage + 1].incoming_body_lateral_delta_m;
        known_step.body_yaw_delta_rad =
            nominal_states[model_stage + 1].incoming_body_yaw_delta_rad;
        known_step.valid =
            nominal_states[model_stage + 1].incoming_transition_valid;
        const int linearized = known_step.valid
            ? mpc_model_linearize_sensitivities_from_known_step(
                &nominal_states[model_stage].plant,
                &nominal_controls[model_stage], prediction_dt,
                linearization_curvature,
                &known_step, plant_linearization)
            : mpc_model_linearize(&nominal_states[model_stage].plant,
                &nominal_controls[model_stage], prediction_dt,
                linearization_curvature, plant_linearization);
        linearization_ok[k] = linearized;
        if (linearized && nominal_states[model_stage + 1].
                incoming_tracking_jacobian_valid) {
            const float (*geometry)[6] = nominal_states[model_stage + 1].
                incoming_tracking_jacobian;
            for (int row = 0; row < 2; ++row) {
                for (int col = 0; col < MPC_MODEL_NX; ++col) {
                    float derivative = col < 2 ? geometry[row][col] : 0.0f;
                    for (int motion = 0; motion < 3; ++motion)
                        derivative += geometry[row][motion + 2] *
                            plant_linearization->body_motion_A[motion][col];
                    plant_linearization->A[row][col] = derivative;
                }
                for (int input = 0; input < MPC_MODEL_NU; ++input) {
                    float derivative = 0.0f;
                    for (int motion = 0; motion < 3; ++motion)
                        derivative += geometry[row][motion + 2] *
                            plant_linearization->body_motion_B[motion][input];
                    plant_linearization->B[row][input] = derivative;
                }
            }
        }
        jacobian_stage_wall_us[k] = monotonic_microseconds() - stage_start_us;
        const double stage_cpu_finish_us = thread_cpu_microseconds();
        if (isfinite(stage_cpu_start_us) && isfinite(stage_cpu_finish_us))
            jacobian_stage_cpu_us[k] =
                stage_cpu_finish_us - stage_cpu_start_us;
    }
    problem->jacobian_wall_us =
        monotonic_microseconds() - jacobian_start_us;
    int worker_seen[MPC_RTI_MAX_JACOBIAN_THREADS] = {};
    for (int k = 0; k < RICCATI_HORIZON; ++k) {
        problem->jacobian_slowest_stage_us = fmax(
            problem->jacobian_slowest_stage_us,
            jacobian_stage_wall_us[k]);
        problem->jacobian_thread_cpu_us += jacobian_stage_cpu_us[k];
        problem->jacobian_stages_completed += linearization_ok[k] != 0;
        problem->jacobian_source_sensitivity_calls +=
            plant_linearizations[k].source_sensitivity_call_count;
        problem->jacobian_finite_difference_columns +=
            plant_linearizations[k].finite_difference_column_count;
        const int worker = jacobian_stage_worker[k];
        if (worker >= 0 && worker < MPC_RTI_MAX_JACOBIAN_THREADS &&
            !worker_seen[worker]) {
            worker_seen[worker] = 1;
            ++problem->jacobian_threads_used;
            problem->jacobian_worker_os_tid[worker] =
                jacobian_stage_worker_os_tid[k];
        }
        if (worker >= 0 && worker < MPC_RTI_MAX_JACOBIAN_THREADS) {
            problem->jacobian_worker_cpu_us[worker] +=
                jacobian_stage_cpu_us[k];
            problem->jacobian_worker_stage_wall_us[worker] +=
                jacobian_stage_wall_us[k];
            problem->jacobian_worker_stage_count[worker] += 1;
        }
    }

    /* The four FIFO coordinates remain in the nonlinear state for causal
     * prediction and diagnostics, but the QP controls the plant directly from
     * its already-predicted command application epoch. */
    float nominal_optimizer_states[PREDICTION_HORIZON + 1][MPC_RTI_NX]{};
    for (int k = 0; k <= horizon; ++k) {
        serialize_state(&nominal_states[k], nominal_optimizer_states[k]);
        if (k <= MPC_COMMAND_DELAY_STEPS) {
            nominal_optimizer_states[k][
                MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
                nominal_states[k].plant.delayed_steering_command_2;
            nominal_optimizer_states[k][
                MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
                nominal_states[k].plant.delayed_throttle_command_2;
        } else {
            nominal_optimizer_states[k][
                MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] =
                nominal_controls[k - MPC_COMMAND_DELAY_STEPS - 1]
                    .steering_target_rad;
            nominal_optimizer_states[k][
                MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] =
                nominal_controls[k - MPC_COMMAND_DELAY_STEPS - 1]
                    .throttle_command;
        }
    }

    for (int solver_stage = 0; solver_stage < RICCATI_HORIZON;
         ++solver_stage) {
        const int model_stage = solver_stage + MPC_COMMAND_DELAY_STEPS;
        RiccatiStepData_t *stage = &problem->steps[solver_stage];
        set_unconstrained_bounds(stage);
        const MpcStageLinearization_t *lin =
            &plant_linearizations[solver_stage];
        if (!linearization_ok[solver_stage]) {
            problem->linearization_failure_stage = model_stage;
            problem->linearization_failure_reason = lin->failure_reason;
            problem->linearization_failure_index = lin->failure_index;
            return 0;
        }
        problem->nonsmooth_jacobian_columns +=
            lin->nonsmooth_column_count;
        const float *next_optimizer_state =
            nominal_optimizer_states[model_stage + 1];

        for (int row = 0; row < MPC_OPT_PHYSICAL_NX; ++row) {
            const int model_row = row < 5 ? row : row == 5 ? 7 : row + 4;
            stage->B[row][0] = lin->B[model_row][0];
            stage->B[row][1] = lin->B[model_row][1];
            for (int col = 0; col < MPC_OPT_PHYSICAL_NX; ++col) {
                const int model_col = col < 5 ? col : col == 5 ? 7 : col + 4;
                stage->A[row][col] = lin->A[model_row][model_col];
            }
            const float applied_steering =
                nominal_controls[model_stage].steering_target_rad;
            const float applied_throttle =
                nominal_controls[model_stage].throttle_command;
            /* Preserve the nominal-point identity for this direct-input
             * stage; accumulate cancellation-sensitive rows in double. */
            double affine_offset = next_optimizer_state[row];
            for (int col = 0; col < MPC_OPT_PHYSICAL_NX; ++col)
                affine_offset -= (double)stage->A[row][col] *
                    nominal_optimizer_states[model_stage][col];
            affine_offset -= (double)stage->B[row][0] * applied_steering;
            affine_offset -= (double)stage->B[row][1] * applied_throttle;
            stage->d[row] = (float)affine_offset;
        }

        /* The nonlinear rollout resamples the reference as progress changes.
         * Carry that same dependence through later LTV stages. This scalar
         * is fixed to zero at the nominal and propagated by dynamics; it has
         * no tracking weight or independently selectable initial value. */
        const int progress_index = MPC_OPT_IDX_PROGRESS_DEVIATION;
        stage->A[progress_index][progress_index] = 1.0f;
        if (nominal_states[model_stage + 1].incoming_tracking_jacobian_valid) {
            const float (*geometry)[6] = nominal_states[model_stage + 1].
                incoming_tracking_jacobian;
            stage->A[0][progress_index] = geometry[0][5];
            stage->A[1][progress_index] = geometry[1][5];
            stage->A[progress_index][progress_index] = geometry[2][5];
            double offset = 0.0;
            for (int col = 0; col < MPC_OPT_PHYSICAL_NX; ++col) {
                const int model_col = col < 5 ? col : col == 5 ? 7 : col + 4;
                float derivative = col < 2 ? geometry[2][col] : 0.0f;
                for (int motion = 0; motion < 3; ++motion)
                    derivative += geometry[2][motion + 2] *
                        lin->body_motion_A[motion][model_col];
                stage->A[progress_index][col] = derivative;
                offset -= (double)derivative *
                    nominal_optimizer_states[model_stage][col];
            }
            const float inputs[2] = {
                nominal_controls[model_stage].steering_target_rad,
                nominal_controls[model_stage].throttle_command};
            for (int input = 0; input < MPC_RTI_NU; ++input) {
                float derivative = 0.0f;
                for (int motion = 0; motion < 3; ++motion)
                    derivative += geometry[2][motion + 2] *
                        lin->body_motion_B[motion][input];
                stage->B[progress_index][input] = derivative;
                offset -= (double)derivative * inputs[input];
            }
            stage->d[progress_index] = (float)offset;
        }

        /* Exact command-slew augmentation. These two auxiliary coordinates
         * are not vehicle states: they carry the previous optimized input so
         * adjacent steering/throttle increments are quadratic costs while
         * the physical plant states and controls stay unchanged. */
        stage->B[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND][0] = 1.0f;
        stage->B[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND][1] = 1.0f;

        /* The state at solver stage 0 is the fixed command-time initial state,
         * so its state cost is constant and omitted. Remaining in-horizon costs
         * are unchanged; the endpoint cost is transferred to terminal_Q/q. */
        if (model_stage > MPC_COMMAND_DELAY_STEPS) {
            add_tracking_cost(stage->Q_diag, stage->q,
                &objective_references[model_stage], configuration, 1.0f,
                &nominal_states[model_stage].plant,
                objective_schedule ? &objective_schedule->speed_weights[model_stage]
                                   : nullptr);
        }

        add_source_transition_slip_cost(stage, lin,
            nominal_optimizer_states[model_stage],
            &nominal_controls[model_stage],
            longitudinal_slip_weight_for_reference(configuration,
                &objective_references[model_stage + 1]),
            model_stage + 1 == horizon ? configuration->terminal_multiplier : 1.0f);

        /* Align the stage steering decision with the path feedforward at the
         * following prediction sample. */
        {
            const int reference_index = solver_stage + 1;
            const float doubled_weight = 2.0f *
                configuration->weight_steering_command *
                (model_stage + 1 == horizon ? configuration->terminal_multiplier : 1.0f);
            stage->R_diag[0] += doubled_weight;
            stage->r[0] -= doubled_weight *
                objective_references[reference_index].steering_command;
        }

        stage->R_diag[0] +=
            2.0f * configuration->weight_steering_target;
        /* A controller steering *reference* is geometric feedforward from
         * path curvature, never an IPOPT open-loop actuator command. Keep
         * this QP term aligned with the exact nonlinear objective. */
        stage->r[0] -= 2.0f * configuration->weight_steering_target *
            objective_references[model_stage].steering_command;
        stage->R_diag[1] +=
            2.0f * configuration->weight_throttle_command;
        const float steering_reference_delta = steering_reference_increment(
            &objective_references[solver_stage],
            &objective_references[solver_stage + 1]);
        const float nominal_previous_delta = solver_stage == 0 ?
            nominal_states[0].plant.delayed_steering_command_2 :
            nominal_controls[solver_stage-1].steering_target_rad;
        const float nominal_correction_delta =
            nominal_controls[solver_stage].steering_target_rad-
            nominal_previous_delta-steering_reference_delta;
        const float pseudo_huber_quadratic =
            mpc_steering_variation_quadratic_weight(
                configuration->weight_steering_correction_variation,
                configuration->steering_correction_variation_epsilon_rad,
                nominal_correction_delta);
        const float steering_slew_hessian =
            2.0f * (configuration->weight_steering_slew+
                      pseudo_huber_quadratic);
        const float throttle_slew_hessian =
            2.0f * configuration->weight_throttle_slew;
        stage->q[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] +=
            steering_slew_hessian * steering_reference_delta;
        stage->r[0] -= steering_slew_hessian * steering_reference_delta;
        stage->Q_diag[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND] +=
            steering_slew_hessian;
        stage->R_diag[0] += steering_slew_hessian;
        stage->N[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND][0] -=
            steering_slew_hessian;
        stage->Q_diag[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND] +=
            throttle_slew_hessian;
        stage->R_diag[1] += throttle_slew_hessian;
        stage->N[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND][1] -=
            throttle_slew_hessian;

        const int bound_index = model_stage;
        const float margin = corridor_margin_at_prediction(configuration, bound_index);
        const float ey_lower = margin - references[bound_index].right_bound;
        const float ey_upper = references[bound_index].left_bound - margin;
        if (!set_state_bounds_for_corridor_polygon(stage->x_lb, stage->x_ub,
                ey_lower, ey_upper, configuration)) return 0;
        if (!mpc_rti_build_corridor_polygon(&nominal_states[bound_index].plant,
                references[bound_index].left_bound, references[bound_index].right_bound,
                configuration, bound_index, &problem->state_polygons[solver_stage])) return 0;
        if(nominal_progress && model_stage>0) {
            float left_slope=0.f,right_slope=0.f;
            if(!source_corridor_progress_slopes(trajectory,trajectory_count,
                    lap_length,nominal_progress[model_stage],
                    previous_progress?previous_progress[model_stage]:NAN,
                    configuration,
                    &left_slope,&right_slope))return 0;
            auto &polygon=problem->state_polygons[solver_stage];
            for(int face=0;face<polygon.count;++face)
                polygon.progress_normal[face]=
                    -(face%2==0?left_slope:right_slope);
            if(left_slope!=0.f || right_slope!=0.f) {
                /* The nominal-progress e_y box would contradict the
                 * progress-coupled polygon, especially in a narrowing turn.
                 * Keep broad physical heading bounds and give delta_s a
                 * nonbinding ±49m split box (the 750ms legal speed envelope
                 * allows at most 12m) so ADMM projects all three axes. */
                stage->x_lb[MPC_OPT_IDX_EY]=-BIG_BOUND+1.f;
                stage->x_ub[MPC_OPT_IDX_EY]= BIG_BOUND-1.f;
                stage->x_lb[MPC_OPT_IDX_PROGRESS_DEVIATION]=-BIG_BOUND+1.f;
                stage->x_ub[MPC_OPT_IDX_PROGRESS_DEVIATION]= BIG_BOUND-1.f;
            }
        }
        if(model_stage>0)
            add_footprint_reserve_merit(stage->Q_diag,stage->q,
                stage->Q_offdiag,&problem->state_polygons[solver_stage],
                &nominal_states[model_stage].plant,configuration);
        if(model_stage>0 &&
           !add_source_front_lateral_slip_merit(
               stage->Q_diag,stage->q,stage->Q_offdiag,
               &nominal_states[model_stage].plant,
               nominal_optimizer_states[model_stage],configuration,1.f))
            return 0;
        stage->u_lb[0] = -configuration->max_steering_rad;
        stage->u_ub[0] = configuration->max_steering_rad;
        stage->u_lb[1] = MPC_THROTTLE_MIN_DRIVE;
        stage->u_ub[1] = MPC_THROTTLE_MAX;
    }

    /* Terminal state costs were previously carried by a held-state tail stage.
     * Put them in the solver's actual terminal cost and omit that dead stage. */
    add_tracking_cost(problem->terminal_Q, problem->terminal_q,
        &objective_references[horizon], configuration,
        configuration->terminal_multiplier, &nominal_states[horizon].plant,
        objective_schedule ? &objective_schedule->speed_weights[horizon]
                           : nullptr);

    const float terminal_margin = corridor_margin_at_prediction(configuration, horizon);
    const float terminal_lower = terminal_margin - references[horizon].right_bound;
    const float terminal_upper = references[horizon].left_bound - terminal_margin;
    if (!set_state_bounds_for_corridor_polygon(problem->terminal_x_lb,
            problem->terminal_x_ub,
            terminal_lower, terminal_upper,
            configuration)) return 0;
    if (!mpc_rti_build_corridor_polygon(&nominal_states[horizon].plant,
            references[horizon].left_bound, references[horizon].right_bound,
            configuration, horizon, &problem->state_polygons[RICCATI_HORIZON])) return 0;
    if(nominal_progress) {
        float left_slope=0.f,right_slope=0.f;
        if(!source_corridor_progress_slopes(trajectory,trajectory_count,
                lap_length,nominal_progress[horizon],
                previous_progress?previous_progress[horizon]:NAN,
                configuration,
                &left_slope,&right_slope))return 0;
        auto &polygon=problem->state_polygons[RICCATI_HORIZON];
        for(int face=0;face<polygon.count;++face)
            polygon.progress_normal[face]=
                -(face%2==0?left_slope:right_slope);
        if(left_slope!=0.f || right_slope!=0.f) {
            problem->terminal_x_lb[MPC_OPT_IDX_EY]=-BIG_BOUND+1.f;
            problem->terminal_x_ub[MPC_OPT_IDX_EY]= BIG_BOUND-1.f;
            problem->terminal_x_lb[MPC_OPT_IDX_PROGRESS_DEVIATION]=-BIG_BOUND+1.f;
            problem->terminal_x_ub[MPC_OPT_IDX_PROGRESS_DEVIATION]= BIG_BOUND-1.f;
        }
    }
    add_footprint_reserve_merit(problem->terminal_Q,problem->terminal_q,
        nullptr,&problem->state_polygons[RICCATI_HORIZON],
        &nominal_states[horizon].plant,configuration);
    if(!add_source_front_lateral_slip_merit(
           problem->terminal_Q,problem->terminal_q,nullptr,
           &nominal_states[horizon].plant,
           nominal_optimizer_states[horizon],configuration,
           configuration->terminal_multiplier))return 0;
    // Transform into the solver's actual incremental-command coordinates
    // BEFORE checking floating-point convexity. The coordinate transform
    // contains sums of large PSD source-slip GN products; independent
    // binary32 rounding can reintroduce negative curvature even when the
    // absolute-command Hessian was PSD. The earlier pre-transform check was
    // therefore not a certificate of the actual QP sent to ADMM.
    if(configuration->command_increment_qp_enabled &&
       !mpc_rti_transform_qp_to_command_increments(
            problem,configuration,prediction_dt))return 0;
    for(int k=0;k<RICCATI_HORIZON;++k) {
        const int model_stage=k+MPC_COMMAND_DELAY_STEPS;
        float nominal_solver_input[MPC_RTI_NU]={
            nominal_controls[model_stage].steering_target_rad,
            nominal_controls[model_stage].throttle_command
        };
        if(configuration->command_increment_qp_enabled) {
            nominal_solver_input[0]-=nominal_optimizer_states[model_stage][
                mpc_rti_steering_qp_state_index(configuration)];
            nominal_solver_input[1]-=nominal_optimizer_states[model_stage][
                MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
        }
        if(!mpc_rti_convexify_float_source_hessian(
            &problem->steps[k],
            nominal_optimizer_states[model_stage],nominal_solver_input,
            nullptr)) {
            problem->linearization_failure_stage=k;
            problem->linearization_failure_reason=7; // Nonconvex QP.
            problem->linearization_failure_index=-1;
            return 0;
        }
    }
    return 1;
}



int mpc_rti_build_ltv_qp_with_objective_schedule(
    const MpcRtiState_t nominal_states[PREDICTION_HORIZON+1],
    const MpcModelControl_t nominal_controls[PREDICTION_HORIZON],
    const MpcRtiReference_t references[PREDICTION_HORIZON+1],
    int horizon,float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    const MpcRtiObjectiveSchedule_t *objective_schedule,
    MpcRtiProblem_t *problem)
{
    return mpc_rti_build_ltv_qp_with_progress_corridor(
        nominal_states,nominal_controls,references,horizon,prediction_dt,
        configuration,objective_schedule,problem,nullptr,nullptr,0,0.);
}

int mpc_rti_build_ltv_qp(
    const MpcRtiState_t nominal_states[PREDICTION_HORIZON + 1],
    const MpcModelControl_t nominal_controls[PREDICTION_HORIZON],
    const MpcRtiReference_t references[PREDICTION_HORIZON + 1],
    int horizon,
    float prediction_dt,
    const MpcRtiConfiguration_t *configuration,
    MpcRtiProblem_t *problem)
{
    return mpc_rti_build_ltv_qp_with_objective_schedule(
        nominal_states, nominal_controls, references, horizon,
        prediction_dt, configuration, nullptr, problem);
}
