#include "mpc_rti_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

namespace {
constexpr int RECOURSE_STEPS=8;
constexpr int PHYSICAL=MPC_OPT_PHYSICAL_NX;
constexpr int NX=MPC_RTI_NX;
int source_channel(int i) {return i<5?i:(i==5?7:i+4);}

/* Source-exact finite-horizon recourse merit. The *original* path sampler
 * and full 33-state/actuator-queue source integrator are used; no bicycle
 * approximation, coordinate-frozen width, or fake feasible fallback.
 *
 * This continuous 16-variable optimization is a numerical successor search,
 * NOT a terminal controlled-invariant certificate. It is optional until
 * failure corpora and Unity races justify promotion. */
struct SourceTailMerit {
    double value{INFINITY};
    double hard_violation_m2{INFINITY};
    double minimum_slack_m{-INFINITY};
    double progress_m{0.};
    int valid{};
};

void project_source_tail_controls(
    MpcModelControl_t *controls, int n,MpcModelControl_t previous,
    const MpcRtiConfiguration_t &model)
{
    for(int k=0;k<n;++k) {
        auto &u=controls[k];
        u.steering_target_rad=std::clamp(u.steering_target_rad,
            -model.max_steering_rad,model.max_steering_rad);
        u.throttle_command=std::clamp(u.throttle_command,
            MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
        if(model.command_increment_qp_enabled) {
            u.steering_target_rad=std::clamp(u.steering_target_rad,
                previous.steering_target_rad-model.max_steering_command_step_rad,
                previous.steering_target_rad+model.max_steering_command_step_rad);
            u.throttle_command=std::clamp(u.throttle_command,
                previous.throttle_command-model.max_throttle_command_step,
                previous.throttle_command+model.max_throttle_command_step);
        }
        previous=u;
    }
}

/* Source-derived 16/32-variable control gradient of the SAME nonlinear tail
 * merit used for physical trial acceptance. Only footprint geometry is
 * differentiated with small central differences. Tire/contact/actuator and
 * path-frame motion sensitivities are native source A/B for each already
 * integrated stage; no 2*n*2 whole-tail finite-difference rollouts.
 *
 * This is a local derivative at the exact source branch. Candidate steps
 * remain independently reintegrated and strictly validated: AD does not
 * pretend to describe a large finite step across a tire/contact switch. */
int source_tail_control_gradient(
    const MpcRtiNominal_t *plan,double initial_progress,
    const MpcModelControl_t anchor[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *path,size_t count,double lap,
    float dt,int n,const MpcRtiConfiguration_t *model,
    double gradient[MPC_SUCCESSOR_PROBE_HORIZON][2])
{
    if(!plan || !plan->valid || !model || !gradient ||
       !path || n<1 || n>MPC_SUCCESSOR_PROBE_HORIZON)return 0;
    constexpr int columns=2*MPC_SUCCESSOR_PROBE_HORIZON;
    double sensitivity[PHYSICAL][columns]{};
    double station[columns]{};
    for(int k=0;k<n;++k) {
        const auto &incoming=plan->states[k+1];
        if(!incoming.incoming_tracking_jacobian_valid ||
           !incoming.incoming_transition_valid)return 0;
        MpcStageResult_t known{};
        known.next=incoming.plant;
        known.valid=incoming.incoming_transition_valid;
        known.branch_flags=incoming.incoming_branch_flags;
        known.body_forward_delta_m=incoming.incoming_body_forward_delta_m;
        known.body_lateral_delta_m=incoming.incoming_body_lateral_delta_m;
        known.body_yaw_delta_rad=incoming.incoming_body_yaw_delta_rad;
        for(int w=0;w<4;++w)
            known.output.maximum_longitudinal_slip_excess_mps[w]=
                incoming.incoming_wheel_slip_excess_mps[w];
        MpcStageLinearization_t lin{};
        if(!mpc_model_linearize_sensitivities_from_known_step(
            &plan->states[k].plant,&plan->controls[k],dt,0.f,&known,&lin))
            return 0;
        const auto &geom=incoming.incoming_tracking_jacobian;
        double next[PHYSICAL][columns]{};
        double next_station[columns]{};
        for(int i=0;i<PHYSICAL;++i) {
            for(int j=0;j<PHYSICAL;++j) {
                double a=lin.A[source_channel(i)][source_channel(j)];
                if(i<2) {
                    a=j<2?geom[i][j]:0.;
                    for(int m=0;m<3;++m)
                        a+=geom[i][m+2]*
                            lin.body_motion_A[m][source_channel(j)];
                }
                for(int c=0;c<2*n;++c)
                    next[i][c]+=a*sensitivity[j][c];
            }
            if(i<2)
                for(int c=0;c<2*n;++c)
                    next[i][c]+=geom[i][5]*station[c];
            for(int input=0;input<2;++input) {
                double b=lin.B[source_channel(i)][input];
                if(i<2) {
                    b=0.;
                    for(int m=0;m<3;++m)
                        b+=geom[i][m+2]*lin.body_motion_B[m][input];
                }
                next[i][2*k+input]+=b;
            }
        }
        for(int c=0;c<2*n;++c)
            next_station[c]=geom[2][5]*station[c];
        for(int j=0;j<PHYSICAL;++j) {
            double a=j<2?geom[2][j]:0.;
            for(int m=0;m<3;++m)
                a+=geom[2][m+2]*lin.body_motion_A[m][source_channel(j)];
            for(int c=0;c<2*n;++c)
                next_station[c]+=a*sensitivity[j][c];
        }
        for(int input=0;input<2;++input)
            for(int m=0;m<3;++m)
                next_station[2*k+input]+=
                    geom[2][m+2]*lin.body_motion_B[m][input];
        std::memcpy(sensitivity,next,sizeof(sensitivity));
        std::memcpy(station,next_station,sizeof(station));

        const auto &x=incoming.plant;
        const double progress=plan->progress[k+1];
        auto slack=[&](const MpcModelState_t &state,double p,
                       double out[2]) {
            MpcRtiReference_t ref{};
            float right=0.f,left=0.f;
            if(!mpc_rti_reference_at_progress(path,count,lap,p,model,&ref) ||
               !mpc_rti_compute_corridor_slack(&state,
                    ref.left_bound,ref.right_bound,model,
                    PREDICTION_HORIZON,&right,&left))return false;
            out[0]=left;out[1]=right;
            return std::isfinite(out[0]) && std::isfinite(out[1]);
        };
        double center[2]{};
        if(!slack(x,progress,center))return 0;
        double partial[2][3]{};
        for(int axis=0;axis<3;++axis) {
            constexpr double ey_eps=1.e-4;
            constexpr double station_eps=.002;
            const double h=axis==2?station_eps:ey_eps;
            MpcModelState_t plus=x,minus=x;
            double pplus=progress,pminus=progress;
            if(axis==0){plus.e_y+=h;minus.e_y-=h;}
            else if(axis==1){plus.e_psi+=h;minus.e_psi-=h;}
            else {pplus+=h;pminus-=h;}
            double upper[2]{},lower[2]{};
            if(!slack(plus,pplus,upper) ||
               !slack(minus,pminus,lower))return 0;
            for(int side=0;side<2;++side)
                partial[side][axis]=(upper[side]-lower[side])/(2.*h);
        }
        for(int side=0;side<2;++side) {
            const double s=center[side];
            const double derivative=2.e6*std::min(0.,s)-
                2.e4*std::max(0.,.03-s);
            if(derivative==0.)continue;
            for(int c=0;c<2*n;++c) {
                const double dslack=partial[side][0]*sensitivity[0][c]+
                    partial[side][1]*sensitivity[1][c]+
                    partial[side][2]*station[c];
                gradient[c/2][c%2]+=derivative*dslack;
            }
        }
    }
    const double travel=plan->progress[n]-initial_progress;
    if(travel<.02)
        for(int c=0;c<2*n;++c)
            gradient[c/2][c%2]-=400.*(.02-travel)*station[c];
    const auto &origin=plan->states[0].plant;
    MpcModelControl_t previous{
        origin.delayed_steering_command_2,
        origin.delayed_throttle_command_2};
    for(int k=0;k<n;++k) {
        const auto &u=plan->controls[k];
        gradient[k][0]+=20.*(u.steering_target_rad-
                            anchor[k].steering_target_rad);
        gradient[k][1]+=4.*(u.throttle_command-
                           anchor[k].throttle_command);
        const double steer_slew=u.steering_target_rad-
                                previous.steering_target_rad;
        const double throttle_slew=u.throttle_command-
                                   previous.throttle_command;
        gradient[k][0]+=4.*steer_slew;
        gradient[k][1]+=.4*throttle_slew;
        if(k>0) {
            gradient[k-1][0]-=4.*steer_slew;
            gradient[k-1][1]-=.4*throttle_slew;
        }
        previous=u;
    }
    for(int k=0;k<n;++k)
        for(double g:gradient[k])
            if(!std::isfinite(g))return 0;
    return 1;
}

int source_tail_merit(
    const MpcRtiState_t *initial, double station,
    const MpcModelControl_t commands[PREDICTION_HORIZON],
    const MpcModelControl_t anchor[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *path,size_t count,double lap,
    float dt,int n,const MpcRtiConfiguration_t *model,
    SourceTailMerit *merit,
    double gradient[MPC_SUCCESSOR_PROBE_HORIZON][2]=nullptr)
{
    if(!merit || !initial || !commands || !anchor || !model)return 0;
    *merit=SourceTailMerit{};
    auto plan=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
    if(!mpc_rti_build_nominal_same_epoch(initial,station,nullptr,commands,
        n,path,count,lap,dt,n,model,plan.get(),refs))return 0;
    double hard=0.,reserve=0.,minimum=INFINITY,control=0.;
    constexpr double reserve_distance=.03;
    const double distance=plan->progress[n]-station;
    if(!std::isfinite(distance))return 0;
    for(int k=1;k<=n;++k) {
        MpcRtiReference_t ref{};
        if(!mpc_rti_reference_at_progress(path,count,lap,
            plan->progress[k],model,&ref))return 0;
        float right=0.f,left=0.f;
        // Use terminal-stage margin across all future stages: the first
        // prediction special margin cannot understate the terminal reserve.
        if(!mpc_rti_compute_corridor_slack(&plan->states[k].plant,
            ref.left_bound,ref.right_bound,model,PREDICTION_HORIZON,
            &right,&left))return 0;
        minimum=std::min({minimum,double(left),double(right)});
        for(double s:{double(left),double(right)}) {
            hard+=std::pow(std::max(0.,-s),2);
            reserve+=std::pow(std::max(0.,reserve_distance-s),2);
        }
    }
    MpcModelControl_t previous{
        initial->plant.delayed_steering_command_2,
        initial->plant.delayed_throttle_command_2};
    for(int k=0;k<n;++k) {
        const double ds=commands[k].steering_target_rad-
            anchor[k].steering_target_rad;
        const double dtc=commands[k].throttle_command-
            anchor[k].throttle_command;
        const double slew_s=commands[k].steering_target_rad-
            previous.steering_target_rad;
        const double slew_t=commands[k].throttle_command-
            previous.throttle_command;
        control+=ds*ds + .2*dtc*dtc + .2*slew_s*slew_s + .02*slew_t*slew_t;
        previous=commands[k];
    }
    // Hard constraints dominate the differentiable reserve. A modest
    // command-change term ensures the terminal control search stays a
    // continuation rather than a separate aggressive racing controller.
    merit->value=1.e6*hard+1.e4*reserve+10.*control+
        200.*std::pow(std::max(0.,.02-distance),2);
    merit->hard_violation_m2=hard;
    merit->minimum_slack_m=minimum;
    merit->progress_m=distance;
    merit->valid=std::isfinite(merit->value);
    if(!merit->valid)return 0;
    if(gradient) {
        for(int k=0;k<n;++k)gradient[k][0]=gradient[k][1]=0.;
        return source_tail_control_gradient(plan.get(),station,anchor,
            path,count,lap,dt,n,model,gradient);
    }
    return 1;
}

int optimize_source_tail(
    const MpcRtiState_t *initial, double station,
    MpcModelControl_t seed[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *path,size_t count,double lap,float dt,int n,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiSuccessorProbe_t *result)
{
    if(!initial || !seed || !configuration || !result)return 0;
    const auto &model=configuration->model;
    MpcModelControl_t current[PREDICTION_HORIZON]{};
    MpcModelControl_t anchor[PREDICTION_HORIZON]{};
    std::copy_n(seed,n,current);
    std::copy_n(seed,n,anchor);
    SourceTailMerit base{};
    ++result->candidates_evaluated;
    if(!source_tail_merit(initial,station,current,anchor,path,count,
                          lap,dt,n,&model,&base))return 0;
    MpcRtiCycleConfiguration_t validation=*configuration;
    validation.terminal_viability=0;
    validation.objective_schedule=nullptr;
    validation.qp_capture=nullptr;
    validation.qp_capture_second_pass=nullptr;
    validation.capture_diagnostic_outputs=0;
    auto checked=std::make_unique<MpcRtiNominal_t>();
    // An exact source recourse optimizer is part of the optional normal SQP
    // merit, NOT the command publication path. In a 40 Hz callback, leave
    // 14ms for QP construction, ADMM and exact source acceptance. Offline
    // replay (no absolute deadline) uses the full continuous search without
    // limiting iterations or changing the physical objective.
    const double optimization_stop_us=
        configuration->absolute_deadline_monotonic_us>0. ?
        configuration->absolute_deadline_monotonic_us-14000. : INFINITY;
    const auto budget_available=[&]() {
        return !std::isfinite(optimization_stop_us) ||
            mpc_rti_monotonic_time_us()+900.<optimization_stop_us;
    };
    const auto maybe_accept=[&](const MpcModelControl_t *trial,
                                const SourceTailMerit &merit) {
        if(merit.minimum_slack_m < 0. ||
           (initial->plant.u>.5f && merit.progress_m<.02))return;
        int failure=-1;
        if(mpc_rti_validate_candidate_plan(initial,station,trial,
            path,count,lap,n,dt,&validation,checked.get(),
            &failure)!=MPC_RTI_ROLLOUT_OK)return;
        ++result->feasible_candidates;
        const auto &terminal=checked->states[n].plant;
        const double clearance=checked->validation_minimum_corridor_slack_m;
        const double travel=checked->progress[n]-station;
        const double score=3.*clearance+.3*travel-
            .15*std::abs(terminal.e_psi)-.07*std::abs(terminal.v)-
            .025*std::abs(terminal.r);
        if(result->feasible && score<=result->best_value)return;
        result->feasible=1;
        result->minimum_clearance_m=clearance;
        result->progress_m=travel;
        result->terminal_heading_error_rad=terminal.e_psi;
        result->terminal_lateral_speed_mps=terminal.v;
        result->best_value=score;
        std::copy_n(trial,n,result->controls);
    };
    maybe_accept(current,base);
    for(int iteration=0;iteration<8;++iteration) {
        if(!budget_available())break;
        // Already well clear and certifiably continuable: no benefit in
        // spending extra source evaluations or modifying the fixed policy.
        if(result->feasible && result->minimum_clearance_m>=.03f)break;
        double gradient[MPC_SUCCESSOR_PROBE_HORIZON][2]{};
        SourceTailMerit differentiated{};
        ++result->candidates_evaluated;
        if(!source_tail_merit(initial,station,current,anchor,
            path,count,lap,dt,n,&model,&differentiated,gradient))break;
        double squared_norm[2]{};
        for(int k=0;k<n;++k)for(int a=0;a<2;++a)
            squared_norm[a]+=gradient[k][a]*gradient[k][a];
        const double normalizer[2]={std::sqrt(squared_norm[0]),
                                    std::sqrt(squared_norm[1])};
        if(normalizer[0]<1.e-9 && normalizer[1]<1.e-9)break;
        MpcModelControl_t best_trial[PREDICTION_HORIZON]{};
        SourceTailMerit best=base;
        int improved=0;
        for(double scale:{.5,1.,2.,4.,.25,.125}) {
            if(!budget_available())break;
            MpcModelControl_t trial[PREDICTION_HORIZON]{};
            std::copy_n(current,n,trial);
            for(int k=0;k<n;++k) {
                trial[k].steering_target_rad-=
                    float(scale*.2*gradient[k][0]/
                        std::max(1.e-9,normalizer[0]));
                trial[k].throttle_command-=
                    float(scale*.35*gradient[k][1]/
                        std::max(1.e-9,normalizer[1]));
            }
            project_source_tail_controls(trial,n,{
                initial->plant.delayed_steering_command_2,
                initial->plant.delayed_throttle_command_2},model);
            SourceTailMerit candidate{};
            ++result->candidates_evaluated;
            if(!source_tail_merit(initial,station,trial,anchor,
                path,count,lap,dt,n,&model,&candidate))continue;
            if(candidate.value>=best.value-1.e-8)continue;
            best=candidate;
            std::copy_n(trial,n,best_trial);
            improved=1;
        }
        if(!improved)break;
        base=best;
        std::copy_n(best_trial,n,current);
        maybe_accept(current,base);
    }
    // When no feasible control was found, provide the best finite source
    // policy for a SOFT terminal objective. The caller MUST NOT interpret
    // it as feasible; exact source validation is still mandatory.
    if(!result->feasible)std::copy_n(current,n,result->controls);
    return 1;
}
 } // namespace

/* Focused gradient-parity regression entrypoint for the exact source recourse
 * merit. No solver state, control queue or controller output is modified. */
int mpc_rti_source_successor_merit_gradient(
    const MpcRtiState_t *terminal,double progress,
    const MpcModelControl_t controls[PREDICTION_HORIZON],
    const MpcModelControl_t anchor[PREDICTION_HORIZON],
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    float dt,int horizon,const MpcRtiConfiguration_t *model,
    double *merit,double gradient[MPC_SUCCESSOR_PROBE_HORIZON][2])
{
    SourceTailMerit evaluated{};
    if(!source_tail_merit(terminal,progress,controls,anchor,trajectory,
         count,lap,dt,horizon,model,&evaluated,gradient))return 0;
    if(merit)*merit=evaluated.value;
    return 1;
}

// The controlled source successor uses exactly the hidden plant/suspension
// state and queued actuator commands in terminal; it never reconstructs these
// states from a 5-state bicycle approximation.
int mpc_rti_source_successor_probe(
    const MpcRtiState_t *terminal, double progress,
    MpcModelControl_t previous_command,
    const MpcTrajectorySample_t *trajectory, size_t count, double lap,
    float dt, const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiSuccessorProbe_t *result, int steps,int candidate_limit)
{
    if (!result) return 0;
    *result = MpcRtiSuccessorProbe_t{};
    if (steps < 8 || steps > MPC_SUCCESSOR_PROBE_HORIZON ||
        (candidate_limit!=3 && candidate_limit!=9 && candidate_limit!=10))return 0;
    result->horizon=steps;
    if (!terminal || !trajectory || count < 3 || !configuration ||
        !std::isfinite(progress) || !std::isfinite(lap) || lap <= 0. ||
        !std::isfinite(dt) || dt <= 0.f ||
        !std::isfinite(previous_command.steering_target_rad) ||
        !std::isfinite(previous_command.throttle_command)) return 0;

    const int n = steps;
    const auto &model = configuration->model;
    MpcRtiCycleConfiguration_t validation = *configuration;
    validation.terminal_viability = 0;
    validation.objective_schedule = nullptr;
    validation.qp_capture = nullptr;
    validation.qp_capture_second_pass = nullptr;
    validation.capture_diagnostic_outputs = 0;

    // Estimate loaded-path feedforward at each successor stage. This is an
    // optimization *seed*; the nine options vary both physical input channels.
    float steer_ref[MPC_SUCCESSOR_PROBE_HORIZON]{};
    float throttle_ref[MPC_SUCCESSOR_PROBE_HORIZON]{};
    const double speed = std::max(.1, double(terminal->plant.u));
    for (int k=0;k<n;++k) {
        const double s = progress+(k+1)*double(dt)*speed;
        MpcTrajectorySample_t sample{};
        if (!mpc_trajectory_sample(trajectory,count,lap,s,&sample)) return 0;
        steer_ref[k] = std::clamp(
            float(sample.source_witness_valid ? sample.source_steering_target
                : vehicle_model_steering_for_curvature(float(sample.curvature))),
            -model.max_steering_rad,model.max_steering_rad);
        // A lower throttle seed is allowed because the source needs to be
        // able to slow down, not merely hold racing throttle until impact.
        throttle_ref[k] = std::clamp(
            sample.source_witness_valid ? float(sample.source_throttle) :
                previous_command.throttle_command,
            MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
    }

    static constexpr float blends[3] = {0.f,.5f,1.f};
    double best_value = -INFINITY;
    auto checked = std::make_unique<MpcRtiNominal_t>();
    // Three representative controllable policies are sufficient for normal
    // R1/R2 recourse policy seeding. The full nine-grid remains available to
    // offline forensics; it is not repeated on every live 40Hz callback.
    constexpr int sparse_choices[3]={0,4,8};
    MpcModelControl_t best_seed[PREDICTION_HORIZON]{};
    double best_seed_merit=INFINITY;
    int seed_valid=0;
    for(int sample=0;sample<(candidate_limit==9?9:3);++sample) {
            const int choice=candidate_limit==9?sample:sparse_choices[sample];
            const int steering_choice=choice/3;
            const int throttle_choice=choice%3;
            MpcModelControl_t controls[PREDICTION_HORIZON]{};
            float previous_steering=previous_command.steering_target_rad;
            float previous_throttle=previous_command.throttle_command;
            for(int k=0;k<n;++k) {
                const float interpolation=float(k+1)/n;
                const float desired_steering = previous_command.steering_target_rad+
                    blends[steering_choice]*interpolation*(steer_ref[k]-
                        previous_command.steering_target_rad);
                // Throttle choice 2 is a controlled slowdown branch, not a
                // full-brake fallback. It is an optimization variable in the
                // candidate successor search, never a published command.
                const float target_throttle=throttle_choice==2
                    ? MPC_THROTTLE_MIN_DRIVE : throttle_ref[k];
                const float desired_throttle = previous_command.throttle_command+
                    (throttle_choice==0 ? 0.f :
                     throttle_choice==1 ? .5f : 1.f)*interpolation*
                    (target_throttle-previous_command.throttle_command);
                auto &u=controls[k];
                u.steering_target_rad=std::clamp(desired_steering,
                    -model.max_steering_rad,model.max_steering_rad);
                u.throttle_command=std::clamp(desired_throttle,
                    MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
                if(model.command_increment_qp_enabled) {
                    u.steering_target_rad=std::clamp(u.steering_target_rad,
                        previous_steering-model.max_steering_command_step_rad,
                        previous_steering+model.max_steering_command_step_rad);
                    u.throttle_command=std::clamp(u.throttle_command,
                        previous_throttle-model.max_throttle_command_step,
                        previous_throttle+model.max_throttle_command_step);
                }
                previous_steering=u.steering_target_rad;
                previous_throttle=u.throttle_command;
            }
            if(candidate_limit==10) {
                MpcModelControl_t anchor[PREDICTION_HORIZON]{};
                std::copy_n(controls,n,anchor);
                SourceTailMerit evaluated{};
                if(source_tail_merit(terminal,progress,controls,anchor,
                    trajectory,count,lap,dt,n,&model,&evaluated) &&
                    evaluated.value<best_seed_merit) {
                    best_seed_merit=evaluated.value;
                    std::copy_n(controls,n,best_seed);
                    seed_valid=1;
                }
            }
            ++result->candidates_evaluated;
            int stage=-1;
            if(mpc_rti_validate_candidate_plan(terminal,progress,controls,
                    trajectory,count,lap,n,dt,&validation,checked.get(),
                    &stage)!=MPC_RTI_ROLLOUT_OK) continue;
            const double distance=checked->progress[n]-progress;
            if (terminal->plant.u>.5f && distance<.02) continue;
            const auto &last=checked->states[n].plant;
            const float clearance=checked->validation_minimum_corridor_slack_m;
            if(!std::isfinite(clearance) || clearance<0.f ||
                !std::isfinite(last.e_psi) || !std::isfinite(last.v) ||
                !std::isfinite(last.r)) continue;
            ++result->feasible_candidates;
            // Controlled terminal safety and progress, in physically
            // interpretable units. This is an offline feasibility quality
            // indicator, not yet a QP cost and not a terminal certificate.
            const double value=3.*clearance + .3*distance -
                .15*std::abs(last.e_psi) - .07*std::abs(last.v) -
                .025*std::abs(last.r);
            if(value <= best_value) continue;
            best_value=value;
            result->feasible=1;
            result->minimum_clearance_m=clearance;
            result->progress_m=distance;
            result->terminal_heading_error_rad=last.e_psi;
            result->terminal_lateral_speed_mps=last.v;
            result->best_value=value;
            std::copy_n(controls,n,result->controls);
        }
    // The 3 exact-source seeds already provide a legal controlled successor
    // when clearance comfortably exceeds the terminal reserve. Continuous
    // re-optimization then solves NO additional feasibility problem, while
    // burning 10-30ms before the actual normal SQP can start. Optimize only
    // genuinely endangered terminal states; this is a mathematically safe
    // early exit because an accepted source-valid continuation already exists.
    if(candidate_limit==10 && seed_valid &&
       !(result->feasible && result->minimum_clearance_m >=
           std::max(.06f,model.near_wall_tracking_reserve_m+.03f))) {
        (void)optimize_source_tail(terminal,progress,best_seed,
            trajectory,count,lap,dt,n,configuration,result);
    }
    return 1;
}

namespace {
// Physical 20-channel optimizer layout is a sparse slice of the full source
// state; source_channel is shared by normal tail sensitivity and its control
// optimizer, so neither uses a reduced bicycle approximation.

bool frozen_recourse_tail(const MpcRtiNominal_t *plan,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model,const MpcRtiObjectiveSchedule_t *schedule,
    MpcRtiNominal_t *tail,MpcRtiReference_t refs[RECOURSE_STEPS+1],
    bool active_control_derivative[RECOURSE_STEPS][2])
{
    if(!plan || !plan->valid || plan->horizon!=PREDICTION_HORIZON ||
       !model || !schedule || !schedule->successor_recourse_valid ||
       schedule->successor_recourse_horizon!=RECOURSE_STEPS || !tail ||
       !refs || !trajectory || !std::isfinite(lap) || lap<=0.)return false;
    MpcModelControl_t commands[PREDICTION_HORIZON]{};
    const auto previous=plan->controls[PREDICTION_HORIZON-1];
    double last[2]={previous.steering_target_rad,
                    previous.throttle_command};
    bool last_derivative[2]={true,true};
    const double physical_min[2]={-model->max_steering_rad,
                                   MPC_THROTTLE_MIN_DRIVE};
    const double physical_max[2]={model->max_steering_rad,
                                   MPC_THROTTLE_MAX};
    const double rate_limit[2]={model->max_steering_command_step_rad,
                                 model->max_throttle_command_step};
    for(int k=0;k<RECOURSE_STEPS;++k) {
        const double desired[2]={
            double(previous.steering_target_rad)+
                schedule->successor_recourse_deltas[k].steering_target_rad,
            double(previous.throttle_command)+
                schedule->successor_recourse_deltas[k].throttle_command};
        double next[2]{};
        bool derivative[2]{};
        for(int a=0;a<2;++a) {
            double lower=physical_min[a],upper=physical_max[a];
            if(model->command_increment_qp_enabled) {
                if(!(rate_limit[a]>0.))return false;
                lower=std::max(lower,last[a]-rate_limit[a]);
                upper=std::min(upper,last[a]+rate_limit[a]);
            }
            if(lower>upper)return false;
            next[a]=std::clamp(desired[a],lower,upper);
            if(desired[a]>lower && desired[a]<upper) {
                derivative[a]=true;
            } else if(desired[a]<=lower && lower>physical_min[a]) {
                // A rate-saturated command follows the previous physical
                // command; chain its terminal-input sensitivity exactly.
                derivative[a]=last_derivative[a];
            } else if(desired[a]>=upper && upper<physical_max[a]) {
                derivative[a]=last_derivative[a];
            } else {
                // Saturation at a true physical actuator bound has zero
                // local derivative with respect to terminal command.
                derivative[a]=false;
            }
            if(active_control_derivative)
                active_control_derivative[k][a]=derivative[a];
        }
        commands[k].steering_target_rad=float(next[0]);
        commands[k].throttle_command=float(next[1]);
        last[0]=next[0];last[1]=next[1];
        last_derivative[0]=derivative[0];
        last_derivative[1]=derivative[1];
    }
    return mpc_rti_build_nominal_same_epoch(
        &plan->states[PREDICTION_HORIZON],
        plan->progress[PREDICTION_HORIZON],nullptr,commands,
        RECOURSE_STEPS,trajectory,count,lap,
        schedule->successor_recourse_dt,RECOURSE_STEPS,model,tail,refs);
}

bool stage_slack(const MpcModelState_t &plant,double progress,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model,int stage,double &left,double &right)
{
    MpcRtiReference_t ref{};
    float l=0.f,r=0.f;
    if(!mpc_rti_reference_at_progress(trajectory,count,lap,progress,model,&ref) ||
       !mpc_rti_compute_corridor_slack(&plant,ref.left_bound,ref.right_bound,
            model,stage,&l,&r))return false;
    left=l;right=r;
    return std::isfinite(left) && std::isfinite(right);
}

// A source-exact scored tail with a frozen, terminal-command-relative policy.
// The ORIGINAL hard 750ms constraints are unchanged; the future controlled
// corridor deficit is a soft extension of that same immutable SQP merit.
int recourse_cost_and_gradient(const MpcRtiNominal_t *plan,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model,const MpcRtiObjectiveSchedule_t *schedule,
    double *cost,double gradient[NX],double majorizer[NX])
{
    if(!cost || !plan || !model || !schedule)return 0;
    *cost=0.;
    if(!schedule->successor_recourse_valid ||
       schedule->successor_recourse_weight<=0.f)return 1;
    if(gradient)std::fill_n(gradient,NX,0.);
    if(majorizer)std::fill_n(majorizer,NX,0.);
    // One full source integration serves BOTH exact cost and local QP.
    auto tail=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[RECOURSE_STEPS+1]{};
    bool active[RECOURSE_STEPS][2]{};
    if(!frozen_recourse_tail(plan,trajectory,count,lap,model,schedule,
            tail.get(),refs,active))return 0;
    /* First score the eight exact-source tail states WITHOUT constructing
     * any expensive source sensitivities. When every future footprint has
     * sufficient reserve, the merit and ALL 23 first derivatives/Hessians
     * are identically zero: skip the 8 native tire/actuator Jacobians. */
    const bool differentiate=gradient && majorizer;
    double deficit_at_stage[RECOURSE_STEPS][2]{};
    int active_deficits=0;
    const double weight=schedule->successor_recourse_weight;
    for(int k=0;k<RECOURSE_STEPS;++k) {
        const auto &source=tail->states[k+1].plant;
        double left=0.,right=0.;
        if(!stage_slack(source,tail->progress[k+1],trajectory,count,lap,
                model,PREDICTION_HORIZON,left,right))return 0;
        for(int side=0;side<2;++side) {
            const double slack=side==0?left:right;
            const double deficit=std::max(0.,double(
                schedule->successor_recourse_reserve_m)-slack);
            deficit_at_stage[k][side]=deficit;
            *cost+=weight*deficit*deficit;
            active_deficits+=deficit>0.;
        }
    }
    if(!differentiate || active_deficits==0)return std::isfinite(*cost);
    double sensitivity[PHYSICAL][NX]{};
    double station[NX]{};
    if(differentiate) {
        for(int i=0;i<PHYSICAL;++i)sensitivity[i][i]=1.;
        station[MPC_OPT_IDX_PROGRESS_DEVIATION]=1.;
    }
    const double scales[NX]={.1,.1,1.,1.,1.,.1,100.,100.,100.,100.,
        .01,.01,.01,.01,.01,.01,.01,.01,.01,.01,1.,.1,.1};
    constexpr double eps=1.e-4;
    for(int k=0;k<RECOURSE_STEPS;++k) {
        const auto &incoming=tail->states[k+1];
        if(differentiate) {
            if(!incoming.incoming_tracking_jacobian_valid)return 0;
            MpcStageResult_t known{};
            known.next=incoming.plant;
            known.valid=incoming.incoming_transition_valid;
            known.branch_flags=incoming.incoming_branch_flags;
            known.body_forward_delta_m=incoming.incoming_body_forward_delta_m;
            known.body_lateral_delta_m=incoming.incoming_body_lateral_delta_m;
            known.body_yaw_delta_rad=incoming.incoming_body_yaw_delta_rad;
            for(int w=0;w<4;++w)
                known.output.maximum_longitudinal_slip_excess_mps[w]=
                    incoming.incoming_wheel_slip_excess_mps[w];
            MpcStageLinearization_t lin{};
            if(!mpc_model_linearize_sensitivities_from_known_step(
                 &tail->states[k].plant,&tail->controls[k],
                 schedule->successor_recourse_dt,0.f,&known,&lin))return 0;
            const auto &geom=incoming.incoming_tracking_jacobian;
            double next[PHYSICAL][NX]{};
            double next_station[NX]{};
            for(int i=0;i<PHYSICAL;++i) {
                for(int j=0;j<PHYSICAL;++j) {
                    double a=lin.A[source_channel(i)][source_channel(j)];
                    if(i<2) {
                        a=j<2?geom[i][j]:0.;
                        for(int m=0;m<3;++m)
                            a+=geom[i][m+2]*
                                lin.body_motion_A[m][source_channel(j)];
                    }
                    for(int c=0;c<NX;++c)next[i][c]+=a*sensitivity[j][c];
                }
                if(i<2)for(int c=0;c<NX;++c)
                    next[i][c]+=geom[i][5]*station[c];
                for(int a=0;a<2;++a)if(active[k][a]) {
                    double b=lin.B[source_channel(i)][a];
                    if(i<2) {
                        b=0.;
                        for(int m=0;m<3;++m)
                            b+=geom[i][m+2]*lin.body_motion_B[m][a];
                    }
                    next[i][MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND+a]+=b;
                }
            }
            for(int c=0;c<NX;++c)next_station[c]=geom[2][5]*station[c];
            for(int j=0;j<PHYSICAL;++j) {
                double a=j<2?geom[2][j]:0.;
                for(int m=0;m<3;++m)
                    a+=geom[2][m+2]*
                        lin.body_motion_A[m][source_channel(j)];
                for(int c=0;c<NX;++c)next_station[c]+=a*sensitivity[j][c];
            }
            for(int a=0;a<2;++a)if(active[k][a])
                for(int m=0;m<3;++m)
                    next_station[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND+a]+=
                        geom[2][m+2]*lin.body_motion_B[m][a];
            std::memcpy(sensitivity,next,sizeof(sensitivity));
            std::memcpy(station,next_station,sizeof(station));
        }
        const auto &state=incoming.plant;
        const double progress=tail->progress[k+1];
        for(int side=0;side<2;++side) {
            const double deficit=deficit_at_stage[k][side];
            if(deficit==0.)continue;
            // Differentiate ONLY the 2D exact footprint/slack sampler, not
            // the source plant; source A/B propagates all hidden slip and
            // actuator dynamics at this exact nonlinear branch.
            double partial[3]{};
            for(int axis=0;axis<3;++axis) {
                auto plus=state,minus=state;
                double pplus=progress,pminus=progress;
                if(axis==0){plus.e_y+=eps;minus.e_y-=eps;}
                if(axis==1){plus.e_psi+=eps;minus.e_psi-=eps;}
                if(axis==2){pplus+=eps;pminus-=eps;}
                double lplus=0.,rplus=0.,lminus=0.,rminus=0.;
                if(!stage_slack(plus,pplus,trajectory,count,lap,model,
                        PREDICTION_HORIZON,lplus,rplus) ||
                   !stage_slack(minus,pminus,trajectory,count,lap,model,
                        PREDICTION_HORIZON,lminus,rminus))return 0;
                partial[axis]=-((side==0?lplus:rplus)-
                    (side==0?lminus:rminus))/(2.*eps);
            }
            double row[NX]{},norm=0.;
            for(int c=0;c<NX;++c) {
                row[c]=partial[0]*sensitivity[0][c]+
                    partial[1]*sensitivity[1][c]+partial[2]*station[c];
                norm+=std::abs(row[c])*scales[c];
            }
            for(int c=0;c<NX;++c) {
                gradient[c]+=2.*weight*deficit*row[c];
                // PSD diagonal GN majorizer; true first derivative retained.
                majorizer[c]+=2.*weight*std::abs(row[c])*norm/scales[c];
            }
        }
    }
    return std::isfinite(*cost);
}
} // namespace

int mpc_rti_initialize_successor_recourse(
    const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,float dt,
    const MpcRtiCycleConfiguration_t *configuration,
    MpcRtiObjectiveSchedule_t *schedule)
{
    if(!schedule || !configuration || !nominal || !nominal->valid ||
       nominal->horizon!=PREDICTION_HORIZON ||
       !std::isfinite(configuration->successor_recourse_weight) ||
       configuration->successor_recourse_weight<=0.f)return 0;
    schedule->successor_recourse_valid=0;
    const auto previous=nominal->controls[PREDICTION_HORIZON-1];
    MpcRtiSuccessorProbe_t sampled{};
    if(!mpc_rti_source_successor_probe(
         &nominal->states[PREDICTION_HORIZON],
         nominal->progress[PREDICTION_HORIZON],previous,trajectory,count,
         lap,dt,configuration,&sampled,RECOURSE_STEPS,
         configuration->successor_recourse_optimize?10:3))return 0;
    schedule->successor_recourse_horizon=RECOURSE_STEPS;
    schedule->successor_recourse_dt=dt;
    schedule->successor_recourse_weight=
        configuration->successor_recourse_weight;
    schedule->successor_recourse_reserve_m=std::max(.03f,
        configuration->model.near_wall_tracking_reserve_m);
    for(int k=0;k<RECOURSE_STEPS;++k) {
        const auto u=(sampled.feasible ||
            configuration->successor_recourse_optimize)
            ? sampled.controls[k] : previous;
        schedule->successor_recourse_deltas[k].steering_target_rad=
            u.steering_target_rad-previous.steering_target_rad;
        schedule->successor_recourse_deltas[k].throttle_command=
            u.throttle_command-previous.throttle_command;
    }
    schedule->successor_recourse_valid=1;
    return 1;
}

int mpc_rti_successor_recourse_cost(
    const MpcRtiNominal_t *plan,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model,
    const MpcRtiObjectiveSchedule_t *schedule,double *cost)
{
    if(recourse_cost_and_gradient(plan,trajectory,count,lap,model,
            schedule,cost,nullptr,nullptr))return 1;
    /* If candidate source dynamics cannot be extended by the frozen
     * controlled tail, assign finite soft recourse cost. The original 750ms
     * horizon retains its own strict physical gate: a *future* model/recourse
     * failure is NOT a collision and must not force full brake. */
    if(!cost || !schedule || !schedule->successor_recourse_valid)return 0;
    *cost=4.*schedule->successor_recourse_weight;
    return 1;
}

int mpc_rti_add_successor_recourse_qp(
    MpcRtiProblem_t *problem,const MpcRtiNominal_t *nominal,
    const MpcTrajectorySample_t *trajectory,size_t count,double lap,
    const MpcRtiConfiguration_t *model,
    const MpcRtiObjectiveSchedule_t *schedule)
{
    if(!problem || !nominal || !schedule)return 0;
    if(!schedule->successor_recourse_valid ||
       schedule->successor_recourse_weight<=0.f)return 1;
    double objective=0.,gradient[NX]{},hessian[NX]{};
    if(!recourse_cost_and_gradient(nominal,trajectory,count,lap,model,
            schedule,&objective,gradient,hessian))return 0;
    float state[NX]{};
    serialize_state(&nominal->states[PREDICTION_HORIZON],state);
    state[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]=
        nominal->controls[PREDICTION_HORIZON-1].steering_target_rad;
    state[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]=
        nominal->controls[PREDICTION_HORIZON-1].throttle_command;
    for(int i=0;i<NX;++i) {
        if(!std::isfinite(gradient[i]) || !std::isfinite(hessian[i]) ||
           hessian[i]>1.e9)return 0;
        problem->terminal_Q[i]+=(float)hessian[i];
        problem->terminal_q[i]+=(float)(gradient[i]-hessian[i]*state[i]);
    }
    return 1;
}
