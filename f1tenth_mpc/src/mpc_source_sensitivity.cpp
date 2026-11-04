#include "mpc_source_sensitivity.hpp"
#include "vehicle_model.h"
#include "unity_source_plant_c.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <memory>
#if defined(SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES) && defined(_OPENMP)
#include <omp.h>
#include <cstdlib>
#if defined(__linux__)
#include <sched.h>
#endif
#endif

namespace sdu_apex::source_sqp {
namespace {
using Clock=std::chrono::steady_clock;
double microseconds_since(Clock::time_point before) {
    return std::chrono::duration<double,std::micro>(
        Clock::now()-before).count();
}
double monotonic_us() {
    return std::chrono::duration<double,std::micro>(
        Clock::now().time_since_epoch()).count();
}
constexpr int N=PREDICTION_HORIZON;
constexpr int L=UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;
constexpr int X=UNITY_SOURCE_PLANT_STATE_SIZE;
constexpr int D=2*N;
struct GaussNewtonFeature {
    double ey{},epsi{},u{},v{},r{},steer_queue2{},rack{};
    double slip_excess[4]{};
};
using GaussNewtonTrace=std::array<std::array<GaussNewtonFeature,D>,N+1>;
// These are the *local* derivatives of exactly one 25 ms source interval.
// Each stage has the original 26 hidden/visible physical source coordinates
// and two physical control inputs, not the reduced Riccati state.
struct StageTransitionJacobian {
    double source[X][X+2]{};
    double body[3][X+2]{};
    double slip[4][X+2]{};
};
using StageJacobianTrace=std::array<StageTransitionJacobian,N>;


constexpr double kCom=0.15532;
constexpr double kCommandScale[2]={.15,.25};
static_assert(X==26 && L==20 && MPC_PLANT_INTERNAL_STATE_COUNT==22);

struct StateTangent {
    double source[X]{};
    double ey{},epsi{},progress{};
    double steer_queue1{},steer_queue2{},throttle_queue1{},throttle_queue2{};
    double slip_excess[4]{};
};
using Trace=std::array<std::array<StateTangent,L>,N+1>;

// The source's 22 physical memory channels have one-to-one coordinates in
// source[3..12,14..25]. The difference between v at the rear axle and
// source COM lateral velocity includes the same 0.15532 m lever arm as
// vehicle_model.cpp. No lost wheel or suspension coordinates.
void apply_state_tangent(MpcRtiState_t *dst,const StateTangent &v,double h) {
    auto &s=dst->plant;
    s.e_y+=static_cast<float>(h*v.ey);
    s.e_psi+=static_cast<float>(h*v.epsi);
    s.u+=static_cast<float>(h*v.source[UNITY_SOURCE_U]);
    s.v+=static_cast<float>(h*(v.source[UNITY_SOURCE_V]-
                                   kCom*v.source[UNITY_SOURCE_YAW_RATE]));
    s.r+=static_cast<float>(h*v.source[UNITY_SOURCE_YAW_RATE]);
    s.actual_steering_angle+=static_cast<float>(
        h*v.source[UNITY_SOURCE_PHYSICAL_STEERING]);
    s.delayed_steering_command_1+=static_cast<float>(h*v.steer_queue1);
    s.delayed_steering_command_2+=static_cast<float>(h*v.steer_queue2);
    s.delayed_throttle_command_1+=static_cast<float>(h*v.throttle_queue1);
    s.delayed_throttle_command_2+=static_cast<float>(h*v.throttle_queue2);
    for(int i=0;i<MPC_PLANT_INTERNAL_STATE_COUNT;++i) {
        const int source=i<4?3+i:i<10?3+i:i+4;
        s.source_plant_internal_state[i]+=h*v.source[source];
    }
    for(int wheel=0;wheel<4;++wheel)
        dst->incoming_wheel_slip_excess_mps[wheel]+=
            static_cast<float>(h*v.slip_excess[wheel]);
}

// Same objective, progress-dependent speed references and physical footprint
// checker as the finite-difference source oracle, but score tangent changes
// to the ALREADY exact-source-propagated nominal. No approximate wheel model.
bool scored_tangent(const MpcRtiNominal_t &nominal,
    const Trace &trace,int lane,int direction,int source_column,double h,
    const MpcTrajectorySample_t *track,std::size_t track_count,double length,
    const MpcRtiCycleConfiguration_t &cfg,MpcRtiNominal_t *candidate,
    double *objective,double margins[2*N],
    bool corridor_only_phase_one) {
    *candidate=nominal;
    // The frozen SQP objective owns its speed targets. The exact scorer
    // already interpolates ALL progress-dependent references when checking
    // the physical corridor. Do not redundantly interpolate each changed
    // tangent stage just to overwrite a speed that the frozen objective
    // ignores. Without a frozen schedule preserve the old full path.
    const bool frozen_objective=cfg.objective_schedule &&
        cfg.objective_schedule->valid;
    if(frozen_objective)candidate->speed_reference_valid=0;
    // The first altered command is applied at stage k; no physical
    // state/progress/speed reference at stage <=k can depend on it.
    // Skip the immutable prefix, including its map interpolation.
    for(int stage=source_column/2+1;stage<=N;++stage) {
        const auto &t=trace[stage][lane];
        const double eps=direction*h;
        apply_state_tangent(&candidate->states[stage],t,eps);
        candidate->progress[stage]+=eps*t.progress;
        if(!frozen_objective && stage>0) {
            MpcRtiReference_t reference{};
            if(!mpc_rti_reference_at_progress(track,track_count,length,
                    candidate->progress[stage],&cfg.model,&reference))
                return false;
            candidate->speed_reference[stage]=reference.u;
        }
    }
    const int control_stage=source_column/2;
    const int input=source_column%2;
    if(input==0)
        candidate->controls[control_stage].steering_target_rad+=
            static_cast<float>(direction*h*kCommandScale[0]);
    else
        candidate->controls[control_stage].throttle_command+=
            static_cast<float>(direction*h*kCommandScale[1]);
    if(corridor_only_phase_one) {
        // The objective is NOT a Phase-I QP objective. Evaluate the exact
        // source physical footprint and progress-dependent track width
        // directly, preserving the SAME reference interpolation and
        // mpc_rti_compute_corridor_slack used by the full scorer.
        // This is not a new plant, geometric approximation or relaxed
        // constraint; it avoids 120 redundant full racing-cost evaluations.
        *objective=0.;
        for(int stage=1;stage<=N;++stage) {
            MpcRtiReference_t reference{};
            if(!mpc_rti_reference_at_progress(
                track,track_count,length,candidate->progress[stage],
                &cfg.model,&reference))return false;
            float lower=NAN,upper=NAN;
            if(!mpc_rti_compute_corridor_slack(
                &candidate->states[stage].plant,
                reference.left_bound,reference.right_bound,
                &cfg.model,stage,&lower,&upper))return false;
            margins[2*(stage-1)]=lower;
            margins[2*(stage-1)+1]=upper;
        }
        return true;
    }
    float raw_margins[N+1][2]{};
    float minimum=NAN;
    if(!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
        candidate,track,track_count,length,&cfg,objective,&minimum,
        raw_margins))return false;
    for(int stage=1;stage<=N;++stage) {
        margins[2*(stage-1)]=raw_margins[stage][0];
        margins[2*(stage-1)+1]=raw_margins[stage][1];
    }
    return std::isfinite(*objective);
}
} // namespace

bool full_source_derivatives(
    const MpcRtiNominal_t *baseline,
    const MpcTrajectorySample_t *trajectory,std::size_t trajectory_count,
    double lap_length,float dt,const MpcRtiCycleConfiguration_t *cfg,
    int decision_count,double normalized_step,double baseline_cost,
    double absolute_deadline_us,FullSourceDerivatives *out,
    bool build_gauss_newton_hessian,
    bool use_stage_local_jacobians,
    bool compress_positive_drive_passive_timers,
    bool corridor_only_phase_one)
{
    if(!baseline||!baseline->valid||baseline->horizon!=N||!trajectory||
       !cfg||!out||!std::isfinite(baseline_cost)||
       !std::isfinite(normalized_step)||normalized_step<=0.||
       decision_count<1||decision_count>2*N)
        return false;
    // Each 20-column batch is causally independent of the others.
    // Their source rolls and tangent scoring can run on separate threads;
    // writes to output matrices are restricted to disjoint column ranges.
    // Workers own persistent scratch and never share nominal buffers.
    struct BatchMeasurement {
        double dual_plant_us{};
        double tangent_scoring_us{};
        int dual_plant_intervals{};
        int score_evaluations{};
        bool deadline_exhausted{};
    };
    // The outer call owns this scratch trace. Explicitly capture its raw
    // pointer for workers: a thread_local variable accessed IN a worker
    // would otherwise resolve to a different per-worker trace.
    static thread_local auto gn_storage=std::make_unique<GaussNewtonTrace>();
    auto *const gn_trace=gn_storage.get();
    if(build_gauss_newton_hessian) *gn_trace={};

    // The primal source transition at each stage is already in baseline.
    // Differentiate the ORIGINAL source model at those exact states into
    // local [A(26x26), B(26x2), motion, slip] matrices. All 30 stages
    // are independent and can be distributed across available CPU cores.
    // Causal multiplication below later assembles all 60 command Jacobians
    // with no 23-state surrogate or frozen physical dynamics.
    static thread_local auto local_storage=
        std::make_unique<StageJacobianTrace>();
    auto *const locals=local_storage.get();
    if(use_stage_local_jacobians) {
        std::array<double,N> worker_plant_us{};
        std::array<int,N> worker_plant_calls{};
        std::array<unsigned char,N> worker_ok{};
        // Determine worker availability without spawning oversubscribed
        // OpenMP teams on affinity-restricted control threads.
        int workers=1;
#if defined(_OPENMP) && defined(SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES)
#if defined(__linux__)
        cpu_set_t allowed_cpus;
        CPU_ZERO(&allowed_cpus);
        if(sched_getaffinity(0,sizeof(allowed_cpus),&allowed_cpus)==0)
            workers=std::min(6,CPU_COUNT(&allowed_cpus));
#endif
        if(std::getenv("OMP_PLACES")) {
            const int places=omp_get_num_places();
            if(places>0)workers=std::min(6,places);
        }
        if(cfg->model.world_sweep_check)workers=1;
#pragma omp parallel for schedule(dynamic,1) num_threads(workers) if(workers>1)
#endif
        for(int stage=0;stage<N;++stage) {
            bool ok=true;
            const auto &state=baseline->states[stage].plant;
            const auto &control=baseline->controls[stage];
            // In positive drive the source resets low-speed sticky-tire
            // histories BEFORE testing either sticky activation. Wheel
            // dynamics and the physical footprint are independent of
            // state[18..25]; all 26 source primal values are still
            // integrated and the timers still exist in the nominal.
            // Only zero-impact sensitivities are left out of the
            // optimization's forward tangent propagation.
            const bool compressed=
                compress_positive_drive_passive_timers &&
                control.throttle_command>0.f &&
                control.throttle_command<=MPC_THROTTLE_MAX;
            const int chunks=compressed?1:2;
            if(compressed) {
                auto &local=(*locals)[stage];
                // loc is persistent thread-local memory: prevent stale
                // inactive columns from another epoch reaching the QP.
                for(int row=0;row<X;++row)
                    for(int col=18;col<26;++col)
                        local.source[row][col]=0.;
                for(int row=0;row<3;++row)
                    for(int col=18;col<26;++col)
                        local.body[row][col]=0.;
                for(int row=0;row<4;++row)
                    for(int col=18;col<26;++col)
                        local.slip[row][col]=0.;
            }
            for(int chunk=0;chunk<chunks && ok;++chunk) {
                double state_seeds[X][L]{};
                double control_seeds[2][L]{};
                for(int lane=0;lane<L;++lane) {
                    const int coordinate=compressed
                        ?(lane<18?lane:X+(lane-18))
                        :chunk*L+lane;
                    if(coordinate<X)
                        state_seeds[coordinate][lane]=1.;
                    else if(coordinate<X+2)
                        control_seeds[coordinate-X][lane]=1.;
                }
                double next[X][L],body[3][L],slip[4][L];
                const auto start=Clock::now();
                ok=vehicle_model_full_source_interval_sensitivities(
                    &state,&control,dt,state_seeds,control_seeds,
                    next,body,slip);
                worker_plant_us[stage]+=microseconds_since(start);
                ++worker_plant_calls[stage];
                if(!ok)break;
                for(int lane=0;lane<L;++lane) {
                    const int coordinate=compressed
                        ?(lane<18?lane:X+(lane-18))
                        :chunk*L+lane;
                    if(coordinate>=X+2)break;
                    for(int row=0;row<X;++row)
                        (*locals)[stage].source[row][coordinate]=
                            next[row][lane];
                    for(int row=0;row<3;++row)
                        (*locals)[stage].body[row][coordinate]=
                            body[row][lane];
                    for(int row=0;row<4;++row)
                        (*locals)[stage].slip[row][coordinate]=
                            slip[row][lane];
                }
            }
            worker_ok[stage]=ok?1:0;
        }
        for(int stage=0;stage<N;++stage) {
            out->dual_plant_us+=worker_plant_us[stage];
            out->dual_plant_intervals+=worker_plant_calls[stage];
            if(!worker_ok[stage])return false;
        }
        if(absolute_deadline_us>0. &&
           monotonic_us()>=absolute_deadline_us) {
            out->deadline_exhausted=true;
            return false;
        }
    }

    const int batches=(decision_count+L-1)/L;
    std::array<BatchMeasurement,3> measurements{};
    std::array<unsigned char,3> succeeded{};

    const auto evaluate_batch=[&](int batch)->bool {
        static thread_local auto trace=std::make_unique<Trace>();
        static thread_local auto positive=std::make_unique<MpcRtiNominal_t>();
        static thread_local auto negative=std::make_unique<MpcRtiNominal_t>();
        auto &measurement=measurements[batch];

        // Every lane in this batch belongs to a command from stage
        // batch*(20/2) onward. Before that stage the exact source state
        // cannot depend on ANY variable in the batch, so zero is the
        // mathematically exact tangent and no source dual integration
        // is necessary. This preserves hidden-state and actuator causality.
        const int first_affected_stage=batch*(L/2);
        // Only the first current stage must be zero. Every later stage is
        // fully overwritten in the causal propagation, and all earlier
        // stages are excluded from scoring. Avoid zeroing ~200 kB/batch.
        (*trace)[first_affected_stage]={};
        for(int stage=first_affected_stage;stage<N;++stage) {
            if(absolute_deadline_us>0. &&
               monotonic_us()>=absolute_deadline_us) {
                measurement.deadline_exhausted=true;
                return false;
            }
            if(!baseline->states[stage+1].incoming_tracking_jacobian_valid)
                return false;
            const auto &g=baseline->states[stage+1].
                incoming_tracking_jacobian;
            double seeds[X][L];
            // Only the two active command lanes are assigned below; the
            // remaining 38 entries MUST remain zero. Never leave inactive
            // dual control seeds uninitialized.
            double control_seeds[2][L]{};
            double next[X][L],body[3][L],slip[4][L];
            for(int lane=0;lane<L;++lane) {
                const auto &t=(*trace)[stage][lane];
                for(int row=0;row<X;++row)seeds[row][lane]=t.source[row];
                const int coordinate=batch*L+lane;
                if(coordinate<decision_count &&
                   coordinate/2==stage)
                    control_seeds[coordinate%2][lane]=
                        kCommandScale[coordinate%2];
            }
            if(use_stage_local_jacobians) {
                const auto &local=(*locals)[stage];
                // Chain local source sensitivities through all 26 physical
                // states and both commands. Each of the 20 lanes still
                // represents the SAME independent 60-command derivative
                // as the original direct full-horizon source tangent.
                for(int lane=0;lane<L;++lane) {
                    for(int row=0;row<X+7;++row) {
                        double value=0.;
                        if(row<X) {
                            for(int col=0;col<X;++col)
                                value+=local.source[row][col]*
                                    seeds[col][lane];
                            value+=local.source[row][X]*
                                control_seeds[0][lane]+
                                local.source[row][X+1]*
                                control_seeds[1][lane];
                            next[row][lane]=value;
                        } else if(row<X+3) {
                            const int m=row-X;
                            for(int col=0;col<X;++col)
                                value+=local.body[m][col]*seeds[col][lane];
                            value+=local.body[m][X]*control_seeds[0][lane]+
                                local.body[m][X+1]*control_seeds[1][lane];
                            body[m][lane]=value;
                        } else {
                            const int wheel=row-X-3;
                            for(int col=0;col<X;++col)
                                value+=local.slip[wheel][col]*
                                    seeds[col][lane];
                            value+=local.slip[wheel][X]*
                                control_seeds[0][lane]+
                                local.slip[wheel][X+1]*
                                control_seeds[1][lane];
                            slip[wheel][lane]=value;
                        }
                    }
                }
            } else {
                const auto dual_start=Clock::now();
                if(!vehicle_model_full_source_interval_sensitivities(
                    &baseline->states[stage].plant,
                    &baseline->controls[stage],
                    dt,seeds,control_seeds,next,body,slip))
                    return false;
                measurement.dual_plant_us+=microseconds_since(dual_start);
                ++measurement.dual_plant_intervals;
            }
            for(int lane=0;lane<L;++lane) {
                const auto &previous=(*trace)[stage][lane];
                auto &following=(*trace)[stage+1][lane];
                for(int row=0;row<X;++row)
                    following.source[row]=next[row][lane];
                for(int wheel=0;wheel<4;++wheel)
                    following.slip_excess[wheel]=slip[wheel][lane];
                const double input[6]={previous.ey,previous.epsi,
                    body[0][lane],body[1][lane],body[2][lane],
                    previous.progress};
                double result[3]{};
                for(int row=0;row<3;++row)
                    for(int axis=0;axis<6;++axis)
                        result[row]+=g[row][axis]*input[axis];
                following.ey=result[0];
                following.epsi=result[1];
                following.progress=result[2];
                following.steer_queue1=previous.steer_queue2;
                following.throttle_queue1=previous.throttle_queue2;
                const int coordinate=batch*L+lane;
                following.steer_queue2=coordinate<decision_count &&
                    coordinate/2==stage && coordinate%2==0
                        ?kCommandScale[0]:0.;
                following.throttle_queue2=coordinate<decision_count &&
                    coordinate/2==stage && coordinate%2==1
                        ?kCommandScale[1]:0.;
                if(build_gauss_newton_hessian && coordinate<decision_count) {
                    auto &f=(*gn_trace)[stage+1][coordinate];
                    f.ey=following.ey;
                    f.epsi=following.epsi;
                    f.u=following.source[UNITY_SOURCE_U];
                    f.v=following.source[UNITY_SOURCE_V]-
                        kCom*following.source[UNITY_SOURCE_YAW_RATE];
                    f.r=following.source[UNITY_SOURCE_YAW_RATE];
                    f.steer_queue2=following.steer_queue2;
                    f.rack=following.source[UNITY_SOURCE_PHYSICAL_STEERING];
                    for(int w=0;w<4;++w)
                        f.slip_excess[w]=following.slip_excess[w];
                }
            }
        }
        for(int lane=0;lane<L;++lane) {
            const int j=batch*L+lane;
            if(j>=decision_count)break;
            if(absolute_deadline_us>0. &&
               monotonic_us()>=absolute_deadline_us) {
                measurement.deadline_exhausted=true;
                return false;
            }
            double cost_plus=NAN,cost_minus=NAN;
            double wall_plus[2*N]{},wall_minus[2*N]{};
            const auto score_start=Clock::now();
            if(!scored_tangent(*baseline,*trace,lane,+1,j,normalized_step,
                    trajectory,trajectory_count,lap_length,*cfg,
                    positive.get(),&cost_plus,wall_plus,
                    corridor_only_phase_one && !cfg->model.world_sweep_check) ||
               !scored_tangent(*baseline,*trace,lane,-1,j,normalized_step,
                    trajectory,trajectory_count,lap_length,*cfg,
                    negative.get(),&cost_minus,wall_minus,
                    corridor_only_phase_one && !cfg->model.world_sweep_check))
                return false;
            measurement.tangent_scoring_us+=microseconds_since(score_start);
            measurement.score_evaluations+=2;
            if(!corridor_only_phase_one || cfg->model.world_sweep_check) {
                out->gradient[j]=(cost_plus-cost_minus)/(2.*normalized_step);
                out->positive_diagonal_curvature[j]=std::clamp(
                    (cost_plus-2.*baseline_cost+cost_minus)/
                        (normalized_step*normalized_step),1.e-4,1.e7);
            } else {
                // Phase I uses neither racing gradient nor racing cost
                // Hessian: its separate feasibility objective is built
                // from exact physical margin Jacobians.
                out->gradient[j]=0.;
                out->positive_diagonal_curvature[j]=1.;
            }
            if(!std::isfinite(out->gradient[j])||
               !std::isfinite(out->positive_diagonal_curvature[j]))
                return false;
            for(int row=0;row<2*N;++row)
                out->footprint[row][j]=(wall_plus[row]-wall_minus[row])/
                    (2.*normalized_step);
            for(int stage=0;stage<N;++stage) {
                // No rate residual at stage < command j/2 can depend on
                // command j. The trace retains unrelated old-batch entries
                // there; never read them for a current causal derivative.
                if(stage<j/2) {
                    for(int row=0;row<4;++row)
                        out->actuator_rate[4*stage+row][j]=0.;
                    continue;
                }
                const auto &t=(*trace)[stage][lane];
                const double direct_j0=(j==2*stage)?kCommandScale[0]:0.;
                const double direct_j1=(j==2*stage+1)?kCommandScale[1]:0.;
                const double rack= t.source[UNITY_SOURCE_PHYSICAL_STEERING];
                const double steer=direct_j0-rack;
                const double throttle=direct_j1-t.throttle_queue2;
                out->actuator_rate[4*stage][j]=-steer;
                out->actuator_rate[4*stage+1][j]=steer;
                out->actuator_rate[4*stage+2][j]=-throttle;
                out->actuator_rate[4*stage+3][j]=throttle;
            }
        }
        // No full-trace clear: future batches reset only their entry stage,
        // and the score ignores all stages before the changed command.
        return true;
    };
#if defined(SDU_APEX_MPC_PARALLEL_SOURCE_SENSITIVITIES) && defined(_OPENMP)
    // Never initialize OpenMP just to run a serialized region on a pinned
    // one-core controller. First-use runtime overhead can miss the deadline.
    bool enough_cpus = false;
#if defined(__linux__)
    cpu_set_t allowed_cpus;
    CPU_ZERO(&allowed_cpus);
    if(sched_getaffinity(0,sizeof(allowed_cpus),&allowed_cpus)==0)
        enough_cpus=CPU_COUNT(&allowed_cpus)>=3;
#endif
    // A bound OpenMP master may itself be pinned to one CPU, while its
    // configured worker places span three available cores.
    if(!enough_cpus && std::getenv("OMP_PLACES"))
        enough_cpus=omp_get_num_places()>=3;
    // World sweep callbacks are potentially stateful: keep serial unless
    // their thread safety is explicitly established.
    if(enough_cpus && batches>1 && !cfg->model.world_sweep_check) {
#pragma omp parallel for schedule(static,1) num_threads(3)
        for(int batch=0;batch<batches;++batch)
            succeeded[batch]=evaluate_batch(batch)?1:0;
    } else
#endif
    {
        // Exact same source arithmetic, without OpenMP startup overhead.
        for(int batch=0;batch<batches;++batch)
            succeeded[batch]=evaluate_batch(batch)?1:0;
    }

    bool all_valid=true;
    for(int batch=0;batch<batches;++batch) {
        const auto &measurement=measurements[batch];
        out->dual_plant_us+=measurement.dual_plant_us;
        out->dual_plant_intervals+=measurement.dual_plant_intervals;
        out->tangent_scoring_us+=measurement.tangent_scoring_us;
        out->score_evaluations+=measurement.score_evaluations;
        out->deadline_exhausted|=measurement.deadline_exhausted;
        all_valid &= succeeded[batch]!=0;
    }
    if(!all_valid || !build_gauss_newton_hessian) return all_valid;
    // Build a PSD J'WJ from the SAME 26-physical-state tangent trace used
    // for the objective gradient. This retains every steering/throttle and
    // cross-stage coupling rather than projecting only diagonal curvature.
    // Only the loss curvature is approximated; full-source rollout and the
    // exact objective gradient, QP physical constraints, merit and certificate
    // remain unchanged. No source dynamics are dropped.
    const auto &model=cfg->model;
    const auto &schedule=*cfg->objective_schedule;
    auto append=[&](double weight,const std::array<double,D> &jac) {
        if(weight<=0.)return;
        for(int i=0;i<decision_count;++i) {
            const double scaled=2.*weight*jac[i];
            if(scaled==0.)continue;
            for(int j=0;j<=i;++j)
                out->gn_hessian[i][j]+=scaled*jac[j];
        }
    };
    for(int stage=0;stage<=N;++stage) {
        const auto &state=baseline->states[stage].plant;
        const auto &ref=schedule.references[stage];
        const double term=stage==N?model.terminal_multiplier:1.;
        std::array<double,D> jac{};
        auto tracking=[&](double weight,auto projector) {
            if(weight<=0.)return;
            jac.fill(0.);
            for(int col=0;col<decision_count;++col)
                jac[col]=projector((*gn_trace)[stage][col]);
            append(term*weight,jac);
        };
        if(std::abs(state.e_y-ref.e_y)>model.tracking_lateral_tolerance_m)
            tracking(model.weight_e_y,[](const auto &t){return t.ey;});
        if(std::abs(state.e_psi-ref.e_psi)>model.tracking_heading_tolerance_rad)
            tracking(model.weight_e_psi,[](const auto &t){return t.epsi;});
        tracking(schedule.speed_weights[stage],[](const auto &t){return t.u;});
        tracking(model.weight_v,[](const auto &t){return t.v;});
        tracking(model.weight_r,[](const auto &t){return t.r;});
        tracking(model.weight_steering_command,
                 [](const auto &t){return t.steer_queue2;});
        tracking(model.weight_steering_command,
                 [](const auto &t){return t.rack;});
        if(stage>0) {
            const double slip_weight=model.weight_longitudinal_slip*
                (1.+model.longitudinal_slip_curvature_gain_m*
                     std::abs(ref.path_curvature));
            for(int wheel=0;wheel<4;++wheel) {
                jac.fill(0.);
                for(int col=0;col<decision_count;++col)
                    jac[col]=(*gn_trace)[stage][col].slip_excess[wheel];
                append(term*slip_weight,jac);
            }
            if(model.weight_source_corridor_reserve>0. &&
               model.near_wall_tracking_reserve_m>0.) {
                MpcRtiReference_t physical_ref{};
                if(!mpc_rti_reference_at_progress(trajectory,trajectory_count,
                         lap_length,baseline->progress[stage],&model,
                         &physical_ref))return false;
                float left=0.,right=0.;
                if(!mpc_rti_compute_corridor_slack(
                        &state,physical_ref.left_bound,physical_ref.right_bound,
                        &model,stage,&left,&right))return false;
                const int side=left<=right?0:1;
                if(std::min(left,right)<model.near_wall_tracking_reserve_m) {
                    jac.fill(0.);
                    for(int col=0;col<decision_count;++col)
                        jac[col]=out->footprint[2*(stage-1)+side][col];
                    append(model.weight_source_corridor_reserve,jac);
                }
            }
        }
        if(stage==N)continue;
        const int steer_col=2*stage,throttle_col=steer_col+1;
        jac.fill(0.);
        jac[steer_col]=kCommandScale[0];
        append(model.weight_steering_target,jac);
        jac.fill(0.);
        jac[throttle_col]=kCommandScale[1];
        append(model.weight_throttle_command,jac);
        jac.fill(0.);
        jac[steer_col]=kCommandScale[0];
        if(stage>0)jac[steer_col-2]=-kCommandScale[0];
        append(model.weight_steering_slew,jac);
        const double steer_delta=
            baseline->controls[stage].steering_target_rad-
            (stage?baseline->controls[stage-1].steering_target_rad:
             baseline->states[0].plant.delayed_steering_command_2)-
            (schedule.references[stage+1].steering_command-
             schedule.references[stage].steering_command);
        const double eps=model.steering_correction_variation_epsilon_rad;
        if(model.weight_steering_correction_variation>0. && eps>0.) {
            const double hyp=std::hypot(eps,steer_delta);
            // Exact C2 curvature of lambda*(sqrt(eps^2+z^2)-eps).
            append(model.weight_steering_correction_variation*eps*eps/
                       (2.*hyp*hyp*hyp),jac);
        }
        jac.fill(0.);
        jac[throttle_col]=kCommandScale[1];
        if(stage>0)jac[throttle_col-2]=-kCommandScale[1];
        append(model.weight_throttle_slew,jac);
    }
    for(int i=0;i<decision_count;++i) {
        for(int j=0;j<i;++j)
            out->gn_hessian[j][i]=out->gn_hessian[i][j];
        // Strictly positive regularization of the local *cost* only, not a
        // hard restriction on physical steering/throttle changes.
        out->gn_hessian[i][i]+=1.e-4;
    }
    return true;
}
} // namespace sdu_apex::source_sqp
