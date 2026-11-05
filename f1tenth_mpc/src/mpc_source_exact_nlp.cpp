#include "mpc_source_exact_nlp.hpp"
#include "vehicle_model.h"
#include "mpc_rti_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sdu_apex::source_nlp {
namespace {
static_assert(kPhysical==26 && kState==29 && kStages==30);
static_assert(MPC_PLANT_INTERNAL_STATE_COUNT==22);
}

void pack_state(const MpcRtiState_t &state,double progress,State *out) {
    if(!out)return;
    // This is the literal state packing used by the authoritative source
    // plant, not a duplicate physical-coordinate conversion.
    if(!vehicle_model_pack_source_state(&state.plant,out->z)) {
        for(auto &value:out->z)value=NAN;
        return;
    }
    out->z[kPhysical]=state.plant.e_y;
    out->z[kPhysical+1]=state.plant.e_psi;
    out->z[kPhysical+2]=progress;
}

bool inject_state(const State &in,MpcRtiState_t *inout,
                  double *progress) {
    if(!inout||!progress)return false;
    for(double value:in.z)
        if(!std::isfinite(value))return false;
    if(!vehicle_model_inject_source_state(&inout->plant,in.z))
        return false;
    inout->plant.e_y=static_cast<float>(in.z[kPhysical]);
    inout->plant.e_psi=static_cast<float>(in.z[kPhysical+1]);
    *progress=in.z[kPhysical+2];
    return std::isfinite(inout->plant.e_y)&&
           std::isfinite(inout->plant.e_psi);
}

bool decision_from_nominal(const MpcRtiNominal_t &nominal,Decision *out) {
    if(!out || !nominal.valid || nominal.horizon!=kStages)return false;
    for(int k=0;k<kStages;++k) {
        pack_state(nominal.states[k+1],nominal.progress[k+1],
                   &out->nodes[k]);
        out->controls[k]=nominal.controls[k];
    }
    return true;
}

bool linearize_original_source_stage(
    const MpcRtiState_t &current,double current_progress,
    const MpcModelControl_t &command,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,StageJacobian *jacobian,
    MpcRtiState_t *next,double *next_progress)
{
    if(!jacobian||!next||!next_progress||!trajectory)return false;
    constexpr int L=UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;
    *jacobian={};
    MpcRtiState_t expected{};
    double predicted_progress=0.;
        if(!mpc_rti_source_exact_stage(&current,current_progress,
              &command,trajectory,trajectory_count,
              lap_length,dt,&expected,&predicted_progress) ||
           !expected.incoming_tracking_jacobian_valid)
            return false;

    const auto &g=expected.incoming_tracking_jacobian;
    auto &stage=*jacobian;
        // 26 physical states + 2 original physical controls. NO
        // derivative-column compression yet: state reduction must not
        // silently omit low-speed timer physics in a full nonlinear NLP.
        for(int batch=0;batch<(kPhysical+2+L-1)/L;++batch) {
            double seed[kPhysical][L]{},input[2][L]{};
            for(int lane=0;lane<L;++lane) {
                const int column=batch*L+lane;
                if(column<kPhysical)seed[column][lane]=1.;
                else if(column<kPhysical+2)
                    input[column-kPhysical][lane]=1.;
            }
            double physical[kPhysical][L]{},body[3][L]{},slip[4][L]{};
            if(!vehicle_model_full_source_interval_sensitivities(
                  &current.plant,&command,dt,
                  seed,input,physical,body,slip))
                return false;

            for(int lane=0;lane<L;++lane) {
                const int column=batch*L+lane;
                if(column>=kPhysical+2)break;
                for(int wheel=0;wheel<4;++wheel) {
                    if(column<kPhysical)
                        stage.peak_slip_state[wheel][column]=slip[wheel][lane];
                    else
                        stage.peak_slip_control[wheel][column-kPhysical]=
                            slip[wheel][lane];
                }
                for(int r=0;r<kPhysical;++r) {
                    if(column<kPhysical)
                        stage.A[r][column]=physical[r][lane];
                    else stage.B[r][column-kPhysical]=physical[r][lane];
                }
                for(int row=0;row<3;++row) {
                    double value=0.;
                    for(int i=0;i<3;++i)
                        value+=g[row][i+2]*body[i][lane];
                    if(column<kPhysical)
                        stage.A[kPhysical+row][column]=value;
                    else stage.B[kPhysical+row][column-kPhysical]=value;
                }
            }
        }
        for(int wheel=0;wheel<4;++wheel) {
            stage.source_peak_slip[wheel]=
                expected.incoming_wheel_slip_excess_mps[wheel];
        }
        for(int row=0;row<3;++row) {
            stage.A[kPhysical+row][kPhysical]=g[row][0];
            stage.A[kPhysical+row][kPhysical+1]=g[row][1];
            stage.A[kPhysical+row][kPhysical+2]=g[row][5];
        }
    *next=expected;
    *next_progress=predicted_progress;
    return true;
}

// The only full-horizon source Jacobian assembler: the 26 physical states,
// 3 original tracking states, 2 original absolute controls and EVERY
// original tire-substep constraint come from the SAME source interval
// and the SAME 28-lane source AD seed. No second plant integration can
// silently select a different friction/contact branch.
bool linearize_dynamics_and_slip_epigraph(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,const Decision &decision,
    DynamicsJacobian *dynamics,SlipEpigraphJacobian *slip_rows,
    const MpcRtiNominal_t *known_original_source_trace,int requested_threads)
{
    if((!dynamics&&!slip_rows)||!trajectory||trajectory_count<3 ||
       !std::isfinite(initial_progress)||!std::isfinite(lap_length)||
       lap_length<=0.||!std::isfinite(dt)||dt<=0.f)
        return false;
    if(known_original_source_trace &&
       (!known_original_source_trace->valid ||
        known_original_source_trace->horizon!=kStages ||
        known_original_source_trace->progress[0]!=initial_progress))
        return false;
    if(dynamics)*dynamics={};
    if(slip_rows) {
        slip_rows->rows.clear();
        slip_rows->rows.reserve(16000);
        slip_rows->original_source_ad_calls=0;
        slip_rows->original_nonlinear_stage_calls=0;
        slip_rows->valid=0;
    }
    constexpr int L=UNITY_SOURCE_PLANT_SENSITIVITY_WIDTH;
    constexpr int capacity=8192;
    // In source-exact reduced NLP, the authoritative source rollout
    // ALREADY contains the 29-state input of every physical interval.
    // Their AD transitions are independent functions of those inputs.
    // Calculate them concurrently, then concatenate all original tire
    // substep rows in the unchanged deterministic stage/tick/wheel order.
    // In a general multiple-shooting linearization without a complete
    // known source trace, preserve the causal sequential stage path.
    std::array<std::vector<SlipEpigraphJacobianRow>,kStages> stage_rows;
    const auto process_stage=[&](
        int k,const MpcRtiState_t &previous,
        const MpcRtiState_t &exact,double exact_progress,
        UnitySourceSlipSubstepSensitivityC *substeps)->bool {
        if(known_original_source_trace) {
            if(known_original_source_trace->controls[k].steering_target_rad!=
                   decision.controls[k].steering_target_rad ||
               known_original_source_trace->controls[k].throttle_command!=
                   decision.controls[k].throttle_command ||
               !exact.incoming_transition_valid)
                return false;
            State packed{};
            pack_state(exact,exact_progress,&packed);
            for(int j=0;j<kState;++j)
                if(packed.z[j]!=decision.nodes[k].z[j])
                    return false;
        }
        if(dynamics && !exact.incoming_tracking_jacobian_valid)
            return false;
        const auto &g=exact.incoming_tracking_jacobian;
        auto &rows=stage_rows[k];
        if(slip_rows) rows.reserve(1200);
        for(int batch=0;batch<(kPhysical+2+L-1)/L;++batch) {
            double seed[kPhysical][L]{},control[2][L]{};
            for(int lane=0;lane<L;++lane) {
                const int column=batch*L+lane;
                if(column<kPhysical)seed[column][lane]=1.;
                else if(column<kPhysical+2)
                    control[column-kPhysical][lane]=1.;
            }
            double physical[kPhysical][L]{},body[3][L]{},peak[4][L]{};
            int count=0;
            if(dynamics && slip_rows) {
                if(!vehicle_model_full_source_interval_and_substep_sensitivities(
                    &previous.plant,&decision.controls[k],dt,
                    seed,control,physical,body,peak,
                    substeps,capacity,&count))return false;
            } else if(dynamics) {
                if(!vehicle_model_full_source_interval_sensitivities(
                    &previous.plant,&decision.controls[k],dt,
                    seed,control,physical,body,peak))return false;
            } else {
                if(!vehicle_model_source_slip_substep_sensitivities(
                    &previous.plant,&decision.controls[k],dt,
                    seed,control,substeps,capacity,&count))
                    return false;
            }
            if(dynamics) {
                auto &stage=dynamics->stages[k];
                for(int lane=0;lane<L;++lane) {
                    const int column=batch*L+lane;
                    if(column>=kPhysical+2)break;
                    for(int wheel=0;wheel<4;++wheel) {
                        if(column<kPhysical)
                            stage.peak_slip_state[wheel][column]=
                                peak[wheel][lane];
                        else stage.peak_slip_control[wheel][
                                column-kPhysical]=peak[wheel][lane];
                    }
                    for(int state=0;state<kPhysical;++state) {
                        if(column<kPhysical)
                            stage.A[state][column]=physical[state][lane];
                        else stage.B[state][column-kPhysical]=
                                physical[state][lane];
                    }
                    for(int row=0;row<3;++row) {
                        double tangent=0.;
                        for(int i=0;i<3;++i)
                            tangent+=g[row][i+2]*body[i][lane];
                        if(column<kPhysical)
                            stage.A[kPhysical+row][column]=tangent;
                        else stage.B[kPhysical+row][column-kPhysical]=
                                tangent;
                    }
                }
            }
            if(slip_rows) {
                if(count<0 || count>capacity)return false;
                if(batch==0) {
                    for(int j=0;j<count;++j) {
                        const auto &r=substeps[j].original;
                        if(r.wheel<0||r.wheel>=4||
                           !std::isfinite(r.positive_excess_mps))
                            return false;
                        SlipEpigraphJacobianRow row{};
                        row.stage=k;
                        row.wheel=r.wheel;
                        row.fixed_tick=r.fixed_tick;
                        row.vehicle_substep=r.vehicle_substep;
                        row.original_source_excess=r.positive_excess_mps;
                        rows.push_back(row);
                    }
                } else if(size_t(count)!=rows.size())return false;
                for(int j=0;j<count;++j) {
                    auto &row=rows[j];
                    const auto &r=substeps[j].original;
                    if(row.wheel!=r.wheel ||
                       row.fixed_tick!=r.fixed_tick ||
                       row.vehicle_substep!=r.vehicle_substep ||
                       row.original_source_excess!=r.positive_excess_mps)
                        return false;
                    for(int lane=0;lane<L;++lane) {
                        const int column=batch*L+lane;
                        if(column<kPhysical)
                            row.previous_physical_state[column]=
                                substeps[j].derivative[lane];
                        else if(column<kPhysical+2)
                            row.control[column-kPhysical]=
                                substeps[j].derivative[lane];
                    }
                }
            }
        }
        if(dynamics) {
            auto &stage=dynamics->stages[k];
            for(int wheel=0;wheel<4;++wheel)
                stage.source_peak_slip[wheel]=
                    exact.incoming_wheel_slip_excess_mps[wheel];
            for(int row=0;row<3;++row) {
                stage.A[kPhysical+row][kPhysical]=g[row][0];
                stage.A[kPhysical+row][kPhysical+1]=g[row][1];
                stage.A[kPhysical+row][kPhysical+2]=g[row][5];
            }
        }
        if(slip_rows) {
            double maximum[4]{};
            for(const auto &r:rows)
                maximum[r.wheel]=std::max(
                    maximum[r.wheel],r.original_source_excess);
            for(int wheel=0;wheel<4;++wheel)
                if(std::abs(maximum[wheel]-
                    exact.incoming_wheel_slip_excess_mps[wheel])>1.e-10)
                    return false;
        }
        return true;
    };

    if(known_original_source_trace) {
        const auto &trace=*known_original_source_trace;
        std::atomic<bool> all_valid{true};
#ifdef _OPENMP
        const int workers=std::min({8,omp_get_max_threads(),std::max(1,requested_threads)});
        const bool parallel=
            workers>1 &&
            std::getenv("MPC_SOURCE_NLP_FORCE_SERIAL_STAGE_AD")==nullptr;
        if(parallel) {
#pragma omp parallel num_threads(workers)
            {
                auto scratch=slip_rows?
                    std::make_unique<UnitySourceSlipSubstepSensitivityC[]>(
                        capacity):nullptr;
#pragma omp for schedule(dynamic,1)
                for(int k=0;k<kStages;++k)
                    if(!process_stage(k,trace.states[k],
                            trace.states[k+1],trace.progress[k+1],
                            scratch.get()))
                        all_valid.store(false,std::memory_order_relaxed);
            }
        } else
#endif
        {
            auto scratch=slip_rows?
                std::make_unique<UnitySourceSlipSubstepSensitivityC[]>(
                    capacity):nullptr;
            for(int k=0;k<kStages;++k)
                if(!process_stage(k,trace.states[k],trace.states[k+1],
                        trace.progress[k+1],scratch.get()))
                    all_valid.store(false,std::memory_order_relaxed);
        }
        if(!all_valid.load(std::memory_order_relaxed))return false;
    } else {
        // Arbitrary 29-state multiple-shooting decisions may be
        // dynamically inconsistent. Linearize each ORIGINAL source
        // transition at exactly that supplied state, sequentially,
        // retaining the original (source-manifold-free) NLP semantics.
        auto scratch=slip_rows?
            std::make_unique<UnitySourceSlipSubstepSensitivityC[]>(
                capacity):nullptr;
        MpcRtiState_t previous=initial;
        double progress=initial_progress;
        for(int k=0;k<kStages;++k) {
            MpcRtiState_t exact{};
            double exact_progress=0.;
            if(!mpc_rti_source_exact_stage(
                &previous,progress,&decision.controls[k],
                trajectory,trajectory_count,lap_length,dt,
                &exact,&exact_progress) ||
               !process_stage(k,previous,exact,exact_progress,scratch.get()))
                return false;
            previous=exact;
            progress=exact_progress;
            if(!inject_state(decision.nodes[k],&previous,&progress))
                return false;
        }
    }
    if(dynamics) {
        dynamics->source_nonlinear_stage_calls=kStages;
        dynamics->original_source_ad_calls=2*kStages;
        dynamics->valid=1;
    }
    if(slip_rows) {
        slip_rows->original_nonlinear_stage_calls=kStages;
        slip_rows->original_source_ad_calls=2*kStages;
        for(auto &stage:stage_rows)
            slip_rows->rows.insert(slip_rows->rows.end(),
                                   stage.begin(),stage.end());
        slip_rows->valid=1;
    }
    return true;
}

bool linearize_dynamics(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,const Decision &decision,
    DynamicsJacobian *result)
{
    if(!result)return false;
    return linearize_dynamics_and_slip_epigraph(
        initial,initial_progress,trajectory,trajectory_count,
        lap_length,dt,decision,result,nullptr);
}

bool source_peak_slip_objective_gradient(
    const MpcRtiCycleConfiguration_t &configuration,
    const DynamicsJacobian &dynamics,PeakSlipObjectiveGradient *result)
{
    if(!result||!dynamics.valid||!configuration.objective_schedule||
       !configuration.objective_schedule->valid ||
       configuration.objective_schedule->horizon!=kStages)return false;
    *result={};
    for(int k=0;k<kStages;++k) {
        const auto &ref=configuration.objective_schedule->references[k+1];
        // EXACT same helper as the original racing-cost scorer. Do not
        // duplicate source curvature scaling or stage-index semantics.
        const double weight=longitudinal_slip_weight_for_reference(
            &configuration.model,&ref)*
            (k==kStages-1?configuration.model.terminal_multiplier:1.);
        if(!std::isfinite(weight) || weight<0.)return false;
        const auto &source=dynamics.stages[k];
        for(int wheel=0;wheel<4;++wheel) {
            const double peak=source.source_peak_slip[wheel];
            if(!std::isfinite(peak)||peak<0.)return false;
            result->cost+=weight*peak*peak;
            if(peak<=0.)continue;
            ++result->active_peak_wheels;
            const double scale=2.*weight*peak;
            if(k>0)
                for(int col=0;col<kState;++col)
                    result->state[k-1][col]+=
                        scale*source.peak_slip_state[wheel][col];
            for(int input=0;input<2;++input)
                result->control[k][input]+=
                    scale*source.peak_slip_control[wheel][input];
        }
    }
    result->valid=std::isfinite(result->cost);
    return result->valid;
}

bool linearize_slip_epigraph(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,const Decision &decision,
    SlipEpigraphJacobian *result)
{
    if(!result)return false;
    return linearize_dynamics_and_slip_epigraph(
        initial,initial_progress,trajectory,trajectory_count,
        lap_length,dt,decision,nullptr,result);
}

static bool evaluate_impl(const MpcRtiState_t &initial,
    double initial_progress,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &decision,Evaluation *result,
    const double (*epigraph_peak)[4],
    SlipEpigraphEvaluation *epigraph,
    MpcRtiNominal_t *original_source_trace,
    const MpcRtiNominal_t *known_source_rollout=nullptr)
{
    if(!result || !trajectory || trajectory_count<3 ||
       !std::isfinite(initial_progress) || !std::isfinite(lap_length) ||
       lap_length<=0. || !std::isfinite(dt) || dt<=0.f)return false;
    *result={};
    // In the live control-only SQP, every candidate has already been
    // integrated through the ORIGINAL source model and verified against
    // its shooting coordinates. Score this trace directly. Making another
    // large nominal copy and injecting every identical physical state is
    // redundant, especially for seven independent exact nonlinear trials.
    // Standalone multiple-shooting evaluations retain their own mutable
    // original source rollout, including the independent dynamics check.
    std::unique_ptr<MpcRtiNominal_t> owned_source;
    const MpcRtiNominal_t *original=known_source_rollout;
    if(!original) {
        owned_source=std::make_unique<MpcRtiNominal_t>();
        original=owned_source.get();
        owned_source->valid=1;
        owned_source->horizon=kStages;
        owned_source->speed_reference_valid=0;
        owned_source->states[0]=initial;
        owned_source->progress[0]=initial_progress;
    }
    if(known_source_rollout) {
        // This is an already completed rollout of the AUTHORITATIVE
        // original source plant. Its physical source state, original
        // tire-slip maxima, contact flags and captured command epochs
        // are exact; reuse them rather than advancing the identical
        // source plant a second time for the SAME score.
        if(!known_source_rollout->valid ||
           known_source_rollout->horizon!=kStages)
            return false;
        for(int k=0;k<kStages;++k) {
            const auto &u=known_source_rollout->controls[k];
            if(u.steering_target_rad!=decision.controls[k].steering_target_rad||
               u.throttle_command!=decision.controls[k].throttle_command)
                return false;
            State source_node{};
            pack_state(known_source_rollout->states[k+1],
                known_source_rollout->progress[k+1],&source_node);
            for(int j=0;j<kState;++j)
                if(source_node.z[j]!=decision.nodes[k].z[j])
                    return false;
        }
    }
    if(epigraph) {
        if(!epigraph_peak||!configuration.objective_schedule||
           !configuration.objective_schedule->valid||
           configuration.objective_schedule->horizon!=kStages)
            return false;
        epigraph->rows.clear();
        epigraph->rows.reserve(16000);
        epigraph->min_epigraph_margin=INFINITY;
        epigraph->min_peak_nonnegative_margin=INFINITY;
        epigraph->physical_source_rows=0;
        epigraph->epigraph_peak_cost=0.;
        epigraph->original_peak_cost=0.;
        epigraph->valid=0;
    }
    result->minimum_command_box_margin=INFINITY;
    result->minimum_command_rate_margin=INFINITY;
    for(int k=0;k<kStages;++k) {
        if(owned_source)owned_source->controls[k]=decision.controls[k];
        const auto &u=decision.controls[k];
        const double steering_limit=configuration.model.max_steering_rad;
        result->command_box[k][0]=steering_limit-double(u.steering_target_rad);
        result->command_box[k][1]=steering_limit+double(u.steering_target_rad);
        result->command_box[k][2]=
            double(MPC_THROTTLE_MAX)-double(u.throttle_command);
        result->command_box[k][3]=
            double(u.throttle_command)-double(MPC_THROTTLE_MIN_DRIVE);
        for(double v:result->command_box[k]) {
            if(!std::isfinite(v))return false;
            result->minimum_command_box_margin=std::min(
                result->minimum_command_box_margin,v);
        }
        // The original C++ source clips out-of-box commands. Such
        // clipping would make a purported NLP optimization inaccurate:
        // here we require the real input domain, not a hidden clamp.
        if(result->minimum_command_box_margin < -1.e-7)return false;
        if(!mpc_rti_source_command_rate_margins(
              &original->states[k].plant,&u,&configuration.model,
              result->command_rate[k]))return false;
        for(double v:result->command_rate[k])
            result->minimum_command_rate_margin=std::min(
                result->minimum_command_rate_margin,v);
        double actual_trace_max[4]{};
        if(epigraph) {
            // The ORIGINAL source traces one event per wheel/internal
            // PhysX substep. No approximate slip curve, stage-end proxy,
            // or frozen max branch is introduced here.
            UnitySourceSlipSubstepC samples[8192]{};
            int count=0;
            if(!vehicle_model_source_slip_substeps(
                &original->states[k].plant,&decision.controls[k],dt,
                samples,8192,&count,actual_trace_max))
                return false;
            for(int wheel=0;wheel<4;++wheel) {
                if(!std::isfinite(epigraph_peak[k][wheel]))
                    return false;
                epigraph->min_peak_nonnegative_margin=std::min(
                    epigraph->min_peak_nonnegative_margin,
                    epigraph_peak[k][wheel]);
                const auto &ref=
                    configuration.objective_schedule->references[k+1];
                const double weight=longitudinal_slip_weight_for_reference(
                    &configuration.model,&ref)*
                    (k==kStages-1?
                        configuration.model.terminal_multiplier:1.);
                epigraph->epigraph_peak_cost+=weight*
                    epigraph_peak[k][wheel]*epigraph_peak[k][wheel];
            }
            for(int j=0;j<count;++j) {
                const auto &v=samples[j];
                const double margin=
                    epigraph_peak[k][v.wheel]-v.positive_excess_mps;
                epigraph->rows.push_back({
                    k,v.wheel,v.fixed_tick,v.vehicle_substep,
                    v.positive_excess_mps,margin});
                epigraph->min_epigraph_margin=std::min(
                    epigraph->min_epigraph_margin,margin);
            }
            epigraph->physical_source_rows+=count;
        }
        MpcRtiState_t predicted{};
        double predicted_progress=0.;
        if(known_source_rollout) {
            predicted=known_source_rollout->states[k+1];
            predicted_progress=known_source_rollout->progress[k+1];
            if(!predicted.incoming_transition_valid)return false;
        } else if(!mpc_rti_source_exact_stage(
                &original->states[k],original->progress[k],
                &decision.controls[k],
                trajectory,trajectory_count,lap_length,dt,
                &predicted,&predicted_progress))
            return false;
        State exact{};
        pack_state(predicted,predicted_progress,&exact);
        for(int j=0;j<kState;++j) {
            const double expected=decision.nodes[k].z[j];
            result->dynamics[k][j]=expected-exact.z[j];
            result->max_dynamics_abs=std::max(result->max_dynamics_abs,
                std::abs(result->dynamics[k][j]));
        }
        result->source_branch_flags[k]=predicted.incoming_branch_flags;
        if(epigraph)
            for(int w=0;w<4;++w)
                if(std::abs(
                    predicted.incoming_wheel_slip_excess_mps[w]-
                    actual_trace_max[w])>1.e-10)
                    return false;
        // Score original source feature without modifying the original
        // racing objective. The independent max-over-substeps is the only
        // legitimate longitudinal slip term for stage k.
        const MpcRtiObjectiveSchedule_t *frozen=
            configuration.objective_schedule;
        if(frozen && frozen->valid && frozen->horizon==kStages) {
            const double weight=longitudinal_slip_weight_for_reference(
                &configuration.model,&frozen->references[k+1])*
                (k==kStages-1?
                    configuration.model.terminal_multiplier:1.);
            for(int wheel=0;wheel<4;++wheel) {
                const double slip=
                    predicted.incoming_wheel_slip_excess_mps[wheel];
                result->source_peak_slip[k][wheel]=slip;
                result->source_peak_slip_cost+=weight*slip*slip;
            }
        }
        // Keep original source branch/slip/max metrics for this stage.
        // Only independently optimized physical+tracking state variables
        // are overridden. Delay/queue metadata is propagated by original
        // source stage from physical controls, not added as fake NLP states.
        if(owned_source) {
            owned_source->states[k+1]=predicted;
            owned_source->progress[k+1]=predicted_progress;
            if(!inject_state(decision.nodes[k],&owned_source->states[k+1],
                             &owned_source->progress[k+1]))
                return false;
        }
    }
    if(!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
        original,trajectory,trajectory_count,lap_length,
        &configuration,&result->objective,
        &result->minimum_corridor_slack_m,result->corridor))
        return false;
    // The exact original Frenet collision check is the MINIMUM of the
    // eight genuine body corners. Each corner is smooth in heading,
    // even where the original max-support active bumper switches.
    // Keep the original reference bound interpolation; do NOT fit a
    // continuous spline or average incompatible one-sided knot slopes.
    for(int k=1;k<=kStages;++k) {
        auto &g=result->corridor_corners[k-1];
        if(!mpc_rti_source_exact_corridor_corners(
            &original->states[k].plant,trajectory,trajectory_count,
            lap_length,original->progress[k],&configuration.model,k,&g))
            return false;
        double right=INFINITY,left=INFINITY;
        for(int corner=0;corner<4;++corner) {
            right=std::min(right,g.right[corner]);
            left=std::min(left,g.left[corner]);
        }
        result->corner_source_parity_max_abs=std::max({
            result->corner_source_parity_max_abs,
            std::abs(right-double(result->corridor[k][0])),
            std::abs(left-double(result->corridor[k][1]))});
        result->corridor_reference_kink_count+=
            g.reference_progress_kink!=0;
        result->corridor_reference_derivative_invalid_count+=
            !g.reference_derivatives_valid;
    }
    if(result->corner_source_parity_max_abs>1.e-9)
        return false;
    if(original_source_trace)
        *original_source_trace=*original;
    result->valid=1;
    if(epigraph) {
        epigraph->source=*result;
        epigraph->original_peak_cost=result->source_peak_slip_cost;
        // Original source cost already contains weight*source_max^2.
        // Replacing only those terms with epigraph peak variables makes
        // the objective EXACT at the minimal feasible epigraph solution.
        epigraph->objective=result->objective-
            result->source_peak_slip_cost+
            epigraph->epigraph_peak_cost;
        epigraph->valid=std::isfinite(epigraph->objective) &&
            std::isfinite(epigraph->min_epigraph_margin) &&
            std::isfinite(epigraph->min_peak_nonnegative_margin);
        if(!epigraph->valid)return false;
    }
    return true;
}

bool evaluate(const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &decision,Evaluation *result)
{
    return evaluate_impl(initial,initial_progress,trajectory,
        trajectory_count,lap_length,dt,configuration,decision,
        result,nullptr,nullptr,nullptr);
}

// Only for a physically consistent, ORIGINAL source-propagated nominal.
// This is the same canonical nonlinear scorer as evaluate(), with the
// preceding exact source rollout's immutable 30 physical transitions
// reused. No stage dynamics are approximated, nor is a second scorer
// substituted for the original physical source.
bool evaluate_existing_source_nominal(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const MpcRtiNominal_t &known_source,
    const Decision &decision,Evaluation *result)
{
    return evaluate_impl(initial,initial_progress,trajectory,
        trajectory_count,lap_length,dt,configuration,decision,
        result,nullptr,nullptr,nullptr,&known_source);
}

bool evaluate_slip_epigraph_with_nominal(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,
    SlipEpigraphEvaluation *result,MpcRtiNominal_t *source_trace,
    const MpcRtiNominal_t *known_original_trace)
{
    if(!result)return false;
    // Original source nonlinear primal, epigraph slip observation and
    // original source state trace are evaluated exactly once, using
    // ONE authoritative source execution and score.
    auto source=std::make_unique<Evaluation>();
    return evaluate_impl(initial,initial_progress,trajectory,
        trajectory_count,lap_length,dt,configuration,decision.shooting,
        source.get(),decision.peak,result,source_trace,
        known_original_trace);
}

bool evaluate_slip_epigraph(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,
    SlipEpigraphEvaluation *result)
{
    return evaluate_slip_epigraph_with_nominal(
        initial,initial_progress,trajectory,trajectory_count,
        lap_length,dt,configuration,decision,result,nullptr);
}

bool evaluate_with_nominal(const MpcRtiState_t &initial,
    double initial_progress,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &decision,Evaluation *result,
    MpcRtiNominal_t *original_source_trace)
{
    if(!original_source_trace)return false;
    return evaluate_impl(initial,initial_progress,trajectory,
        trajectory_count,lap_length,dt,configuration,decision,
        result,nullptr,nullptr,original_source_trace);
}
} // namespace sdu_apex::source_nlp
