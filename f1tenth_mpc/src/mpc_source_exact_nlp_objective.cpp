#include "mpc_source_exact_nlp.hpp"
#include "mpc_rti_internal.hpp"
#include "mpc_steering_variation.hpp"
#include "vehicle_model.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace sdu_apex::source_nlp {
namespace {
constexpr double kCom=.15532;

double tracking_excess(double e,double tolerance) {
    const double magnitude=std::max(0.,std::abs(e)-tolerance);
    return std::copysign(magnitude,e);
}
}
bool original_source_objective_gradient_from_source_trace(
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,
    const MpcRtiNominal_t &source_trace,
    const Evaluation &source_score,
    ExactSourceObjectiveGradient *result)
{
    if(!result||!configuration.objective_schedule||
       !configuration.objective_schedule->valid||
       configuration.objective_schedule->horizon!=kStages ||
       configuration.model.world_sweep_check ||
       !source_trace.valid||source_trace.horizon!=kStages||
       !source_score.valid)
        return false;
    *result={};
    const auto *nominal=&source_trace;
    const auto *score=&source_score;
    const auto &cfg=configuration.model;
    const auto &schedule=*configuration.objective_schedule;
    result->original_source_objective=score->objective;
    result->epigraph_objective=score->objective-
        score->source_peak_slip_cost;
    double last_steer=
        nominal->states[0].plant.delayed_steering_command_2;
    double last_throttle=
        nominal->states[0].plant.delayed_throttle_command_2;
    for(int k=0;k<=kStages;++k) {
        const auto &plant=nominal->states[k].plant;
        const auto &ref=schedule.references[k];
        const double terminal=k==kStages?
            double(cfg.terminal_multiplier):1.;
        if(k>0) {
            auto &g=result->state[k-1];
            // The two tracking deadzones contribute the exact derivative
            // of w*max(0,abs(error)-tolerance)^2 at every smooth point.
            g[kPhysical]+=2.*terminal*cfg.weight_e_y*
                tracking_excess(double(plant.e_y)-ref.e_y,
                    cfg.tracking_lateral_tolerance_m);
            g[kPhysical+1]+=2.*terminal*cfg.weight_e_psi*
                tracking_excess(double(plant.e_psi)-ref.e_psi,
                    cfg.tracking_heading_tolerance_rad);
            g[UNITY_SOURCE_U]+=2.*terminal*schedule.speed_weights[k]*
                (double(plant.u)-ref.u);
            const double rear_v_gradient=2.*terminal*cfg.weight_v*
                (double(plant.v)-ref.v);
            g[UNITY_SOURCE_V]+=rear_v_gradient;
            g[UNITY_SOURCE_YAW_RATE]+=
                -kCom*rear_v_gradient+
                2.*terminal*cfg.weight_r*(double(plant.r)-ref.r);
            g[UNITY_SOURCE_PHYSICAL_STEERING]+=
                2.*terminal*cfg.weight_steering_command*
                (double(plant.actual_steering_angle)-
                 ref.steering_command);

            // The direct-command source queues carry u_(k-1) into the
            // next stage's state. They are NOT extra independent shooting
            // coordinates and must not be silently differentiated as such.
            result->control[k-1][0]+=
                2.*terminal*cfg.weight_steering_command*
                (double(plant.delayed_steering_command_2)-
                 ref.steering_command);

            if(cfg.weight_source_lateral_slip>0.f) {
                double source[kPhysical]{};
                if(!vehicle_model_pack_source_state(&plant,source))
                    return false;
                double actual_slip[4]{};
                double exact_jac[4][kPhysical]{};
                if(!unity_source_plant_instantaneous_lateral_slip_ad_c(
                    source,actual_slip,exact_jac))
                    return false;
                ++result->evaluated_source_lateral_slip_stages;
                for(int front=0;front<2;++front) {
                    const double slip=actual_slip[front];
                    const double excess=std::max(0.,
                        std::abs(slip)-
                        kMpcSourceFrontLateralSlipReserveRad);
                    if(excess<=0.)continue;
                    const double scale=2.*terminal*
                        cfg.weight_source_lateral_slip*excess*
                        (slip<0.?-1.:1.);
                    for(int j=0;j<kPhysical;++j)
                        g[j]+=scale*exact_jac[front][j];
                }
            }

            if(cfg.weight_source_corridor_reserve>0.f &&
               cfg.near_wall_tracking_reserve_m>0.f) {
                const auto &geometry=score->corridor_corners[k-1];
                const double source_margin=std::min(
                    double(score->corridor[k][0]),
                    double(score->corridor[k][1]));
                const double deficit=std::max(0.,
                    double(cfg.near_wall_tracking_reserve_m)-
                    source_margin);
                if(deficit>0.) {
                    int active_corner=-1,active_side=-1;
                    int ties=0;
                    for(int side=0;side<2;++side)
                        for(int j=0;j<4;++j) {
                            const double margin=side==0?
                                geometry.left[j]:geometry.right[j];
                            if(std::abs(margin-source_margin)<1.e-10) {
                                ++ties;
                                if(active_corner<0) {
                                    active_corner=j;
                                    active_side=side;
                                }
                            }
                        }
                    if(active_corner<0)return false;
                    result->nonsmooth_wall_reserve_stages+=ties>1;
                    const double scale=-2.*cfg.weight_source_corridor_reserve*
                        deficit;
                    const bool left=active_side==0;
                    g[kPhysical]+=scale*(left?-1.:1.);
                    g[kPhysical+1]+=scale*(left?
                        geometry.left_d_psi[active_corner]:
                        geometry.right_d_psi[active_corner]);
                    const double fwd=left?
                        geometry.left_d_progress_forward:
                        geometry.right_d_progress_forward;
                    const double bwd=left?
                        geometry.left_d_progress_backward:
                        geometry.right_d_progress_backward;
                    if(geometry.reference_progress_kink)
                        ++result->nonsmooth_reference_progress_stages;
                    g[kPhysical+2]+=scale*fwd;
                    result->state_progress_backward[k-1]=
                        g[kPhysical+2]-scale*fwd+scale*bwd;
                } else {
                    result->state_progress_backward[k-1]=g[kPhysical+2];
                }
            } else {
                result->state_progress_backward[k-1]=g[kPhysical+2];
            }
        }

        if(k<kStages) {
            const auto &u=nominal->controls[k];
            const double steer=u.steering_target_rad;
            const double throttle=u.throttle_command;
            const double reference_increment=
                steering_reference_increment(&schedule.references[k],
                                             &schedule.references[k+1]);
            const double steering_correction=
                steer-last_steer-reference_increment;
            const double throttle_delta=throttle-last_throttle;
            result->control[k][0]+=
                2.*cfg.weight_steering_target*
                (steer-ref.steering_command);
            result->control[k][1]+=
                2.*cfg.weight_throttle_command*throttle;
            const double correction_slope=
                2.*cfg.weight_steering_slew*steering_correction+
                (cfg.weight_steering_correction_variation>0.f?
                 cfg.weight_steering_correction_variation*
                 steering_correction/
                 std::hypot(cfg.steering_correction_variation_epsilon_rad,
                            steering_correction):0.);
            const double throttle_slope=
                2.*cfg.weight_throttle_slew*throttle_delta;
            result->control[k][0]+=correction_slope;
            result->control[k][1]+=throttle_slope;
            if(k>0) {
                result->control[k-1][0]-=correction_slope;
                result->control[k-1][1]-=throttle_slope;
            }
            last_steer=steer;
            last_throttle=throttle;
        }
    }
    // Original longitudinal slip maximum has become an epigraph
    // variable; no selected-max derivative enters this racing objective.
    // Every source-substep dependence lives in the nonlinear inequalities.
    for(int k=0;k<kStages;++k) {
        const double weight=longitudinal_slip_weight_for_reference(
            &cfg,&schedule.references[k+1])*
            (k==kStages-1?cfg.terminal_multiplier:1.);
        for(int wheel=0;wheel<4;++wheel) {
            const double t=decision.peak[k][wheel];
            if(!std::isfinite(t))return false;
            result->epigraph_objective+=weight*t*t;
            result->peak[k][wheel]=2.*weight*t;
        }
    }
    result->valid=std::isfinite(result->epigraph_objective);
    for(int k=0;k<kStages;++k) {
        for(double value:result->state[k])
            if(!std::isfinite(value))return false;
        for(double value:result->control[k])
            if(!std::isfinite(value))return false;
        for(double value:result->peak[k])
            if(!std::isfinite(value))return false;
    }
    return result->valid;
}

bool original_source_objective_gradient(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,
    ExactSourceObjectiveGradient *result)
{
    if(!result||!configuration.objective_schedule||
       !configuration.objective_schedule->valid||
       configuration.objective_schedule->horizon!=kStages ||
       configuration.model.world_sweep_check)
        return false;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    auto score=std::make_unique<Evaluation>();
    if(!evaluate_with_nominal(initial,initial_progress,trajectory,
         trajectory_count,lap_length,dt,configuration,
         decision.shooting,score.get(),nominal.get()))
        return false;
    return original_source_objective_gradient_from_source_trace(
        configuration,decision,*nominal,*score,result);
}
} // namespace sdu_apex::source_nlp
