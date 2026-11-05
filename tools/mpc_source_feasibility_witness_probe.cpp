// Offline ORIGINAL-SOURCE feasibility witness search for a recorded Unity
// failure. Diagnostic only: not built/linked into ROS MPC, and never a live
// controller, SQP fallback, alpha search or alternative racing optimizer.
#include "mpc_qp_failure_context.hpp"
#include "mpc_source_exact_nlp.hpp"
#include "vehicle_model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

namespace {
constexpr int N=PREDICTION_HORIZON;
struct Record {
    double violation{std::numeric_limits<double>::infinity()};
    double minimum{-std::numeric_limits<double>::infinity()};
    int first_negative_stage{-1};
    int worst_stage{-1};
    double final_speed{};
    double final_progress{};
    bool valid{};
};
using Commands=std::array<MpcModelControl_t,N>;
Record evaluate(
    const sdu_apex::qp_fixture::FailureReplayContext &ctx,
    const Commands &commands)
{
    Record r{};
    r.violation=0.;
    r.minimum=std::numeric_limits<double>::infinity();
    MpcRtiState_t current=ctx.initial_state;
    double progress=ctx.progress_m;
    for(int k=0;k<N;++k) {
        MpcRtiState_t next{};
        double next_progress{};
        if(!mpc_rti_source_exact_stage(&current,progress,&commands[k],
            ctx.trajectory.data(),ctx.trajectory.size(),ctx.lap_length_m,
            ctx.prediction_dt,&next,&next_progress))return Record{};
        MpcRtiSourceCorridorCorners_t c{};
        if(!mpc_rti_source_exact_corridor_corners(
            &next.plant,ctx.trajectory.data(),ctx.trajectory.size(),
            ctx.lap_length_m,next_progress,&ctx.model,k+1,&c))return Record{};
        for(int corner=0;corner<4;++corner)
            for(double value:{c.left[corner],c.right[corner]}) {
                if(!std::isfinite(value))return Record{};
                if(value<r.minimum) {
                    r.minimum=value;
                    r.worst_stage=k+1;
                }
                const double negative=std::min(0.,value-.03);
                r.violation+=.5*negative*negative;
                if(value<.03 && r.first_negative_stage<0)
                    r.first_negative_stage=k+1;
            }
        current=next;
        progress=next_progress;
    }
    r.valid=true;
    r.final_speed=current.plant.u;
    r.final_progress=progress;
    return r;
}
}

int main(int argc,char **argv) {
    if(argc!=3) {
        std::cerr<<"usage: mpc_source_feasibility_witness_probe CASE_DIR "
                    "CONTROLS.txt\n";
        return 2;
    }
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (std::filesystem::path(argv[1])/"source_context.mctx").string(),
        ctx.get()))return 3;
    Commands nominal{};
    std::ifstream in(argv[2]);
    if(!in)return 4;
    for(auto &u:nominal)
        if(!(in>>u.steering_target_rad>>u.throttle_command))return 5;
    if(!vehicle_model_set_active_target_speed_ceiling(
        ctx->model.active_speed_ceiling_mps))return 6;
    const Record base=evaluate(*ctx,nominal);
    if(!base.valid)return 7;
    Record best=base;
    Commands best_commands=nominal;
    int feasible=0,valid=0,checked=0;
    const auto maybe=[&](const Commands &v){
        ++checked;
        const Record x=evaluate(*ctx,v);
        if(!x.valid)return;
        ++valid;
        if(x.minimum>=.03)++feasible;
        if(x.violation<best.violation ||
           (std::abs(x.violation-best.violation)<1.e-9 &&
            x.minimum>best.minimum)) {
            best=x;
            best_commands=v;
        }
    };
    // Straight physical-command policies over a single 750ms source
    // horizon, including true full-steer and coast/drive authority.
    for(int i=0;i<=24;++i) {
        const float steer=-.5235987f+i*(1.0471974f/24.f);
        for(float throttle:{.001f,.04f,.12f,.25f,.5f,1.f}) {
            Commands v=nominal;
            for(auto &u:v){u.steering_target_rad=steer;u.throttle_command=throttle;}
            maybe(v);
        }
    }
    // Two distinct original-source steering targets with one switch.
    for(int split:{3,6,9,12,15,18,22}) {
        for(int a=0;a<=12;++a) {
            const float s0=-.5235987f+a*(1.0471974f/12.f);
            for(int b=0;b<=12;++b) {
                const float s1=-.5235987f+b*(1.0471974f/12.f);
                for(float throttle:{.001f,.10f,.25f}) {
                    Commands v=nominal;
                    for(int k=0;k<N;++k) {
                        v[k].steering_target_rad=k<split?s0:s1;
                        v[k].throttle_command=throttle;
                    }
                    maybe(v);
                }
            }
        }
    }
    // Preserve the shape of the actual SQP warm start but test physical
    // steering and braking margin across the entire original horizon.
    for(int j=-24;j<=24;++j) {
        const float offset=.025f*j;
        for(float throttle:{.001f,.08f,.15f,.25f}) {
            Commands v=nominal;
            for(auto &u:v) {
                u.steering_target_rad=std::clamp(
                    u.steering_target_rad+offset,-.5235987f,.5235987f);
                u.throttle_command=throttle;
            }
            maybe(v);
        }
    }
    // Local block-coordinate exploration OF THE ORIGINAL EXACT PHYSICS
    // exclusively for a recoverability witness; never used by live MPC.
    for(int pass=0;pass<3;++pass) {
        bool improved=false;
        for(int start=0;start<N;start+=5) {
            const Record before=best;
            const Commands anchor=best_commands;
            for(float change:{-.25f,-.12f,-.05f,.05f,.12f,.25f})
                for(float throttle:{.001f,.08f,.25f}) {
                    Commands v=anchor;
                    for(int k=start;k<std::min(start+5,N);++k){
                        v[k].steering_target_rad=std::clamp(
                            v[k].steering_target_rad+change,
                            -.5235987f,.5235987f);
                        v[k].throttle_command=throttle;
                    }
                    maybe(v);
                }
            improved|=best.violation<before.violation-1.e-8;
        }
        if(!improved)break;
    }
    std::cout<<std::setprecision(10)
        <<"{\"original_minimum_clearance_m\":"<<base.minimum
        <<",\"original_violation_half_squared\":"<<base.violation
        <<",\"original_first_violating_stage\":"<<base.first_negative_stage
        <<",\"original_worst_stage\":"<<base.worst_stage
        <<",\"evaluated\":"<<checked
        <<",\"valid_rollouts\":"<<valid
        <<",\"physically_feasible_witnesses\":"<<feasible
        <<",\"best_minimum_clearance_m\":"<<best.minimum
        <<",\"best_violation_half_squared\":"<<best.violation
        <<",\"best_first_violating_stage\":"<<best.first_negative_stage
        <<",\"best_worst_stage\":"<<best.worst_stage
        <<",\"best_final_speed_mps\":"<<best.final_speed
        <<"}\n";
    std::ofstream out(std::filesystem::path(argv[1])/"witness.controls");
    if(!out)return 8;
    out<<std::setprecision(10);
    for(const auto &u:best_commands)
        out<<u.steering_target_rad<<' '<<u.throttle_command<<'\n';
    return 0;
}
