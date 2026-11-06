#include "mpc_qp_failure_context.hpp"
#include "mpc_rti.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
// Offline viability *sample*, NOT a proof and NEVER a controller or
// publishable policy. Evaluate extreme positive-drive full-source controls.
int main(int argc,char**argv) {
    if(argc<2||argc>3) {
        std::cerr<<"usage: source_viability_probe CASE_DIR [--deep]\n";return 2;}
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
            (std::filesystem::path(argv[1])/"source_context.mctx").string(),
            ctx.get()))return 3;
    MpcRtiConfiguration_t model=ctx->model;
    model.world_sweep_check=nullptr;model.world_sweep_context=nullptr;
    const double max_steer=model.max_steering_rad;
    const double throttles[]={.001,.02,.05,.15,.3};
    auto rollout=std::make_unique<MpcRtiNominal_t>();
    MpcModelControl_t controls[PREDICTION_HORIZON]{};
    struct Result {
        double min09{-1e9},min12{-1e9},min30{-1e9};
        int first_failure{};
        double steer{}, throttle{};
        int change{}, sign{},method{};
        std::array<MpcModelControl_t,PREDICTION_HORIZON> plan{};
    };
    std::vector<Result> results;
    const auto evaluate=[&](int method,int sign,double steering,
                            double throttle,int change) {
        for(int k=0;k<PREDICTION_HORIZON;++k) {
            controls[k]=ctx->r1_nominal[k];
            const double forced=method==0?steering:
                method==1?(k<change?steering:-steering):
                method==2?(k<change?steering:ctx->r1_nominal[k].steering_target_rad):
                method==3?(k<change?ctx->r1_nominal[k].steering_target_rad:steering):
                (k<change?steering:std::copysign(max_steer,
                        ctx->r1_nominal[k].steering_target_rad));
            controls[k].steering_target_rad=std::clamp(
                float(forced),-model.max_steering_rad,model.max_steering_rad);
            controls[k].throttle_command=float(throttle);
        }
        MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
        if(!mpc_rti_build_nominal_same_epoch(
                &ctx->initial_state,ctx->progress_m,nullptr,controls,
                PREDICTION_HORIZON,ctx->trajectory.data(),
                ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
                PREDICTION_HORIZON,&model,rollout.get(),refs))return;
        Result result{};result.first_failure=-1;
        result.min09=result.min12=result.min30=1e9;
        for(int k=1;k<=PREDICTION_HORIZON;++k) {
            MpcRtiReference_t ref{};
            if(!mpc_rti_reference_at_progress(ctx->trajectory.data(),
                ctx->trajectory.size(),ctx->lap_length_m,
                rollout->progress[k],&model,&ref))return;
            float left=0,right=0;
            if(!mpc_rti_compute_corridor_slack(
                &rollout->states[k].plant,ref.left_bound,ref.right_bound,
                &model,k,&left,&right))return;
            const double m=std::min(left,right);
            if(k<=9)result.min09=std::min(result.min09,m);
            if(k<=12)result.min12=std::min(result.min12,m);
            result.min30=std::min(result.min30,m);
            if(m<0&&result.first_failure<0)result.first_failure=k;
        }
        result.steer=steering;result.throttle=throttle;
        result.method=method;result.change=change;result.sign=sign;
        std::copy_n(controls,PREDICTION_HORIZON,result.plan.begin());
        results.push_back(result);
    };
    // Holds, early reversal, early return to raceline, and delayed
    // extreme controls. Every candidate respects the physical steering
    // target range and exclusively positive-drive throttle.
    for(int method=0;method<4;++method)
        for(int sign:{-1,1})
            for(double fraction:{.25,.5,.75,1.})
                for(double throttle:throttles)
                    for(int change:{2,5,8,12,18}) {
                        evaluate(method,sign,sign*max_steer*fraction,
                                 throttle,change);
                    }
    if(argc==3&&std::string(argv[2])=="--deep") {
        std::mt19937 rng(20261103);
        std::normal_distribution<double> steer_dist(-.27,.13);
        std::uniform_int_distribution<int> switch_dist(2,21);
        std::uniform_real_distribution<double> uniform(-1.,1.);
        // Offline search only. The real controller MUST still solve its
        // own normal SQP; these sample families are not a recovery policy.
        // Each trial uses four piecewise-constant physical steering targets
        // with a legal positive throttle range.
        for(int trial=0;trial<18000;++trial) {
            int switches[3]={switch_dist(rng),switch_dist(rng),switch_dist(rng)};
            std::sort(switches,switches+3);
            double values[4]={};
            for(auto &v:values)v=std::clamp(steer_dist(rng),-max_steer,max_steer);
            const double throttle=(trial%8==0?.05:
                trial%8==1?.02:.001);
            for(int k=0;k<PREDICTION_HORIZON;++k) {
                const int seg=k<switches[0]?0:k<switches[1]?1:
                    k<switches[2]?2:3;
                controls[k].steering_target_rad=static_cast<float>(values[seg]);
                controls[k].throttle_command=static_cast<float>(throttle);
            }
            MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
            if(!mpc_rti_build_nominal_same_epoch(
                &ctx->initial_state,ctx->progress_m,nullptr,controls,
                PREDICTION_HORIZON,ctx->trajectory.data(),
                ctx->trajectory.size(),ctx->lap_length_m,ctx->prediction_dt,
                PREDICTION_HORIZON,&model,rollout.get(),refs))continue;
            Result result{};result.first_failure=-1;
            result.min09=result.min12=result.min30=1e9;
            for(int k=1;k<=PREDICTION_HORIZON;++k) {
                MpcRtiReference_t ref{};
                if(!mpc_rti_reference_at_progress(ctx->trajectory.data(),
                    ctx->trajectory.size(),ctx->lap_length_m,
                    rollout->progress[k],&model,&ref))break;
                float left=0,right=0;
                if(!mpc_rti_compute_corridor_slack(
                    &rollout->states[k].plant,ref.left_bound,
                    ref.right_bound,&model,k,&left,&right))break;
                double m=std::min(left,right);
                if(k<=9)result.min09=std::min(result.min09,m);
                if(k<=12)result.min12=std::min(result.min12,m);
                result.min30=std::min(result.min30,m);
                if(m<0 && result.first_failure<0)result.first_failure=k;
            }
            result.steer=values[0];result.throttle=throttle;
            result.method=4;result.change=switches[0];
            result.sign=trial;
            std::copy_n(controls,PREDICTION_HORIZON,result.plan.begin());
        results.push_back(result);
        }
    }
    if(argc==3&&std::string(argv[2])=="--scan") {
        for(int stage=0;stage<12;++stage) {
            for(double target:{-.5235987756,-.45,-.40,-.35,-.30,
                               -.26,-.23,-.20,-.15,-.10,0.0}) {
                for(int k=0;k<PREDICTION_HORIZON;++k)
                    controls[k]=ctx->r1_nominal[k];
                controls[stage].steering_target_rad=float(target);
                MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
                if(!mpc_rti_build_nominal_same_epoch(
                    &ctx->initial_state,ctx->progress_m,nullptr,controls,
                    PREDICTION_HORIZON,ctx->trajectory.data(),
                    ctx->trajectory.size(),ctx->lap_length_m,
                    ctx->prediction_dt,PREDICTION_HORIZON,&model,
                    rollout.get(),refs))continue;
                double minimum=1e9,at_stage9=1e9;
                for(int j=1;j<=PREDICTION_HORIZON;++j) {
                    MpcRtiReference_t ref{};
                    if(!mpc_rti_reference_at_progress(ctx->trajectory.data(),
                        ctx->trajectory.size(),ctx->lap_length_m,
                        rollout->progress[j],&model,&ref))break;
                    float left=0,right=0;
                    if(!mpc_rti_compute_corridor_slack(
                        &rollout->states[j].plant,ref.left_bound,
                        ref.right_bound,&model,j,&left,&right))break;
                    minimum=std::min(minimum,double(std::min(left,right)));
                    if(j==9)at_stage9=std::min(left,right);
                }
                std::cout<<"SINGLE_AXIS "<<stage<<" "<<target
                         <<" STAGE9 "<<at_stage9
                         <<" MIN30 "<<minimum<<"\n";
            }
        }
        return 0;
    }
    // Do not extrapolate from finite candidate sweeps to a formal
    // viability certificate. Report the best minimum clearance in each
    // prefix and the number of fully source-corridor feasible candidates.
    for(int horizon:{9,12,30}) {
        const auto margin=[&](const Result &r) {
            return horizon==9?r.min09:horizon==12?r.min12:r.min30;
        };
        auto it=std::max_element(results.begin(),results.end(),
            [&](const Result&a,const Result&b){return margin(a)<margin(b);});
        int feasible=std::count_if(results.begin(),results.end(),
            [&](const Result&r){return margin(r)>=0.;});
        if(horizon==30 && margin(*it)>=0.) {
            // The independent production validator is the authority for
            // ALL physical conditions. A positive corridor margin alone
            // cannot justify calling a plan publishable.
            MpcRtiCycleConfiguration_t configuration{};
            configuration.model=model;
            auto certified=std::make_unique<MpcRtiNominal_t>();
            int failure=-1;
            const auto status=mpc_rti_validate_candidate_plan(
                &ctx->initial_state,ctx->progress_m,it->plan.data(),
                ctx->trajectory.data(),ctx->trajectory.size(),
                ctx->lap_length_m,PREDICTION_HORIZON,ctx->prediction_dt,
                &configuration,certified.get(),&failure);
            std::cout<<"BEST_30_STRICT_SOURCE_STATUS "<<int(status)
                     <<" VALID "<<int(certified->valid)
                     <<" FAILURE_STAGE "<<failure<<"\n";
            if(status==MPC_RTI_ROLLOUT_OK && certified->valid) {
                // Data for offline comparison of source-SQP warm-start
                // sensitivity. Never an online control or fallback output.
                std::cout<<"BEST_30_CONTROLS";
                for(const auto &u:it->plan)
                    std::cout<<" "<<std::setprecision(9)
                             <<u.steering_target_rad<<","
                             <<u.throttle_command;
                std::cout<<"\n";
            }
        }
        std::cout<<std::setprecision(9)<<"HORIZON "<<horizon
                 <<" TESTED "<<results.size()<<" FEASIBLE "<<feasible
                 <<" BEST_MIN_CLEARANCE_M "<<margin(*it)
                 <<" FIRST_VIOLATION "<<it->first_failure
                 <<" METHOD "<<it->method<<" CHANGE "<<it->change
                 <<" STEERING "<<it->steer<<" THROTTLE "<<it->throttle<<"\n";
    }
    return 0;
}
