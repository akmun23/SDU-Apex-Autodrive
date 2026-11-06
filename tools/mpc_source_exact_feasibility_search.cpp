#include "mpc_qp_failure_context.hpp"
#include "mpc_source_sensitivity.hpp"
#include "mpc_rti.h"
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
using json=nlohmann::json;
namespace fs=std::filesystem;
namespace {
constexpr int N=PREDICTION_HORIZON;
constexpr int D=2*N;
using Vec=Eigen::Matrix<double,D,1>;
constexpr double SCALE[2]={.15,.25};
struct Result {
    double objective{INFINITY},min_wall{INFINITY},cost{INFINITY};
    Vec gradient=Vec::Zero();
    std::array<double,D> walls{};
    bool valid{},certified{};
    double tangent_us{};
};
struct Search {
    std::unique_ptr<sdu_apex::qp_fixture::FailureReplayContext> ctx=
        std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    MpcRtiCycleConfiguration_t cfg{};
    std::unique_ptr<MpcRtiNominal_t> nominal=
        std::make_unique<MpcRtiNominal_t>();
    std::unique_ptr<MpcRtiNominal_t> certified=
        std::make_unique<MpcRtiNominal_t>();
    std::unique_ptr<sdu_apex::source_sqp::FullSourceDerivatives> diff=
        std::make_unique<sdu_apex::source_sqp::FullSourceDerivatives>();
    int evaluations{},gradient_calls{},strict_checks{};
    bool start(const fs::path &case_dir) {
        if(!sdu_apex::qp_fixture::read_failure_context(
            (case_dir/"source_context.mctx").string(),ctx.get()))
            return false;
        cfg.model=ctx->model;
        cfg.model.world_sweep_check=nullptr;
        cfg.model.world_sweep_context=nullptr;
        cfg.model.world_map_rejection_only=0;
        cfg.objective_schedule=&ctx->objective;
        return true;
    }
    static Vec clamp(const Vec &candidate, const MpcRtiConfiguration_t &model) {
        Vec value=candidate;
        for(int j=0;j<D;++j) {
            const double low=j%2
                ?double(MPC_THROTTLE_MIN_DRIVE)/SCALE[1]
                :-double(model.max_steering_rad)/SCALE[0];
            const double high=j%2
                ?double(MPC_THROTTLE_MAX)/SCALE[1]
                :double(model.max_steering_rad)/SCALE[0];
            value[j]=std::clamp(value[j],low,high);
        }
        return value;
    }
    bool evaluate(const Vec &candidate,Result &out,bool gradient) {
        ++evaluations;
        const Vec y=clamp(candidate,cfg.model);
        MpcModelControl_t controls[N]{};
        for(int k=0;k<N;++k) {
            controls[k].steering_target_rad=float(y[2*k]*SCALE[0]);
            controls[k].throttle_command=float(y[2*k+1]*SCALE[1]);
        }
        MpcRtiReference_t references[N+1]{};
        if(!mpc_rti_build_nominal_same_epoch(
                &ctx->initial_state,ctx->progress_m,nullptr,
                controls,N,ctx->trajectory.data(),ctx->trajectory.size(),
                ctx->lap_length_m,ctx->prediction_dt,N,&cfg.model,
                nominal.get(),references))
            return false;
        float margins[N+1][2]{},minimum=NAN;
        if(!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
                nominal.get(),ctx->trajectory.data(),ctx->trajectory.size(),
                ctx->lap_length_m,&cfg,&out.cost,&minimum,margins))
            return false;
        out.min_wall=INFINITY;
        out.objective=0.;
        for(int stage=1;stage<=N;++stage)
            for(int side=0;side<2;++side) {
                const int row=2*(stage-1)+side;
                const double wall=margins[stage][side];
                out.walls[row]=wall;
                out.min_wall=std::min(out.min_wall,wall);
                const double violated=std::min(0.,wall);
                out.objective+=violated*violated;
            }
        if(!std::isfinite(out.objective))return false;
        out.valid=true;
        if(out.objective<1.e-10) {
            ++strict_checks;
            int failure=-1;
            out.certified=mpc_rti_validate_candidate_plan(
                &ctx->initial_state,ctx->progress_m,controls,
                ctx->trajectory.data(),ctx->trajectory.size(),
                ctx->lap_length_m,N,ctx->prediction_dt,&cfg,
                certified.get(),&failure)==MPC_RTI_ROLLOUT_OK;
            if(out.certified)return true;
        }
        if(!gradient)return true;
        ++gradient_calls;
        *diff={};
        const auto start=std::chrono::steady_clock::now();
        if(!sdu_apex::source_sqp::full_source_derivatives(
             nominal.get(),ctx->trajectory.data(),ctx->trajectory.size(),
             ctx->lap_length_m,ctx->prediction_dt,&cfg,D,.002,out.cost,
             0.,diff.get(),false,true,true))
            return false;
        out.tangent_us=std::chrono::duration<double,std::micro>(
            std::chrono::steady_clock::now()-start).count();
        for(int col=0;col<D;++col) {
            double value=0.;
            for(int row=0;row<D;++row)
                value+=2.*std::min(0.,out.walls[row])*
                   diff->footprint[row][col];
            out.gradient[col]=value;
        }
        return out.gradient.allFinite();
    }
};
struct Memory {
    Vec s{},y{};
    double sy{};
};
Vec projected_gradient(const Vec &x,const Vec &gradient,
                       const MpcRtiConfiguration_t &model) {
    Vec g=gradient;
    for(int j=0;j<D;++j) {
        const double lo=j%2?double(MPC_THROTTLE_MIN_DRIVE)/SCALE[1]:
                                -double(model.max_steering_rad)/SCALE[0];
        const double hi=j%2?double(MPC_THROTTLE_MAX)/SCALE[1]:
                                double(model.max_steering_rad)/SCALE[0];
        if((x[j]<=lo+1.e-7 && g[j]>0.)||
           (x[j]>=hi-1.e-7 && g[j]<0.))
            g[j]=0.;
    }
    return g;
}
Vec lbfgs_direction(const Vec &g,const std::vector<Memory> &memory) {
    Vec v=g;
    std::vector<double> alpha(memory.size());
    for(int i=int(memory.size())-1;i>=0;--i) {
        alpha[i]=memory[i].s.dot(v)/memory[i].sy;
        v.noalias()-=alpha[i]*memory[i].y;
    }
    double initial=1.;
    if(!memory.empty()) {
        const auto &last=memory.back();
        initial=std::clamp(last.sy/
            std::max(last.y.squaredNorm(),1.e-16),1.e-6,1.e3);
    }
    v*=initial;
    for(std::size_t i=0;i<memory.size();++i) {
        const double beta=memory[i].y.dot(v)/memory[i].sy;
        v.noalias()+=memory[i].s*(alpha[i]-beta);
    }
    return -v;
}
bool load_controls(const fs::path &file,Vec &value) {
    std::ifstream input(file);
    if(!input)return false;
    for(int stage=0;stage<N;++stage) {
        double steer=NAN,throttle=NAN;
        if(!(input>>steer>>throttle))return false;
        value[2*stage]=steer/SCALE[0];
        value[2*stage+1]=throttle/SCALE[1];
    }
    return true;
}
void save_controls(const fs::path &file,const Vec &value) {
    std::ofstream output(file);
    output<<std::setprecision(12);
    for(int k=0;k<N;++k)
        output<<float(value[2*k]*SCALE[0])<<" "
              <<float(value[2*k+1]*SCALE[1])<<"\n";
}
void run(Search &search,const Vec &seed,int max_iteration,
         std::uint32_t seed_index,const fs::path &output,
         json &history,double &best_global,Vec &best_y) {
    Vec x=Search::clamp(seed,search.cfg.model);
    Result state{};
    if(!search.evaluate(x,state,true))return;
    std::vector<Memory> memory;
    double initial_value=state.objective;
    int it=0,accepted=0;
    for(;it<max_iteration && !state.certified;++it) {
        if(state.gradient.lpNorm<Eigen::Infinity>()<1.e-10)break;
        Vec g=projected_gradient(x,state.gradient,search.cfg.model);
        Vec d=memory.empty()?-g:lbfgs_direction(g,memory);
        // Strict projected directional descent, including active bounds.
        const Vec direction=projected_gradient(x,-d,search.cfg.model);
        d=-direction;
        if(!d.allFinite()||g.dot(d)>-1.e-12 ||
           d.norm()<1.e-9)d=-g;
        const double maxabs=d.cwiseAbs().maxCoeff();
        if(maxabs<=1.e-9)break;
        d/=maxabs;
        bool found=false;
        Vec candidate{};
        Result next{};
        for(double step:{1.,.5,.25,.125,.0625,.03125,
                         .015625,.0078125,.00390625,.001953125,
                         .0009765625,.00048828125,.000244140625}) {
            candidate=Search::clamp(x+step*d,search.cfg.model);
            const Vec displacement=candidate-x;
            if(displacement.norm()<1.e-9)continue;
            Result checked{};
            if(!search.evaluate(candidate,checked,false))continue;
            if(checked.certified ||
               checked.objective<state.objective-
                   1.e-6*std::max(1.e-7,-state.gradient.dot(displacement))) {
                next=checked;
                found=true;
                break;
            }
        }
        if(!found) {
            memory.clear();
            break;
        }
        if(!next.certified && !search.evaluate(candidate,next,true))break;
        if(!next.certified) {
            const Vec displacement=candidate-x;
            const Vec gradient_change=next.gradient-state.gradient;
            const double sy=displacement.dot(gradient_change);
            if(sy>1.e-10 && std::isfinite(sy)) {
                if(memory.size()>=10)memory.erase(memory.begin());
                memory.push_back({displacement,gradient_change,sy});
            } else memory.clear();
        }
        ++accepted;x=candidate;state=next;
        if(state.objective<best_global) {
            best_global=state.objective;best_y=x;
            save_controls(output,best_y);
        }
        if(it%15==0 || state.certified)
            std::cout<<"RESTART="<<seed_index<<" iteration="<<it
                     <<" violation="<<std::setprecision(10)
                     <<state.objective<<" min_wall="<<state.min_wall
                     <<" accepted="<<accepted
                     <<" strict="<<state.certified<<"\n"<<std::flush;
    }
    history.push_back({{"restart",seed_index},{"iterations",it},
         {"accepted",accepted},{"initial_violation_m2",initial_value},
         {"final_violation_m2",state.objective},
         {"final_min_wall_m",state.min_wall},
         {"strict_certified",state.certified}});
    if(state.objective<best_global) {
        best_global=state.objective;best_y=x;
        save_controls(output,best_y);
    }
}
} //namespace
int main(int argc,char **argv) {
    if(argc<3||argc>6) {
        std::cerr<<"usage: mpc_source_exact_feasibility_search CASE_DIR "
             "OUT_CONTROLS.txt [iterations=100] [restarts=1] "
             "[start_controls.txt]\n";return 2;
    }
    Search search;
    if(!search.start(argv[1]))return 3;
    const int max_iterations=argc>3?
        std::clamp(std::atoi(argv[3]),1,500):100;
    const int restarts=argc>4?
        std::clamp(std::atoi(argv[4]),1,64):1;
    Vec starting;
    for(int k=0;k<N;++k) {
        starting[2*k]=search.ctx->r1_nominal[k].steering_target_rad/SCALE[0];
        starting[2*k+1]=search.ctx->r1_nominal[k].throttle_command/SCALE[1];
    }
    if(argc>5 && !load_controls(argv[5],starting))return 4;
    std::mt19937 generator(9001);
    std::normal_distribution<double> unit(0.,1.);
    double best=INFINITY;
    Vec best_y=starting;
    json history=json::array();
    for(int restart=0;restart<restarts;++restart) {
        Vec start=starting;
        if(restart>0) {
            // Every trial remains in the ORIGINAL physical vehicle
            // steering and positive drive throttle domain. This is an
            // offline existence search, NEVER a live controller policy.
            const double amplitude=restart%4==0?.1:
                                   restart%4==1?.35:
                                   restart%4==2?.75:1.25;
            for(int j=0;j<D;++j)
                start[j]+=amplitude*unit(generator);
        }
        run(search,start,max_iterations,restart,argv[2],
            history,best,best_y);
        if(best<1.e-12)break;
    }
    std::cout<<std::setprecision(15)
       <<json({{"case",argv[1]},
        {"best_violation_m2",best},
        {"source_evaluations",search.evaluations},
        {"gradient_calls",search.gradient_calls},
        {"strict_certifications",search.strict_checks},
        {"trials",history}}).dump(2)<<std::endl;
    return 0;
}
