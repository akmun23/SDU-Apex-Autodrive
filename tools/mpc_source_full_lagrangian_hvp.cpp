// Offline condensed exact-source Lagrangian Hessian-vector oracle.
//
// At the SAME captured 26-physical-state command epoch, independently
// integrate the full 30-stage native source plant at U+h*v and U-h*v,
// then differentiate the ORIGINAL full-source racing cost and 60 physical
// footprint constraints by the existing 26-state source tangent.
//
// H_L v := (gradient L(U+h*v)-gradient L(U-h*v))/(2*h)
// L(U) := racing_cost(U) - lambda^T physical_wall_margin(U)
//
// This includes true source transition/actuator/slip coupling and active
// geometry curvature automatically, not a fitted bicycle or GN surrogate.
// Lambda is FROZEN for all evaluations; the default is a labeled merit
// proxy lambda_i=2*rho*max(0,-wall_i), NOT measured KKT duals.
//
// It intentionally runs OFFLINE; never call this costly numerical oracle
// directly in the 22ms ROS control callback.
#include "mpc_qp_failure_context.hpp"
#include "mpc_source_sensitivity.hpp"
#include "mpc_rti.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>
using json=nlohmann::json;
namespace fs=std::filesystem;
namespace {
constexpr int N=PREDICTION_HORIZON;
constexpr int D=2*N;
double elapsed_us(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-start).count();
}
bool load_direction(const fs::path &file,std::array<double,D> &v) {
    std::ifstream input(file);
    if(!input)return false;
    int column=0;
    std::string line;
    while(std::getline(input,line)) {
        if(line.empty()||line[0]=='#')continue;
        std::istringstream row(line);
        int index=-1; double value=NAN;
        if(!(row>>index>>value) || index!=column ||
           !std::isfinite(value)||column>=D)return false;
        v[column++]=value;
    }
    return column==D;
}
struct Eval {
    double cost{};
    std::array<double,D> wall{};
    std::array<double,D> gradient{};
    std::array<double,D> lagrangian_gradient{};
    std::array<double,D*D> hessian_gn{};
    double eval_us{},tangent_us{};
};
bool evaluate(
    const sdu_apex::qp_fixture::FailureReplayContext &ctx,
    const MpcRtiCycleConfiguration_t &cfg,
    const std::array<double,D> &y,
    const std::array<double,D> &lambda,
    Eval &out)
{
    MpcModelControl_t controls[N]{};
    const double scale[2]={.15,.25};
    for(int stage=0;stage<N;++stage) {
        const double steer=y[2*stage]*scale[0];
        const double throttle=y[2*stage+1]*scale[1];
        // The model is defined in the physical domain. Do not evaluate
        // clipped controls or Unity's distinct zero-throttle braking mode.
        if(steer>cfg.model.max_steering_rad+1.e-7 ||
           steer<-cfg.model.max_steering_rad-1.e-7 ||
           throttle<MPC_THROTTLE_MIN_DRIVE-1.e-8 ||
           throttle>MPC_THROTTLE_MAX+1.e-7)return false;
        controls[stage].steering_target_rad=float(steer);
        controls[stage].throttle_command=float(throttle);
    }
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    MpcRtiReference_t refs[N+1]{};
    const auto before=std::chrono::steady_clock::now();
    if(!mpc_rti_build_nominal_same_epoch(
        &ctx.initial_state,ctx.progress_m,nullptr,controls,N,
        ctx.trajectory.data(),ctx.trajectory.size(),
        ctx.lap_length_m,ctx.prediction_dt,N,
        &cfg.model,nominal.get(),refs))return false;
    float margins[N+1][2]{},minimum=NAN;
    if(!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
        nominal.get(),ctx.trajectory.data(),ctx.trajectory.size(),
        ctx.lap_length_m,&cfg,&out.cost,&minimum,margins))return false;
    for(int stage=1;stage<=N;++stage) {
        out.wall[2*(stage-1)]=margins[stage][0];
        out.wall[2*(stage-1)+1]=margins[stage][1];
    }
    out.eval_us=elapsed_us(before);
    auto derivative=std::make_unique<sdu_apex::source_sqp::FullSourceDerivatives>();
    const auto start_derivative=std::chrono::steady_clock::now();
    const bool valid=sdu_apex::source_sqp::full_source_derivatives(
        nominal.get(),ctx.trajectory.data(),ctx.trajectory.size(),
        ctx.lap_length_m,ctx.prediction_dt,&cfg,D,.002,out.cost,
        0.,derivative.get(),true,true);
    out.tangent_us=elapsed_us(start_derivative);
    if(!valid)return false;
    for(int j=0;j<D;++j) {
        out.gradient[j]=derivative->gradient[j];
        double v=derivative->gradient[j];
        for(int row=0;row<D;++row)
            v-=lambda[row]*derivative->footprint[row][j];
        out.lagrangian_gradient[j]=v;
        for(int k=0;k<D;++k)
            out.hessian_gn[j*D+k]=derivative->gn_hessian[j][k];
    }
    return true;
}
double dot(const std::array<double,D> &a,
           const std::array<double,D> &b) {
    double s=0.;
    for(int i=0;i<D;++i)s+=a[i]*b[i];
    return s;
}
double lagrangian(const Eval &a,const std::array<double,D> &lambda) {
    return a.cost-dot(a.wall,lambda);
}
} // namespace

int main(int argc,char **argv) {
    if(argc<3||argc>6) {
        std::cerr<<"usage: mpc_source_full_lagrangian_hvp CASE_DIR "
            "DIRECTION.txt [RHO=10000] "
            "[--qp-wall-duals=QP_DUALS.txt] "
            "[--controls=SOURCE_SEED.controls]\n";
        return 2;
    }
    auto ctx=std::make_unique<
        sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (fs::path(argv[1])/"source_context.mctx").string(),ctx.get()))
        return 3;
    std::array<double,D> v{};
    if(!load_direction(argv[2],v))return 4;
    double rho=10000.;
    std::string multiplier_file,control_file;
    for(int i=3;i<argc;++i) {
        const std::string option(argv[i]);
        constexpr std::string_view prefix="--qp-wall-duals=";
        constexpr std::string_view controls_prefix="--controls=";
        if(option.rfind(prefix,0)==0)
            multiplier_file=option.substr(prefix.size());
        else if(option.rfind(controls_prefix,0)==0)
            control_file=option.substr(controls_prefix.size());
        else rho=std::stod(option);
    }
    if(!std::isfinite(rho)||rho<0.||rho>1.e9)return 5;
    std::array<double,D> original{},lambda{};
    for(int k=0;k<N;++k) {
        original[2*k]=ctx->r1_nominal[k].steering_target_rad/.15;
        original[2*k+1]=ctx->r1_nominal[k].throttle_command/.25;
    }
    if(!control_file.empty()) {
        std::ifstream stream(control_file);
        if(!stream)return 11;
        for(int k=0;k<N;++k) {
            double steer=NAN,throttle=NAN;
            if(!(stream>>steer>>throttle))return 12;
            original[2*k]=steer/.15;
            original[2*k+1]=throttle/.25;
        }
    }
    MpcRtiCycleConfiguration_t cfg{};
    cfg.model=ctx->model;
    cfg.model.world_sweep_check=nullptr;
    cfg.model.world_sweep_context=nullptr;
    cfg.model.world_map_rejection_only=0;
    cfg.objective_schedule=&ctx->objective;
    Eval base{};
    if(!evaluate(*ctx,cfg,original,lambda,base))return 6;
    int active=0;
    if(!multiplier_file.empty()) {
        std::ifstream input(multiplier_file);
        if(!input)return 8;
        std::string line;
        int count=0;
        while(std::getline(input,line)) {
            if(line.empty()||line[0]=='#')continue;
            std::istringstream row(line);
            int index=-1;double value=NAN;
            if(!(row>>index>>value)||index!=count||count>=D||
               !std::isfinite(value)||value<0.)return 9;
            lambda[count++]=value;
            if(value>1.e-6)++active;
        }
        if(count!=D)return 10;
    } else {
        for(int k=0;k<D;++k)
            if(base.wall[k]<0.) {
                lambda[k]=2.*rho*(-base.wall[k]);
                ++active;
            }
    }
    // Rebuild baseline cost gradient with frozen duals; this evaluation
    // is used for cross-checking source local first-order consistency.
    if(!evaluate(*ctx,cfg,original,lambda,base))return 7;
    double gn_direction=0.;
    for(int i=0;i<D;++i)
        for(int j=0;j<D;++j)
            gn_direction+=v[i]*base.hessian_gn[i*D+j]*v[j];
    json result;
    result["schema"]="source-lagrangian-directional-hvp/v2";
    result["controls_source"]=control_file.empty()?"captured_r1_nominal":control_file;
    result["lambda_source"]=multiplier_file.empty()
        ?"frozen_wall_merit_proxy_not_KKT_dual"
        :"recorded_first_QP_wall_duals_not_nonlinear_KKT";
    result["wall_merit_rho"]=rho;
    result["active_wall_multipliers"]=active;
    result["qp_normalized_direction_norm"]=std::sqrt(dot(v,v));
    result["base_cost"]=base.cost;
    result["base_source_nominal_wall_V"]=0.;
    for(double m:base.wall)
        result["base_source_nominal_wall_V"]=
            double(result["base_source_nominal_wall_V"]) +
                std::pow(std::min(0.,m),2);
    result["gauss_newton_objective_directional_curvature"]=gn_direction;
    result["base_cost_evaluation_us"]=base.eval_us;
    result["base_full_source_tangent_us"]=base.tangent_us;
    result["scales"]=json::array();
    for(double h:{.0005,.001,.002,.005,.01}) {
        std::array<double,D> yp=original,ym=original;
        for(int j=0;j<D;++j) {
            yp[j]+=h*v[j];ym[j]-=h*v[j];
        }
        Eval pos{},neg{};
        json stage={{"h",h}};
        const bool ok1=evaluate(*ctx,cfg,yp,lambda,pos);
        const bool ok2=ok1 && evaluate(*ctx,cfg,ym,lambda,neg);
        stage["valid"]=ok1 && ok2;
        if(ok2) {
            const double second_scalar=(
                lagrangian(pos,lambda)-
                2.*lagrangian(base,lambda)+
                lagrangian(neg,lambda))/(h*h);
            std::array<double,D> hvp{};
            for(int j=0;j<D;++j)
                hvp[j]=(pos.lagrangian_gradient[j]-
                        neg.lagrangian_gradient[j])/(2.*h);
            stage["source_frozen_lagrangian_cost_second_fd"]=
                second_scalar;
            stage["source_frozen_lagrangian_gradient_hvp_dot_direction"]=
                dot(v,hvp);
            stage["local_second_consistency_abs_error"]=
                std::abs(second_scalar-dot(v,hvp));
            stage["source_racing_cost_second_fd"]=
                (pos.cost-2.*base.cost+neg.cost)/(h*h);
            stage["source_hvp"]=
                std::vector<double>(hvp.begin(),hvp.end());
            stage["source_evaluation_us"]=pos.eval_us+neg.eval_us;
            stage["source_tangent_us"]=pos.tangent_us+neg.tangent_us;
        }
        result["scales"].push_back(stage);
    }
    std::cout<<std::setprecision(17)<<result.dump(2)<<"\n";
    return 0;
}
