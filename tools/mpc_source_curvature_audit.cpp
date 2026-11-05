// Offline finite-difference HVP of the exact condensed source Lagrangian.
// All eight body corners, physical command limits and frozen racing samples
// retain their original row identity. No production numerical FD is introduced.
#include "mpc_qp_failure_context.hpp"
#include "mpc_source_exact_compact.hpp"
#include "mpc_source_exact_qp.hpp"
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <array>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>

using namespace sdu_apex::source_nlp;
using json=nlohmann::json;
constexpr int D=2*kStages;
using V=Eigen::Matrix<double,D,1>;
struct Oracle {
    std::unique_ptr<MpcRtiNominal_t> nominal{std::make_unique<MpcRtiNominal_t>()};
    std::unique_ptr<Evaluation> score{std::make_unique<Evaluation>()};
    std::unique_ptr<Decision> decision{std::make_unique<Decision>()};
    std::unique_ptr<ReducedSourceGradient> gradient{std::make_unique<ReducedSourceGradient>()};
    std::unique_ptr<SparseNonlinearProgram> program{std::make_unique<SparseNonlinearProgram>()};
    std::unique_ptr<SourceConstrainedQpResult> qp{std::make_unique<SourceConstrainedQpResult>()};
};
int main(int argc,char **argv) {
    if(argc<2 || argc>3) {
        std::cerr<<"usage: mpc_source_curvature_audit CASE_DIR [CONTROLS]\n";return 2;
    }
    auto ctx=std::make_unique<sdu_apex::qp_fixture::FailureReplayContext>();
    if(!sdu_apex::qp_fixture::read_failure_context(
        (std::filesystem::path(argv[1])/"source_context.mctx").string(),ctx.get()))return 3;
    std::array<MpcModelControl_t,kStages> controls{};
    std::copy_n(ctx->r1_nominal,kStages,controls.begin());
    if(argc==3) {
        std::ifstream in(argv[2]);
        for(auto &u:controls)if(!(in>>u.steering_target_rad>>u.throttle_command))return 4;
    }
    MpcRtiCycleConfiguration_t cfg{};cfg.model=ctx->model;cfg.objective_schedule=&ctx->objective;
    const auto evaluate=[&](const std::array<MpcModelControl_t,kStages> &u,Oracle &o) {
        auto &n=*o.nominal;
        n.states[0]=ctx->initial_state;n.progress[0]=ctx->progress_m;n.horizon=kStages;
        for(int k=0;k<kStages;++k) {
            n.controls[k]=u[k];
            if(!mpc_rti_source_exact_stage(&n.states[k],n.progress[k],&n.controls[k],
                ctx->trajectory.data(),ctx->trajectory.size(),ctx->lap_length_m,
                ctx->prediction_dt,&n.states[k+1],&n.progress[k+1]))return false;
        }
        n.valid=1;
        if(!decision_from_nominal(n,o.decision.get()) || !evaluate_existing_source_nominal(
            ctx->initial_state,ctx->progress_m,ctx->trajectory.data(),ctx->trajectory.size(),
            ctx->lap_length_m,ctx->prediction_dt,cfg,n,*o.decision,o.score.get()))return false;
        const ConstraintMeritWeights weights{0.,0.,.03};
        if(!source_exact_compact_qp_linearization(ctx->initial_state,ctx->progress_m,
            ctx->trajectory.data(),ctx->trajectory.size(),ctx->lap_length_m,
            ctx->prediction_dt,cfg,*o.decision,*o.score,n,weights,o.gradient.get(),o.program.get()))return false;
        SourceConstrainedQpOptions opts;
        opts.source_configuration=&cfg;opts.source_nominal=&n;
        opts.capture_curvature_model=true;
        return source_exact_constrained_qp_direction(*o.program,
            o.gradient->original_source_cost_control,u.data(),opts,o.qp.get());
    };
    Oracle base;
    if(!evaluate(controls,base))return 5;
    std::vector<double> lambda(base.qp->constraint_margins.size(),0.);
    for(size_t i=0;i<lambda.size();++i) {
        const int row=base.qp->constraint_row_indices[i];
        lambda[i]=base.qp->inequality_multipliers[row];
        if(base.program->rows[row].type==NonlinearRowType::BodyCorner)
            lambda[i]+=1.e6*std::max(0.,.03-base.qp->constraint_margins[i]);
    }
    const auto gradient=[&](const Oracle &o) {
        V g;
        for(int j=0;j<D;++j)g(j)=o.gradient->original_source_cost_control[j/2][j%2]*(j%2?.25:.15);
        for(size_t r=0;r<lambda.size();++r)
            for(int j=0;j<D;++j)g(j)-=lambda[r]*o.qp->constraint_jacobian_scaled[r*D+j];
        return g;
    };
    const auto lagrangian=[&](const Oracle &o) {
        double v=o.score->objective;
        for(size_t r=0;r<lambda.size();++r)v-=lambda[r]*o.qp->constraint_margins[r];
        return v;
    };
    const V g0=gradient(base);
    json report={{"schema","source-lagrangian-hvp/v1"},
        {"cost",base.score->objective},{"minimum_clearance_m",base.score->minimum_corridor_slack_m},
        {"multipliers","frozen hard QP duals plus current elastic reserve duals"},
        {"directions",json::array()}};
    for(int axis=0;axis<4;++axis) {
        V direction=V::Zero();
        for(int k=0;k<kStages;++k) {
            if(axis==0 || axis==2)direction(2*k)=std::sin(.37*(k+1));
            if(axis==1 || axis==2)direction(2*k+1)=std::cos(.29*(k+1));
            if(axis==3) {
                direction(2*k)=base.qp->direction[k][0]/.15;
                direction(2*k+1)=base.qp->direction[k][1]/.25;
            }
        }
        // Central FD is only meaningful inside the original physical domain.
        for(int j=0;j<D;++j) {
            const double value=j%2?controls[j/2].throttle_command:controls[j/2].steering_target_rad;
            if(j%2 ? value<.003 || value>.998 : std::abs(value)>ctx->model.max_steering_rad-.002)
                direction(j)=0.;
        }
        if(direction.norm()<1.e-12)continue;
        direction.normalize();
        V gn=V::Zero();
        for(int i=0;i<D;++i)
            for(int j=0;j<D;++j)gn(i)+=base.qp->racing_hessian_scaled[i*D+j]*direction(j);
        json samples=json::array();
        for(double epsilon:{.0005,.001,.002,.005}) {
            auto plus=controls,minus=controls;
            for(int j=0;j<D;++j) {
                float &a=j%2?plus[j/2].throttle_command:plus[j/2].steering_target_rad;
                float &b=j%2?minus[j/2].throttle_command:minus[j/2].steering_target_rad;
                const double change=epsilon*direction(j)*(j%2?.25:.15);
                a=static_cast<float>(a+change);b=static_cast<float>(b-change);
            }
            Oracle p,m;
            if(!evaluate(plus,p) || !evaluate(minus,m))return 6;
            const V hvp=(gradient(p)-gradient(m))/(2.*epsilon);
            const double second=(lagrangian(p)-2.*lagrangian(base)+lagrangian(m))/(epsilon*epsilon);
            const double first_fd=(lagrangian(p)-lagrangian(m))/(2.*epsilon);
            samples.push_back({{"epsilon",epsilon},{"source_hvp_norm",hvp.norm()},
                {"gn_hvp_norm",gn.norm()},{"relative_hvp_error",(hvp-gn).norm()/std::max(1.,hvp.norm())},
                {"source_directional_curvature",direction.dot(hvp)},
                {"gn_directional_curvature",direction.dot(gn)},
                {"lagrangian_second_difference",second},
                {"first_derivative_fd",first_fd},{"first_derivative_ad",g0.dot(direction)}});
        }
        report["directions"].push_back({{"name",std::array<const char *,4>{"steering","throttle","coupled","sqp"}[axis]},
                                         {"samples",samples}});
    }
    std::cout<<report.dump(2)<<'\n';
    return 0;
}
