#include "mpc_source_reduced_sqp.hpp"
#include "mpc_source_sensitivity.hpp"

// The 60-variable dense source QP is intentionally single-threaded.
// The source-plant tangent batches use three independent OpenMP workers,
// but Eigen spawning an additional QP worker team oversubscribes pinned
// control CPUs and introduces multi-millisecond (or worse) timing spikes.
// This changes scheduling only, not the physical model or QP arithmetic.
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <cstring>

/*
 * Source-exact 60-control reduced SQP.
 *
 * Only the 60 normalized commands are decision variables. Hidden wheel,
 * jounce, roll, pitch, command FIFO, yaw and world progress are always
 * recomputed with the identical checked-in source model, never dropped by
 * a 23-state tangent reduction.
 *
 * Infeasible starts use a lexicographic PHASE I: minimize the sum of 60
 * affine physical footprint slacks with all 120 actuator-rate inequalities
 * and 120 absolute control bounds hard, then minimize the correction norm
 * at the achieved slack optimum. This is NOT a wall-weight heuristic.
 * Feasible starts use a convexified diagonal exact-source cost Hessian,
 * subsequently corrected by a damped BFGS secant from exact-source costs.
 *
 * The local QP is never a safety certificate: only an independent strict
 * 30-stage nonlinear source validation can authorize the returned plan.
 */
namespace sdu_apex::source_sqp {
namespace {
constexpr int N=PREDICTION_HORIZON;
constexpr int D=2*N;
constexpr int BaseW=2*N;
constexpr int TailW=6;
constexpr int W=BaseW+TailW;
constexpr int R=4*N;
constexpr double kMarginBuffer=.00025;
constexpr double kScale[2]={.15,.25};
constexpr double kAlpha[]={1.,.5,.25,.125,.0625,.03125,.015625,
    .0078125,.00390625,.001953125,.0009765625};
using Vec=Eigen::Matrix<double,D,1>;
using Mat=Eigen::Matrix<double,D,D>;
using Wall=Eigen::Matrix<double,W,1>;
using Rate=Eigen::Matrix<double,R,1>;
using JWall=Eigen::Matrix<double,W,D>;
using JRate=Eigen::Matrix<double,R,D>;
using Clock=std::chrono::steady_clock;
double now_us() {
    return std::chrono::duration<double,std::micro>(
        Clock::now().time_since_epoch()).count();
}
bool finite(double x) { return std::isfinite(x); }
Vec to_y(const MpcModelControl_t controls[N]) {
    Vec y;
    for(int k=0;k<N;++k) {
        y[2*k]=controls[k].steering_target_rad/kScale[0];
        y[2*k+1]=controls[k].throttle_command/kScale[1];
    }
    return y;
}
void from_y(const Vec &y,MpcModelControl_t controls[N]) {
    for(int k=0;k<N;++k) {
        controls[k].steering_target_rad=float(y[2*k]*kScale[0]);
        controls[k].throttle_command=float(y[2*k+1]*kScale[1]);
    }
}
double wall_violation(const Wall &w,bool tightened) {
    const auto deficit=(w.array()-(tightened?kMarginBuffer:0.))
        .min(0.).matrix();
    return deficit.squaredNorm();
}
double rate_violation(const Rate &r) {
    return r.array().min(0.).matrix().squaredNorm();
}
double minimum_clearance(const Wall &w) {
    return w.minCoeff();
}
struct Sample {
    Wall wall=Wall::Zero();
    Rate rate=Rate::Zero();
    double stages[N+1]{};
    double cost{NAN};
    double wall_v{NAN},rate_v{NAN};
    double margin{NAN};
    double tail_wall_v{};
    double tail_minimum{0.};
    int failure_stage{-1};
    bool strict_valid{};
};
struct Evaluation {
    const MpcRtiState_t *initial{};
    double progress{};
    const MpcTrajectorySample_t *trajectory{};
    size_t trajectory_count{};
    double lap_length{};
    float dt{};
    const MpcRtiCycleConfiguration_t *cfg{};
    Statistics *stats{};
    double deadline_us{};
    int terminal_lookahead_steps{};
    bool reuse_unperturbed_fd_prefix{};
    bool captured_source_prefix{};
    // Reusable buffers avoid allocating another 30-stage trajectory for
    // every one of the 120 source derivative probes.
    std::unique_ptr<MpcRtiNominal_t> nominal{
        std::make_unique<MpcRtiNominal_t>()};
    std::unique_ptr<MpcRtiNominal_t> strict{
        std::make_unique<MpcRtiNominal_t>()};
    std::unique_ptr<MpcRtiNominal_t> tail;
    std::unique_ptr<MpcRtiNominal_t> source_prefix;
    std::unique_ptr<MpcRtiNominal_t> suffix;
    void capture_source_prefix() {
        if(reuse_unperturbed_fd_prefix && nominal->valid &&
           nominal->horizon==N) {
            if(!source_prefix)
                source_prefix=std::make_unique<MpcRtiNominal_t>();
            *source_prefix=*nominal;
            captured_source_prefix=true;
        } else {
            captured_source_prefix=false;
        }
    }

    bool evaluate(const Vec &y,bool strict_check,Sample &sample,
                  int first_changed_command_stage=0) {
        const double begin=now_us();
        // Every expensive finite-difference oracle checks its REAL absolute
        // deadline before starting. A rejected epoch never publishes an
        // intermediate source-infeasible plan.
        if(deadline_us>0. && begin>=deadline_us) {
            if(stats)stats->deadline_exhausted=true;
            return false;
        }
        MpcModelControl_t commands[N]{};
        from_y(y,commands);
        MpcRtiReference_t refs[N+1]{};
        // No source-plant substitution or affine rollout is allowed here.
        if(reuse_unperturbed_fd_prefix && captured_source_prefix &&
           !strict_check && first_changed_command_stage>0) {
            const int k=first_changed_command_stage;
            if(k>=N || !source_prefix->valid || source_prefix->horizon!=N)
                return false;
            if(!suffix)suffix=std::make_unique<MpcRtiNominal_t>();
            MpcRtiReference_t suffix_refs[N+1]{};
            if(!mpc_rti_build_nominal_same_epoch(
                    &source_prefix->states[k],source_prefix->progress[k],
                    nullptr,commands+k,N-k,trajectory,trajectory_count,
                    lap_length,dt,N-k,&cfg->model,suffix.get(),
                    suffix_refs))
                return false;
            *nominal=*source_prefix;
            for(int stage=k;stage<=N;++stage) {
                nominal->states[stage]=suffix->states[stage-k];
                nominal->progress[stage]=suffix->progress[stage-k];
                // The full-horizon racing objective uses stage-indexed
                // source-progress references, not the suffix's artificial
                // stage-0 speed initialization.
                MpcRtiReference_t target{};
                if(!mpc_rti_reference_at_progress(
                        trajectory,trajectory_count,lap_length,
                        nominal->progress[stage],&cfg->model,&target))
                    return false;
                nominal->speed_reference[stage]=
                    stage==0?source_prefix->speed_reference[0]:target.u;
            }
            for(int stage=k;stage<N;++stage)
                nominal->controls[stage]=commands[stage];
            nominal->valid=1;
            nominal->horizon=N;
            nominal->speed_reference_valid=1;
        } else {
            if(!mpc_rti_build_nominal_same_epoch(initial,progress,nullptr,
                    commands,N,trajectory,trajectory_count,lap_length,dt,N,
                    &cfg->model,nominal.get(),refs))
                return false;
        }
        MpcRtiCycleConfiguration_t score_config=*cfg;
        // The same immutable objective is provided to every source probe.
        float computed_slacks[N+1][2]{};
        if(!mpc_rti_score_candidate_plan_and_corridor_for_diagnostics(
                nominal.get(),trajectory,trajectory_count,lap_length,
                &score_config,&sample.cost,&score_min_dummy_,
                computed_slacks))return false;

        // The scorer already evaluated physical left and right footprint
        // margins for stages 1..N. Only measured stage zero needs a separate
        // corridor query. No altered weights, geometry or relaxed margins.
        for(int k=0;k<=N;++k) {
            float left=0.f,right=0.f;
            if(k>0) {
                left=computed_slacks[k][0];
                right=computed_slacks[k][1];
            } else {
                MpcRtiReference_t r{};
                if(!mpc_rti_reference_at_progress(trajectory,
                        trajectory_count,lap_length,nominal->progress[k],
                        &cfg->model,&r) ||
                   !mpc_rti_compute_corridor_slack(&nominal->states[k].plant,
                        r.left_bound,r.right_bound,&cfg->model,k,
                        &left,&right))return false;
            }
            if(!std::isfinite(left)||!std::isfinite(right))return false;
            sample.stages[k]=std::min(double(left),double(right));
            if(k>0) {
                sample.wall[2*(k-1)]=left;
                sample.wall[2*(k-1)+1]=right;
            }
        }
        if(sample.stages[0]<-1e-10)return false;
        const auto &model=cfg->model;
        // Exact optional three-stage source continuation under the
        // last command. Original 30-stage QP remains unchanged if disabled.
        sample.wall.tail(TailW).setConstant(1.e3);
        if(terminal_lookahead_steps==3) {
            if(!tail)tail=std::make_unique<MpcRtiNominal_t>();
            MpcModelControl_t held[N]{};
            for(int i=0;i<3;++i)held[i]=commands[N-1];
            MpcRtiReference_t tail_refs[N+1]{};
            if(!mpc_rti_build_nominal_same_epoch(
                &nominal->states[N],nominal->progress[N],nullptr,held,3,
                trajectory,trajectory_count,lap_length,dt,3,
                &model,tail.get(),tail_refs))return false;
            for(int i=1;i<=3;++i) {
                MpcRtiReference_t ref{};
                if(!mpc_rti_reference_at_progress(trajectory,trajectory_count,
                    lap_length,tail->progress[i],&model,&ref))return false;
                float left=NAN,right=NAN;
                if(!mpc_rti_compute_corridor_slack(&tail->states[i].plant,
                    ref.left_bound,ref.right_bound,&model,N,
                    &left,&right))return false;
                sample.wall[BaseW+2*(i-1)]=left;
                sample.wall[BaseW+2*(i-1)+1]=right;
            }
            sample.tail_minimum=sample.wall.tail(TailW).minCoeff();
            sample.tail_wall_v=sample.wall.tail(TailW).array()
                .min(0.).square().sum();
        }
        const double rack_step=std::min(
            double(model.max_steering_command_step_rad),
            .995*3.2*double(dt));
        for(int k=0;k<N;++k) {
            const double ds=commands[k].steering_target_rad-
                nominal->states[k].plant.actual_steering_angle;
            const double throttle_delta=commands[k].throttle_command-
                nominal->states[k].plant.delayed_throttle_command_2;
            sample.rate[4*k]=rack_step-ds;
            sample.rate[4*k+1]=rack_step+ds;
            sample.rate[4*k+2]=model.max_throttle_command_step-throttle_delta;
            sample.rate[4*k+3]=model.max_throttle_command_step+throttle_delta;
        }
        sample.wall_v=wall_violation(sample.wall,false);
        sample.rate_v=rate_violation(sample.rate);
        sample.margin=minimum_clearance(sample.wall);
        if(strict_check) {
            int stage=-1;
            const auto status=mpc_rti_validate_candidate_plan(
                initial,progress,commands,trajectory,trajectory_count,
                lap_length,N,dt,cfg,strict.get(),&stage);
            sample.strict_valid=status==MPC_RTI_ROLLOUT_OK;
            sample.failure_stage=stage;
            if(sample.strict_valid && terminal_lookahead_steps==3) {
                MpcModelControl_t held[N]{};
                for(int i=0;i<3;++i)held[i]=commands[N-1];
                MpcRtiCycleConfiguration_t tail_cfg=*cfg;
                tail_cfg.terminal_viability=0;
                tail_cfg.objective_schedule=nullptr;
                int failure=-1;
                const auto tail_status=mpc_rti_validate_candidate_plan(
                    &nominal->states[N],nominal->progress[N],held,
                    trajectory,trajectory_count,lap_length,3,dt,
                    &tail_cfg,strict.get(),&failure);
                if(tail_status!=MPC_RTI_ROLLOUT_OK) {
                    sample.strict_valid=false;
                    sample.failure_stage=N+failure;
                }
            }
        }
        if(stats) {
            ++stats->source_evaluations;
            stats->exact_source_evaluation_us+=now_us()-begin;
        }
        return finite(sample.cost) && sample.wall.allFinite() &&
            sample.rate.allFinite();
    }
    float score_min_dummy_{NAN};
};
struct Derivatives {
    Vec gradient=Vec::Zero();
    Mat hessian=Mat::Identity();
    JWall wall=JWall::Zero();
    JRate rate=JRate::Zero();
};
bool deriv(Evaluation &eval,const Vec &y,const Sample &base,
    double h,bool feasible,int decision_count,
    bool use_full_source_forward_sensitivities,Derivatives &out) {
    if(use_full_source_forward_sensitivities) {
        // No heap churn for ~90 kB of derivative columns every callback.
        // The output is always reset before propagation; thread-local means
        // independent ROS executor threads never share intermediate data.
        static thread_local auto result=
            std::make_unique<FullSourceDerivatives>();
        *result={};
        if(!full_source_derivatives(eval.nominal.get(),eval.trajectory,
                eval.trajectory_count,eval.lap_length,eval.dt,eval.cfg,
                decision_count,h,base.cost,eval.deadline_us,result.get())) {
            if(result->deadline_exhausted && eval.stats)
                eval.stats->deadline_exhausted=true;
            return false;
        }
        for(int j=0;j<decision_count;++j) {
            out.gradient[j]=result->gradient[j];
            out.hessian(j,j)=feasible?
                result->positive_diagonal_curvature[j]:1.;
            for(int row=0;row<BaseW;++row)
                out.wall(row,j)=result->footprint[row][j];
            for(int row=0;row<R;++row)
                out.rate(row,j)=result->actuator_rate[row][j];
        }
        if(eval.stats) {
            eval.stats->source_forward_sensitivity_intervals+=
                result->dual_plant_intervals;
            eval.stats->source_tangent_objective_evaluations+=
                result->score_evaluations;
            eval.stats->full_forward_dual_plant_us+=
                result->dual_plant_us;
            eval.stats->full_forward_tangent_scoring_us+=
                result->tangent_scoring_us;
        }
        return out.gradient.allFinite()&&out.hessian.allFinite()&&
            out.wall.allFinite()&&out.rate.allFinite();
    }
    for(int j=0;j<decision_count;++j) {
        Vec plus=y,minus=y;
        plus[j]+=h;minus[j]-=h;
        Sample a{},b{};
        const int changed_stage=j/2;
        if(!eval.evaluate(plus,false,a,changed_stage) ||
           !eval.evaluate(minus,false,b,changed_stage))
            return false;
        out.wall.col(j)=(a.wall-b.wall)/(2*h);
        out.rate.col(j)=(a.rate-b.rate)/(2*h);
        out.gradient[j]=(a.cost-b.cost)/(2*h);
        // Central full-source directional second derivative: all hidden
        // plant states and progress are included. Preserve only nonnegative
        // diagonal curvature in the convex local QP, then use BFGS to
        // introduce source-observed inter-command couplings.
        const double curvature=(a.cost-2*base.cost+b.cost)/(h*h);
        out.hessian(j,j)=feasible
            ?std::clamp(curvature,1.e-4,1.e7):1.;
    }
    return out.gradient.allFinite()&&out.hessian.allFinite()&&
        out.wall.allFinite()&&out.rate.allFinite();
}
struct Qp {
    Eigen::MatrixXd H,A;
    Eigen::VectorXd q,b;
};
struct QpAnswer {
    Eigen::VectorXd x;
    int iterations{},status{};
    double max_violation{INFINITY},relative_dual{INFINITY},gap{INFINITY};
};
double max_violation(const Qp &qp,const Eigen::VectorXd &x) {
    if(x.size()!=qp.q.size() || !x.allFinite())return INFINITY;
    return std::max(0.,(qp.A*x-qp.b).maxCoeff());
}
double fraction(const Eigen::VectorXd &x,
                const Eigen::VectorXd &delta,double tau) {
    double a=1.;
    for(int i=0;i<x.size();++i)
        if(delta[i]<0.)a=std::min(a,-tau*x[i]/delta[i]);
    return std::clamp(a,0.,1.);
}
bool newton(const Qp &qp,const Eigen::LDLT<Eigen::MatrixXd> &ldlt,
    const Eigen::VectorXd &s,const Eigen::VectorXd &lambda,
    const Eigen::VectorXd &rd,const Eigen::VectorXd &rp,
    const Eigen::VectorXd &rc,
    Eigen::VectorXd &dx,Eigen::VectorXd &ds,Eigen::VectorXd &dlambda) {
    // Both Mehrotra predictor and corrector use the SAME (H + A' D A)
    // coefficient matrix at this IPM iteration. Factor only once.
    const Eigen::VectorXd rhs=-rd+qp.A.transpose()*
        ((rc.array()-lambda.array()*rp.array())/s.array()).matrix();
    dx=ldlt.solve(rhs);
    if(ldlt.info()!=Eigen::Success || !dx.allFinite())return false;
    ds=-rp-qp.A*dx;
    dlambda=(-rc.array()-lambda.array()*ds.array())/s.array();
    return ds.allFinite() && dlambda.allFinite();
}
// Dense primal-dual Mehrotra IPM, all inequalities in original units.
// No "solved" flag is accepted without an independent primal certificate.
QpAnswer solve_qp(const Qp &qp,int maximum,double deadline_us=0.) {
    QpAnswer result{};
    const int m=qp.b.size(),n=qp.q.size();
    if(n==0||m==0||qp.H.rows()!=n||qp.H.cols()!=n||
       qp.A.rows()!=m||qp.A.cols()!=n||!qp.H.allFinite()||
       !qp.q.allFinite()||!qp.A.allFinite()||!qp.b.allFinite())return result;
    Eigen::VectorXd x=Eigen::VectorXd::Zero(n);
    Eigen::VectorXd s=Eigen::VectorXd::Ones(m);
    Eigen::VectorXd lambda=Eigen::VectorXd::Ones(m);
    const Eigen::VectorXd ax=qp.A*x;
    for(int i=0;i<m;++i)
        s[i]=std::max(1.,qp.b[i]-ax[i]+1.);
    // Reuse all Eigen workspaces and the LDLT factor buffer through the
    // interior-point iterations. This preserves the original normal-matrix
    // expression and its strict KKT tests, unlike the rejected structured
    // numerical variant.
    Eigen::ArrayXd weights(m);
    Eigen::MatrixXd weighted(m,n),W(n,n);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(n);
    Eigen::VectorXd rd(n),rp(m),rc(m),corr(m);
    Eigen::VectorXd dx(n),ds(m),dl(m),step(n),sstep(m),lstep(m);
    for(int it=0;it<maximum;++it) {
        // The solver has no right to consume an entire control deadline
        // trying to repair an already impossible physical corner.
        if(deadline_us>0. && now_us()>=deadline_us) {
            result.status=-2; // deadline, not numerical feasibility
            break;
        }
        rd=qp.H*x+qp.q+qp.A.transpose()*lambda;
        rp=qp.A*x+s-qp.b;
        const double mu=s.dot(lambda)/m;
        const double dualnorm=rd.lpNorm<Eigen::Infinity>();
        const double scale=std::max({1.,qp.q.lpNorm<Eigen::Infinity>(),
            (qp.H*x).lpNorm<Eigen::Infinity>(),
            (qp.A.transpose()*lambda).lpNorm<Eigen::Infinity>()});
        result.iterations=it;
        result.relative_dual=dualnorm/scale;
        result.gap=mu;
        const double violation=rp.lpNorm<Eigen::Infinity>();
        if(violation<1.e-7 && result.relative_dual<1.e-7 && mu<1.e-7) {
            result.status=1;break;
        }
        if(!finite(mu)||mu<=0.||mu>1e30||!finite(violation))break;
        weights=lambda.array()/s.array();
        weighted=qp.A.array().colwise()*weights;
        W=qp.H+qp.A.transpose()*weighted;
        W.diagonal().array()+=1.e-11*std::max(
            1.,W.diagonal().cwiseAbs().maxCoeff());
        ldlt.compute(W);
        if(ldlt.info()!=Eigen::Success || !ldlt.isPositive())break;
        rc=s.cwiseProduct(lambda);
        if(!newton(qp,ldlt,s,lambda,rd,rp,rc,dx,ds,dl))break;
        const double ap=fraction(s,ds,1.),ad=fraction(lambda,dl,1.);
        const double mua=(s+ap*ds).dot(lambda+ad*dl)/m;
        const double sigma=std::pow(std::clamp(mua/mu,0.,1.),3.);
        corr=rc+ds.cwiseProduct(dl)-
            Eigen::VectorXd::Constant(m,sigma*mu);
        if(!newton(qp,ldlt,s,lambda,rd,rp,corr,step,sstep,lstep))break;
        const double a=fraction(s,sstep,.995);
        const double b=fraction(lambda,lstep,.995);
        if(a<=0.||b<=0.)break;
        x+=a*step;s+=a*sstep;lambda+=b*lstep;
        if(!x.allFinite()||!s.allFinite()||!lambda.allFinite()||
           (s.array()<=0.).any()||(lambda.array()<=0.).any())break;
    }
    result.x=x;
    result.max_violation=max_violation(qp,x);
    if(result.status!=-2 &&
       (result.status!=1 || result.max_violation>1e-5 ||
        !finite(result.max_violation)))result.status=0;
    return result;
}
// Constraints -Jwall*dy<=wall-buffer; -Jrate*dy<=rate; physical boxes.
Qp assemble_hard(const Vec &y,const Sample &base,
    const Derivatives &deriv,const MpcRtiConfiguration_t &model,
    const Mat &H,const Vec &g,int walls) {
    Qp q{};
    q.H=H;q.q=g;
    q.A=Eigen::MatrixXd::Zero(walls+R+2*D,D);
    q.b=Eigen::VectorXd::Zero(walls+R+2*D);
    q.A.topRows(walls)=-deriv.wall.topRows(walls);
    q.b.head(walls)=base.wall.head(walls).array()-kMarginBuffer;
    q.A.middleRows(walls,R)=-deriv.rate;q.b.segment(walls,R)=base.rate;
    const int first=walls+R;
    for(int j=0;j<D;++j) {
        const int axis=j%2;
        const double lo=(axis==0 ?-model.max_steering_rad:.001)/
            kScale[axis]-y[j];
        const double hi=(axis==0 ? model.max_steering_rad:1.)/
            kScale[axis]-y[j];
        q.A(first+2*j,j)=1.;q.b[first+2*j]=hi;
        q.A(first+2*j+1,j)=-1.;q.b[first+2*j+1]=-lo;
    }
    return q;
}
Qp assemble_slack(const Qp &hard,bool objective_lp,
    int walls,double slack_bound=INFINITY) {
    Qp q{};
    const int variables=hard.q.size();
    const int hard_rows=hard.b.size();
    const int m=hard_rows+walls+(objective_lp?0:1);
    q.H=Eigen::MatrixXd::Zero(variables+walls,variables+walls);
    q.q=Eigen::VectorXd::Zero(variables+walls);
    q.H.diagonal().array()+=objective_lp?1.e-10:1.e-8;
    if(objective_lp)q.q.tail(walls).setOnes();
    else q.H.topLeftCorner(variables,variables).diagonal().array()=1.;
    q.A=Eigen::MatrixXd::Zero(m,variables+walls);
    q.b=Eigen::VectorXd::Zero(m);
    q.A.topLeftCorner(walls,variables)=hard.A.topRows(walls);
    q.A.topRightCorner(walls,walls)=
        -Eigen::MatrixXd::Identity(walls,walls);
    q.A.block(walls,0,hard_rows-walls,variables)=
        hard.A.bottomRows(hard_rows-walls);
    q.b.head(hard_rows)=hard.b;
    q.A.block(hard_rows,variables,walls,walls)=
        -Eigen::MatrixXd::Identity(walls,walls);
    if(!objective_lp) {
        q.A.bottomRows(1).rightCols(walls).setOnes();
        q.b[m-1]=slack_bound;
    }
    return q;
}

bool qp_step(const Vec &y,const Sample &base,
    const Derivatives &d,const Mat &H,const MpcRtiConfiguration_t &model,
    int max_qp_iter,int walls,int decision_count,double deadline_us,
    Vec &step,Statistics &stats) {
    const bool phase1=!base.strict_valid ||base.wall_v>1.e-12 ||
        base.rate_v>1.e-10;
    // Build full physical constraints, then eliminate fixed-tail variables
    // exactly from the QP. Their 30-stage source effects remain in the
    // right-hand sides; no physics or safety constraints are truncated.
    Qp hard=assemble_hard(y,base,d,model,
        phase1?Mat::Identity():H,phase1?Vec::Zero():d.gradient,walls);
    if(decision_count<D) {
        // For a 20-control-stage experiment, algebraically eliminate
        // frozen variables. The selected 30-stage production candidate
        // needs NO duplicate 300x60 allocation/copy of the same hard QP.
        hard.H=hard.H.topLeftCorner(decision_count,decision_count).eval();
        hard.q=hard.q.head(decision_count).eval();
        hard.A=hard.A.leftCols(decision_count).eval();
    }
    step.setZero();
    auto solve=[&](const Qp &problem) {
        const double start=now_us();
        auto res=solve_qp(problem,max_qp_iter,deadline_us);
        if(res.status==-2)stats.deadline_exhausted=true;
        stats.qp_us+=now_us()-start;
        stats.qp_iterations+=res.iterations;
        ++stats.qp_solves;stats.last_qp_status=res.status;
        stats.max_affine_violation=std::max(
            stats.max_affine_violation,
            finite(res.max_violation)?res.max_violation:0.);
        return res;
    };
    if(!phase1) {
        const auto result=solve(hard);
        if(result.status!=1)return false;
        step.head(decision_count)=result.x;
        return step.allFinite();
    }
    // A source-infeasible nominal does NOT imply its linearized wall
    // constraints are infeasible. Test the exact hard min-norm Phase-I QP
    // first. An LP stopped with a micrometre residual must not cause the
    // second QP to choose a different elastic search direction when the
    // hard linearized problem is in fact solvable. Only enter the true
    // slack LP when this hard QP is independently uncertified.
    const auto direct=solve(hard);
    if(direct.status==1) {
        step.head(decision_count)=direct.x;
        return step.allFinite();
    }
    const auto stage1=solve(assemble_slack(hard,true,walls));
    if(stage1.status!=1)return false;
    const double bound=std::max(0.,stage1.x.tail(walls).sum());
    stats.phase1_slack_sum[stats.iterations-1]=bound;
    if(bound>2.e-7) {
        const auto stage2=solve(assemble_slack(hard,false,walls,
            bound+std::max(1e-7,1.e-5*bound)));
        if(stage2.status!=1)return false;
        step.head(decision_count)=stage2.x.head(decision_count);
    } else {
        // Affine footprint is fully repairable; use hard QP minimal norm.
        const auto stage2=solve(hard);
        if(stage2.status!=1)return false;
        step.head(decision_count)=stage2.x;
    }
    return step.allFinite();
}
bool corrected_step(const Vec &trial,const Sample &at_trial,
    const Derivatives &d,const MpcRtiConfiguration_t &model,
    int max_qp_iter,int walls,int decision_count,double deadline_us,
    Vec &corrected,Statistics &stats) {
    // Source exact second-order constraint defect: c(y+p)-
    // [c(y)+Jp]. The source value at trial is the correction RHS.
    Derivatives probe=d;
    probe.gradient.setZero();
    Qp q=assemble_hard(trial,at_trial,probe,model,
        Mat::Identity(),Vec::Zero(),walls);
    if(decision_count<D) {
        q.H=q.H.topLeftCorner(decision_count,decision_count).eval();
        q.q=q.q.head(decision_count).eval();
        q.A=q.A.leftCols(decision_count).eval();
    }
    const double start=now_us();
    const auto res=solve_qp(q,max_qp_iter,deadline_us);
    if(res.status==-2)stats.deadline_exhausted=true;
    stats.qp_us+=now_us()-start;++stats.qp_solves;
    stats.qp_iterations+=res.iterations;
    if(res.status!=1)return false;
    corrected=trial;
    corrected.head(decision_count)+=res.x;
    return corrected.allFinite();
}
bool fully_feasible(const Sample &s) {
    return s.strict_valid && s.wall_v<=1e-12 &&s.rate_v<=1e-10;
}
bool acceptable(const Sample &base,const Sample &trial) {
    if(fully_feasible(base)) {
        return fully_feasible(trial) &&
            trial.cost<base.cost-1.e-6;
    }
    if(fully_feasible(trial))return true;
    const double old_v=wall_violation(base.wall,true);
    const double new_v=wall_violation(trial.wall,true);
    return new_v<old_v-std::max(1.e-11,old_v*1.e-5) &&
        trial.rate_v<=base.rate_v+1.e-8;
}
} // namespace

bool optimize(const MpcRtiState_t *initial,double progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t *configuration,
    const MpcModelControl_t initial_controls[N],
    const Options &options,Statistics *statistics,
    MpcModelControl_t selected_controls[N],MpcRtiNominal_t *certified) {
    if(!initial||!trajectory||trajectory_count<3||!configuration||
       !configuration->objective_schedule||
       !configuration->objective_schedule->valid||
       !initial_controls||!statistics||!selected_controls||!certified||
       !finite(progress)||!finite(lap_length)||lap_length<=0.||
       !finite(dt)||dt<=0.f||options.max_iterations<1||
       options.max_iterations>12||options.qp_max_iterations<1||
       options.derivative_step<=0.||options.derivative_step>.05||
       (options.terminal_lookahead_steps!=0 &&
        options.terminal_lookahead_steps!=3) ||
       (options.control_decision_stages!=20 &&
        options.control_decision_stages!=N) ||
       (options.use_full_source_forward_sensitivities &&
        options.terminal_lookahead_steps!=0))
        return false;
    const double start=now_us();
    *statistics={};*certified={};
    Statistics &stats=*statistics;
    Evaluation evaluator{};
    evaluator.initial=initial;evaluator.progress=progress;
    evaluator.trajectory=trajectory;evaluator.trajectory_count=trajectory_count;
    evaluator.lap_length=lap_length;evaluator.dt=dt;
    evaluator.cfg=configuration;evaluator.stats=&stats;
    evaluator.deadline_us=options.absolute_deadline_us;
    evaluator.terminal_lookahead_steps=options.terminal_lookahead_steps;
    evaluator.reuse_unperturbed_fd_prefix=
        options.reuse_unperturbed_fd_prefix;
    stats.terminal_lookahead_steps=options.terminal_lookahead_steps;
    stats.control_decision_stages=options.control_decision_stages;
    const int decision_count=2*options.control_decision_stages;
    Vec y=to_y(initial_controls);
    if(!y.allFinite())return false;
    Sample base{};
    if(!evaluator.evaluate(y,true,base))return false;
    // Each strict candidate evaluation already invokes the production
    // complete nonlinear source validator. Preserve its exact certificate
    // when that candidate is selected, instead of replaying all 30 plant
    // steps a THIRD time in the final return path. This is never a QP-only
    // certificate: the validated command bytes MUST match the output.
    auto cache_source_certificate=[&](const Vec &candidate,
                                       const Sample &sample) {
        if(options.terminal_lookahead_steps!=0 || !fully_feasible(sample) ||
           !evaluator.strict->valid || evaluator.strict->horizon!=N)
            return;
        MpcModelControl_t commands[N]{};
        from_y(candidate,commands);
        if(std::memcmp(evaluator.strict->controls,commands,
                sizeof(commands))==0)
            *certified=*evaluator.strict;
    };
    cache_source_certificate(y,base);
    stats.initial_cost=base.cost;
    stats.initial_wall_violation_m2=base.wall_v;
    stats.initial_rate_violation_m2=base.rate_v;
    stats.initial_strict_source_valid=base.strict_valid;
    stats.initial_min_clearance_m=base.margin;
    stats.initial_tail_violation_m2=base.tail_wall_v;
    stats.initial_tail_minimum_clearance_m=base.tail_minimum;
    stats.initial_source_failure_stage=base.failure_stage;
    std::copy_n(base.stages,N+1,stats.initial_stage_min_clearance_m);
    // Only exact strict source feasibility certifies a nominal, never an
    // unchecked infeasible intermediate or a relaxed local wall.
    Vec accepted_y=y;
    bool have_accepted=false;
    Mat H=Mat::Identity();
    // Snapshot previous-epoch KKT duals before updating the current
    // memory. Never apply an uncommitted QP from a rejected RTI cycle.
    Vec previous_y=Vec::Zero(),previous_grad=Vec::Zero();
    bool previous_feasible=false;
    for(int iteration=0;iteration<options.max_iterations;++iteration) {
        if(options.absolute_deadline_us>0. &&
           now_us()>=options.absolute_deadline_us) {
            stats.deadline_exhausted=true;break;
        }
        ++stats.iterations;
        const double derivative_start=now_us();
        Derivatives derivative{};
        if((options.reuse_unperturbed_fd_prefix ||
            options.use_full_source_forward_sensitivities) &&
            iteration>0) {
            // Line search / SOC can leave the temporary rollout buffer
            // holding an unaccepted candidate. Recreate the EXACT accepted
            // command nominal once so every stage-k prefix is causal.
            Sample confirmed{};
            if(!evaluator.evaluate(y,false,confirmed) ||
               (confirmed.wall-base.wall).cwiseAbs().maxCoeff()>2e-6 ||
               (confirmed.rate-base.rate).cwiseAbs().maxCoeff()>2e-6 ||
               std::abs(confirmed.cost-base.cost)>2e-3)
                break;
        }
        evaluator.capture_source_prefix();
        const bool feasible=fully_feasible(base);
        if(!deriv(evaluator,y,base,options.derivative_step,feasible,
                decision_count,options.use_full_source_forward_sensitivities,
                derivative))break;
        if(stats.deadline_exhausted)break;
        if(!options.use_full_source_forward_sensitivities)
            stats.source_finite_difference_columns+=decision_count;
        if(iteration==0 && options.first_derivative_snapshot) {
            auto &snap=*options.first_derivative_snapshot;
            snap.objective=base.cost;
            for(int row=0;row<BaseW;++row) {
                snap.wall[row]=base.wall[row];
                for(int col=0;col<D;++col)
                    snap.wall_jacobian[row][col]=derivative.wall(row,col);
            }
            for(int row=0;row<R;++row) {
                snap.rate[row]=base.rate[row];
                for(int col=0;col<D;++col)
                    snap.rate_jacobian[row][col]=derivative.rate(row,col);
            }
            for(int col=0;col<D;++col)
                snap.gradient[col]=derivative.gradient[col];
        }
        stats.derivative_us+=now_us()-derivative_start;
        if(feasible) {
            if(previous_feasible) {
                // Powell-damped BFGS with exact source FD gradients.
                const Vec s=y-previous_y;
                const Vec diff=derivative.gradient-previous_grad;
                const Vec Bs=H*s;
                const double sBs=s.dot(Bs),sy=s.dot(diff);
                if(s.norm()>1e-9&&sBs>1e-10&&finite(sy)) {
                    const double theta=sy>=.2*sBs?1.:
                        .8*sBs/(sBs-sy);
                    const Vec adjusted=theta*diff+(1.-theta)*Bs;
                    if(s.dot(adjusted)>1e-10) {
                        H-=Bs*Bs.transpose()/sBs;
                        H+=adjusted*adjusted.transpose()/s.dot(adjusted);
                        H=.5*(H+H.transpose()).eval();
                    }
                }
            } else {
                H=derivative.hessian;
            }
            previous_y=y;
            previous_grad=derivative.gradient;
            previous_feasible=true;
        } else {
            previous_feasible=false;
            H.setIdentity();
        }
        Vec delta;
        if(!qp_step(y,base,derivative,H,configuration->model,
                options.qp_max_iterations,
                options.terminal_lookahead_steps==3?W:BaseW,
                decision_count,options.absolute_deadline_us,delta,stats))
            break;
        stats.first_qp_direction_norm[iteration]=delta.norm();
        double chosen_alpha=0.;
        bool chosen=false;
        Sample full_step{};
        bool full_step_evaluated=false;
        Vec selected=Vec::Zero();
        Sample measured{};
        for(double alpha:kAlpha) {
            if(options.absolute_deadline_us>0. &&
               now_us()>=options.absolute_deadline_us) {
                stats.deadline_exhausted=true;break;
            }
            const Vec trial=y+alpha*delta;
            Sample attempt{};
            ++stats.line_search_trials;
            if(!evaluator.evaluate(trial,true,attempt))continue;
            if(alpha==1.) {
                full_step=attempt;
                full_step_evaluated=true;
            }
            if(!acceptable(base,attempt))continue;
            selected=trial;measured=attempt;chosen=true;chosen_alpha=alpha;
            cache_source_certificate(selected,measured);
            // First source-exact improving globalization step. No
            // exhaustive trial search or additional "recovery controller".
            break;
        }
        if(full_step_evaluated) {
            const Wall linear_wall=base.wall+derivative.wall*delta;
            const Rate linear_rate=base.rate+derivative.rate*delta;
            stats.full_step_max_wall_defect_m[iteration]=
                (full_step.wall-linear_wall).cwiseAbs().maxCoeff();
            stats.full_step_max_rate_defect[iteration]=
                (full_step.rate-linear_rate).cwiseAbs().maxCoeff();
            stats.full_step_local_wall_violation_m2[iteration]=
                wall_violation(linear_wall,false);
            stats.full_step_source_wall_violation_m2[iteration]=
                full_step.wall_v;
        }
        // In the validated 1-SQP-iteration realtime regime, an already
        // source-feasible incumbent is itself a publishable full trajectory.
        // The source-exact SOC is a feasibility/globalization correction,
        // not a prerequisite to accepting that incumbent. On 60 normal
        // frozen epochs SOC ran 21 extra QPs and never changed a single
        // strict nonlinear accepted control or objective. Preserve it for
        // genuinely source-infeasible starts and multi-iteration solves.
        const bool soc_needed=!feasible || options.max_iterations>1;
        if(options.second_order_correction && soc_needed &&
           !stats.deadline_exhausted && full_step_evaluated) {
            const Vec trial=y+delta;
            // Reuse the exact alpha=1 source rollout already measured for
            // globalization. A second identical 750ms integration is never
            // necessary merely to start the SOC subproblem.
            if(!(full_step.strict_valid &&
                 full_step.margin>=kMarginBuffer)) {
                Vec corrected;
                if(corrected_step(trial,full_step,derivative,
                       configuration->model,
                       options.qp_max_iterations,
                       options.terminal_lookahead_steps==3?W:BaseW,
                       decision_count,options.absolute_deadline_us,
                       corrected,stats)) {
                    Sample soc{};
                    if(evaluator.evaluate(corrected,true,soc) &&
                       acceptable(base,soc) &&
                       (!chosen ||
                         (fully_feasible(base)
                            ?soc.cost<measured.cost
                            :wall_violation(soc.wall,true)<
                             wall_violation(measured.wall,true)))) {
                        selected=corrected;measured=soc;chosen=true;
                        chosen_alpha=1.;
                        cache_source_certificate(selected,measured);
                    }
                }
            }
        }
        if(!chosen)break;
        if(!have_accepted && feasible) {
            // Only compare racing-cost Hessian against real racing cost
            // when the feasible-start QP actually optimizes that cost.
            // Phase-I minimizes wall slack instead: its Hessian is I and
            // is not a source racing-cost prediction.
            const Vec applied_step=selected-y;
            stats.first_cost_prediction_error=measured.cost-base.cost-
                (derivative.gradient.dot(applied_step)+
                 .5*applied_step.dot(H*applied_step));
        }
        const bool stop_on_full_source_merit=
            options.source_exact_early_stop &&
            iteration==0 && fully_feasible(measured) &&
            measured.margin>=options.min_early_stop_clearance_m &&
            (base.cost-measured.cost)/
                std::max(1.,std::abs(base.cost))>=
                    options.min_accepted_relative_cost_improvement;
        have_accepted=true;
        y=selected;base=measured;accepted_y=y;
        stats.accepted_cost[iteration]=base.cost;
        stats.accepted_wall_violation_m2[iteration]=base.wall_v;
        stats.accepted_alpha[iteration]=chosen_alpha;
        ++stats.accepted_iterations;
        if(fully_feasible(base)) {
            if(!stats.first_feasible_iteration)
                stats.first_feasible_iteration=iteration+1;
            ++stats.accepted_full_source_horizons;
        }
        if(stop_on_full_source_merit) {
            stats.source_exact_early_stop_selected=true;
            break;
        }
    }
    from_y(accepted_y,selected_controls);
    stats.final_cost=base.cost;
    stats.final_wall_violation_m2=base.wall_v;
    stats.final_rate_violation_m2=base.rate_v;
    stats.final_strict_source_valid=base.strict_valid;
    stats.final_source_failure_stage=base.failure_stage;
    stats.final_min_clearance_m=base.margin;
    stats.final_tail_violation_m2=base.tail_wall_v;
    stats.final_tail_minimum_clearance_m=base.tail_minimum;
    stats.feasible=fully_feasible(base);
    std::copy_n(base.stages,N+1,stats.final_stage_min_clearance_m);
    if(stats.feasible) {
        const bool certified_commands_identical=
            options.terminal_lookahead_steps==0 && certified->valid &&
            certified->horizon==N &&
            std::memcmp(certified->controls,selected_controls,
                sizeof(MpcModelControl_t)*N)==0;
        if(certified_commands_identical) {
            // Strict physics/corridor validation ALREADY succeeded for
            // these exact command bytes. Only omit redundant integration.
            stats.final_source_certificate_reused=true;
        } else {
            int failure_stage=-1;
            if(mpc_rti_validate_candidate_plan(initial,progress,
                    selected_controls,trajectory,trajectory_count,
                    lap_length,N,dt,configuration,
                    certified,&failure_stage)!=MPC_RTI_ROLLOUT_OK) {
                *certified={};stats.feasible=false;
            }
        }
    }
    stats.total_us=now_us()-start;
    // A full source certification, numerical QP solve, or a scheduler stall
    // may cross the deadline AFTER the last pre-work budget check. The
    // caller is never allowed to publish that late plan. Preserve the
    // candidate only as diagnostic evidence; the RTI selector returns
    // WAITING_FOR_SOLVER_BUDGET before filling candidate controls.
    if(options.absolute_deadline_us>0. &&
       now_us()>=options.absolute_deadline_us)
        stats.deadline_exhausted=true;
    return true;
}
} // namespace sdu_apex::source_sqp
