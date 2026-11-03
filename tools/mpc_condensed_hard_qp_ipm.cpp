/*
 * Offline condensed hard-corridor QP primal-dual interior-point experiment.
 * NOT linked into the production solver and NOT a controller policy.
 *
 * Exact affine condensing: eliminate all x[k] = c[k] + G[k]*u[0:N-1]
 * and all 23x30 LTV dynamics equality rows. Solve 60 control variables
 * with all actual 2D/3D physical hard wall and command/state boxes.
 *
 * The full original QP is written independently by the capture format.
 * Test accuracy against the Python condensed OSQP and physical C++ source
 * rollout before considering any native controller integration.
 */
#include "mpc_qp_fixture_format.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int nx=RICCATI_MAX_NX,nu=RICCATI_MAX_NU;
constexpr int horizon=RICCATI_HORIZON, vars=nu*horizon;
using VecX=Eigen::Matrix<double,nx,1>;
using VecU=Eigen::Matrix<double,vars,1>;
using MatG=Eigen::Matrix<double,nx,vars>;
using MatH=Eigen::Matrix<double,vars,vars>;
using Row=Eigen::Matrix<double,1,vars>;
struct Candidate {
    MatH hessian=MatH::Zero();
    VecU linear=VecU::Zero();
    std::array<MatG,horizon+1> state_map{};
    std::array<VecX,horizon+1> constant{};
    Eigen::MatrixXd C;
    Eigen::VectorXd b;
    double cost_constant{};
    int redundant{};
};
struct Result {
    bool success{};
    int iterations{};
    double primal{},dual{},relative_dual{},duality{}, objective{};
    double solve_us{},setup_us{};
    VecU controls=VecU::Zero();
};

Candidate condense(const MpcRtiQpCapture_t& capture) {
    Candidate p{};
    const auto &qp=capture.problem;
    if(qp.horizon!=horizon)throw std::runtime_error("unexpected horizon");
    for(int i=0;i<nx;++i)p.constant[0](i)=qp.x0[i];
    for(int k=0;k<horizon;++k) {
        const auto &step=qp.steps[k];
        Eigen::Matrix<double,nx,nx> A{};
        Eigen::Matrix<double,nx,nu> B{};
        VecX d{},q{};
        Eigen::Matrix<double,nx,nx> Q{};
        Eigen::Matrix<double,nx,nu> N{};
        Eigen::Matrix<double,nu,nu> R{};
        Eigen::Matrix<double,nu,1> linear_r{};
        for(int i=0;i<nx;++i) {
            d(i)=step.d[i];q(i)=step.q[i];
            for(int j=0;j<nx;++j) {
                A(i,j)=step.A[i][j];
                Q(i,j)=step.Q_offdiag[i][j]+(i==j?step.Q_diag[i]:0.);
            }
            for(int a=0;a<nu;++a) {
                B(i,a)=step.B[i][a];
                N(i,a)=step.N[i][a];
            }
        }
        for(int a=0;a<nu;++a) {
            linear_r(a)=step.r[a];
            R(a,a)=step.R_diag[a];
        }
        R(0,1)=R(1,0)=step.R_cross;
        p.constant[k+1]=A*p.constant[k]+d;
        p.state_map[k+1]=A*p.state_map[k];
        p.state_map[k+1].template block<nx,nu>(0,nu*k)+=B;
        const auto &G=p.state_map[k];
        const auto &c=p.constant[k];
        p.hessian.noalias()+=G.transpose()*Q*G;
        p.linear.noalias()+=G.transpose()*(Q*c+q);
        p.hessian.block(0,nu*k,vars,nu).noalias()+=G.transpose()*N;
        p.hessian.block(nu*k,0,nu,vars).noalias()+=N.transpose()*G;
        p.hessian.block<nu,nu>(nu*k,nu*k)+=R;
        p.linear.segment<nu>(nu*k)+=N.transpose()*c+linear_r;
        p.cost_constant+=.5*c.dot(Q*c)+q.dot(c);
    }
    const auto &c=p.constant[horizon];
    const auto &G=p.state_map[horizon];
    VecX q{};
    VecX diag{};
    for(int i=0;i<nx;++i) {
        q(i)=qp.terminal_q[i];
        diag(i)=qp.terminal_Q[i];
    }
    p.hessian.noalias()+=G.transpose()*diag.asDiagonal()*G;
    p.linear.noalias()+=G.transpose()*(diag.asDiagonal()*c+q);
    p.cost_constant+=.5*c.dot(diag.asDiagonal()*c)+q.dot(c);
    p.hessian=.5*(p.hessian+p.hessian.transpose()).eval();

    std::vector<Row> rows;
    std::vector<double> bounds;
    rows.reserve(800);bounds.reserve(800);
    const auto add=[&](const Row &row,double upper) {
        if(!row.allFinite() || !std::isfinite(upper))
            throw std::runtime_error("non-finite condensed inequality");
        if(row.lpNorm<Eigen::Infinity>() < 1.e-9) {
            ++p.redundant;
            if(upper< -1.e-7)
                throw std::runtime_error("state0 hard infeasible");
        } else {
            rows.push_back(row);bounds.push_back(upper);
        }
    };
    // Input boxes are exact; no wall relaxation and no box widening.
    for(int k=0;k<horizon;++k) {
        const auto &stage=qp.steps[k];
        for(int a=0;a<nu;++a) {
            Row positive=Row::Zero();positive(nu*k+a)=1.;
            add( positive,stage.u_ub[a]);
            add(-positive,-stage.u_lb[a]);
        }
    }
    // State boxes at x1..xH; x0 is fixed, per native Riccati convention.
    for(int k=1;k<=horizon;++k) {
        const float *lb=k<horizon?qp.steps[k].x_lb:qp.terminal_x_lb;
        const float *ub=k<horizon?qp.steps[k].x_ub:qp.terminal_x_ub;
        for(int i=0;i<nx;++i) {
            const Row row=p.state_map[k].row(i);
            if(ub[i]<BIG_BOUND)add(row,ub[i]-p.constant[k](i));
            if(lb[i]>-BIG_BOUND)add(-row,p.constant[k](i)-lb[i]);
        }
        const auto &poly=qp.state_polygons[k];
        for(int face=0;face<poly.count;++face) {
            // Even if the frozen capture contains an *elastic* normal QP,
            // this experiment solves the corresponding ORIGINAL HARD walls.
            Eigen::Matrix<double,1,nx> direction=
                Eigen::Matrix<double,1,nx>::Zero();
            direction(0)=poly.normal[face][0];
            direction(1)=poly.normal[face][1];
            direction(MPC_OPT_IDX_PROGRESS_DEVIATION)=
                poly.progress_normal[face];
            add(direction*p.state_map[k],
                poly.upper[face]-(direction*p.constant[k])(0));
        }
    }
    const auto &terminal=qp.terminal_constraints;
    for(int face=0;face<terminal.count;++face) {
        Eigen::Matrix<double,1,nx> direction{};
        for(int j=0;j<nx;++j)direction(j)=terminal.normal[face][j];
        add(direction*p.state_map[horizon],
            terminal.upper[face]-(direction*p.constant[horizon])(0));
    }
    p.C.resize(rows.size(),vars);
    p.b.resize(bounds.size());
    for(size_t i=0;i<rows.size();++i) {
        p.C.row(i)=rows[i];
        p.b(i)=bounds[i];
    }
    return p;
}

double maximum_violation(const Candidate &qp,const VecU &u) {
    const auto violated=qp.C*u-qp.b;
    return std::max(0.,violated.maxCoeff());
}
double energy(const Candidate& qp,const VecU& u) {
    return .5*u.dot(qp.hessian*u)+qp.linear.dot(u)+qp.cost_constant;
}

double positive_fraction(const Eigen::VectorXd &x,
                         const Eigen::VectorXd &direction,
                         double tau) {
    double alpha=1.;
    for(int i=0;i<x.size();++i)
        if(direction(i)<0.)
            alpha=std::min(alpha,-tau*x(i)/direction(i));
    return std::clamp(alpha,0.,1.);
}
bool newton(const Candidate &problem,const VecU &u,
            const Eigen::VectorXd &s,const Eigen::VectorXd &lambda,
            const VecU &dual,const Eigen::VectorXd &primal,
            const Eigen::VectorXd &centering,
            VecU &du,Eigen::VectorXd &ds,Eigen::VectorXd &dlam) {
    const Eigen::ArrayXd weight=lambda.array()/s.array();
    Eigen::MatrixXd weighted=problem.C.array().colwise()*weight;
    MatH W=problem.hessian+problem.C.transpose()*weighted;
    // Strict numerical regularization is relative to Hessian scale and
    // only protects singular Schur complements in the offline experiment.
    const double max_diag=W.diagonal().cwiseAbs().maxCoeff();
    W.diagonal().array()+=1.e-12*std::max(1.,max_diag);
    const Eigen::VectorXd aux=(centering.array()-lambda.array()*primal.array())/
                               s.array();
    const VecU rhs=-dual+problem.C.transpose()*aux;
    Eigen::LDLT<MatH> ldlt(W);
    if(ldlt.info()!=Eigen::Success || !ldlt.isPositive())return false;
    du=ldlt.solve(rhs);
    if(ldlt.info()!=Eigen::Success || !du.allFinite())return false;
    ds=-primal-problem.C*du;
    dlam=(-centering.array()-lambda.array()*ds.array())/s.array();
    return ds.allFinite() && dlam.allFinite();
}

Result ipm(const Candidate &problem,int max_iter,double tolerance) {
    Result result{};
    const auto start=std::chrono::steady_clock::now();
    const int m=problem.b.size();
    VecU u=VecU::Zero();
    Eigen::VectorXd s=Eigen::VectorXd::Constant(m,1.);
    Eigen::VectorXd lambda=Eigen::VectorXd::Ones(m);
    const auto cu=problem.C*u;
    for(int i=0;i<m;++i)
        s(i)=std::max(1.,problem.b(i)-cu(i)+1.);
    for(int it=0;it<max_iter;++it) {
        const VecU rd=problem.hessian*u+problem.linear+
            problem.C.transpose()*lambda;
        const Eigen::VectorXd rp=problem.C*u+s-problem.b;
        const double mu=s.dot(lambda)/m;
        const double max_rp=rp.lpNorm<Eigen::Infinity>();
        const double max_rd=rd.lpNorm<Eigen::Infinity>();
        /* Absolute KKT dual residual is meaningless for a physically
         * mixed-unit condensed racing cost with 3.27e6 control gradients:
         * 1e-3 absolute stationarity corresponds to 3e-10 relative. The
         * correct numerical requirement is scale-aware dual stationarity,
         * while hard constraint residual must remain measured ABSOLUTELY
         * in its physical units (metres, radians, commands).
         */
        const double dual_scale=std::max({
            1.,problem.linear.lpNorm<Eigen::Infinity>(),
            (problem.hessian*u).lpNorm<Eigen::Infinity>(),
            (problem.C.transpose()*lambda).lpNorm<Eigen::Infinity>()});
        const double relative_dual=max_rd/dual_scale;
        result.iterations=it;
        result.primal=max_rp;
        result.dual=max_rd;
        result.relative_dual=relative_dual;
        result.duality=mu;
        if(max_rp<tolerance && relative_dual<1.e-8 && mu<tolerance) {
            result.success=true;break;
        }
        if(!std::isfinite(mu) || !std::isfinite(max_rp) ||
           !std::isfinite(max_rd) || mu>1.e30)break;
        VecU affine{};
        Eigen::VectorXd ds_aff(m),dl_aff(m);
        const Eigen::VectorXd rc=s.cwiseProduct(lambda);
        if(!newton(problem,u,s,lambda,rd,rp,rc,
                   affine,ds_aff,dl_aff))break;
        const double ap=positive_fraction(s,ds_aff,1.);
        const double ad=positive_fraction(lambda,dl_aff,1.);
        const double mua=(s+ap*ds_aff).dot(lambda+ad*dl_aff)/m;
        const double sigma=std::pow(std::clamp(mua/mu,0.,1.),3.);
        const Eigen::VectorXd corr=rc+ds_aff.cwiseProduct(dl_aff)-
                                    Eigen::VectorXd::Constant(m,sigma*mu);
        VecU step{};
        Eigen::VectorXd ds(m),dl(m);
        if(!newton(problem,u,s,lambda,rd,rp,corr,step,ds,dl))break;
        const double alpha_primal=positive_fraction(s,ds,.995);
        const double alpha_dual=positive_fraction(lambda,dl,.995);
        if(alpha_primal<=0. || alpha_dual<=0.)break;
        u+=alpha_primal*step;
        s+=alpha_primal*ds;
        lambda+=alpha_dual*dl;
        if(!u.allFinite() || !s.allFinite() || !lambda.allFinite())break;
    }
    result.controls=u;
    result.objective=energy(problem,u);
    result.solve_us=std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-start).count();
    return result;
}

} // namespace

int main(int argc,char**argv) {
    if(argc<2 || argc>4) {
        std::cerr<<"usage: mpc_condensed_hard_qp_ipm FILE.fqp [iterations] [output_controls.txt]\n";
        return 2;
    }
    try {
        auto fixture=std::make_unique<MpcRtiQpCapture_t>();
        if(!sdu_apex::qp_fixture::read_capture(argv[1],fixture.get()))
            throw std::runtime_error("invalid frozen FQP capture");
        const auto start=std::chrono::steady_clock::now();
        const Candidate problem=condense(*fixture);
        const double preparation_us=std::chrono::duration<double,std::micro>(
            std::chrono::steady_clock::now()-start).count();
        const int iterations=argc>=3?std::clamp(std::atoi(argv[2]),1,100):80;
        const Result result=ipm(problem,iterations,1.e-7);
        const double wall=maximum_violation(problem,result.controls);
        if(argc>=4 && result.success) {
            std::ofstream out(argv[3]);
            out<<std::setprecision(12);
            for(int k=0;k<horizon;++k)
                out<<result.controls(2*k)<<" "<<result.controls(2*k+1)<<"\n";
        }
        std::cout<<std::setprecision(12)
                 <<"{\"status\":\""<<(result.success?"solved":"not_solved")
                 <<"\",\"iterations\":"<<result.iterations
                 <<",\"primal_residual\":"<<result.primal
                 <<",\"dual_residual\":"<<result.dual
                 <<",\"relative_dual_residual\":"<<result.relative_dual
                 <<",\"complementarity\":"<<result.duality
                 <<",\"maximum_original_hard_wall_and_box_violation\":"<<wall
                 <<",\"objective\":"<<result.objective
                 <<",\"first_steer_decision\":"<<result.controls(0)
                 <<",\"first_throttle_decision\":"<<result.controls(1)
                 <<",\"condense_us\":"<<preparation_us
                 <<",\"ipm_us\":"<<result.solve_us
                 <<",\"inequalities\":"<<problem.C.rows()
                 <<",\"eliminated_dynamics\":"<<nx*horizon
                 <<"}\n";
        return result.success ? 0 : 1;
    } catch(const std::exception &error) {
        std::cerr<<"mpc_condensed_hard_qp_ipm: "<<error.what()<<"\n";
        return 2;
    }
}
