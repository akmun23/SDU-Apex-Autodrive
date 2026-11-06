#include "mpc_source_convex_qp.hpp"
#include "mpc_rti.h"

#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <vector>
#include <algorithm>
#include <cmath>

namespace sdu_apex::source_nlp {
namespace {
using V=Eigen::VectorXd;
using M=Eigen::MatrixXd;
double inf(const V &v) {return v.size()?v.lpNorm<Eigen::Infinity>():0.;}
double fraction(const V &v,const V &direction,double safety) {
    double step=1.;
    for(Eigen::Index i=0;i<v.size();++i)
        if(direction(i)<0.)step=std::min(step,-safety*v(i)/direction(i));
    return step;
}
}

bool solve_source_convex_qp(const SourceConvexQp &p,int maximum_iterations,
                           double tolerance,SourceConvexQpResult *out,
                           double absolute_deadline_monotonic_us)
{
    const int n=p.gradient.size(),mh=p.hard_margin.size(),me=p.elastic_margin.size();
    const int groups=p.elastic_weight.size();
    if(!out || n<1 || maximum_iterations<1 || !(tolerance>0.) || !std::isfinite(tolerance) ||
       !std::isfinite(absolute_deadline_monotonic_us) || absolute_deadline_monotonic_us<0. ||
       (p.equality_rhs.size() && (p.equality_jacobian.rows()!=p.equality_rhs.size() || p.equality_jacobian.cols()!=n)) ||
       !p.equality_jacobian.allFinite() || !p.equality_rhs.allFinite() ||
       p.hessian.rows()!=n || p.hessian.cols()!=n ||
       p.hard_jacobian.rows()!=mh || p.hard_jacobian.cols()!=n ||
       p.elastic_jacobian.rows()!=me || p.elastic_jacobian.cols()!=n ||
       (p.elastic_groups.size()!=0 && p.elastic_groups.size()!=me) ||
       (p.elastic_groups.size()==0 && groups!=me) ||
       (p.elastic_groups.size() && (p.elastic_groups.minCoeff()<0 || p.elastic_groups.maxCoeff()>=groups)) ||
       (p.elastic_linear_cost.size()!=0 && p.elastic_linear_cost.size()!=groups) ||
       !p.elastic_linear_cost.allFinite() ||
       (p.initial_hard_multipliers.size()!=0 && p.initial_hard_multipliers.size()!=mh) ||
       !p.initial_hard_multipliers.allFinite() || !p.hessian.allFinite() || !p.gradient.allFinite() ||
       !p.hard_jacobian.allFinite() || !p.hard_margin.allFinite() ||
       !p.elastic_jacobian.allFinite() || !p.elastic_margin.allFinite() ||
       !p.elastic_weight.allFinite() || (me && p.elastic_weight.minCoeff()<0.) ||
       (p.elastic_linear_cost.size() && p.elastic_linear_cost.minCoeff()<0.))return false;
    *out={};
    if(p.equality_rhs.size()) {
        M equations=p.equality_jacobian;
        V rhs=p.equality_rhs,equation_scale=V::Ones(rhs.size());
        for(int i=0;i<rhs.size();++i) {
            equation_scale(i)=1./std::max({1.e-12,equations.row(i).norm(),std::abs(rhs(i))});
            equations.row(i)*=equation_scale(i);rhs(i)*=equation_scale(i);
        }
        Eigen::JacobiSVD<M> decomposition(equations,Eigen::ComputeFullU|Eigen::ComputeFullV);
        decomposition.setThreshold(1.e-12);
        const int rank=decomposition.rank();
        const V anchor=decomposition.solve(rhs);
        if(!anchor.allFinite() || inf(p.equality_jacobian*anchor-p.equality_rhs)>1.e-8)return false;
        const M nullspace=decomposition.matrixV().rightCols(n-rank);
        SourceConvexQp reduced=p;
        reduced.equality_jacobian.resize(0,0);reduced.equality_rhs.resize(0);
        reduced.hessian=nullspace.transpose()*p.hessian*nullspace;
        reduced.gradient=nullspace.transpose()*(p.hessian*anchor+p.gradient);
        std::vector<int> active_hard;
        const M projected=p.hard_jacobian*nullspace;
        const V shifted=p.hard_margin+p.hard_jacobian*anchor;
        for(int i=0;i<mh;++i) {
            if(projected.row(i).norm()>1.e-12*std::max(1.,p.hard_jacobian.row(i).norm()))active_hard.push_back(i);
            else if(shifted(i)<-1.e-8)return false;
        }
        reduced.hard_jacobian.resize(active_hard.size(),n-rank);
        reduced.hard_margin.resize(active_hard.size());
        reduced.initial_hard_multipliers.resize(p.initial_hard_multipliers.size()?active_hard.size():0);
        for(size_t i=0;i<active_hard.size();++i) {
            reduced.hard_jacobian.row(i)=projected.row(active_hard[i]);
            reduced.hard_margin(i)=shifted(active_hard[i]);
            if(p.initial_hard_multipliers.size())reduced.initial_hard_multipliers(i)=p.initial_hard_multipliers(active_hard[i]);
        }
        reduced.elastic_jacobian=p.elastic_jacobian*nullspace;
        reduced.elastic_margin=p.elastic_margin+p.elastic_jacobian*anchor;
        SourceConvexQpResult solved;
        if(n>rank) {
            if(!solve_source_convex_qp(reduced,maximum_iterations,tolerance,&solved,absolute_deadline_monotonic_us))return false;
            *out=solved;out->step=anchor+nullspace*solved.step;
        } else {
            out->step=anchor;out->iterations=1;out->converged=true;
            out->elastic_values=V::Zero(groups);out->elastic_multipliers=V::Zero(me);
            for(int i=0;i<me;++i) {
                const int group=p.elastic_groups.size()?p.elastic_groups(i):i;
                out->elastic_values(group)=std::max(out->elastic_values(group),-reduced.elastic_margin(i));
            }
            for(int group=0;group<groups;++group)if(out->elastic_values(group)>0.)
                for(int i=0;i<me;++i)
                    if((p.elastic_groups.size()?p.elastic_groups(i):i)==group &&
                       std::abs(out->elastic_values(group)+reduced.elastic_margin(i))<1.e-10) {
                        out->elastic_multipliers(i)=p.elastic_weight(group)*out->elastic_values(group)+
                            (p.elastic_linear_cost.size()?p.elastic_linear_cost(group):0.);
                        break;
                    }
        }
        out->hard_multipliers=V::Zero(mh);
        for(size_t i=0;i<active_hard.size();++i)out->hard_multipliers(active_hard[i])=solved.hard_multipliers(i);
        const V gradient=p.hessian*out->step+p.gradient-p.hard_jacobian.transpose()*out->hard_multipliers-
            p.elastic_jacobian.transpose()*out->elastic_multipliers;
        out->equality_multipliers=decomposition.matrixU().leftCols(rank)*
            decomposition.singularValues().head(rank).cwiseInverse().asDiagonal()*
            decomposition.matrixV().leftCols(rank).transpose()*gradient;
        out->equality_multipliers.array()*=equation_scale.array();
        out->equality_dual_reconstruction_inf=inf(gradient-p.equality_jacobian.transpose()*out->equality_multipliers);
        out->equality_basis_multipliers=decomposition.matrixV().leftCols(rank).transpose()*gradient;
        // Certify stationarity using the equivalent orthonormal equations;
        // do not subtract enormous cancelling original equality multipliers.
        out->dual_inf=std::max(out->dual_inf,inf(gradient-
            decomposition.matrixV().leftCols(rank)*out->equality_basis_multipliers));
        const V margins=p.hard_margin+p.hard_jacobian*out->step;
        out->primal_inf=std::max({out->primal_inf,mh?std::max(0.,-margins.minCoeff()):0.,
            inf(p.equality_jacobian*out->step-p.equality_rhs)});
        out->converged&=out->primal_inf<=1.e-8 && out->dual_inf<=tolerance && out->complementarity_inf<=tolerance;
        return out->step.allFinite() && out->equality_multipliers.allFinite();
    }
    // Scaling changes neither the primal problem nor physical multipliers.
    const double objective_scale=std::max({1.,inf(p.gradient),
        p.hessian.diagonal().cwiseAbs().maxCoeff(),me?inf(p.elastic_weight):0.,
        inf(p.elastic_linear_cost)});
    const M H=p.hessian/objective_scale;
    const V g=p.gradient/objective_scale,w=p.elastic_weight/objective_scale;
    const V linear=p.elastic_linear_cost.size()?V(p.elastic_linear_cost/objective_scale):V::Zero(groups);
    M A=p.hard_jacobian,C=p.elastic_jacobian;
    V b=p.hard_margin,d=p.elastic_margin;
    V hard_scale=V::Ones(mh);
    for(int i=0;i<mh;++i) {
        hard_scale(i)=1./std::max({1.e-3,A.row(i).norm(),std::abs(b(i))});
        A.row(i)*=hard_scale(i);b(i)*=hard_scale(i);
    }
    // Elastic t stays in its physical units, preserving its actual cost.
    // Do not independently normalize C without also transforming t and w.
    const auto group=[&](int row) {return p.elastic_groups.size()?p.elastic_groups(row):row;};
    const auto expand=[&](const V &v) {
        V result(me);for(int i=0;i<me;++i)result(i)=v(group(i));return result;
    };
    const auto sum_groups=[&](const V &v) {
        V result=V::Zero(groups);for(int i=0;i<me;++i)result(group(i))+=v(i);return result;
    };
    const auto epigraph=[&](const V &margins) {
        V t=V::Zero(groups);
        for(int i=0;i<me;++i)t(group(i))=std::max(t(group(i)),-margins(i));
        return t;
    };
    V x=V::Zero(n),t=epigraph(d);
    V sh=b.array().max(1.),sr=(d+expand(t)).array().max(1.),st=t.array().max(1.);
    V lh=V::Ones(mh),lr=V::Ones(me),lt=V::Ones(groups);
    if(p.initial_hard_multipliers.size())
        lh=(p.initial_hard_multipliers.array()/hard_scale.array()/objective_scale).max(1.e-3);
    const int inequalities=mh+me+groups;
    if(inequalities==0) {
        Eigen::LLT<M> factor(H);
        if(factor.info()!=Eigen::Success)return false;
        out->step=factor.solve(-g);out->hard_multipliers=V(0);
        out->dual_inf=inf(p.hessian*out->step+p.gradient);
        out->converged=out->dual_inf<=tolerance;out->iterations=1;
        return out->step.allFinite();
    }
    for(int iteration=0;iteration<maximum_iterations;++iteration) {
        if(absolute_deadline_monotonic_us>0. &&
           mpc_rti_monotonic_time_us()>=absolute_deadline_monotonic_us) {
            out->deadline_exceeded=true;break;
        }
        const V ph=b+A*x-sh,pr=d+C*x+expand(t)-sr,pt=t-st;
        const V dx_residual=H*x+g-A.transpose()*lh-C.transpose()*lr;
        const V dt_residual=(w.array()*t.array()).matrix()+linear-sum_groups(lr)-lt;
        const double mu=(sh.dot(lh)+sr.dot(lr)+st.dot(lt))/inequalities;
        out->iterations=iteration+1;
        // Certify the eliminated elastic problem itself. Small epigraph
        // complementarity alone can still hide w*slack gradient error.
        const V actual_hard=p.hard_margin+p.hard_jacobian*x;
        const V actual_lambda=lh.array()*hard_scale.array()*objective_scale;
        const V actual_peak=epigraph(p.elastic_margin+p.elastic_jacobian*x);
        const V actual_slack=p.elastic_margin+p.elastic_jacobian*x+expand(actual_peak);
        const V actual_gradient=p.hessian*x+p.gradient-p.elastic_jacobian.transpose()*(lr*objective_scale)-
            p.hard_jacobian.transpose()*actual_lambda;
        const V peak_gradient=(p.elastic_weight.array()*actual_peak.array()).matrix()+linear*objective_scale-
            sum_groups(lr*objective_scale)-lt*objective_scale;
        if((!mh || actual_hard.minCoeff()>=-1.e-8) && inf(actual_gradient)<=tolerance &&
           inf(peak_gradient)<=tolerance &&
           inf((actual_hard.array()*actual_lambda.array()).matrix())<=tolerance &&
           inf((actual_slack.array()*lr.array()).matrix())*objective_scale<=tolerance &&
           inf((actual_peak.array()*lt.array()).matrix())*objective_scale<=tolerance) {
            out->converged=true;break;
        }
        const V Dh=lh.array()/sh.array(),Dr=lr.array()/sr.array(),Dt=lt.array()/st.array();
        const V diagonal=w+sum_groups(Dr)+Dt;
        // Form each elastic Schur contribution as a weighted covariance.
        // C' Dr C - v v'/diagonal loses all significant digits when an
        // active elastic row has Dr >> w+Dt. The equivalent positive sum
        // preserves curvature for both independent and shared epigraphs.
        const V group_dr=sum_groups(Dr);
        const V remaining=w+Dt;
        M mean=M::Zero(n,groups);
        for(int i=0;i<me;++i)
            mean.col(group(i)).noalias()+=C.row(i).transpose()*(Dr(i)/group_dr(group(i)));
        M weighted=A;
        for(int i=0;i<mh;++i)weighted.row(i)*=std::sqrt(Dh(i));
        M normal=H+weighted.transpose()*weighted;
        for(int i=0;i<me;++i) {
            const V centered=C.row(i).transpose()-mean.col(group(i));
            normal.noalias()+=Dr(i)*centered*centered.transpose();
        }
        for(int i=0;i<groups;++i)
            normal.noalias()+=(group_dr(i)/diagonal(i))*remaining(i)*
                mean.col(i)*mean.col(i).transpose();
        // Symmetric diagonal equilibration also protects nearly parallel
        // active hard rows without deleting any physical restriction.
        const V equilibration=normal.diagonal().cwiseMax(1.e-300).cwiseSqrt().cwiseInverse();
        const M equilibrated=equilibration.asDiagonal()*normal*equilibration.asDiagonal();
        Eigen::LLT<M> factor(equilibrated);
        if(factor.info()!=Eigen::Success)break;
        const auto solve=[&](const V &rhs)->V {
            V value=equilibration.asDiagonal()*factor.solve(equilibration.asDiagonal()*rhs);
            for(int refinement=0;refinement<2;++refinement) {
                const V residual=rhs-normal*value;
                if(inf(residual)<=1.e-12*std::max(1.,inf(rhs)))break;
                value+=equilibration.asDiagonal()*factor.solve(equilibration.asDiagonal()*residual);
            }
            return value;
        };
        const auto newton=[&](double target,const V &cross_h,const V &cross_r,const V &cross_t,
                             V &dx,V &dt,V &dsh,V &dsr,V &dst,V &dlh,V &dlr,V &dlt) {
            const V qh=(-sh.array()*lh.array()+target-lh.array()*ph.array()-cross_h.array()).matrix();
            const V qr=(-sr.array()*lr.array()+target-lr.array()*pr.array()-cross_r.array()).matrix();
            const V qt=(-st.array()*lt.array()+target-lt.array()*pt.array()-cross_t.array()).matrix();
            const V rhs_t=-dt_residual+sum_groups((qr.array()/sr.array()).matrix())+(qt.array()/st.array()).matrix();
            const V reduced_r=(qr.array()/sr.array()).matrix();
            const V group_r=sum_groups(reduced_r);
            V rhs=-dx_residual+A.transpose()*(qh.array()/sh.array()).matrix();
            for(int i=0;i<me;++i)
                rhs.noalias()+=(C.row(i).transpose()-mean.col(group(i)))*reduced_r(i);
            const V mean_rhs=(remaining.array()/diagonal.array()*group_r.array()+
                group_dr.array()/diagonal.array()*(dt_residual.array()-qt.array()/st.array())).matrix();
            rhs.noalias()+=mean*mean_rhs;
            dx=solve(rhs);
            dt=rhs_t.array()/diagonal.array()-
                group_dr.array()/diagonal.array()*(mean.transpose()*dx).array();
            dsh=A*dx+ph;dsr=C*dx+expand(dt)+pr;dst=dt+pt;
            dlh=qh.array()/sh.array()-Dh.array()*(A*dx).array();
            dlr=qr.array()/sr.array()-Dr.array()*(C*dx+expand(dt)).array();
            dlt=qt.array()/st.array()-Dt.array()*dt.array();
        };
        V dx,dt,dsh,dsr,dst,dlh,dlr,dlt;
        newton(0.,V::Zero(mh),V::Zero(me),V::Zero(groups),dx,dt,dsh,dsr,dst,dlh,dlr,dlt);
        if(!dx.allFinite() || !dt.allFinite())break;
        const double ap=std::min({fraction(sh,dsh,1.),fraction(sr,dsr,1.),fraction(st,dst,1.)});
        const double ad=std::min({fraction(lh,dlh,1.),fraction(lr,dlr,1.),fraction(lt,dlt,1.)});
        const double mu_affine=((sh+ap*dsh).dot(lh+ad*dlh)+
            (sr+ap*dsr).dot(lr+ad*dlr)+(st+ap*dst).dot(lt+ad*dlt))/inequalities;
        const double sigma=std::pow(std::clamp(mu_affine/std::max(1.e-300,mu),0.,1.),3.);
        const V cross_h=dsh.array()*dlh.array(),cross_r=dsr.array()*dlr.array(),cross_t=dst.array()*dlt.array();
        newton(sigma*mu,cross_h,cross_r,cross_t,dx,dt,dsh,dsr,dst,dlh,dlr,dlt);
        if(!dx.allFinite() || !dt.allFinite() || !dlh.allFinite() || !dlr.allFinite() || !dlt.allFinite())break;
        const double primal_step=std::min({fraction(sh,dsh,.995),fraction(sr,dsr,.995),fraction(st,dst,.995)});
        const double dual_step=std::min({fraction(lh,dlh,.995),fraction(lr,dlr,.995),fraction(lt,dlt,.995)});
        x+=primal_step*dx;t+=primal_step*dt;
        sh+=primal_step*dsh;sr+=primal_step*dsr;st+=primal_step*dst;
        lh+=dual_step*dlh;lr+=dual_step*dlr;lt+=dual_step*dlt;
        if(!x.allFinite() || !t.allFinite() || (mh && sh.minCoeff()<=0.) ||
           (me && (sr.minCoeff()<=0. || st.minCoeff()<=0.)))break;
    }
    out->step=x;
    out->hard_multipliers=lh.array()*hard_scale.array()*objective_scale;
    out->elastic_multipliers=lr*objective_scale;
    const V hard=p.hard_margin+p.hard_jacobian*x;
    const V actual_peak=epigraph(p.elastic_margin+p.elastic_jacobian*x);
    out->elastic_values=actual_peak;
    const V actual_slack=p.elastic_margin+p.elastic_jacobian*x+expand(actual_peak);
    const V grad=p.hessian*x+p.gradient-p.elastic_jacobian.transpose()*out->elastic_multipliers-
        p.hard_jacobian.transpose()*out->hard_multipliers;
    const V peak_grad=(p.elastic_weight.array()*actual_peak.array()).matrix()+linear*objective_scale-
        sum_groups(out->elastic_multipliers)-lt*objective_scale;
    out->primal_inf=mh?std::max(0.,-hard.minCoeff()):0.;
    out->dual_inf=std::max(inf(grad),inf(peak_grad));
    out->complementarity_inf=std::max({inf((out->hard_multipliers.array()*hard.array()).matrix()),
        inf((out->elastic_multipliers.array()*actual_slack.array()).matrix()),
        inf((lt.array()*actual_peak.array()).matrix())*objective_scale});
    out->converged=out->primal_inf<=1.e-8 && out->dual_inf<=tolerance &&
        out->complementarity_inf<=tolerance;
    return x.allFinite() && out->hard_multipliers.allFinite();
}
} // namespace sdu_apex::source_nlp
