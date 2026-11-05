#include "mpc_source_convex_qp.hpp"
#include "mpc_rti.h"

#include <Eigen/Cholesky>
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
        M coupling=M::Zero(n,groups);
        for(int i=0;i<me;++i)coupling.col(group(i)).noalias()+=C.row(i).transpose()*Dr(i);
        M weighted=A;
        for(int i=0;i<mh;++i)weighted.row(i)*=std::sqrt(Dh(i));
        M normal=H+weighted.transpose()*weighted;
        weighted=C;
        for(int i=0;i<me;++i)weighted.row(i)*=std::sqrt(Dr(i));
        normal.noalias()+=weighted.transpose()*weighted;
        M scaled_coupling=coupling;
        for(int i=0;i<groups;++i)scaled_coupling.col(i)/=std::sqrt(diagonal(i));
        normal.noalias()-=scaled_coupling*scaled_coupling.transpose();
        Eigen::LLT<M> factor(normal);
        if(factor.info()!=Eigen::Success)break;
        const auto newton=[&](double target,const V &cross_h,const V &cross_r,const V &cross_t,
                             V &dx,V &dt,V &dsh,V &dsr,V &dst,V &dlh,V &dlr,V &dlt) {
            const V qh=(-sh.array()*lh.array()+target-lh.array()*ph.array()-cross_h.array()).matrix();
            const V qr=(-sr.array()*lr.array()+target-lr.array()*pr.array()-cross_r.array()).matrix();
            const V qt=(-st.array()*lt.array()+target-lt.array()*pt.array()-cross_t.array()).matrix();
            const V rhs_t=-dt_residual+sum_groups((qr.array()/sr.array()).matrix())+(qt.array()/st.array()).matrix();
            const V rhs=-dx_residual+A.transpose()*(qh.array()/sh.array()).matrix()+
                C.transpose()*(qr.array()/sr.array()).matrix()-coupling*(rhs_t.array()/diagonal.array()).matrix();
            dx=factor.solve(rhs);
            dt=(rhs_t-coupling.transpose()*dx).array()/diagonal.array();
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
