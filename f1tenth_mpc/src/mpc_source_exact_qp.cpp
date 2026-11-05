#include "mpc_source_exact_qp.hpp"
#include "unity_source_plant_c.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace sdu_apex::source_nlp {
namespace {
constexpr int N=2*kStages;
constexpr int E=kStages*kState;
constexpr double input_scale[2]={.15,.25};
using Vector=Eigen::VectorXd;
using Matrix=Eigen::MatrixXd;
using Tangent=Eigen::Matrix<double,kState,N,Eigen::RowMajor>;

struct PhysicalRow {
    explicit PhysicalRow(int n):slope(Vector::Zero(n)) {}
    Vector slope;
    double margin{};
    double weight{};
    double multiplier{};
    size_t row_index{};
    bool hard{};
    NonlinearRowType type{};
};
bool finite(const Vector &x) {return x.allFinite();}
} // namespace

bool source_exact_constrained_qp_direction(
    const SparseNonlinearProgram &program,
    const double original_objective_control_gradient[kStages][2],
    const MpcModelControl_t nominal_controls[kStages],
    const SourceConstrainedQpOptions &options,
    SourceConstrainedQpResult *output)
{
    if(!output || !program.valid || !original_objective_control_gradient ||
       !nominal_controls ||
       program.original_source_stage_derivatives.size()!=kStages ||
       program.equality_rows!=E ||
       !(options.curvature_regularization>0.) ||
       !(options.inequality_penalty_weight>0.) ||
       options.maximum_qp_iterations<1 ||
       options.maximum_qp_iterations>5000 ||
       !(options.kkt_tolerance>0.) ||
       !std::isfinite(options.requested_corner_clearance_m) ||
       options.requested_corner_clearance_m<0. ||
       (options.inequality_multipliers &&
        options.inequality_multipliers->size()!=program.rows.size()))
        return false;
    *output={};
    const auto begin=std::chrono::steady_clock::now();
    const int n=N;

    // The state tangent and sparse physical rows are evaluated at exactly
    // the source model's own nonlinear nominal, never a QP surrogate plant.
    // S[k] = d state[k+1] / d all 60 ORIGINAL control commands.
    std::vector<Tangent> sensitivity(kStages,Tangent::Zero());
    for(int k=0;k<kStages;++k) {
        const auto &stage=program.original_source_stage_derivatives[k];
        auto &S=sensitivity[k];
        for(int i=0;i<kState;++i) {
            for(int j=0;j<N;++j) {
                double v=0.;
                if(k>0)
                    for(int state=0;state<kState;++state)
                        v+=stage.A[i][state]*sensitivity[k-1](state,j);
                if(j/2==k)v+=stage.B[i][j%2];
                S(i,j)=v;
                if(!std::isfinite(v))return false;
            }
        }
    }
    // The sparse assembly is canonical row ordered. Include both the
    // source-through-state derivatives and direct original-control terms.
    std::vector<size_t> first(program.rows.size()+1,0);
    size_t pos=0;
    for(size_t r=0;r<program.rows.size();++r) {
        first[r]=pos;
        while(pos<program.jacobian.size() &&
              program.jacobian[pos].row==static_cast<int>(r))++pos;
        if(pos<program.jacobian.size() &&
           program.jacobian[pos].row<static_cast<int>(r))
            return false;
    }
    first.back()=pos;
    if(pos!=program.jacobian.size())return false;

    std::vector<PhysicalRow> rows;
    rows.reserve(16*kStages);
    output->inequality_multipliers.assign(program.rows.size(),0.);
    const auto append=[&](PhysicalRow &&row) {
        if(!finite(row.slope) || !std::isfinite(row.margin))
            return false;
        rows.emplace_back(std::move(row));
        return true;
    };
    for(size_t r=0;r<program.rows.size();++r) {
        const auto &src=program.rows[r];
        if(src.type!=NonlinearRowType::BodyCorner &&
           src.type!=NonlinearRowType::PhysicalCommandRate &&
           src.type!=NonlinearRowType::PhysicalCommandBox)
            continue;
        PhysicalRow row(n);
        row.type=src.type;
        row.margin=src.value-
            (src.type==NonlinearRowType::BodyCorner?
             options.requested_corner_clearance_m:0.);
        row.hard=src.type==NonlinearRowType::PhysicalCommandBox;
        row.weight=options.inequality_penalty_weight;
        row.row_index=r;
        row.multiplier=options.inequality_multipliers?
            (*options.inequality_multipliers)[r]:0.;
        if(!std::isfinite(row.multiplier) || row.multiplier<0.)
            return false;
        for(size_t p=first[r];p<first[r+1];++p) {
            const auto &term=program.jacobian[p];
            if(term.row!=static_cast<int>(r))return false;
            if(term.column>=0 && term.column<E) {
                const int stage=term.column/kState;
                const int state=term.column%kState;
                for(int j=0;j<N;++j)
                    row.slope(j)+=term.forward*
                        sensitivity[stage](state,j)*input_scale[j%2];
            } else if(term.column>=E && term.column<E+N) {
                const int j=term.column-E;
                row.slope(j)+=term.forward*input_scale[j%2];
            }
        }
        if(src.type==NonlinearRowType::BodyCorner)
            ++output->original_footprint_rows;
        else if(src.type==NonlinearRowType::PhysicalCommandRate)
            ++output->original_rate_rows;
        else
            ++output->original_box_rows;
        if(!append(std::move(row)))return false;
    }
    if(rows.empty())return false;
    Vector g=Vector::Zero(n);
    // Always optimize the ORIGINAL racing cost, including at an infeasible
    // iterate. Constraints enter the same QP via augmented-Lagrangian
    // multipliers; no alternate objective or Phase-I controller.
    for(int k=0;k<kStages;++k)
        for(int axis=0;axis<2;++axis)
            g(2*k+axis)=original_objective_control_gradient[k][axis]*
                input_scale[axis];
    if(!finite(g))return false;
    for(int k=0;k<kStages;++k)
        if(!std::isfinite(nominal_controls[k].steering_target_rad) ||
           !std::isfinite(nominal_controls[k].throttle_command))
            return false;

    // The physically enforced input-box inequalities each select one
    // ORIGINAL control coordinate. Their feasible set is a box in the
    // scaled correction; no additional previous-iterate trust box exists.
    Vector lower=Vector::Constant(n,-1.e30);
    Vector upper=Vector::Constant(n,1.e30);
    for(const auto &row:rows) {
        if(!row.hard) {
            if(row.margin<0.)
                output->predicted_linearized_violation_before+=
                    .5*row.margin*row.margin;
            continue;
        }
        int nonzero=-1;
        for(int j=0;j<n;++j)
            if(std::abs(row.slope(j))>1.e-12) {
                if(nonzero>=0)return false;
                nonzero=j;
            }
        if(nonzero<0) {
            if(row.margin<0.)return false;
            continue;
        }
        const double limit=-row.margin/row.slope(nonzero);
        if(row.slope(nonzero)>0.)
            lower(nonzero)=std::max(lower(nonzero),limit);
        else
            upper(nonzero)=std::min(upper(nonzero),limit);
    }
    for(int j=0;j<n;++j)
        if(lower(j)>upper(j)+1.e-10)return false;

    // Equivalent convex QP:
    // min 0.5*h*||x||^2 + g'x + 0.5*w*||s||^2
    // s.t. source_margin + source_J*x + s >= 0, s >= 0,
    //      physical_input_lower <= nominal+scaled*x <= upper.
    // Eliminate s analytically: s=max(0,-source_margin-source_J*x).
    // Semismooth primal active-set Newton solves the resulting exact
    // piecewise-quadratic QP with one 60x60 SPD system per iteration.
    // This replaces unconverged ADMM sweeps, not the source dynamics.
    const double h=options.curvature_regularization;
    // Source-derived PSD racing-cost curvature. All feature gradients use
    // the SAME stage source A/B used in the feasibility QP and the exact
    // nonlinear score. Including the cross-stage steering-throttle
    // sensitivity avoids the previously misleading isotropic Hessian.
    // Only the positive-semidefinite Gauss-Newton part is represented:
    // residual-weighted source second derivatives are not fabricated.
    Matrix source_cost_hessian=Matrix::Identity(n,n)*h;
    if(options.source_configuration || options.source_nominal) {
        if(!options.source_configuration || !options.source_nominal ||
           !options.source_configuration->objective_schedule ||
           !options.source_configuration->objective_schedule->valid ||
           options.source_configuration->objective_schedule->horizon!=kStages ||
           !options.source_nominal->valid ||
           options.source_nominal->horizon!=kStages)
            return false;
        const auto &cfg=options.source_configuration->model;
        const auto &schedule=*options.source_configuration->objective_schedule;
        const auto &nominal=*options.source_nominal;
        const auto add=[&](const Vector &feature,double weight) {
            if(weight<=0.)return true;
            if(!std::isfinite(weight) || !feature.allFinite())return false;
            source_cost_hessian.noalias()+=
                (2.*weight)*feature*feature.transpose();
            return true;
        };
        const auto feature_for_state=[&](int stage,int state) {
            Vector feature=Vector::Zero(n);
            for(int j=0;j<N;++j)
                feature(j)=sensitivity[stage](state,j)*
                    input_scale[j%2];
            return feature;
        };
        for(int k=0;k<kStages;++k) {
            const auto &plant=nominal.states[k+1].plant;
            const auto &ref=schedule.references[k+1];
            const double terminal=k==kStages-1?
                cfg.terminal_multiplier:1.;
            if(std::abs(double(plant.e_y)-ref.e_y)>
               cfg.tracking_lateral_tolerance_m &&
               !add(feature_for_state(k,kPhysical),
                    terminal*cfg.weight_e_y))return false;
            if(std::abs(double(plant.e_psi)-ref.e_psi)>
               cfg.tracking_heading_tolerance_rad &&
               !add(feature_for_state(k,kPhysical+1),
                    terminal*cfg.weight_e_psi))return false;
            if(!add(feature_for_state(k,UNITY_SOURCE_U),
                    terminal*schedule.speed_weights[k+1]))return false;
            Vector rear=feature_for_state(k,UNITY_SOURCE_V)-
                .15532*feature_for_state(k,UNITY_SOURCE_YAW_RATE);
            if(!add(rear,terminal*cfg.weight_v) ||
               !add(feature_for_state(k,UNITY_SOURCE_YAW_RATE),
                    terminal*cfg.weight_r) ||
               !add(feature_for_state(k,UNITY_SOURCE_PHYSICAL_STEERING),
                    terminal*cfg.weight_steering_command))return false;

            // Direct command and source-delayed command queues, with
            // original stage reference/slew effects. Unlike diagonal
            // curvature, these features couple consecutive controls.
            Vector steer=Vector::Zero(n);
            Vector throttle=Vector::Zero(n);
            steer(2*k)=input_scale[0];
            throttle(2*k+1)=input_scale[1];
            if(!add(steer,cfg.weight_steering_target+
                     terminal*cfg.weight_steering_command) ||
               !add(throttle,cfg.weight_throttle_command))
                return false;
            if(k>0) {
                steer(2*k-2)=-input_scale[0];
                throttle(2*k-1)=-input_scale[1];
            }
            if(!add(steer,cfg.weight_steering_slew) ||
               !add(throttle,cfg.weight_throttle_slew))
                return false;
        }
        if(!source_cost_hessian.allFinite())return false;
    }
    // Full-step source curvature calibration: the physical plant's
    // actual nonlinear response is authoritative. A rejected full-step
    // contributes one nonnegative rank-one secant in the original control
    // coordinates. No sample of a scaled physical step is created here.
    if(options.source_curvature_secants)
        for(const auto &secant:*options.source_curvature_secants) {
            if(!std::isfinite(secant.curvature) ||
               secant.curvature<0. || secant.curvature>1.e9)
                return false;
            Eigen::Matrix<double,N,1> unit;
            for(int j=0;j<N;++j)
                unit(j)=secant.unit_direction[j];
            if(!unit.allFinite() ||
               std::abs(unit.squaredNorm()-1.)>1.e-5)
                return false;
            source_cost_hessian.noalias()+=
                secant.curvature*unit*unit.transpose();
        }
    // Original source-NLP KKT residual at the current physical iterate.
    // Multipliers apply to unmodified original source constraints.  Box
    // normal-cone projection is computed at a zero control correction.
    Vector stationarity=g;
    for(const auto &row:rows) {
        const double infeasibility=std::max(0.,-row.margin);
        output->original_nlp_inequality_residual_inf=std::max(
            output->original_nlp_inequality_residual_inf,infeasibility);
        if(row.hard)continue;
        stationarity.noalias()-=row.multiplier*row.slope;
        output->original_nlp_complementarity_inf=std::max(
            output->original_nlp_complementarity_inf,
            std::abs(row.multiplier*row.margin));
    }
    for(int j=0;j<n;++j) {
        if((lower(j)>=-1.e-10 && stationarity(j)>0.) ||
           (upper(j)<=1.e-10 && stationarity(j)<0.))
            stationarity(j)=0.;
    }
    output->original_nlp_stationarity_inf=
        stationarity.lpNorm<Eigen::Infinity>();
    if(!std::isfinite(output->original_nlp_stationarity_inf) ||
       !std::isfinite(output->original_nlp_complementarity_inf))
        return false;
    const auto clamp=[&](const Vector &v) {
        return v.cwiseMax(lower).cwiseMin(upper);
    };
    const auto model=[&](const Vector &v,Vector *gradient,
                         Matrix *hessian) {
        double cost=.5*v.dot(source_cost_hessian*v)+g.dot(v);
        if(gradient)*gradient=source_cost_hessian*v+g;
        if(hessian)*hessian=source_cost_hessian;
        for(const auto &row:rows) {
            if(row.hard)continue;
            // Powell-Hestenes-Rockafellar inequality AL model:
            // λ >= 0, c(U)>=0.  The shifted elastic residual carries
            // active physical constraint multipliers across SQP passes.
            // It is a single NLP, not a separate feasibility optimizer.
            const double shifted_margin=
                row.margin+row.slope.dot(v)-
                row.multiplier/row.weight;
            if(shifted_margin>=0.)continue;
            cost+=.5*row.weight*shifted_margin*shifted_margin;
            if(gradient)
                gradient->noalias()+=
                    row.weight*shifted_margin*row.slope;
            if(hessian)
                hessian->noalias()+=
                    row.weight*row.slope*row.slope.transpose();
        }
        return cost;
    };
    Vector x=clamp(Vector::Zero(n));
    Vector gradient=Vector::Zero(n);
    Matrix H=Matrix::Zero(n,n);
    double value=model(x,&gradient,&H);
    for(int it=0;it<options.maximum_qp_iterations;++it) {
        Vector projected=gradient;
        std::vector<bool> active(n,false);
        for(int j=0;j<n;++j) {
            active[j]=(x(j)<=lower(j)+1.e-10 && gradient(j)>0.) ||
                      (x(j)>=upper(j)-1.e-10 && gradient(j)<0.);
            if(active[j])projected(j)=0.;
        }
        output->iterations=it+1;
        output->dual_residual=
            projected.lpNorm<Eigen::Infinity>();
        if(output->dual_residual<=options.kkt_tolerance) {
            output->converged=1;
            break;
        }
        for(int j=0;j<n;++j)
            if(active[j]) {
                H.row(j).setZero();
                H.col(j).setZero();
                H(j,j)=1.;
            }
        Eigen::LLT<Matrix> factor(H);
        if(factor.info()!=Eigen::Success)return false;
        Vector step=factor.solve(-projected);
        if(!finite(step))return false;
        bool accepted=false;
        // QP-model globalization only; after this solve the accepted MPC
        // trajectory is STILL separately checked by the exact nonlinear
        // source rollout and exact racing cost.
        for(int t=0;t<22;++t) {
            const Vector next=clamp(x+std::ldexp(1.,-t)*step);
            if((next-x).squaredNorm()<1.e-24)continue;
            const double next_cost=model(next,nullptr,nullptr);
            if(std::isfinite(next_cost) &&
               next_cost<value-1.e-13*
                   std::max(1.,std::abs(value))) {
                x=next;
                value=model(x,&gradient,&H);
                accepted=true;
                break;
            }
        }
        if(!accepted) {
            // Globally valid projected-gradient direction for this
            // convex quadratic model; no new vehicle-model approximation.
            const double lipschitz=std::max(h,H.trace());
            const Vector next=clamp(x-gradient/lipschitz);
            const double next_cost=model(next,nullptr,nullptr);
            if(!std::isfinite(next_cost) || !(next_cost<value))
                break;
            x=next;
            value=model(x,&gradient,&H);
        }
    }
    Vector projected=gradient;
    for(int j=0;j<n;++j)
        if((x(j)<=lower(j)+1.e-10 && gradient(j)>0.) ||
           (x(j)>=upper(j)-1.e-10 && gradient(j)<0.))
            projected(j)=0.;
    output->dual_residual=projected.lpNorm<Eigen::Infinity>();
    output->converged|=
        output->dual_residual<=options.kkt_tolerance;
    for(const auto &row:rows) {
        const double margin=row.margin+row.slope.dot(x);
        if(row.hard)
            output->primal_residual=std::max(
                output->primal_residual,std::max(0.,-margin));
        else {
            const double violation=std::min(0.,margin);
            output->predicted_linearized_violation_after+=
                .5*violation*violation;
            output->inequality_multipliers[row.row_index]=
                std::max(0.,row.multiplier-row.weight*margin);
        }
    }
    // The QP may solve only for feasibility (zero racing gradient), but
    // its cost predictor must ALWAYS use the ORIGINAL source objective
    // gradient and never include artificial numerical regularization.
    for(int k=0;k<kStages;++k)
        for(int axis=0;axis<2;++axis)
            output->source_cost_linear_slope+=
                original_objective_control_gradient[k][axis]*
                x(2*k+axis)*input_scale[axis];
    output->source_cost_psd_quadratic=std::max(0.,
        x.dot(source_cost_hessian*x)-h*x.squaredNorm());
    output->predicted_original_objective_change=
        output->source_cost_linear_slope+
        .5*output->source_cost_psd_quadratic;
    output->valid=true;
    for(int k=0;k<kStages;++k)
        for(int axis=0;axis<2;++axis)
            output->direction[k][axis]=
                x(2*k+axis)*input_scale[axis];
    output->wall_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    output->valid=finite(x) &&
        std::isfinite(output->predicted_original_objective_change);
    return output->valid;
}
} // namespace sdu_apex::source_nlp
