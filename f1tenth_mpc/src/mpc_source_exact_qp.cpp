#include "mpc_source_exact_qp.hpp"
#include "mpc_source_convex_qp.hpp"
#include "mpc_rti_internal.hpp"
#include "unity_source_plant_c.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
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
    // Every reduced QP has exactly 2*kStages controls. Keep constraint
    // slopes inline rather than allocating one dynamic Eigen vector for
    // every one of the 480 original physical inequalities, per SQP pass.
    Eigen::Matrix<double,N,1> slope=Eigen::Matrix<double,N,1>::Zero();
    double margin{};
    double original_margin{};
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
       !(options.curvature_regularization>0.) || !std::isfinite(options.curvature_regularization) ||
       !(options.inequality_penalty_weight>0.) || !std::isfinite(options.inequality_penalty_weight) ||
       options.maximum_qp_iterations<1 ||
       options.maximum_qp_iterations>5000 ||
       !(options.kkt_tolerance>0.) ||
       !std::isfinite(options.requested_corner_clearance_m) ||
       options.requested_corner_clearance_m<0. ||
       !std::isfinite(options.terminal_speed_viability_weight) ||
       options.terminal_speed_viability_weight<0. ||
       (options.inequality_multipliers &&
        options.inequality_multipliers->size()!=program.rows.size()) ||
       (options.constraint_margin_corrections &&
        options.constraint_margin_corrections->size()!=program.rows.size()))
        return false;
    *output={};
    const auto begin=std::chrono::steady_clock::now();
    const int n=N;

    // The state tangent and sparse physical rows are evaluated at exactly
    // the source model's own nonlinear nominal, never a QP surrogate plant.
    // S[k] = d state[k+1] / d all 60 ORIGINAL control commands.
    std::vector<Tangent> sensitivity(kStages);
    for(int k=0;k<kStages;++k) {
        const auto &stage=program.original_source_stage_derivatives[k];
        auto &S=sensitivity[k];
        // Dense state-to-control sensitivities are the dominant QP assembly
        // operation. Use a single contiguous matrix product instead of
        // 29 x 60 separately dispatched 29-term dot products. The exact
        // source Jacobians, control scaling and shooting convention remain
        // unchanged; B enters only the two controls of this stage.
        if(k==0) S.setZero();
        else {
            const Eigen::Map<const Eigen::Matrix<double,kState,kState,Eigen::RowMajor>>
                A(&stage.A[0][0]);
            S.noalias()=A*sensitivity[k-1];
        }
        for(int i=0;i<kState;++i) {
            S(i,2*k)+=stage.B[i][0];
            S(i,2*k+1)+=stage.B[i][1];
        }
        if(!S.allFinite())return false;
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
    output->predicted_constraint_margins.assign(program.rows.size(),0.);
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
           src.type!=NonlinearRowType::PhysicalCommandBox &&
           src.type!=NonlinearRowType::TerminalSpeedViability)
            continue;
        PhysicalRow row;
        row.type=src.type;
        row.margin=src.value;
        row.original_margin=src.value;
        if(options.constraint_margin_corrections)
            row.margin+=(*options.constraint_margin_corrections)[r];
        row.hard=src.type!=NonlinearRowType::TerminalSpeedViability;
        row.weight=
            src.type==NonlinearRowType::TerminalSpeedViability?
            options.terminal_speed_viability_weight:
            options.inequality_penalty_weight;
        if(!(row.weight>0.))return false;
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
        else if(src.type==NonlinearRowType::PhysicalCommandBox)
            ++output->original_box_rows;
        if(!append(std::move(row)))return false;
    }
    if(rows.empty())return false;
    Vector g=Vector::Zero(n);
    Vector peak_gradient=Vector::Zero(n);
    Matrix peak_gauss_newton=Matrix::Zero(n,n);
    const bool peak_epigraph=!program.original_source_slip_substeps.empty();
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

    // Hard source geometry, command-rate and command-box rows are distinct
    // from the preferred clearance epigraph. Preferred clearance contributes
    // a finite quadratic cost, never a persistent constraint multiplier.
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

            // The exact objective penalizes each wheel's source-selected
            // peak slip. Omitting these GN features discarded the dominant
            // steering/throttle/body coupling on the failing corner.
            const auto &stage=program.original_source_stage_derivatives[k];
            const double slip_weight=terminal*longitudinal_slip_weight_for_reference(
                &cfg,&schedule.references[k+1]);
            for(int wheel=0;wheel<4;++wheel) {
                Vector feature=Vector::Zero(n);
                for(int j=0;j<N;++j) {
                    double slope=j/2==k?stage.peak_slip_control[wheel][j%2]:0.;
                    if(k>0)
                        for(int state=0;state<kState;++state)
                            slope+=stage.peak_slip_state[wheel][state]*sensitivity[k-1](state,j);
                    feature(j)=slope*input_scale[j%2];
                }
                if(!add(feature,slip_weight))return false;
                // Peak-slip epigraph curvature/gradient are needed only
                // when those explicit auxiliary variables are enabled.
                // The live original-max objective already has the feature
                // curvature in source_cost_hessian via add() above.
                if(peak_epigraph) {
                    peak_gauss_newton.noalias()+=2.*slip_weight*feature*feature.transpose();
                    peak_gradient.noalias()+=2.*slip_weight*stage.source_peak_slip[wheel]*feature;
                }
            }

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
    const Matrix racing_hessian=source_cost_hessian-Matrix::Identity(n,n)*h;
    const Matrix base_gauss_newton=racing_hessian-
        (peak_epigraph?peak_gauss_newton:Matrix::Zero(n,n));
    const Vector base_gradient=g-
        (peak_epigraph?peak_gradient:Vector::Zero(n));
    if(peak_epigraph)source_cost_hessian-=peak_gauss_newton;
    const auto slip_slope=[&](const SlipEpigraphJacobianRow &row) {
        Vector slope(n);
        for(int j=0;j<n;++j) {
            double a=j/2==row.stage?row.control[j%2]:0.;
            if(row.stage>0)
                for(int state=0;state<kPhysical;++state)
                    a+=row.previous_physical_state[state]*sensitivity[row.stage-1](state,j);
            slope(j)=-a*input_scale[j%2];
        }
        return slope;
    };
    if(auto *memory=options.lagrangian_memory) {
        std::array<double,N> coordinates{};
        for(int j=0;j<N;++j)coordinates[j]=(j%2?
            double(nominal_controls[j/2].throttle_command):
            double(nominal_controls[j/2].steering_target_rad))/input_scale[j%2];
        if(!memory->valid) {
            for(int i=0;i<N;++i)
                for(int j=0;j<N;++j) {
                    memory->hessian[i*N+j]=base_gauss_newton(i,j);
                    memory->racing_gauss_newton[i*N+j]=base_gauss_newton(i,j);
                }
            memory->valid=true;
        } else if(memory->anchor_valid &&
                  (!memory->linearized_controls_valid || coordinates!=memory->linearized_controls)) {
            Vector s(n),y(n),lagrangian=base_gradient;
            if(memory->frozen_curvature_multipliers.size()!=program.rows.size())return false;
            for(const auto &row:rows)
                lagrangian.noalias()-=memory->frozen_curvature_multipliers[row.row_index]*row.slope;
            bool same_slip_structure=true;
            for(const auto &multiplier:memory->frozen_slip_multipliers) {
                if(multiplier.source_row_index>=program.original_source_slip_substeps.size()) {
                    same_slip_structure=false;break;
                }
                const auto &row=program.original_source_slip_substeps[multiplier.source_row_index];
                if(row.stage!=multiplier.stage || row.wheel!=multiplier.wheel ||
                   row.fixed_tick!=multiplier.fixed_tick || row.vehicle_substep!=multiplier.vehicle_substep) {
                    same_slip_structure=false;break;
                }
                lagrangian.noalias()-=multiplier.multiplier*slip_slope(row);
            }
            for(int j=0;j<N;++j) {
                s(j)=coordinates[j]-memory->anchor_controls[j];
                y(j)=lagrangian(j)-memory->anchor_gradient[j];
            }
            Matrix B(n,n);
            for(int i=0;i<N;++i)
                for(int j=0;j<N;++j)B(i,j)=memory->hessian[i*N+j]+
                    (options.refresh_known_gauss_newton?
                     base_gauss_newton(i,j)-memory->racing_gauss_newton[i*N+j]:0.);
            // Refresh known source GN curvature at every accepted primal;
            // retain only the learned residual curvature. A stale initial
            // tire-slip metric can otherwise dominate after slip disappears.
            Eigen::LLT<Matrix> positive(B+Matrix::Identity(n,n)*1.e-6);
            if(positive.info()!=Eigen::Success || !same_slip_structure)B=base_gauss_newton;
            const Vector Bs=B*s;
            const double sBs=s.dot(Bs),sy=s.dot(y);
            if(same_slip_structure && s.squaredNorm()>1.e-18 && sBs>1.e-12 && std::isfinite(sy)) {
                const double damping=sy>=.2*sBs?1.:.8*sBs/(sBs-sy);
                const Vector r=damping*y+(1.-damping)*Bs;
                const double sr=s.dot(r);
                if(sr>1.e-12) {
                    B.noalias()-=Bs*Bs.transpose()/sBs;
                    B.noalias()+=r*r.transpose()/sr;
                    if(!B.allFinite())return false;
                    output->lagrangian_bfgs_updated=true;
                    output->lagrangian_bfgs_damping=damping;
                }
            }
            for(int i=0;i<N;++i)
                for(int j=0;j<N;++j) {
                    memory->hessian[i*N+j]=.5*(B(i,j)+B(j,i));
                    memory->racing_gauss_newton[i*N+j]=base_gauss_newton(i,j);
                }
        }
        memory->linearized_controls=coordinates;
        memory->linearized_controls_valid=true;
        for(int i=0;i<N;++i)
            for(int j=0;j<N;++j)source_cost_hessian(i,j)=memory->hessian[i*N+j]+(i==j?h:0.);
    }
    if(options.lagrangian_hessian_scaled) {
        for(int i=0;i<n;++i)
            for(int j=0;j<n;++j)source_cost_hessian(i,j)=
                (*options.lagrangian_hessian_scaled)[i*n+j]+(i==j?h:0.);
        if(!source_cost_hessian.allFinite())return false;
    }
    if(options.capture_curvature_model) {
        output->racing_hessian_scaled.resize(n*n);
        output->stabilized_hessian_scaled.resize(n*n);
        for(int i=0;i<n;++i)
            for(int j=0;j<n;++j) {
                output->racing_hessian_scaled[i*n+j]=racing_hessian(i,j);
                output->stabilized_hessian_scaled[i*n+j]=source_cost_hessian(i,j);
            }
        for(const auto &row:rows) {
            output->constraint_row_indices.push_back(static_cast<int>(row.row_index));
            output->constraint_margins.push_back(row.margin);
            for(int j=0;j<n;++j)output->constraint_jacobian_scaled.push_back(row.slope(j));
        }
    }
    if(options.linearization_audit_only) {
        if(!options.capture_curvature_model)return false;
        output->valid=true;
        output->wall_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-begin).count();
        return true;
    }
    if(options.capture_curvature_model) {
        Eigen::SelfAdjointEigenSolver<Matrix> eigen(source_cost_hessian,Eigen::EigenvaluesOnly);
        if(eigen.info()!=Eigen::Success)return false;
        output->hessian_min_eigenvalue=eigen.eigenvalues().minCoeff();
        output->hessian_max_eigenvalue=eigen.eigenvalues().maxCoeff();
    }
    SourceConvexQp problem;
    problem.hessian=source_cost_hessian;
    problem.gradient=base_gradient;
    int hard_count=0,elastic_count=0;
    for(const auto &row:rows) {
        hard_count+=row.hard;
        elastic_count+=!row.hard;
    }
    problem.hard_jacobian.resize(hard_count,n);
    problem.hard_margin.resize(hard_count);
    problem.initial_hard_multipliers.resize(hard_count);
    std::vector<const SlipEpigraphJacobianRow *> peak_rows;
    if(peak_epigraph) {
        std::array<const SlipEpigraphJacobianRow *,4*kStages> largest{};
        for(const auto &row:program.original_source_slip_substeps) {
            const int group=4*row.stage+row.wheel;
            if(row.original_source_excess>0. && (!largest[group] ||
               row.original_source_excess>largest[group]->original_source_excess))largest[group]=&row;
        }
        for(const auto *row:largest)if(row)peak_rows.push_back(row);
    }
    const int base_elastic=elastic_count;
    problem.elastic_jacobian.resize(elastic_count+peak_rows.size(),n);
    problem.elastic_margin.resize(elastic_count+peak_rows.size());
    problem.elastic_groups.resize(elastic_count+peak_rows.size());
    problem.elastic_weight=Vector::Ones(elastic_count+(peak_epigraph?4*kStages:0));
    problem.elastic_linear_cost=Vector::Zero(problem.elastic_weight.size());
    int hi=0,ei=0;
    for(const auto &row:rows) {
        if(row.hard) {
            problem.hard_jacobian.row(hi)=row.slope.transpose();
            problem.initial_hard_multipliers(hi)=row.multiplier;
            problem.hard_margin(hi++)=row.margin;
        }
        if(!row.hard) {
            const double reserve=row.type==NonlinearRowType::BodyCorner?
                options.requested_corner_clearance_m:0.;
            problem.elastic_jacobian.row(ei)=row.slope.transpose();
            problem.elastic_margin(ei)=row.margin-reserve;
            problem.elastic_weight(ei++)=row.weight;
        }
    }
    for(int i=0;i<base_elastic;++i)problem.elastic_groups(i)=i;
    if(peak_epigraph) {
        if(!options.source_configuration || !options.source_configuration->objective_schedule)return false;
        for(int k=0;k<kStages;++k) {
            const auto &cfg=options.source_configuration->model;
            const auto &ref=options.source_configuration->objective_schedule->references[k+1];
            const double weight=2.*longitudinal_slip_weight_for_reference(&cfg,&ref)*
                (k==kStages-1?cfg.terminal_multiplier:1.);
            for(int wheel=0;wheel<4;++wheel)problem.elastic_weight(base_elastic+4*k+wheel)=weight;
        }
    }
    const auto append_peak_rows=[&](int first_row) {
        problem.elastic_jacobian.conservativeResize(base_elastic+peak_rows.size(),n);
        problem.elastic_margin.conservativeResize(base_elastic+peak_rows.size());
        problem.elastic_groups.conservativeResize(base_elastic+peak_rows.size());
        for(int index=first_row;index<static_cast<int>(peak_rows.size());++index) {
            const auto *row=peak_rows[index];
            const int target=base_elastic+index,k=row->stage;
            problem.elastic_groups(target)=base_elastic+4*k+row->wheel;
            problem.elastic_margin(target)=-row->original_source_excess;
            for(int j=0;j<n;++j) {
                double a=j/2==k?row->control[j%2]:0.;
                if(k>0)
                    for(int state=0;state<kPhysical;++state)
                        a+=row->previous_physical_state[state]*sensitivity[k-1](state,j);
                problem.elastic_jacobian(target,j)=-a*input_scale[j%2];
            }
        }
    };
    append_peak_rows(0);
    SourceConvexQpResult solved;

    // Inexact SQP forcing: do not solve a local QP to 1e-4 stationarity
    // when its original nonlinear KKT residual is millions. Tighten as
    // the nonlinear primal/dual iterate converges; hard primal tolerance
    // remains unchanged. Always expose the actual requested accuracy.
    Vector current_lagrangian=g;
    for(const auto &row:rows) {
        if(row.hard)current_lagrangian.noalias()-=row.multiplier*row.slope;
        if(!row.hard) {
            const double reserve=row.type==NonlinearRowType::BodyCorner?
                options.requested_corner_clearance_m:0.;
            current_lagrangian.noalias()+=row.weight*
                std::min(0.,row.original_margin-reserve)*row.slope;
        }
    }
    output->tolerance_used=std::max(options.kkt_tolerance,
        std::min(.01,.0001*current_lagrangian.lpNorm<Eigen::Infinity>()));
    int total_qp_iterations=0;
    bool all_peak_rows_satisfied=!peak_epigraph;
    // Constraint generation solves ONE convex epigraph subproblem. Check every
    // original source-substep affine row after each solve; never certify a QP
    // from a selected maximum alone. No extra nonlinear trial or alpha grid.
    for(int round=0;round<8;++round) {
        const bool hard_solved=solve_source_convex_qp(problem,options.maximum_qp_iterations,
            output->tolerance_used,&solved,options.absolute_deadline_monotonic_us);
        // A certified hard QP already supplies the constrained SQP step.
        // Replacing it with an elastic solve can introduce nonzero normal
        // residuals solely from normal-step regularization, relaxing an
        // otherwise consistent hard linearization and corrupting its duals.
        // Only a failed hard solve needs the internal normal/tangent pair.
        // Its independently certified elastic optimum distinguishes affine
        // inconsistency from failure of the hard-QP numerical algorithm.
        if((!hard_solved || !solved.converged) && !solved.deadline_exceeded &&
           !output->normal_step_computed) {
            SourceConvexQp normal;
            // Regularized affine normal step, in normalized controls.
            // Its fixed small norm term chooses a bounded, stable normal;
            // racing-step rejections must never increase this coefficient.
            // At its exact optimum V_aff(n)+mu*||n||^2/2 <= V_aff(0),
            // so every nonzero normal predicts physical violation descent.
            constexpr double normal_norm_weight=1.e-6;
            normal.hessian=normal_norm_weight*Matrix::Identity(n,n);
            normal.gradient=Vector::Zero(n);
            int boxes=0,physical=0;
            for(const auto &row:rows) {
                boxes+=row.type==NonlinearRowType::PhysicalCommandBox;
                physical+=row.hard && row.type!=NonlinearRowType::PhysicalCommandBox;
            }
            normal.hard_jacobian.resize(boxes,n);normal.hard_margin.resize(boxes);
            normal.elastic_jacobian.resize(physical,n);normal.elastic_margin.resize(physical);
            normal.elastic_weight=Vector::Ones(physical);
            int box=0,elastic=0;
            for(const auto &row:rows) {
                if(row.type==NonlinearRowType::PhysicalCommandBox) {
                    normal.hard_jacobian.row(box)=row.slope.transpose();
                    normal.hard_margin(box++)=row.margin;
                } else if(row.hard) {
                    normal.elastic_jacobian.row(elastic)=row.slope.transpose();
                    normal.elastic_margin(elastic++)=row.margin;
                }
            }
            SourceConvexQpResult normal_result;
            if(!solve_source_convex_qp(normal,std::max(60,options.maximum_qp_iterations),1.e-9,
                &normal_result,options.absolute_deadline_monotonic_us) || !normal_result.converged) {
                output->iterations=total_qp_iterations+normal_result.iterations;
                output->normal_primal_residual=normal_result.primal_inf;
                output->normal_dual_residual=normal_result.dual_inf;
                output->primal_residual=normal_result.primal_inf;
                output->dual_residual=normal_result.dual_inf;
                output->complementarity_residual=normal_result.complementarity_inf;
                output->deadline_exceeded=normal_result.deadline_exceeded;
                output->failure_reason="normal_affine_least_squares_qp_failed";
                return false;
            }
            total_qp_iterations+=normal_result.iterations;
            output->normal_step_computed=true;
            output->normal_primal_residual=normal_result.primal_inf;
            output->normal_dual_residual=normal_result.dual_inf;
            int hard=0;
            std::vector<const PhysicalRow *> normal_active;
            for(const auto &row:rows) {
                if(!row.hard)continue;
                const double value=row.margin+row.slope.dot(normal_result.step);
                const double relaxation=row.type==NonlinearRowType::PhysicalCommandBox?0.:std::min(0.,value);
                output->normal_linearized_violation+=.5*relaxation*relaxation;
                problem.hard_margin(hard++)=row.margin-relaxation;
                if(relaxation<-1.e-8)normal_active.push_back(&row);

            }
            // Preserve the active normal residual in an exact tangent
            // nullspace. Keeping these as inequalities creates a feasible
            // set with no strict interior and destroys barrier conditioning.
            problem.equality_jacobian.resize(normal_active.size(),n);
            problem.equality_rhs.resize(normal_active.size());
            for(size_t i=0;i<normal_active.size();++i) {
                problem.equality_jacobian.row(i)=normal_active[i]->slope.transpose();
                problem.equality_rhs(i)=normal_active[i]->slope.dot(normal_result.step);
            }
            for(int j=0;j<n;++j)output->normal_direction[j/2][j%2]=
                normal_result.step(j)*input_scale[j%2];
            problem.initial_hard_multipliers=Vector::Zero(hard_count);
            const bool tangent_valid=solve_source_convex_qp(problem,std::max(60,options.maximum_qp_iterations),
                output->tolerance_used,&solved,options.absolute_deadline_monotonic_us);
            if(!tangent_valid || !solved.converged) {
                // An accurate normal step remains useful when the objective
                // correction is uncertified. Keep the failed tangential
                // residuals/flag; never relabel it as a converged hard QP.
                output->normal_only_step=true;
                solved.converged=false;
                solved.step=normal_result.step;
                solved.hard_multipliers=Vector::Zero(hard_count);
                solved.elastic_multipliers=Vector::Zero(problem.elastic_margin.size());
                solved.elastic_values=Vector::Zero(problem.elastic_weight.size());
                for(int i=0;i<problem.elastic_margin.size();++i) {
                    const int group=problem.elastic_groups.size()?problem.elastic_groups(i):i;
                    solved.elastic_values(group)=std::max(solved.elastic_values(group),
                        -problem.elastic_margin(i)-problem.elastic_jacobian.row(i).dot(solved.step));
                }
                if(!tangent_valid) {
                    solved.primal_inf=std::numeric_limits<double>::infinity();
                    solved.dual_inf=std::numeric_limits<double>::infinity();
                }
            }
        } else if(!hard_solved)return false;
        total_qp_iterations+=solved.iterations;
        if(!peak_epigraph || solved.deadline_exceeded || output->normal_only_step)break;
        std::array<Eigen::Matrix<double,kState,1>,kStages> state_delta;
        for(int k=0;k<kStages;++k) {
            Vector physical=solved.step;
            for(int j=0;j<n;++j)physical(j)*=input_scale[j%2];
            state_delta[k]=sensitivity[k]*physical;
        }
        std::array<const SlipEpigraphJacobianRow *,4*kStages> worst{};
        std::array<double,4*kStages> defect{};
        for(const auto &row:program.original_source_slip_substeps) {
            const int k=row.stage,group=4*k+row.wheel;
            double value=row.original_source_excess;
            for(int j=0;j<2;++j)value+=row.control[j]*solved.step(2*k+j)*input_scale[j];
            if(k>0)
                for(int j=0;j<kPhysical;++j)value+=row.previous_physical_state[j]*state_delta[k-1](j);
            const double violation=value-solved.elastic_values(base_elastic+group);
            if(violation>std::max(1.e-7,defect[group])) {
                defect[group]=violation;worst[group]=&row;
            }
        }
        const int first_added=peak_rows.size();
        for(const auto *row:worst)
            if(row && std::find(peak_rows.begin(),peak_rows.end(),row)==peak_rows.end())peak_rows.push_back(row);
        all_peak_rows_satisfied=std::none_of(worst.begin(),worst.end(),[](const auto *p){return p!=nullptr;});
        if(all_peak_rows_satisfied || first_added==static_cast<int>(peak_rows.size()) || round==7) {
            peak_rows.resize(first_added);break;
        }
        append_peak_rows(first_added);
    }
    const Vector &x=solved.step;
    output->iterations=total_qp_iterations;
    output->converged=solved.converged && all_peak_rows_satisfied;
    output->deadline_exceeded=solved.deadline_exceeded;
    output->primal_residual=solved.primal_inf;
    output->dual_residual=solved.dual_inf;
    output->complementarity_residual=solved.complementarity_inf;
    for(const auto &row:rows)if(row.hard)
        output->linearized_physical_violation_inf=std::max(output->linearized_physical_violation_inf,
            std::max(0.,-row.margin-row.slope.dot(x)));
    // A small residual obtained by cancellation of huge dual terms is not
    // a trustworthy original-NLP multiplier estimate. Bound accumulation
    // roundoff in the actual, unscaled stationarity equation.
    Vector magnitude=problem.gradient.cwiseAbs()+problem.hessian.cwiseAbs()*x.cwiseAbs();
    magnitude.noalias()+=problem.hard_jacobian.cwiseAbs().transpose()*solved.hard_multipliers.cwiseAbs();
    magnitude.noalias()+=problem.elastic_jacobian.cwiseAbs().transpose()*solved.elastic_multipliers.cwiseAbs();
    const double roundoff_bound=(n+hard_count+problem.elastic_margin.size()+2)*
        std::numeric_limits<double>::epsilon()*magnitude.lpNorm<Eigen::Infinity>();
    output->original_duals_trustworthy=std::isfinite(roundoff_bound) && roundoff_bound<=options.kkt_tolerance &&
        output->converged &&
        output->linearized_physical_violation_inf<=1.e-8 && problem.equality_rhs.size()==0;
    // The original NLP stationarity must NEVER use the infeasible-start
    // squared-violation restoration gradient from the convex subproblem.
    Vector stationarity=base_gradient;
    Vector curvature_gradient=base_gradient;
    output->curvature_multipliers.assign(program.rows.size(),0.);
    hi=0;
    for(const auto &row:rows) {
        output->predicted_constraint_margins[row.row_index]=row.margin+row.slope.dot(x);
        if(row.hard) {
            const double estimate=solved.hard_multipliers(hi++);
            const double lambda=output->original_duals_trustworthy?estimate:0.;
            output->inequality_multipliers[row.row_index]=lambda;
            output->curvature_multipliers[row.row_index]=lambda;
            stationarity.noalias()-=lambda*row.slope;
            output->original_nlp_dual_feasibility_inf=std::max(
                output->original_nlp_dual_feasibility_inf,std::max(0.,-lambda));
            output->original_nlp_inequality_residual_inf=std::max(
                output->original_nlp_inequality_residual_inf,std::max(0.,-row.original_margin));
            output->original_nlp_complementarity_inf=std::max(
                output->original_nlp_complementarity_inf,std::abs(lambda*row.original_margin));
        }
        if(!row.hard) {
            const double reserve=row.type==NonlinearRowType::BodyCorner?
                options.requested_corner_clearance_m:0.;
            const double before=std::min(0.,row.original_margin-reserve);
            const double after=std::min(0.,row.margin+row.slope.dot(x)-reserve);
            output->curvature_multipliers[row.row_index]-=row.weight*after;
            // Preferred 3 cm reserve is an internal numerical penalty,
            // not part of the frozen original racing objective J. It must
            // never appear in an original-NLP KKT certificate.
            if(!row.hard)stationarity.noalias()+=row.weight*before*row.slope;
            if(row.type==NonlinearRowType::TerminalSpeedViability) {
                output->predicted_terminal_speed_violation_before=std::max(
                    output->predicted_terminal_speed_violation_before,.5*before*before);
                output->predicted_terminal_speed_violation_after=std::max(
                    output->predicted_terminal_speed_violation_after,.5*after*after);
            } else {
                output->predicted_linearized_violation_before+=.5*before*before;
                output->predicted_linearized_violation_after+=.5*after*after;
            }
        } else if(row.hard) {
            const double before=std::min(0.,row.margin);
            const double after=std::min(0.,row.margin+row.slope.dot(x));
            output->predicted_linearized_violation_before+=.5*before*before;
            output->predicted_linearized_violation_after+=.5*after*after;
        }
        curvature_gradient.noalias()-=output->curvature_multipliers[row.row_index]*row.slope;
    }
    if(peak_epigraph) {
        Vector peak_dual=Vector::Zero(4*kStages);
        for(int i=0;i<static_cast<int>(peak_rows.size());++i) {
            const auto &row=*peak_rows[i];
            const double lambda=output->original_duals_trustworthy?solved.elastic_multipliers(base_elastic+i):0.;
            stationarity.noalias()-=lambda*problem.elastic_jacobian.row(base_elastic+i).transpose();
            curvature_gradient.noalias()-=lambda*problem.elastic_jacobian.row(base_elastic+i).transpose();
            if(lambda>1.e-8)output->slip_curvature_multipliers.push_back({
                row.stage,row.wheel,row.fixed_tick,row.vehicle_substep,
                static_cast<size_t>(peak_rows[i]-program.original_source_slip_substeps.data()),lambda});
            peak_dual(4*row.stage+row.wheel)+=lambda;
            const double nominal_peak=program.original_source_stage_derivatives[row.stage].source_peak_slip[row.wheel];
            output->original_nlp_complementarity_inf=std::max(output->original_nlp_complementarity_inf,
                std::abs(lambda*(nominal_peak-row.original_source_excess)));
        }
        for(int k=0;k<kStages;++k)
            for(int wheel=0;wheel<4;++wheel) {
                const int index=4*k+wheel;
                const double value=program.original_source_stage_derivatives[k].source_peak_slip[wheel];
                const double residual=problem.elastic_weight(base_elastic+index)*value-peak_dual(index);
                // t>=0 is a genuine epigraph bound. At t=0, its dual can
                // absorb a nonnegative derivative, never a negative one.
                output->original_nlp_stationarity_inf=std::max(output->original_nlp_stationarity_inf,
                    value>0.?std::abs(residual):std::max(0.,-residual));
            }
    }
    // D_u * grad_U L is stationarity in normalized physical coordinates:
    // one unit is 0.15 rad steering / 0.25 throttle. Multipliers and
    // complementarity remain in original row and racing-cost units.
    for(int j=0;j<n;++j)output->lagrangian_gradient_at_qp_duals[j]=curvature_gradient(j);
    output->original_nlp_stationarity_inf=std::max(output->original_nlp_stationarity_inf,
        stationarity.lpNorm<Eigen::Infinity>());
    if(!std::isfinite(output->original_nlp_stationarity_inf) ||
       !std::isfinite(output->original_nlp_complementarity_inf))return false;
    // Predict original racing cost only. Stabilization and calibrated
    // constraint-merit curvature do not belong to the racing Hessian.
    for(int k=0;k<kStages;++k)
        for(int axis=0;axis<2;++axis)
            output->source_cost_linear_slope+=
                original_objective_control_gradient[k][axis]*
                x(2*k+axis)*input_scale[axis];
    output->source_cost_psd_quadratic=std::max(0.,
        x.dot(racing_hessian*x));
    output->predicted_original_objective_change=
        output->source_cost_linear_slope+
        .5*output->source_cost_psd_quadratic;
    if(peak_epigraph) {
        Vector peaks=Vector::Zero(4*kStages);
        for(int i=base_elastic;i<problem.elastic_margin.size();++i) {
            const int group=problem.elastic_groups(i)-base_elastic;
            peaks(group)=std::max(peaks(group),-problem.elastic_margin(i)-
                problem.elastic_jacobian.row(i).dot(x));
        }
        output->predicted_original_objective_change=problem.gradient.dot(x)+
            .5*x.dot((racing_hessian-peak_gauss_newton)*x);
        for(int k=0;k<kStages;++k)
            for(int wheel=0;wheel<4;++wheel) {
                const int group=4*k+wheel;
                const double previous=program.original_source_stage_derivatives[k].source_peak_slip[wheel];
                output->predicted_original_objective_change+=.5*problem.elastic_weight(base_elastic+group)*
                    (peaks(group)*peaks(group)-previous*previous);
            }
    }
    output->valid=true;
    for(int k=0;k<kStages;++k)
        for(int axis=0;axis<2;++axis)
            output->direction[k][axis]=
                x(2*k+axis)*input_scale[axis];
    output->wall_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    output->valid=finite(x) &&
        std::isfinite(output->predicted_original_objective_change);
    if(output->valid)output->failure_reason="none";
    return output->valid;
}
} // namespace sdu_apex::source_nlp
