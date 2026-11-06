#include "mpc_source_exact_nlp.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace sdu_apex::source_nlp {
bool source_exact_reduced_gradient(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &source_feasible_decision,
    ReducedSourceGradient *out,ConstraintMeritWeights penalties,
    SparseNonlinearProgram *source_linearization,
    const Evaluation *known_original_score,
    const MpcRtiNominal_t *known_original_trace)
{
    if(!out || !std::isfinite(penalties.footprint) ||
       !std::isfinite(penalties.command_rate) ||
       penalties.footprint<0. || penalties.command_rate<0. ||
       !std::isfinite(penalties.minimum_corner_clearance_m) ||
       penalties.minimum_corner_clearance_m<0.)return false;
    *out={};
    // The current iterate was already scored by the authoritative full
    // original-source nonlinear oracle. Reuse ONLY when both the exact
    // score and corresponding same-epoch physical trace were supplied.
    // Standalone derivative callers retain their own independent source
    // validation path.
    if(bool(known_original_score)!=bool(known_original_trace))
        return false;
    auto owned_score=std::make_unique<Evaluation>();
    const Evaluation *primal=known_original_score;
    if(!primal) {
        if(!evaluate(initial,initial_progress,trajectory,trajectory_count,
            lap_length,dt,configuration,source_feasible_decision,
            owned_score.get()))return false;
        primal=owned_score.get();
    }
    if(!primal->valid || primal->max_dynamics_abs>1.e-8)
        return false;
    auto z=std::make_unique<SlipEpigraphDecision>();
    z->shooting=source_feasible_decision;
    for(int k=0;k<kStages;++k)
        for(int w=0;w<4;++w)
            z->peak[k][w]=primal->source_peak_slip[k][w];

    auto program=std::make_unique<SparseNonlinearProgram>();
    if(!assemble_sparse_nonlinear_program(initial,initial_progress,trajectory,
        trajectory_count,lap_length,dt,configuration,*z,program.get(),
        known_original_trace,known_original_score) ||
       !program->valid || program->equality_rows!=kDynamicsEquations ||
       program->num_variables!=kDecisionVariables+4*kStages ||
       program->original_source_objective!=primal->objective)
        return false;

    constexpr int control0=kStages*kState;
    constexpr int peak0=kDecisionVariables;
    // Objective derivatives at the original nonlinear source trajectory,
    // plus source-active epigraph rows (the exact max's generalized
    // derivative). No frozen QP cost and no fitted slip curve.
    std::vector<double> local=program->objective_gradient;
    std::array<double,4*kStages> best_margin{};
    best_margin.fill(std::numeric_limits<double>::infinity());
    for(const auto &row:program->rows)
        if(row.type==NonlinearRowType::OriginalSlipSubstep)
            best_margin[4*row.stage+row.index]=std::min(
                best_margin[4*row.stage+row.index],row.value);

    std::vector<double> row_weight(program->rows.size(),0.);
    std::array<int,4*kStages> counts{};
    for(size_t r=0;r<program->rows.size();++r) {
        const auto &row=program->rows[r];
        if(row.type!=NonlinearRowType::OriginalSlipSubstep)continue;
        const int peak=4*row.stage+row.index;
        if(std::abs(row.value-best_margin[peak])<=1.e-11 &&
           z->peak[row.stage][row.index]>0.)
            ++counts[peak];
    }
    for(int k=0;k<kStages;++k)
        for(int w=0;w<4;++w)
            if(counts[4*k+w]>1)++out->source_branch_ties;

    for(size_t r=0;r<program->rows.size();++r) {
        const auto &row=program->rows[r];
        if(row.type!=NonlinearRowType::OriginalSlipSubstep)continue;
        const int peak=4*row.stage+row.index;
        if(counts[peak] && std::abs(row.value-best_margin[peak])<=1.e-11) {
            row_weight[r]=local[peak0+peak]/double(counts[peak]);
            ++out->active_peak_rows;
        }
    }
    // All original substep inequalities are represented. At a solution
    // of the epigraph constraints, d peak = d source slip of active row.
    // Since g=peak-slip, its non-peak derivative is -d(source slip).
    for(const auto &entry:program->jacobian) {
        const double weight=row_weight[entry.row];
        if(weight==0. || entry.column>=peak0)continue;
        local[entry.column]-=weight*entry.forward;
    }

    // Preserve the exact ORIGINAL racing-objective differential before
    // adding feasibility penalties. The constrained QP uses this gradient
    // once and enforces physical constraints in its OWN inequality rows;
    // reusing the penalty gradient would double-count every wall/rate row.
    const std::vector<double> original_cost_local=local;

    // A single nonlinear merit for the ORIGINAL source inequalities.
    // Merit=C_source + (rho_wall/2) sum min(corner,0)^2
    //                + (rho_rate/2) sum min(rate,0)^2.
    // The source dynamics remain EXACT equalities under the same
    // constraint Jacobian. The extra derivative is the exact derivative
    // of this numerical merit on a smooth physical branch, NOT an
    // approximation of the original source racing objective.
    std::vector<double> constraint_weight(program->rows.size(),0.);
    for(size_t row=0;row<program->rows.size();++row) {
        const auto &r=program->rows[row];
        const double margin=
            r.type==NonlinearRowType::BodyCorner?
            r.value-penalties.minimum_corner_clearance_m:r.value;
        const double v=std::min(0.,margin);
        if(r.type==NonlinearRowType::BodyCorner) {
            out->exact_source_corridor_violation_half_squared+=.5*v*v;
            out->maximum_source_corridor_violation=std::max(
                out->maximum_source_corridor_violation,-v);
            // Exact nonsmooth penalty: rho*(-min(g,0)) plus a
            // squared term for useful curvature when far infeasible.
            constraint_weight[row]=v<0.?
                penalties.footprint*
                v:0.;
        } else if(r.type==NonlinearRowType::PhysicalCommandRate) {
            out->exact_source_rate_violation_half_squared+=.5*v*v;
            constraint_weight[row]=v<0.?
                penalties.command_rate*
                v:0.;
        }
    }
    out->exact_source_violation_half_squared=
        out->exact_source_corridor_violation_half_squared+
        out->exact_source_rate_violation_half_squared;
    out->exact_source_constraint_merit=
        out->exact_source_violation_half_squared;
    // Footprint and rate can have independent numerical penalties.
    // No smooth surrogate replaces the ORIGINAL physical functions.
    out->merit_objective=primal->objective;
    for(const auto &r:program->rows) {
        if(r.type!=NonlinearRowType::BodyCorner &&
           r.type!=NonlinearRowType::PhysicalCommandRate)continue;
        const double margin=
            r.type==NonlinearRowType::BodyCorner?
            r.value-penalties.minimum_corner_clearance_m:r.value;
        if(margin>=0.)continue;
        const double v=-margin;
        const double weight=r.type==NonlinearRowType::BodyCorner?
            penalties.footprint:penalties.command_rate;
        out->merit_objective+=weight*
            (.5*v*v);
    }
    for(const auto &e:program->jacobian) {
        const double multiplier=constraint_weight[e.row];
        if(multiplier!=0.) {
            // Original one-sided active reference slope is retained
            // in the sparse formulation. This uses its FORWARD branch;
            // exact nonlinear trial acceptance handles branch switches.
            local[e.column]+=multiplier*e.forward;
        }
    }

    std::array<std::array<double,kState>,kStages> adj{};
    std::array<std::array<double,kState>,kStages> original_cost_adj{};
    for(int k=0;k<kStages;++k)
        for(int j=0;j<kState;++j) {
            adj[k][j]=local[k*kState+j];
            original_cost_adj[k][j]=original_cost_local[k*kState+j];
        }

    // The source multiple-shooting Jacobian has exact sparse structure:
    // z[k+1]-F_source(z[k],u[k])=0. Its transpose produces an exact
    // active-branch reduced gradient without dense 60x60 QP curvature.
    std::vector<int> start(kDynamicsEquations+1,0);
    for(const auto &entry:program->jacobian)
        if(entry.row<kDynamicsEquations)
            ++start[entry.row+1];
    for(int i=1;i<=kDynamicsEquations;++i)start[i]+=start[i-1];
    for(int k=kStages-1;k>=0;--k) {
        double gradient[2]{local[control0+2*k],
                           local[control0+2*k+1]};
        double racing_gradient[2]{
            original_cost_local[control0+2*k],
            original_cost_local[control0+2*k+1]};
        for(int j=0;j<kState;++j) {
            const int row=k*kState+j;
            const double dual=adj[k][j];
            const double racing_dual=original_cost_adj[k][j];
            // Accumulate original objective and penalty-merit adjoints
            // through the SAME original-source dynamics Jacobian.
            for(int t=start[row];t<start[row+1];++t) {
                const auto &e=program->jacobian[t];
                if(e.row!=row)return false;
                if(e.column>=control0+2*k &&
                   e.column<control0+2*k+2) {
                    gradient[e.column-(control0+2*k)]-=e.forward*dual;
                    racing_gradient[e.column-(control0+2*k)]-=
                        e.forward*racing_dual;
                } else if(k>0 && e.column>=(k-1)*kState &&
                          e.column<k*kState) {
                    const int state=e.column-(k-1)*kState;
                    adj[k-1][state]-=e.forward*dual;
                    original_cost_adj[k-1][state]-=
                        e.forward*racing_dual;
                }
            }
        }
        for(int j=0;j<2;++j) {
            if(!std::isfinite(gradient[j]) ||
               !std::isfinite(racing_gradient[j]))return false;
            out->control[k][j]=gradient[j];
            out->original_source_cost_control[k][j]=racing_gradient[j];
        }
    }
    out->source_cost=primal->objective;
    out->source_minimum_corridor=primal->minimum_corridor_slack_m;
    out->minimum_rate_margin=primal->minimum_command_rate_margin;
    out->minimum_command_box_margin=primal->minimum_command_box_margin;
    out->nonsmooth_reference_stages=
        program->nonlinear_reference_knots+
        program->objective_nonsmooth_reference_progress_stages;
    out->valid=1;
    // Transfer this exact same assembled original-source Jacobian to
    // the nonlinear search. This avoids a second inconsistent source
    // AD evaluation and guarantees directional/KKT diagnostics use
    // precisely the Jacobian used by the reduced adjoint.
    if(source_linearization)
        *source_linearization=std::move(*program);
    return true;
}
} // namespace sdu_apex::source_nlp
