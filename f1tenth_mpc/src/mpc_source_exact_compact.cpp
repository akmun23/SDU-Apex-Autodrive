#include "mpc_source_exact_compact.hpp"
#include "mpc_source_terminal_speed_viability.hpp"
#include "vehicle_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace sdu_apex::source_nlp {
bool source_exact_compact_qp_linearization(
    const MpcRtiState_t &initial,double progress,
    const MpcTrajectorySample_t *track,size_t count,double lap,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &decision,const Evaluation &original_score,
    const MpcRtiNominal_t &original_nominal,
    ConstraintMeritWeights penalties,
    ReducedSourceGradient *out,SparseNonlinearProgram *program)
{
    if(!out||!program||!original_score.valid||!original_nominal.valid||
       original_nominal.horizon!=kStages ||
       original_score.max_dynamics_abs>1.e-8 ||
       progress!=original_nominal.progress[0] ||
       penalties.footprint<0. || penalties.command_rate<0. ||
       penalties.minimum_corner_clearance_m<0. ||
       !std::isfinite(penalties.minimum_corner_clearance_m))
        return false;
    *out={};
    *program={};
    constexpr int state_count=kState*kStages;
    constexpr int control_count=2*kStages;
    constexpr int nvar=kDecisionVariables+4*kStages;
    program->equality_rows=state_count;
    program->num_variables=nvar;
    program->original_source_objective=original_score.objective;
    auto dynamics=std::make_unique<DynamicsJacobian>();
    auto slip_rows=std::make_unique<SlipEpigraphJacobian>();
    if(!linearize_dynamics_and_slip_epigraph(
        initial,progress,track,count,lap,dt,decision,
        dynamics.get(),penalties.include_original_slip_epigraph?slip_rows.get():nullptr,&original_nominal,
        configuration.model.jacobian_threads) || !dynamics->valid)
        return false;
    program->original_source_stage_derivatives.assign(
        dynamics->stages,dynamics->stages+kStages);
    program->original_source_slip_substeps=std::move(slip_rows->rows);

    auto epigraph=std::make_unique<SlipEpigraphDecision>();
    epigraph->shooting=decision;
    for(int k=0;k<kStages;++k)
        for(int wheel=0;wheel<4;++wheel)
            epigraph->peak[k][wheel]=
                original_score.source_peak_slip[k][wheel];
    auto objective=std::make_unique<ExactSourceObjectiveGradient>();
    auto peak=std::make_unique<PeakSlipObjectiveGradient>();
    if(!original_source_objective_gradient_from_source_trace(
         configuration,*epigraph,original_nominal,
         original_score,objective.get()) ||
       !source_peak_slip_objective_gradient(
         configuration,*dynamics,peak.get()) ||
       !objective->valid || !peak->valid ||
       std::abs(peak->cost-original_score.source_peak_slip_cost)>
         1.e-8*std::max(1.,original_score.source_peak_slip_cost) ||
       std::abs(objective->epigraph_objective-original_score.objective)>
         1.e-7*std::max(1.,std::abs(original_score.objective)))
        return false;

    std::array<std::array<double,kState>,kStages> original_state{};
    std::array<std::array<double,2>,kStages> original_control{};
    for(int k=0;k<kStages;++k) {
        for(int j=0;j<kState;++j)
            original_state[k][j]=objective->state[k][j]+peak->state[k][j];
        for(int j=0;j<2;++j)
            original_control[k][j]=
                objective->control[k][j]+peak->control[k][j];
    }
    auto merit_state=original_state;
    auto merit_control=original_control;

    program->rows.reserve(state_count+kStages*16);
    program->jacobian.reserve(kStages*75);
    // Placeholder dynamics equality rows let physical Jacobian coordinates
    // reuse the canonical original 29-state/2-command shooting layout.
    // Only the original source AD A/B are needed for reduced-space QP
    // elimination; inactive substep epigraph derivatives are not needed.
    for(int k=0;k<kStages;++k)
        for(int j=0;j<kState;++j)
            program->rows.push_back(
                {NonlinearRowType::DynamicsEquality,k,j,-1,-1,0.});
    const auto add=[&](NonlinearRowType type,int stage,int index,
                        double margin,
                        std::initializer_list<std::pair<int,double>> terms) {
        if(!std::isfinite(margin))return false;
        const int row=static_cast<int>(program->rows.size());
        program->rows.push_back({type,stage,index,-1,-1,margin});
        for(const auto &[col,slope]:terms) {
            if(col<0||col>=state_count+control_count||
               !std::isfinite(slope))
                return false;
            if(slope!=0.)
                program->jacobian.push_back({row,col,slope,slope});
        }
        if(type==NonlinearRowType::BodyCorner ||
           type==NonlinearRowType::PhysicalCommandRate) {
            const double shifted_margin=margin-
                (type==NonlinearRowType::BodyCorner?
                 penalties.minimum_corner_clearance_m:0.);
            const double v=std::min(0.,shifted_margin);
            if(type==NonlinearRowType::BodyCorner) {
                out->exact_source_corridor_violation_half_squared+=.5*v*v;
                out->maximum_source_corridor_violation=
                    std::max(out->maximum_source_corridor_violation,-v);
            } else
                out->exact_source_rate_violation_half_squared+=.5*v*v;
            if(v<0.) {
                const double weight=type==NonlinearRowType::BodyCorner?
                    penalties.footprint:penalties.command_rate;
                const double multiplier=weight*
                    v;
                for(const auto &[col,slope]:terms) {
                    if(col<state_count)
                        merit_state[col/kState][col%kState]+=
                            multiplier*slope;
                    else
                        merit_control[(col-state_count)/2][
                            (col-state_count)%2]+=multiplier*slope;
                }
            }
        }
        return true;
    };
    for(int k=0;k<kStages;++k) {
        const auto &geometry=original_score.corridor_corners[k];
        if(!geometry.reference_derivatives_valid)return false;
        program->nonlinear_reference_knots+=
            geometry.reference_progress_kink!=0;
        for(int side=0;side<2;++side)
            for(int corner=0;corner<4;++corner) {
                const bool left=side==0;
                const int col=k*kState;
                const double margin=left?
                    geometry.left[corner]:geometry.right[corner];
                const double heading=left?
                    geometry.left_d_psi[corner]:
                    geometry.right_d_psi[corner];
                const double path=left?
                    geometry.left_d_progress_forward:
                    geometry.right_d_progress_forward;
                if(!add(NonlinearRowType::BodyCorner,k,side*4+corner,
                    margin,{{col+kPhysical,left?-1.:1.},
                            {col+kPhysical+1,heading},
                            {col+kPhysical+2,path}}))
                    return false;
            }
    }
    if(penalties.terminal_speed_viability_weight>0.) {
        if(!std::isfinite(penalties.terminal_speed_allowance_mps) ||
           penalties.terminal_speed_allowance_mps<0.)return false;
        TerminalSpeedViability terminal{};
        if(!terminal_source_speed_viability(
             track,count,lap,original_nominal,
             penalties.terminal_speed_allowance_mps,&terminal))
            return false;
        // At a real speed knot the periodic piecewise-linear reference
        // has two distinct one-sided progress derivatives. Both cuts are
        // included in the SAME constrained SQP model; no invented spline,
        // tangent averaging or additional nonlinear candidate search.
        const int stage=kStages-1;
        const int col=stage*kState;
        const bool kink=terminal.at_reference_knot &&
            std::abs(terminal.slope_forward-
                     terminal.slope_backward)>1.e-9;
        const int cuts=kink?2:1;
        for(int side=0;side<cuts;++side) {
            const double slope=side==0?
                terminal.slope_forward:terminal.slope_backward;
            if(!add(NonlinearRowType::TerminalSpeedViability,
                stage,side,terminal.margin_mps,
                {{col+UNITY_SOURCE_U,-1.},
                 {col+kPhysical+2,slope}}))
                return false;
        }
    }
    for(int k=0;k<kStages;++k) {
        const int u=state_count+2*k;
        for(int side=0;side<4;++side) {
            const double sign=side%2==0?-1.:1.;
            const double margin=original_score.command_rate[k][side];
            if(side<2) {
                if(k==0) {
                    if(!add(NonlinearRowType::PhysicalCommandRate,
                        k,side,margin,{{u,sign}}))return false;
                } else {
                    if(!add(NonlinearRowType::PhysicalCommandRate,
                        k,side,margin,
                        {{u,sign},
                         {(k-1)*kState+
                          UNITY_SOURCE_PHYSICAL_STEERING,-sign}}))
                        return false;
                }
            } else {
                if(k==0) {
                    if(!add(NonlinearRowType::PhysicalCommandRate,
                        k,side,margin,{{u+1,sign}}))return false;
                } else {
                    if(!add(NonlinearRowType::PhysicalCommandRate,
                        k,side,margin,{{u+1,sign},{u-1,-sign}}))
                        return false;
                }
            }
        }
    }
    for(int k=0;k<kStages;++k)
        for(int side=0;side<4;++side) {
            const double sign=side%2==0?-1.:1.;
            const int col=state_count+2*k+(side<2?0:1);
            if(!add(NonlinearRowType::PhysicalCommandBox,k,side,
                original_score.command_box[k][side],
                {{col,sign}}))return false;
        }

    const auto adjoint=[&](
        std::array<std::array<double,kState>,kStages> state,
        const std::array<std::array<double,2>,kStages> &direct,
        double result[kStages][2]) {
        for(int k=kStages-1;k>=0;--k) {
            const auto &stage=dynamics->stages[k];
            for(int input=0;input<2;++input) {
                double v=direct[k][input];
                for(int j=0;j<kState;++j)
                    v+=stage.B[j][input]*state[k][j];
                result[k][input]=v;
                if(!std::isfinite(v))return false;
            }
            if(k>0)
                for(int col=0;col<kState;++col) {
                    double value=0.;
                    for(int j=0;j<kState;++j)
                        value+=stage.A[j][col]*state[k][j];
                    state[k-1][col]+=value;
                }
        }
        return true;
    };
    if(!adjoint(original_state,original_control,
         out->original_source_cost_control) ||
       !adjoint(merit_state,merit_control,out->control))
        return false;
    out->source_cost=original_score.objective;
    out->source_minimum_corridor=original_score.minimum_corridor_slack_m;
    out->minimum_rate_margin=original_score.minimum_command_rate_margin;
    out->minimum_command_box_margin=original_score.minimum_command_box_margin;
    out->exact_source_violation_half_squared=
        out->exact_source_corridor_violation_half_squared+
        out->exact_source_rate_violation_half_squared;
    out->exact_source_constraint_merit=
        out->exact_source_violation_half_squared;
    out->merit_objective=original_score.objective+
        penalties.footprint*
            out->exact_source_corridor_violation_half_squared+
        penalties.command_rate*
            out->exact_source_rate_violation_half_squared;

    out->nonsmooth_reference_stages=
        program->nonlinear_reference_knots+
        objective->nonsmooth_reference_progress_stages;
    program->original_source_corridor_reserve_weight=
        configuration.model.weight_source_corridor_reserve;
    program->original_source_corridor_reserve_m=
        configuration.model.near_wall_tracking_reserve_m;
    program->objective_nonsmooth_reference_progress_stages=
        objective->nonsmooth_reference_progress_stages;
    program->valid=1;
    out->valid=1;
    return true;
}
} // namespace sdu_apex::source_nlp
