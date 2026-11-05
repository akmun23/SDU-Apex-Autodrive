#include "mpc_source_exact_nlp.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace sdu_apex::source_nlp {
namespace {
constexpr int FirstControl=kStages*kState;
constexpr int FirstEpigraph=FirstControl+2*kStages;
int state_column(int stage,int index) {return stage*kState+index;}
int control_column(int stage,int index) {
    return FirstControl+2*stage+index;
}
int peak_column(int stage,int wheel) {
    return FirstEpigraph+4*stage+wheel;
}
}

bool assemble_sparse_nonlinear_program(const MpcRtiState_t &initial,
    double initial_progress,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,SparseNonlinearProgram *result,
    const MpcRtiNominal_t *known_original_trace,
    const Evaluation *known_original_score)
{
    if(!result ||
       (known_original_score && !known_original_trace))return false;
    result->rows.clear();
    result->jacobian.clear();
    result->valid=0;
    result->num_variables=FirstEpigraph+4*kStages;
    result->equality_rows=kStages*kState;
    result->nonlinear_reference_knots=0;
    result->invalid_reference_derivatives=0;
    auto primal=std::make_unique<SlipEpigraphEvaluation>();
    auto source=std::make_unique<DynamicsJacobian>();
    auto slip=std::make_unique<SlipEpigraphJacobian>();
    auto objective=std::make_unique<ExactSourceObjectiveGradient>();
    auto trace=std::make_unique<MpcRtiNominal_t>();
    if(known_original_score) {
        // The original physical nonlinear oracle already evaluated the
        // SAME controls, hidden source states, 8 footprint corners,
        // objective and active tire maxima. The exact original source AD
        // below produces the physical substep slip VALUES along with
        // their Jacobians. Re-running the original physical substep loop
        // solely to enumerate identical epigraph margins is redundant.
        // This fast path is ONLY valid at the original minimal epigraph
        // peaks, not at an arbitrary independent shooting iterate.
        if(!known_original_score->valid ||
           known_original_score->max_dynamics_abs>1.e-8 ||
           !known_original_trace->valid ||
           known_original_trace->horizon!=kStages ||
           known_original_trace->progress[0]!=initial_progress)
            return false;
        for(int k=0;k<kStages;++k)
            for(int w=0;w<4;++w)
                if(decision.peak[k][w]!=
                   known_original_score->source_peak_slip[k][w])
                    return false;
        *trace=*known_original_trace;
        if(!linearize_dynamics_and_slip_epigraph(
              initial,initial_progress,trajectory,trajectory_count,
              lap_length,dt,decision.shooting,source.get(),slip.get(),
              trace.get()))return false;
        primal->source=*known_original_score;
        primal->objective=known_original_score->objective;
        primal->original_peak_cost=
            known_original_score->source_peak_slip_cost;
        primal->epigraph_peak_cost=primal->original_peak_cost;
        primal->min_epigraph_margin=INFINITY;
        primal->min_peak_nonnegative_margin=INFINITY;
        primal->rows.reserve(slip->rows.size());
        for(int k=0;k<kStages;++k)
            for(int w=0;w<4;++w)
                primal->min_peak_nonnegative_margin=std::min(
                    primal->min_peak_nonnegative_margin,decision.peak[k][w]);
        double original_active_max[kStages][4]{};
        for(const auto &row:slip->rows) {
            if(row.stage<0 || row.stage>=kStages ||
               row.wheel<0 || row.wheel>=4 ||
               !std::isfinite(row.original_source_excess))return false;
            original_active_max[row.stage][row.wheel]=std::max(
                original_active_max[row.stage][row.wheel],
                row.original_source_excess);
            const double margin=decision.peak[row.stage][row.wheel]-
                                row.original_source_excess;
            primal->rows.push_back({row.stage,row.wheel,
                row.fixed_tick,row.vehicle_substep,
                row.original_source_excess,margin});
            primal->min_epigraph_margin=std::min(
                primal->min_epigraph_margin,margin);
        }
        // Every ORIGINAL independent slip maximum must equal the max
        // from those original physical substep values; fail closed on
        // any AD/source branch mismatch instead of hiding in a QP.
        for(int k=0;k<kStages;++k)
            for(int w=0;w<4;++w)
                if(std::abs(original_active_max[k][w]-
                     known_original_score->source_peak_slip[k][w])>1.e-10)
                    return false;
        primal->physical_source_rows=int(primal->rows.size());
        primal->valid=std::isfinite(primal->objective) &&
            std::isfinite(primal->min_epigraph_margin) &&
            std::isfinite(primal->min_peak_nonnegative_margin);
    } else if(!evaluate_slip_epigraph_with_nominal(
         initial,initial_progress,trajectory,trajectory_count,
         lap_length,dt,configuration,decision,primal.get(),trace.get(),
         known_original_trace) ||
       !linearize_dynamics_and_slip_epigraph(
         initial,initial_progress,trajectory,trajectory_count,
         lap_length,dt,decision.shooting,source.get(),slip.get(),
         trace.get()))return false;
    if(!original_source_objective_gradient_from_source_trace(
          configuration,decision,*trace,primal->source,objective.get()))
        return false;
    if(!primal->valid || !source->valid || !slip->valid ||
       !objective->valid ||
       primal->rows.size()!=slip->rows.size() ||
       std::abs(primal->objective-objective->epigraph_objective)>1.e-8)
        return false;
    result->original_source_stage_derivatives.assign(
        source->stages,source->stages+kStages);
    result->objective_gradient.assign(result->num_variables,0.);
    result->objective_gradient_progress_backward.assign(
        result->num_variables,0.);
    for(int k=0;k<kStages;++k) {
        for(int j=0;j<kState;++j) {
            const int index=state_column(k,j);
            result->objective_gradient[index]=objective->state[k][j];
            result->objective_gradient_progress_backward[index]=
                j==kPhysical+2?
                objective->state_progress_backward[k]:
                objective->state[k][j];
        }
        for(int j=0;j<2;++j) {
            const int index=control_column(k,j);
            result->objective_gradient[index]=objective->control[k][j];
            result->objective_gradient_progress_backward[index]=
                objective->control[k][j];
        }
        for(int w=0;w<4;++w) {
            const int index=peak_column(k,w);
            result->objective_gradient[index]=objective->peak[k][w];
            result->objective_gradient_progress_backward[index]=
                objective->peak[k][w];
        }
    }
    result->objective_nonsmooth_wall_reserve_stages=
        objective->nonsmooth_wall_reserve_stages;
    result->objective_nonsmooth_reference_progress_stages=
        objective->nonsmooth_reference_progress_stages;

    result->rows.reserve(kStages*(kState+8+4+4+4)+slip->rows.size());
    result->jacobian.reserve(200000+slip->rows.size()*29);
    const auto add_row=[&](NonlinearRowType type,int stage,int index,
                           double value,int tick=-1,int substep=-1) {
        const int row=static_cast<int>(result->rows.size());
        result->rows.push_back({type,stage,index,tick,substep,value});
        return row;
    };
    const auto add=[&](int row,int col,double forward,double backward) {
        if(!std::isfinite(forward)||!std::isfinite(backward))
            return false;
        if(forward!=0.||backward!=0.)
            result->jacobian.push_back({row,col,forward,backward});
        return true;
    };

    // Nonlinear equality E_k = z_(k+1) - source_F(z_k, u_k) = 0.
    // No eliminated/simplified physical states and no surrogate plant.
    for(int k=0;k<kStages;++k) {
        const auto &st=source->stages[k];
        for(int i=0;i<kState;++i) {
            const int row=add_row(
                NonlinearRowType::DynamicsEquality,k,i,
                primal->source.dynamics[k][i]);
            if(!add(row,state_column(k,i),1.,1.))return false;
            if(k>0)
                for(int j=0;j<kState;++j)
                    if(!add(row,state_column(k-1,j),
                            -st.A[i][j],-st.A[i][j]))return false;
            for(int j=0;j<2;++j)
                if(!add(row,control_column(k,j),
                        -st.B[i][j],-st.B[i][j]))return false;
        }
    }
    if(static_cast<int>(result->rows.size())!=result->equality_rows)
        return false;

    // Original physical rectangle expressed as all eight source corners.
    // No single active bumper/cosine support chosen at e_psi=0.
    for(int k=0;k<kStages;++k) {
        const auto &g=primal->source.corridor_corners[k];
        result->nonlinear_reference_knots+=g.reference_progress_kink!=0;
        result->invalid_reference_derivatives+=
            !g.reference_derivatives_valid;
        if(!g.reference_derivatives_valid)
            return false;  // don't invent a progress derivative
        for(int side=0;side<2;++side)
            for(int corner=0;corner<4;++corner) {
                const bool left=side==0;
                const int row=add_row(
                    NonlinearRowType::BodyCorner,k,4*side+corner,
                    left?g.left[corner]:g.right[corner]);
                if(!add(row,state_column(k,kPhysical),
                        left?-1.:1.,left?-1.:1.))return false;
                const double dpsi=left?
                    g.left_d_psi[corner]:g.right_d_psi[corner];
                if(!add(row,state_column(k,kPhysical+1),dpsi,dpsi))
                    return false;
                const double fwd=left?
                    g.left_d_progress_forward:g.right_d_progress_forward;
                const double bwd=left?
                    g.left_d_progress_backward:g.right_d_progress_backward;
                if(!add(row,state_column(k,kPhysical+2),fwd,bwd))
                    return false;
            }
    }

    // Steering authority is a genuine physical rack/target relationship,
    // not a trust region around the preceding optimizer iterate.
    for(int k=0;k<kStages;++k) {
        for(int side=0;side<4;++side) {
            const int row=add_row(
                NonlinearRowType::PhysicalCommandRate,k,side,
                primal->source.command_rate[k][side]);
            const int control=side<2?0:1;
            const double sign=side%2==0?-1.:1.;
            if(!add(row,control_column(k,control),sign,sign))
                return false;
            if(side<2) {
                if(k>0 && !add(row,
                       state_column(k-1,UNITY_SOURCE_PHYSICAL_STEERING),
                       -sign,-sign))return false;
            } else {
                // The original source appends u_(k-1) directly into
                // delayed_throttle_command_2 at the preceding stage.
                // Thus it is a command-history coordinate, not a missing
                // 30th independent physical shooting state.
                if(k>0 && !add(row,control_column(k-1,1),
                              -sign,-sign))return false;
            }
        }
    }

    for(int k=0;k<kStages;++k)
        for(int side=0;side<4;++side) {
            const int row=add_row(
                NonlinearRowType::PhysicalCommandBox,k,side,
                primal->source.command_box[k][side]);
            const double sign=side%2==0?-1.:1.;
            if(!add(row,control_column(k,side<2?0:1),
                    sign,sign))return false;
        }

    for(int k=0;k<kStages;++k)
        for(int wheel=0;wheel<4;++wheel) {
            const int row=add_row(
                NonlinearRowType::PeakNonnegative,k,wheel,
                decision.peak[k][wheel]);
            if(!add(row,peak_column(k,wheel),1.,1.))return false;
        }

    // Original tire-slip EPIGRAPH inequalities, one row for every actual
    // internal tire substep, not just an active maximum linearization.
    for(size_t i=0;i<slip->rows.size();++i) {
        const auto &s=slip->rows[i];
        const auto &p=primal->rows[i];
        if(s.stage!=p.stage||s.wheel!=p.wheel||
           s.fixed_tick!=p.fixed_tick||
           s.vehicle_substep!=p.vehicle_substep||
           s.original_source_excess!=p.source_excess)return false;
        const int row=add_row(
            NonlinearRowType::OriginalSlipSubstep,s.stage,s.wheel,
            p.margin,s.fixed_tick,s.vehicle_substep);
        if(!add(row,peak_column(s.stage,s.wheel),1.,1.))
            return false;
        if(s.stage>0)
            for(int j=0;j<kPhysical;++j)
                if(!add(row,state_column(s.stage-1,j),
                        -s.previous_physical_state[j],
                        -s.previous_physical_state[j]))
                    return false;
        for(int j=0;j<2;++j)
            if(!add(row,control_column(s.stage,j),-s.control[j],
                    -s.control[j]))return false;
    }

    result->original_source_objective=primal->source.objective;
    result->epigraph_objective=primal->objective;
    result->original_source_corridor_reserve_weight=
        configuration.model.weight_source_corridor_reserve;
    result->original_source_corridor_reserve_m=
        configuration.model.near_wall_tracking_reserve_m;
    if(result->num_variables!=kDecisionVariables+4*kStages ||
       static_cast<int>(result->rows.size())!=
       result->equality_rows+8*kStages+4*kStages+
       4*kStages+4*kStages+static_cast<int>(slip->rows.size()))
        return false;
    result->valid=1;
    return true;
}
} // namespace sdu_apex::source_nlp
