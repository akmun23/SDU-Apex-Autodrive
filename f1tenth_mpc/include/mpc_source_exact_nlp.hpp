#pragma once
/* Authoritative nonlinear source evaluation.
 *
 * No quadratic/QP objective, approximated plant, or substituted reference.
 * The runtime optimizes only 30 original steering/throttle pairs. Stored
 * 29-state nodes describe their exact condensed rollout. Independent-node
 * evaluation remains available for offline derivative/oracle checks; it is
 * not a multiple-shooting runtime solver. The initial state is fixed.
 *
 * This is the nonlinear problem oracle, NOT yet a solver. Source-model exact
 * does not imply bit-exact equivalence with the private Unity/PhysX solver.
 */
#include "mpc_rti.h"
#include "unity_source_plant_c.h"
#include <array>
#include <vector>

namespace sdu_apex::source_nlp {
inline constexpr int kStages=PREDICTION_HORIZON;
inline constexpr int kMaxControlledTerminalStages=24;
inline constexpr int kPhysical=UNITY_SOURCE_PLANT_STATE_SIZE;
inline constexpr int kState=kPhysical+3;
inline constexpr int kDynamicsEquations=kStages*kState;
inline constexpr int kDecisionVariables=kDynamicsEquations+2*kStages;

struct State {
    double z[kState]{};
};
struct Decision {
    // Node k stores the state at the END of prediction interval k.
    State nodes[kStages]{};
    MpcModelControl_t controls[kStages]{};
};
struct Evaluation {
    double dynamics[kStages][kState]{};
    double max_dynamics_abs{};
    double objective{};
    // The ORIGINAL max within the original 25ms physical integration;
    // these are score features, not state coordinates or fitted proxies.
    double source_peak_slip[kStages][4]{};
    double source_peak_slip_cost{};
    float minimum_corridor_slack_m{};
    float corridor[kStages+1][2]{};
    // Eight nonlinear corner constraints per stage (4 left + 4 right).
    // Unlike the source support-extent max, their heading derivatives
    // exist at a zero-heading corner tie.
    MpcRtiSourceCorridorCorners_t corridor_corners[kStages]{};
    int corridor_reference_kink_count{};
    int corridor_reference_derivative_invalid_count{};
    double corner_source_parity_max_abs{};
    // Original 120 source command-rate margins; positive is legal.
    double command_rate[kStages][4]{};
    // The original source legal input box (absolute physical targets).
    double command_box[kStages][4]{};
    double minimum_command_rate_margin{};
    double minimum_command_box_margin{};
    unsigned int source_branch_flags[kStages]{};
    // At a shooting iterate, the source transition starts from its
    // independent x_k variable, NOT the preceding predicted state.
    // This canonical reconstructed nominal is valid as the exact source
    // objective oracle even when the dynamics equalities are violated.
    int valid{};
};

// Smooth Lagrangian curvature is reusable only on the same original physical,
// peak-slip and progress-reference branches. Comparison never modifies a primal.
struct SourceCurvatureBranches {
    UnitySourcePhysicalBranchComparisonC physical{};
    int peak_changes{};
    int tracking_reference_changes{};
    int reference_changes{};
    bool same{};
};
bool compare_source_curvature_branches(const MpcRtiNominal_t &base,const Evaluation &base_score,
    const MpcRtiNominal_t &candidate,const Evaluation &candidate_score,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,double lap_length,float dt,
    SourceCurvatureBranches *result);

// Directional local constraint model for U(t)=U+t*dU. Canonical physical
// row order is all 240 corners, 120 rates, then 120 boxes. The second
// derivative includes propagated physical, implicit path-frame and corner
// curvature. Discontinuous source ties invalidate this model, never physics.
struct SourceConstraintCurvature {
    std::array<double,16*kStages> first{},second{};
    double original_objective_first{},original_objective_second{};
    int first_invalid_stage{-1};
    int physical_switch_ties{},selected_switch_ties{},reference_knots{};
    int contact_switches{},substep_switches{},brake_switches{};
    bool valid{};
};
bool source_constraint_path_curvature(const MpcRtiNominal_t &nominal,
    const Evaluation &score,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const double direction[kStages][2],SourceConstraintCurvature *result);

// Expensive offline reference in z=(steer/.15, throttle/.25) coordinates.
// Polarize exact source directional derivatives, then independently check
// their quadratic consistency. A nonsmooth nonquadratic path is rejected.
struct SourceLagrangianHessian {
    std::array<double,4*kStages*kStages> scaled{};
    int source_directional_evaluations{};
    double quadratic_consistency_relative{};
    bool valid{};
};
bool source_lagrangian_hessian(const MpcRtiNominal_t &nominal,
    const Evaluation &score,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const std::array<double,16*kStages> &physical_multipliers,
    SourceLagrangianHessian *result,double absolute_deadline_monotonic_us=0.);

// The EXACT active-branch Jacobian of the ORIGINAL nonlinear equalities.
 // Rows are grouped by source interval: r_k=z_{k+1}-F_source(z_k,u_k).
 // The initial state z_0 is fixed. No condensed or fitted dynamics.
struct StageJacobian {
    double A[kState][kState]{};
    double B[kState][2]{};
    // Derivative of the ORIGINAL max over all 1ms/internal tire substeps.
    // One source-selected active substep at a time; at a tie this is a
    // selected generalized gradient, not a fictitious smooth Hessian.
    double source_peak_slip[4]{};
    double peak_slip_state[4][kState]{};
    double peak_slip_control[4][2]{};
};
struct DynamicsJacobian {
    StageJacobian stages[kStages]{};
    int original_source_ad_calls{};
    int source_nonlinear_stage_calls{};
    int valid{};
};
// Exact active-branch gradient of the original LONGITUDINAL SLIP feature
// of the nonlinear racing objective, in INDEPENDENT shooting coordinates.
// The complete racing objective also contains other terms and is not yet
// represented by this feature gradient alone.
struct PeakSlipObjectiveGradient {
    double cost{};
    double state[kStages][kState]{}; // x_1..x_30
    double control[kStages][2]{};    // u_0..u_29
    int active_peak_wheels{};
    int valid{};
};
bool source_peak_slip_objective_gradient(
    const MpcRtiCycleConfiguration_t &configuration,
    const DynamicsJacobian &dynamics,
    PeakSlipObjectiveGradient *result);

/* Single authoritative source-stage physical/tracking Jacobian, also
 * used for exact terminal-continuation derivatives beyond the nominal
 * 30-stage horizon. No separate stage model, no omitted hidden states.
 */
bool linearize_original_source_stage(
    const MpcRtiState_t &current,double current_progress,
    const MpcModelControl_t &command,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,StageJacobian *jacobian,
    MpcRtiState_t *next,double *next_progress);

bool linearize_dynamics(const MpcRtiState_t &initial,
    double initial_progress,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const Decision &decision,DynamicsJacobian *result);

void pack_state(const MpcRtiState_t &state,double progress,State *out);
bool inject_state(const State &in,MpcRtiState_t *inout,double *progress);
bool decision_from_nominal(const MpcRtiNominal_t &nominal,Decision *out);

/* Exact source longitudinal-slip MAX as nonlinear epigraph constraints.
 * For every original internal tire substep:
 *      peak[k,w] - source_excess[k,tick,substep,w] >= 0.
 * and peak[k,w] >= 0. Minimizing original objective with
 * weight * peak[k,w]^2 is THEN EXACTLY EQUIVALENT to source-max scoring,
 * without fitting or differentiating a single frozen peak branch.
 *
 * This is an offline NLP objective/constraints oracle, NOT a solver.
 */
struct SlipEpigraphDecision {
    Decision shooting{};
    double peak[kStages][4]{};
};
struct SlipEpigraphRow {
    int stage{};
    int wheel{};
    int fixed_tick{};
    int vehicle_substep{};
    double source_excess{};
    double margin{};
};
struct SlipEpigraphEvaluation {
    Evaluation source{};
    double objective{};
    double original_peak_cost{};
    double epigraph_peak_cost{};
    double min_epigraph_margin{};
    double min_peak_nonnegative_margin{};
    int physical_source_rows{};
    std::vector<SlipEpigraphRow> rows;
    int valid{};
};
/* Sparse nonlinear epigraph Jacobian of the ORIGINAL source
 * slip substep expressions. For a row
 *     g = peak[k,w] - physical_slip_excess(z_k, u_k)
 * the exact active-branch Jacobian is:
 *     dg/dpeak=1, dg/dz_k=-source_state_derivative (26 physical entries),
 *     dg/du_k=-source_control_derivative (2 original controls).
 * There are NO arbitrary full-horizon couplings or fitted tire models.
 * At an actual branch kink, one-sided selected derivatives are reported.
 */
struct SlipEpigraphJacobianRow {
    int stage{};
    int wheel{};
    int fixed_tick{};
    int vehicle_substep{};
    double original_source_excess{};
    double previous_physical_state[26]{};
    double control[2]{};
};
struct SlipEpigraphJacobian {
    std::vector<SlipEpigraphJacobianRow> rows;
    int original_source_ad_calls{};
    int original_nonlinear_stage_calls{};
    int valid{};
};
// One sparse nonlinear source AD pass for dynamics equality AND all
// source tire-slip epigraph inequality Jacobians. The selected physical
// branch for both is identical by construction.
bool linearize_dynamics_and_slip_epigraph(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,const Decision &decision,
    DynamicsJacobian *dynamics,SlipEpigraphJacobian *slip,
    const MpcRtiNominal_t *known_original_source_trace=nullptr,
    int requested_threads=0);
/* Derivative of the ORIGINAL nonlinear stage racing objective in sparse
 * shooting coordinates with slip represented by exact epigraph variables.
 * Uses the scorer's frozen objective schedule, actual source lateral-slip
 * AD, physical tracking/steering/command residuals and source footprint.
 *
 * This is NOT the full Unity/PhysX equivalence certificate. The source
 * callback's float state/command ABI is preserved, so independent numerical
 * secants must use physically resolvable perturbations.
 * Where a wall-reserve cost has multiple equally active corners, the
 * gradient is a selected subgradient and classical differentiability is
 * NOT asserted; the corresponding nonsmooth flag remains set.
 */
struct ExactSourceObjectiveGradient {
    double state[kStages][kState]{};
    double control[kStages][2]{};
    double peak[kStages][4]{};
    // One-sided value if ORIGINAL loaded corridor width has a knot.
    double state_progress_backward[kStages]{};
    double original_source_objective{};
    double epigraph_objective{};
    int nonsmooth_wall_reserve_stages{};
    int nonsmooth_reference_progress_stages{};
    int evaluated_source_lateral_slip_stages{};
    int valid{};
};
bool original_source_objective_gradient_from_source_trace(
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,
    const MpcRtiNominal_t &source_trace,
    const Evaluation &source_score,
    ExactSourceObjectiveGradient *result);
bool original_source_objective_gradient(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,
    ExactSourceObjectiveGradient *result);

/* Accuracy-first SPARSE constraint Jacobian of the complete nonlinear
 * source NLP (including one-sided reference knot derivatives).
 * Columns: 870 independent 29-state shooting coordinates, 60 source
 * controls, 120 exact physical slip-epigraph variables.
 * Rows: 870 nonlinear source dynamics equalities; 240 original body-corner
 * clearances; 120 physical actuator rates; 120 command-box margins;
 * 120 nonnegative peak-slip conditions; and all real tire substep rows.
 *
 * Jacobian coordinates hold BOTH forward and backward directional entries,
 * so the solver must select according to the ACTUAL signed perturbation at
 * a loaded reference knot. No averaging into a fictitious smooth slope.
 */
enum class NonlinearRowType : unsigned char {
    DynamicsEquality, BodyCorner, PhysicalCommandRate,
    PhysicalCommandBox, PeakNonnegative, OriginalSlipSubstep,
    TerminalSpeedViability
};
struct SparseNonlinearRow {
    NonlinearRowType type{};
    int stage{};
    int index{};
    int source_fixed_tick{-1};
    int source_vehicle_substep{-1};
    double value{};
};
// Persistent SQP dual and Lagrangian memory is attached to the IDENTITY
// of each physical source constraint, not merely to a vector position.
// Two branches may produce the same row count with different constraints.
// Ignore the changing numerical margin; keep stage, corner/rate index and
// source fixed-tick/substep identity exact. No fuzzy value comparison.
inline bool same_original_source_constraint_identity(
    const SparseNonlinearRow &a,const SparseNonlinearRow &b)
{
    return a.type==b.type && a.stage==b.stage && a.index==b.index &&
           a.source_fixed_tick==b.source_fixed_tick &&
           a.source_vehicle_substep==b.source_vehicle_substep;
}
inline bool same_original_source_constraint_layout(
    const std::vector<SparseNonlinearRow> &previous,
    const std::vector<SparseNonlinearRow> &current)
{
    if(previous.size()!=current.size())return false;
    for(size_t i=0;i<previous.size();++i)
        if(!same_original_source_constraint_identity(previous[i],current[i]))
            return false;
    return true;
}

struct SparseNonlinearDerivative {
    int row{};
    int column{};
    double forward{};
    double backward{};
};
struct SparseNonlinearProgram {
    std::vector<SparseNonlinearRow> rows;
    std::vector<SparseNonlinearDerivative> jacobian;
    // The source AD stage blocks already used to construct the sparse
    // nonlinear dynamics equalities. Retain them for a consistent,
    // source-derived reduced constraint-search metric (no refitting).
    std::vector<StageJacobian> original_source_stage_derivatives;
    std::vector<SlipEpigraphJacobianRow> original_source_slip_substeps;
    // Exact first derivatives of the same original nonlinear stage scorer.
    // Two-sided progress slopes remain separate at a reference-bound knot.
    std::vector<double> objective_gradient;
    std::vector<double> objective_gradient_progress_backward;
    int objective_nonsmooth_wall_reserve_stages{};
    int objective_nonsmooth_reference_progress_stages{};
    int num_variables{};
    int equality_rows{};
    int nonlinear_reference_knots{};
    int invalid_reference_derivatives{};
    double original_source_objective{};
    double epigraph_objective{};
    // ORIGINAL scoring weight/reserve; used ONLY to resolve the exact
    // directional derivative at a tied physical bumper/corner maximum.
    double original_source_corridor_reserve_weight{};
    double original_source_corridor_reserve_m{};
    int valid{};
};

/* Exact-dynamics elimination of the source NLP for offline control-only
 * nonlinear optimization. This is not a quadratic/QP cost approximation:
 * every trial still runs the original source integrator and original scorer.
 * The 870 local source dynamics equality residuals are eliminated
 * exactly by the original source transition Jacobians. The 120 epigraph peaks are the original exact within-tick
 * maxima; the derivative is assembled from the active ORIGINAL substep
 * inequalities and recomputed after every accepted nonlinear trial.
 *
 * At source-branch ties the returned gradient is a selected generalized
 * derivative. It is never claimed to predict large cross-branch steps.
 */
struct ConstraintMeritWeights {
    // Numerical weights used to differentiate the original physical
    // inequalities. These are not alternative racing objectives.
    double footprint{};
    double command_rate{};
    double minimum_corner_clearance_m{};
    double terminal_speed_viability_weight{};
    double terminal_speed_allowance_mps{};
    bool include_original_slip_epigraph{};
};
struct ReducedSourceGradient {
    double control[kStages][2]{};
    double original_source_cost_control[kStages][2]{};
    double merit_objective{};
    double exact_source_violation_half_squared{};
    double exact_source_constraint_merit{};
    double exact_source_corridor_violation_half_squared{};
    double exact_source_rate_violation_half_squared{};
    double maximum_source_corridor_violation{};
    double source_cost{};
    double source_minimum_corridor{};
    double minimum_rate_margin{};
    double minimum_command_box_margin{};
    int active_peak_rows{};
    int source_branch_ties{};
    int nonsmooth_reference_stages{};
    int valid{};
};
bool source_exact_reduced_gradient(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &source_feasible_decision,
    ReducedSourceGradient *out,
    ConstraintMeritWeights penalties={},
    SparseNonlinearProgram *source_linearization=nullptr,
    // Optional same-epoch original-source primal/trace, previously scored
    // by the exact nonlinear acceptance oracle. BOTH must be supplied.
    const Evaluation *known_original_score=nullptr,
    const MpcRtiNominal_t *known_original_trace=nullptr);

bool assemble_sparse_nonlinear_program(const MpcRtiState_t &initial,
    double initial_progress,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,SparseNonlinearProgram *result,
    const MpcRtiNominal_t *known_original_trace=nullptr,
    const Evaluation *known_original_score=nullptr);

bool linearize_slip_epigraph(const MpcRtiState_t &initial,
    double initial_progress,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const Decision &decision,SlipEpigraphJacobian *result);

bool evaluate_slip_epigraph_with_nominal(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,
    SlipEpigraphEvaluation *result,MpcRtiNominal_t *source_trace,
    const MpcRtiNominal_t *known_original_trace=nullptr);
bool evaluate_slip_epigraph(const MpcRtiState_t &initial,
    double initial_progress,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const SlipEpigraphDecision &decision,SlipEpigraphEvaluation *result);

/* Return the exact same shooting state trace that the authoritative
 * nonlinear source scorer actually consumed. Derivative clients may
 * differentiate its original physical residuals without rebuilding a
 * different rollout or copying an independent reference implementation.
 */
bool evaluate_existing_source_nominal(
    const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const MpcRtiNominal_t &known_source,
    const Decision &decision,Evaluation *result);

bool evaluate_with_nominal(const MpcRtiState_t &initial,
    double initial_progress,const MpcTrajectorySample_t *trajectory,
    size_t trajectory_count,double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &decision,Evaluation *result,
    MpcRtiNominal_t *original_source_trace);

bool evaluate(const MpcRtiState_t &initial,double initial_progress,
    const MpcTrajectorySample_t *trajectory,size_t trajectory_count,
    double lap_length,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &decision,Evaluation *result);
} // namespace sdu_apex::source_nlp
