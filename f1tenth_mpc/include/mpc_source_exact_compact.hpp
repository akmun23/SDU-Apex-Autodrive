#pragma once
#include "mpc_source_exact_nlp.hpp"

// Exact-source reduced-space linearization for the constrained-QP MPC.
// Avoid materializing *inactive* source tire-slip epigraph Jacobian rows:
// the source-supplied selected max-substep derivative is already the
// generalized gradient of the ORIGINAL max-slip cost. At a physical
// branch tie the subgradient can differ from the full epigraph tie-average;
// source-fidelity parity tests must qualify that mode before production.
// Original source forward dynamics, nonlinear cost and exact full source
// trial acceptance are unchanged.
namespace sdu_apex::source_nlp {
bool source_exact_compact_qp_linearization(
    const MpcRtiState_t &initial,double progress,
    const MpcTrajectorySample_t *track,size_t count,double lap,float dt,
    const MpcRtiCycleConfiguration_t &configuration,
    const Decision &decision,const Evaluation &original_score,
    const MpcRtiNominal_t &original_nominal,
    ConstraintMeritWeights penalties,
    ReducedSourceGradient *gradient,
    SparseNonlinearProgram *physical_program);
} // namespace sdu_apex::source_nlp
