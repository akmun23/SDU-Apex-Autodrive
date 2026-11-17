#pragma once

// Shared validation and objective helpers for the source-derived MPC.
#include "mpc_cycle.h"
#include <cmath>

double monotonic_microseconds(void);
int finite_cycle_state(const MpcCycleState_t *state);
int finite_reference(const MpcCycleReference_t *reference);
int valid_configuration(const MpcSolverConfiguration_t *configuration);

struct MpcCycleReferenceSamples {
    struct Bounds { float left_bound, right_bound; };
    Bounds center;
    Bounds endpoints[2];
};

struct MpcCycleStageReference {
    MpcCycleReference_t reference;
    MpcCycleReferenceSamples samples;
};

int mpc_cycle_sample_stage_reference(
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, double progress,
    const MpcSolverConfiguration_t *configuration, MpcCycleStageReference *stage);

// References include the candidate's speed profile and yaw-rate alignment.
// Corridor geometry uses the same samples and float slacks as the scorer.
int mpc_cycle_score_candidate_with_references(
    const MpcCycleNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcCycleConfiguration_t *configuration,
    const MpcCycleStageReference stages[PREDICTION_HORIZON + 1],
    double *objective, float *minimum_corridor_slack,
    int allow_hard_constraint_violations_for_metrics,
    float (*source_corridor_by_stage)[2], MpcCycleGripMetrics_t *grip_metrics,
    MpcCycleSourceCorridorCorners_t *corners);

int minimum_candidate_corridor_slack(
    const MpcCycleNominal_t *candidate,
    const MpcTrajectorySample_t *trajectory, size_t trajectory_count,
    double lap_length, const MpcSolverConfiguration_t *configuration,
    float *minimum_slack);

float corridor_margin_at_prediction(
    const MpcSolverConfiguration_t *configuration,int prediction_index);
float corridor_tolerance_at_prediction(
    const MpcSolverConfiguration_t *configuration,int prediction_index);
float overspeed_weight_for_reference(
    const MpcSolverConfiguration_t *configuration,
    const MpcCycleReference_t *reference);
float lateral_weight_for_reference(
    const MpcSolverConfiguration_t *configuration,
    const MpcCycleReference_t *reference);

float heading_weight_for_reference(
    const MpcSolverConfiguration_t *configuration,
    const MpcCycleReference_t *reference);

float underspeed_weight_for_reference(
    const MpcSolverConfiguration_t *configuration,
    const MpcCycleReference_t *reference);
float steering_reference_increment(
    const MpcCycleReference_t *current,const MpcCycleReference_t *next);
