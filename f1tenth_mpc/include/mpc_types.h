/**
 * @file mpc_types.h
 * @brief Type definitions and compile-time constants for the MPC system.
 * @details Defines all shared constants and data structures used by the MPC
 *          pipeline, including model parameters, trajectory references, and
 *          Riccati-ADMM solver data.
 * @dependencies none
 */

#ifndef MPC_TYPES_H
#define MPC_TYPES_H

/*===========================================================================
 * Defines
 *===========================================================================*/

/* Riccati dimensions. The RTI state contains the 26-coordinate tracking and
 * Unity-derived plant state plus the two prior-input memory coordinates. */
#define RICCATI_MAX_NX  28                               /* Full Unity-derived plant, tracking, and prior-input states. */
#define RICCATI_MAX_NU  2                                /* Steering-rate and normalized throttle. */

/* Math and timing */
#define PREDICTION_DT_SECONDS 0.025f                    /* Nominal 40 Hz prediction-step duration. */

/* Other swept MPC defaults */
#define MAX_ITERATIONS 200                             /* Default solver iteration budget per control update. */
#define ADMM_RHO 7.0f                                  /* Primary ADMM penalty balancing feasibility and optimality progress. */
#define ADMM_RHO_U 7.0f                                /* ADMM penalty applied to control-variable projection terms. */
#define CONVERGENCE_TOLERANCE 0.01f                     /* Residual threshold used to declare solver convergence. */
#define PREDICTION_HORIZON 30                            /* 30 commands at 40 Hz = 0.75 s. */
#define TIME_STEP_SECONDS 0.025f                         /* Default model integration period per horizon stage. */
#define MPC_MODEL_NX 26                                  /* Tracking, actuator queues, and all 18 Unity source-plant states without duplicated body channels. */

/* Solver and model safeguards */
#define BIG_BOUND 50.0f                                  /* Sentinel magnitude representing an effectively unconstrained bound. */

/* Current Unity source anchors and controller output policy. */
#define SOURCE_MAX_STEERING_RAD 0.5235987756f            /* VehicleController SteeringLimit: 30 deg. */
#define SOURCE_STEERING_RATE_RADPS 3.2f                  /* Prefab SteeringRate: 183.346 deg/s. */
#define SOURCE_STEERING_WHEELBASE_M 0.324f               /* VehicleController Wheelbase: 324 mm. */
#define MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS 0.015f /* Practice-bag system identification: held-out 40 Hz yaw-rate steps across independent laps. */
#define MPC_YAW_RATE_STEERING_GAIN_PER_M 2.95f          /* Held-out yaw-rate gain per u*tan(delta), distinct from source wheelbase. */
#define MPC_MAX_COMMAND_SPEED_MPS 16.0f                  /* Project command envelope, not simulator physics. */
#define MPC_THROTTLE_MIN 0.0f
#define MPC_THROTTLE_MAX 1.0f

/* AutoDRIVE vehicle footprint dimensions used for MPC corridor projection.
 * They are geometry only, not tire or vehicle-dynamics parameters. */
#define MPC_CAR_WIDTH_M 0.273f
#define MPC_REAR_OVERHANG_M 0.080f
#define MPC_REAR_AXLE_TO_FRONT_BUMPER_M 0.430f
/* Hard minimum clearance between the projected vehicle footprint and wall. */
#define MPC_REQUIRED_WALL_CLEARANCE_M 0.150f

/*===========================================================================
 * Structs
 *===========================================================================*/

/**
 * @brief Current-source anchors and MPC output constraints.
 * @details These are only the Unity command-wrapper quantities exercised by
 *          the current source-command baseline.
 */
typedef struct
{
    float steering_wheelbase_m;           /* Unity VehicleController geometry. */
    float max_steering_angle;             /* Unity steering limit. */
    float steering_rate_radps;            /* Unity physical steering rate. */
    float maximum_command_speed_mps;      /* Project command ceiling. */
} VehicleParameters_t;

/**
 * @brief One trajectory reference sample for a single horizon stage.
 * @details Defines the tracking targets and corridor bounds consumed by each
 *          MPC prediction step.
 * @param reference_lateral_error Target lateral error [meters].
 * @param reference_heading_error Target heading error [radians].
 * @param reference_velocity Target longitudinal velocity [meters per second].
 * @param reference_lateral_velocity Target lateral velocity [meters per second].
 * @param reference_yaw_rate Target yaw rate [radians per second].
 * @param path_curvature Reference path curvature [radians per meter].
 * @param left_wall_bound Maximum leftward allowed offset from centerline [meters].
 * @param right_wall_bound Maximum rightward allowed offset from centerline [meters].
 */
typedef struct
{
    float reference_lateral_error;       /* Reference lateral error [meters]. */
    float reference_heading_error;       /* Reference heading error [radians]. */
    float reference_velocity;            /* Target longitudinal velocity [meters per second]. */
    float reference_lateral_velocity;    /* Target lateral velocity [meters per second]. */
    float reference_yaw_rate;            /* Target yaw rate [radians per second]. */
    float path_curvature;                /* Path curvature [radians per meter]. */
    float left_wall_bound;               /* Maximum leftward bound from path centerline [meters]. */
    float right_wall_bound;              /* Maximum rightward bound from path centerline [meters]. */
} TrajectoryReferencePoint_t;

/*===========================================================================
 * Riccati-ADMM Solver Types
 *===========================================================================*/

/**
 * @brief Per-stage dynamics, cost, and box-constraint data for Riccati-ADMM.
 * @details Represents one prediction step used by the constrained LQR solve.
 *          Q and R are diagonal weights, N is an optional cross-cost term,
 *          and x/u bounds define box constraints for ADMM projection.
 */
typedef struct
{
    float A[RICCATI_MAX_NX][RICCATI_MAX_NX];
    float B[RICCATI_MAX_NX][RICCATI_MAX_NU];
    float d[RICCATI_MAX_NX];
    float Q_diag[RICCATI_MAX_NX];
    float q[RICCATI_MAX_NX];
    float R_diag[RICCATI_MAX_NU];
    float r[RICCATI_MAX_NU];
    float N[RICCATI_MAX_NX][RICCATI_MAX_NU];
    float x_lb[RICCATI_MAX_NX];
    float x_ub[RICCATI_MAX_NX];
    float u_lb[RICCATI_MAX_NU];
    float u_ub[RICCATI_MAX_NU];
} RiccatiStepData_t;

/**
 * @brief ADMM configuration parameters for the Riccati solver.
 * @details Controls augmented-Lagrangian penalties, convergence criteria,
 *          iteration budget, and optional adaptive-rho logic. `tolerance` is
 *          the absolute maximum allowed raw primal and dual residual; it is
 *          not scaled by unrelated state magnitudes because MPC channels have
 *          different physical units.
 */
typedef struct
{
    float rho;
    float rho_u;
    float tolerance;
    int max_iterations;
    int adaptive_rho;
    /* Re-factor only every Nth ADMM iteration when adapting rho; zero keeps
     * the historical per-iteration update cadence. */
    int adaptive_rho_update_period;
    int shared_rho;
    /* ADMM over-relaxation in [1, 2]; zero selects the unrelaxed baseline. */
    float over_relaxation;
    int use_prefactorization;
    /* Cooperative wall-time bound for one ADMM solve. Zero disables the
     * bound. The solver checks after each complete primal/projection pass. */
    double max_solve_time_us;
    int use_scaling;
    float state_scale[RICCATI_MAX_NX];
    float input_scale[RICCATI_MAX_NU];
} RiccatiAdmmConfig_t;

/**
 * @brief Termination status codes returned by Riccati-ADMM.
 * @details Distinguishes converged solves, iteration-limit exits, and hard
 *          input/runtime errors.
 */
typedef enum
{
    RICCATI_STATUS_OPTIMAL = 0,
    RICCATI_STATUS_MAX_ITERATIONS = 1,
    RICCATI_STATUS_ERROR = 2
} RiccatiStatus_t;

/**
 * @brief Output trajectories and convergence metrics from Riccati-ADMM.
 * @details Stores the solved state/control horizon together with residuals,
 *          iteration count, and final status.
 */
typedef struct
{
    float x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX];
    float u[PREDICTION_HORIZON][RICCATI_MAX_NU];
    int iterations;
    float primal_residual;
    float dual_residual;
    RiccatiStatus_t status;
} RiccatiSolution_t;

/**
 * @brief Persistent ADMM warm-start buffers used across solve calls.
 * @details Retains projected variables, dual variables, and adapted penalty
 *          parameters so sequential MPC solves can converge faster.
 */
typedef struct
{
    float z_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX];
    float z_u[PREDICTION_HORIZON][RICCATI_MAX_NU];
    float y_x[PREDICTION_HORIZON + 1][RICCATI_MAX_NX];
    float y_u[PREDICTION_HORIZON][RICCATI_MAX_NU];
    float rho;
    float rho_u;
    int nx;
    int nu;
    int horizon;
    int initialized;
    int scaling_enabled;
    float state_scale[RICCATI_MAX_NX];
    float input_scale[RICCATI_MAX_NU];
} RiccatiAdmmState_t;

#endif /* MPC_TYPES_H */
