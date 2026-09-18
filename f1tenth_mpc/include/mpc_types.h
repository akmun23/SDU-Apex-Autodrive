/**
 * @file mpc_types.h
 * @brief Type definitions and compile-time constants for the MPC system.
 * @details Defines all shared constants and data structures used by the MPC
 *          pipeline, including vehicle state, control inputs, physical
 *          parameters, solver configuration, trajectory references, and
 *          solver outputs.
 * @dependencies util_math.h, <stdint.h>
 */

#ifndef MPC_TYPES_H
#define MPC_TYPES_H

#include "util_math.h"
#include <stdint.h>

/*===========================================================================
 * Defines
 *===========================================================================*/

/* Global dimensions */
#define NX_GLOBAL 7                                      /* Global/body state plus commanded speed target. */
#define NX_FRENET 6                                      /* Frenet state plus actuator speed setpoint. */
#define NX_AUG 10                                        /* Six vehicle states, two steering states, two previous-input states. */
#define IDX_EY 0                                         /* Position of lateral error (ey) in the augmented vector. */
#define IDX_EPSI 1                                       /* Position of heading error in the augmented vector. */
#define IDX_LONG_VEL 2                                   /* Position of body longitudinal velocity in the augmented vector. */
#define IDX_LAT_VEL 3                                    /* Position of body lateral velocity in the augmented vector. */
#define IDX_YAW_RATE 4                                   /* Position of yaw rate in the augmented vector. */
#define IDX_TARGET_SPEED_STATE 5                         /* Actuator target carried through horizon stages. */
#define IDX_DELTA_COMMAND 6                              /* Position of commanded front-wheel steering angle. */
#define IDX_DELTA_EFFECTIVE 7                            /* Position of steering angle acting on the vehicle model. */
#define IDX_DRATE_PREV 8                                 /* Previous steering-rate state in the augmented vector. */
#define IDX_TARGET_SPEED_RATE_PREV 9                     /* Previous target-speed slew in the augmented vector. */
#define NU 2                                             /* Control vector width: steering-rate and target-speed slew. */
#define RICCATI_MAX_NX  10                               /* Maximum Riccati state dimension (augmented Frenet model). */
#define RICCATI_MAX_NU  2                                /* Maximum Riccati control dimension (steering-rate and speed slew). */

/* Math and timing */
#define TWO_PI (2.0 * M_PI)                              /* Full-angle constant used for heading wrap operations. */
#define CONTROL_RATE_HZ 40.0f                            /* Nominal dev-side source/control rate; runtime may provide measured dt. */
#define CONTROL_DT_SECONDS (1.0f / CONTROL_RATE_HZ)      /* Controller sample period derived from control-rate definition. */
#define PREDICTION_DT_SECONDS 0.025f                    /* Nominal 40 Hz prediction-step duration. */
#define CROSS_CALL_RATE_SCALE (CONTROL_DT_SECONDS / PREDICTION_DT_SECONDS) /* Normalizes first-step rate penalties across sample times. */
#define STEERING_EFFECTIVE_TIME_CONSTANT_SECONDS 0.025f  /* Recorded-data-selected zero-dead-time effective-steering pole. */

/* Default MPC objective weights */
#define WEIGHT_LAT_ERROR 1500.0f                         /* Penalizes lateral tracking deviation from the reference path. */
#define WEIGHT_HEADING 50.0f                            /* Penalizes heading misalignment relative to path tangent. */
#define WEIGHT_VELOCITY 200.0f                           /* Penalizes deviation from target longitudinal speed profile. */
#define WEIGHT_LAT_VEL 5.0f                             /* Penalizes lateral velocity to suppress side-slip growth. */
#define WEIGHT_YAW_RATE 1.5f                            /* Penalizes yaw-rate mismatch against reference curvature dynamics. */
#define WEIGHT_TARGET_SPEED_STATE 20.0f                  /* Tracks actuator target speed to the raceline profile. */
#define WEIGHT_COMMAND_STEERING 1.0f                     /* Tracks commanded steering to curvature feedforward. */
#define WEIGHT_STEER_EFFORT 2.0f                        /* Penalizes steering-rate effort to limit aggressive steering actuation. */
#define WEIGHT_TARGET_SPEED_RATE_EFFORT 0.5f           /* Penalizes target-speed slew magnitude. */
#define WEIGHT_STEER_RATE 5.0f                         /* Penalizes steering-rate change to reduce steering jerk. */
#define WEIGHT_TARGET_SPEED_RATE_CHANGE 5.0f           /* Penalizes target-speed slew change between calls. */
#define WEIGHT_EFFECTIVE_STEERING 1.0f                /* Penalizes effective-steering bias away from curvature feedforward. */

/* Other swept MPC defaults */
#define MAX_ITERATIONS 50                              /* Default solver iteration budget per control update. */
#define WALL_MARGIN 0.00f                                 /* Safety offset subtracted from both wall boundaries. */
#define ADMM_RHO 7.0f                                  /* Primary ADMM penalty balancing feasibility and optimality progress. */
#define ADMM_RHO_U 7.0f                                /* ADMM penalty applied to control-variable projection terms. */
#define CONVERGENCE_TOLERANCE 0.01f                     /* Residual threshold used to declare solver convergence. */
#define PREDICTION_HORIZON 30                            /* 30 commands at 40 Hz = 0.75 s. */
#define TIME_STEP_SECONDS 0.025f                         /* Default model integration period per horizon stage. */

/* Solver and model safeguards */
#define RICCATI_COST_FACTOR 2.0f                         /* Global scaling factor applied to stage and terminal costs. */
#define STEERING_RATE_LIMIT 3.2f                         /* Unity-measured steering-rate limit [rad/s]. */
#define STEERING_FEEDFORWARD_CLAMP_FACTOR 1.0f           /* Limits feedforward steering around linearization operating point. */
#define BIG_BOUND 50.0f                                  /* Sentinel magnitude representing an effectively unconstrained bound. */
#define MIN_LINEARIZATION_VELOCITY 0.5f                  /* Lower numeric velocity clamp for low-speed recovery. */
#define STABILITY_LIMIT 0.95f                            /* Clamp on selected discrete self-coupling to preserve numerical stability. */
/* CPU warm-start / cold-start policy. */
#define MPC_WS_CURVATURE_THRESH 0.25f                    /* Curvature jump that forces a cold start. */
#define MPC_WS_BOUND_THRESH 0.05f                        /* Slack on ey box before a stale warm start is treated as bound-incompatible. */
#define MPC_MODEL_SIGNATURE 6                            /* Identified yaw and speed response with carried setpoint. */

/* Default MPC configuration values */
#define TRAJECTORY_MAXIMUM_WAYPOINTS 4000                /* Maximum trajectory samples accepted by MPC reference buffers. */
#define TRAJECTORY_MAXIMUM_VELOCITY 16.0f                /* Project command ceiling accepted from trajectory input. */
#define MIN_TRAJECTORY_SPEED_MPS 0.5f                    /* Lower bound used when trajectory speed is missing or too small. */

/* Current Unity source anchors and controller output policy. */
#define SOURCE_MAX_STEERING_RAD 0.5235987756f            /* VehicleController SteeringLimit: 30 deg. */
#define SOURCE_STEERING_RATE_RADPS 3.2f                  /* Prefab SteeringRate: 183.346 deg/s. */
#define SOURCE_STEERING_WHEELBASE_M 0.324f               /* VehicleController Wheelbase: 324 mm. */
#define MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS 0.087735f /* Held-out AutoDRIVE yaw response fit, not a real-car tire constant. */
#define MPC_YAW_RATE_STEERING_GAIN_PER_M 3.011897f       /* Held-out gain per u*tan(delta), distinct from source wheelbase. */
#define MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2 (-0.37356440f) /* PP development-trace fit; truth used offline as target only. */
#define MPC_LONGITUDINAL_SPEED_COEFF_PER_S (-0.06389858f) /* Coefficient on predicted body speed. */
#define MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S 9.11426915f /* Coefficient on target minus body speed. */
#define MPC_LONGITUDINAL_TARGET_RATE_COEFF (-0.06687204f) /* Coefficient on commanded speed slew. */
#define MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2 6.0f           /* Recursive replay saturation, not a Unity force. */
#define MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2 5.36267417f /* 12/14 m/s fit; independent 15.3 m/s holdout. */
#define MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV 0.27655518f /* Measured full-brake envelope per current speed. */
#define MPC_MAX_COMMAND_SPEED_MPS 16.0f                  /* Project command envelope, not simulator physics. */
#define MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2 3.0f   /* Output target-speed increase policy. */
#define MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2 8.0f  /* Output target-speed reduction policy. */

/*===========================================================================
 * Structs
 *===========================================================================*/

/**
 * @brief Vehicle state in world/body coordinates.
 * @details Holds the global pose and body-frame velocities used by nonlinear
 *          propagation, simulation updates, and conversion to Frenet errors.
 * @param pos_x X position in world frame [meters].
 * @param pos_y Y position in world frame [meters].
 * @param heading Yaw angle relative to world X-axis [radians].
 * @param long_vel Longitudinal velocity in body frame [meters per second].
 * @param lat_vel Lateral velocity in body frame [meters per second].
 * @param yaw_rate Yaw rate [radians per second].
 */
typedef struct
{
    float pos_x;                /* X position in world frame [meters]. */
    float pos_y;                /* Y position in world frame [meters]. */
    float heading;              /* Yaw angle relative to world X-axis [radians]. */
    float long_vel;             /* Longitudinal velocity in body frame [meters per second]. */
    float lat_vel;              /* Lateral velocity in body frame [meters per second]. */
    float yaw_rate;             /* Yaw rate [radians per second]. */
    float target_speed_mps;     /* Actuator target carried through MPC prediction. */
} VehicleState_t;

/**
 * @brief Frenet-frame state relative to the active reference path.
 * @details Stores path-relative tracking errors and body-frame dynamics in the
 *          coordinate system consumed by the MPC optimizer.
 * @param flat_error Lateral displacement from path centerline [meters].
 * @param fhead_error Heading error relative to path tangent [radians].
 * @param flong_vel Longitudinal velocity in vehicle body frame [meters per second].
 * @param flat_vel Lateral velocity in vehicle body frame [meters per second].
 * @param fyaw_rate Vehicle yaw rate [radians per second].
 */
typedef struct
{
    float flat_error;           /* Lateral error [meters]. */
    float fhead_error;          /* Heading error [radians]. */
    float flong_vel;            /* Longitudinal velocity [meters per second]. */
    float flat_vel;             /* Lateral velocity [meters per second]. */
    float fyaw_rate;            /* Yaw rate [radians per second]. */
    float ftarget_speed_mps;    /* Actuator target speed [meters per second]. */
} FrenetState_t;

/**
 * @brief Control command issued to the vehicle model/controller.
 * @details Represents the command pair produced by MPC.  The longitudinal
 *          field is converted to an Ackermann target-speed update by the ROS
 *          adapter; it is not a direct Unity acceleration actuator.
 * @param steer_ang Front wheel steering angle command [radians].
 * @param target_speed_rate Requested target-speed change [m/s^2].
 */
typedef struct
{
    float steer_ang;            /* Front wheel steering angle [radians]. */
    float target_speed_rate;      /* Requested target-speed change [m/s^2]. */
} ControlInput_t;

/**
 * @brief One raceline waypoint sample used by ROS nodes when loading CSV tracks.
 * @details Stores geometric, kinematic, and corridor-bound data in double
 *          precision for interpolation and closest-point queries.
 * @param s_meters Arc length along the reference path [meters].
 * @param x_meters Global X position in map/world frame [meters].
 * @param y_meters Global Y position in map/world frame [meters].
 * @param heading_radians Path tangent heading at this waypoint [radians].
 * @param velocity_meters_per_second Reference speed at this waypoint [m/s].
 * @param curvature_radians_per_meter Path curvature at this waypoint [rad/m].
 * @param left_bound_meters Left corridor bound from centerline [meters].
 * @param right_bound_meters Right corridor bound from centerline [meters].
 * @param sin_heading Cached sin(heading_radians) for fast projections.
 * @param cos_heading Cached cos(heading_radians) for fast projections.
 */
typedef struct
{
    double s_meters;
    double x_meters;
    double y_meters;
    double heading_radians;
    double velocity_meters_per_second;
    double curvature_radians_per_meter;
    double left_bound_meters;
    double right_bound_meters;
    double sin_heading;
    double cos_heading;
} TrajectoryWaypoint_t;

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
    float maximum_target_speed_rate_increase_mps2; /* Controller output policy. */
    float maximum_target_speed_rate_reduction_mps2;/* Controller output policy. */
} VehicleParameters_t;

/**
 * @brief MPC solver configuration and objective policy parameters.
 * @details Collects horizon timing, objective weights, and solver stopping
 *          settings that define one complete MPC tuning profile.
 * @param prediction_horizon_steps Number of prediction stages in the horizon.
 * @param time_step Prediction stage duration [seconds].
 * @param weight_lateral_error Weight on lateral tracking error.
 * @param weight_heading_error Weight on heading tracking error.
 * @param weight_velocity Weight on longitudinal velocity tracking error.
 * @param weight_lateral_velocity Weight on lateral velocity tracking error.
 * @param weight_yaw_rate Weight on yaw-rate tracking error.
 * @param weight_target_speed_state Weight on actuator target-speed tracking.
 * @param weight_commanded_steering Weight on steering-command tracking.
 * @param weight_steering_effort Weight on steering effort command magnitude.
 * @param weight_target_speed_rate_effort Weight on target-speed slew magnitude.
 * @param weight_steering_rate Weight on steering-rate change.
 * @param weight_target_speed_rate_change Weight on target-speed slew change.
 * @param weight_effective_steering Weight on effective-steering centering term.
 * @param cross_call_rate_scale Scale factor for first-step cross-call penalties.
 * @param wall_margin Effective lateral wall margin [meters] used in e_y constraints.
 * @param max_solver_iterations Maximum QP/ADMM iterations per solve.
 * @param solver_convergence_tolerance Residual tolerance for solve termination.
 */
typedef struct
{
    float time_step;                     /* Prediction step duration [seconds]. */
    float weight_lateral_error;          /* Weight for lateral error tracking. */
    float weight_heading_error;          /* Weight for heading error tracking. */
    float weight_velocity;               /* Weight for longitudinal velocity tracking error. */
    float weight_lateral_velocity;       /* Weight for lateral velocity tracking error. */
    float weight_yaw_rate;               /* Weight for yaw-rate tracking error. */
    float weight_target_speed_state;     /* Weight for actuator target-speed state. */
    float weight_commanded_steering;     /* Weight for commanded steering angle. */
    float weight_steering_effort;        /* Weight for steering effort. */
    float weight_target_speed_rate_effort;    /* Weight for target-speed slew. */
    float weight_steering_rate;          /* Weight for steering jerk/rate change. */
    float weight_target_speed_rate_change;      /* Weight for target-speed slew change. */
    float weight_effective_steering;     /* Weight for effective-steering centering. */
    float cross_call_rate_scale;         /* First-step cross-call rate penalty scale factor. */
    float wall_margin;                   /* Effective wall margin for lateral corridor bounds [m]. */

    uint16_t max_solver_iterations;      /* Maximum QP solver iterations. */
    float solver_convergence_tolerance;  /* Solver convergence tolerance. */
} MpcConfiguration_t;

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

/** 
 * @brief Enumeration of possible MPC solver termination statuses.
 * @details Used to communicate solver outcomes to higher-level logic for
 * @param MPC_STATUS_SUCCESS Optimal solution found.
 * @param MPC_STATUS_MAXIMUM_ITERATIONS_REACHED Solver reached max iterations without convergence.
 * @param MPC_STATUS_INFEASIBLE No feasible solution exists for the given constraints.
*/
typedef enum
{
    MPC_STATUS_SUCCESS = 0,                    /* Optimal solution found. */
    MPC_STATUS_MAXIMUM_ITERATIONS_REACHED = 1, /* Solver reached max iterations. */
    MPC_STATUS_INFEASIBLE = 2,                 /* No feasible solution for constraints. */
    MPC_STATUS_ERROR = 3                       /* Solver error or invalid input. */
} MpcSolverStatus_t;

/**
 * @brief MPC solve output for one control cycle.
 * @details Captures solver termination metadata and the first control action
 *          applied to the plant for the current control tick.
 * @param solver_status Solver termination state.
 * @param optimal_control First-step optimal control command.
 * @param iterations_used Number of iterations consumed by the solver.
 * @param primal_residual Final primal residual metric reported by the solver.
 * @param dual_residual Final dual residual from ADMM convergence checks.
 */
typedef struct
{
    MpcSolverStatus_t solver_status;  /* Solver termination status. */
    ControlInput_t optimal_control;   /* First-step optimal control command. */
    uint16_t iterations_used;         /* Number of solver iterations used. */
    float primal_residual;            /* Final solver primal residual metric. */
    float dual_residual;              /* Final dual residual (ADMM metric). */
} MpcSolverResult_t;


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
    int shared_rho;
    int use_prefactorization;
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
