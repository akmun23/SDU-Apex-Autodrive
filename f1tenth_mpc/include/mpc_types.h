/**
 * @file mpc_types.h
 * @brief Type definitions and compile-time constants for the MPC system.
 * @details Defines all shared constants and data structures used by the MPC
 *          pipeline, including source model parameters and trajectory references.
 * @dependencies none
 */

#ifndef MPC_TYPES_H
#define MPC_TYPES_H

/*===========================================================================
 * Defines
 *===========================================================================*/

/* One source of truth for the production RTI state layout. The source plant's
 * u/v/yaw-rate and physical steering are represented once here; only its
 * wheel, sprung-body, and suspension states are additional hidden channels. */
enum
{
    MPC_STATE_IDX_EY = 0,
    MPC_STATE_IDX_EPSI = 1,
    MPC_STATE_IDX_U = 2,
    MPC_STATE_IDX_V = 3,
    MPC_STATE_IDX_R = 4,
    MPC_STATE_IDX_DELAYED_STEERING_COMMAND_1 = 5,
    MPC_STATE_IDX_DELAYED_STEERING_COMMAND_2 = 6,
    MPC_STATE_IDX_ACTUAL_STEERING_ANGLE = 7,
    MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_1 = 8,
    MPC_STATE_IDX_DELAYED_THROTTLE_COMMAND_2 = 9,
    MPC_STATE_IDX_WHEEL_OMEGA_FL = 10,
    MPC_STATE_IDX_WHEEL_OMEGA_FR = 11,
    MPC_STATE_IDX_WHEEL_OMEGA_RL = 12,
    MPC_STATE_IDX_WHEEL_OMEGA_RR = 13,
    MPC_STATE_IDX_BODY_HEIGHT = 14,
    MPC_STATE_IDX_BODY_HEIGHT_RATE = 15,
    MPC_STATE_IDX_ROLL = 16,
    MPC_STATE_IDX_ROLL_RATE = 17,
    MPC_STATE_IDX_PITCH = 18,
    MPC_STATE_IDX_PITCH_RATE = 19,
    MPC_STATE_IDX_SUSPENSION_JOUNCE_FL = 20,
    MPC_STATE_IDX_SUSPENSION_JOUNCE_FR = 21,
    MPC_STATE_IDX_SUSPENSION_JOUNCE_RL = 22,
    MPC_STATE_IDX_SUSPENSION_JOUNCE_RR = 23,
    MPC_STATE_COUNT = 24
};

#define MPC_MODEL_NX MPC_STATE_COUNT
#define MPC_MODEL_NU 2
/* Compact RTI state layout. The four logged actuator samples remain in the
 * nonlinear plant representation, but practice request/response timing shows
 * only the newest sample is in flight at a sensor epoch. The older sample is
 * history. The QP carries only the 20 path/body/plant coordinates. */
enum
{
    MPC_OPT_IDX_EY = 0,
    MPC_OPT_IDX_EPSI = 1,
    MPC_OPT_IDX_U = 2,
    MPC_OPT_IDX_V = 3,
    MPC_OPT_IDX_R = 4,
    MPC_OPT_IDX_ACTUAL_STEERING_ANGLE = 5,
    MPC_OPT_IDX_WHEEL_OMEGA_FL = 6,
    MPC_OPT_IDX_WHEEL_OMEGA_FR = 7,
    MPC_OPT_IDX_WHEEL_OMEGA_RL = 8,
    MPC_OPT_IDX_WHEEL_OMEGA_RR = 9,
    MPC_OPT_IDX_BODY_HEIGHT = 10,
    MPC_OPT_IDX_BODY_HEIGHT_RATE = 11,
    MPC_OPT_IDX_ROLL = 12,
    MPC_OPT_IDX_ROLL_RATE = 13,
    MPC_OPT_IDX_PITCH = 14,
    MPC_OPT_IDX_PITCH_RATE = 15,
    MPC_OPT_IDX_SUSPENSION_JOUNCE_FL = 16,
    MPC_OPT_IDX_SUSPENSION_JOUNCE_FR = 17,
    MPC_OPT_IDX_SUSPENSION_JOUNCE_RL = 18,
    MPC_OPT_IDX_SUSPENSION_JOUNCE_RR = 19,
    MPC_OPT_PHYSICAL_NX = 20,
    /* Deterministic displacement from nominal progress. Without this scalar,
     * speed corrections disappear from later path-frame linearizations. */
    MPC_OPT_IDX_PROGRESS_DEVIATION = 20,
    MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND = 21,
    MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND = 22,
    MPC_OPTIMIZER_NX = 23
};
/* command-time prediction advances the measured state to the next bridge
 * request epoch before solving. The newly optimized input is active at that
 * boundary, so the MPC horizon starts with a direct-input plant transition. */
#define MPC_COMMAND_DELAY_STEPS 0

/* Math and timing */
#define PREDICTION_DT_SECONDS 0.025f                    /* Nominal 40 Hz prediction-step duration. */

/* Other swept MPC defaults */
#define PREDICTION_HORIZON 30
/* Equal-duration, 75 ms exact validation for rejected-plan proposals and
 * retained actions. Independent of in-horizon delay: command-time prediction
 * has already handled the external actuator queue before this horizon. */
/* Prediction and optimization use the same 30 transitions from the command
 * application epoch; there is no extra in-horizon queued-command stage. */
#define TIME_STEP_SECONDS 0.025f                         /* Default model integration period per horizon stage. */

/* Solver and model safeguards */

/* Current Unity source anchors and controller output policy. */
#define SOURCE_MAX_STEERING_RAD 0.5235987756f            /* VehicleController SteeringLimit: 30 deg. */
#define SOURCE_STEERING_RATE_RADPS 3.2f                  /* Prefab SteeringRate: 183.346 deg/s. */
#define SOURCE_STEERING_WHEELBASE_M 0.324f               /* VehicleController Wheelbase: 324 mm. */
#define MPC_MAX_COMMAND_SPEED_MPS 16.0f                  /* Project command envelope, not simulator physics. */
#define MPC_THROTTLE_MIN 0.0f
#define MPC_THROTTLE_MAX 1.0f
/* The bridge rounds normalized actuator commands to 0.001. Unity's controller
 * treats every nonzero positive throttle as motor torque, so 0.001 is the
 * smallest unambiguous positive drive command; exactly zero remains its
 * separate full-brake mode. */
#define MPC_THROTTLE_MIN_DRIVE 0.001f
/* The 1 ms hard-brake plant sweep bottoms out at about -7.3 mm/s before
 * settling to rest; retain this measured numerical stopping transient. */
#define MPC_PHYSX_BRAKE_REVERSE_TRANSIENT_MPS 0.008f

/* AutoDRIVE vehicle footprint dimensions used for MPC corridor projection.
 * They are geometry only, not tire or vehicle-dynamics parameters. */
#define MPC_CAR_WIDTH_M 0.273f
#define MPC_REAR_OVERHANG_M 0.080f
#define MPC_REAR_AXLE_TO_FRONT_BUMPER_M 0.430f
/* Default physical clearance; the loaded runtime profile explicitly sets it. */
#define MPC_DEFAULT_WALL_CLEARANCE_M 0.050f

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

#endif /* MPC_TYPES_H */
