/* Shared MPC types, source-model constants and trajectory references. */

#ifndef MPC_TYPES_H
#define MPC_TYPES_H

/*===========================================================================
 * Defines
 *===========================================================================*/

/* Math and timing */
#define PREDICTION_DT_SECONDS 0.025f                    /* Nominal 40 Hz prediction-step duration. */

/* MPC defaults */
#ifndef PREDICTION_HORIZON
#define PREDICTION_HORIZON 30                           /* Stages of PREDICTION_DT_SECONDS; set by SDU_APEX_MPC_HORIZON. */
#endif
/* Prediction and optimization use the same 30 transitions from the command
 * application epoch; there is no extra in-horizon queued-command stage. */
#define TIME_STEP_SECONDS 0.025f                         /* Default model integration period per horizon stage. */

/* Solver and model safeguards */

/* Current Unity source anchors and controller output policy. */
#define SOURCE_MAX_STEERING_RAD 0.5235987756f            /* VehicleController SteeringLimit: 30 deg. */
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

/* Unity command-wrapper geometry and MPC output constraints. */
typedef struct
{
    float steering_wheelbase_m;           /* Unity VehicleController geometry. */
    float max_steering_angle;             /* Unity steering limit. */
    float maximum_command_speed_mps;      /* Project command ceiling. */
} VehicleParameters_t;

/* Tracking targets and corridor bounds for one horizon stage. */
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
