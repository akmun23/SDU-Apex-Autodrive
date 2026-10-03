"""Canonical units, frames, and timing rules for offline vehicle-plant work.

This module describes existing recorded signals; it does not change runtime
odometry or simulator physics.  A wheel-rotation measurement is never equated
with tire contact speed unless it is explicitly called a kinematic proxy.
"""

from __future__ import annotations

import math
from typing import Any


DT_S = 0.025
WHEEL_RADIUS_M = 0.059
REAR_TRACK_WIDTH_M = 0.236
REAR_AXLE_TO_COM_X_M = 0.15532
ENCODER_SOURCE_DT_GATE_S = (0.015, 0.035)
ENCODER_ALIGNMENT_LIMIT_S = 0.030


SIGNAL_DICTIONARY: dict[str, dict[str, Any]] = {
    "body_u_rear": {
        "name": "rear-axle longitudinal body velocity", "units": "m/s",
        "source": "/autodrive/roboracer_1/odom.twist.twist.linear.x; frames[:,0]",
        "frame_reference": "vehicle body axes, rear-axle/base_link point",
        "observation": "direct bridge odometry target; simulator-derived in Explore",
        "formula": "copied COM u; longitudinal component unchanged by x-axis point shift",
        "causal_window": "one source packet", "nominal_delay": "packet transport/alignment; not identified as sensor group delay",
        "validity": "finite odometry twist and matched packet; fixed 25 ms dataset steps",
        "training_use": "supervised state/transition target; offline truth only",
        "rollout_use": "predicted recursively; no future measurement",
        "role": "physical-state candidate and training label",
    },
    "body_v_rear": {
        "name": "rear-axle lateral body velocity", "units": "m/s",
        "source": "/autodrive/roboracer_1/odom.twist.twist.linear.y; frames[:,1]",
        "frame_reference": "vehicle body axes, rear-axle/base_link point",
        "observation": "derived from copied COM odometry velocity",
        "formula": "v_rear = v_com - yaw_rate * 0.15532 m",
        "causal_window": "one source packet", "nominal_delay": "packet transport/alignment; not identified as sensor group delay",
        "validity": "finite twist and yaw rate; source frame convention checked in body-dynamics loader",
        "training_use": "supervised state/transition target; offline truth only",
        "rollout_use": "predicted recursively; no future measurement",
        "role": "physical-state candidate and training label",
    },
    "yaw_rate": {
        "name": "body yaw rate", "units": "rad/s",
        "source": "/autodrive/roboracer_1/odom.twist.twist.angular.z; frames[:,2]",
        "frame_reference": "vehicle body z axis; same rigid-body rate at COM/rear axle",
        "observation": "direct bridge odometry / aligned simulator angular-velocity label",
        "formula": "no point-shift term for angular velocity",
        "causal_window": "one source packet", "nominal_delay": "packet transport/alignment; not separately measured",
        "validity": "finite and source-packet aligned",
        "training_use": "supervised transition target",
        "rollout_use": "predicted recursively",
        "role": "physical-state candidate and training label",
    },
    "rear_axle_world_pose": {
        "name": "rear-axle world pose", "units": "m, m, rad",
        "source": "/autodrive/roboracer_1/odom.pose.pose; dataset odom_pose_xyyaw",
        "frame_reference": "world pose at base_link/rear axle",
        "observation": "direct bridge odometry pose; simulator pose kept as a separate truth label",
        "formula": "world x/y/yaw; yaw differences must be wrapped",
        "causal_window": "one source packet", "nominal_delay": "packet transport/alignment; not separately measured",
        "validity": "finite pose; run/reset boundaries preserved",
        "training_use": "offline trajectory label/diagnostic",
        "rollout_use": "integrated from predicted body motion; never a future input",
        "role": "pose target/diagnostic",
    },
    "com_world_pose": {
        "name": "COM world pose", "units": "m, m, rad",
        "source": "bridge packet JSON on /autodrive/roboracer_1/bridge_packet_timing; simulator_rigid_state / simulator_pose_xyyaw",
        "frame_reference": "simulator world, rigid-body COM",
        "observation": "simulator truth carried by debug packet; offline label only",
        "formula": "rear-to-COM body offset is +0.15532 m; rotate by world yaw for pose conversion",
        "causal_window": "same simulator packet", "nominal_delay": "same source packet after offline alignment",
        "validity": "packet source stamp and packet sequence agree; truth channel may be absent",
        "training_use": "offline supervision/diagnostic only",
        "rollout_use": "must be propagated from predicted state if modeled; no future truth",
        "role": "label only",
    },
    "steering_command": {
        "name": "steering command", "units": "rad",
        "source": "/autodrive/roboracer_1/steering_command; frames[:,7]",
        "frame_reference": "front steering command convention",
        "observation": "direct command",
        "formula": "none",
        "causal_window": "sample-and-hold command history", "nominal_delay": "command-to-feedback delay is a fitted model quantity, not assumed here",
        "validity": "finite; paired to packet through existing causal capture alignment",
        "training_use": "past history and known future exogenous command sequence",
        "rollout_use": "future commanded input allowed; future feedback not allowed",
        "role": "control input",
    },
    "throttle_command": {
        "name": "throttle/brake command", "units": "normalized command",
        "source": "/autodrive/roboracer_1/throttle_command; frames[:,8]",
        "frame_reference": "actuator command convention used by the sim bridge",
        "observation": "direct command; sign semantics must follow recorded interface, not a presumed coast category",
        "formula": "none",
        "causal_window": "sample-and-hold command history", "nominal_delay": "command-to-feedback delay is a fitted model quantity, not assumed here",
        "validity": "finite; paired to packet through existing causal capture alignment",
        "training_use": "past history and known future exogenous command sequence",
        "rollout_use": "future commanded input allowed; future feedback not allowed",
        "role": "control input",
    },
    "steering_feedback": {
        "name": "measured steering feedback", "units": "rad",
        "source": "/autodrive/roboracer_1/steering; frames[:,3]",
        "frame_reference": "reported steering actuator angle",
        "observation": "direct feedback",
        "formula": "none",
        "causal_window": "single feedback sample plus model history", "nominal_delay": "alignment tolerance <=30 ms; physical actuator lag is dynamic",
        "validity": "finite and feedback history marked valid",
        "training_use": "past measured state; future feedback only in named oracle intervention",
        "rollout_use": "predicted actuator state in normal free run",
        "role": "actuator state / measurement",
    },
    "throttle_feedback": {
        "name": "measured throttle feedback", "units": "normalized command",
        "source": "/autodrive/roboracer_1/throttle; frames[:,4]",
        "frame_reference": "reported throttle actuator feedback",
        "observation": "direct feedback",
        "formula": "none",
        "causal_window": "single feedback sample plus model history", "nominal_delay": "alignment tolerance <=30 ms; physical actuator lag is dynamic",
        "validity": "finite and feedback history marked valid",
        "training_use": "past measured state; future feedback only in named oracle intervention",
        "rollout_use": "predicted actuator state in normal free run",
        "role": "actuator state / measurement",
    },
    "encoder_left_angle": {
        "name": "left cumulative rear encoder angle", "units": "rad",
        "source": "/autodrive/roboracer_1/left_encoder.position[0]",
        "frame_reference": "left rear wheel rotation; cumulative angle",
        "observation": "direct sensor packet",
        "formula": "not differenced until a named wheel-rate view is formed",
        "causal_window": "raw cumulative sample", "nominal_delay": "source-stamp cadence is treated as 40 Hz; stamp jitter is alignment metadata",
        "validity": "strictly increasing source stamps, 15–35 ms interval gate, same reset epoch, causal sample no later than query time",
        "training_use": "offline target construction and sensor history",
        "rollout_use": "no future angle; recursive predicted wheel or latent state instead",
        "role": "measurement",
    },
    "encoder_right_angle": {
        "name": "right cumulative rear encoder angle", "units": "rad",
        "source": "/autodrive/roboracer_1/right_encoder.position[0]",
        "frame_reference": "right rear wheel rotation; cumulative angle",
        "observation": "direct sensor packet",
        "formula": "not differenced until a named wheel-rate view is formed",
        "causal_window": "raw cumulative sample", "nominal_delay": "source-stamp cadence is treated as 40 Hz; stamp jitter is alignment metadata",
        "validity": "strictly increasing source stamps, 15–35 ms interval gate, same reset epoch, causal sample no later than query time",
        "training_use": "offline target construction and sensor history",
        "rollout_use": "no future angle; recursive predicted wheel or latent state instead",
        "role": "measurement",
    },
    "encoder_rate_25ms_fixed": {
        "name": "fixed-40-Hz encoder surface-rate estimate", "units": "m/s",
        "source": "causal adjacent cumulative encoder angles; fixed sidecar encoder_raw_surface_mps",
        "frame_reference": "left/right wheel rotation projected to tire radius",
        "observation": "derived measurement",
        "formula": "(angle[k]-angle[k-1])*0.059/0.025; source interval only gates 15–35 ms",
        "causal_window": "one 25 ms encoder interval", "nominal_delay": "sample endpoint plus one-period backward difference; not a centered filter",
        "validity": "causal aligned pair, valid source interval, no reset/gap, same left/right source packet",
        "training_use": "fixed40 sidecar target; previous raw sidecar is a separate source-stamp-divided view",
        "rollout_use": "not an observed future input; use predicted wheel state if causal experiment supports it",
        "role": "derived measurement/target",
    },
    "encoder_rate_variable_stamp_diagnostic": {
        "name": "variable-source-stamp diagnostic encoder rate", "units": "m/s",
        "source": "same adjacent causal encoder angles and source stamps",
        "frame_reference": "left/right wheel rotation projected to tire radius",
        "observation": "diagnostic alternate calculation, not canonical physical cadence",
        "formula": "(angle[k]-angle[k-1])*0.059/(stamp[k]-stamp[k-1])",
        "causal_window": "one observed encoder interval", "nominal_delay": "sample endpoint",
        "validity": "same source-pair gate as fixed rate; must be labeled stamp-divided",
        "training_use": "diagnostic comparison only",
        "rollout_use": "none",
        "role": "diagnostic only",
    },
    "encoder_rate_100ms_stored": {
        "name": "stored approximately-100-ms encoder surface-speed proxy", "units": "m/s",
        "source": "frames[:,5:7] / sensor_frames[:,2:4]",
        "frame_reference": "left/right rear wheel rotation projected to tire radius",
        "observation": "existing causal filtered encoder feature; not body speed",
        "formula": "cumulative angle difference across four 25 ms packets * radius / elapsed source time; existing data accepts 75–125 ms elapsed",
        "causal_window": "approximately 100 ms trailing window", "nominal_delay": "trailing window effective delay about half the window; exact filter group delay not identified",
        "validity": "valid encoder history within one run/reset epoch and established dataset mask",
        "training_use": "historical model state/measurement feature; retain its name and timing definition",
        "rollout_use": "predicted only in normal recursive plant; measured future values are oracle diagnostics",
        "role": "filtered measurement / candidate internal state under test",
    },
    "production_wheel_quantities": {
        "name": "production odometry wheel outputs", "units": "m/s",
        "source": "production SensorPacketAssembler + OdometryObserver outputs: wheel_raw_mps, wheel_mapped_mps, wheel_packet_mps",
        "frame_reference": "mean rear-wheel forward surface speed, magnitude-valued in production observer",
        "observation": "replayed offline from recorded left/right encoder and IMU packets using exact local C++ source and deployment YAML",
        "formula": "rolling source-time window (configured 0.10 s), abs mean of left/right; mapped uses speed calibration; packet uses adjacent source-time interval",
        "causal_window": "raw rolling window 0.10 s; packet estimate one source interval",
        "nominal_delay": "rolling window; packet endpoint; implementation exact, no separate filter-delay fit",
        "validity": "production packet assembler receives matching source-stamped L/R/IMU; observer reset/burst/output flags retained",
        "training_use": "audit comparator only unless a later handoff gate explicitly changes topology",
        "rollout_use": "not a future measurement",
        "role": "measurement output / diagnostic",
    },
    "imu_accel_yaw": {
        "name": "IMU body acceleration and yaw rate", "units": "m/s^2, rad/s",
        "source": "/autodrive/roboracer_1/imu linear_acceleration.{x,y}, angular_velocity.z; sensor_frames[:,4:7]",
        "frame_reference": "IMU sensor axes; acceleration reference includes sensor lever arm and gravity convention",
        "observation": "direct sensor measurements; not simulator truth acceleration",
        "formula": "none; any rotation/reference correction must be explicit",
        "causal_window": "one IMU message; sample timestamp alignment <=30 ms",
        "nominal_delay": "source and receive timing are audited separately; no assumed filter group delay",
        "validity": "finite fields, matched source stamp, per-run reset boundaries",
        "training_use": "sensor observer inputs; plant diagnostic/history only when explicit",
        "rollout_use": "only if predicted internally; no future measured IMU",
        "role": "measurement",
    },
    "imu_roll_roll_rate": {
        "name": "IMU roll and roll rate", "units": "rad, rad/s",
        "source": "/autodrive/roboracer_1/imu orientation and angular_velocity.x; imu_attitude_frames[:,0],[:,2]",
        "frame_reference": "IMU/body attitude axes; quaternion converted to roll; angular x is measured gyro rate",
        "observation": "direct measurement; simulator roll is not a rollout input",
        "formula": "roll from quaternion; rate from body gyro x",
        "causal_window": "one IMU sample plus model history", "nominal_delay": "alignment tolerance <=30 ms; filtering not inferred beyond recorded channel",
        "validity": "attitude validity mask and finite sample",
        "training_use": "past-only diagnostic or supervised latent target; prior roll test did not promote it",
        "rollout_use": "must be predicted as internal state if ever admitted; never use future measured roll",
        "role": "measurement / optional latent target",
    },
    "linear_acceleration_labels": {
        "name": "simulator linear-acceleration labels", "units": "m/s^2",
        "source": "bridge_packet_timing JSON; simulator_linear_acceleration_{x,y,z} / simulator_linear_acceleration array",
        "frame_reference": "simulator rigid-body body-frame acceleration at the COM",
        "observation": "simulator truth, aligned by bridge receive source stamp and packet sequence",
        "formula": "none; labels are not IMU acceleration",
        "causal_window": "same simulator packet", "nominal_delay": "same packet label",
        "validity": "all required acceleration components finite and packet matched",
        "training_use": "offline supervision/diagnostic only",
        "rollout_use": "none as a measured input; model predicts its own transition",
        "role": "label only",
    },
}


def fixed_period_surface_rate(previous_angle_rad: float,
                              current_angle_rad: float,
                              radius_m: float = WHEEL_RADIUS_M,
                              period_s: float = DT_S) -> float:
    """Surface rate for known physical cadence, independent of stamp jitter."""
    if not all(math.isfinite(value) for value in
               (previous_angle_rad, current_angle_rad, radius_m, period_s)):
        raise ValueError("encoder inputs must be finite")
    if radius_m <= 0.0 or period_s <= 0.0:
        raise ValueError("radius and physical period must be positive")
    return (current_angle_rad - previous_angle_rad) * radius_m / period_s


def variable_stamp_surface_rate(previous_angle_rad: float,
                                current_angle_rad: float,
                                source_dt_s: float,
                                radius_m: float = WHEEL_RADIUS_M) -> float:
    """Alternate stamp-divided diagnostic; never the canonical 40 Hz rate."""
    low, high = ENCODER_SOURCE_DT_GATE_S
    if not math.isfinite(source_dt_s) or not low <= source_dt_s <= high:
        raise ValueError("source interval is outside the encoder validity gate")
    return fixed_period_surface_rate(previous_angle_rad, current_angle_rad,
                                    radius_m, source_dt_s)


def fixed_n_period_surface_rate(older_angle_rad: float,
                                current_angle_rad: float,
                                periods: int,
                                radius_m: float = WHEEL_RADIUS_M) -> float:
    """Causal n-sample wheel rate assuming every physical step is 25 ms."""
    if not isinstance(periods, int) or periods < 1:
        raise ValueError("periods must be a positive integer")
    return fixed_period_surface_rate(older_angle_rad, current_angle_rad,
                                     radius_m, periods * DT_S)


def elapsed_window_surface_rate(older_angle_rad: float,
                                current_angle_rad: float,
                                elapsed_s: float,
                                radius_m: float = WHEEL_RADIUS_M) -> float:
    """Reconstruct the historical stored proxy using its measured window."""
    if not math.isfinite(elapsed_s) or not 0.075 <= elapsed_s <= 0.125:
        raise ValueError("stored 100 ms proxy window must be in [75, 125] ms")
    return fixed_period_surface_rate(older_angle_rad, current_angle_rad,
                                     radius_m, elapsed_s)


def com_to_rear_axle_velocity(u_com_mps: float, v_com_mps: float,
                              yaw_rate_radps: float,
                              rear_to_com_x_m: float = REAR_AXLE_TO_COM_X_M
                              ) -> tuple[float, float]:
    """Shift planar velocity from COM to rear axle in body coordinates."""
    if not all(math.isfinite(value) for value in
               (u_com_mps, v_com_mps, yaw_rate_radps, rear_to_com_x_m)):
        raise ValueError("kinematic inputs must be finite")
    return u_com_mps, v_com_mps - yaw_rate_radps * rear_to_com_x_m


def rear_contact_speeds(u_rear_mps: float, yaw_rate_radps: float,
                        track_width_m: float = REAR_TRACK_WIDTH_M
                        ) -> tuple[float, float]:
    """Rigid-body rear contact speeds (left, right), not tire-slip truth."""
    if not all(math.isfinite(value) for value in
               (u_rear_mps, yaw_rate_radps, track_width_m)):
        raise ValueError("kinematic inputs must be finite")
    if track_width_m <= 0.0:
        raise ValueError("track width must be positive")
    half_track = 0.5 * track_width_m
    return (u_rear_mps - yaw_rate_radps * half_track,
            u_rear_mps + yaw_rate_radps * half_track)


def wrapped_angle_difference(angle_a_rad: float, angle_b_rad: float) -> float:
    """Return the shortest signed angular difference in [-pi, pi]."""
    return math.atan2(math.sin(angle_a_rad - angle_b_rad),
                      math.cos(angle_a_rad - angle_b_rad))
