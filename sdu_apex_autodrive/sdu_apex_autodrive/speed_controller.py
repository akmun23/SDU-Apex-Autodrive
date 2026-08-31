"""Measured-feedforward speed and longitudinal acceleration controllers.

The AutoDRIVE actuator interface has a normalized throttle input, not a direct
acceleration input.  The speed feed-forward table supplies the throttle needed
to hold a body speed, while the acceleration controller adds the calibrated
speed-dependent throttle required to produce a requested longitudinal
acceleration.  The online controller only consumes allowed odometry and IMU
data; simulator truth is used offline to identify the tables.
"""

from bisect import bisect_right
from dataclasses import dataclass
import math


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


@dataclass(frozen=True)
class SpeedControllerConfig:
    kp: float
    ki: float
    ka: float
    integral_limit: float
    throttle_max_forward: float
    throttle_rise_rate_per_sec: float
    throttle_fall_rate_per_sec: float
    stop_speed_threshold_mps: float
    overspeed_coast_threshold_mps: float
    feedforward_speed_mps: tuple[float, ...]
    feedforward_throttle: tuple[float, ...]
    # Speed error is converted to a requested body acceleration before the
    # throttle inverse is applied.  These are not throttle ceilings.
    speed_error_to_accel_gain: float = 1.25
    speed_error_integral_to_accel_gain: float = 0.05
    max_acceleration_mps2: float = 6.0
    max_deceleration_mps2: float = 8.0
    acceleration_feedback_gain: float = 0.25
    acceleration_integral_gain: float = 0.003
    acceleration_integral_limit: float = 2.0
    acceleration_speed_mps: tuple[float, ...] = (
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 23.0,
    )
    # Throttle added per requested m/s^2.  It rises with speed because the
    # open-ground measurements show a progressively smaller acceleration
    # response and more wheel slip at high speed.
    acceleration_throttle_per_mps2: tuple[float, ...] = (
        0.158521, 0.160881, 0.163312, 0.165818, 0.168402, 0.171068,
        0.173819, 0.176661, 0.179597, 0.182632, 0.185771, 0.190688,
    )

    def validate(self) -> None:
        scalars = (
            self.kp, self.ki, self.ka, self.integral_limit,
            self.throttle_max_forward, self.throttle_rise_rate_per_sec,
            self.throttle_fall_rate_per_sec, self.stop_speed_threshold_mps,
            self.overspeed_coast_threshold_mps,
        )
        if not all(math.isfinite(v) for v in scalars):
            raise ValueError("speed-controller values must be finite")
        if self.kp < 0.0 or self.ki < 0.0 or self.integral_limit < 0.0:
            raise ValueError("kp, ki and integral_limit must be >= 0")
        if not 0.0 < self.throttle_max_forward <= 1.0:
            raise ValueError("throttle_max_forward must be in (0, 1]")
        if self.throttle_rise_rate_per_sec <= 0.0 or self.throttle_fall_rate_per_sec <= 0.0:
            raise ValueError("throttle slew rates must be > 0")
        if self.stop_speed_threshold_mps < 0.0:
            raise ValueError("stop_speed_threshold_mps must be >= 0")
        if self.overspeed_coast_threshold_mps < 0.0:
            raise ValueError("overspeed_coast_threshold_mps must be >= 0")
        if (
            self.speed_error_to_accel_gain < 0.0 or
            self.speed_error_integral_to_accel_gain < 0.0 or
            self.max_acceleration_mps2 <= 0.0 or
            self.max_deceleration_mps2 <= 0.0 or
            self.acceleration_feedback_gain < 0.0 or
            self.acceleration_integral_gain < 0.0 or
            self.acceleration_integral_limit < 0.0
        ):
            raise ValueError("invalid acceleration-controller gains or limits")

        speeds = self.feedforward_speed_mps
        throttles = self.feedforward_throttle
        if len(speeds) != len(throttles) or not speeds:
            raise ValueError("feedforward arrays must be non-empty and equal length")
        if any(speeds[i] >= speeds[i + 1] for i in range(len(speeds) - 1)):
            raise ValueError("feedforward speeds must be strictly increasing")
        if not all(math.isfinite(v) for v in (*speeds, *throttles)):
            raise ValueError("feedforward values must be finite")
        if any(v < 0.0 or v > self.throttle_max_forward for v in throttles):
            raise ValueError("feedforward throttle outside configured limit")

        accel_speeds = self.acceleration_speed_mps
        accel_gains = self.acceleration_throttle_per_mps2
        if len(accel_speeds) != len(accel_gains) or not accel_speeds:
            raise ValueError("acceleration arrays must be non-empty and equal length")
        if any(accel_speeds[i] >= accel_speeds[i + 1]
               for i in range(len(accel_speeds) - 1)):
            raise ValueError("acceleration speeds must be strictly increasing")
        if not all(math.isfinite(v) for v in (*accel_speeds, *accel_gains)):
            raise ValueError("acceleration values must be finite")
        if accel_speeds[0] < 0.0 or any(v <= 0.0 for v in accel_gains):
            raise ValueError("acceleration speed/gain values must be positive")


class AccelerationController:
    """Convert desired body acceleration into a throttle correction.

    ``base_throttle`` is the measured steady-speed feed-forward value.  The
    correction is an inverse acceleration model with IMU acceleration
    feedback.  It deliberately returns a correction instead of clamping the
    final throttle, so the caller retains the complete normalized [0, 1]
    actuator range.
    """

    def __init__(self, config: SpeedControllerConfig) -> None:
        self.config = config
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def throttle_per_acceleration(self, speed_mps: float) -> float:
        speeds = self.config.acceleration_speed_mps
        gains = self.config.acceleration_throttle_per_mps2
        speed = max(0.0, speed_mps)
        if speed <= speeds[0]:
            return gains[0]
        if speed >= speeds[-1]:
            return gains[-1]
        upper = bisect_right(speeds, speed)
        lower = upper - 1
        ratio = (speed - speeds[lower]) / (speeds[upper] - speeds[lower])
        return gains[lower] + ratio * (gains[upper] - gains[lower])

    def throttle_for_acceleration(
        self,
        speed_mps: float,
        base_throttle: float,
        desired_accel_mps2: float,
    ) -> float:
        """Return the unbounded normalized throttle for an acceleration.

        The calibration identifies the incremental throttle needed around a
        steady operating point, so the absolute command is

        ``base_throttle + desired_acceleration * gain(speed)``.

        This method intentionally does not clamp the result.  Saturation is a
        responsibility of the actuator boundary, where the full normalized
        protocol range is visible.  It is useful both for the runtime inverse
        model and for offline calibration checks.
        """
        if not all(math.isfinite(v) for v in (
            speed_mps, base_throttle, desired_accel_mps2
        )):
            raise ValueError("invalid acceleration-to-throttle input")
        return base_throttle + (
            self.throttle_per_acceleration(speed_mps) * desired_accel_mps2
        )

    def acceleration_from_throttle(
        self,
        speed_mps: float,
        throttle: float,
        base_throttle: float,
    ) -> float:
        """Estimate longitudinal acceleration from a throttle command.

        ``base_throttle`` is the steady-speed feed-forward command at the
        current body speed.  The calibrated inverse map describes the extra
        normalized throttle required per m/s^2 above that operating point.
        This is an estimator/diagnostic primitive; the runtime controller
        closes the loop with the allowed IMU acceleration measurement.
        """
        if not all(math.isfinite(v) for v in (
            speed_mps, throttle, base_throttle
        )):
            raise ValueError("invalid throttle-to-acceleration input")
        gain = self.throttle_per_acceleration(speed_mps)
        return (throttle - base_throttle) / gain

    def correction(
        self,
        speed_mps: float,
        desired_accel_mps2: float,
        measured_accel_mps2: float,
        dt_seconds: float,
    ) -> float:
        if not all(math.isfinite(v) for v in (
            speed_mps, desired_accel_mps2, measured_accel_mps2, dt_seconds
        )) or dt_seconds <= 0.0:
            raise ValueError("invalid acceleration-controller input")

        accel_error = desired_accel_mps2 - measured_accel_mps2
        candidate_integral = clamp(
            self.integral + accel_error * dt_seconds,
            -self.config.acceleration_integral_limit,
            self.config.acceleration_integral_limit,
        )
        effective_accel = (
            desired_accel_mps2
            + self.config.acceleration_feedback_gain * accel_error
            + self.config.acceleration_integral_gain * candidate_integral
        )
        # Do not build an integral wind-up around the unavailable reverse/brake
        # channel.  The outer speed controller handles passive coasting.
        if desired_accel_mps2 >= 0.0 or measured_accel_mps2 > 0.0:
            self.integral = candidate_integral
        else:
            self.integral = 0.0
        return self.throttle_for_acceleration(
            speed_mps, 0.0, effective_accel)


class LongitudinalStateEstimator:
    """Allowed-input longitudinal observer for the calibrated odometry.

    The C++ odometry node already converts the documented wheel angle into a
    body-speed estimate and applies the measured high-speed slip map.  That
    odometry stream is therefore the absolute speed reference for control.
    The IMU is retained as a filtered acceleration signal and as a diagnostic
    disagreement indicator; its open-loop integral is not allowed to mask a
    valid odometry update with accumulated bias.
    """

    def __init__(
        self,
        acceleration_filter_alpha: float = 0.35,
        slip_threshold_mps: float = 0.75,
        slip_ratio: float = 0.20,
        odom_correction_gain: float = 0.25,
    ) -> None:
        if not 0.0 < acceleration_filter_alpha <= 1.0:
            raise ValueError("acceleration_filter_alpha must be in (0, 1]")
        if slip_threshold_mps < 0.0 or slip_ratio < 0.0:
            raise ValueError("slip thresholds must be >= 0")
        if not 0.0 <= odom_correction_gain <= 1.0:
            raise ValueError("odom_correction_gain must be in [0, 1]")
        self.acceleration_filter_alpha = acceleration_filter_alpha
        self.slip_threshold_mps = slip_threshold_mps
        self.slip_ratio = slip_ratio
        self.odom_correction_gain = odom_correction_gain
        self.reset()

    def reset(self, speed_mps: float = 0.0) -> None:
        self.speed_mps = max(0.0, float(speed_mps))
        self.filtered_acceleration_mps2 = 0.0
        self.last_acceleration_time = None
        self.last_odom_time = None
        self.has_acceleration = False
        self.has_odom = False
        self.slip_detected = False

    def update_acceleration(self, acceleration_mps2: float, now_seconds: float) -> None:
        if not all(math.isfinite(v) for v in (acceleration_mps2, now_seconds)):
            return
        acceleration = clamp(acceleration_mps2, -25.0, 25.0)
        if self.last_acceleration_time is not None:
            dt = now_seconds - self.last_acceleration_time
            if 1.0e-4 < dt <= 0.5:
                previous = self.filtered_acceleration_mps2
                self.filtered_acceleration_mps2 = (
                    self.acceleration_filter_alpha * acceleration
                    + (1.0 - self.acceleration_filter_alpha) * previous
                )
                # Before the first odometry sample, integration gives the
                # actuator a bounded startup estimate. Once /odom is live,
                # keep this observer from overwriting its calibrated absolute
                # speed with IMU bias between two odometry events.
                if not self.has_odom:
                    self.speed_mps = max(
                        0.0, self.speed_mps + self.filtered_acceleration_mps2 * dt)
        else:
            self.filtered_acceleration_mps2 = acceleration
        self.last_acceleration_time = now_seconds
        self.has_acceleration = True

    def update_odometry(self, speed_mps: float, now_seconds: float) -> float:
        if not all(math.isfinite(v) for v in (speed_mps, now_seconds)):
            return self.speed_mps
        odom_speed = max(0.0, speed_mps)
        if not self.has_odom:
            self.speed_mps = odom_speed
            self.has_odom = True
        elif not self.has_acceleration:
            self.speed_mps = odom_speed
        elif odom_speed <= 0.20 and self.speed_mps > 0.50:
            # A reset or a genuine stop must not leave the IMU observer moving.
            self.speed_mps = odom_speed
            self.slip_detected = False
        else:
            discrepancy = abs(odom_speed - self.speed_mps)
            threshold = max(
                self.slip_threshold_mps,
                self.slip_ratio * max(odom_speed, self.speed_mps),
            )
            self.slip_detected = discrepancy > threshold
            # /odom is the calibrated absolute body-speed measurement. Always
            # accept it for control; ``slip_detected`` remains available for
            # diagnostics but must not leave the speed controller operating on
            # a stale IMU integral.
            self.speed_mps = odom_speed
        self.last_odom_time = now_seconds
        return self.speed_mps

    @property
    def acceleration_mps2(self) -> float:
        return self.filtered_acceleration_mps2


class TargetSpeedController:
    def __init__(self, config: SpeedControllerConfig) -> None:
        config.validate()
        self.config = config
        self.acceleration_controller = AccelerationController(config)
        self.integral = 0.0
        self.last_output = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.last_output = 0.0
        self.acceleration_controller.reset()

    def reconfigure(self, config: SpeedControllerConfig) -> None:
        config.validate()
        self.config = config
        self.acceleration_controller = AccelerationController(config)
        self.reset()

    def feedforward(self, target_speed_mps: float) -> float:
        speeds = self.config.feedforward_speed_mps
        throttles = self.config.feedforward_throttle
        if len(speeds) == 1:
            return throttles[0]
        if target_speed_mps <= speeds[0]:
            return throttles[0]
        if target_speed_mps >= speeds[-1]:
            return throttles[-1]
        upper = bisect_right(speeds, target_speed_mps)
        lower = upper - 1
        ratio = (target_speed_mps - speeds[lower]) / (speeds[upper] - speeds[lower])
        return throttles[lower] + ratio * (throttles[upper] - throttles[lower])

    def update(
        self,
        target_speed_mps: float,
        measured_speed_mps: float,
        requested_accel_mps2: float,
        dt_seconds: float,
        measured_accel_mps2: float = 0.0,
    ) -> float:
        if not all(math.isfinite(v) for v in (
            target_speed_mps, measured_speed_mps, requested_accel_mps2,
            dt_seconds, measured_accel_mps2,
        )) or dt_seconds <= 0.0:
            raise ValueError("invalid speed-controller input")

        target = max(0.0, target_speed_mps)
        measured = max(0.0, measured_speed_mps)
        if target <= self.config.stop_speed_threshold_mps:
            self.reset()
            return 0.0

        error = target - measured
        candidate_integral = clamp(
            self.integral + error * dt_seconds,
            -self.config.integral_limit,
            self.config.integral_limit,
        )
        requested_accel = clamp(
            requested_accel_mps2
            + self.config.speed_error_to_accel_gain * error
            + self.config.speed_error_integral_to_accel_gain * candidate_integral,
            -self.config.max_deceleration_mps2,
            self.config.max_acceleration_mps2,
        )
        acceleration_correction = self.acceleration_controller.correction(
            measured,
            requested_accel,
            measured_accel_mps2,
            dt_seconds,
        )
        unsaturated = (
            self.feedforward(target)
            + acceleration_correction
            + self.config.ka * requested_accel_mps2
            + self.config.kp * error
            + self.config.ki * candidate_integral
        )
        # AutoDRIVE has no active brake channel in the competition actuator
        # contract. Once measured speed is materially above the requested
        # speed, positive feed-forward would prolong the overspeed. Coast
        # immediately and clear the integral so the next target does not
        # inherit stale acceleration demand.
        materially_overspeed = (
            measured - target > self.config.overspeed_coast_threshold_mps
        )
        if materially_overspeed:
            desired = 0.0
        else:
            desired = clamp(unsaturated, 0.0, self.config.throttle_max_forward)

        if not materially_overspeed and (
            0.0 < unsaturated < self.config.throttle_max_forward
            or (unsaturated >= self.config.throttle_max_forward and error < 0.0)
            or (unsaturated <= 0.0 and error > 0.0)
        ):
            self.integral = candidate_integral
        elif materially_overspeed:
            # No active brake channel is available in the competition
            # actuator contract. Do not retain acceleration demand while
            # waiting for passive coast-down.
            self.integral = 0.0

        rate = (
            self.config.throttle_rise_rate_per_sec
            if desired >= self.last_output
            else self.config.throttle_fall_rate_per_sec
        )
        max_step = rate * dt_seconds
        self.last_output = clamp(
            desired,
            max(0.0, self.last_output - max_step),
            min(self.config.throttle_max_forward, self.last_output + max_step),
        )
        return self.last_output
