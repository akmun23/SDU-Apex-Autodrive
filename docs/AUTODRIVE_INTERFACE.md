# AutoDRIVE actuator interface

No physical-vehicle actuator calibration belongs at the AutoDRIVE boundary.
The simulator exposes normalized actuator inputs directly.

## Source-backed contract

The installed 2026 bridge declares both command values as `[-1, 1]`, rounds
incoming ROS values to three decimal places, then forwards them unchanged to
the simulator.

| Quantity | Native value or unit |
| --- | --- |
| Throttle command | normalized `[-1, 1]` |
| Steering command | normalized `[-1, 1]` |
| Neutral command | `0.0` for both |
| Centre steering-angle limit | `[-0.5236, 0.5236] rad` |
| Steering rate | `3.2 rad/s` |
| Vehicle top speed | `22.88 m/s` |
| Wheelbase | `0.3240 m` |
| Vehicle length | `0.5000 m` |
| Vehicle width | `0.2700 m` |
| Track width | `0.2360 m` |
| Rear-axle to COM | `0.15532 m` |
| LiDAR x from rear axle | `0.2733 m` |
| Vehicle mass | `3.906 kg` |

Official reference:
<https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/>

The native ROS topics are:

| Topic | Meaning |
| --- | --- |
| `/autodrive/roboracer_1/steering_command` | normalized actuator input |
| `/autodrive/roboracer_1/throttle_command` | normalized actuator input |
| `/autodrive/roboracer_1/steering` | actual centre steering angle in radians |
| `/autodrive/roboracer_1/throttle` | normalized actuator feedback |
| `/autodrive/roboracer_1/odom` | pose and velocity feedback |

The steering conversion is therefore:

```text
steering_command = clamp(steering_angle_rad / 0.5236, -1, 1)
```

This is a unit conversion from the documented interface, not a fitted gain.
The static neutral-throttle check also returned `0.0262 rad` for a `0.05`
normalized steering command, matching `0.05 * 0.5236`.

## Why throttle still needs control

`AckermannDrive.speed` is a target speed in metres per second. AutoDRIVE does
not expose a target-speed command; it exposes normalized motor throttle.
Consequently, no exact `speed * gain = throttle` conversion exists in the
bridge or simulator interface.

The adapter preserves the correct Ackermann contract and uses odometry
feedback to control normalized forward throttle. Its proportional/integral
values are controller tuning, not actuator calibration. The default controller
uses target-speed feedforward plus signed PI feedback, then a throttle slew
rate. It does not cut throttle to zero merely because speed is slightly above
target. `maximum_forward_throttle: 0.07` remains an operating cap;
AutoDRIVE's native hard limit remains `1.0`.

At a zero target, the adapter publishes zero throttle. The simulator documents
idle braking at zero throttle. Reverse is not used by the initial racing stack.

## Safety behavior

The adapter remains disarmed at startup and publishes neutral for:

- disarm
- missing or stale controller command
- missing or stale odometry
- invalid numeric input
- shutdown

Arm:

```bash
ros2 topic pub --once /autodrive/adapter/enable \
  std_msgs/msg/Bool '{data: true}'
```

Disarm:

```bash
ros2 topic pub --once /autodrive/adapter/enable \
  std_msgs/msg/Bool '{data: false}'
```

Controller speed behavior may still need tuning on track. Tune the speed
controller and operational caps, not the native actuator range or steering
conversion.
