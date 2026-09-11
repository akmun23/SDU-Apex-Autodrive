# F1Tenth control

This package contains the two active simulator controllers:

- `f1tenth_control::FTGNode` for LiDAR-only mapping
- `f1tenth_control::PurePursuitNode` for raceline driving

Both publish physical-unit `AckermannDriveStamped` speed commands on
`/cmd/speed`. The single `sdu_apex_autodrive/actuator_interface` owns the
conversion to simulator steering and throttle topics.

Build and run through the single launch entry point. For a complete simulator
run, use the repository-level [startup guide](../STARTUP_GUIDE.md); the direct
commands are useful only when the simulator and bridge lifecycle are already
under deliberate manual control:

```bash
ros2 launch sdu_apex_autodrive controller.launch.py controller:=ftg
ros2 launch sdu_apex_autodrive controller.launch.py controller:=pure_pursuit
```

Pure Pursuit consumes `/current_map_pose`, `/ekf_odom`, and the configured
raceline. FTG consumes the native simulator LiDAR and odometry topics. The
active vehicle envelope is a `0.324 m` wheelbase, `±0.5236 rad` steering limit,
and `3.2 rad/s` steering-rate limit.

Pure Pursuit starts at `1.5 m/s` and ramps its command cap to the configured
maximum over one completed raceline lap. The cap is then removed; the
raceline and controller regulation determine the requested speed.
