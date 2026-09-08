# FTG run log

The FTG node publishes two diagnostic topics in addition to its drive command:

- `/ftg/processed_scan` (`sensor_msgs/msg/LaserScan`): the median-filtered scan after wall margin and disparity extension. `intensities` is a bit mask: `1` means disparity-blocked and `2` means bubble-blocked.
- `/ftg/diagnostics` (`std_msgs/msg/Float64MultiArray`): one fixed-width record for every processed LiDAR scan.

The decision record fields are:

```text
0  node time [s]
1  closest range [m]
2  closest angle [rad]
3  target angle [rad]
4  raw steering [rad]
5  commanded steering [rad]
6  commanded speed [m/s]
7  committed recovery sign (-1, 0, +1)
8  front-left clearance [m]
9  front-right clearance [m]
10 rear-left context clearance [m]
11 rear-right context clearance [m]
12 selected gap start [rad]
13 selected gap end [rad]
14 selected gap minimum range [m]
15 selected gap maximum range [m]
16 selected gap width [rad]
17 detected gap count
18 emergency-stop flag
19 footprint-limited flag
20 side-recovery flag
21 valid beam count
22 disparity-blocked beam count
```

A complete run record should also include the raw LiDAR, simulator odometry,
`/cmd/speed`, actuator feedback, collision count, and lap count. The mapping
run command records all of these together, so the odometry is the trajectory
and the two LiDAR topics are the raw and controller-shaped wall evidence.
