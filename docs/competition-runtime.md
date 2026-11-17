# Competition runtime

This document covers startup, CPU pinning, bridge socket settings and recording for
the competition image. See [architecture.md](architecture.md) for nodes and topics,
and [mpc.md](mpc.md) for the controller.

## Build and run

```bash
tools/build_competition.sh    # runs the topic-policy check, then builds the image
tools/run_competition.sh      # starts the fixed competition launch
```

The image builds the MPC for the CPU that builds it, with link-time optimization
(`SDU_APEX_MPC_NATIVE_TUNING=ON`, `SDU_APEX_MPC_LTO=ON`), so build it on the machine that
runs it. The container entrypoint starts
[competition.launch.py](../sdu_apex_autodrive/launch/competition.launch.py) and nothing
else. The map and raceline default to the competition track; `SDU_APEX_MAP_REL` and
`SDU_APEX_TRAJECTORY_REL` select another pair under `f1tenth_planning/`.

## Startup order

Press Unity's Connect button at any time. The launch waits for data before starting
AMCL and MPC:

1. The bridge, sensor odometry, EKF, map server and actuator interface start.
2. A one-shot sensor gate waits for four consecutive fresh IMU and odometry samples
   (age at most 0.12 s, gap at most 0.18 s). Then AMCL starts.
3. A second gate waits for four consecutive fresh, valid `/current_map_pose` samples.
   Then the MPC container starts.

A failed gate shuts down the launch. AMCL uses the IROS spawn prior from
[amcl_iros.yaml](../f1tenth_localization/config/amcl_iros.yaml); the first raceline
waypoint is not the spawn pose.

## CPU layout

On the 6-P-core / 4-E-core host topology the run script pins the MPC container to CPUs
`0,2,4,6,8,10` with six Jacobian workers and bound OpenMP places, the other ROS processes
to `9,11`, and leaves `12-15` for the simulator and recorder. The MPC runs in a
multi-threaded component container (four executor threads). On other hosts nothing is
pinned. `SDU_APEX_COMPETITION_CPU_ISOLATION=off` disables the pinning.

## Bridge socket

The bridge keeps the 40 Hz Devkit request pacing. For each accepted IPv4 WebSocket
connection it sets `TCP_NODELAY` and re-arms `TCP_QUICKACK` every 5 ms, which removes the
Nagle and delayed-ACK stalls of about 40 ms on the loopback link. The options are
best-effort and produce no logs. A 40 Hz request rate alone does not prove timely sensor
delivery; with Unity connected, check the frame intervals and inspect the socket with
`ss -tinm 'sport = :4567'`. The command listener runs as its own node and does not inherit
the bridge's `__node` remap.

## Recording

```bash
tools/record_bag.sh                      # bags/competition_<timestamp>/ as MCAP
SDU_APEX_BAG_NAME=lap_test tools/record_bag.sh
SDU_APEX_RECORDER_CPUSET_CPUS=12-15 tools/record_bag.sh
```

The recorder runs in a separate container on the host network and does not affect the
autonomous launch. The bag is one MCAP file with zstd chunks and embedded message
definitions for every topic, so Foxglove opens it directly, including `DriveCommand` and
`UnityPlantState`. Poses carry `frame_id: map`; the stack publishes no `/tf`.

## Graphical tuning

[competition-tuner.md](competition-tuner.md) describes the desktop tool that edits the MPC
weights and restarts only the controller while the graphical Unity player stays open.
