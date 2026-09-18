# MPC rewrite Phase 3: source-time state synchronization

`MpcStateSynchronizer` keeps a bounded, source-stamped `/odom` history and the
latest `/current_map_pose` global anchor. It interpolates odometry at the map
anchor timestamp, propagates the anchor to the newest odometry timestamp with
the SE(2) odometry delta, and returns newest legal body velocities plus
separate source-age and pose/odometry-skew measurements. Timestamp reversal,
unsupported source gaps, missing brackets, future anchors, excessive skew,
and excessive age are distinct faults. Zero stamps are not replaced with
callback time.

The recorded failed 4 m/s run was replayed using only `/odom` and
`/current_map_pose` records from `events.csv`; simulator truth and other topics
were not consumed. 2,255 of 2,306 odometry events synchronized (97.8%), with
accepted source age p95 85.05 ms and maximum 118.11 ms. There were no ordering
faults or source-gap faults. This demonstrates that the previous 75 ms raw-age
gate was rejecting a recoverable timing band.

This phase aligns the global pose to newest odometry but does not yet propagate
the state from that source timestamp all the way to command application time.
That causal forward prediction remains coupled to the new nonlinear stage
model. No live controller was enabled and no simulator was started.
