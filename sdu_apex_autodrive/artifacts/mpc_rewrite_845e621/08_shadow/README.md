# Phase 10 — shadow controller validation (blocked before telemetry)

# The 9-state RTI core has passed offline replay and is integrated into the ROS
# node. Static preflight confirmed the shadow component loads alongside Pure
# Pursuit and command authority is false. Shadow JSON separates source/ROS
# stamps from host-monotonic callback/synchronization/publish times and records
# the observed Pure Pursuit command history.

The first batchmode capture attempt is blocked before data: the local
competition player segfaulted in `GameAssembly.so` during `il2cpp_init()`.
The recorder subscribed to all requested topics but recorded zero messages.
See `BLOCKED.json` and the worklist for the exact command and evidence. No
Unity files or build outputs were changed. Do not substitute the older May
simulator image, and do not enable MPC authority.

Once a verified matching competition player bundle is available, repeat a
Pure Pursuit + shadow run in batchmode (never `-no-graphics`), record only
`/odom`, `/current_map_pose`, `/cmd/speed`, and `/mpc_shadow/diagnostics`, then
score N1/N5/N10/N20/N30 against subsequent legal observed states.
