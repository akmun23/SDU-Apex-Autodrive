# Automatic IROS Unity/MPC validation

## Workflow

The validation supervisor watches source-certified exports of the map-only
optimizer. It never runs uncertified IPOPT candidates. On first startup it
queues only the latest currently certified reference and marks earlier exports
as seen. Each subsequent certified export enters a serialized FIFO queue.

It starts the existing production IROS launcher with the immutable, SHA-256
identified trajectory and the IROS map. The watcher stops the run on collision,
MPC rejection, motion stall, timeout, or completion of **11 official laps**:
one warmup plus 10 scored laps. A failed trial is not automatically retried.

This is an independent process. It pauses the verified optimizer's dedicated
systemd cgroup during controller building, Unity driving and bag analysis,
then resumes the exact same iterate. Only one heavy workload runs at once.
It never changes the optimizer's best manifest or runtime controller reference
because of a Unity trial. A durable pause lease is released on normal shutdown
and recovered on supervisor restart after a crash.

## Services and commands

Check whether service is running:

    systemctl --user status sdu-apex-mpc-validation-iros-conditioned.service
    tail -F live_runs/optimizer_iros_conditioned_20261029/mpc_validation/supervisor.log

Stop the validation supervisor only:

    systemctl --user stop sdu-apex-mpc-validation-iros-conditioned.service

Single queue attempt when the service is stopped:

    python3 tools/optimizer_mpc_validation.py \
      --workspace live_runs/optimizer_iros_conditioned_20261029 --once

Run a foreground supervisor when the service is stopped:

    source tools/docker_env.sh
    python3 tools/optimizer_mpc_validation.py \
      --workspace live_runs/optimizer_iros_conditioned_20261029 --watch

Request a new attempt for a failed witness:

    python3 tools/optimizer_mpc_validation.py \
      --workspace live_runs/optimizer_iros_conditioned_20261029 \
      --retry f8c5b636134d8b64

The viewer at http://127.0.0.1:8765/ has a Unity + MPC trajectory trials
panel, including live official-lap progress, clear model-versus-Unity results,
actual supervisor status and a **Retry this driving test** button. This single
write endpoint accepts same-origin JSON only and queues a previously failed,
independently source-certified candidate. It cannot execute arbitrary commands.
The supervisor owns launching and serializes all trials. Repeated clicks do
not create duplicate queued attempts. A terminal command can also be copied.

## Results

Every attempt has an independent run folder:

    live_runs/apex_mpc_<witness-id>_aNN/

The source witness snapshot and host launch logs remain under the optimizer's
mpc_validation/inputs/ folder. The run folder retains map/trajectory copies,
controller/simulator/recorder/watcher/build logs, the complete development
ROS bag, a versioned MPC report JSON, cycle detail CSV, and full diagnostics
JSONL (if bag analysis succeeds).

Analysis runs inside the controller's ROS image, rather than depending on
host Python having rclpy and the generated vehicle messages. New trials save
`controller_image_id.txt` and the decoder uses that immutable image ID; older
recordings without it use the configured controller image. Failed runs are
decoded as well, with zero completed laps distinguished from missing data.

The report includes official per-lap durations and per-lap RMS reference
cross-track/heading/speed error, as well as full-run average, total, median,
minimum/maximum, standard deviation, MPC convergence statuses, fault reasons,
timings (median/p95/max), per-progress-bin diagnostics and raw cycles.

A run is successful only if the watcher completes without abort, the process
returns successfully and the ROS bag contains ten scored completed lap-time
events after warmup. Missing diagnostic or lap evidence is explicitly flagged.

The optimizer source-only certificates are unaffected. Online controller
completion is not proof that the source model matches private Unity PhysX.

## Resource and scope limitations

This integration is specifically for the IROS competition track in Unity-confirmed racing direction.
Each trial builds a fresh production controller image, so stale binaries are
not silently reused. Simulation, builds and analysis run with IPOPT paused;
the GUI remains responsive. The default watcher wall timeout is 900 seconds,
and a separate outer watchdog guards the launch process. Existing active
Unity/controller/recorder containers defer the next auto trial rather than
being stopped. It does not automatically promote any result into the
competition runtime or modify the optimizer solver.

## Initial failure and corrected optimizer direction (2026-10-29)

The first automatic trial used the source-validated 7.350 s witness
(8c2ef06394b9af69). Unity, MPC, and the recorder launched, but the
controller could not localize: AMCL repeatedly rejected heading errors
of about 2.89 rad and MPC received no localized map pose.

**Root cause: the raceline optimizer generated the track in the opposite
travel orientation from the previously completed Unity/MPC laps.**
The simulator's verified travel direction is authoritative. The original
optimizer enforced negative yaw winding and negative signed area in the
ROS map XY frame; the Unity-driven reference and first successful IROS
two-lap trial use positive winding and positive signed area. Near the start,
the old witness differed by -3.138 rad (-179.8 degrees). Do NOT change
Unity's spawn, force opposite traversal, or relax AMCL heading checks.

The original wrong-direction optimizer workspace and best.json are
preserved as diagnostic artifacts, not to be fed into autonomous driving.
The corrected optimizer reads the reference from
live_runs/iros_mpc_initial_20261029_02/raceline.csv, uses its direction
when building map geometry and periodic state/command seeds, and checks
the sign of yaw winding and signed path area on every newly certified witness.
It uses a new output workspace and a separate source-contract hash to prevent
resume from the incompatible campaign. The saved opposite-direction
witness cannot be reversed while retaining dynamics certification.

Single active dashboard: http://127.0.0.1:8765/
Older workspace results remain on disk; no second dashboard is required.

The dashboard retains expanded failure panels, preserves selected text
across two-second polls, updates unchanged regions without rebuilding
them, and provides a Pause live display button. Pausing the display does
not stop IPOPT or Unity automation.

A 75-second MPC startup watchdog remains in place in case localization
fails for another reason. The verifier independently checks candidate
heading relative to the Unity-driven start before launching expensive
ten-lap trials.
