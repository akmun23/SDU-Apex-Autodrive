# Phase 1 field inventory — 2026-09-16

`existing_field_inventory.csv` is the complete source-by-field matrix required by the handoff. It has eight candidate sources and separate columns for body state, steering/next command, each FL/FR/RL/RR wheel quantity, six suspension coordinates, reset segment, fixed-step time, command sequence, 40 Hz source index, and coverage notes. Cell values use exactly the handoff categories: `exact recorded`, `derived causally`, `derivable only offline`, and `missing`.

The old 1 kHz open-raceline trace has 84,036 fixed steps and rich four-wheel/contact fields, but its bridge command sequence is constant zero and it has no matching 40 Hz source-packet index. The competition full trace has 70,028 steps and reaches only 5.76 m/s; it also lacks the causal bridge sequence. The two accepted track runs have legal `/odom`, `/ekf_odom`, `/current_map_pose`, `/cmd/speed`, sensor, and bridge timing records, but they are not paired with the Unity fixed-step wheel/suspension traces. The selected v5 axle-force rows are force diagnostics, not continuous N30 source histories. Hashes and run provenance are in `existing_field_inventory.json`.

## Join-path pilot

A disposable competition diagnostic player was built from Unity commit `bb0c5018b48b888faea405a0b0342729b5841dc9`. Its passive fixed-step logger records `VehicleController.AppliedCommandSequence`; the bridge recorder stores the same consumed sequence and simulator source time. A Pure Pursuit batchmode run demonstrated that these fields join directly without interpolation. However, after the 45-second recorder ended, the still-running ROS launch and simulator continued; the bridge later rejected a 10.004 ms source interval and latched neutral. The partial run is not accepted as a clean Phase 1 data source, and the trace was not copied from `/tmp` into the project.

The Unity diagnostic writer has since been buffered, and collision `stay` logs are now sampled at 10 Hz while exact enter/exit rows and every 1 ms wheel/body trace row are retained. Vehicle physics and commands are unchanged. The buffered build still requires a bounded live cadence pilot before the 12-run campaign.

## Gate

**Not passed.** There is still no accepted straight source and combined/track source with synchronized four-wheel RPM/contact, reconstructable heave/pitch/roll state, causal source command, and uninterrupted accepted source cadence. Do not run the Phase 2 oracle initialization ablation yet.
