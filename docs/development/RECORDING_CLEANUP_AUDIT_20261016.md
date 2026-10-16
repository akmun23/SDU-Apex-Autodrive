# Recording cleanup audit — 2026-10-16

## Scope and preservation rule

This is a dependency-aware cleanup, not a claim that every remaining file has
already been proven useful. Raw simulator captures were retained unless their
contents and provenance showed that they were empty, invalid, or strictly
duplicated by a retained capture. Age, filename, file size, or low sample rate
alone were not used to delete a recording. Model work is source-derived under
the current requirement: Unity/PhysX source and configured values are used
directly; captures are for diagnosis and validation, not parameter fitting.

## Current inventory

The refreshed generated index is `live_runs/INDEX_20261016.csv`. It contains
**804 bags totaling 32.813 GiB**:

| Category | Bags | Size (GiB) |
|---|---:|---:|
| open-plane dynamics/throttle | 342 | 19.212269 |
| odometry/localization | 232 | 6.327267 |
| practice/racing | 144 | 3.722153 |
| MPC/controller | 52 | 1.707691 |
| competition/bridge | 7 | 0.419495 |
| mapping/SLAM | 2 | 0.534492 |
| other | 25 | 0.889633 |

The displayed total is rounded to three decimals; category values come from
the index's byte sizes: 35,232,690,176 bytes (32.812999725 GiB). The broader
`live_runs` directory is about 108 GiB;
the difference is mostly non-bag content, including Unity player/build
snapshots and source captures. Those are outside this recording-only cleanup.

No full semantic classification of all 804 bags is claimed. The index is a
manageable inventory, not proof every retained bag is independently useful.
The large throttle sweeps, full-spectrum yaw captures, practice runs,
Unity-source captures, and bags referenced by current analysis/configuration
were deliberately preserved. Exact-content duplicates among some derived
`.npz` files remain in separate manifest-owned dataset/cache layouts; they
were not removed because each path is part of its local dataset contract.
An earlier SHA-256 scan of same-size candidate DB3s larger than 1 MiB and of
`wheel_dynamics*.csv` files found no exact duplicates in those scanned groups.
That does not rule out duplicates outside those candidate sets or logically
redundant captures whose bytes differ.

## Removed, with evidence

### Earlier cleanup pass

- **69 generated replay databases**, 641,568,768 bytes (0.598 GiB): each was
  exactly a `*/replayed/replayed_0.db3` artifact, held only
  `/replayed_odom` and diagnostics, had replay provenance pointing to an
  existing source bag, and was not the replay explicitly used by the yaw
  candidate configuration. The original bags, replay metadata, reports, and
  score files were retained. These outputs can be regenerated.
- **Six empty/zero-message derived databases and two invalid derived bag
  payloads:** failed derivatives, with source inputs retained. No valid raw
  driving capture was removed in that pass.

### Post-outage cleanup

- Removed the marker-only `run_0.db3` files from the `r12d` and `r12e`
  mechanics-validation attempts. Each contained only one experiment-phase
  marker and a reset command, with no vehicle sensor, actuator, or truth
  stream. The captures had no commanded motion and no dynamic response.
- Removed the `r12e` `wheel_dynamics` CSV because its 55,456 data rows were an
  prefix duplicate of the retained `r12d` static-settle CSV according to the
  comparison performed at deletion time, with only clock-row differences
  reported. The retained `r12d` CSV remains the representative stationary
  settle record; the `r12d` and `r12e` logs/manifests were kept. However, the
  comparison output/checksum was not saved before deleting `r12e`; an
  independent reviewer therefore cannot now reproduce that duplicate claim.
  Treat this particular deletion as supported by the contemporaneous analysis
  but not independently auditable from retained artifacts.
- Removed one invalid Ghidra pseudo-decompilation text report. It was not a
  recording or a valid binary/source artifact; the generic Ghidra scripts,
  matching Unity player/debug binary, and PhysX source checkout remain.

The two marker-only DB3 files were 32,768 bytes each; the removed `r12e` CSV
was 60,925,790 bytes, for 60,991,326 known bytes total, excluding the
pseudo-decompilation report whose old size was not retained. The index after
cleanup is internally consistent, but no complete before/after byte ledger
was preserved for every earlier derived-file deletion.

No valid original driving bag or unique moving 1 kHz Unity capture was removed.
The 1 kHz `r07`, `r11`, and `r12g` source captures used for the current
no-fit suspension comparison remain available.

An independent read-only review confirmed the 804-bag count, category and byte
totals, and that `r12d`/`r12e` bags had no vehicle/truth stream; it also
confirmed the retained `r12d` CSV is a stationary settle sequence. The review
could not verify the removed `r12e` CSV comparison because its source and
comparison artifact are no longer present. This is recorded as an evidence
gap, not silently treated as a verified content hash.

## Outage recovery state

The previous long-running IPOPT process ended with the power outage. No
simulator process or live optimizer process is currently running. The old
partial optimizer output was not accepted as a result: its plant still
contained fitted wheel-state/force components when that run was stopped. The
source-derived wheel equations have since been updated and unit-tested, but no
new IPOPT solve or simulator validation has been run against them.

## Important retained partial experiments

- The long `r03` throttle capture is not a completed 1,508-condition sweep,
  but it contains a long approximately 40 Hz truth/IMU response segment for
  the first commanded condition; retain it for recursive/source-model checks.
- The exhaustive straight `r01` throttle capture stopped at 245 complete of
  5,050 expected transitions. Its fine one-percent steps are unique relative
  to the stratified five-percent surface, so it remains as a partial
  validation record, not as a completed sweep.
- Lower-rate original bags remain where they provide held-out evaluation,
  distinct conditions, or analysis dependencies. Lower rate by itself is not
  evidence of uselessness.

## Source parity context

Unity 2022.3.52f1 project files expose the vehicle scripts and configuration;
the deployed Unity player has a matching debug-symbol file. The pinned public
PhysX 4.1 source supplies the upstream vehicle algorithms. The current
source-parity work translates wheel inertia, sprung masses, suspension and
tire-load behavior, slip, and the configured tire callback without fitting
coefficients. Existing recordings are being used only to check the translation
and identify remaining source/runtime timing differences. This is not yet a
bit-identical Unity rigid-body/contact simulator and has not been integrated
into runtime odometry or MPC. Details and residuals are in
[`PHYSX_41_WHEELCOLLIDER_SOURCE_PARITY_20261016.md`](PHYSX_41_WHEELCOLLIDER_SOURCE_PARITY_20261016.md).

## Next cleanup pass

Continue by classifying original bags by experiment purpose and dependency,
then compare candidate duplicates by content and measured coverage. Remove
only records whose complete source/outcome/evaluation evidence is represented
elsewhere and whose consumers/manifests can be updated safely. Do not treat
the bag index as a substitute for understanding what a capture contains. No
simulator was run as part of this audit.
