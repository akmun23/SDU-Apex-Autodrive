# v2.2.8 — Native steering symbol compatibility (in-place)

## Problem reproduced

v2.2.7 can stop at startup with `UnresolvedCppSymbol: Cannot resolve kSteeringLimitDeg: no source definition` if the current local `physx_odom_runtime.cpp` uses a different constant naming scheme or external build-time macros. No IPOPT or PhysX evaluation is started.

## Fix

- Source discovery no longer requires `kSteeringLimitDeg` and `kSteeringRateDegPerS` declarations to exist in native C++.
- Read local project headers for these symbols, where present.
- Read explicitly implemented steering update constants from `f1tenth_mpc/src/unity_source_plant.cpp` (`last_internal_deg / ...`, `-command_norm * ...`, `... * kDt`) as source-backed fallback, cross-checking the target limit against the threshold.
- Cross-check all available plant update, plant declared, and native declared constants; abort on mismatches or ambiguous definitions.
- Warn in the console and emit provenance metadata when an alias depends on unavailable compiler `-D` values or the native actuator constant cannot be independently recovered. Such values are *not proven to match the compiled vehicle*.
- The exact recovered IPOPT historical source, the v2.2.7 elastic Phase-I implementation, and the 1-ms native-PhysX acceptance gate are unchanged.

## Verification

50 tests pass, including six new regression cases. The full returned IPOPT/PhysX/Unity lap has **not** been executed in this environment. Feasibility and sub-4.9-second performance remain unverified.

## Installation

Unzip this archive at the repository root with `unzip -o`, overlaying the existing `SDU_Apex_IPOPT_PhysX_MinTime_v2` directory. It does not contain or touch `.venv/`, `native_physx_vehicle_step`, `_restored_git_history/`, `_working_solver_limited_memory/` or `optimized_*` outputs.
