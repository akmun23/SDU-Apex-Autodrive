# v2.2.2 — In-place steering alias fix

The v2.2.1 setup check failed with `Cannot resolve kMaxSteeringRad expression 'UNITY_SOURCE_MAX_STEERING_RAD'` in a local checkout containing named C++ constants.

- Support simple scalar `constexpr`/`const` and `#define` name aliases, nested aliases, finite arithmetic, and `kPi` without executing C++ or arbitrary Python.
- Resolve aliases from local `#include "..."` headers of `unity_source_plant.cpp`.
- When the *specific* steering aliases `UNITY_SOURCE_MAX_STEERING_RAD` or `UNITY_SOURCE_STEERING_RATE_RADPS` are supplied by an external C++ compile definition unavailable to this Python loader, use the native PhysX actuator's source-backed steering value **with a RuntimeWarning and explicit unverified-alias provenance**. Do not claim independent source-plant parity for that unresolved alias.
- Reject unknown unresolved names, circular aliases, nonfinite constants, and any explicit numeric mismatch with the native PhysX actuator.
- No change to the IPOPT physical model, friction constraints, phase-I numerical overlay, or PhysX binary. No evidence of a completed sub-4.9-second lap.
- Package unit tests: 27 passing, including six alias/headers/unknown-name cases.

**Update in place:** Extract this ZIP from the repository root (`unzip -o ... -d .`). Only the `SDU_Apex_IPOPT_PhysX_MinTime_v2/` code directory is in the archive. Existing `.venv`, compiled `native_physx_vehicle_step`, `_restored_git_history/`, `_working_solver_limited_memory/`, and prior `optimized_*` outputs are excluded and preserved.
