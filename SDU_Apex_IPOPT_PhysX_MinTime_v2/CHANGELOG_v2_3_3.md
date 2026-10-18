# v2.3.3 — Phase-I CasADi numeric snapshot and evaluation performance

## User-observed problem

A two-iteration debug run of v2.3.2 remained silent before the IPOPT banner for over 120 seconds. `faulthandler` captured the Python thread repeatedly inside CasADi `Opti.initial()` and `Opti.value()` called by the generated working solver's `snapshot_current` and `phase1_measure` functions. The user's earlier run entered IPOPT but advanced only one iteration during a five-minute window.

## Fix

The historical source and user inputs are unchanged. Its generated working solver now:

- Caches the initialized decision variables (states, controls, algebraics and elastic defects) numerically instead of reconstructing `opti.initial()` for every variable.
- Compiles **one** CasADi numeric function of `opti.x` and `opti.p` to evaluate the exact original constraint graph, its current bounds, original dynamics residuals and elastic objective.
- Evaluates candidate checkpoints with one `opti.debug.value(opti.x)` per callback instead of separate `Opti` queries for the full constraint vector, every state and every dynamic residual.
- Rechecks original hard constraints and strict zero-elastic-defect bounds. The physical vehicle equations, tire laws, and minimum-time objective are **not modified**.
- Reconstructs the byte-exact v2.3.2 cache for safe in-place upgrading, still refusing unknown local edits.

The generated optimizer prints `Phase-I numeric seed diagnostics ready; starting IPOPT` when it has finished numerical initialization. This separates pre-IPOPT overhead from the IPOPT iteration loop.

## One-command diagnostic

After installing over `SDU_Apex_IPOPT_PhysX_MinTime_v2/`, run:

```bash
cd ~/Documents/GitHub/SDU-Apex-Autodrive/SDU_Apex_IPOPT_PhysX_MinTime_v2
bash diagnose_phase1_v233.sh
```

This uses the existing saved PhysX warm start, disables PhysX **validation** for the short debug run, limits IPOPT to 12 feasibility iterations and stops after 180 seconds. Output is saved to `ipopt_v233_debug.log`. It does not certify a physical or faster lap.

## Verification scope

Tested numerical-cache ordering, constraint residuals, live zero-defect slack bounds, generated-code syntax and cache reconstruction with toy CasADi NLPs. No complete Porto IPOPT solve, native PhysX replay, or Unity run was performed by the packager. Solver speedup on the user's hardware remains to be measured.
