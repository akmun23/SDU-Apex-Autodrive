# v2.4.0 — Complete source-to-convergence review and Phase-I workflow overhaul

- Freeze the vehicle equations at **v2.3.16**, including full yaw/roll/pitch tire contact velocity, corrected native force/slip coordinates, and corrected suspension mounts. NO new conjectural physical equation changes.
- Export the best *hard-feasible* but strictly dynamics-infeasible Phase-I checkpoint with all physical variables and spatial phase lengths; resume additional IPOPT solves from it under hard safety constraints. Stop on stagnation, never certify a nonzero defect.
- Compare two initializations (native unprojected vs symbolic algebraic projection) under identical Phase-I options, up to two additional restarts each; optional exact-Hessian experiment. Distinct reports/logs for each attempt, ranked JSON/Markdown summary and upload ZIP.
- Correct v2.3.16 Phase-I diagnostic output inadvertently directed at the v2.3.15 directory; fix provenance labeling and support trusted between-Hessian cache upgrades.
- Full 2,100-word engineering audit documenting model boundaries, numerical behavior, false equivalences, code risks, and formulation-level next moves.
- Source historical cache protected; user .venv, native PhysX executable, original native seed, and prior results are **never packaged for overwrite**.
- 143 tests passed (runtime-independent). The original full external repository, IPOPT high-dimensional solve and native validation are not available in this sandbox; results must be verified locally.
