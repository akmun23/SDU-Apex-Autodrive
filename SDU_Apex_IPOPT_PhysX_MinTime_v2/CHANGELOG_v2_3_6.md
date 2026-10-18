# v2.3.6: forward-speed wheel substep source bug + projected-seed diagnostic

User's v2.3.5 50-iteration failure report shows a single accepted, hard-constraint-
feasible backtrack: 1.0647761866538858 -> 1.0627929937238791 strict max
original-dynamics defect. This is only ~0.19%; no fully feasible minimum-time
solution has been obtained. The diagnostic script used the flag
`--no-seed-force-projection`, so the observed lateral-force defect reflects
measured native accelerations/loads, not a symbolic-algebraic projected seed.

Source-derived OCP physics correction: the historical recovered source used
`interval_start_forward_speed = x[0, k]` inside the PhysX NoDrive wheel
substep branch. The actual OCP state order is `x[0]=ey_m`, `x[2]=u_mps`.
Consequently the former code selected 5 internal substeps for most track states
even when `u>=5 m/s`, whereas the native source uses 1 substep in that speed
regime. In the GENERATED working solver only, repair it to `x[2,k]` using a
uniquely matched fail-closed source patch. The historical recovered solver file
and PhysX native binary are not altered. This is a real correction to the
symbolic wheel dynamics above 5 m/s, not a mere numerical tolerance change.

`diagnose_phase1_v236.sh` tests a *projected* native physical-state initialization
for 50 Phase-I IPOPT iterations (the previous v235 test turned that projection
off). It deliberately runs drive-only and skips native replay to isolate the
feasibility mismatch. It is NOT a brake-aware minimum-time result.

Automatically recognizes and upgrades the exact v2.3.5 cache; unrecognized
local edits remain protected. Existing `.venv`, compiled native PhysX binary,
restored source history, warm-start NPZ and prior results survive extraction.
No full IPOPT solve or native/Unity validation was run by the packager.
