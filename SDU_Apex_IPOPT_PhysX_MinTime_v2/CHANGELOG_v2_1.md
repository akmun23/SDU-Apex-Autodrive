# v2.1 Phase-I feasibility correction (2026-10-17)

- Fixed the restored full-lap IPOPT program inheriting a captured-motion squared-distance cap of 25 despite no historical Unity-state capture being available. The reported failure had objective 31.7362664 and max violation 6.7362661 (~31.7362664 - 25). Default Phase-I cap is now 200, user adjustable. The nonlinear vehicle, map, contact, periodicity, and minimum-time constraints are unchanged.
- Failures print named maximal constraint violations and retain JSON for analysis. Added offline report diagnostics.
- Preserved original solver source byte-for-byte in Git recovery; change is passed via the upstream `feasibility_trust_region` argument.
- No PhysX/Unity execution or minimum-time convergence claimed. No new physics model or empirical results supplied.
