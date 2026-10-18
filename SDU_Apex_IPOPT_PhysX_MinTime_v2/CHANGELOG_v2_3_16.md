# v2.3.16 — Full native contact-point velocity in wheel-slip dynamics

- Fix omitted roll/pitch rate contribution from angular velocity cross the
  PhysX source-defined wheel force-application point. Front contact arm is
  0.063435m below COM; rear 0.073340m. No parameter fitting or soft dynamics.
- Symbolic OCP passes its actual roll/pitch rate states into tire_forces_casadi;
  optional defaults maintain compatibility with older diagnostic calls.
- Source-consistent algebraic seed projection also passes roll/pitch rates.
- Existing native contact telemetry + archived 143-point warm-start implies
  lateral slip RMS 0.000547 -> 0.000059 rad; lateral tire force RMS
  0.577 -> 0.068 N; 25ms lateral acceleration RMS 0.719 -> 0.219 m/s2.
- Distinct validation against source-matched CasADi function required before
  IPOPT; 40-iteration Phase-I diagnostic follows, without brake nodes.
- Historic optimizer and PhysX source preserved; unknown manual cache edits
  fail closed. Feasible minimum-time and Unity/native validation NOT claimed.
