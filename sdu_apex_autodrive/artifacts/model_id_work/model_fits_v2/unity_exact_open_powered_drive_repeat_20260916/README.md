# Exact Unity powered-drive repeat

This is a diagnostic-only open-plane capture from the disposable Unity
identification player. It uses the unchanged vehicle physics and records at
the internal 1 kHz fixed-step rate. The positive-drive throttle sequence was
repeated across the reachable speed range up to the project ceiling of 16
m/s; no `-nographics` option was used.

The CSV is 57 MB and contains 51,018 rows. The companion audit is
`../unity_exact_open_powered_drive_repeat_wheel_drive_audit_20260916.json`.
The data is ground truth for offline identification only. It does not alter
the simulator or promote a model into production MPC.
