# v2.3.9 — Native/symbolic suspension frame audit

The v2.3.8 per-wheel file shows the native left-right load transfer and the
symbolic load transfer have correlations approximately -0.9993 at both axles.
In 97.9% of track nodes their left/right sign is opposite. The native and
symbolic four-wheel load sums also differ by approximately 3.22 N on average.

The native source state uses an x-forward/y-left/z-up frame while the Unity
wheel mount coordinates are x-right/y-up/z-forward. The original flat-ground
suspension travel expression uses `sin(roll)*mount_x`, a possible lateral
reflection mismatch. The new diagnostic compares native loads against the
existing model and several physically motivated **sign-only candidates**:
reflect the mount lateral x coordinate, negate measured roll and roll rate,
negate pitch and pitch rate, and negate both axes. It requires reflected mount
and reversed-roll predictions to agree to numerical precision.

The script also reports whether applying each candidate's loads to the
unchanged tire-force law reduces reconstructed lateral-acceleration mismatch.
This is **not a fitted correction** and is **not an optimizer physics change**.
It runs no IPOPT solve or PhysX replay. The working cached solver and historical
sources are preserved untouched. We will change the model only if the native
per-wheel data supports a specific coordinate correction.

Run: `bash diagnose_suspension_v239.sh`, inspect
`suspension_frame_v239/summary.json` and `per_node.csv`.
