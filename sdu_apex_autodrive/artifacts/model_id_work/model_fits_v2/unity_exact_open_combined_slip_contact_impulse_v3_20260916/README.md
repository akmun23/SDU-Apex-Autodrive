# Exact Unity open-plane contact-impulse audit

- Build: `exact_open_contact_impulse_v3_20260916`
- Unity: `2022.3.52f1` Linux player
- Run mode: `-batchmode`; no `-nographics`
- Experiment: `combined_slip_matrix_v1`
- Duration: `70.028 s`
- Wheel trace: 70,028 fixed-step rows at 1 kHz
- Collision trace: 290,572 callbacks with per-contact impulse/separation
- Physics/controller/scene behavior: unchanged; diagnostic observations only

All `10,918` `Chassis-1-solid1`--`Floor` callbacks had positive separation
(`0.0183--0.0200 m`) and zero `ContactPoint.impulse` and `Collision.impulse`.
Only one wheel-child callback had a nonzero contact impulse. The chassis
callbacks therefore do not identify an applied body-force channel.
