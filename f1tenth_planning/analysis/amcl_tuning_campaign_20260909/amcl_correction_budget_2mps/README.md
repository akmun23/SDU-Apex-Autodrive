# AMCL correction-budget experiment

This is a real graphical AutoDRIVE ICRA-compete run with `-batchmode` and no
`-no-graphics`, using the 2 m/s controller and the successful 0.60 m minimum
lookahead from `turn_entry_lookahead_2mps`.

The production AMCL correction gain is 1.0 and the accumulated local XY
correction budget is 0.50 m. The values are based on the preceding run's
accepted 0.267 m scan correction that was suppressed by the old 0.12 m cap.
All other runtime parameters remain unchanged. Ground truth is diagnostics
only and is not supplied to the estimator or controller.
