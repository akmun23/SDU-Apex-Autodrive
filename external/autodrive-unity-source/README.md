# Unity vehicle reference subset

This directory is deliberately not a runnable Unity project. It retains only
the RoboRacer vehicle prefab, its race-scene wiring, the vehicle/command/sensor
scripts relevant to modeling, and the Unity physics timing settings. The
simulator itself is run from its Docker image; this subset is a source
reference, not a second simulator or a build input.
