# Development-to-graphical latency matching (IROS)

Keep race-completion and lap-time results separate for GUI and development
batchmode until measured transport and actuator response are comparable.
Qualification also requires identical frame/physics, initialization, AMCL and
MPC configuration, plus checks at the same failure corner. A fixed delay cannot
reproduce every Unity GUI scheduling effect.

The official IROS image digest is
`sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`.
For IROS runs with the practice image digest `sha256:b4bb...` in
`run_manifest.json`, that manifest value is incorrect: the corresponding
`simulator.log` files confirm that the launched image is IROS.
`tools/sim/run_trial.sh` records the correct image digest for the selected track.

## Development-only controls

- `SDU_APEX_DEV_SENSOR_DELAY_MS=0..75` delays ROS sensor publication after
  the simulator request timestamp has been assigned. This models older sensor
  information arriving at the MPC. Additional serialization within the packet
  callback can change packet delivery cadence at high delays.
- `SDU_APEX_DEV_ACTUATOR_DELAY_MS=0..75` delays actuator setpoint selection at
  the existing 40 Hz Bridge request clock. This is a zero-order hold on the
  latest old-enough ROS command; actual added delay is quantized by the Bridge
  request schedule. It does not delay or modify Unity's physics integrator.
- Both default to 0. The official competition image does not read these
  variables and remains unchanged. Reset and socket reconnect flush command
  queues so controls from one run do not carry into another.

Development-only test (do not run alongside graphical Unity):

```bash
cd ~/Documents/GitHub/SDU-Apex-Autodrive
SDU_APEX_DEV_SENSOR_DELAY_MS=5 \
SDU_APEX_DEV_ACTUATOR_DELAY_MS=15 \
SDU_APEX_TRIAL_TOTAL_LAPS=2 \
tools/sim/run_trial.sh competition dev_latency_test_001
```

Each `live_runs/<id>/latency_profile.json` records the selected artificial
sensor and command delays. The Bridge request timer stays at 40 Hz.

## Calibration

Record both runtimes with `tools/analysis/measure_transport_latency.py`. The
script observes IMU, odometry, map pose, MPC commands, exact Bridge requests,
and Unity-reported actuator feedback. It reports sensor age (including
sample/request age), interarrival timing, and effective
MPC-command-to-feedback alignment.

To start a 15-second read-only capture inside the development container:

```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
python3 /workspace/src/tools/analysis/measure_transport_latency.py --seconds 15 \
  --json-output /workspace/src/live_runs/<id>/transport_latency.json
```

For the competition container, pipe the same script into `python3 -` from the
host; add `--wait-for-imu` when a human must press Unity's Connect button.
The reporter does not change any MPC command and uses permitted ROS topics.

```bash
python3 tools/analysis/compare_transport_profiles.py \
  --development live_runs/<id>/transport_latency.json \
  --graphical /tmp/sdu_gui_transport_parity_20261110.log
```

A passing gate checks that observed sensor age, 40 Hz delivery, command
continuity, and command-to-feedback alignment are comparable. It does not prove
that nonlinear feasibility, physical contacts, slip or lap completion match.
Graphical qualification is the final acceptance criterion. A development-only
completion does not count as a successful graphical race.

## 10 November 2026 baseline

- Unmodified development batchmode: IMU age p50 28.65 ms, p95 29.82 ms; in
  that 10-second capture IMU rate 34.0 Hz, MPC commands 36.8 Hz. Two lap
  times: 7.517 and 7.055 seconds.
- Development batchmode with +5 ms sensor and +15 ms actuator selection delay:
  IMU age p50 34.13 ms, p95 34.86 ms; IMU/MPC about 40 Hz. Two lap times:
  7.695 and 7.002 seconds; estimated command-to-feedback alignment: 35 ms.
  This delay pair is an experimental hypothesis without graphical qualification.
- Graphical measurements with a different instrument give IMU p95 ages of
  approximately 25–35 ms. These measurements are insufficient to establish
  parity. A GUI capture with the same instrument is necessary.
