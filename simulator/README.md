# AutoDRIVE simulator telemetry patch

The deployed Docker image contains a prebuilt Unity IL2CPP player, not the
Unity project. The source-side change therefore lives as an explicit patch
against the official `AutoDRIVE-Simulator` source branch.

This patch makes the numeric telemetry path suitable for the 40 Hz gate:

- camera capture and camera fields are disabled by default at the simulator,
  before JPEG/base64 serialization;
- unused LIDAR intensity serialization is disabled by default;
- the API can explicitly keep both settings disabled in every `Bridge`
  command;
- each packet carries Unity `Simulation Time`, `Simulation Frame`, and a
  telemetry sequence for diagnosing repeated physics states and bursty
  delivery.

The ROS API change is already in
`sdu_apex_autodrive/sdu_apex_autodrive/bridge_40hz.py`. It accepts packets with
no camera field, preserves the numeric topic contract, and includes the Unity
metadata in `/autodrive/roboracer_1/bridge_packet_timing`.

## Build the simulator player

A Unity build is required before this can affect the running simulator image.
The source branch currently targets Unity `2022.3.52f1`.

```bash
git clone --branch AutoDRIVE-Simulator --depth 1 \
  https://github.com/Tinker-Twins/AutoDRIVE.git autodrive-simulator
cd autodrive-simulator
git apply /workspace/src/simulator/patches/0001-clean-40hz-telemetry.patch
```

Open the project with Unity `2022.3.52f1` and build the Linux x86_64 player.
Use the resulting player directory as the simulator image payload, or build a
new image and set `AUTODRIVE_SIMULATOR_IMAGE` when starting this workspace.

The current prebuilt `2026-iros-explore` image cannot be changed by the ROS
container alone: it will continue to transmit its compiled camera path until a
patched player is installed.
