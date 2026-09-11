# Optional simulator-side patches

The repository contains source-side patches and tools for a rebuilt Unity
player. They cannot change a prebuilt IL2CPP player by themselves.

The ROS bridge requests a 40 Hz command/polling cadence using a bounded
four-request pipeline; the native Unity player needs more than one request in
flight to avoid halving the round-trip rate. The supported local player is the rebuilt source player at
`/home/akselmo/Documents/GitHub/AutoDRIVE/Builds/AutoDRIVE-Simulator.x86_64`.
Its native LiDAR and odometry streams were measured at approximately 40 Hz on
the compete track. The older prebuilt competition image is not the 40 Hz
workflow; no duplicate or synthetic samples are accepted as a substitute.

If Unity source becomes available, apply the patches to the matching
AutoDRIVE-Simulator source branch and rebuild the Linux player:

```bash
git apply simulator/patches/0001-clean-40hz-telemetry.patch
git apply simulator/patches/0002-parallel-lidar-raycasts.patch
git apply simulator/patches/0003-reusable-lidar-buffers.patch
```

The first patch removes camera/intensity serialization from the numeric path
and adds timing metadata. The second moves independent LiDAR raycasts off the
Unity main thread. The third reuses the LiDAR job buffers between scans. The
compete scene also disables the optional laser-point visualization, which is
not needed for the GUI and avoids per-scan visualization allocations.
These patches are not active in the prebuilt competition image until a rebuilt
player is installed.

The repository-level [startup guide](../STARTUP_GUIDE.md) is the supported way
to start the simulator and controller together. If the player is started
manually, run the HDRP player with its normal GUI. Do not use `-batchmode` or
`-no-graphics`:

```bash
./AutoDRIVE-Simulator.x86_64 \
  -ip 127.0.0.1 -port 4567 -logFile /tmp/autodrive.log
```

The verified player includes the exact official SRL 2026 ICRA compete mesh as
both official submeshes, with the official materials and matching collider,
because the Linux Unity editor cannot import the original SketchUp prefab
directly.
