# Optional simulator-side patches

The repository contains source-side patches and tools for a rebuilt Unity
player. They cannot change a prebuilt IL2CPP player by themselves.

The ROS bridge requests a 40 Hz command/polling cadence, but that is not a
claim that the simulator produces 40 Hz physics or LiDAR data. The current
prebuilt compete player must be measured from source timestamps; the retained
live run measured approximately 20 Hz on the track. No duplicate or synthetic
samples are accepted as a substitute.

If Unity source becomes available, apply the patches to the matching
AutoDRIVE-Simulator source branch and rebuild the Linux player:

```bash
git apply simulator/patches/0001-clean-40hz-telemetry.patch
git apply simulator/patches/0002-parallel-lidar-raycasts.patch
```

The first patch removes camera/intensity serialization from the numeric path
and adds timing metadata. The second moves independent LiDAR raycasts off the
Unity main thread. Neither patch is active in the prebuilt competition image
until a rebuilt player is installed.

Run the HDRP player in graphical batch mode. Do not use `-no-graphics`:

```bash
./AutoDRIVE\ Simulator.x86_64 -batchmode \
  -ip 127.0.0.1 -port 4567 -logFile /tmp/autodrive.log
```
