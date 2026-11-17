# Competition MPC graphical tuning

Run `tools/tuner/run_competition_tuner.sh` or use the SDU Apex Competition Tuner desktop shortcut. The launcher runs `tools/tuner/competition_tuner.py`, which starts the official graphical IROS Unity simulator and the competition controller. It does not launch a development controller.

By default, the controller uses `sdu-apex-autodrive:competition`, the image built by `tools/build_competition.sh`. See [competition-runtime.md](competition-runtime.md) for build and runtime details.

## Tuning workflow

1. Set the host to Performance power mode and log in to the graphical desktop.
2. Click Start Unity + MPC, then Connect in the Unity window. AMCL and MPC start only after fresh sensor and localization data arrive.
3. Watch the car drive and change the weights in the tuner. Apply changes only when the car is stopped.
4. Click Apply & Restart MPC to save a new YAML snapshot and restart only the controller. The Unity window stays open. Click Connect if the bridge is disconnected, then press Unity's Restart to reset the vehicle.
5. Repeat as needed. Click Stop Both, close the tuner, or press Ctrl+C in its launching terminal to stop the two containers owned by the tuning session.

Unity's Restart button does not reload MPC parameters. The controller declares these ROS 2 parameters at startup and does not update them through runtime parameter callbacks. The tuner restarts the complete controller process to load the parameters and avoid reusing a solver warm start after changing the objective.

If Unity requires reconnection after a controller restart, press Connect again. Unity may keep applying an old command until the bridge reconnects. Apply changes only when the car is stopped.

## Files and controls

- Base test settings: `tools/tuner/profiles/competition_gui.yaml` (overspeed 50, overspeed curvature 1.25, heading 2500, heading curvature 0.5, steering slew 500).
- Active settings: `~/.local/share/sdu-apex/competition-tuner/active.yaml`.
- Automatically archived configurations: `~/.local/share/sdu-apex/competition-tuner/history/`.
- Use Load Profile and Save Profile As in the GUI. Saving alone does not change the running MPC.
- Controller launcher event file, if needed: `~/.local/share/sdu-apex/competition-tuner/controller_launcher.log`. No permanent development diagnostics are added to the competition image.
- To use a different prebuilt image, run `tools/tuner/run_competition_tuner.sh --image IMAGE_TAG`. YAML-only weight changes do not require an image rebuild.

The controller reads the active YAML through a read-only bind mount at the installed ROS parameter path. `tools/run_competition.sh` uses this override only when `SDU_APEX_MPC_CONFIG_OVERRIDE` is set. Both test containers have a unique tuning session ID label. Shutdown stops only containers with that label. The tuner does not automatically replace or stop an external container using the same name.

## Limitations and verification

The 40 Hz controller can have corner tracking and feasibility failures. These GUI controls do not guarantee collision-free laps. The tuning tool does not send simulator reset or ground-truth commands from the competition controller. Use Unity's visible Restart button to reset the car. Unity GUI frames and communication can differ from batchmode, so evaluate candidates in graphical mode before retaining them.

To validate the container lifecycle without pressing Connect:

```bash
tools/tuner/run_competition_tuner.sh --smoke-test
```

This checks that restarting the controller keeps the same Unity container and that stopping removes both owned containers and frees bridge port 4567.
