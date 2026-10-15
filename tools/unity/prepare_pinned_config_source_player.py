#!/usr/bin/env python3
"""Create a persistent Unity-source player project with pinned car settings."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "external/autodrive-unity-source"
DESTINATION = (
    REPO / "live_runs/unity_exact_binary_audit_20261012/"
    "source_project_pinned_config"
)
PREFAB_RELATIVE = Path("Assets/Prefabs/F1TENTH/F1TENTH.prefab")
IMAGE_DIGEST = "sha256:0511fb6d6db7f31ec0d20787c0498eb05f3bd45ae6b9683165e8680b61e9b159"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    patch_existing = "--patch-existing" in sys.argv[1:]
    if DESTINATION.exists() and not patch_existing:
        raise SystemExit(f"Refusing to overwrite existing project: {DESTINATION}")
    if patch_existing and not DESTINATION.is_dir():
        raise SystemExit(f"Cannot patch missing project: {DESTINATION}")

    for required in (SOURCE / "Assets", SOURCE / "Packages", SOURCE / "ProjectSettings"):
        if not required.is_dir():
            raise SystemExit(f"Missing Unity project directory: {required}")

    if not patch_existing:
        DESTINATION.mkdir(parents=True)
        for name in ("Assets", "Packages", "ProjectSettings"):
            shutil.copytree(SOURCE / name, DESTINATION / name)
    practice_resource = SOURCE / "Assets/Resources/OfflinePracticeTrack"
    if not practice_resource.is_dir():
        raise SystemExit(f"Missing extracted practice-track resource: {practice_resource}")
    (DESTINATION / "Assets/Resources").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "Assets/Resources.meta",
                 DESTINATION / "Assets/Resources.meta")
    shutil.copytree(
        practice_resource,
        DESTINATION / "Assets/Resources/OfflinePracticeTrack",
        dirs_exist_ok=True)
    shutil.copy2(SOURCE / "Assets/Resources/OfflinePracticeTrack.meta",
                 DESTINATION / "Assets/Resources/OfflinePracticeTrack.meta")
    for name in ("BatchModeSocketBootstrap.cs", "BatchModeSocketBootstrap.cs.meta"):
        shutil.copy2(REPO / "tools/unity" / name,
                     DESTINATION / "Assets/Scripts" / name)
    shutil.copy2(
        SOURCE / "Assets/Scripts/OfflineCommandProfileRunner.cs",
        DESTINATION / "Assets/Scripts/OfflineCommandProfileRunner.cs")
    shutil.copy2(
        SOURCE / "Assets/Scripts/OfflineCommandCollisionMonitor.cs",
        DESTINATION / "Assets/Scripts/OfflineCommandCollisionMonitor.cs")
    for name in ("BuildWheelDynamicsCapture.cs", "BuildWheelDynamicsCapture.cs.meta"):
        shutil.copy2(
            SOURCE / "Assets/Scripts/Editor" / name,
            DESTINATION / "Assets/Scripts/Editor" / name)

    prefab = DESTINATION / PREFAB_RELATIVE
    original = prefab.read_text(encoding="utf-8")
    controller_blocks = [
        block for block in original.split("--- !u!")
        if "driveType: 5" in block and "MotorTorque: 85.6" in block
        and "DrivingMode: 0" in block
    ]
    if len(controller_blocks) != 1:
        raise SystemExit(f"Expected one public controller prefab block, found {len(controller_blocks)}")

    block = controller_blocks[0]
    required_fields = {
        "COM: {x: -0.00008, y: 0.06434, z: -0.00468}": "exact COM",
        "SteeringRate: 183.346": "exact steering rate",
        "Wheelbase: 324": "exact Ackermann wheelbase",
        "TrackWidth: 236": "exact track width",
        "WheelRadius: 0.0325": "controller radius",
        "SteeringLimit: 30": "steering limit",
    }
    for field, description in required_fields.items():
        if field not in block:
            raise SystemExit(f"Public prefab does not contain expected {description}: {field}")

    explore_scene = DESTINATION / "Assets/Scenes/RoboRacer - Sim Racing - Explore.unity"
    scene_text = explore_scene.read_text(encoding="utf-8")
    for field, value in (("DrivingMode", "1"), ("MotorTorque", "428")):
        if f"propertyPath: {field}\n      value: {value}" not in scene_text:
            raise SystemExit(f"Explore scene does not override {field} to pinned value {value}")

    # Preserve the GUI scene's disabled socket objects. The connection button
    # belongs to an inactive Menu subtree in batch mode, so the batch-only
    # bootstrap lives on the active UI root and references the existing toggle.
    bootstrap_file_id = "7654321009876543210"
    bootstrap_component = (
        f"  - component: {{fileID: {bootstrap_file_id}}}\n")
    scene_text = scene_text.replace(bootstrap_component, "")
    scene_text = re.sub(
        rf"(?ms)^--- !u!114 &{bootstrap_file_id}\n.*?(?=^--- !u!|\Z)",
        "", scene_text, count=1)
    if bootstrap_file_id in scene_text:
        raise SystemExit("Batch bootstrap file ID is unexpectedly duplicated")

    def set_game_object_active(scene: str, file_id: str, active: bool) -> str:
        marker = f"--- !u!1 &{file_id}\n"
        start = scene.find(marker)
        end = scene.find("--- !u!", start + len(marker))
        if start < 0 or end < 0:
            raise SystemExit(f"Explore scene is missing GameObject {file_id}")
        block = scene[start:end]
        state = "1" if active else "0"
        if block.count("  m_IsActive: ") != 1:
            raise SystemExit(f"GameObject {file_id} has an unexpected active-state field")
        block = re.sub(
            r"(?m)^  m_IsActive: [01]$", f"  m_IsActive: {state}", block, count=1)
        return scene[:start] + block + scene[end:]

    scene_text = set_game_object_active(scene_text, "971678322", False)
    scene_text = set_game_object_active(scene_text, "3857243730762531608", False)

    # The public Explore scene retains a stripped front-camera reference. Its
    # Socket.EmitTelemetry callback calls FrameGrabber without a null/texture
    # check, so keep image serialization disabled in this kinematics-only
    # development player. Motion telemetry and all vehicle components remain.
    socket_script_marker = "--- !u!114 &971678324\n"
    socket_script_start = scene_text.find(socket_script_marker)
    socket_script_end = scene_text.find("--- !u!", socket_script_start + len(socket_script_marker))
    if socket_script_start < 0 or socket_script_end < 0:
        raise SystemExit("Explore scene Socket component anchor changed")
    socket_script_block = scene_text[socket_script_start:socket_script_end]
    camera_reference = "  FrontCameras:\n  - {fileID: 1105803435}\n  RearCameras: []"
    if socket_script_block.count(camera_reference) == 1:
        socket_script_block = socket_script_block.replace(
            camera_reference, "  FrontCameras: []\n  RearCameras: []", 1)
        scene_text = (scene_text[:socket_script_start] + socket_script_block
                      + scene_text[socket_script_end:])
    elif socket_script_block.count("  FrontCameras: []\n  RearCameras: []") != 1:
        raise SystemExit("Explore Socket camera serialization changed unexpectedly")

    ui_object = "--- !u!1 &419672553\n"
    ui_start = scene_text.find(ui_object)
    ui_end = scene_text.find("--- !u!", ui_start + len(ui_object))
    if ui_start < 0 or ui_end < 0:
        raise SystemExit("Explore scene is missing the active User Interface root")
    ui_block = scene_text[ui_start:ui_end]
    if "  m_Name: User Interface\n" not in ui_block or "  m_IsActive: 1\n" not in ui_block:
        raise SystemExit("User Interface root is not the expected active scene object")
    first_ui_component = re.search(r"(?m)^  - component: \{fileID: \d+\}$", ui_block)
    if first_ui_component is None:
        raise SystemExit("User Interface root has no component list")
    ui_block = (ui_block[:first_ui_component.end()] + "\n" + bootstrap_component.rstrip()
                + ui_block[first_ui_component.end():])
    scene_text = scene_text[:ui_start] + ui_block + scene_text[ui_end:]

    socket_component_marker = "--- !u!114 &419672555\n"
    socket_component = (
        f"--- !u!114 &{bootstrap_file_id}\n"
        "MonoBehaviour:\n"
        "  m_ObjectHideFlags: 0\n"
        "  m_CorrespondingSourceObject: {fileID: 0}\n"
        "  m_PrefabInstance: {fileID: 0}\n"
        "  m_PrefabAsset: {fileID: 0}\n"
        "  m_GameObject: {fileID: 419672553}\n"
        "  m_Enabled: 1\n"
        "  m_EditorHideFlags: 0\n"
        "  m_Script: {fileID: 11500000, guid: 830b61a23c624f76b16a90c1de5039c4, type: 3}\n"
        "  m_Name: \n"
        "  m_EditorClassIdentifier: \n"
        "  socketIO: {fileID: 3857243730769114872}\n"
        "  connection: {fileID: 541364191}\n")
    if scene_text.count(socket_component_marker) != 1:
        raise SystemExit("User Interface component anchor changed")
    scene_text = scene_text.replace(
        socket_component_marker, socket_component + socket_component_marker, 1)
    explore_scene.write_text(scene_text, encoding="utf-8")

    # The candidate must retain the source's read-only development capture hook.
    hook = DESTINATION / "Assets/Scripts/OpenPlaneWheelDynamicsCapture.cs"
    controller = DESTINATION / "Assets/Scripts/VehicleController.cs"
    build_script = DESTINATION / "Assets/Scripts/Editor/BuildWheelDynamicsCapture.cs"
    if not hook.is_file() or "OpenPlaneWheelDynamicsCapture.Capture(this)" not in controller.read_text(encoding="utf-8"):
        raise SystemExit("Development capture hook is not wired into the copied source")
    if not build_script.is_file():
        raise SystemExit("Development player build entry point is missing")

    manifest = {
        "purpose": "Instrumented development player; not a replacement competition image",
        "source_commit": "75b7df5216524a3c6b3150d698139190c6c44f8f",
        "reference_competition_image_digest": IMAGE_DIGEST,
        "controller_configuration": {
            "prefab_defaults": {"MotorTorque": 85.6, "DrivingMode": 0},
            "active_explore_scene_overrides": {"MotorTorque": 428.0, "DrivingMode": 1},
            "pinned_player_values": {"MotorTorque": 428.0, "DrivingMode": 1},
        },
        "vehicle_physics_or_serialization_changed": False,
        "batch_mode_network_bootstrap": (
            "Socket and SocketIO begin inactive as in the GUI scene. A "
            "development-only component on the active UI root invokes the "
            "existing GUI toggle after scene startup and connects after "
            "Socket.Start registers callbacks; no vehicle or physics fields "
            "changed."),
        "development_sensor_scope": (
            "Camera image arrays are empty because the source Explore scene "
            "contains a stripped camera reference that crashes FrameGrabber; "
            "motion, IMU, encoder, actuator, and LiDAR telemetry are retained."),
        "recorder": "read-only DEVELOPMENT_BUILD capture, 1 kHz, environment-variable gated",
        "recorder_capacity_samples": 300000,
        "files": {
            str(PREFAB_RELATIVE): sha256(prefab),
            "Assets/Scenes/RoboRacer - Sim Racing - Explore.unity": sha256(explore_scene),
            "Assets/Scripts/VehicleController.cs": sha256(controller),
            "Assets/Scripts/OfflineCommandProfileRunner.cs": sha256(
                DESTINATION / "Assets/Scripts/OfflineCommandProfileRunner.cs"),
            "Assets/Scripts/OfflineCommandCollisionMonitor.cs": sha256(
                DESTINATION / "Assets/Scripts/OfflineCommandCollisionMonitor.cs"),
            "Assets/Scripts/OpenPlaneWheelDynamicsCapture.cs": sha256(hook),
            "Assets/Scripts/BatchModeSocketBootstrap.cs": sha256(
                DESTINATION / "Assets/Scripts/BatchModeSocketBootstrap.cs"),
            "Assets/Scripts/Editor/BuildWheelDynamicsCapture.cs": sha256(build_script),
            "Assets/Resources/OfflinePracticeTrack/PracticeCourse.obj": sha256(
                DESTINATION / "Assets/Resources/OfflinePracticeTrack/PracticeCourse.obj"),
            "Assets/Resources/OfflinePracticeTrack/PracticeCourse.obj.meta": sha256(
                DESTINATION / "Assets/Resources/OfflinePracticeTrack/PracticeCourse.obj.meta"),
        },
        "acceptance": "Do not use telemetry as pinned-player labels until matched-command public-motion A/B passes.",
    }
    (DESTINATION / "pinned_config_overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"project": str(DESTINATION), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
