#!/usr/bin/env python3
"""Replay only encoders/IMU through the image-built production odometry node.

The image must contain source and configuration hashes identical to this
worktree. The bag is mounted read-only, ROS is isolated from host/simulator
networks, TF publication is disabled, and only derived odometry/diagnostics are
written to the requested output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as analysis


DEFAULT_IMAGE = "sdu-apex-autodrive:dev-practice-speed-screen-20260927"
SOURCE_FILES = (
    "f1tenth_localization/src/sensor_odometry_node.cpp",
    "f1tenth_localization/src/odometry_observer.cpp",
    "f1tenth_localization/src/sensor_packet_assembler.cpp",
    "f1tenth_localization/include/f1tenth_localization/odometry_observer.hpp",
    "f1tenth_localization/include/f1tenth_localization/sensor_packet_assembler.hpp",
    "f1tenth_localization/config/sensor_odometry.yaml",
)
SENSOR_TOPICS = (
    "/autodrive/roboracer_1/left_encoder",
    "/autodrive/roboracer_1/right_encoder",
    "/autodrive/roboracer_1/imu",
    "/autodrive/roboracer_1/steering",
    "/autodrive/roboracer_1/throttle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _docker_environment(repo_root: Path) -> dict[str, str]:
    setup = subprocess.run(
        ["bash", "-c", 'source "$1"; env -0', "replay-env",
         str(repo_root / "tools/docker_env.sh")],
        check=True, capture_output=True,
    )
    environment = os.environ.copy()
    for item in setup.stdout.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            environment[key.decode()] = value.decode()
    return environment


def _docker(image: str, args: list[str], env: dict[str, str],
            capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], env=env, check=True, text=True,
                          capture_output=capture_output)


def _verify_image_sources(image: str, repo_root: Path,
                          docker_env: dict[str, str]) -> str:
    image_id = _docker(
        image, ["image", "inspect", "--format", "{{.Id}}", image],
        docker_env, capture_output=True,
    ).stdout.strip()
    image_hash_command = "cd /workspace/src && sha256sum " + " ".join(
        shlex.quote(name) for name in SOURCE_FILES)
    image_hashes = _docker(
        image,
        ["run", "--rm", "--network=none", "--entrypoint", "/bin/bash",
         image, "-lc", image_hash_command],
        docker_env, capture_output=True,
    ).stdout.splitlines()
    built = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
             for line in image_hashes}
    mismatches = []
    for relative in SOURCE_FILES:
        workspace_hash = _sha256(repo_root / relative)
        if built.get(relative) != workspace_hash:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(
            "odometry source/config differs from the installed image snapshot: "
            + ", ".join(mismatches))
    return image_id


def _container_script(integrate_lateral_acceleration_in_turn: bool = False,
                      wheel_burst_catchup_accel_mps2: float | None = None,
                      wheel_speed_window_s: float | None = None,
                      wheel_update_beta: float | None = None,
                      turn_current_packet_max_decrease_mps: float | None = None,
                      turn_speed_bias_yaw_rate_abs_mps: float | None = None,
                      turn_speed_bias_max_mps: float | None = None,
                      turn_slip_speed_bias_fade_start_mps: float | None = None,
                      turn_slip_speed_bias_fade_end_mps: float | None = None,
                      start_offset_s: float | None = None,
                      duration_wall_s: float | None = None,
                      disable_turn_speed_bias_lateral_accel_gate: bool = False,
                      disable_coherent_packet_pose_velocity: bool = False,
                      coherent_packet_pose_blend: float | None = None,
                      enable_high_yaw_wheel_slip_observer: bool = False,
                      lateral_velocity_yaw_rate_gain_m: float | None = None,
                      lateral_velocity_speed_yaw_rate_gain_s: float | None = None,
                      enable_learned_u_residual_model: bool = False,
                      disable_learned_u_residual_model: bool = False,
                      learned_u_residual_model: dict[str, Any] | None = None,
                      unity_actuator_u_model: dict[str, Any] | None = None,
                      unity_actuator_feedback_settle_delay_ms: float = 0.0,
                      disable_unity_actuator_u_experts: bool = False,
                      high_steer_u_model: dict[str, Any] | None = None,
                      decel_ax_scale: float | None = None,
                      decel_ax_offset_mps2: float | None = None,
                      ) -> str:
    topics = " ".join(shlex.quote(topic) for topic in SENSOR_TOPICS)
    lateral_override = (
        "  -p integrate_lateral_acceleration_in_turn:=true \\\n"
        if integrate_lateral_acceleration_in_turn else "")
    catchup_override = (
        "  -p wheel_burst_catchup_accel_mps2:="
        f"{float(wheel_burst_catchup_accel_mps2)!r} \\\n"
        if wheel_burst_catchup_accel_mps2 is not None else "")
    wheel_window_override = (
        "  -p wheel_speed_window_s:="
        f"{float(wheel_speed_window_s)!r} \\\n"
        if wheel_speed_window_s is not None else "")
    wheel_update_override = (
        "  -p wheel_update_beta:="
        f"{float(wheel_update_beta)!r} \\\n"
        if wheel_update_beta is not None else "")
    decel_model_override = ""
    if decel_ax_scale is not None:
        decel_model_override += (
            "  -p decel_ax_scale:="
            f"{float(decel_ax_scale)!r} \\\n")
    if decel_ax_offset_mps2 is not None:
        decel_model_override += (
            "  -p decel_ax_offset_mps2:="
            f"{float(decel_ax_offset_mps2)!r} \\\n")
    turn_packet_decrease_override = (
        "  -p turn_current_packet_max_decrease_mps:="
        f"{float(turn_current_packet_max_decrease_mps)!r} \\\n"
        if turn_current_packet_max_decrease_mps is not None else "")
    turn_bias_override = ""
    if turn_speed_bias_yaw_rate_abs_mps is not None:
        turn_bias_override += (
            "  -p turn_speed_bias_yaw_rate_abs_mps:="
            f"{float(turn_speed_bias_yaw_rate_abs_mps)!r} \\\n")
    if turn_speed_bias_max_mps is not None:
        turn_bias_override += (
            "  -p turn_speed_bias_max_mps:="
            f"{float(turn_speed_bias_max_mps)!r} \\\n")
    turn_slip_fade_override = ""
    if turn_slip_speed_bias_fade_start_mps is not None:
        turn_slip_fade_override = (
            "  -p turn_slip_speed_bias_fade_start_mps:="
            f"{float(turn_slip_speed_bias_fade_start_mps)!r} \\\n"
            "  -p turn_slip_speed_bias_fade_end_mps:="
            f"{float(turn_slip_speed_bias_fade_end_mps)!r} \\\n")
    lateral_gate_override = (
        "  -p use_turn_speed_bias_lateral_accel_gate:=false \\\n"
        if disable_turn_speed_bias_lateral_accel_gate else "")
    pose_velocity_override = (
      "  -p use_coherent_packet_velocity_for_pose:=false \\\n"
      if disable_coherent_packet_pose_velocity else "")
    pose_blend_override = (
      "  -p use_coherent_packet_velocity_for_pose:=true \\\n"
      f"  -p coherent_packet_pose_blend:={float(coherent_packet_pose_blend)!r} \\\n"
      if coherent_packet_pose_blend is not None else "")
    high_yaw_slip_override = (
      "  -p use_high_yaw_wheel_slip_observer:=true \\\n"
      if enable_high_yaw_wheel_slip_observer else "")
    learned_u_residual_override = ""
    if enable_learned_u_residual_model:
        learned_u_residual_override = (
          "  -p use_learned_u_residual_model:=true \\\n")
    elif disable_learned_u_residual_model:
        learned_u_residual_override = (
          "  -p use_learned_u_residual_model:=false \\\n")
    if learned_u_residual_model is not None:
        coefficients = "[" + ",".join(
            repr(float(value))
            for value in learned_u_residual_model["coefficients"]
        ) + "]"
        gate = learned_u_residual_model["activation_gate"]
        learned_u_residual_override = (
            "  -p use_learned_u_residual_model:=true \\\n"
            "  -p learned_u_residual_min_speed_mps:="
            f"{float(gate['baseline_observer_u_min_mps'])!r} \\\n"
            "  -p learned_u_residual_max_speed_mps:="
            f"{float(gate['baseline_observer_u_max_exclusive_mps'])!r} \\\n"
            "  -p learned_u_residual_intercept_mps:="
            f"{float(learned_u_residual_model['intercept_mps'])!r} \\\n"
            "  -p learned_u_residual_coefficients:="
            f"{shlex.quote(coefficients)} \\\n")
    lateral_velocity_model_override = ""
    if lateral_velocity_yaw_rate_gain_m is not None:
        lateral_velocity_model_override = (
            "  -p lateral_velocity_yaw_rate_gain_m:="
            f"{float(lateral_velocity_yaw_rate_gain_m)!r} \\\n"
            "  -p lateral_velocity_speed_yaw_rate_gain_s:="
            f"{float(lateral_velocity_speed_yaw_rate_gain_s)!r} \\\n")
    unity_actuator_u_override = ""
    if unity_actuator_u_model is not None:
        model = unity_actuator_u_model["runtime_model"]
        zero = model["unity_zero_throttle_brake"]
        motor = model["unity_nonzero_motor_torque"]
        zero_coefficients = "[" + ",".join(
            repr(float(value)) for value in zero["coefficients"]) + "]"
        motor_coefficients = "[" + ",".join(
            repr(float(value)) for value in motor["coefficients"]) + "]"
        minimum_speed, maximum_speed = model["speed_gate_mps"]
        unity_actuator_u_override = (
          "  -p use_unity_actuator_u_experts:=true \\\n"
          "  -p unity_actuator_u_min_speed_mps:="
          f"{float(minimum_speed)!r} \\\n"
          "  -p unity_actuator_u_max_speed_mps:="
          f"{float(maximum_speed)!r} \\\n"
          "  -p unity_zero_throttle_intercept_mps:="
          f"{float(zero['intercept_mps'])!r} \\\n"
          "  -p unity_zero_throttle_coefficients:="
          f"{shlex.quote(zero_coefficients)} \\\n"
          "  -p unity_nonzero_throttle_intercept_mps:="
          f"{float(motor['intercept_mps'])!r} \\\n"
          "  -p unity_nonzero_throttle_coefficients:="
          f"{shlex.quote(motor_coefficients)} \\\n")
        if unity_actuator_feedback_settle_delay_ms > 0.0:
            unity_actuator_u_override += (
                "  -p unity_actuator_feedback_settle_delay_ms:="
                f"{unity_actuator_feedback_settle_delay_ms!r} \\\n")
    unity_actuator_u_disable_override = (
        "  -p use_unity_actuator_u_experts:=false \\\n"
        if disable_unity_actuator_u_experts else "")
    high_steer_u_override = ""
    if high_steer_u_model is not None:
        support = high_steer_u_model["support"]
        affine = high_steer_u_model["runtime_affine"]
        coefficients = "[" + ",".join(
            repr(float(value)) for value in affine["coefficients"]) + "]"
        high_steer_u_override = (
            "  -p use_high_steer_unity_actuator_u_expert:=true \\\n"
            "  -p high_steer_unity_actuator_u_min_speed_mps:="
            f"{float(support['published_forward_speed_min_mps'])!r} \\\n"
            "  -p high_steer_unity_actuator_u_max_speed_mps:="
            f"{float(support['published_forward_speed_max_inclusive_mps'])!r} \\\n"
            "  -p high_steer_unity_actuator_u_max_speed_low_throttle_mps:="
            f"{float(support.get('published_forward_speed_max_low_throttle_mps', support['published_forward_speed_max_inclusive_mps']))!r} \\\n"
            "  -p high_steer_unity_actuator_u_throttle_split:="
            f"{float(support.get('throttle_feedback_split', support['throttle_feedback_min_inclusive']))!r} \\\n"
            "  -p high_steer_unity_actuator_u_min_abs_yaw_rate_radps:="
            f"{float(support.get('abs_yaw_rate_min_radps', 0.0))!r} \\\n"
            "  -p high_steer_unity_actuator_u_min_abs_steering_rad:="
            f"{float(support['abs_steering_min_exclusive_rad'])!r} \\\n"
            "  -p high_steer_unity_actuator_u_max_abs_steering_rad:="
            f"{float(support['abs_steering_max_inclusive_rad'])!r} \\\n"
            "  -p high_steer_unity_actuator_u_min_throttle_feedback:="
            f"{float(support['throttle_feedback_min_inclusive'])!r} \\\n"
            "  -p high_steer_unity_actuator_u_max_throttle_feedback:="
            f"{float(support['throttle_feedback_max_inclusive'])!r} \\\n"
            "  -p high_steer_unity_actuator_u_intercept_mps:="
            f"{float(affine['intercept_mps'])!r} \\\n"
            "  -p high_steer_unity_actuator_u_coefficients:="
            f"{shlex.quote(coefficients)} \\\n")
    playback_window = ""
    playback_timeout = ""
    if start_offset_s is not None:
        playback_window = f" --start-offset {start_offset_s:.9g}"
    if duration_wall_s is not None:
        playback_timeout = (
            "set +e\n"
            f"timeout --signal=INT --kill-after=3 {duration_wall_s:.9g}s "
            f"ros2 bag play /input --rate \"$BAG_PLAY_RATE\""
            f"{playback_window} --topics {topics}\n"
            "play_rc=$?\n"
            "set -e\n"
            "if [[ \"$play_rc\" != 0 && \"$play_rc\" != 124 && \"$play_rc\" != 130 ]]; then\n"
            "  echo \"rosbag playback failed: $play_rc\" >&2\n"
            "  exit \"$play_rc\"\n"
            "fi\n"
        )
    else:
        playback_timeout = (
            f"ros2 bag play /input --rate \"$BAG_PLAY_RATE\""
            f"{playback_window} --topics {topics}\n"
        )
    return f"""set -Ee
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
source /workspace/install/setup.bash
set -u
set -o pipefail
export ROS_LOCALHOST_ONLY=1
node_pid= record_pid=
cleanup() {{
  rc=$?
  set +e
  if [[ -n "$record_pid" ]] && kill -0 "$record_pid" 2>/dev/null; then
    kill -INT "$record_pid" 2>/dev/null
    wait "$record_pid"
  fi
  if [[ -n "$node_pid" ]] && kill -0 "$node_pid" 2>/dev/null; then
    kill -INT "$node_pid" 2>/dev/null
    wait "$node_pid"
  fi
  exit "$rc"
}}
trap cleanup EXIT
/workspace/install/lib/f1tenth_localization/sensor_odometry_node \\
  --ros-args --params-file /tmp/sensor_odometry.yaml \\
  -p publish_tf:=false -p odom_topic:=/replayed_odom \\
  -p diagnostics_topic:=/replayed_odom_diagnostics \\
{lateral_override}{catchup_override}{wheel_window_override}{wheel_update_override}{decel_model_override}{turn_packet_decrease_override}{turn_bias_override}{turn_slip_fade_override}{lateral_gate_override}{pose_velocity_override}{pose_blend_override}{high_yaw_slip_override}{lateral_velocity_model_override}{learned_u_residual_override}{unity_actuator_u_override}{unity_actuator_u_disable_override}{high_steer_u_override}  >/results/observer.log 2>&1 &
node_pid=$!
sleep 2
if ! kill -0 "$node_pid" 2>/dev/null; then
  tail -80 /results/observer.log >&2
  exit 1
fi
ros2 bag record --max-cache-size 67108864 -o /results/replayed \\
  /replayed_odom /replayed_odom_diagnostics >/results/recorder.log 2>&1 &
record_pid=$!
ready=0
for attempt in $(seq 1 100); do
  if grep -q "All requested topics are subscribed" /results/recorder.log; then
    ready=1
    break
  fi
  if ! kill -0 "$record_pid" 2>/dev/null; then
    tail -80 /results/recorder.log >&2
    exit 1
  fi
  sleep 0.1
done
if [[ "$ready" != 1 ]]; then
  tail -80 /results/recorder.log >&2
  exit 1
fi
{playback_timeout.rstrip()}
sleep 1
kill -INT "$record_pid"
wait "$record_pid" || true
record_pid=
kill -INT "$node_pid"
wait "$node_pid"
node_pid=
ros2 bag info /results/replayed
tail -20 /results/observer.log
"""


def _yaw(quaternion: object) -> float:
    q = quaternion
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def _read_odometry(path: Path, topic: str) -> dict[int, object]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        if topic not in topics:
            return {}
        return {
            analysis._stamp_ns(message.header.stamp): message
            for _, message in analysis._messages(connection, topics, topic)
            if analysis._stamp_ns(message.header.stamp) > 0
        }
    finally:
        connection.close()


def _verify_against_recorded_odom(input_bag: Path, replay_bag: Path) -> str:
    """Require exact state equality at every source time when reference exists."""
    reference = _read_odometry(input_bag, "/odom")
    if not reference:
        return "no recorded team /odom reference in input; source/config hash only"
    replayed = _read_odometry(replay_bag, "/replayed_odom")
    missing = set(reference) - set(replayed)
    if missing:
        raise ValueError(
            f"observer replay omitted {len(missing)}/{len(reference)} recorded /odom stamps")

    twist_residuals = [[], [], []]
    pose_residuals = []
    first_stamp = min(reference)
    first_ref, first_replay = reference[first_stamp], replayed[first_stamp]
    yaw_offset = (_yaw(first_ref.pose.pose.orientation)
                  - _yaw(first_replay.pose.pose.orientation))
    cos_yaw, sin_yaw = math.cos(yaw_offset), math.sin(yaw_offset)
    replay_origin = np.asarray((first_replay.pose.pose.position.x,
                                first_replay.pose.pose.position.y))
    reference_origin = np.asarray((first_ref.pose.pose.position.x,
                                   first_ref.pose.pose.position.y))

    for stamp, original in reference.items():
        replay = replayed[stamp]
        original_twist = original.twist.twist
        replay_twist = replay.twist.twist
        twist_residuals[0].append(
            float(replay_twist.linear.x - original_twist.linear.x))
        twist_residuals[1].append(
            float(replay_twist.linear.y - original_twist.linear.y))
        twist_residuals[2].append(
            float(replay_twist.angular.z - original_twist.angular.z))
        delta = np.asarray((replay.pose.pose.position.x,
                            replay.pose.pose.position.y)) - replay_origin
        aligned = reference_origin + np.asarray((
            cos_yaw * delta[0] - sin_yaw * delta[1],
            sin_yaw * delta[0] + cos_yaw * delta[1],
        ))
        position = np.asarray((original.pose.pose.position.x,
                               original.pose.pose.position.y))
        pose_residuals.append(float(np.linalg.norm(position - aligned)))

    twist_arrays = [np.asarray(axis, dtype=float) for axis in twist_residuals]
    axis_stats = [(
        math.sqrt(float(np.mean(residual ** 2))),
        float(np.max(np.abs(residual))),
        float(np.mean(residual)),
    ) for residual in twist_arrays]
    max_twist_error = max((stats[1] for stats in axis_stats), default=math.inf)
    p95_position_residual = float(np.quantile(pose_residuals, 0.95))
    axis_details = ", ".join(
        f"{name} RMSE/max/bias={rmse:.4g}/{maximum:.4g}/{bias:+.4g}"
        for name, (rmse, maximum, bias) in zip(("u", "v", "yaw"), axis_stats))
    if max_twist_error > 1e-9 or p95_position_residual > 1e-6:
        raise ValueError(
            "observer replay differs from recorded team /odom after one SE(2) "
            f"frame alignment: {axis_details}; "
            f"p95_position_residual={p95_position_residual:g} m")
    return (f"PASS: exact {len(reference)}/{len(reference)} source-stamped states; "
            f"{axis_details}; "
            f"SE(2)-aligned pose p95={p95_position_residual:.3g} m")


def replay(bag: Path, output_dir: Path, image: str, rate: float,
           domain_id: int, integrate_lateral_acceleration_in_turn: bool = False,
           wheel_burst_catchup_accel_mps2: float | None = None,
           turn_current_packet_max_decrease_mps: float | None = None,
           turn_speed_bias_yaw_rate_abs_mps: float | None = None,
           turn_speed_bias_max_mps: float | None = None,
           start_offset_s: float | None = None,
           duration_wall_s: float | None = None,
           source_sequence_index: int | None = None,
           build_mounted_workspace: bool = False,
           disable_turn_speed_bias_lateral_accel_gate: bool = False,
           disable_coherent_packet_pose_velocity: bool = False,
           coherent_packet_pose_blend: float | None = None,
           enable_high_yaw_wheel_slip_observer: bool = False,
           skip_recorded_odom_equivalence: bool = False,
           turn_slip_speed_bias_fade_start_mps: float | None = None,
           turn_slip_speed_bias_fade_end_mps: float | None = None,
           lateral_velocity_yaw_rate_gain_m: float | None = None,
           lateral_velocity_speed_yaw_rate_gain_s: float | None = None,
           enable_learned_u_residual_model: bool = False,
           disable_learned_u_residual_model: bool = False,
           learned_u_residual_model_path: Path | None = None,
           wheel_speed_window_s: float | None = None,
           decel_ax_scale: float | None = None,
           decel_ax_offset_mps2: float | None = None,
           unity_actuator_u_model_path: Path | None = None,
           unity_actuator_feedback_settle_delay_ms: float = 0.0,
           disable_unity_actuator_u_experts: bool = False,
           high_steer_u_model_path: Path | None = None,
           high_steer_u_gain_scale: float = 1.0,
           wheel_update_beta: float | None = None,
           ) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    bag = bag.resolve()
    output_dir = output_dir.resolve()
    if not bag.is_file() or bag.name != "run_0.db3":
        raise ValueError("--bag must be an existing closed run_0.db3 file")
    if not (bag.parent / "metadata.yaml").is_file():
        raise ValueError("bag metadata.yaml is required for read-only playback")
    if enable_learned_u_residual_model and disable_learned_u_residual_model:
        raise ValueError("learned Odom-u residual cannot be enabled and disabled together")
    if learned_u_residual_model_path is not None and (
            enable_learned_u_residual_model or disable_learned_u_residual_model):
        raise ValueError("a learned Odom-u model file replaces the enable/disable flags")
    learned_u_residual_model = None
    learned_u_residual_model_sha256 = None
    if learned_u_residual_model_path is not None:
        learned_u_residual_model_path = learned_u_residual_model_path.resolve()
        learned_u_residual_model = json.loads(
            learned_u_residual_model_path.read_text(encoding="utf-8"))
        gate = learned_u_residual_model.get("activation_gate", {})
        coefficients = learned_u_residual_model.get("coefficients", [])
        minimum_speed = gate.get("baseline_observer_u_min_mps")
        maximum_speed = gate.get("baseline_observer_u_max_exclusive_mps")
        numeric_values = [learned_u_residual_model.get("intercept_mps")]
        numeric_values.extend(coefficients)
        numeric_values.extend((minimum_speed, maximum_speed))
        if (len(coefficients) != 22 or
                not all(isinstance(value, (int, float)) and math.isfinite(value)
                        for value in numeric_values) or
                minimum_speed >= maximum_speed):
            raise ValueError("learned Odom-u model file has invalid parameters")
        learned_u_residual_model_sha256 = _sha256(learned_u_residual_model_path)
    unity_actuator_u_model = None
    unity_actuator_u_model_sha256 = None
    if unity_actuator_u_model_path is not None:
        unity_actuator_u_model_path = unity_actuator_u_model_path.resolve()
        unity_actuator_u_model = json.loads(
            unity_actuator_u_model_path.read_text(encoding="utf-8"))
        runtime_model = unity_actuator_u_model.get("runtime_model", {})
        feature_names = runtime_model.get("feature_names", [])
        minimum_speed, maximum_speed = runtime_model.get("speed_gate_mps", [None, None])
        numeric_values = [minimum_speed, maximum_speed]
        for key in ("unity_zero_throttle_brake", "unity_nonzero_motor_torque"):
            expert = runtime_model.get(key, {})
            coefficients = expert.get("coefficients", [])
            numeric_values.append(expert.get("intercept_mps"))
            numeric_values.extend(coefficients)
            if len(coefficients) != 34:
                raise ValueError(f"Unity actuator Odom expert {key} must have 34 coefficients")
        if (len(feature_names) != 34 or
                not all(isinstance(value, (int, float)) and math.isfinite(value)
                        for value in numeric_values) or
                minimum_speed >= maximum_speed):
            raise ValueError("Unity actuator Odom model file has invalid parameters")
        unity_actuator_u_model_sha256 = _sha256(unity_actuator_u_model_path)
    high_steer_u_model = None
    high_steer_u_model_sha256 = None
    if high_steer_u_model_path is not None:
        high_steer_u_model_path = high_steer_u_model_path.resolve()
        high_steer_u_model = json.loads(
            high_steer_u_model_path.read_text(encoding="utf-8"))
        affine = dict(high_steer_u_model.get("runtime_affine", {}))
        affine["intercept_mps"] = (
            float(affine["intercept_mps"]) * high_steer_u_gain_scale)
        affine["coefficients"] = [
            float(value) * high_steer_u_gain_scale
            for value in affine.get("coefficients", [])]
        high_steer_u_model = dict(high_steer_u_model)
        high_steer_u_model["runtime_affine"] = affine
        support = high_steer_u_model.get("support", {})
        coefficients = affine.get("coefficients", [])
        numeric_values = [
            affine.get("intercept_mps"),
            support.get("published_forward_speed_min_mps"),
            support.get("published_forward_speed_max_inclusive_mps"),
            support.get("published_forward_speed_max_low_throttle_mps",
                        support.get("published_forward_speed_max_inclusive_mps")),
            support.get("throttle_feedback_split",
                        support.get("throttle_feedback_min_inclusive")),
            support.get("abs_steering_min_exclusive_rad"),
            support.get("abs_steering_max_inclusive_rad"),
            support.get("throttle_feedback_min_inclusive"),
            support.get("throttle_feedback_max_inclusive"),
            *coefficients,
        ]
        if (len(coefficients) != 34 or
                len(high_steer_u_model.get("feature_names", [])) != 34 or
                not all(isinstance(value, (int, float)) and math.isfinite(value)
                        for value in numeric_values) or
                support["published_forward_speed_min_mps"] >=
                support["published_forward_speed_max_inclusive_mps"] or
                support.get("published_forward_speed_max_low_throttle_mps",
                            support["published_forward_speed_max_inclusive_mps"]) <=
                support["published_forward_speed_min_mps"] or
                support.get("published_forward_speed_max_low_throttle_mps",
                            support["published_forward_speed_max_inclusive_mps"]) >
                support["published_forward_speed_max_inclusive_mps"] or
                support.get("throttle_feedback_split",
                            support["throttle_feedback_min_inclusive"]) <
                support["throttle_feedback_min_inclusive"] or
                support.get("throttle_feedback_split",
                            support["throttle_feedback_min_inclusive"]) >
                support["throttle_feedback_max_inclusive"] or
                support["abs_steering_min_exclusive_rad"] >=
                support["abs_steering_max_inclusive_rad"] or
                support["throttle_feedback_min_inclusive"] >
                support["throttle_feedback_max_inclusive"]):
            raise ValueError("high-steer Odom-u model file has invalid parameters")
        high_steer_u_model_sha256 = _sha256(high_steer_u_model_path)
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    if not math.isfinite(rate) or not 0.1 <= rate <= 4.0:
        raise ValueError("--rate must be finite and between 0.1 and 4.0")
    if (not math.isfinite(unity_actuator_feedback_settle_delay_ms) or
            not 0.0 <= unity_actuator_feedback_settle_delay_ms <= 10.0):
        raise ValueError("--unity-actuator-feedback-settle-delay-ms must be in [0, 10]")
    if unity_actuator_feedback_settle_delay_ms > 0.0 and unity_actuator_u_model is None:
        raise ValueError("feedback settle delay is only valid with a Unity actuator model")
    if disable_unity_actuator_u_experts and unity_actuator_u_model is not None:
        raise ValueError("Unity actuator experts cannot be disabled and supplied together")
    if disable_unity_actuator_u_experts and unity_actuator_feedback_settle_delay_ms > 0.0:
        raise ValueError("feedback settle delay is invalid when Unity actuator experts are disabled")
    if not math.isfinite(high_steer_u_gain_scale) or not 0.0 <= high_steer_u_gain_scale <= 1.0:
        raise ValueError("high-steer Odom-u gain scale must be finite and in [0, 1]")
    if high_steer_u_model_path is None and high_steer_u_gain_scale != 1.0:
        raise ValueError("high-steer Odom-u gain scale requires a model JSON")
    if not 0 <= domain_id <= 232:
        raise ValueError("--domain-id must be between 0 and 232")
    if ((start_offset_s is None) != (duration_wall_s is None)
            or start_offset_s is not None and
            (not math.isfinite(start_offset_s) or start_offset_s < 0.0)
            or duration_wall_s is not None and
            (not math.isfinite(duration_wall_s) or duration_wall_s <= 0.0)):
        raise ValueError("playback window requires nonnegative start offset and positive wall duration")
    if (wheel_speed_window_s is not None and
            (not math.isfinite(wheel_speed_window_s) or wheel_speed_window_s < 0.0)):
        raise ValueError("wheel speed window must be finite and nonnegative")
    if (wheel_update_beta is not None and
            (not math.isfinite(wheel_update_beta) or
             not 0.0 <= wheel_update_beta <= 1.0)):
        raise ValueError("wheel update beta must be finite and in [0, 1]")
    if any(value is not None and not math.isfinite(value)
           for value in (decel_ax_scale, decel_ax_offset_mps2)):
        raise ValueError("deceleration model overrides must be finite")
    if source_sequence_index is not None and (
            start_offset_s is None or source_sequence_index < 0):
        raise ValueError("--source-sequence-index requires a playback window and nonnegative index")
    if (wheel_burst_catchup_accel_mps2 is not None and
            (not math.isfinite(wheel_burst_catchup_accel_mps2) or
             wheel_burst_catchup_accel_mps2 < 0.0)):
        raise ValueError("wheel burst catch-up acceleration must be finite and nonnegative")
    if (turn_current_packet_max_decrease_mps is not None and
            (not math.isfinite(turn_current_packet_max_decrease_mps) or
             turn_current_packet_max_decrease_mps < 0.0)):
        raise ValueError("turn current-packet maximum decrease must be finite and nonnegative")
    if ((turn_speed_bias_yaw_rate_abs_mps is None) !=
            (turn_speed_bias_max_mps is None)):
        raise ValueError("turn speed bias coefficient and limit must be set together")
    if ((turn_slip_speed_bias_fade_start_mps is None) !=
            (turn_slip_speed_bias_fade_end_mps is None)):
        raise ValueError("turn speed-bias fade start and end must be set together")
    if (turn_slip_speed_bias_fade_start_mps is not None and
            (not math.isfinite(turn_slip_speed_bias_fade_start_mps) or
             not math.isfinite(turn_slip_speed_bias_fade_end_mps) or
             turn_slip_speed_bias_fade_start_mps < 0.0 or
             turn_slip_speed_bias_fade_end_mps <=
             turn_slip_speed_bias_fade_start_mps)):
        raise ValueError("turn speed-bias fade must have finite 0 <= start < end")
    if ((lateral_velocity_yaw_rate_gain_m is None) !=
            (lateral_velocity_speed_yaw_rate_gain_s is None)):
        raise ValueError("both lateral-velocity model coefficients must be set together")
    for name, value in (
            ("lateral velocity yaw-rate gain", lateral_velocity_yaw_rate_gain_m),
            ("lateral velocity speed/yaw-rate gain",
             lateral_velocity_speed_yaw_rate_gain_s)):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if (disable_coherent_packet_pose_velocity and
            coherent_packet_pose_blend is not None):
        raise ValueError("pose-velocity disable and pose blend override are mutually exclusive")
    if (coherent_packet_pose_blend is not None and
            (not math.isfinite(coherent_packet_pose_blend) or
             not 0.0 <= coherent_packet_pose_blend <= 1.0)):
        raise ValueError("coherent packet pose blend must be between 0 and 1")
    for name, value in (("turn speed bias coefficient",
                         turn_speed_bias_yaw_rate_abs_mps),
                        ("turn speed bias limit", turn_speed_bias_max_mps)):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")

    docker_env = _docker_environment(repo_root)
    if build_mounted_workspace:
        image_id = _docker(
            image, ["image", "inspect", "--format", "{{.Id}}", image],
            docker_env, capture_output=True,
        ).stdout.strip()
        source_mode = (
            "read-only worktree mounted; f1tenth_localization rebuilt in isolated replay container")
    else:
        image_id = _verify_image_sources(image, repo_root, docker_env)
        source_mode = "installed image sources hash-verified against worktree"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    print(f"Replay image: {image} ({image_id})", flush=True)
    print(f"Input bag, read-only: {bag}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Playback rate: {rate:g}x; isolated ROS domain: {domain_id}", flush=True)
    if start_offset_s is not None:
        print(f"Playback window: start={start_offset_s:g}s, "
              f"wall duration={duration_wall_s:g}s; fresh observer process",
              flush=True)
    if integrate_lateral_acceleration_in_turn:
        print("Observer-only override: integrate_lateral_acceleration_in_turn=true",
              flush=True)
    if wheel_burst_catchup_accel_mps2 is not None:
        print("Offline-only override: "
              f"wheel_burst_catchup_accel_mps2={wheel_burst_catchup_accel_mps2:g}",
              flush=True)
    if wheel_speed_window_s is not None:
        print("Offline-only encoder-window override: "
              f"wheel_speed_window_s={wheel_speed_window_s:g} s", flush=True)
    if wheel_update_beta is not None:
        print("Offline-only wheel-fusion override: "
              f"wheel_update_beta={wheel_update_beta:g}", flush=True)
    if decel_ax_scale is not None or decel_ax_offset_mps2 is not None:
        print("Offline-only deceleration-model override: "
              f"scale={decel_ax_scale}, offset={decel_ax_offset_mps2} m/s^2",
              flush=True)
    if turn_current_packet_max_decrease_mps is not None:
        print("Offline-only override: "
              "turn_current_packet_max_decrease_mps="
              f"{turn_current_packet_max_decrease_mps:g}", flush=True)
    if turn_speed_bias_yaw_rate_abs_mps is not None:
        print("Offline-only turn calibration override: "
              f"yaw-rate coefficient={turn_speed_bias_yaw_rate_abs_mps:g}, "
              f"absolute limit={turn_speed_bias_max_mps:g} m/s", flush=True)
    if turn_slip_speed_bias_fade_start_mps is not None:
        print("Offline-only turn speed-bias fade override: "
              f"{turn_slip_speed_bias_fade_start_mps:g}-"
              f"{turn_slip_speed_bias_fade_end_mps:g} m/s", flush=True)
    if disable_turn_speed_bias_lateral_accel_gate:
        print("Offline-only observer override: "
              "use_turn_speed_bias_lateral_accel_gate=false", flush=True)
    if disable_coherent_packet_pose_velocity:
        print("Offline-only observer override: "
              "use_coherent_packet_velocity_for_pose=false", flush=True)
    if coherent_packet_pose_blend is not None:
        print("Offline-only observer override: "
              f"coherent_packet_pose_blend={coherent_packet_pose_blend:g}",
              flush=True)
    if enable_high_yaw_wheel_slip_observer:
        print("Offline-only observer override: "
              "use_high_yaw_wheel_slip_observer=true", flush=True)
    if high_steer_u_model is not None:
        print("Offline-only high-steer Odom-u expert override: "
              f"{high_steer_u_model['support']}", flush=True)
    print("Input topics: " + ", ".join(SENSOR_TOPICS), flush=True)

    command = [
        "run", "--rm", "--network=none", "--log-opt", "max-size=10m",
        "--log-opt", "max-file=2", "-e", f"ROS_DOMAIN_ID={domain_id}",
        "-e", f"BAG_PLAY_RATE={rate:g}",
        "--mount", f"type=bind,source={bag.parent},target=/input,readonly",
        "--mount", f"type=bind,source={output_dir},target=/results",
        "--mount",
        f"type=bind,source={repo_root / SOURCE_FILES[-1]},"
        "target=/tmp/sensor_odometry.yaml,readonly",
    ]
    replay_script = _container_script(
        integrate_lateral_acceleration_in_turn=integrate_lateral_acceleration_in_turn,
        wheel_burst_catchup_accel_mps2=wheel_burst_catchup_accel_mps2,
        wheel_speed_window_s=wheel_speed_window_s,
        wheel_update_beta=wheel_update_beta,
        decel_ax_scale=decel_ax_scale,
        decel_ax_offset_mps2=decel_ax_offset_mps2,
        turn_current_packet_max_decrease_mps=turn_current_packet_max_decrease_mps,
        turn_speed_bias_yaw_rate_abs_mps=turn_speed_bias_yaw_rate_abs_mps,
        turn_speed_bias_max_mps=turn_speed_bias_max_mps,
        turn_slip_speed_bias_fade_start_mps=turn_slip_speed_bias_fade_start_mps,
        turn_slip_speed_bias_fade_end_mps=turn_slip_speed_bias_fade_end_mps,
        start_offset_s=start_offset_s,
        duration_wall_s=duration_wall_s,
        disable_turn_speed_bias_lateral_accel_gate=disable_turn_speed_bias_lateral_accel_gate,
        disable_coherent_packet_pose_velocity=disable_coherent_packet_pose_velocity,
        coherent_packet_pose_blend=coherent_packet_pose_blend,
        enable_high_yaw_wheel_slip_observer=enable_high_yaw_wheel_slip_observer,
        lateral_velocity_yaw_rate_gain_m=lateral_velocity_yaw_rate_gain_m,
        lateral_velocity_speed_yaw_rate_gain_s=lateral_velocity_speed_yaw_rate_gain_s,
        enable_learned_u_residual_model=enable_learned_u_residual_model,
        disable_learned_u_residual_model=disable_learned_u_residual_model,
        learned_u_residual_model=learned_u_residual_model,
        unity_actuator_u_model=unity_actuator_u_model,
        unity_actuator_feedback_settle_delay_ms=unity_actuator_feedback_settle_delay_ms,
        disable_unity_actuator_u_experts=disable_unity_actuator_u_experts,
        high_steer_u_model=high_steer_u_model)
    if build_mounted_workspace:
        command.extend([
            "-e", "SDU_APEX_BUILD_LOCALIZATION=1",
            "--mount", f"type=bind,source={repo_root},target=/workspace/src,readonly",
            "--entrypoint", "/bin/bash", image, "-lc",
            "/bin/bash /workspace/src/docker/development_entrypoint.sh /bin/bash -lc " +
            shlex.quote(replay_script),
        ])
    else:
        command.extend([
            "--entrypoint", "/bin/bash", image, "-lc", replay_script,
        ])
    _docker(image, command, docker_env)
    replay_bag = output_dir / "replayed" / "replayed_0.db3"
    if skip_recorded_odom_equivalence:
        print("Observer equivalence skipped by request; the input bag's "
              "recorded /odom is not asserted to match this replay", flush=True)
    elif (integrate_lateral_acceleration_in_turn or
            wheel_burst_catchup_accel_mps2 is not None or
            wheel_speed_window_s is not None or
            wheel_update_beta is not None or
            decel_ax_scale is not None or decel_ax_offset_mps2 is not None or
            turn_current_packet_max_decrease_mps is not None or
            turn_speed_bias_yaw_rate_abs_mps is not None or
            turn_slip_speed_bias_fade_start_mps is not None or
            disable_turn_speed_bias_lateral_accel_gate or
            disable_coherent_packet_pose_velocity or
            coherent_packet_pose_blend is not None or
            enable_high_yaw_wheel_slip_observer or
            lateral_velocity_yaw_rate_gain_m is not None or
            enable_learned_u_residual_model or
            disable_learned_u_residual_model or
            learned_u_residual_model is not None or
            unity_actuator_u_model is not None or
            high_steer_u_model is not None or
            disable_unity_actuator_u_experts or
            unity_actuator_feedback_settle_delay_ms > 0.0):
        print("Observer equivalence: parameter variant; recorded /odom equality "
              "is not expected", flush=True)
    else:
        verification = _verify_against_recorded_odom(bag, replay_bag)
        print(f"Observer equivalence: {verification}", flush=True)
    metadata = {
        "image": image,
        "image_id": image_id,
        "localization_source_mode": source_mode,
        "localization_source_sha256": {
            relative: _sha256(repo_root / relative)
            for relative in SOURCE_FILES
        },
        "input_bag": str(bag),
        "input_bag_sha256": _sha256(bag),
        "playback_rate": rate,
        "playback_window": ({
            "start_offset_s": start_offset_s,
            "duration_wall_s": duration_wall_s,
            "source_sequence_index": source_sequence_index,
            "observer_reinitialized_for_segment": True,
        } if start_offset_s is not None else None),
        "domain_id": domain_id,
        "sensor_topics": list(SENSOR_TOPICS),
        "skip_recorded_odom_equivalence": skip_recorded_odom_equivalence,
        "recorded_odom_equivalence": (
            "skipped_by_request_for_source_or_configuration_variant"
            if skip_recorded_odom_equivalence else "verified_or_not_applicable"),
        "parameter_overrides": {
            **({"integrate_lateral_acceleration_in_turn": True}
               if integrate_lateral_acceleration_in_turn else {}),
            **({"wheel_burst_catchup_accel_mps2":
                wheel_burst_catchup_accel_mps2}
               if wheel_burst_catchup_accel_mps2 is not None else {}),
            **({"wheel_speed_window_s": wheel_speed_window_s}
               if wheel_speed_window_s is not None else {}),
            **({"wheel_update_beta": wheel_update_beta}
               if wheel_update_beta is not None else {}),
            **({"decel_ax_scale": decel_ax_scale}
               if decel_ax_scale is not None else {}),
            **({"decel_ax_offset_mps2": decel_ax_offset_mps2}
               if decel_ax_offset_mps2 is not None else {}),
            **({"turn_current_packet_max_decrease_mps":
                turn_current_packet_max_decrease_mps}
               if turn_current_packet_max_decrease_mps is not None else {}),
            **({"turn_speed_bias_yaw_rate_abs_mps":
                turn_speed_bias_yaw_rate_abs_mps,
                "turn_speed_bias_max_mps": turn_speed_bias_max_mps}
               if turn_speed_bias_yaw_rate_abs_mps is not None else {}),
            **({"turn_slip_speed_bias_fade_start_mps":
                turn_slip_speed_bias_fade_start_mps,
                "turn_slip_speed_bias_fade_end_mps":
                turn_slip_speed_bias_fade_end_mps}
               if turn_slip_speed_bias_fade_start_mps is not None else {}),
            **({"use_turn_speed_bias_lateral_accel_gate": False}
               if disable_turn_speed_bias_lateral_accel_gate else {}),
            **({"use_coherent_packet_velocity_for_pose": False}
               if disable_coherent_packet_pose_velocity else {}),
            **({"use_coherent_packet_velocity_for_pose": True,
                "coherent_packet_pose_blend": coherent_packet_pose_blend}
               if coherent_packet_pose_blend is not None else {}),
            **({"use_high_yaw_wheel_slip_observer": True}
               if enable_high_yaw_wheel_slip_observer else {}),
            **({"lateral_velocity_yaw_rate_gain_m":
                lateral_velocity_yaw_rate_gain_m,
                "lateral_velocity_speed_yaw_rate_gain_s":
                lateral_velocity_speed_yaw_rate_gain_s}
               if lateral_velocity_yaw_rate_gain_m is not None else {}),
            **({"use_learned_u_residual_model": True}
               if enable_learned_u_residual_model else {}),
            **({"use_learned_u_residual_model": False}
               if disable_learned_u_residual_model else {}),
            **({"learned_u_residual_model_path":
                str(learned_u_residual_model_path),
                "learned_u_residual_model_sha256":
                learned_u_residual_model_sha256,
                "learned_u_residual_model_id":
                learned_u_residual_model.get("model_id")}
               if learned_u_residual_model is not None else {}),
            **({"use_unity_actuator_u_experts": True,
                "unity_actuator_u_model_path": str(unity_actuator_u_model_path),
                "unity_actuator_u_model_sha256": unity_actuator_u_model_sha256,
                "unity_actuator_u_training_runs":
                    unity_actuator_u_model["runtime_model"]["training_runs"],
                "unity_actuator_feedback_settle_delay_ms":
                    unity_actuator_feedback_settle_delay_ms}
               if unity_actuator_u_model is not None else {}),
            **({"use_unity_actuator_u_experts": False}
               if disable_unity_actuator_u_experts else {}),
            **({"use_high_steer_unity_actuator_u_expert": True,
                "high_steer_unity_actuator_u_model_path":
                    str(high_steer_u_model_path),
                "high_steer_unity_actuator_u_model_sha256":
                    high_steer_u_model_sha256,
                "high_steer_unity_actuator_u_support":
                    high_steer_u_model["support"],
                "high_steer_unity_actuator_u_gain_scale":
                    high_steer_u_gain_scale,
                "high_steer_unity_actuator_u_training_runs":
                    high_steer_u_model["training_runs"]}
               if high_steer_u_model is not None else {}),
        },
    }
    (output_dir / "replay_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True,
                        help="closed ROS 2 run_0.db3 input bag")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for the derived odometry bag/logs")
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="existing Humble development image with matching source")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="rosbag playback rate (validated maximum: 4x)")
    parser.add_argument(
        "--build-mounted-workspace", action="store_true",
        help=("mount this worktree read-only and rebuild f1tenth_localization "
              "inside the isolated replay container when the image's installed "
              "sources do not match"),
    )
    parser.add_argument(
        "--skip-recorded-odom-equivalence", action="store_true",
        help=("skip automatic comparison with the input bag's recorded /odom "
              "when replaying a deliberate source/configuration variant"),
    )
    parser.add_argument("--domain-id", type=int, default=96,
                        help="isolated ROS domain for the offline replay")
    parser.add_argument("--integrate-lateral-acceleration-in-turn", action="store_true",
                        help="offline-only observer A/B; override that parameter to true")
    parser.add_argument("--wheel-burst-catchup-accel-mps2", type=float,
                        help="offline-only parameter override for controlled burst-recovery replay")
    parser.add_argument("--wheel-speed-window-s", type=float,
                        help="offline-only encoder-angle window override for an Odom replay")
    parser.add_argument("--wheel-update-beta", type=float,
                        help="offline-only straight-speed wheel/IMU fusion override in [0, 1]")
    parser.add_argument("--decel-ax-scale", type=float,
                        help="offline-only longitudinal deceleration scale override")
    parser.add_argument("--decel-ax-offset-mps2", type=float,
                        help="offline-only longitudinal deceleration offset override")
    parser.add_argument(
        "--turn-current-packet-max-decrease-mps", type=float,
        help="offline-only recovery-gate override for paired odometry replay",
    )
    parser.add_argument("--turn-speed-bias-yaw-rate-abs-mps", type=float,
                        help="offline-only turn speed correction per rad/s yaw rate")
    parser.add_argument("--turn-speed-bias-max-mps", type=float,
                        help="offline-only absolute bound paired with turn speed correction")
    parser.add_argument(
        "--turn-slip-speed-bias-fade-start-mps", type=float,
        help="offline-only start speed for tapering the turn-speed correction",
    )
    parser.add_argument(
        "--turn-slip-speed-bias-fade-end-mps", type=float,
        help="offline-only end speed for tapering the turn-speed correction",
    )
    parser.add_argument(
        "--disable-turn-speed-bias-lateral-accel-gate", action="store_true",
        help="offline-only A/B: use the un-gated high-speed turn-speed correction",
    )
    parser.add_argument(
        "--disable-coherent-packet-pose-velocity", action="store_true",
        help=("offline-only A/B: integrate pose from the published filtered speed "
              "instead of the fresh encoder-packet speed"),
    )
    parser.add_argument(
        "--coherent-packet-pose-blend", type=float,
        help="offline-only pose A/B: blend [0,1] between filtered and coherent packet speed",
    )
    parser.add_argument(
        "--enable-high-yaw-wheel-slip-observer", action="store_true",
        help=("offline-only A/B: publish the causal rigid-body IMU latent-u estimate "
              "when high-yaw wheel speed exceeds it by the configured innovation"),
    )
    parser.add_argument(
        "--lateral-velocity-yaw-rate-gain-m", type=float,
        help="offline-only override for the lateral COM velocity yaw-rate gain",
    )
    parser.add_argument(
        "--lateral-velocity-speed-yaw-rate-gain-s", type=float,
        help="offline-only override for the speed-dependent lateral velocity gain",
    )
    parser.add_argument(
        "--enable-learned-u-residual-model", action="store_true",
        help="offline-only A/B: enable the support-gated learned Odom-u residual",
    )
    parser.add_argument(
        "--disable-learned-u-residual-model", action="store_true",
        help="offline-only A/B: disable the learned Odom-u residual from YAML",
    )
    parser.add_argument(
        "--learned-u-residual-model-json", type=Path,
        help="offline-only: recursively replay a fitted model JSON over YAML coefficients",
    )
    parser.add_argument(
        "--unity-actuator-u-model-json", type=Path,
        help="offline-only: apply the Unity source-branch Odom candidate from a fitted report",
    )
    parser.add_argument(
        "--high-steer-u-model-json", type=Path,
        help="offline-only: apply the support-gated high-steering Odom-u expert",
    )
    parser.add_argument(
        "--high-steer-u-gain-scale", type=float, default=1.0,
        help="offline-only: scale a fitted high-steer expert correction in [0, 1]",
    )
    parser.add_argument(
        "--unity-actuator-feedback-settle-delay-ms", type=float, default=0.0,
        help="offline-only: defer packet updates to settle allowed actuator feedback",
    )
    parser.add_argument(
        "--disable-unity-actuator-u-experts", action="store_true",
        help="offline-only A/B: disable Unity actuator experts configured in YAML",
    )
    parser.add_argument("--start-offset-s", type=float,
                        help="play one isolated segment from this bag offset")
    parser.add_argument("--duration-wall-s", type=float,
                        help="wall-clock playback window; pairs with --start-offset-s")
    parser.add_argument("--source-sequence-index", type=int,
                        help="source reset-sequence represented by this isolated segment")
    args = parser.parse_args()
    try:
        replay(args.bag, args.output_dir, args.image, args.rate, args.domain_id,
               args.integrate_lateral_acceleration_in_turn,
               args.wheel_burst_catchup_accel_mps2,
               args.turn_current_packet_max_decrease_mps,
               args.turn_speed_bias_yaw_rate_abs_mps,
               args.turn_speed_bias_max_mps,
               args.start_offset_s, args.duration_wall_s,
               args.source_sequence_index,
               args.build_mounted_workspace,
               args.disable_turn_speed_bias_lateral_accel_gate,
               args.disable_coherent_packet_pose_velocity,
               args.coherent_packet_pose_blend,
               args.enable_high_yaw_wheel_slip_observer,
               args.skip_recorded_odom_equivalence,
               args.turn_slip_speed_bias_fade_start_mps,
               args.turn_slip_speed_bias_fade_end_mps,
               args.lateral_velocity_yaw_rate_gain_m,
               args.lateral_velocity_speed_yaw_rate_gain_s,
               args.enable_learned_u_residual_model,
               args.disable_learned_u_residual_model,
               args.learned_u_residual_model_json,
               args.wheel_speed_window_s,
               args.decel_ax_scale,
               args.decel_ax_offset_mps2,
               args.unity_actuator_u_model_json,
               args.unity_actuator_feedback_settle_delay_ms,
               args.disable_unity_actuator_u_experts,
               args.high_steer_u_model_json,
               args.high_steer_u_gain_scale,
               args.wheel_update_beta)
    except (OSError, ValueError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"sensor-odometry replay failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
