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
                      turn_speed_bias_yaw_rate_abs_mps: float | None = None,
                      turn_speed_bias_max_mps: float | None = None,
                      start_offset_s: float | None = None,
                      duration_wall_s: float | None = None,
                      ) -> str:
    topics = " ".join(shlex.quote(topic) for topic in SENSOR_TOPICS)
    lateral_override = (
        "  -p integrate_lateral_acceleration_in_turn:=true \\\n"
        if integrate_lateral_acceleration_in_turn else "")
    catchup_override = (
        "  -p wheel_burst_catchup_accel_mps2:="
        f"{float(wheel_burst_catchup_accel_mps2)!r} \\\n"
        if wheel_burst_catchup_accel_mps2 is not None else "")
    turn_bias_override = ""
    if turn_speed_bias_yaw_rate_abs_mps is not None:
        turn_bias_override += (
            "  -p turn_speed_bias_yaw_rate_abs_mps:="
            f"{float(turn_speed_bias_yaw_rate_abs_mps)!r} \\\n")
    if turn_speed_bias_max_mps is not None:
        turn_bias_override += (
            "  -p turn_speed_bias_max_mps:="
            f"{float(turn_speed_bias_max_mps)!r} \\\n")
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
{lateral_override}{catchup_override}{turn_bias_override}  >/results/observer.log 2>&1 &
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
           turn_speed_bias_yaw_rate_abs_mps: float | None = None,
           turn_speed_bias_max_mps: float | None = None,
           start_offset_s: float | None = None,
           duration_wall_s: float | None = None,
           source_sequence_index: int | None = None,
           ) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    bag = bag.resolve()
    output_dir = output_dir.resolve()
    if not bag.is_file() or bag.name != "run_0.db3":
        raise ValueError("--bag must be an existing closed run_0.db3 file")
    if not (bag.parent / "metadata.yaml").is_file():
        raise ValueError("bag metadata.yaml is required for read-only playback")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    if not math.isfinite(rate) or not 0.1 <= rate <= 4.0:
        raise ValueError("--rate must be finite and between 0.1 and 4.0")
    if not 0 <= domain_id <= 232:
        raise ValueError("--domain-id must be between 0 and 232")
    if ((start_offset_s is None) != (duration_wall_s is None)
            or start_offset_s is not None and
            (not math.isfinite(start_offset_s) or start_offset_s < 0.0)
            or duration_wall_s is not None and
            (not math.isfinite(duration_wall_s) or duration_wall_s <= 0.0)):
        raise ValueError("playback window requires nonnegative start offset and positive wall duration")
    if source_sequence_index is not None and (
            start_offset_s is None or source_sequence_index < 0):
        raise ValueError("--source-sequence-index requires a playback window and nonnegative index")
    if (wheel_burst_catchup_accel_mps2 is not None and
            (not math.isfinite(wheel_burst_catchup_accel_mps2) or
             wheel_burst_catchup_accel_mps2 < 0.0)):
        raise ValueError("wheel burst catch-up acceleration must be finite and nonnegative")
    if ((turn_speed_bias_yaw_rate_abs_mps is None) !=
            (turn_speed_bias_max_mps is None)):
        raise ValueError("turn speed bias coefficient and limit must be set together")
    for name, value in (("turn speed bias coefficient",
                         turn_speed_bias_yaw_rate_abs_mps),
                        ("turn speed bias limit", turn_speed_bias_max_mps)):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")

    docker_env = _docker_environment(repo_root)
    image_id = _verify_image_sources(image, repo_root, docker_env)
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
    if turn_speed_bias_yaw_rate_abs_mps is not None:
        print("Offline-only turn calibration override: "
              f"yaw-rate coefficient={turn_speed_bias_yaw_rate_abs_mps:g}, "
              f"absolute limit={turn_speed_bias_max_mps:g} m/s", flush=True)
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
        "--entrypoint", "/bin/bash", image, "-lc",
        _container_script(integrate_lateral_acceleration_in_turn,
                          wheel_burst_catchup_accel_mps2,
                          turn_speed_bias_yaw_rate_abs_mps,
                          turn_speed_bias_max_mps,
                          start_offset_s, duration_wall_s),
    ]
    _docker(image, command, docker_env)
    replay_bag = output_dir / "replayed" / "replayed_0.db3"
    if (integrate_lateral_acceleration_in_turn or
            wheel_burst_catchup_accel_mps2 is not None or
            turn_speed_bias_yaw_rate_abs_mps is not None):
        print("Observer equivalence: parameter variant; recorded /odom equality "
              "is not expected", flush=True)
    else:
        verification = _verify_against_recorded_odom(bag, replay_bag)
        print(f"Observer equivalence: {verification}", flush=True)
    metadata = {
        "image": image,
        "image_id": image_id,
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
        "parameter_overrides": {
            **({"integrate_lateral_acceleration_in_turn": True}
               if integrate_lateral_acceleration_in_turn else {}),
            **({"wheel_burst_catchup_accel_mps2":
                wheel_burst_catchup_accel_mps2}
               if wheel_burst_catchup_accel_mps2 is not None else {}),
            **({"turn_speed_bias_yaw_rate_abs_mps":
                turn_speed_bias_yaw_rate_abs_mps,
                "turn_speed_bias_max_mps": turn_speed_bias_max_mps}
               if turn_speed_bias_yaw_rate_abs_mps is not None else {}),
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
    parser.add_argument("--domain-id", type=int, default=96,
                        help="isolated ROS domain for the offline replay")
    parser.add_argument("--integrate-lateral-acceleration-in-turn", action="store_true",
                        help="offline-only observer A/B; override that parameter to true")
    parser.add_argument("--wheel-burst-catchup-accel-mps2", type=float,
                        help="offline-only parameter override for controlled burst-recovery replay")
    parser.add_argument("--turn-speed-bias-yaw-rate-abs-mps", type=float,
                        help="offline-only turn speed correction per rad/s yaw rate")
    parser.add_argument("--turn-speed-bias-max-mps", type=float,
                        help="offline-only absolute bound paired with turn speed correction")
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
               args.turn_speed_bias_yaw_rate_abs_mps,
               args.turn_speed_bias_max_mps,
               args.start_offset_s, args.duration_wall_s,
               args.source_sequence_index)
    except (OSError, ValueError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"sensor-odometry replay failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
