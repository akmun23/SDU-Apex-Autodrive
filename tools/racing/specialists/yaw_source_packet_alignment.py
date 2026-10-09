"""Exact-source IMU joins used by yaw-model offline diagnostics and fitting."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tools import analyze_open_plane_dynamics as dynamics
from tools.evaluate_open_plane_body_dynamics import _roll_pitch_from_quaternion


def exact_imu_by_odom_receipt(
        bag_path: Path) -> tuple[dict[int, tuple[float, ...]], int]:
    """Return odometry-receipt -> IMU values with the identical source stamp.

    Values are ax, ay, gyro-z, roll, pitch, gyro-x, gyro-y. This reproduces
    the source-stamp coherence expected by SensorPacketAssembler; it does not
    align on receipt-time proximity or use simulator truth.
    """
    connection = sqlite3.connect(
        bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = dynamics._topic_map(connection)
        odom_topic = topics[dynamics.ODOM]
        imu_topic = topics[dynamics.IMU]
        odom_type = get_message(odom_topic[1])
        imu_type = get_message(imu_topic[1])
        source_by_receipt: dict[int, int] = {}
        cursor = connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
            (odom_topic[0],))
        for receipt_ns, payload in cursor:
            message = deserialize_message(bytes(payload), odom_type)
            source_by_receipt[int(receipt_ns)] = dynamics._stamp_ns(
                message.header.stamp)

        imu_by_source: dict[int, tuple[float, ...]] = {}
        cursor = connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
            (imu_topic[0],))
        for _receipt_ns, payload in cursor:
            message = deserialize_message(bytes(payload), imu_type)
            source_ns = dynamics._stamp_ns(message.header.stamp)
            roll_pitch = _roll_pitch_from_quaternion(
                float(message.orientation.x), float(message.orientation.y),
                float(message.orientation.z), float(message.orientation.w))
            if roll_pitch is None:
                continue
            values = (
                float(message.linear_acceleration.x),
                float(message.linear_acceleration.y),
                float(message.angular_velocity.z), roll_pitch[0], roll_pitch[1],
                float(message.angular_velocity.x),
                float(message.angular_velocity.y),
            )
            if source_ns > 0 and np.isfinite(values).all():
                imu_by_source[source_ns] = values
    finally:
        connection.close()
    return ({receipt: imu_by_source[source]
             for receipt, source in source_by_receipt.items()
             if source in imu_by_source}, len(source_by_receipt))


def odom_source_stamp_by_receipt(bag_path: Path) -> dict[int, int]:
    """Return each odometry bag-receipt timestamp's source/header timestamp."""
    connection = sqlite3.connect(
        bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = dynamics._topic_map(connection)
        odom_topic = topics[dynamics.ODOM]
        odom_type = get_message(odom_topic[1])
        result: dict[int, int] = {}
        cursor = connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
            (odom_topic[0],))
        for receipt_ns, payload in cursor:
            message = deserialize_message(bytes(payload), odom_type)
            source_ns = dynamics._stamp_ns(message.header.stamp)
            if source_ns > 0:
                result[int(receipt_ns)] = source_ns
    finally:
        connection.close()
    return result
