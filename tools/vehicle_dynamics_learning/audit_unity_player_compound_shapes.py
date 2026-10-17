#!/usr/bin/env python3
"""Extract active vehicle compound shapes from a Unity player and audit clearance.

Requires the optional UnityPy package only for extraction. The output retains
source-mesh convex-hull support vertices and serialized transforms; it does not
claim to recover Unity's runtime-cooked PhysX convex meshes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "live_runs/unity_exact_binary_audit_20261012/Data"
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "vehicle_compound_shapes_clearance_r07_20261016.json")
CONTACT_OFFSET_PAIR_M = 0.020
WHEEL_VISUALS = {"FL Wheel": "fl", "FR Wheel": "fr",
                 "RL Wheel": "rl", "RR Wheel": "rr"}
WHEEL_COLLIDERS = ("fl", "fr", "rl", "rr")


def _unitypy():
    try:
        import UnityPy
        from UnityPy.helpers.MeshHelper import MeshHandler
    except ImportError as error:
        raise SystemExit(
            "This optional extraction tool requires UnityPy; install it in the "
            "analysis environment, then rerun. No repo/runtime dependency is added.") from error
    return UnityPy, MeshHandler


def _quat_matrix(value: Any) -> np.ndarray:
    x, y, z, w = (float(getattr(value, name)) for name in ("x", "y", "z", "w"))
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm <= 1e-12:
        raise ValueError("Unity transform contains a zero quaternion")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.asarray((
        (1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
        (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
        (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)),
    ), dtype=np.float64)


def _trs(transform: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_matrix(transform.m_LocalRotation) @ np.diag((
        float(transform.m_LocalScale.x), float(transform.m_LocalScale.y),
        float(transform.m_LocalScale.z)))
    matrix[:3, 3] = (float(transform.m_LocalPosition.x),
                     float(transform.m_LocalPosition.y),
                     float(transform.m_LocalPosition.z))
    return matrix


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ points.T).T + matrix[:3, 3]


def _summary(clearance: np.ndarray, support_velocity: np.ndarray | None = None,
             contact_offset_pair_m: float = CONTACT_OFFSET_PAIR_M
             ) -> dict[str, Any]:
    minimum_row = int(np.argmin(clearance))
    result: dict[str, Any] = {
        "samples": int(clearance.size),
        "minimum_clearance_m": float(clearance[minimum_row]),
        "minimum_clearance_row": minimum_row,
        "fraction_negative": float(np.mean(clearance < 0.0)),
        "fraction_within_20mm_contact_offset_pair": float(
            np.mean(clearance <= contact_offset_pair_m)),
    }
    if support_velocity is not None:
        within = clearance <= contact_offset_pair_m
        result["fraction_within_20mm_and_approaching"] = float(
            np.mean(within & (support_velocity < 0.0)))
        result["fraction_within_20mm_and_receding"] = float(
            np.mean(within & (support_velocity > 0.0)))
        result["minimum_support_normal_velocity_mps"] = float(
            support_velocity.min())
    return result


def _serialized_global_settings(data_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    UnityPy, _ = _unitypy()
    environment = UnityPy.load(str(data_dir / "globalgamemanagers"))
    objects = [obj for obj in environment.objects]
    physics = next(obj.read() for obj in objects
                   if obj.type.name == "PhysicsManager")
    time = next(obj.read() for obj in objects if obj.type.name == "TimeManager")
    build = next(obj.read() for obj in objects if obj.type.name == "BuildSettings")
    layer_matrix = physics.m_LayerCollisionMatrix
    default_material = physics.m_DefaultMaterial

    def optional(name: str) -> Any:
        return getattr(physics, name, None)

    physics_values = {
        "gravity_mps2": [float(physics.m_Gravity.x),
                         float(physics.m_Gravity.y),
                         float(physics.m_Gravity.z)],
        "default_contact_offset_m": float(physics.m_DefaultContactOffset),
        "default_max_angular_speed_radps": float(
            physics.m_DefaultMaxAngularSpeed),
        "default_max_depenetration_velocity_mps": float(
            physics.m_DefaultMaxDepenetrationVelocity),
        "default_solver_position_iterations": int(
            physics.m_DefaultSolverIterations),
        "default_solver_velocity_iterations": int(
            physics.m_DefaultSolverVelocityIterations),
        "bounce_threshold_mps": float(physics.m_BounceThreshold),
        "sleep_threshold_m2ps2": float(physics.m_SleepThreshold),
        "fast_motion_threshold_m": float(physics.m_FastMotionThreshold),
        "broadphase_type_serialized": int(physics.m_BroadphaseType),
        "contacts_generation_serialized": int(physics.m_ContactsGeneration),
        "solver_type_serialized": int(physics.m_SolverType),
        "friction_type_serialized": int(physics.m_FrictionType),
        "improved_patch_friction": bool(physics.m_ImprovedPatchFriction),
        "adaptive_force": bool(physics.m_EnableAdaptiveForce),
        "enhanced_determinism": bool(physics.m_EnableEnhancedDeterminism),
        "queries_hit_triggers": bool(physics.m_QueriesHitTriggers),
        "queries_hit_backfaces": bool(physics.m_QueriesHitBackfaces),
        "auto_sync_transforms": bool(physics.m_AutoSyncTransforms),
        "invoke_collision_callbacks": bool(physics.m_InvokeCollisionCallbacks),
        "reuse_collision_callbacks": bool(physics.m_ReuseCollisionCallbacks),
        "enable_pcm_serialized": optional("m_EnablePCM"),
        "simulation_mode_serialized": int(physics.m_SimulationMode),
        "default_material": {
            "file_id": int(default_material.m_FileID),
            "path_id": int(default_material.m_PathID),
            "explicit": bool(default_material.m_PathID),
        },
        "layer_collision_matrix_rows": [int(value) for value in layer_matrix],
        "default_layer0_self_collision_enabled": bool(layer_matrix[0] & 1),
        "default_layer0_collision_row_hex": hex(int(layer_matrix[0])),
        "world_bounds_center_m": [float(physics.m_WorldBounds.m_Center.x),
                                   float(physics.m_WorldBounds.m_Center.y),
                                   float(physics.m_WorldBounds.m_Center.z)],
        "world_bounds_extent_m": [float(physics.m_WorldBounds.m_Extent.x),
                                  float(physics.m_WorldBounds.m_Extent.y),
                                  float(physics.m_WorldBounds.m_Extent.z)],
        "world_subdivisions": int(physics.m_WorldSubdivisions),
    }
    time_values = {
        "unity_version": str(build.m_Version),
        "fixed_timestep_s": float(time.Fixed_Timestep),
        "maximum_allowed_timestep_s": float(time.Maximum_Allowed_Timestep),
        "time_scale": float(time.m_TimeScale),
    }
    return physics_values, time_values


def _wheel_collider_values(environment: Any,
                           game_objects: dict[int, tuple[Any, list[Any]]]
                           ) -> dict[str, Any]:
    wheel_names = {
        "Front Left Wheel Collider": "fl",
        "Front Right Wheel Collider": "fr",
        "Rear Left Wheel Collider": "rl",
        "Rear Right Wheel Collider": "rr",
    }

    def curve(value: Any) -> dict[str, float]:
        return {
            "extremum_slip": float(value.m_ExtremumSlip),
            "extremum_value": float(value.m_ExtremumValue),
            "asymptote_slip": float(value.m_AsymptoteSlip),
            "asymptote_value": float(value.m_AsymptoteValue),
            "stiffness": float(value.m_Stiffness),
        }

    result: dict[str, Any] = {}
    for obj in environment.objects:
        if obj.type.name != "WheelCollider":
            continue
        wheel = obj.read()
        game_object_id = int(wheel.m_GameObject.m_PathID)
        game_object = game_objects[game_object_id][0]
        name = game_object.m_Name
        if name not in wheel_names:
            continue
        transform = next(component.read()
                         for component in game_objects[game_object_id][1]
                         if component.type.name == "Transform")
        spring = wheel.m_SuspensionSpring
        result[wheel_names[name]] = {
            "component_path_id": int(obj.path_id),
            "game_object": name,
            "enabled": bool(wheel.m_Enabled),
            "layer": int(game_object.m_Layer),
            "center_local_m": [float(wheel.m_Center.x),
                                float(wheel.m_Center.y),
                                float(wheel.m_Center.z)],
            "transform_local_position_m": [float(transform.m_LocalPosition.x),
                                            float(transform.m_LocalPosition.y),
                                            float(transform.m_LocalPosition.z)],
            "transform_local_rotation_xyzw": [
                float(transform.m_LocalRotation.x),
                float(transform.m_LocalRotation.y),
                float(transform.m_LocalRotation.z),
                float(transform.m_LocalRotation.w)],
            "radius_m": float(wheel.m_Radius),
            "mass_kg": float(wheel.m_Mass),
            "wheel_damping_rate": float(wheel.m_WheelDampingRate),
            "suspension_distance_m": float(wheel.m_SuspensionDistance),
            "suspension_spring_npm": float(spring.spring),
            "suspension_damper_nspm": float(spring.damper),
            "suspension_target_position": float(spring.targetPosition),
            "force_application_point_distance_m": float(
                wheel.m_ForceAppPointDistance),
            "forward_friction": curve(wheel.m_ForwardFriction),
            "sideways_friction": curve(wheel.m_SidewaysFriction),
            "include_layers_bits": int(wheel.m_IncludeLayers.m_Bits),
            "exclude_layers_bits": int(wheel.m_ExcludeLayers.m_Bits),
            "layer_override_priority": int(wheel.m_LayerOverridePriority),
            "provides_contacts": bool(wheel.m_ProvidesContacts),
        }
    if set(result) != set(WHEEL_COLLIDERS):
        raise ValueError(f"expected four active WheelColliders, got {sorted(result)}")
    return result


def _metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, separator, value = line[2:].strip().partition("=")
            if separator:
                values[key] = value
    return values


def _hash(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def extract_and_audit(data_dir: Path, capture: Path,
                      inspect_rows: tuple[int, ...]) -> dict[str, Any]:
    UnityPy, MeshHandler = _unitypy()
    level0 = data_dir / "level0"
    environment = UnityPy.load(str(level0))
    objects = {obj.path_id: obj for obj in environment.objects}
    game_objects: dict[int, tuple[Any, list[Any]]] = {}
    for obj in environment.objects:
        if obj.type.name == "GameObject":
            game_object = obj.read()
            components = [objects[item.m_PathID]
                          for item in game_object.m_Components
                          if item.m_PathID in objects]
            game_objects[obj.path_id] = (game_object, components)

    def transform_for(game_object_id: int) -> Any:
        return next((component.read() for component in game_objects[game_object_id][1]
                     if component.type.name == "Transform"), None)

    def relative_matrix(game_object_id: int, ancestor_id: int) -> np.ndarray:
        chain: list[np.ndarray] = []
        current = game_object_id
        while current != ancestor_id:
            transform = transform_for(current)
            if transform is None or transform.m_Father.m_PathID == 0:
                raise ValueError("collider is not parented to the vehicle Rigidbody")
            chain.append(_trs(transform))
            father = objects[transform.m_Father.m_PathID].read()
            current = father.m_GameObject.m_PathID
        matrix = np.eye(4, dtype=np.float64)
        for local in reversed(chain):
            matrix = matrix @ local
        return matrix

    def is_active_in_hierarchy(game_object_id: int) -> bool:
        current = game_object_id
        while current:
            game_object = game_objects[current][0]
            if not game_object.m_IsActive:
                return False
            transform = transform_for(current)
            if transform is None or transform.m_Father.m_PathID == 0:
                break
            current = objects[transform.m_Father.m_PathID].read().m_GameObject.m_PathID
        return True

    rigidbodies = [obj for obj in environment.objects
                   if obj.type.name == "Rigidbody"]
    vehicle_body = next((obj for obj in rigidbodies
                         if game_objects[obj.read().m_GameObject.m_PathID][0].m_Name
                         == "Roboracer"), None)
    if vehicle_body is None:
        raise ValueError("could not uniquely locate the Roboracer Rigidbody")
    body = vehicle_body.read()
    body_game_object_id = body.m_GameObject.m_PathID

    meshes: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    disabled_vehicle_mesh_colliders: list[dict[str, Any]] = []
    wheel_hulls: dict[str, dict[str, Any]] = {}

    for obj in environment.objects:
        if obj.type.name not in ("MeshCollider", "BoxCollider"):
            continue
        collider = obj.read()
        game_object_id = collider.m_GameObject.m_PathID
        game_object = game_objects[game_object_id][0]
        try:
            body_matrix = relative_matrix(game_object_id, body_game_object_id)
        except ValueError:
            continue
        active = (bool(collider.m_Enabled)
                  and is_active_in_hierarchy(game_object_id))
        trigger = bool(getattr(collider, "m_IsTrigger", False))
        name = game_object.m_Name

        if obj.type.name == "MeshCollider":
            if not active:
                disabled_vehicle_mesh_colliders.append({
                    "component_path_id": obj.path_id,
                    "game_object": name,
                    "mesh_collider_enabled": bool(collider.m_Enabled),
                })
                continue
            if trigger or not bool(collider.m_Convex):
                continue
            mesh = collider.m_Mesh.read()
            handler = MeshHandler(mesh)
            handler.process()
            source_vertices = np.asarray(handler.m_Vertices, dtype=np.float64)
            if source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
                raise ValueError(f"invalid source mesh vertices for {mesh.m_Name}")
            parent_id = transform_for(game_object_id).m_Father.m_PathID
            parent_name = game_objects[
                objects[parent_id].read().m_GameObject.m_PathID][0].m_Name
            if name == "Collider" and parent_name in WHEEL_VISUALS:
                side = WHEEL_VISUALS[parent_name]
                child_matrix = relative_matrix(game_object_id,
                                               objects[parent_id].read().m_GameObject.m_PathID)
                hull = source_vertices[ConvexHull(source_vertices).vertices]
                wheel_hulls[side] = {
                    "component_path_id": obj.path_id,
                    "visual_parent": parent_name,
                    "mesh_name": mesh.m_Name,
                    "mesh_path_id": int(mesh.object_reader.path_id),
                    "convex": True,
                    "cooking_options_serialized": int(collider.m_CookingOptions),
                    "game_object_layer": int(game_object.m_Layer),
                    "material": {
                        "file_id": int(collider.m_Material.m_FileID),
                        "path_id": int(collider.m_Material.m_PathID),
                        "explicit": bool(collider.m_Material.m_PathID),
                    },
                    "include_layers_bits": int(collider.m_IncludeLayers.m_Bits),
                    "exclude_layers_bits": int(collider.m_ExcludeLayers.m_Bits),
                    "layer_override_priority": int(collider.m_LayerOverridePriority),
                    "provides_contacts": bool(collider.m_ProvidesContacts),
                    "source_vertex_count": int(source_vertices.shape[0]),
                    "source_hull_vertex_count": int(hull.shape[0]),
                    "source_hull_vertices_mesh_local_m": hull.tolist(),
                    "collider_local_to_wheel_pose_matrix": child_matrix.tolist(),
                }
                continue

            transformed = _apply(body_matrix, source_vertices)
            hull = transformed[ConvexHull(transformed).vertices]
            meshes.append({
                "component_path_id": obj.path_id,
                "game_object": name,
                "mesh_name": mesh.m_Name,
                "mesh_path_id": int(mesh.object_reader.path_id),
                "convex": True,
                "trigger": False,
                "cooking_options_serialized": int(collider.m_CookingOptions),
                "game_object_layer": int(game_object.m_Layer),
                "material": {
                    "file_id": int(collider.m_Material.m_FileID),
                    "path_id": int(collider.m_Material.m_PathID),
                    "explicit": bool(collider.m_Material.m_PathID),
                },
                "include_layers_bits": int(collider.m_IncludeLayers.m_Bits),
                "exclude_layers_bits": int(collider.m_ExcludeLayers.m_Bits),
                "layer_override_priority": int(collider.m_LayerOverridePriority),
                "provides_contacts": bool(collider.m_ProvidesContacts),
                "source_vertex_count": int(source_vertices.shape[0]),
                "source_hull_vertex_count": int(hull.shape[0]),
                "source_hull_vertices_body_local_m": hull.tolist(),
            })
        else:
            if not active or trigger:
                continue
            center = np.asarray((collider.m_Center.x, collider.m_Center.y,
                                 collider.m_Center.z), dtype=np.float64)
            half = np.asarray((collider.m_Size.x, collider.m_Size.y,
                               collider.m_Size.z), dtype=np.float64) * 0.5
            corners = np.asarray([
                (x, y, z)
                for x in (-half[0], half[0])
                for y in (-half[1], half[1])
                for z in (-half[2], half[2])], dtype=np.float64) + center
            boxes.append({
                "component_path_id": obj.path_id,
                "game_object": name,
                "trigger": False,
                "game_object_layer": int(game_object.m_Layer),
                "material": {
                    "file_id": int(collider.m_Material.m_FileID),
                    "path_id": int(collider.m_Material.m_PathID),
                    "explicit": bool(collider.m_Material.m_PathID),
                },
                "include_layers_bits": int(collider.m_IncludeLayers.m_Bits),
                "exclude_layers_bits": int(collider.m_ExcludeLayers.m_Bits),
                "layer_override_priority": int(collider.m_LayerOverridePriority),
                "provides_contacts": bool(collider.m_ProvidesContacts),
                "center_collider_local_m": center.tolist(),
                "size_collider_local_m": (half * 2.0).tolist(),
                "corners_body_local_m": _apply(body_matrix, corners).tolist(),
            })

    if set(wheel_hulls) != set(WHEEL_COLLIDERS):
        raise ValueError(f"expected four wheel mesh colliders, got {sorted(wheel_hulls)}")

    physics_settings, time_settings = _serialized_global_settings(data_dir)

    floor_shape = None
    for obj in environment.objects:
        if obj.type.name != "BoxCollider":
            continue
        collider = obj.read()
        game_object = game_objects[collider.m_GameObject.m_PathID][0]
        if game_object.m_Name != "Floor" or not collider.m_Enabled:
            continue
        floor_transform = transform_for(collider.m_GameObject.m_PathID)
        center = np.asarray((collider.m_Center.x, collider.m_Center.y,
                             collider.m_Center.z), dtype=np.float64)
        half = np.asarray((collider.m_Size.x, collider.m_Size.y,
                           collider.m_Size.z), dtype=np.float64) * 0.5
        corners = np.asarray([
            (x, y, z)
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])], dtype=np.float64) + center
        chain: list[np.ndarray] = []
        current = collider.m_GameObject.m_PathID
        while current:
            transform = transform_for(current)
            if transform is None:
                break
            chain.append(_trs(transform))
            if transform.m_Father.m_PathID == 0:
                break
            current = objects[transform.m_Father.m_PathID].read().m_GameObject.m_PathID
        world = np.eye(4, dtype=np.float64)
        for local in reversed(chain):
            world = world @ local
        world_corners = _apply(world, corners)
        floor_shape = {
            "component_path_id": int(obj.path_id),
            "layer": int(game_object.m_Layer),
            "material": {
                "file_id": int(collider.m_Material.m_FileID),
                "path_id": int(collider.m_Material.m_PathID),
                "explicit": bool(collider.m_Material.m_PathID),
            },
            "include_layers_bits": int(collider.m_IncludeLayers.m_Bits),
            "exclude_layers_bits": int(collider.m_ExcludeLayers.m_Bits),
            "layer_override_priority": int(collider.m_LayerOverridePriority),
            "provides_contacts": bool(collider.m_ProvidesContacts),
            "world_y_min_m": float(world_corners[:, 1].min()),
            "world_y_max_m": float(world_corners[:, 1].max()),
            "collider_size_local_m": (half * 2.0).tolist(),
        }
        break
    if floor_shape is None:
        raise ValueError("active scene Floor BoxCollider was not found")

    metadata = _metadata(capture)
    wheel_collider_settings = _wheel_collider_values(environment, game_objects)
    if float(metadata["fixed_delta_time_s"]) != 0.001:
        raise ValueError("clearance audit capture must use a 1 ms physics step")
    wheel_columns = [f"{wheel}_wheel_pose_body_{axis}"
                     for wheel in WHEEL_COLLIDERS for axis in "xyz"]
    wheel_columns += [f"{wheel}_wheel_pose_q{axis}"
                      for wheel in WHEEL_COLLIDERS for axis in "xyzw"]
    columns = ["sim_time_s", "pos_world_y_m", "rot_world_x", "rot_world_y",
               "rot_world_z", "rot_world_w", "vel_body_x_mps",
               "vel_body_y_mps", "vel_body_z_mps", "omega_body_x_rps",
               "omega_body_y_rps", "omega_body_z_rps", *wheel_columns]
    frame = pd.read_csv(capture, comment="#", usecols=columns)
    position_y = frame["pos_world_y_m"].to_numpy(dtype=np.float64)
    body_q = frame[["rot_world_x", "rot_world_y", "rot_world_z",
                    "rot_world_w"]].to_numpy(dtype=np.float64)
    body_q /= np.maximum(np.linalg.norm(body_q, axis=1)[:, None], 1e-12)
    qx, qy, qz, qw = body_q.T
    # R(body)^T * world-up; yaw therefore does not alter this vector.
    body_up_world = np.column_stack((
        2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),
        2*(qy*qz - qx*qw)))
    body_velocity = frame[["vel_body_x_mps", "vel_body_y_mps",
                           "vel_body_z_mps"]].to_numpy(dtype=np.float64)
    body_omega = frame[["omega_body_x_rps", "omega_body_y_rps",
                        "omega_body_z_rps"]].to_numpy(dtype=np.float64)

    body_clearance: dict[str, Any] = {}
    for shape in meshes:
        points = np.asarray(shape["source_hull_vertices_body_local_m"],
                            dtype=np.float64)
        support_index = np.argmin(body_up_world @ points.T, axis=1)
        support = points[support_index]
        clearance = position_y + np.einsum("ij,ij->i", body_up_world, support)
        point_velocity = body_velocity + np.cross(body_omega, support)
        support_velocity = np.einsum("ij,ij->i", body_up_world, point_velocity)
        stats = _summary(clearance, support_velocity,
                         2.0 * physics_settings["default_contact_offset_m"])
        stats["minimum_clearance_time_s"] = float(
            frame.iloc[stats["minimum_clearance_row"]]["sim_time_s"])
        body_clearance[
            f"{shape['game_object']}#{shape['component_path_id']}"] = stats
    for shape in boxes:
        points = np.asarray(shape["corners_body_local_m"], dtype=np.float64)
        support_index = np.argmin(body_up_world @ points.T, axis=1)
        support = points[support_index]
        clearance = position_y + np.einsum("ij,ij->i", body_up_world, support)
        point_velocity = body_velocity + np.cross(body_omega, support)
        support_velocity = np.einsum("ij,ij->i", body_up_world, point_velocity)
        stats = _summary(clearance, support_velocity,
                         2.0 * physics_settings["default_contact_offset_m"])
        stats["minimum_clearance_time_s"] = float(
            frame.iloc[stats["minimum_clearance_row"]]["sim_time_s"])
        body_clearance[
            f"{shape['game_object']}#{shape['component_path_id']}"] = stats

    wheel_clearance: dict[str, Any] = {}
    for wheel, shape in wheel_hulls.items():
        local = np.asarray(shape["source_hull_vertices_mesh_local_m"],
                           dtype=np.float64)
        matrix = np.asarray(shape["collider_local_to_wheel_pose_matrix"],
                            dtype=np.float64)
        wheel_points = _apply(matrix, local)
        position_columns = [f"{wheel}_wheel_pose_body_{axis}" for axis in "xyz"]
        rotation_columns = [f"{wheel}_wheel_pose_q{axis}" for axis in "xyzw"]
        wheel_position = frame[position_columns].to_numpy(dtype=np.float64)
        wheel_q = frame[rotation_columns].to_numpy(dtype=np.float64)
        wheel_q /= np.maximum(np.linalg.norm(wheel_q, axis=1)[:, None], 1e-12)
        clearances: list[np.ndarray] = []
        block = 4096
        for start in range(0, len(frame), block):
            stop = min(start + block, len(frame))
            q = wheel_q[start:stop]
            xyz, scalar = q[:, :3], q[:, 3, None]
            points = np.broadcast_to(wheel_points, (len(q), *wheel_points.shape))
            twice_cross = 2.0 * np.cross(xyz[:, None, :], points)
            rotated = (points + scalar[:, None, :] * twice_cross
                       + np.cross(xyz[:, None, :], twice_cross))
            in_body = wheel_position[start:stop, None, :] + rotated
            support = np.min(np.sum(
                in_body * body_up_world[start:stop, None, :], axis=2), axis=1)
            clearances.append(position_y[start:stop] + support)
        values = np.concatenate(clearances)
        stats = _summary(values, contact_offset_pair_m=(
            2.0 * physics_settings["default_contact_offset_m"]))
        stats["minimum_clearance_time_s"] = float(
            frame.iloc[stats["minimum_clearance_row"]]["sim_time_s"])
        wheel_clearance[wheel] = stats

    inspected_rows: dict[str, Any] = {}
    for row in inspect_rows:
        if row < 0 or row >= len(frame):
            raise ValueError(f"inspection row {row} outside capture length {len(frame)}")
        direction = body_up_world[row]
        entry: dict[str, Any] = {
            "time_s": float(frame.iloc[row]["sim_time_s"]),
            "clearance_m_by_body_shape": {},
            "support_normal_velocity_mps_by_body_shape": {},
            "clearance_m_by_wheel_shape": {},
        }
        for shape in meshes:
            points = np.asarray(shape["source_hull_vertices_body_local_m"],
                                dtype=np.float64)
            support = points[int(np.argmin(points @ direction))]
            label = f"{shape['game_object']}#{shape['component_path_id']}"
            entry["clearance_m_by_body_shape"][label] = float(
                position_y[row] + direction @ support)
            velocity = body_velocity[row] + np.cross(body_omega[row], support)
            entry["support_normal_velocity_mps_by_body_shape"][label] = float(
                direction @ velocity)
        for shape in boxes:
            points = np.asarray(shape["corners_body_local_m"], dtype=np.float64)
            support = points[int(np.argmin(points @ direction))]
            label = f"{shape['game_object']}#{shape['component_path_id']}"
            entry["clearance_m_by_body_shape"][label] = float(
                position_y[row] + direction @ support)
            velocity = body_velocity[row] + np.cross(body_omega[row], support)
            entry["support_normal_velocity_mps_by_body_shape"][label] = float(
                direction @ velocity)
        for wheel, shape in wheel_hulls.items():
            source = np.asarray(shape["source_hull_vertices_mesh_local_m"],
                                dtype=np.float64)
            matrix = np.asarray(shape["collider_local_to_wheel_pose_matrix"],
                                dtype=np.float64)
            local_points = _apply(matrix, source)
            q = frame.loc[row, [f"{wheel}_wheel_pose_q{axis}" for axis in "xyzw"]]
            q = np.asarray(q, dtype=np.float64)
            q /= max(float(np.linalg.norm(q)), 1e-12)
            xyz, scalar = q[:3], q[3]
            twice_cross = 2.0 * np.cross(xyz, local_points)
            rotated = (local_points + scalar * twice_cross
                       + np.cross(xyz, twice_cross))
            wheel_position = frame.loc[
                row, [f"{wheel}_wheel_pose_body_{axis}" for axis in "xyz"]
            ].to_numpy(dtype=np.float64)
            in_body = wheel_position + rotated
            entry["clearance_m_by_wheel_shape"][wheel] = float(
                position_y[row] + np.min(in_body @ direction))
        inspected_rows[str(row)] = entry

    binary_files = [data_dir / name for name in (
        "globalgamemanagers", "level0", "sharedassets0.assets",
        "sharedassets0.assets.resS")
        if (data_dir / name).is_file()]
    game_assembly = data_dir.parent / "GameAssembly.so"
    if game_assembly.is_file():
        binary_files.append(game_assembly)
    return {
        "provenance": {
            "unity_version_from_capture": metadata.get("unity_version"),
            "unity_version_from_player_build_settings": time_settings[
                "unity_version"],
            "run_id": metadata.get("run_id"),
            "capture": str(capture.resolve().relative_to(ROOT)),
            "capture_rows": int(len(frame)),
            "physics_step_s": float(metadata["fixed_delta_time_s"]),
            "player_asset_files": [_hash(path) for path in binary_files],
        },
        "vehicle_rigidbody": {
            "game_object": "Roboracer",
            "component_path_id": int(vehicle_body.path_id),
            "mass_kg": float(body.m_Mass),
            "use_gravity": bool(body.m_UseGravity),
            "collision_detection_mode_serialized": int(body.m_CollisionDetection),
            "implicit_center_of_mass": bool(body.m_ImplicitCom),
            "implicit_inertia_tensor": bool(body.m_ImplicitTensor),
            "runtime_center_of_mass_body_m": [float(value) for value in
                metadata["center_of_mass_body"].split(",")],
            "runtime_inertia_tensor_body_kgm2": [float(value) for value in
                metadata["inertia_tensor_body"].split(",")],
            "runtime_inertia_tensor_rotation_body_xyzw": [float(value) for value in
                metadata["inertia_tensor_rotation_body"].split(",")],
        },
        "serialized_physics_settings": physics_settings,
        "serialized_time_settings": time_settings,
        "serialized_wheel_collider_settings": wheel_collider_settings,
        "active_open_plane_floor": floor_shape,
        "active_vehicle_compound_shape_counts": {
            "convex_mesh_colliders": len(meshes) + len(wheel_hulls),
            "box_colliders": len(boxes),
            "total": len(meshes) + len(wheel_hulls) + len(boxes),
            "inactive_vehicle_mesh_colliders_excluded": len(
                disabled_vehicle_mesh_colliders),
        },
        "active_body_convex_meshes_source_hulls": meshes,
        "active_body_box_colliders": boxes,
        "active_wheel_mesh_colliders": wheel_hulls,
        "inactive_vehicle_mesh_colliders": disabled_vehicle_mesh_colliders,
        "body_shape_clearance_to_open_plane_y0": body_clearance,
        "wheel_shape_clearance_to_open_plane_y0": wheel_clearance,
        "inspected_rows": inspected_rows,
        "contact_offset_pair_m": 2.0 * physics_settings[
            "default_contact_offset_m"],
        "interpretation": (
            "The support points are convex hulls of source Mesh vertices after "
            "serialized body-local transforms; wheel meshes use captured "
            "WheelCollider.GetWorldPose plus the exact player child transform. "
            "Unity's serialized Mesh records contain no baked convex-cooking "
            "bytes, so these are not asserted to equal PhysX's runtime-cooked "
            "hulls. A gap below 20 mm is within the two-collider default "
            "contact-offset band, not proof of a solver impulse. Negative gap "
            "is source-hull penetration only, not measured contact-manifold data."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inspect-row", type=int, action="append")
    args = parser.parse_args()
    rows = tuple(args.inspect_row if args.inspect_row is not None
                 else (142075, 181366))
    report = extract_and_audit(args.player_data, args.capture, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "active_vehicle_compound_shape_counts": report[
            "active_vehicle_compound_shape_counts"],
        "body_shape_clearance": {
            name: {key: value for key, value in stats.items()
                   if key in ("minimum_clearance_m", "fraction_negative",
                              "fraction_within_20mm_contact_offset_pair",
                              "fraction_within_20mm_and_approaching")}
            for name, stats in report["body_shape_clearance_to_open_plane_y0"].items()},
        "wheel_shape_clearance": report["wheel_shape_clearance_to_open_plane_y0"],
        "inspected_rows": report["inspected_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
