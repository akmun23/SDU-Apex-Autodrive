"""Create historical CasADi wheel model from *current* checked-in source values.

No removed Unity capture is required. In contrast with the original v1.3
model, the source of each current parameter is a checked-in C++/JSON file.
The mathematical force integration is still the historical approximation;
validation against native PhysX is a separate mandatory stage.
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np


class UnresolvedCppSymbol(RuntimeError):
    """A named constant has no readable definition in the source text."""


def _validate_reset_wheel_geometry(index, wheel, wheel_x, wheel_y,
                                   transform_z, radius):
    """Require reset-state wheel centres and force points to match source.

    Reset JSON uses Unity right/up/forward; the plant/OCP use
    forward/left/up. Keep both checks together so a source-frame error cannot
    silently pass just because the centre happens to match.
    """
    reset_wheel = tuple(float(value) for value in
                        wheel['wheel_center_offset_com_body_m'])
    reset_in_source = (reset_wheel[2], -reset_wheel[0], reset_wheel[1])
    expected_center = (wheel_x, wheel_y, transform_z)
    if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8)
           for actual, expected in zip(reset_in_source, expected_center)):
        raise RuntimeError(
            f'Reset wheel {index} centre disagrees with production plant')

    reset_force_point = tuple(float(value) for value in
                              wheel['force_application_offset_com_body_m'])
    expected_force_point = (-wheel_y, transform_z - radius, wheel_x)
    if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8)
           for actual, expected in zip(reset_force_point,
                                      expected_force_point)):
        raise RuntimeError(
            f'Reset wheel {index} force point disagrees with production plant')


def _unity_scene_vehicle_parameters(repo: Path) -> dict:
    """Resolve the practice-scene overrides for the F1TENTH controller.

    The prefab default is not the runtime value: the practice scene overrides
    MotorTorque. Bind the override to the VehicleController component through
    Unity's serialized prefab GUID and source file ID so a stale/default-only
    parse cannot silently change the OCP torque model.
    """
    unity = Path(repo) / 'external/autodrive-unity-source'
    prefab_path = unity / 'Assets/Prefabs/F1TENTH/F1TENTH.prefab'
    prefab_meta_path = Path(str(prefab_path) + '.meta')
    controller_meta_path = unity / 'Assets/Scripts/VehicleController.cs.meta'
    controller_path = unity / 'Assets/Scripts/VehicleController.cs'
    scene_path = unity / 'Assets/Scenes/RoboRacer - Sim Racing.unity'
    required = (prefab_path, prefab_meta_path, controller_meta_path,
                controller_path, scene_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Cannot verify practice Unity vehicle config; missing: '
                           + ', '.join(missing))

    def guid(path: Path) -> str:
        matches = re.findall(r'(?m)^guid:\s*([0-9a-f]+)\s*$',
                             path.read_text(encoding='utf-8'))
        if len(matches) != 1:
            raise RuntimeError(f'Expected one Unity asset GUID in {path}')
        return matches[0]

    prefab_guid = guid(prefab_meta_path)
    controller_guid = guid(controller_meta_path)
    prefab = prefab_path.read_text(encoding='utf-8')
    scene = scene_path.read_text(encoding='utf-8')
    controller_source = controller_path.read_text(encoding='utf-8')
    controller_code = re.sub(r'/\*.*?\*/|//[^\n]*', '', controller_source,
                             flags=re.S)
    compact_controller = re.sub(r'\s+', '', controller_code)
    source_contract = (
        'VehicleRigidBody.centerOfMass=COM;',
        'BrakeTorque=MotorTorque;',
        'elseDriveTorque=ThrottleLimit*AutonomousThrottle*MotorTorque;',
        'FrontLeftWheelCollider.motorTorque=DriveTorque/4;',
        'FrontRightWheelCollider.motorTorque=DriveTorque/4;',
        'RearLeftWheelCollider.motorTorque=DriveTorque/4;',
        'RearRightWheelCollider.motorTorque=DriveTorque/4;',
        'if(brakeType==BrakeType.CAWB){RearLeftWheelCollider.brakeTorque=BrakeTorque;',
        'RearRightWheelCollider.brakeTorque=BrakeTorque;',
        'FrontLeftWheelCollider.brakeTorque=BrakeTorque;',
        'FrontRightWheelCollider.brakeTorque=BrakeTorque;',
        'publicenumDriveType{IRWD,IFWD,IAWD,CRWD,CFWD,CAWD,SkidSteer};',
        'publicenumSteerType{FrontWheelSteer,FourWheelSteer,WedgeWheelSteer};',
        'elseif(driveType==DriveType.CAWD){FrontLeftWheelCollider.motorTorque=DriveTorque/4;',
        'if(driveType!=DriveType.SkidSteer){currentT=DriveTorque/MotorTorque;',
        'if(steerType==SteerType.FrontWheelSteer)',
        'FrontLeftWheelCollider.steerAngle=Mathf.Rad2Deg*(Mathf.Atan(',
    )
    missing_contract = [item for item in source_contract
                        if item not in compact_controller]
    if missing_contract:
        raise RuntimeError('Unity VehicleController control logic changed; '
                           're-derive the plant before IPOPT: '
                           + ', '.join(missing_contract))
    blocks = re.split(r'(?m)(?=^--- !u!\d+ &)', prefab)
    controller_blocks = [block for block in blocks
                         if re.search(r'(?m)^\s*m_Script:\s*\{fileID:\s*11500000,\s*'
                                      r'guid:\s*' + re.escape(controller_guid) + r'\b', block)]
    if len(controller_blocks) != 1:
        raise RuntimeError('Practice F1TENTH prefab must contain exactly one '
                           'VehicleController component')
    component_block = controller_blocks[0]
    component_match = re.match(r'(?m)^--- !u!\d+ &(\d+)\s*$', component_block)
    if component_match is None:
        raise RuntimeError('Could not resolve serialized VehicleController file ID')
    component_file_id = component_match.group(1)

    prefab_blocks = {}
    for block in blocks:
        header = re.match(r'(?m)^--- !u!\d+ &(\d+)\s*$', block)
        if header is not None:
            prefab_blocks[header.group(1)] = block

    def serialized_vector(block: str, field: str, keys: str) -> tuple[float, ...]:
        number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
        names = tuple(keys)
        pattern = (r'(?m)^\s*' + re.escape(field) + r':\s*\{'
                   + r',\s*'.join(re.escape(key) + r':\s*(' + number + r')'
                                   for key in names)
                   + r'\}\s*$')
        matches = re.findall(pattern, block)
        if len(matches) != 1:
            raise RuntimeError(f'Expected one serialized {field} vector in F1TENTH prefab')
        return tuple(float(value) for value in matches[0])

    def serialized_number(block: str, field: str) -> float:
        matches = re.findall(r'(?m)^\s*' + re.escape(field) + r':\s*'
                             r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$',
                             block)
        if len(matches) != 1:
            raise RuntimeError(f'Expected one serialized {field} in F1TENTH prefab')
        return float(matches[0])

    drive_type = int(serialized_number(component_block, 'driveType'))
    brake_type = int(serialized_number(component_block, 'brakeType'))
    steer_type = int(serialized_number(component_block, 'steerType'))
    prefab_torque = serialized_number(component_block, 'MotorTorque')
    throttle_limit = serialized_number(component_block, 'ThrottleLimit')
    steering_limit_deg = serialized_number(component_block, 'SteeringLimit')
    steering_rate_degps = serialized_number(component_block, 'SteeringRate')
    wheelbase_mm = serialized_number(component_block, 'Wheelbase')
    track_width_mm = serialized_number(component_block, 'TrackWidth')
    controller_wheel_radius_m = serialized_number(component_block, 'WheelRadius')
    prefab_driving_mode = int(serialized_number(component_block, 'DrivingMode'))
    prefab_com = serialized_vector(component_block, 'COM', 'xyz')
    vehicle_refs = re.findall(
        r'(?m)^\s*Vehicle:\s*\{fileID:\s*(\d+)\}\s*$',
        component_block)
    if len(vehicle_refs) != 1:
        raise RuntimeError('Could not resolve practice vehicle root object')
    vehicle_transforms = [
        block for block in prefab_blocks.values()
        if re.match(r'(?m)^--- !u!4 &\d+\s*$', block)
        and re.search(r'(?m)^\s*m_GameObject:\s*\{fileID:\s*'
                      + re.escape(vehicle_refs[0]) + r'\}\s*$', block)]
    if len(vehicle_transforms) != 1:
        raise RuntimeError('Could not resolve practice vehicle root Transform')
    vehicle_transform_id = re.match(
        r'(?m)^--- !u!4 &(\d+)\s*$', vehicle_transforms[0]).group(1)
    rigidbody_refs = re.findall(
        r'(?m)^\s*VehicleRigidBody:\s*\{fileID:\s*(\d+)\}\s*$',
        component_block)
    if len(rigidbody_refs) != 1 or rigidbody_refs[0] not in prefab_blocks:
        raise RuntimeError('Could not resolve practice Vehicle Rigidbody component')
    rigidbody_mass = serialized_number(
        prefab_blocks[rigidbody_refs[0]], 'm_Mass')
    rigidbody_drag = serialized_number(
        prefab_blocks[rigidbody_refs[0]], 'm_Drag')
    rigidbody_angular_drag = serialized_number(
        prefab_blocks[rigidbody_refs[0]], 'm_AngularDrag')
    rigidbody_use_gravity = int(serialized_number(
        prefab_blocks[rigidbody_refs[0]], 'm_UseGravity'))
    rigidbody_is_kinematic = int(serialized_number(
        prefab_blocks[rigidbody_refs[0]], 'm_IsKinematic'))
    rigidbody_constraints = int(serialized_number(
        prefab_blocks[rigidbody_refs[0]], 'm_Constraints'))

    wheel_geometry = []
    wheel_parameters = []
    wheel_parent_ids = set()
    wheel_component_index = {}
    wheel_transform_ids = set()
    for wheel_name in ('FrontLeft', 'FrontRight', 'RearLeft', 'RearRight'):
        component_ids = re.findall(
            r'(?m)^\s*' + wheel_name
            + r'WheelCollider:\s*\{fileID:\s*(\d+)\}\s*$',
            component_block)
        if len(component_ids) != 1 or component_ids[0] not in prefab_blocks:
            raise RuntimeError(f'Could not resolve {wheel_name} WheelCollider component')
        wheel_component_index[component_ids[0]] = len(wheel_parameters)
        collider_block = prefab_blocks[component_ids[0]]
        game_objects = re.findall(r'(?m)^\s*m_GameObject:\s*\{fileID:\s*(\d+)\}\s*$',
                                  collider_block)
        if len(game_objects) != 1:
            raise RuntimeError(f'Could not resolve {wheel_name} WheelCollider object')
        transform_blocks = [
            block for block in prefab_blocks.values()
            if re.match(r'(?m)^--- !u!4 &\d+\s*$', block)
            and re.search(r'(?m)^\s*m_GameObject:\s*\{fileID:\s*'
                          + re.escape(game_objects[0]) + r'\}\s*$', block)]
        if len(transform_blocks) != 1:
            raise RuntimeError(f'Expected one {wheel_name} WheelCollider transform')
        transform_block = transform_blocks[0]
        transform_id = re.match(
            r'(?m)^--- !u!4 &(\d+)\s*$', transform_block).group(1)
        wheel_transform_ids.add(transform_id)
        fathers = re.findall(r'(?m)^\s*m_Father:\s*\{fileID:\s*(\d+)\}\s*$',
                             transform_block)
        if len(fathers) != 1:
            raise RuntimeError(f'Could not resolve {wheel_name} transform parent')
        wheel_parent_ids.add(fathers[0])
        forward_block = re.search(
            r'(?ms)^\s*m_ForwardFriction:\s*\n(.*?)'
            r'(?=^\s*m_SidewaysFriction:)', collider_block)
        sideways_block = re.search(
            r'(?ms)^\s*m_SidewaysFriction:\s*\n(.*?)'
            r'(?=^\s*m_IncludeLayers:)', collider_block)
        if forward_block is None or sideways_block is None:
            raise RuntimeError(f'Could not resolve {wheel_name} serialized tire curves')
        curve_fields = ('m_ExtremumSlip', 'm_ExtremumValue',
                        'm_AsymptoteSlip', 'm_AsymptoteValue', 'm_Stiffness')
        wheel_parameters.append({
            'name': wheel_name,
            'radius_m': serialized_number(collider_block, 'm_Radius'),
            'mass_kg': serialized_number(collider_block, 'm_Mass'),
            'damping_rate_nms': serialized_number(
                collider_block, 'm_WheelDampingRate'),
            'suspension_distance_m': serialized_number(
                collider_block, 'm_SuspensionDistance'),
            'suspension_spring_npm': serialized_number(
                collider_block, 'spring'),
            'suspension_damper_ns_per_m': serialized_number(
                collider_block, 'damper'),
            'suspension_target_position': serialized_number(
                collider_block, 'targetPosition'),
            'force_app_point_distance_m': serialized_number(
                collider_block, 'm_ForceAppPointDistance'),
            'forward_curve': tuple(serialized_number(forward_block.group(1), key)
                                   for key in curve_fields),
            'sideways_curve': tuple(serialized_number(sideways_block.group(1), key)
                                   for key in curve_fields),
        })
        wheel_geometry.append({
            'name': wheel_name,
            'transform_id': int(transform_id),
            'transform_position_right_up_forward_m': serialized_vector(
                transform_block, 'm_LocalPosition', 'xyz'),
            'transform_rotation_xyzw': serialized_vector(
                transform_block, 'm_LocalRotation', 'xyzw'),
            'center_right_up_forward_m': serialized_vector(
                collider_block, 'm_Center', 'xyz'),
        })
    if len(wheel_parent_ids) != 1:
        raise RuntimeError('Practice WheelColliders do not share one transform parent')
    parent_id = next(iter(wheel_parent_ids))
    parent_block = prefab_blocks.get(parent_id)
    if parent_block is None or not re.match(r'(?m)^--- !u!4 &\d+\s*$', parent_block):
        raise RuntimeError('Practice WheelCollider parent is not a serialized Transform')
    parent_position = serialized_vector(parent_block, 'm_LocalPosition', 'xyz')
    parent_rotation = serialized_vector(parent_block, 'm_LocalRotation', 'xyzw')
    parent_scale = serialized_vector(parent_block, 'm_LocalScale', 'xyz')
    parent_fathers = re.findall(
        r'(?m)^\s*m_Father:\s*\{fileID:\s*(\d+)\}\s*$',parent_block)
    if len(parent_fathers)!=1 or parent_fathers[0]!=vehicle_transform_id:
        raise RuntimeError('WheelCollider geometry is not directly in vehicle-local space')
    if (any(abs(value) > 1e-9 for value in parent_position)
            or parent_rotation != (0.0, 0.0, 0.0, 1.0)
            or parent_scale != (1.0, 1.0, 1.0)):
        raise RuntimeError('WheelCollider parent transform is not identity; compose the hierarchy')

    scene_blocks = re.split(r'(?m)(?=^--- !u!\d+ &)', scene)
    instance_blocks = [block for block in scene_blocks
                       if re.search(r'm_SourcePrefab:\s*\{fileID:\s*100100000,\s*'
                                    r'guid:\s*' + re.escape(prefab_guid) + r'\b', block)]
    torque_values = []
    driving_mode_values = []
    hmi_type_values = []
    unhandled_controller_overrides = []
    unhandled_wheel_overrides=[]
    curve_override_index = {
        'm_ExtremumSlip': 0,
        'm_ExtremumValue': 1,
        'm_AsymptoteSlip': 2,
        'm_AsymptoteValue': 3,
        'm_Stiffness': 4,
    }
    scene_curve_overrides = {}
    for block in instance_blocks:
        overrides = re.finditer(
            r'target:\s*\{fileID:\s*(\d+),\s*guid:\s*'
            + re.escape(prefab_guid)
            + r',\s*\n\s*type:\s*\d+\}\s*\n\s*propertyPath:\s*([^\n]+)'
            + r'\s*\n\s*value:\s*([^\n]*)',
            block)
        for match in overrides:
            target_id, property_path, raw_value = match.groups()
            property_path = property_path.strip()
            raw_value = raw_value.strip()
            if target_id == component_file_id:
                if property_path not in ('HMIType', 'DrivingMode', 'MotorTorque'):
                    unhandled_controller_overrides.append(property_path)
                    continue
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise RuntimeError(
                        f'Practice scene controller override {property_path} '
                        'is not numeric') from exc
                if property_path == 'MotorTorque':
                    torque_values.append(value)
                elif property_path == 'DrivingMode':
                    driving_mode_values.append(int(value))
                else:
                    hmi_type_values.append(int(value))
                continue
            if target_id in wheel_component_index:
                curve_match = re.fullmatch(
                    r'm_(Forward|Sideways)Friction\.(m_[A-Za-z]+)',
                    property_path)
                if curve_match is not None and curve_match.group(2) in curve_override_index:
                    try:
                        value = float(raw_value)
                    except ValueError as exc:
                        raise RuntimeError(
                            f'Practice scene tire override {property_path} '
                            'is not numeric') from exc
                    wheel_index = wheel_component_index[target_id]
                    curve_name = ('forward_curve' if curve_match.group(1) == 'Forward'
                                  else 'sideways_curve')
                    curve = list(wheel_parameters[wheel_index][curve_name])
                    curve[curve_override_index[curve_match.group(2)]] = value
                    wheel_parameters[wheel_index][curve_name] = tuple(curve)
                    scene_curve_overrides[(wheel_index, curve_name,
                                           curve_match.group(2))] = value
                    continue
                unhandled_wheel_overrides.append((target_id, property_path))
            elif target_id in wheel_transform_ids:
                unhandled_wheel_overrides.append((target_id, property_path))
    if unhandled_wheel_overrides:
        raise RuntimeError(
            'Practice scene has an unhandled WheelCollider/transform override; '
            'apply it before building the vehicle model: '
            + repr(unhandled_wheel_overrides))
    if unhandled_controller_overrides:
        raise RuntimeError(
            'Practice scene has unhandled VehicleController overrides; apply '
            'them before building the vehicle model: '
            + repr(unhandled_controller_overrides))
    if (len(instance_blocks) != 1 or len(torque_values) != 1
            or len(driving_mode_values) != 1 or len(hmi_type_values) != 1):
        raise RuntimeError('Could not resolve exactly one practice-scene MotorTorque '
                           'and HMIType/DrivingMode override for the F1TENTH VehicleController')
    scene_torque = torque_values[0]
    if drive_type != 5 or brake_type != 0 or steer_type != 0:
        raise RuntimeError('Practice prefab drivetrain/steering changed; re-derive vehicle control semantics')
    if driving_mode_values[0] != 1:
        raise RuntimeError('Practice scene does not select autonomous VehicleController mode')
    if not math.isclose(throttle_limit, 1.0, abs_tol=1.e-9):
        raise RuntimeError('Practice prefab ThrottleLimit is not 1.0')
    return {
        'unity_scene': str(scene_path.relative_to(Path(repo))),
        'unity_prefab': str(prefab_path.relative_to(Path(repo))),
        'unity_controller_source': str(controller_path.relative_to(Path(repo))),
        'controller_component_file_id': int(component_file_id),
        'prefab_default_motor_torque_nm': prefab_torque,
        'scene_motor_torque_override_nm': scene_torque,
        'drive_type': 'CAWD',
        'brake_type': 'CAWB',
        'steer_type': 'FrontWheelSteer',
        'prefab_driving_mode': prefab_driving_mode,
        'scene_driving_mode': driving_mode_values[0],
        'scene_hmi_type': hmi_type_values[0],
        'throttle_limit': throttle_limit,
        'steering_limit_deg': steering_limit_deg,
        'steering_rate_degps': steering_rate_degps,
        'wheelbase_mm': wheelbase_mm,
        'track_width_mm': track_width_mm,
        # This VehicleController field is consumed only by its skid-steer
        # ExtendedDifferentialDrive branch. The active CAWD path uses the
        # physical WheelCollider radii parsed below.
        'controller_wheel_radius_m_inactive_for_cawd': controller_wheel_radius_m,
        'drive_torque_per_wheel_at_full_command_nm': scene_torque / 4.0,
        'brake_torque_per_wheel_at_zero_command_nm': scene_torque,
        'rigidbody_com_right_up_forward_m': prefab_com,
        'rigidbody_mass_kg': rigidbody_mass,
        'rigidbody_drag_per_s': rigidbody_drag,
        'rigidbody_angular_drag_per_s': rigidbody_angular_drag,
        'rigidbody_use_gravity': bool(rigidbody_use_gravity),
        'rigidbody_is_kinematic': bool(rigidbody_is_kinematic),
        'rigidbody_constraints': rigidbody_constraints,
        'wheel_geometry_vehicle_local': wheel_geometry,
        'wheel_physics_parameters': wheel_parameters,
        'scene_tire_curve_overrides': [
            {'wheel_index': wheel_index, 'curve': curve_name,
             'field': field, 'value': value}
            for (wheel_index, curve_name, field), value
            in sorted(scene_curve_overrides.items())],
    }


def _cpp_number(source: str, name: str) -> float:
    """Resolve simple constant C++ expressions and named constexpr/#define aliases.

    This deliberately does not evaluate C++ or arbitrary Python. If a symbol
    depends on unavailable headers or compiler -D definitions, it raises an
    explicit UnresolvedCppSymbol; the caller decides whether a separate
    authoritative source can supply that *specific* parameter.
    """
    without_comments = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)

    def definition(identifier: str):
        # Explicit scalar declarations, including constexpr/static const C++.
        # Avoid confusing a use of a symbol in an expression with a definition.
        declarations = re.findall(
            r'(?m)^\s*(?:(?:inline|static|extern|const|constexpr|volatile)\s+)*'
            r'(?:unsigned\s+)?(?:float|double|int|long\s+double|auto)\s+'
            + re.escape(identifier) + r'\s*=\s*([^;]+);',
            without_comments)
        # Also recognize a macro definition that is not function-like.
        macros = re.findall(r'(?m)^\s*#\s*define\s+'
                            + re.escape(identifier) + r'(?![\w(])\s+([^\n]+)',
                            without_comments)
        # Resolve unique consistent declarations only; no guess on conflicts.
        found = declarations + macros
        if not found:
            raise UnresolvedCppSymbol(f'Cannot resolve {identifier}: no source definition')
        if len(found) != 1:
            raise RuntimeError(f'Ambiguous C++ constant definition for {identifier}')
        return found[0].strip()

    def numeric_expr(expr: str, seen: frozenset[str]):
        expr = re.sub(
            r'(?<![\w.])((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)[fFlL]\b',
            r'\1', expr)
        expr = re.sub(r'\bstd::numbers::pi_v\s*<\s*(?:float|double)\s*>',
                      repr(math.pi), expr)
        expr = re.sub(r'\bstd::numbers::pi\b', repr(math.pi), expr)
        try:
            parsed = ast.parse(expr, mode='eval')
        except SyntaxError as exc:
            raise RuntimeError(f'Unsupported C++ expression {expr!r}') from exc

        def calc(node):
            if isinstance(node, ast.Expression):
                return calc(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (float, int):
                return float(node.value)
            if isinstance(node, ast.Name):
                identifier = node.id
                if identifier in ('kPi', 'M_PI'):
                    return math.pi
                if identifier in seen:
                    raise RuntimeError(f'Circular C++ constant alias involving {identifier}')
                return numeric_expr(definition(identifier), seen | {identifier})
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                val = calc(node.operand)
                return val if isinstance(node.op, ast.UAdd) else -val
            if isinstance(node, ast.BinOp) and isinstance(node.op,
                    (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                a,b=calc(node.left),calc(node.right)
                if isinstance(node.op,ast.Add):return a+b
                if isinstance(node.op,ast.Sub):return a-b
                if isinstance(node.op,ast.Mult):return a*b
                return a/b
            raise RuntimeError(f'Unsupported C++ constant syntax: {expr!r}')
        try:
            value = float(calc(parsed))
        except (ZeroDivisionError, OverflowError) as exc:
            raise RuntimeError(f'Invalid C++ constant expression {expr!r}') from exc
        if not math.isfinite(value):
            raise RuntimeError(f'Non-finite C++ constant {name}')
        return value

    return numeric_expr(definition(name), frozenset({name}))


def _literal_plant_actuator(source: str) -> tuple[float | None, float | None]:
    """Read the *actual* checked-in C++ steer update, not arbitrary defaults.

    Some branches supply kMaxSteeringRad as a build-system macro. Even when
    that macro is invisible to source parsing, the implemented Unity steering
    update uses independent degree-valued literals in its threshold, rate and
    target computations. Those are useful cross-checks and fallback evidence.
    Only accept literals matching the recognizable Unity-update equations.
    """
    clean = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    number = r'([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?(?:[fF])?)'
    def one(pattern):
        matches = re.findall(pattern, clean)
        if len(matches) > 1:
            raise RuntimeError('Ambiguous C++ actuator update declarations')
        return float(matches[0].rstrip('fF')) if matches else None
    threshold = one(r'\bthreshold\s*=\s*last_internal_deg\s*/\s*' + number + r'\s*;')
    target = one(r'\btarget_deg\s*=\s*-\s*command_norm\s*\*\s*' + number + r'\s*;')
    rate = one(r'\bincrement\s*=\s*' + number + r'\s*\*\s*kDt\s*;')
    if rate is None:
        # The production plant names the Unity prefab value and multiplies it
        # by the 1 ms step at the use site. Resolve that named constant only
        # when the same function uses it in the steering increment; this keeps
        # the parser tied to executable source rather than a remembered value.
        named_rate_use = re.search(
            r'\bincrement\s*=\s*kSteeringRateDegps\s*\*\s*kDt\s*;',
            clean)
        if named_rate_use:
            rate_degps = one(
                r'\bconstexpr\s+double\s+kSteeringRateDegps\s*=\s*'
                + number + r'\s*;')
            if rate_degps is None:
                raise RuntimeError(
                    'Plant uses kSteeringRateDegps without one numeric definition')
            rate = rate_degps
    if threshold is not None and target is not None and not math.isclose(threshold, target, abs_tol=1.e-6):
        raise RuntimeError('Source steering target and threshold use different limits')
    limit_deg = threshold if threshold is not None else target
    if limit_deg is not None and (not math.isfinite(limit_deg) or limit_deg <= 0):
        raise RuntimeError('Invalid C++ steering limit in actuator update')
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise RuntimeError('Invalid C++ steering rate in actuator update')
    return (math.radians(limit_deg) if limit_deg is not None else None,
            math.radians(rate) if rate is not None else None)


def _current_steering_parameters(repo: Path, source: str) -> tuple[float, float, dict]:
    """Resolve source actuator constants without assuming particular C++ names.

    Read the production plant and its owned headers. Reject unresolved aliases
    and conflicting implemented updates; never resurrect a retired runtime to
    supply missing constants. current_model cross-checks Unity's active prefab.
    """
    plant_path = Path(repo) / 'f1tenth_mpc/src/unity_source_plant.cpp'

    def includes(raw: str, path: Path, root: Path) -> str:
        text = raw
        for include in re.findall(r'(?m)^\s*#\s*include\s*"([^"\n]+)"', raw):
            part = Path(include)
            if part.is_absolute() or '..' in part.parts:
                continue
            for base in (path.parent, root):
                found = base / part
                if found.is_file():
                    text += '\n' + found.read_text(encoding='utf-8')
                    break
        return text

    plant_text = includes(source, plant_path, Path(repo) / 'f1tenth_mpc/include')
    plant_limit_update, plant_rate_update = _literal_plant_actuator(source)

    evidence = {
        'limit': ('kMaxSteeringRad', plant_limit_update, 1e-5),
        'rate': ('kSteeringRateRadps', plant_rate_update, 1e-4),
    }
    values: dict[str, float] = {}
    provenance = {'plant_update_limit_rad': plant_limit_update,
                  'plant_update_rate_radps': plant_rate_update}
    for label, (name, update_value, atol) in evidence.items():
        try:
            plant_value = _cpp_number(plant_text, name)
        except UnresolvedCppSymbol as exc:
            rhs = re.findall(r'\b' + re.escape(name) + r'\s*=\s*([^;]+);', source)
            if not rhs and update_value is not None:
                plant_value = None
            else:
                raise RuntimeError(f'Cannot resolve {name}: {exc}') from exc
        sources = [(kind, v) for kind, v in (
            ('unity_source_plant.cpp definition', plant_value),
            ('unity_source_plant.cpp actuator update', update_value))
            if v is not None]
        if not sources:
            raise RuntimeError(
                f'Cannot verify {name}: no numerical definition in source plant, '
                'or recognized steering-update equations')
        # Reject mismatch rather than silently selecting an arbitrary value.
        # The active plant's declared constants take precedence where known;
        # otherwise the plant's implemented update is the best numeric source.
        chosen_source, chosen_value = sources[0]
        for other_source, v in sources[1:]:
            if not math.isclose(chosen_value, v, rel_tol=0, abs_tol=atol):
                raise RuntimeError(
                    f'{name}: {chosen_source} ({chosen_value:.10g}) disagrees '
                    f'with {other_source} ({v:.10g}). Check the car config.')
        if not math.isfinite(chosen_value) or chosen_value <= 0:
            raise RuntimeError(f'{name} must be positive and finite')
        values[label] = chosen_value
        provenance[label + '_source'] = chosen_source
    return values['limit'], values['rate'], provenance


def _quat_rotation(q):
    x,y,z,w=q
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])



def _cpp_array(source: str, name: str, count: int):
    match=re.search(r'\b'+re.escape(name)+r'\s*\{([^}]+)\}',source)
    if not match: raise RuntimeError(f'Missing source-model constant array {name}')
    values=tuple(float(v.strip()) for v in match.group(1).split(',') if v.strip())
    if len(values)!=count or not all(math.isfinite(v) for v in values):
        raise RuntimeError(f'Malformed source-model constant array {name}')
    return values


def _cpp_numeric_array(source: str, name: str, count: int):
    """Resolve one-dimensional source arrays, including numeric macro entries."""
    match = re.search(r'\b' + re.escape(name) + r'\s*\{([^}]+)\}', source)
    if match is None:
        raise UnresolvedCppSymbol(f'Cannot resolve source array {name}')
    values = []
    for token in match.group(1).split(','):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(re.sub(r'[fFlL]$', '', token)))
        except ValueError:
            values.append(_cpp_number(source, token))
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f'Malformed source array {name}')
    return tuple(values)


def bind_current_wheel_moment_arms(repo: Path, *modules):
    """Use the current source wheel contact/moment arms in historical CasADi.

    The recovered implementation originally used rounded +/-0.118m legacy
    wheel offsets. Current source uses distinct FL/FR/RL/RR geometry.
    """
    source=(Path(repo)/'f1tenth_mpc/src/unity_source_plant.cpp').read_text()
    wx=_cpp_array(source,'kWheelX',4)
    wy=_cpp_array(source,'kWheelY',4)
    transform_z=_cpp_array(source,'kWheelColliderTransformZCom',4)
    wheel_radius=_cpp_number(source,'kWheelRadiusM')
    wz=tuple(value-wheel_radius for value in transform_z)
    for module in modules:
        module.WHEEL_X_M=wx
        module.WHEEL_Y_M=wy
        module.WHEEL_FORCE_APPLICATION_Z_M=wz
    return {'x_m':wx,'y_m':wy,'force_application_z_m':wz}

def convert_unity_suspension_mounts_to_model_frame(unity_mounts):
    """Unity x-right/y-up/z-forward -> model left/up/forward mount signs.

    The restored suspension travel equation takes roll in the model forward /
    left / up coordinate frame, but reads the first entry of each mount as
    the roll-moment lateral arm. Consequently Unity's x-right mount arm must
    be sign-reflected BEFORE using it with that model-frame roll angle.
    The front/rear arms, mount heights, wheel order, and actual tire slip
    coordinate conventions are not changed. No fitted forces are introduced.
    """
    mounts=tuple(tuple(float(z) for z in p) for p in unity_mounts)
    if len(mounts)!=4 or any(len(p)!=3 for p in mounts):
        raise ValueError('Expected four Unity suspension mounts in FL FR RL RR order')
    if not (mounts[0][0]<0<mounts[1][0] and mounts[2][0]<0<mounts[3][0]):
        raise ValueError('Unity suspension mounts are not x-right, FL FR RL RR')
    return tuple((-x,y,z) for x,y,z in mounts)


def current_model(repo: Path, unity_wheel_module):
    repo=Path(repo).resolve()
    source_path=repo/'f1tenth_mpc/src/unity_source_plant.cpp'
    source_header_path=repo/'f1tenth_mpc/include/unity_source_plant_c.h'
    config_path=repo/'f1tenth_localization/config/physx_odom_reset_initial.json'
    raw=source_path.read_text()
    source_with_header=raw+'\n'+source_header_path.read_text()
    initial=json.loads(config_path.read_text())['initial']
    wheels=initial['wheel_inputs']; body=initial['body']
    steering_limit, steering_rate, steering_provenance = _current_steering_parameters(repo, raw)
    # Historical source geometry remains appropriate only if the current
    # source and native reset still agree on the critical dimensions.
    motor=_cpp_number(raw,'kMotorTorqueNm')
    scene_config=_unity_scene_vehicle_parameters(repo)
    if not math.isclose(motor, scene_config['scene_motor_torque_override_nm'],
                        rel_tol=0.0, abs_tol=1.e-6):
        raise RuntimeError('C++ plant MotorTorque differs from the active practice '
                           'Unity scene override')
    if scene_config['scene_driving_mode'] != 1:
        raise RuntimeError('Source model requires the practice autonomous drive mode')
    if scene_config['rigidbody_is_kinematic'] or not scene_config['rigidbody_use_gravity']:
        raise RuntimeError('Practice Rigidbody must be dynamic and gravity-enabled')
    if scene_config['rigidbody_constraints'] != 0:
        raise RuntimeError('Practice Rigidbody constraints are not represented in the plant')
    radius=_cpp_number(raw,'kWheelRadiusM')
    wheelbase=_cpp_number(raw,'kWheelbaseM')
    track=_cpp_number(raw,'kTrackM')
    if not math.isclose(scene_config['wheelbase_mm'] / 1000.0, wheelbase,
                        rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError('Production wheelbase differs from active VehicleController Ackermann geometry')
    if not math.isclose(scene_config['track_width_mm'] / 1000.0, track,
                        rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError('Production track width differs from active VehicleController Ackermann geometry')
    if not math.isclose(math.radians(scene_config['steering_limit_deg']),
                        steering_limit, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError('Production steering limit differs from active VehicleController')
    if not math.isclose(math.radians(scene_config['steering_rate_degps']),
                        steering_rate, rel_tol=0.0, abs_tol=2e-7):
        raise RuntimeError('Production steering rate differs from active VehicleController')
    if not math.isclose(scene_config['rigidbody_drag_per_s'],
                        _cpp_number(raw, 'kDragPerS'),
                        rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError('Production linear drag differs from active practice Rigidbody')
    if not math.isclose(scene_config['rigidbody_angular_drag_per_s'],
                        _cpp_number(raw, 'kAngularDampingPerS'),
                        rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError('Production angular damping differs from active practice Rigidbody')
    wheel_mass=_cpp_number(raw,'kWheelMassKg')
    wheel_inertia=0.5*wheel_mass*radius*radius
    wheel_damping=_cpp_number(raw,'kWheelDampingRate')
    suspension_distance=_cpp_number(raw,'kSuspensionDistanceM')
    suspension_target=_cpp_number(raw,'kSuspensionTargetPosition')
    suspension_spring=_cpp_number(raw,'kSuspensionSpringRateNPerM')
    suspension_damper=_cpp_number(raw,'kSuspensionDamperRateNsPerM')
    max_compression=_cpp_number(raw,'kSuspensionMaxCompressionM')
    max_droop=_cpp_number(raw,'kSuspensionMaxDroopM')
    target_travel=suspension_distance*suspension_target
    if not (0.0 < suspension_target < 1.0 and
            math.isclose(max_compression,target_travel,
                         rel_tol=0.0,abs_tol=1e-10) and
            math.isclose(max_compression,
                         suspension_distance*(1.0-suspension_target),
                         rel_tol=0.0,abs_tol=1e-10) and
            math.isclose(max_droop,suspension_distance*suspension_target,
                         rel_tol=0.0,abs_tol=1e-10)):
        raise RuntimeError('Production suspension travel constants disagree')
    if (abs(motor-428)>1e-6 or abs(radius-.059)>1e-7
            or abs(wheelbase-.324)>1e-7 or abs(track-.236)>1e-7):
        raise RuntimeError('Current source anchors changed: review model compatibility before optimization')
    source_wheel_values={
        'radius_m':radius,
        'mass_kg':wheel_mass,
        'moment_of_inertia_kgm2':wheel_inertia,
        'wheel_damping_rate_nms':wheel_damping,
        'max_compression_m':max_compression,
        'max_droop_m':max_droop,
        'spring_rate_npm':suspension_spring,
        'damper_rate_nspm':suspension_damper,
    }
    for index,wheel in enumerate(wheels):
        for name,expected in source_wheel_values.items():
            if not math.isclose(float(wheel[name]),expected,
                                rel_tol=0.0,abs_tol=1e-8):
                raise RuntimeError(
                    f'PhysX reset wheel {index} {name} disagrees with '
                    'the production source-derived value')
    prefab_wheels=scene_config['wheel_physics_parameters']
    if len(prefab_wheels)!=4:
        raise RuntimeError('Practice prefab must expose four wheel-physics parameter sets')
    source_forward_curve=_cpp_numeric_array(
        source_with_header,'kForwardCurve',5)
    source_side_curve=_cpp_numeric_array(source_with_header,'kSideCurve',5)
    source_prefab_values={
        'radius_m':radius,
        'mass_kg':wheel_mass,
        'damping_rate_nms':wheel_damping,
        'suspension_distance_m':suspension_distance,
        'suspension_spring_npm':suspension_spring,
        'suspension_damper_ns_per_m':suspension_damper,
        'suspension_target_position':suspension_target,
    }
    for index,(reset_wheel,prefab_wheel) in enumerate(zip(wheels,prefab_wheels)):
        for field,expected in source_prefab_values.items():
            if not math.isclose(float(prefab_wheel[field]),expected,
                                rel_tol=0.0,abs_tol=2e-7):
                raise RuntimeError(
                    f'Production plant wheel {index} {field} differs from active prefab')
        if not math.isclose(prefab_wheel['force_app_point_distance_m'],0.0,
                            rel_tol=0.0,abs_tol=1e-12):
            raise RuntimeError(
                'IPOPT force-point geometry assumes zero WheelCollider forceAppPointDistance')
        for field,expected in (('forward_curve',source_forward_curve),
                               ('sideways_curve',source_side_curve)):
            if any(not math.isclose(float(actual),source_value,
                                    rel_tol=0.0,abs_tol=2e-7)
                   for actual,source_value in zip(prefab_wheel[field],expected)):
                raise RuntimeError(
                    f'Production tire curve {field} differs from active practice prefab')
        if any(not math.isclose(float(reset_wheel[field]),
                                float(prefab_wheel[prefab_field]),
                                rel_tol=0.0,abs_tol=2e-7)
               for field,prefab_field in (
                   ('radius_m','radius_m'),('mass_kg','mass_kg'),
                   ('wheel_damping_rate_nms','damping_rate_nms'),
                   ('spring_rate_npm','suspension_spring_npm'),
                   ('damper_rate_nspm','suspension_damper_ns_per_m'))):
            raise RuntimeError(
                f'Reset wheel {index} physics differs from active practice prefab')
        for reset_field,prefab_field in (
                ('forward_curve','forward_curve'),
                ('sideways_curve','sideways_curve')):
            if any(not math.isclose(float(actual),float(expected),
                                    rel_tol=0.0,abs_tol=2e-7)
                   for actual,expected in zip(reset_wheel[reset_field],
                                              prefab_wheel[prefab_field])):
                raise RuntimeError(
                    f'Reset wheel {index} {reset_field} differs from active prefab')
    mass=float(body['mass_kg'])
    if not math.isclose(mass, _cpp_number(raw,'kMassKg'),
                        rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError('Reset rigidbody mass disagrees with production plant')
    if not math.isclose(scene_config['rigidbody_mass_kg'],
                        _cpp_number(raw,'kMassKg'),
                        rel_tol=0.0,abs_tol=2e-7):
        raise RuntimeError('Production plant mass disagrees with practice prefab')
    if not math.isclose(_cpp_number(raw,'kWheelInertiaKgM2'),wheel_inertia,
                        rel_tol=0.0,abs_tol=1e-14):
        raise RuntimeError('Production wheel inertia is not 0.5*m*r^2')
    for index,wheel in enumerate(wheels):
        for field,expected in (('forward_curve',source_forward_curve),
                               ('sideways_curve',source_side_curve)):
            measured=tuple(float(value) for value in wheel[field])
            if (len(measured)!=len(expected) or any(
                    not math.isclose(actual,source_value,
                                     rel_tol=0.0,abs_tol=1e-8)
                    for actual,source_value in zip(measured,expected))):
                raise RuntimeError(
                    f'Reset wheel {index} {field} disagrees with production source')
    gravity=_cpp_number(raw,'kGravity')
    if not math.isclose(float(unity_wheel_module.UNITY_PHYSICS_GRAVITY_MPS2),
                        gravity,rel_tol=0.0,abs_tol=1e-12):
        raise RuntimeError('IPOPT gravity constant disagrees with production plant')
    source_filter=tuple(_cpp_number(raw,name) for name in (
        'kTireLoadFilterMin','kTireLoadFilterMinValue',
        'kTireLoadFilterMax','kTireLoadFilterMaxValue'))
    if tuple(unity_wheel_module.PHYSX_TIRE_LOAD_FILTER)!=source_filter:
        raise RuntimeError('IPOPT tire-load filter disagrees with production plant')
    tolerance_scale=_cpp_number(raw,'kPhysXToleranceScaleLengthM')
    longitudinal_slip_offset=_cpp_number(raw,'kLongitudinalSlipSpeedOffsetM')
    lateral_slip_offset=_cpp_number(raw,'kMinLateralSpeedForTireModelMps')
    for field,expected in (
            ('PHYSX_TOLERANCE_SCALE_LENGTH_M',tolerance_scale),
            ('PHYSX_APPLIED_TORQUE_SLIP_OFFSET_M',longitudinal_slip_offset),
            ('PHYSX_LATERAL_SLIP_SPEED_OFFSET_MPS',lateral_slip_offset)):
        if not math.isclose(float(getattr(unity_wheel_module,field)),expected,
                            rel_tol=0.0,abs_tol=1e-12):
            raise RuntimeError(f'IPOPT {field} disagrees with production plant')
    # Construct suspension ray geometry from the same resolved source arrays
    # used by the production C++ contact query. Do not rebuild it from the old
    # Python prefab quaternion constants: those constants encoded sin/cos(5°)
    # as quaternion components (a 10° rotation), while the active prefab's
    # local transform and production C++ use a 5° rotation.
    source_com_forward,source_com_left,source_com_up=_cpp_array(
        raw,'kComOffsetFromActorBody',3)
    prefab_com=tuple(float(value) for value in
                     scene_config['rigidbody_com_right_up_forward_m'])
    prefab_com_in_source=(prefab_com[2],-prefab_com[0],prefab_com[1])
    if any(not math.isclose(actual,expected,rel_tol=0.0,abs_tol=2e-7)
           for actual,expected in zip(prefab_com_in_source,
                                      (source_com_forward,source_com_left,
                                       source_com_up))):
        raise RuntimeError('Production plant COM disagrees with active practice prefab')
    com_in_wheel_model=(-source_com_left,source_com_up,source_com_forward)
    reset_com=tuple(float(value) for value in body['center_of_mass_body_m'])
    reset_com_in_source=(reset_com[2],-reset_com[0],reset_com[1])
    if any(not math.isclose(actual,expected,rel_tol=0.0,abs_tol=1e-8)
           for actual,expected in zip(reset_com_in_source,
                                      (source_com_forward,source_com_left,
                                       source_com_up))):
        raise RuntimeError('Reset Rigidbody COM disagrees with production plant')
    unity_wheel_module.RIGIDBODY_CENTER_OF_MASS_BODY_M=com_in_wheel_model

    wheel_x=_cpp_array(raw,'kWheelX',4)
    wheel_y=_cpp_array(raw,'kWheelY',4)
    transform_z=_cpp_array(raw,'kWheelColliderTransformZCom',4)
    transform_y=_cpp_array(raw,'kWheelColliderTransformYCom',4)
    center_offset=_cpp_number(raw,'kWheelColliderCenterOffsetM')
    front_camber_sin=_cpp_number(raw,'kFrontCamberSin')
    front_camber_cos=_cpp_number(raw,'kFrontCamberCos')
    center_target_offset=_cpp_number(raw,'kWheelCentreTargetOffsetM')
    unity_wheel_module.WHEEL_X_M=wheel_x
    unity_wheel_module.WHEEL_Y_M=wheel_y
    unity_wheel_module.WHEEL_FORCE_APPLICATION_Z_M=tuple(
        value-radius for value in transform_z)
    base=[]
    prefab_wheels=scene_config['wheel_geometry_vehicle_local']
    if len(prefab_wheels)!=4:
        raise RuntimeError('Practice prefab must expose all four WheelColliders')
    for index,wheel in enumerate(wheels):
        _validate_reset_wheel_geometry(
            index, wheel, wheel_x[index], wheel_y[index],
            transform_z[index], radius)
        prefab_wheel=prefab_wheels[index]
        position=np.asarray(
            prefab_wheel['transform_position_right_up_forward_m'],dtype=float)
        rotation=_quat_rotation(prefab_wheel['transform_rotation_xyzw'])
        center=np.asarray(prefab_wheel['center_right_up_forward_m'],dtype=float)
        wheel_center=position+rotation@center
        com_right=-source_com_left
        prefab_wheel_center_source=(
            wheel_center[2]-source_com_forward,
            -(wheel_center[0]-com_right))
        prefab_transform_source=(
            -(position[0]-com_right),position[1]-source_com_up)
        production_wheel_center=(wheel_x[index],wheel_y[index])
        production_transform=(transform_y[index],transform_z[index])
        if any(not math.isclose(actual,expected,rel_tol=0.0,abs_tol=3e-7)
               for actual,expected in zip(prefab_wheel_center_source,
                                          production_wheel_center)):
            raise RuntimeError(
                f'Production plant wheel {index} centre disagrees with active prefab')
        if any(not math.isclose(actual,expected,rel_tol=0.0,abs_tol=3e-7)
               for actual,expected in zip(prefab_transform_source,
                                          production_transform)):
            raise RuntimeError(
                f'Production plant wheel {index} transform disagrees with active prefab')
        center_offset_up=center_offset*(front_camber_cos if index<2 else 1.0)
        if not math.isclose(
                wheel_center[1]-position[1],center_offset_up,
                rel_tol=0.0,abs_tol=3e-7):
            raise RuntimeError(
                f'Production plant wheel {index} center rotation disagrees with prefab quaternion')
        ray_start_up=(transform_z[index]+center_offset_up
                      -center_target_offset+radius+max_compression)
        # The CasADi contact helper expresses the collider mount relative to
        # actor origin, then subtracts COM and adds its ray-radius/travel terms.
        base.append((wheel_y[index]+source_com_left,
                     ray_start_up+target_travel-radius
                     -max_compression+source_com_up,
                     wheel_x[index]+source_com_forward))
    camber_angle=math.atan2(front_camber_sin,front_camber_cos)
    camber=(-camber_angle,camber_angle,0.0,0.0)

    # The current plant stores the diagonal inverse inertia used by the
    # vehicle-force update (not the separate rotated inertia used by the
    # scene constraint solver). The historical OCP expects principal moments,
    # so invert the source constants instead of depending on the retired
    # kVehicleMassSpaceInertiaKgM2 declaration.
    inverse_inertia = _cpp_array(raw, 'kInverseInertiaVehicleBodyKgM2', 3)
    if any(value <= 0.0 for value in inverse_inertia):
        raise RuntimeError('Current source inverse inertia must be positive')
    inertia = [1.0 / value for value in inverse_inertia]
    g=gravity
    sprung=_cpp_array(raw,'kSprungMassKg',4)
    cls=unity_wheel_module.UnityWheelModel
    model=cls(
        mass_kg=mass, yaw_inertia_kgm2=inertia[2],
        drag_per_s=_cpp_number(raw,'kDragPerS'),
        wheel_radius_m=radius, wheel_radius_scale=1.,
        wheel_inertia_kgm2=wheel_inertia,
        wheel_damping_nms=wheel_damping,
        steering_limit_rad=steering_limit,
        steering_rate_radps=steering_rate,
        wheelbase_m=wheelbase, track_m=track,
        motor_torque_nm=motor,
        body_com_height_m=float(body['position_world_m'][1])+source_com_up,
        # Reset JSON sprung masses are rounded initial-state values. The
        # optimizer must use the exact configured PxVehicleComputeSprungMasses
        # result pinned by the production model, not those rounded snapshot
        # values.
        sprung_mass_kg=sprung,
        forward_curve=tuple(tuple(float(z) for z in w['forward_curve']) for w in wheels),
        lateral_curve=tuple(tuple(float(z) for z in w['sideways_curve']) for w in wheels),
        static_normal_load_n=tuple(g*value for value in sprung),
        suspension_target_travel_m=(max_compression,)*4,
        suspension_spring_n_per_m=(suspension_spring,)*4,
        suspension_damper_ns_per_m=(suspension_damper,)*4,
        suspension_distance_m=suspension_distance,
        suspension_base_xyz_body_m=tuple(base),
        suspension_camber_rad=tuple(camber),
        suspension_ground_y_m=0.,
        suspension_reference_body_height_m=0.00464804098,
        inertia_pitch_kgm2=inertia[1], inertia_roll_kgm2=inertia[0],
    )
    # Avoid reporting a deleted measured capture as the model's provenance.
    original_meta=cls.metadata
    def metadata(self):
        result=original_meta(self)
        result['source_capture']=None
        result['source_origin']=('f1tenth_mpc/src/unity_source_plant.cpp + '
                                'f1tenth_mpc/include/unity_source_plant_c.h + '
                                'f1tenth_localization/config/physx_odom_reset_initial.json')
        result['inertia_source']='reciprocal of kInverseInertiaVehicleBodyKgM2'
        result['suspension_mount_lateral_frame']='production ray/contact geometry in forward-left-up basis, derived from kWheelX/kWheelY/kWheelColliderTransformZCom/kComOffsetFromActorBody'
        result['suspension_coordinate_correction']='production wheel ray/contact geometry replaces legacy Python transform/quaternion reconstruction; zero fitted parameters'
        result['tire_callback_parity']=(
            'UnityPlayer callback curve arithmetic; raw suspensionForceMag '
            'is normalized, passed through the configured PhysX tire-load '
            'filter, then scaled by rest load before callback')
        result['tire_load_filter_applied_by_ocp']=True
        result['sprung_mass_source']='production kSprungMassKg from PxVehicleComputeSprungMasses'
        result['reset_json_sprung_mass_kg']=[float(w['sprung_mass_kg']) for w in wheels]
        result['production_sprung_mass_kg']=list(sprung)
        result['production_wheel_constants']={
            **source_wheel_values,
            'inertia_derivation':'0.5 * wheel_mass * wheel_radius^2',
            'suspension_distance_m':suspension_distance,
            'suspension_target_position':suspension_target,
        }
        result['native_callback_force_rms_evidence_n']=1.344035278849403e-7
        result['model_boundary']='Historical CasADi differential/algebraic tire+wheel+spring model; not native PhysX scene solver, collision impulses, wheel-contact recursion or exact 1 ms map. Mandatory native replay follows.'
        result['unity_scene_vehicle_config']=scene_config
        return result
    cls.metadata=metadata
    interface_path=repo/'f1tenth_mpc/include/unity_source_plant_c.h'
    fingerprint=hashlib.sha256(source_path.read_bytes()
                               +interface_path.read_bytes()
                               +config_path.read_bytes()
                               +(Path(repo)/scene_config['unity_scene']).read_bytes()
                               +(Path(repo)/scene_config['unity_prefab']).read_bytes()
                               +(Path(repo)/scene_config['unity_controller_source']).read_bytes()).hexdigest()
    return model,{'source_sha256':fingerprint,'model_metadata':model.metadata(),
                  'steering_provenance':steering_provenance}
