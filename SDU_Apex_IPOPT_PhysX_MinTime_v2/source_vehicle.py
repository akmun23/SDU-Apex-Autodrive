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

    The Unity-source plant's explicit steering update and the native runtime's
    declarations are independent evidence. Reject disagreements, and clearly
    label source parity as unverified if one side is missing a definition.
    """
    native_path = Path(repo) / 'f1tenth_localization/src/physx_odom_runtime.cpp'
    native = native_path.read_text(encoding='utf-8')
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
    native_text = includes(native, native_path, Path(repo) / 'f1tenth_localization/include')
    plant_limit_update, plant_rate_update = _literal_plant_actuator(source)

    def optional_cpp(raw: str, name: str) -> float | None:
        try:
            return _cpp_number(raw, name)
        except UnresolvedCppSymbol:
            return None

    native_limit_deg = optional_cpp(native_text, 'kSteeringLimitDeg')
    native_rate_deg = optional_cpp(native_text, 'kSteeringRateDegPerS')
    native_limit = math.radians(native_limit_deg) if native_limit_deg is not None else None
    native_rate = math.radians(native_rate_deg) if native_rate_deg is not None else None

    evidence = {
        'limit': ('kMaxSteeringRad', 'UNITY_SOURCE_MAX_STEERING_RAD',
                  native_limit, plant_limit_update, 1e-5),
        'rate': ('kSteeringRateRadps', 'UNITY_SOURCE_STEERING_RATE_RADPS',
                 native_rate, plant_rate_update, 1e-4),
    }
    values: dict[str, float] = {}
    provenance = {'native_actuator_file': str(native_path.relative_to(repo)),
                  'native_limit_rad': native_limit,
                  'native_rate_radps': native_rate,
                  'plant_update_limit_rad': plant_limit_update,
                  'plant_update_rate_radps': plant_rate_update}
    for label, (name, known_alias, native_value, update_value, atol) in evidence.items():
        try:
            plant_value = _cpp_number(plant_text, name)
            declared = True
        except UnresolvedCppSymbol as exc:
            rhs = re.findall(r'\b' + re.escape(name) + r'\s*=\s*([^;]+);', source)
            # Exactly the known symbolic alias may come from a build-system -D
            # definition. Unknown symbols must still be rejected.
            if len(rhs) == 1 and rhs[0].strip() == known_alias:
                declared = False
                plant_value = None
                provenance[label + '_alias_unverified'] = known_alias
            elif not rhs:
                declared = False
                plant_value = None
            else:
                raise RuntimeError(f'Cannot resolve {name}: {exc}') from exc
        sources = [(kind, v) for kind, v in (
            ('unity_source_plant.cpp definition', plant_value),
            ('unity_source_plant.cpp actuator update', update_value),
            ('physx_odom_runtime.cpp actuator declaration', native_value))
            if v is not None]
        if not sources:
            raise RuntimeError(
                f'Cannot verify {name}: no numerical definition in source plant, '
                'native runtime, or recognized steering-update equations')
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
        if native_value is None:
            provenance[label + '_native_unverified'] = True
            import warnings
            warnings.warn(
                f'{name}: native runtime does not expose a resolvable '
                f"{'kSteeringLimitDeg' if label == 'limit' else 'kSteeringRateDegPerS'}; "
                'verified only against the C++ plant/update source.',
                RuntimeWarning, stacklevel=2)
        if label + '_alias_unverified' in provenance:
            import warnings
            warnings.warn(
                f'{name} uses unresolved C++ alias {known_alias}; using '
                f'{chosen_source}, but cannot verify plant alias parity or compiler -D values.',
                RuntimeWarning, stacklevel=2)
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


def bind_current_wheel_moment_arms(repo: Path, *modules):
    """Use the current source wheel contact/moment arms in historical CasADi.

    The recovered implementation originally used rounded +/-0.118m legacy
    wheel offsets. Current source uses distinct FL/FR/RL/RR geometry.
    """
    source=(Path(repo)/'f1tenth_mpc/src/unity_source_plant.cpp').read_text()
    wx=_cpp_array(source,'kWheelX',4)
    wy=_cpp_array(source,'kWheelY',4)
    for module in modules:
        module.WHEEL_X_M=wx
        module.WHEEL_Y_M=wy
    return {'x_m':wx,'y_m':wy}

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
    config_path=repo/'f1tenth_localization/config/physx_odom_reset_initial.json'
    raw=source_path.read_text()
    initial=json.loads(config_path.read_text())['initial']
    wheels=initial['wheel_inputs']; body=initial['body']
    steering_limit, steering_rate, steering_provenance = _current_steering_parameters(repo, raw)
    from tools.vehicle_dynamics_learning import unity_vehicle_mechanics as legacy
    # Historical source geometry remains appropriate only if the current
    # source and native reset still agree on the critical dimensions.
    motor=_cpp_number(raw,'kMotorTorqueNm')
    radius=_cpp_number(raw,'kWheelRadiusM')
    wheelbase=_cpp_number(raw,'kWheelbaseM')
    track=_cpp_number(raw,'kTrackM')
    if (abs(motor-428)>1e-6 or abs(radius-.059)>1e-7
            or abs(wheelbase-.324)>1e-7 or abs(track-.236)>1e-7):
        raise RuntimeError('Current source anchors changed: review model compatibility before optimization')
    if any(abs(float(w['radius_m'])-radius)>1e-7 for w in wheels):
        raise RuntimeError('PhysX reset wheel radius mismatches current source')
    if any(tuple(round(float(v),6) for v in w['forward_curve']) != (.15,.9,.25,.58,.8) for w in wheels):
        raise RuntimeError('Current tire curve differs from verified IPOPT symbolic force law')
    # Historical suspension base positions derive from source-captured prefab
    # WheelCollider transforms, including the +/- 5 degree front-wheel toe.
    base=[]; camber=[]
    for pose,q in zip(legacy.WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M,
                      legacy.WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW):
        r=_quat_rotation(q)
        base.append(tuple(np.asarray(pose) + r @ np.array([0.,.025,0.])))
        down=r@np.array([0.,-1.,0.])
        camber.append(float(np.arctan2(down[0],-down[1])))
    # The native yaw/roll/body states use the model forward/LEFT/up basis,
    # but the Unity WheelCollider prefab supplies x-RIGHT/y-UP/z-FORWARD
    # mount coordinates. The legacy suspension ray-height expression
    # ``sin(roll) * mount_x`` is written in the model basis. Passing Unity's
    # x directly reversed all four per-wheel lateral load-transfer signs.
    # Reflect ONLY the lateral mount coordinate; wheel ordering and the
    # actual native tyre/wheel longitudinal coordinates remain unchanged.
    # Unlike a fitted load correction, this is a source coordinate mapping.
    base=convert_unity_suspension_mounts_to_model_frame(base)

    # Inertia is the current source-derived mass-space inertia used by the
    # MPC plant. This value changed after the historical optimizer was built.
    inertia_match=re.search(r'kVehicleMassSpaceInertiaKgM2\s*\{([^}]+)\}',raw)
    if not inertia_match: raise RuntimeError('Missing current PhysX inertia anchor')
    inertia=[float(z.strip()) for z in inertia_match.group(1).split(',')]
    mass=float(body['mass_kg'])
    g=9.81
    cls=unity_wheel_module.UnityWheelModel
    model=cls(
        mass_kg=mass, yaw_inertia_kgm2=inertia[2],
        drag_per_s=_cpp_number(raw,'kDragPerS'),
        wheel_radius_m=radius, wheel_radius_scale=1.,
        wheel_inertia_kgm2=float(wheels[0]['moment_of_inertia_kgm2']),
        wheel_damping_nms=float(wheels[0]['wheel_damping_rate_nms']),
        steering_limit_rad=steering_limit,
        steering_rate_radps=steering_rate,
        wheelbase_m=wheelbase, track_m=track,
        motor_torque_nm=motor,
        body_com_height_m=float(body['position_world_m'][1])+float(body['center_of_mass_body_m'][1]),
        sprung_mass_kg=tuple(float(w['sprung_mass_kg']) for w in wheels),
        forward_curve=tuple(tuple(float(z) for z in w['forward_curve']) for w in wheels),
        lateral_curve=tuple(tuple(float(z) for z in w['sideways_curve']) for w in wheels),
        static_normal_load_n=tuple(g*float(w['sprung_mass_kg']) for w in wheels),
        suspension_target_travel_m=tuple(float(w['max_compression_m']) for w in wheels),
        suspension_spring_n_per_m=tuple(float(w['spring_rate_npm']) for w in wheels),
        suspension_damper_ns_per_m=tuple(float(w['damper_rate_nspm']) for w in wheels),
        suspension_distance_m=float(wheels[0]['max_compression_m'])+float(wheels[0]['max_droop_m']),
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
        result['source_origin']='f1tenth_mpc/src/unity_source_plant.cpp + f1tenth_localization/config/physx_odom_reset_initial.json'
        result['suspension_mount_lateral_frame']='converted Unity x-right to model y-left for roll/travel (v2.3.10)'
        result['suspension_coordinate_correction']='sign of wheel-mount lateral coordinate only; zero fitted parameters'
        result['tire_callback_parity']='v2.3.12: native tireLoad=measured suspension force; un-stiffened curve normalized before final stiffness'
        result['tire_load_filter_applied_by_ocp']=False
        result['native_callback_force_rms_evidence_n']=1.344035278849403e-7
        result['model_boundary']='Historical CasADi differential/algebraic tire+wheel+spring model; not native PhysX scene solver, collision impulses, wheel-contact recursion or exact 1 ms map. Mandatory native replay follows.'
        return result
    cls.metadata=metadata
    native_path=repo/'f1tenth_localization/src/physx_odom_runtime.cpp'
    fingerprint=hashlib.sha256(source_path.read_bytes()+config_path.read_bytes()
                               +native_path.read_bytes()).hexdigest()
    return model,{'source_sha256':fingerprint,'model_metadata':model.metadata(),
                  'steering_provenance':steering_provenance}
