"""Exercise the bridge's pure telemetry boundary without ROS or AutoDRIVE."""
import ast
from pathlib import Path
import typing
import unittest

from tools.verify_runtime_topic_policy import subscription_errors


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / 'sdu_apex_autodrive/sdu_apex_autodrive/bridge_40hz.py'


def diagnostic_namespace(enabled):
    tree = ast.parse(BRIDGE.read_text())
    names = {'_development_packet_details', '_parse_optional_int',
             '_record_packet_arrival'}
    selected = ast.Module(body=[node for node in tree.body
                               if isinstance(node, ast.FunctionDef)
                               and node.name in names], type_ignores=[])
    namespace = {'Any': typing.Any,
                 '_development_packet_diagnostics_enabled': enabled}
    exec(compile(selected, str(BRIDGE), 'exec'), namespace)
    return namespace


class BridgeRuntimeContractTests(unittest.TestCase):
    def test_live_endpoint_audit_rejects_debug_and_transform_aliases(self):
        allowed = {'sensor_odometry': ['/autodrive/roboracer_1/imu', '/parameter_events'],
                   'mpc_controller_node': ['/odom', '/odom/unity_plant_state']}
        self.assertEqual(subscription_errors(allowed), [])
        for forbidden in ('/autodrive/roboracer_1/odom', '/autodrive/reset_command',
                          '/autodrive/roboracer_1/bridge_packet_timing',
                          '/autodrive/roboracer_1/development_actuator_timing',
                          '/tf', '/sdu/tf'):
            with self.subTest(topic=forbidden):
                self.assertTrue(subscription_errors({'sensor_odometry': [forbidden]}))

    def test_disabled_diagnostics_never_inspect_packet(self):
        class ForbiddenPacket(dict):
            def get(self, *_args):
                raise AssertionError('competition path inspected diagnostic telemetry')
        namespace = diagnostic_namespace(False)
        self.assertEqual(namespace['_development_packet_details'](ForbiddenPacket()), {})
        # No ROS, clock, locks or publisher exist here: all must be bypassed.
        namespace['_record_packet_arrival']()

    def test_development_only_parser_preserves_recording_fields(self):
        parse = diagnostic_namespace(True)['_development_packet_details']
        result = parse({'V1 Position': '1 2 3', 'V1 Linear Velocity': '4 5 6',
                        'V1 Throttle': '.2', 'V1 Collisions': '0'})
        self.assertEqual(result['simulator_position_x'], 1.)
        self.assertEqual(result['simulator_linear_velocity_y'], 5.)
        self.assertEqual(result['simulator_feedback_throttle_norm'], .2)
        self.assertEqual(result['simulator_collision_count'], 0)
        self.assertIsNone(result['simulator_development_command_sequence'])

    def test_bridge_publishes_exact_request_command_before_socket_send(self):
        source = BRIDGE.read_text()
        self.assertIn('BRIDGE_REQUEST_COMMAND_TOPIC = "/odom/bridge_request_command"', source)
        self.assertIn('message.header.stamp = _copy_stamp(stamp)', source)
        self.assertIn('message.steering_angle_rad = steering_normalized * (math.pi / 6.0)', source)
        self.assertIn('message.throttle_normalized = throttle_normalized', source)
        sender = source[source.index('def _run_command_sender('):]
        self.assertLess(sender.index('_publish_exact_request_command(request_stamp, command)'),
                        sender.index('original_emit("Bridge", data=command)'))

    def test_exact_bridge_command_publisher_uses_payload_not_latest_value(self):
        import math
        from types import SimpleNamespace

        tree = ast.parse(BRIDGE.read_text())
        selected = ast.Module(body=[
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == '_publish_exact_request_command'
        ], type_ignores=[])

        class FakeDriveCommand:
            def __init__(self):
                self.header = SimpleNamespace(stamp=None)
                self.steering_angle_rad = None
                self.throttle_normalized = None

        class FakePublisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        class FakeNode:
            def __init__(self, publisher):
                self.publisher = publisher

            def create_publisher(self, message_type, topic, depth):
                assert message_type is FakeDriveCommand
                assert topic in ('/odom/bridge_request_command', '/odom/bridge_actuator_feedback')
                assert depth == 100
                return self.publisher

        pub = FakePublisher()
        node = FakeNode(pub)
        ns = {
            'Any': typing.Any,
            'math': math,
            'DriveCommand': FakeDriveCommand,
            'BRIDGE_REQUEST_COMMAND_TOPIC': '/odom/bridge_request_command',
            'BRIDGE_ACTUATOR_FEEDBACK_TOPIC': '/odom/bridge_actuator_feedback',
            '_copy_stamp': lambda stamp: stamp,
            'official_bridge': SimpleNamespace(autodrive_bridge=node),
            '_bridge_request_command_publisher': None,
            '_bridge_actuator_feedback_publisher': None,
        }
        exec(compile(selected, str(BRIDGE), 'exec'), ns)
        stamp = SimpleNamespace(sec=100, nanosec=217)
        ns['_publish_exact_request_command'](
            stamp, {'V1 Steering': '-0.397', 'V1 Throttle': '0.079'})
        self.assertEqual(len(pub.messages), 1)
        message = pub.messages[0]
        self.assertEqual(message.header.stamp, stamp)
        self.assertAlmostEqual(message.steering_angle_rad,
                               -0.397 * math.pi / 6.0)
        self.assertEqual(message.throttle_normalized, 0.079)
        with self.assertRaises(RuntimeError):
            ns['_publish_exact_request_command'](
                stamp, {'V1 Steering': '0', 'V1 Throttle': 'nan'})
        self.assertEqual(len(pub.messages), 1)

    def test_command_payload_has_only_official_actuator_reset_fields(self):
        tree = ast.parse(BRIDGE.read_text())
        assignment = next(node for node in tree.body
                          if isinstance(node, ast.AnnAssign)
                          and isinstance(node.target, ast.Name)
                          and node.target.id == '_latest_command')
        self.assertEqual(set(ast.literal_eval(assignment.value)),
                         {'V1 Throttle', 'V1 Steering', 'V1 Reset'})

    def test_packet_feedback_preserves_radians_and_source_stamp(self):
        import math
        from types import SimpleNamespace
        tree = ast.parse(BRIDGE.read_text())
        selected = ast.Module(body=[node for node in tree.body
                                   if isinstance(node, ast.FunctionDef)
                                   and node.name == '_publish_packet_actuator_feedback'],
                              type_ignores=[])
        messages = []
        pub = SimpleNamespace(publish=messages.append)
        node = SimpleNamespace(create_publisher=lambda *_args: pub)
        ns = dict(Any=typing.Any, math=math,
                  DriveCommand=lambda: SimpleNamespace(header=SimpleNamespace()),
                  official_bridge=SimpleNamespace(autodrive_bridge=node),
                  _bridge_actuator_feedback_publisher=None,
                  BRIDGE_ACTUATOR_FEEDBACK_TOPIC='/odom/bridge_actuator_feedback',
                  _copy_stamp=lambda stamp: stamp)
        exec(compile(selected, str(BRIDGE), 'exec'), ns)
        publish = ns['_publish_packet_actuator_feedback']
        stamp = SimpleNamespace(sec=100, nanosec=217)
        publish(stamp, {'V1 Steering': '-0.30', 'V1 Throttle': '.4'})
        self.assertIs(messages[0].header.stamp, stamp)
        self.assertEqual(messages[0].steering_angle_rad, -.30)
        self.assertEqual(messages[0].throttle_normalized, .4)
        # Unity Socket.cs formats physical steering with ToString("F4");
        # saturated pi/6 becomes 0.5236, which is a LEGAL measured value.
        publish(stamp, {'V1 Steering': '0.5236', 'V1 Throttle': '1.0000'})
        self.assertAlmostEqual(messages[1].steering_angle_rad, math.pi/6)
        self.assertEqual(messages[1].throttle_normalized, 1.0)
        publish(stamp, {'V1 Steering': '-0.5236', 'V1 Throttle': '0.0000'})
        self.assertAlmostEqual(messages[2].steering_angle_rad, -math.pi/6)
        self.assertEqual(messages[2].throttle_normalized, 0.0)
        for steering, throttle in [('nan', '.4'), ('.8', '.4'),
                                   ('0.5237', '.4'), ('-0.5237', '.4'),
                                   ('.3', '-.1')]:
            with self.assertRaises(RuntimeError):
                publish(stamp, {'V1 Steering': steering, 'V1 Throttle': throttle})
        self.assertEqual(len(messages), 3)
        source = BRIDGE.read_text()
        handler = source[source.index('    def _packet_handler_body('):]
        self.assertLess(handler.index('_publish_packet_actuator_feedback('),
                        handler.index('result = original_handler(sid, data)'))

    def test_drive_pair_is_discretized_atomically(self):
        import math
        from types import SimpleNamespace
        tree = ast.parse(BRIDGE.read_text())
        selected = ast.Module(body=[node for node in tree.body
                                   if isinstance(node, ast.FunctionDef)
                                   and node.name == '_update_drive_command'],
                              type_ignores=[])
        payload = {'V1 Steering': '0', 'V1 Throttle': '0', 'V1 Reset': 'False'}
        snapshots = []
        class Lock:
            def __enter__(self):
                snapshots.append(dict(payload))
            def __exit__(self, *_args):
                snapshots.append(dict(payload))
        ns = dict(Any=typing.Any, math=math, time=SimpleNamespace(monotonic_ns=lambda: 123),
                  _command_lock=Lock(), _latest_command=payload)
        exec(compile(selected, str(BRIDGE), 'exec'), ns)
        update = ns['_update_drive_command']
        update(SimpleNamespace(steering_angle_rad=-.397 * math.pi / 6,
                               throttle_normalized=.079))
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]['V1 Steering'], '0')
        self.assertEqual(snapshots[1]['V1 Steering'], '-0.397')
        self.assertEqual(snapshots[1]['V1 Throttle'], '0.079')
        update(SimpleNamespace(steering_angle_rad=0., throttle_normalized=float('nan')))
        self.assertEqual(len(snapshots), 2)
        listener = BRIDGE.read_text().split('def _run_command_listener()', 1)[1]
        self.assertIn('DriveCommand, "/actuators/drive_command", _update_drive_command', listener)
        self.assertNotIn('Float32,', listener)

    def test_competition_launch_forces_development_features_off(self):
        tree = ast.parse((ROOT / 'sdu_apex_autodrive/launch/competition.launch.py').read_text())
        environments = [ast.literal_eval(keyword.value)
                        for call in ast.walk(tree) if isinstance(call, ast.Call)
                        for keyword in call.keywords
                        if keyword.arg == 'additional_env']
        self.assertEqual(len(environments), 1)
        self.assertEqual(environments[0]['SDU_APEX_DEV_SIM_RESET_ENABLED'], '0')
        self.assertEqual(environments[0]['SDU_APEX_DEV_PACKET_DIAGNOSTICS_ENABLED'], '0')


if __name__ == '__main__':
    unittest.main()
