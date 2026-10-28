"""Exercise the bridge's pure telemetry boundary without ROS or AutoDRIVE."""
import ast
from pathlib import Path
import typing
import unittest


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

    def test_command_payload_has_only_official_actuator_reset_fields(self):
        tree = ast.parse(BRIDGE.read_text())
        assignment = next(node for node in tree.body
                          if isinstance(node, ast.AnnAssign)
                          and isinstance(node.target, ast.Name)
                          and node.target.id == '_latest_command')
        self.assertEqual(set(ast.literal_eval(assignment.value)),
                         {'V1 Throttle', 'V1 Steering', 'V1 Reset'})

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
