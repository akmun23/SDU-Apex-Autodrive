import pathlib
import re
import sys
import tempfile
import unittest

from ctypes import sizeof

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physics import (  # noqa: E402
    SOURCE_PLANT_CORE_STATE_SIZE,
    SOURCE_PLANT_STATE_SIZE,
    SourcePlant,
    _CInput,
    _COutput,
)


class SourcePlantCtypesAbiTest(unittest.TestCase):
    @staticmethod
    def c_struct_field_names(header, struct_name):
        body = re.search(
            r'typedef\s+struct\s*\{([^}]+)\}\s*' + re.escape(struct_name),
            header, re.DOTALL)
        if body is None:
            raise AssertionError(f'{struct_name} definition not found in C header')
        return re.findall(
            r'\b([A-Za-z_]\w*)\s*(?:\[[^\]]+\])?\s*;', body.group(1))

    def test_state_dimension_matches_c_header(self):
        header = (REPO_ROOT / 'f1tenth_mpc/include/unity_source_plant_c.h').read_text()
        match = re.search(
            r'UNITY_SOURCE_PLANT_STATE_SIZE\s*=\s*(\d+)', header)
        self.assertIsNotNone(match)
        self.assertEqual(SOURCE_PLANT_STATE_SIZE, int(match.group(1)))
        self.assertEqual(SOURCE_PLANT_CORE_STATE_SIZE, 18)
        self.assertEqual(_CInput._fields_[0][1]._length_,
                         SOURCE_PLANT_STATE_SIZE)
        self.assertEqual(_COutput._fields_[0][1]._length_,
                         SOURCE_PLANT_STATE_SIZE)

    def test_python_field_order_matches_c_header(self):
        header = (REPO_ROOT / 'f1tenth_mpc/include/unity_source_plant_c.h').read_text()
        for c_name, ctypes_struct in (
                ('UnitySourcePlantInputC', _CInput),
                ('UnitySourcePlantOutputC', _COutput)):
            with self.subTest(struct=c_name):
                self.assertEqual(
                    [name for name, _ in ctypes_struct._fields_],
                    self.c_struct_field_names(header, c_name))

    def test_input_struct_matches_c_layout(self):
        self.assertEqual(_CInput.wheel_rotation_angle_rad.offset, 208)
        self.assertEqual(_CInput.steering_target_rad.offset, 240)
        self.assertEqual(_CInput.throttle_normalized.offset, 248)
        self.assertEqual(_CInput.suspension_history_initialized.offset, 256)
        self.assertEqual(sizeof(_CInput), 264)

    def test_output_struct_matches_c_layout(self):
        expected_offsets = {
            'integrated_body_forward_displacement_m': 208,
            'integrated_body_lateral_displacement_m': 216,
            'integrated_yaw_delta_rad': 224,
            'wheel_rotation_angle_rad': 232,
            'suspension_history_initialized': 264,
            'longitudinal_slip': 272,
            'maximum_positive_longitudinal_slip': 304,
            'maximum_longitudinal_slip_excess_mps': 336,
            'lateral_slip': 368,
            'normal_load_n': 400,
            'suspension_force_n': 432,
            'suspension_jounce_rate_mps': 464,
            'tire_longitudinal_force_n': 496,
            'tire_lateral_force_n': 528,
            'longitudinal_acceleration_mps2': 560,
            'lateral_acceleration_mps2': 568,
            'yaw_acceleration_radps2': 576,
        }
        for field, offset in expected_offsets.items():
            with self.subTest(field=field):
                self.assertEqual(getattr(_COutput, field).offset, offset)
        self.assertEqual(sizeof(_COutput), 584)

    def test_compiled_source_plant_accepts_and_returns_full_structs(self):
        with tempfile.TemporaryDirectory(prefix='sdu-source-plant-abi-') as tmp:
            plant = SourcePlant(REPO_ROOT, pathlib.Path(tmp))
            try:
                plant.reset(0.0, 0.0, 0.0, 2.4)
                for _ in range(4):
                    state = plant.step(0.0, 0.1)
                    self.assertTrue(np.isfinite(
                        [state.u, state.v, state.r, state.steering]).all())
                self.assertGreater(state.u, 2.4)
            finally:
                plant.close()


if __name__ == '__main__':
    unittest.main()
