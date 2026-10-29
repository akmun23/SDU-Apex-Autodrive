"""v2.3.10 source-frame correction; explicitly constrain the wheel-order sign."""
from dataclasses import dataclass
import numpy as np
import pytest
from source_vehicle import convert_unity_suspension_mounts_to_model_frame
from diagnostic_fixtures.diagnose_suspension_frame_v239 import compare_candidates
from test_v239_suspension_frame import SuspensionModel,SuspensionWheel
from test_v238_scalar_force_audit import seed


def test_native_unity_mounts_convert_to_model_left_positive_and_preserve_other_axes():
    unity=((-0.1157,0.061,0.1747),(.1159,0.060,0.1747),
           (-.1179,0.050,-.1553),(.1181,.05,-.1553))
    model=convert_unity_suspension_mounts_to_model_frame(unity)
    assert (model[0][0]>0>model[1][0]) and (model[2][0]>0>model[3][0])
    for i in range(4):
        assert model[i]==(-unity[i][0],unity[i][1],unity[i][2])
    assert unity[0][0]<0


def test_fails_closed_on_unexpected_wheel_order_or_missing_mount():
    with pytest.raises(ValueError,match='four'):
        convert_unity_suspension_mounts_to_model_frame([(0,0,0)])
    with pytest.raises(ValueError,match='FL FR RL RR'):
        convert_unity_suspension_mounts_to_model_frame([(.1,.05,.17),(-.1,.05,.17),(.1,.05,-.16),(-.1,.05,-.16)])


def test_legacy_vs_corrected_audit_remains_byte_equivalent_after_model_conversion():
    s=seed(25)
    s['roll_rad']=np.linspace(-.04,.03,25)
    s['roll_rate_radps']=np.linspace(-.3,.2,25)
    s['pitch_rad']=np.full(25,.02)
    legacy=SuspensionModel()
    corrected=SuspensionModel(suspension_base_xyz_body_m=convert_unity_suspension_mounts_to_model_frame(legacy.suspension_base_xyz_body_m))
    report_old,_=compare_candidates(s,legacy,SuspensionWheel)
    report_new,_=compare_candidates(s,corrected,SuspensionWheel)
    for key in ('original_unmodified','mirrored_lateral_mount_x'):
        np.testing.assert_allclose(report_old['candidate_results'][key]['global_rms_load_error_n'],
                                   report_new['candidate_results'][key]['global_rms_load_error_n'],atol=1e-13)
    assert report_new['geometry_reflection_equals_sign_reversed_native_roll_max_error_n']<1e-10
