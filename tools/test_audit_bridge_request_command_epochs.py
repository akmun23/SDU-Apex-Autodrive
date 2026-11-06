import unittest
from tools.audit_bridge_request_command_epochs import score_epochs


class BridgeRequestEpochAuditTest(unittest.TestCase):
    def test_exactly_matched_epochs_with_identical_commands_certify(self):
        request = {1234: (-.4, .079), 2234: (-.5, .1)}
        summary = score_epochs(request, request, {1234, 2234})
        self.assertTrue(summary['certified'])
        self.assertEqual(summary['odometry_command_mismatch_count'], 0)

    def test_future_command_cannot_replace_missing_request_command(self):
        requests = {1234: (-.4, .079), 2234: (-.5, .121)}
        odom = {1234: (-.5, .121)}
        summary = score_epochs(requests, odom, {1234})
        self.assertFalse(summary['certified'])
        self.assertEqual(summary['odometry_command_mismatch_count'], 1)

    def test_one_epoch_gap_is_rejected_even_if_commands_unchanged(self):
        requests = {1234: (-.4, .079)}
        odom = {1235: (-.4, .079)}
        summary = score_epochs(requests, odom, {1235})
        self.assertFalse(summary['certified'])
        self.assertEqual(summary['odometry_exact_stamp_missing'], 1)


if __name__ == '__main__':
    unittest.main()
