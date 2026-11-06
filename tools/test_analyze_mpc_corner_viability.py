"""Offline corner-attribution regression: wrap, lap boundaries, and safety events."""
import json
import tempfile
import unittest
from pathlib import Path
from tools.analyze_mpc_corner_viability import summarize


class CornerViabilityTest(unittest.TestCase):
    def test_sector_wrap_and_lap_command_change_reset(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            (p / 'mpc_run_report.json').write_text(json.dumps({
                'per_official_lap_tracking': [
                    {'bag_receipt_start_s': 0, 'bag_receipt_end_s': 10},
                    {'bag_receipt_start_s': 10, 'bag_receipt_end_s': 20},
                    {'bag_receipt_start_s': 20, 'bag_receipt_end_s': 30},
                ]}))
            events = [
                (10.01, 27.0, 'accepted_optimal', -0.01, 3.0),
                (10.04, 28.4, 'accepted_viability_recovery', -0.12, 2.5),
                (21.02, 54.8, 'accepted_optimal', 0.20, 3.5),
            ]
            with (p / 'mpc_cycle_detail_diagnostics.jsonl').open('w') as f:
                for epoch, progress, status, steering, speed in events:
                    f.write(json.dumps({
                        'bag_receipt_epoch_s': epoch,
                        'payload': {
                            'progress_m': progress, 'state': [.07, .03, speed],
                            'reference': {'speed_reference_mps': 4.0},
                            'status': status,
                            'issued_action': {'steering_rad': steering, 'throttle': .2},
                            'nonlinear_failure_stage': 26,
                            'nonlinear_failure_reason': 'corridor',
                            'solver': {'line_search_trials': 4},
                        }}) + '\n')
            output = summarize(p, 26.6)
            self.assertEqual(output['num_scored_laps'], 2)
            self.assertEqual([x['time_s'] for x in output['laps']], [10.0, 10.0])
            first = output['laps'][0]['sectors'][1]
            self.assertEqual(first['accepted_recoveries'], 1)
            self.assertEqual(first['command_steer_jumps_gt_0p08'], 1)
            self.assertEqual(first['line_search_four_trials'], 1)
            self.assertEqual(output['laps'][1]['sectors'][1]['command_steer_jumps_gt_0p08'], 0)
            self.assertEqual(output['laps'][0]['recoveries'][0]['source_failure_stage'], 26)


if __name__ == '__main__':
    unittest.main()
