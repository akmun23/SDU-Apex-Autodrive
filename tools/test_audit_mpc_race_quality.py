import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.audit_mpc_race_quality import summarize


class RaceQualityTests(unittest.TestCase):
    def create_run(self, root, name, laps, gate=0, collision=None):
        p=root/name
        p.mkdir()
        (p/'mpc_run_report.json').write_text(json.dumps({'summary':{
            'recorded_lap_times_s':[5.1]+laps,
            'gate_ms':{'at_or_over_budget':gate},
            'score_cutoff_reason':None,
            'first_collision_from_start_s':collision,
            'status_counts':{'accepted_optimal':80,'accepted_prior_plan':1}}}))
        with (p/'mpc_cycle_detail.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['status','u_mps','target_u_mps',
                 'e_y_m','e_psi_rad','issued_command_throttle',
                 'issued_command_steering_rad','corridor_slack_m',
                 'callback_to_gate_ms','node_solve_ms'])
            w.writeheader();w.writerow({'status':'accepted_optimal','u_mps':4.5,
                 'target_u_mps':5,'e_y_m':.01,'e_psi_rad':-.02,
                 'issued_command_throttle':.2,'issued_command_steering_rad':-.1,
                 'corridor_slack_m':.03,
                 'callback_to_gate_ms':17.2,'node_solve_ms':15.6})
        return p

    def test_pass_requires_both_timing_thresholds_and_zero_deadlines(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            fast=self.create_run(root,'fast',[4.9]*10)
            d=summarize(fast)
            self.assertTrue(d['clean_ten_lap_completion'])
            self.assertTrue(d['passed_mean_target'])
            self.assertTrue(d['passed_total_target'])
            self.assertAlmostEqual(d['timing']['callback_headroom_ms']['minimum'],2.8)
            self.assertEqual(d['timing']['callback_to_gate_ms']['max'],17.2)
            slow=self.create_run(root,'not_fast',[4.97]*10)
            d=summarize(slow)
            self.assertFalse(d['passed_mean_target'])
            self.assertTrue(d['passed_total_target'])
            faulty=self.create_run(root,'faulty',[4.9]*10,gate=1)
            self.assertFalse(summarize(faulty)['clean_ten_lap_completion'])

    def test_independent_three_run_gate_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            paths=[self.create_run(root,'r'+str(i),[4.9]*10) for i in range(3)]
            script=Path(__file__).with_name('audit_mpc_race_quality.py')
            ok=subprocess.run([sys.executable,str(script),*map(str,paths),
                '--enforce-repeatability'],capture_output=True,text=True)
            self.assertEqual(ok.returncode,0,ok.stderr)
            self.assertTrue(json.loads(ok.stdout)['repeatability_gate']['passed'])
            bad=subprocess.run([sys.executable,str(script),str(paths[0]),
                str(paths[0]),str(paths[1]),'--enforce-repeatability'],
                capture_output=True,text=True)
            self.assertEqual(bad.returncode,1)
            self.assertFalse(json.loads(bad.stdout)['repeatability_gate']['passed'])


if __name__=='__main__':
    unittest.main()
