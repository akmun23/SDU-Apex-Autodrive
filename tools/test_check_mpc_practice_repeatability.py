import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.check_mpc_practice_repeatability import check_runs


class CheckPracticeRepeatabilityTest(unittest.TestCase):
    def setUp(self):
        self.workspace=TemporaryDirectory()
        self.root=Path(self.workspace.name)

    def tearDown(self):
        self.workspace.cleanup()

    def make(self,name,mean=4.94,gate_misses=0):
        p=self.root/name;p.mkdir()
        (p/'mpc_parameters.yaml').write_text('controlled: true\n')
        (p/'run_manifest.json').write_text(json.dumps({
            'trajectory_sha256':'same_reference',
            'map_sha256':'same_map',
            'source':{'commit_sha':'same_revision','tracked_diff_sha256':'same_tree'}}))
        (p/'mpc_run_report.json').write_text(json.dumps({'summary':{
            'recorded_lap_times_s':[6.0]+[mean]*10,
            'first_collision_from_start_s':None,
            'score_cutoff_reason':None,
            'gate_ms':{'at_or_over_budget':gate_misses},
            'status_counts':{'accepted_line_search':120},
        }}))
        return p

    def test_three_identical_safe_runs_pass(self):
        a=[self.make(f'run{i}') for i in range(3)]
        _,errors=check_runs(a)
        self.assertEqual(errors,[])

    def test_rejects_one_fast_run_or_deadlines(self):
        a=[self.make('one')]
        _,errors=check_runs(a)
        self.assertTrue(any('require 3' in x for x in errors))
        b=[self.make('two',gate_misses=1),self.make('three')]
        _,errors=check_runs(a+b)
        self.assertTrue(any('publication gate' in x for x in errors))

    def test_rejects_insufficient_average_or_different_config(self):
        a=[self.make(f'bad{i}',mean=4.97) for i in range(3)]
        _,errors=check_runs(a)
        self.assertTrue(any('average' in e for e in errors))
        (a[1]/'mpc_parameters.yaml').write_text('changed: true\n')
        _,errors=check_runs(a)
        self.assertTrue(any('fingerprint' in e for e in errors))

    def test_duplicate_run_cannot_count_as_repeat(self):
        a=self.make('same')
        with self.assertRaises(ValueError):check_runs([a,a,a])


if __name__=='__main__':unittest.main()
