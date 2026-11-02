import csv
import json
import tempfile
import unittest
from pathlib import Path
from tools.extract_mpc_rejection_source_case import extract


class RejectionFixtureTest(unittest.TestCase):
    def test_nominal_controls_are_not_shifted_before_replay_injection(self):
        failure={
            'source_stamp_ns':1793543092414731421,
            'candidate_command_stamp_ns':1793543092464731421,
            'progress_m':227.063158,
            'active_speed_ceiling_mps':16,
            'state':[.1,.3,4.,0.,-.6,.37,-.2,-.2,.17,.17],
            'plant_internal_state':[float(i) for i in range(22)],
            'nominal_controls':[[.001*i,.05+.01*i] for i in range(30)],
            'plant_state_flags':[1,0,1],
            'nonlinear_failure_reason':'corridor',
            'nonlinear_failure_stage':15,
        }
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            log=base/'watcher.log'
            log.write_text('MPC_CYCLE_REJECTED: '+json.dumps(failure)+'\n')
            report=extract(log,base/'fixture')
            self.assertEqual(report['internal_state_length'],22)
            with (base/'fixture'/'nominal_source_seed.csv').open() as stream:
                seed=next(csv.reader(stream))
            self.assertEqual(len(seed),61)
            # The replay's warm-start importer already duplicates seed[0]
            # once; this generated CSV MUST preserve all 30 direct commands.
            self.assertAlmostEqual(float(seed[1]),0.)
            self.assertAlmostEqual(float(seed[2]),.05)
            self.assertAlmostEqual(float(seed[3]),.001)
            self.assertAlmostEqual(float(seed[-2]),.029)
            self.assertAlmostEqual(float(seed[-1]),.34)
            with (base/'fixture'/'physical_state.csv').open() as stream:
                state_row=next(csv.reader(stream))
                self.assertEqual(len(state_row),38)
                self.assertEqual(int(state_row[0]),failure['candidate_command_stamp_ns'])
                self.assertNotEqual(int(state_row[0]),failure['source_stamp_ns'])
                self.assertEqual(float(state_row[13]),0.)
            self.assertEqual(report['stamp_ns'],failure['candidate_command_stamp_ns'])
            del failure['candidate_command_stamp_ns']
            log.write_text('MPC_CYCLE_REJECTED: '+json.dumps(failure)+'\n')
            with self.assertRaisesRegex(ValueError,'missing command epoch'):
                extract(log,base/'ambiguous')


if __name__=='__main__':
    unittest.main()
