import unittest

from tools import analyze_mpc_prefix_chains as chains


def _record(status, bag_time, steering=None, throttle=0.2,
            primal=0.1, dual=0.2, objective=4.0, slack=0.05):
    diagnostic = {"status": status, "progress_m": bag_time,
                  "source_stamp_ns": int(bag_time * 1e9)}
    if status == chains.PREFIX_STATUS:
        diagnostic.update({
            "first_action": [steering, throttle],
            "consecutive_candidate_prefix_actions": 1,
            "minimum_predicted_corridor_slack_m": slack,
            "candidate_corridor_slack_by_stage_m": [
                {"stage": 1, "lower": slack + 0.1, "upper": slack + 0.2}],
            "solver": {"iterations": 100, "admm_status": 1,
                       "admm_stop_reason": "iteration_cap",
                       "primal_residual": primal,
                       "dual_residual": dual,
                       "objective": objective},
        })
    return {"bag_time": bag_time, "json": diagnostic}


class PrefixChainAnalysisTests(unittest.TestCase):
    def test_groups_adjacent_prefixes_and_records_the_followup_status(self):
        records = [
            _record("accepted_optimal", 1.0),
            _record(chains.PREFIX_STATUS, 2.0, -0.2, slack=0.04),
            _record(chains.PREFIX_STATUS, 3.0, 0.1, primal=0.3, dual=0.4,
                    objective=8.0, slack=-0.01),
            _record("accepted_optimal", 4.0),
            _record(chains.PREFIX_STATUS, 5.0, 0.0),
            _record("rejected_nonlinear_rollout", 6.0),
        ]
        report, rows = chains.analyze_records(records)
        self.assertEqual(report["candidate_prefix_action_count"], 3)
        self.assertEqual(report["candidate_prefix_chain_count"], 2)
        self.assertEqual(report["maximum_contiguous_chain_length"], 2)
        self.assertEqual([chain["length"] for chain in report["chains"]], [2, 1])
        first, second = report["chains"]
        self.assertEqual(first["followup_status"], "accepted_optimal")
        self.assertEqual(first["steering_sign_counts"], {
            "negative": 1, "zero": 0, "positive": 1})
        self.assertEqual(first["steering_sign_reversals"], 1)
        self.assertEqual(first["primal_residual"]["p50"], 0.2)
        self.assertEqual(first["predicted_corridor_slack_m"]["min"], -0.01)
        self.assertEqual(second["followup_status"],
                         "rejected_nonlinear_rollout")
        self.assertEqual(len(rows), 3)

    def test_uses_issued_action_when_present_and_first_action_otherwise(self):
        fallback = _record(chains.PREFIX_STATUS, 1.0, -0.3, throttle=0.15)
        issued = _record(chains.PREFIX_STATUS, 2.0, 0.0, throttle=0.0)
        issued["json"]["issued_action"] = {
            "steering_rad": 0.25, "throttle": 0.45}
        _, rows = chains.analyze_records([fallback, issued])
        self.assertEqual(rows[0]["steering_rad"], -0.3)
        self.assertEqual(rows[0]["throttle"], 0.15)
        self.assertEqual(rows[1]["steering_rad"], 0.25)
        self.assertEqual(rows[1]["throttle"], 0.45)

    def test_empty_bag_has_an_empty_but_valid_report(self):
        report, rows = chains.analyze_records([])
        self.assertEqual(rows, [])
        self.assertEqual(report["candidate_prefix_chain_count"], 0)
        self.assertEqual(report["maximum_contiguous_chain_length"], 0)


if __name__ == "__main__":
    unittest.main()
