from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.family_condition_sampler import (
    FamilyConditionSampler,
)


class FamilyConditionSamplerTest(unittest.TestCase):
    def test_balances_families_then_runs_then_conditions(self):
        groups = {
            0: [("a", 0)] * 40 + [("a", 1)],
            1: [("a", 2)],
            2: [("b", 3)],
        }
        sampler = FamilyConditionSampler(
            groups, ["large", "large", "small"],
            lambda run, item: item[1])
        rng = np.random.default_rng(9182)
        families = {"large": 0, "small": 0}
        runs = {0: 0, 1: 0, 2: 0}
        conditions = {0: 0, 1: 0}
        for _ in range(30000):
            run, item, family, condition = sampler.sample(rng)
            families[family] += 1
            runs[run] += 1
            if family == "large" and run == 0:
                conditions[condition] += 1
            self.assertEqual(item[1], condition)
        self.assertAlmostEqual(families["large"] / 30000, 0.5, delta=0.015)
        self.assertAlmostEqual(runs[0] / runs[1], 1.0, delta=0.08)
        self.assertAlmostEqual(conditions[0] / conditions[1], 1.0, delta=0.08)
        self.assertEqual(sampler.metadata["family_probabilities"],
                         {"large": 0.5, "small": 0.5})

    def test_rejects_invalid_weights_and_empty_eligible_groups(self):
        with self.assertRaises(ValueError):
            FamilyConditionSampler({}, ["a"], lambda _run, _item: 0)
        with self.assertRaises(ValueError):
            FamilyConditionSampler({0: [1]}, ["a"], lambda _run, _item: 0,
                                   {"a": -1.0})


if __name__ == "__main__":
    unittest.main()
