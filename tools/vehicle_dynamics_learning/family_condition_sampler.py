"""Hierarchical run-family/condition sampling for correlated vehicle data."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Hashable, Mapping, Sequence, TypeVar

import numpy as np


Item = TypeVar("Item", bound=Hashable)


class FamilyConditionSampler:
    """Sample family, then run, then condition, then one eligible item.

    ``items_by_run`` contains only training-eligible sequences/windows. Items
    sharing a schema-7 condition ID are sampled as one condition, preventing a
    condition split into many reset/gap segments from gaining extra weight.
    """

    def __init__(self, items_by_run: Mapping[int, Sequence[Item]],
                 run_families: Sequence[str],
                 condition_for: Callable[[int, Item], Hashable | None],
                 family_weights: Mapping[str, float] | None = None) -> None:
        nested: dict[str, dict[int, dict[Hashable, list[Item]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        for run_value, items in items_by_run.items():
            run = int(run_value)
            if run < 0 or run >= len(run_families):
                raise ValueError(f"run index {run} has no family metadata")
            family = str(run_families[run])
            for item in items:
                condition = condition_for(run, item)
                if condition is None:
                    condition = "__unlabelled__"
                nested[family][run][condition].append(item)
        if not nested:
            raise ValueError("family sampler received no eligible items")

        present = sorted(nested)
        if family_weights is None:
            weights = {family: 1.0 for family in present}
        else:
            weights = {family: float(family_weights.get(family, 0.0))
                       for family in present}
        if any(not np.isfinite(value) or value < 0.0
               for value in weights.values()) or sum(weights.values()) <= 0.0:
            raise ValueError("family weights must be finite, nonnegative, and nonzero")
        total = sum(weights.values())
        self.family_probabilities = {
            family: weights[family] / total for family in present
        }
        self._nested = dict(nested)
        self._families = tuple(present)
        self._probabilities = np.asarray(
            [self.family_probabilities[family] for family in present],
            dtype=np.float64)
        self.metadata = {
            "hierarchy": ["family", "run", "condition", "eligible_sequence_or_window"],
            "family_probabilities": self.family_probabilities,
            "run_count_by_family": {
                family: len(self._nested[family]) for family in present
            },
            "condition_count_by_family": {
                family: sum(len(conditions)
                            for conditions in self._nested[family].values())
                for family in present
            },
        }

    def sample(self, rng: np.random.Generator) -> tuple[int, Item, str, Hashable]:
        family_index = int(rng.choice(len(self._families), p=self._probabilities))
        family = self._families[family_index]
        runs = tuple(sorted(self._nested[family]))
        run = int(runs[int(rng.integers(0, len(runs)))])
        conditions = tuple(self._nested[family][run])
        condition = conditions[int(rng.integers(0, len(conditions)))]
        items = self._nested[family][run][condition]
        item = items[int(rng.integers(0, len(items)))]
        return run, item, family, condition


def sequence_condition_lookup(data: Mapping[str, Any]
                              ) -> dict[tuple[int, int], Hashable | None]:
    """Map absolute sequence bounds to the schema-7 condition ID."""
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    conditions = data.get("sequence_condition_id")
    if conditions is None:
        return {(int(start), int(end)): None for start, end in bounds}
    conditions = np.asarray(conditions, dtype=np.int64)
    if conditions.shape != (len(bounds),):
        raise ValueError("sequence condition IDs do not align with sequence bounds")
    return {(int(start), int(end)): int(condition)
            for (start, end), condition in zip(bounds, conditions)}


def condition_id_for_item(data: Mapping[str, Any], run: int, item: Any,
                          exact_bounds: Mapping[tuple[int, int], Hashable | None]
                          ) -> Hashable | None:
    """Resolve an item (sequence ID or contained interval) to its condition."""
    conditions = data.get("sequence_condition_id")
    if conditions is None:
        return None
    if isinstance(item, (int, np.integer)):
        sequence_id = int(item)
        if (sequence_id < 0 or sequence_id >= len(conditions)
                or int(data["seq_run"][sequence_id]) != run):
            raise ValueError("sequence item and run index disagree")
        return int(conditions[sequence_id])
    start, end = map(int, item[:2])
    exact = exact_bounds.get((start, end))
    if exact is not None:
        return exact
    for sequence_id, (sequence_start, sequence_end) in enumerate(data["bounds"]):
        if (int(data["seq_run"][sequence_id]) == run
                and int(sequence_start) <= start and end <= int(sequence_end)):
            return int(conditions[sequence_id])
    raise ValueError("sample interval is not contained in a source sequence")


def build_sequence_sampler(data: Mapping[str, Any],
                           items_by_run: Mapping[int, Sequence[Item]],
                           family_weights: Mapping[str, float] | None = None
                           ) -> FamilyConditionSampler:
    """Build a sampler for sequence IDs or intervals derived from sequences."""
    families = run_family_labels(data)
    if (family_weights is None
            and data.get("training_family_names") is not None
            and data.get("training_family_probabilities") is not None):
        names = np.asarray(data["training_family_names"]).astype(str)
        probabilities = np.asarray(
            data["training_family_probabilities"], dtype=np.float64)
        if names.shape != probabilities.shape:
            raise ValueError("training family names and weights do not align")
        family_weights = dict(zip(names.tolist(), probabilities.tolist()))
    exact_bounds = sequence_condition_lookup(data)
    return FamilyConditionSampler(
        items_by_run, families,
        lambda run, item: condition_id_for_item(data, run, item, exact_bounds),
        family_weights=family_weights)


def run_family_labels(data: Mapping[str, Any]) -> np.ndarray:
    """Return archived family labels, with a narrow legacy name fallback."""
    training_families = data.get("training_families")
    if training_families is not None:
        return np.asarray(training_families).astype(str)
    families = data.get("run_families")
    if families is not None:
        return np.asarray(families).astype(str)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    return np.asarray([
        "practice_track" if run_id.startswith("practice_") else "open_plane"
        for run_id in run_ids
    ])
