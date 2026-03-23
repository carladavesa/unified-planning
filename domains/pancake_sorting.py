"""pancake-sorting planning domain.

This module implements the pancake sorting benchmark as a `Domain`.
It is intended to be executed via `run.py`.

Example:
  python run.py --domain pancake_sorting --compilation int --solving fast-downward
"""

import random
import re
from typing import Dict, Optional

from unified_planning.shortcuts import (
    ArrayType,
    Equals,
    Fluent,
    Int,
    IntType,
    InstantaneousAction,
    MinimizeActionCosts,
    Problem,
    RangeVariable,
)

from domains.base import Domain

INSTANCES: Dict[str, list[int]] = {
    "n5_1": [3, 4, 2, 1, 0],
    "n5_2": [3, 0, 1, 2, 4],
    "n5_3": [1, 4, 0, 3, 2],
    "n5_4": [0, 2, 3, 4, 1],
    "n5_5": [2, 4, 0, 1, 3],
    "n5_6": [4, 2, 0, 3, 1],
    "n5_7": [2, 3, 1, 0, 4],
    "n5_8": [1, 3, 2, 4, 0],
    "n5_9": [2, 3, 0, 4, 1],
    "n5_10": [0, 4, 1, 2, 3],
    "n10_1": [3, 4, 6, 8, 1, 0, 2, 9, 5, 7],
    "n10_2": [1, 8, 9, 6, 7, 5, 3, 0, 2, 4],
    "n10_3": [6, 8, 2, 0, 5, 3, 7, 1, 4, 9],
    "n10_4": [7, 9, 6, 1, 2, 8, 5, 0, 4, 3],
    "n10_5": [3, 7, 9, 4, 2, 8, 5, 6, 1, 0],
    "n10_6": [5, 3, 2, 9, 4, 8, 6, 1, 0, 7],
    "n10_7": [5, 2, 7, 4, 8, 6, 1, 0, 9, 3],
    "n10_8": [7, 5, 4, 8, 6, 2, 0, 1, 3, 9],
    "n10_9": [4, 5, 8, 6, 2, 3, 1, 9, 7, 0],
    "n10_10": [8, 6, 4, 2, 3, 5, 1, 0, 9, 7],
    "n15_1": [12, 7, 3, 4, 14, 5, 2, 9, 0, 6, 13, 8, 10, 11, 1],
    "n15_2": [1, 9, 14, 6, 5, 8, 4, 7, 0, 2, 3, 12, 13, 10, 11],
    "n15_3": [2, 0, 9, 7, 5, 4, 12, 3, 13, 14, 6, 1, 10, 11, 8],
    "n15_4": [12, 10, 13, 2, 5, 8, 3, 4, 7, 14, 11, 0, 1, 6, 9],
    "n15_5": [1, 13, 11, 10, 3, 8, 2, 5, 14, 6, 4, 7, 9, 0, 12],
    "n15_6": [4, 6, 1, 14, 8, 12, 5, 11, 2, 13, 0, 10, 9, 3, 7],
    "n15_7": [12, 10, 0, 4, 1, 14, 13, 9, 8, 3, 5, 7, 6, 11, 2],
    "n15_8": [11, 4, 10, 2, 13, 0, 14, 8, 5, 6, 1, 7, 3, 12, 9],
    "n15_9": [8, 2, 1, 11, 3, 6, 10, 9, 13, 4, 12, 14, 0, 5, 7],
    "n15_10": [0, 2, 11, 9, 8, 6, 14, 5, 3, 1, 7, 12, 10, 13, 4],
    "n20_1": [3, 17, 0, 6, 7, 15, 10, 16, 18, 11, 9, 8, 14, 1, 13, 12, 2, 5, 4, 19],
    "n20_2": [11, 13, 7, 18, 16, 14, 1, 2, 15, 6, 8, 19, 0, 12, 10, 3, 17, 5, 9, 4],
    "n20_3": [3, 17, 13, 9, 19, 15, 12, 6, 14, 10, 5, 0, 16, 2, 18, 1, 7, 8, 4, 11],
    "n20_4": [2, 10, 5, 17, 9, 7, 3, 18, 14, 12, 4, 1, 0, 19, 8, 15, 16, 6, 13, 11],
    "n20_5": [17, 3, 5, 19, 10, 18, 0, 15, 4, 6, 13, 12, 1, 14, 7, 2, 8, 16, 11, 9],
    "n20_6": [9, 0, 8, 11, 2, 7, 15, 13, 4, 5, 19, 17, 16, 14, 18, 1, 3, 10, 6, 12],
    "n20_7": [10, 8, 15, 13, 11, 4, 6, 14, 5, 3, 2, 17, 9, 0, 19, 16, 12, 18, 7, 1],
    "n20_8": [3, 18, 4, 13, 8, 19, 12, 15, 17, 11, 1, 10, 9, 0, 7, 2, 14, 5, 6, 16],
    "n20_9": [10, 14, 2, 1, 11, 8, 7, 18, 12, 6, 16, 15, 9, 5, 3, 17, 0, 19, 4, 13],
    "n20_10": [13, 8, 14, 5, 15, 17, 3, 12, 1, 6, 2, 19, 7, 18, 11, 0, 10, 4, 16, 9],
    "n25_1": [19, 17, 20, 3, 5, 13, 24, 16, 22, 0, 1, 10, 23, 2, 4, 14, 8, 18, 21, 7, 6, 15, 11, 12, 9],
    "n25_2": [16, 12, 20, 7, 10, 11, 17, 18, 14, 8, 23, 2, 4, 6, 19, 21, 1, 0, 9, 5, 3, 13, 15, 24, 22],
    "n25_3": [19, 4, 0, 23, 21, 3, 20, 6, 12, 13, 1, 15, 9, 17, 8, 11, 5, 10, 16, 18, 14, 22, 24, 7, 2],
    "n25_4": [16, 3, 23, 12, 14, 10, 24, 19, 6, 13, 7, 21, 11, 0, 15, 8, 5, 2, 9, 22, 1, 17, 4, 20, 18],
    "n25_5": [13, 20, 7, 22, 15, 19, 8, 14, 18, 24, 17, 9, 4, 23, 1, 10, 3, 5, 6, 16, 0, 2, 11, 21, 12],
    "n25_6": [15, 4, 0, 6, 18, 2, 10, 19, 16, 17, 20, 3, 22, 14, 5, 21, 23, 9, 11, 12, 1, 7, 13, 8, 24],
    "n25_7": [13, 16, 20, 1, 17, 11, 5, 12, 10, 9, 3, 19, 0, 22, 15, 24, 7, 21, 8, 14, 6, 18, 2, 4, 23],
    "n25_8": [23, 15, 11, 1, 19, 17, 0, 20, 16, 3, 8, 9, 5, 14, 22, 24, 13, 12, 18, 2, 10, 4, 21, 7, 6],
    "n25_9": [11, 8, 5, 18, 15, 19, 16, 21, 17, 22, 9, 4, 1, 2, 7, 12, 24, 0, 20, 23, 13, 6, 14, 3, 10],
    "n25_10": [17, 22, 7, 16, 5, 13, 0, 18, 12, 19, 9, 24, 11, 15, 4, 20, 23, 10, 1, 14, 21, 8, 3, 2, 6],
}


class PancakeSortingDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {"stack": v} for k, v in self._instances.items()}

    def get_instance(self, instance: Optional[str] = None) -> list[int]:
        """Return a stack instance.

        Prefer a pre-defined instance from the table. If the requested instance is
        not present, generate one randomly based on the instance name.

        If ``instance`` matches ``n<size>_<seed>`` (e.g. ``n10_1``), the name is
        used to infer both stack size and RNG seed.
        """

        # Use pre-defined instances if available.
        if instance and instance in self._instances:
            return list(self._instances[instance])

        # Try to infer size/seed from instance name like "n10_1".
        inferred_size: Optional[int] = None
        inferred_seed: Optional[int] = None
        if instance:
            m = re.match(r"^n(\d+)_(\d+)$", instance)
            if m:
                inferred_size = int(m.group(1))
                inferred_seed = int(m.group(2))

        size = inferred_size or 5
        seed = inferred_seed or 0
        return self._generate_random_stack(size, seed)

    def _generate_random_stack(self, size: int, seed: int) -> list[int]:
        rng = random.Random(seed)
        stack = list(range(size))
        rng.shuffle(stack)
        return stack

    def build_problem(self, instance: str | None = None) -> "Problem":
        stack = self.get_instance(instance)
        n = len(stack)
        lower_bound = 0
        upper_bound = n - 1

        problem = Problem("pancake_problem")

        pancake = Fluent("pancake", ArrayType(n, IntType(lower_bound, upper_bound)))
        problem.add_fluent(pancake, default_initial_value=lower_bound)
        problem.set_initial_value(pancake, stack)

        flip = InstantaneousAction("flip", f=IntType(1, n - 1))
        f = flip.parameter("f")
        b = RangeVariable("b", 0, f)
        flip.add_effect(pancake[b], pancake[f - b], forall=[b])
        problem.add_action(flip)

        for i in range(n):
            problem.add_goal(Equals(pancake[i], i))

        costs = {flip: Int(1)}
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem


DOMAIN = PancakeSortingDomain()