"""Counters planning domain.

Example:
  python run.py --domain counters --compilation uti --solving fast-downward
"""

from typing import Dict, Optional
from unified_planning.model import Action, Expression, Object
from unified_planning.shortcuts import (
    Fluent,
    Int,
    IntType,
    InstantaneousAction,
    MinimizeActionCosts,
    Problem, UserType, Minus, Plus, LE, GE,
)
from domains.base import Domain

INSTANCES: Dict[str, tuple[int, list[int]]] = {
    "pfile1": (8, [6, 4, 2, 0]),
    "pfile2": (8, [1, 3, 7, 1]),
    "pfile3": (8, [0, 0, 0, 0]),
    "pfile4": (16, [3, 13, 0, 13, 7, 10, 8, 5]),
    "pfile5": (16, [5, 7, 0, 15, 4, 14, 10, 1]),
    "pfile6": (24, [22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]),
    "pfile7": (24, [15, 16, 1, 0, 17, 19, 20, 13, 7, 3, 0, 15]),
    "pfile8": (24, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "pfile9": (32, [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]),
    "pfile10": (40, [38, 26, 8, 3, 4, 2, 1, 5, 29, 23, 27, 7, 1, 39, 19, 15, 24, 32]),
    "pfile11": (48, [19, 17, 41, 6, 8, 29, 34, 14, 32, 43, 36, 2, 9, 26, 10, 2, 0, 17, 21, 28, 27, 30, 14, 15]),
    "pfile12": (56, [52, 18, 20, 34, 50, 50, 34, 20, 16, 13, 1, 4, 10, 18, 14, 40, 11, 26, 41, 20, 29, 37, 27, 31, 33, 13, 31, 16]),
    "pfile13": (64, [31, 20, 4, 10, 6, 57, 1, 10, 34, 29, 55, 57, 40, 39, 54, 14, 24, 34, 24, 42, 16, 37, 54, 25, 35, 58, 2, 18, 4, 49, 1, 35]),
    "pfile14": (64, [26, 7, 55, 26, 34, 34, 49, 59, 18, 36, 44, 33, 54, 33, 27, 2, 29, 29, 8, 22, 16, 45, 23, 31, 0, 8, 54, 56, 23, 10, 31, 3]),
    "pfile15": (72, [70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]),
    "pfile16": (72, [68, 2, 7, 50, 45, 7, 21, 30, 39, 43, 49, 23, 23, 11, 26, 27, 29, 62, 31, 50, 38, 9, 23, 1, 56, 47, 53, 32, 56, 34, 9, 52, 7, 67, 39, 67]),
    "pfile17": (80, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "pfile18": (80, [24, 32, 17, 3, 13, 51, 52, 75, 69, 58, 47, 23, 78, 31, 3, 42, 39, 74, 39, 54, 50, 79, 50, 62, 16, 67, 16, 9, 25, 71, 49, 76, 28, 55, 74, 76, 13, 77, 52, 22]),
    "pfile19": (80, [32, 45, 64, 11, 27, 48, 69, 42, 31, 50, 77, 26, 31, 9, 27, 26, 58, 31, 53, 17, 19, 25, 20, 31, 78, 42, 44, 22, 50, 70, 21, 3, 51, 65, 40, 1, 53, 23, 35, 71]),
    "pfile20": (80, [52, 55, 14, 37, 68, 76, 79, 74, 46, 52, 62, 72, 52, 46, 19, 62, 75, 72, 22, 20, 13, 38, 44, 26, 44, 50, 68, 76, 65, 71, 37, 48, 22, 68, 27, 79, 67, 46, 49, 22]),
}


class CountersDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict[str, list]]:
        return {k: {f"{c} counters": v} for k, (c, v) in self._instances.items()}

    def get_instance(self, instance: Optional[str] = None) -> list[int]:
        """Return a stack instance."""

        # Use pre-defined instances if available.
        if instance and instance in self._instances:
            return list(self._instances[instance])
        else:
            raise ValueError(f"Instance '{instance}' not found!")


    def build_problem(self, instance: str | None = None) -> "Problem":
        max_int_n, values = self.get_instance(instance)
        counters = len(values)

        # --- Problem ---

        problem = Problem('counters_problem')

        Counter = UserType('Counter')

        value = Fluent('value', IntType(0, max_int_n), c=Counter)
        max_int = Fluent('max_int', IntType(max_int_n, max_int_n))
        problem.add_fluent(value)
        problem.add_fluent(max_int)

        # --- Initialisation ---
        problem.set_initial_value(max_int, max_int_n)
        for c in range(counters):
            c_object = Object(f'c{c}', Counter)
            problem.add_object(c_object)
            problem.set_initial_value(value(c_object), values[c])

        # --- Actions ---
        increment = InstantaneousAction('increment', c=Counter)
        c = increment.parameter('c')
        increment.add_precondition(LE(Plus(value(c), 1), max_int_n))
        increment.add_increase_effect(value(c), 1)

        decrement = InstantaneousAction('decrement', c=Counter)
        c = decrement.parameter('c')
        decrement.add_precondition(GE(Minus(value(c), 1), 0))
        decrement.add_decrease_effect(value(c), 1)

        problem.add_actions([increment, decrement])

        for c in range(counters - 1):
            problem.add_goal(
                LE(Plus(value(problem.object(f'c{c}')), 1), value(problem.object(f'c{c + 1}')))
            )

        costs: Dict[Action, Expression] = {
            increment: Int(1),
            decrement: Int(1)
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem


DOMAIN = CountersDomain()