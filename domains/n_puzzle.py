"""n_puzzle planning domain.

This module implements the n_puzzle benchmark as a `Domain`.
It is intended to be executed via `run.py`.

Example:
  python run.py --domain n_puzzle --compilation uti --solving fast-downward
"""
import math
from typing import Dict, Optional
from unified_planning.model import Action, Expression
from unified_planning.shortcuts import (
    ArrayType,
    Equals,
    Fluent,
    Int,
    IntType,
    InstantaneousAction,
    MinimizeActionCosts,
    Problem,
    Not,
)

from domains.base import Domain

INSTANCES: Dict[str, list[int]] = {
    "korf1": [14,13,15,7,11,12,9,5,6,0,2,1,4,8,10,3],
    "korf2": [13,5,4,10,9,12,8,14,2,3,7,1,0,15,11,6],
    "korf3": [14,7,8,2,13,11,10,4,9,12,5,0,3,6,1,15],
    "korf4": [5,12,10,7,15,11,14,0,8,2,1,13,3,4,9,6],
    "korf5": [4,7,14,13,10,3,9,12,11,5,6,15,1,2,8,0],
    "korf6": [14,7,1,9,12,3,6,15,8,11,2,5,10,0,4,13],
    "korf7": [2,11,15,5,13,4,6,7,12,8,10,1,9,3,14,0],
    "korf8": [12,11,15,3,8,0,4,2,6,13,9,5,14,1,10,7],
    "korf9": [3,14,9,11,5,4,8,2,13,12,6,7,10,1,15,0],
    "korf10": [13,11,8,9,0,15,7,10,4,3,6,14,5,12,2,1],
    "korf11": [5,9,13,14,6,3,7,12,10,8,4,0,15,2,11,1],
    "korf12": [14,1,9,6,4,8,12,5,7,2,3,0,10,11,13,15],
    "korf13": [3,6,5,2,10,0,15,14,1,4,13,12,9,8,11,7],
    "korf14": [7,6,8,1,11,5,14,10,3,4,9,13,15,2,0,12],
    "korf15": [13,11,4,12,1,8,9,15,6,5,14,2,7,3,10,0],
    "korf16": [1,3,2,5,10,9,15,6,8,14,13,11,12,4,7,0],
    "korf17": [15,14,0,4,11,1,6,13,7,5,8,9,3,2,10,12],
    "korf18": [6,0,14,12,1,15,9,10,11,4,7,2,8,3,5,13],
    "korf19": [7,11,8,3,14,0,6,15,1,4,13,9,5,12,2,10],
    "korf20": [6,12,11,3,13,7,9,15,2,14,8,10,4,1,5,0],
    "korf21": [12,8,14,6,11,4,7,0,5,1,10,15,3,13,9,2],
    "korf22": [14,3,9,1,15,8,4,5,11,7,10,13,0,2,12,6],
    "korf23": [10,9,3,11,0,13,2,14,5,6,4,7,8,15,1,12],
    "korf24": [7,3,14,13,4,1,10,8,5,12,9,11,2,15,6,0],
    "korf25": [11,4,2,7,1,0,10,15,6,9,14,8,3,13,5,12],
    "korf26": [5,7,3,12,15,13,14,8,0,10,9,6,1,4,2,11],
    "korf27": [14,1,8,15,2,6,0,3,9,12,10,13,4,7,5,11],
    "korf28": [13,14,6,12,4,5,1,0,9,3,10,2,15,11,8,7],
    "korf29": [9,8,0,2,15,1,4,14,3,10,7,5,11,13,6,12],
    "korf30": [12,15,2,6,1,14,4,8,5,3,7,0,10,13,9,11],
    "korf31": [12,8,15,13,1,0,5,4,6,3,2,11,9,7,14,10],
    "korf32": [14,10,9,4,13,6,5,8,2,12,7,0,1,3,11,15],
    "korf33": [14,3,5,15,11,6,13,9,0,10,2,12,4,1,7,8],
    "korf34": [6,11,7,8,13,2,5,4,1,10,3,9,14,0,12,15],
    "korf35": [1,6,12,14,3,2,15,8,4,5,13,9,0,7,11,10],
    "korf36": [12,6,0,4,7,3,15,1,13,9,8,11,2,14,5,10],
    "korf37": [8,1,7,12,11,0,10,5,9,15,6,13,14,2,3,4],
    "korf38": [7,15,8,2,13,6,3,12,11,0,4,10,9,5,1,14],
    "korf39": [9,0,4,10,1,14,15,3,12,6,5,7,11,13,8,2],
    "korf40": [11,5,1,14,4,12,10,0,2,7,13,3,9,15,6,8],
    "korf41": [8,13,10,9,11,3,15,6,0,1,2,14,12,5,4,7],
    "korf42": [4,5,7,2,9,14,12,13,0,3,6,11,8,1,15,10],
    "korf43": [11,15,14,13,1,9,10,4,3,6,2,12,7,5,8,0],
    "korf44": [12,9,0,6,8,3,5,14,2,4,11,7,10,1,15,13],
    "korf45": [3,14,9,7,12,15,0,4,1,8,5,6,11,10,2,13],
    "korf46": [8,4,6,1,14,12,2,15,13,10,9,5,3,7,0,11],
    "korf47": [6,10,1,14,15,8,3,5,13,0,2,7,4,9,11,12],
    "korf48": [8,11,4,6,7,3,10,9,2,12,15,13,0,1,5,14],
    "korf49": [10,0,2,4,5,1,6,12,11,13,9,7,15,3,14,8],
    "korf50": [12,5,13,11,2,10,0,9,7,8,4,3,14,6,15,1],
    "korf51": [10,2,8,4,15,0,1,14,11,13,3,6,9,7,5,12],
    "korf52": [10,8,0,12,3,7,6,2,1,14,4,11,15,13,9,5],
    "korf53": [14,9,12,13,15,4,8,10,0,2,1,7,3,11,5,6],
    "korf54": [12,11,0,8,10,2,13,15,5,4,7,3,6,9,14,1],
    "korf55": [13,8,14,3,9,1,0,7,15,5,4,10,12,2,6,11],
    "korf56": [3,15,2,5,11,6,4,7,12,9,1,0,13,14,10,8],
    "korf57": [5,11,6,9,4,13,12,0,8,2,15,10,1,7,3,14],
    "korf58": [5,0,15,8,4,6,1,14,10,11,3,9,7,12,2,13],
    "korf59": [15,14,6,7,10,1,0,11,12,8,4,9,2,5,13,3],
    "korf60": [11,14,13,1,2,3,12,4,15,7,9,5,10,6,8,0],
    "korf61": [6,13,3,2,11,9,5,10,1,7,12,14,8,4,0,15],
    "korf62": [4,6,12,0,14,2,9,13,11,8,3,15,7,10,1,5],
    "korf63": [8,10,9,11,14,1,7,15,13,4,0,12,6,2,5,3],
    "korf64": [5,2,14,0,7,8,6,3,11,12,13,15,4,10,9,1],
    "korf65": [7,8,3,2,10,12,4,6,11,13,5,15,0,1,9,14],
    "korf66": [11,6,14,12,3,5,1,15,8,0,10,13,9,7,4,2],
    "korf67": [7,1,2,4,8,3,6,11,10,15,0,5,14,12,13,9],
    "korf68": [7,3,1,13,12,10,5,2,8,0,6,11,14,15,4,9],
    "korf69": [6,0,5,15,1,14,4,9,2,13,8,10,11,12,7,3],
    "korf70": [2,11,15,5,13,4,6,7,12,8,10,1,9,3,14,0],
    "korf71": [5,7,0,11,12,1,9,10,15,6,2,3,8,4,13,14],
    "korf72": [12,15,11,10,4,5,14,0,13,7,1,2,9,8,3,6],
    "korf73": [6,14,10,5,15,8,7,1,3,4,2,0,12,9,11,13],
    "korf74": [14,13,4,11,15,8,6,9,0,7,3,1,2,10,12,5],
    "korf75": [14,4,0,10,6,5,1,3,9,2,13,15,12,7,8,11],
    "korf76": [15,10,8,3,0,6,9,5,1,14,13,11,7,2,12,4],
    "korf77": [0,13,2,4,12,14,6,9,15,1,10,3,11,5,8,7],
    "korf78": [3,14,13,6,4,15,8,9,5,12,10,0,2,7,1,11],
    "korf79": [0,1,9,7,11,13,5,3,14,12,4,2,8,6,10,15],
    "korf80": [11,0,15,8,13,12,3,5,10,1,4,6,14,9,7,2],
    "korf81": [13,0,9,12,11,6,3,5,15,8,1,10,4,14,2,7],
    "korf82": [14,10,2,1,13,9,8,11,7,3,6,12,15,5,4,0],
    "korf83": [12,3,9,1,4,5,10,2,6,11,15,0,14,7,13,8],
    "korf84": [15,8,10,7,0,12,14,1,5,9,6,3,13,11,4,2],
    "korf85": [4,7,13,10,1,2,9,6,12,8,14,5,3,0,11,15],
    "korf86": [6,0,5,10,11,12,9,2,1,7,4,3,14,8,13,15],
    "korf87": [9,5,11,10,13,0,2,1,8,6,14,12,4,7,3,15],
    "korf88": [15,2,12,11,14,13,9,5,1,3,8,7,0,10,6,4],
    "korf89": [11,1,7,4,10,13,3,8,9,14,0,15,6,5,2,12],
    "korf90": [5,4,7,1,11,12,14,15,10,13,8,6,2,0,9,3],
    "korf91": [9,7,5,2,14,15,12,10,11,3,6,1,8,13,0,4],
    "korf92": [3,2,7,9,0,15,12,4,6,11,5,14,8,13,10,1],
    "korf93": [13,9,14,6,12,8,1,2,3,4,0,7,5,10,11,15],
    "korf94": [5,7,11,8,0,14,9,13,10,12,3,15,6,1,4,2],
    "korf95": [4,3,6,13,7,15,9,0,10,5,8,11,2,12,1,14],
    "korf96": [1,7,15,14,2,6,4,9,12,11,13,3,0,8,5,10],
    "korf97": [9,14,5,7,8,15,1,2,10,4,13,6,12,0,11,3],
    "korf98": [0,11,3,12,5,2,1,9,8,10,14,15,7,4,13,6],
    "korf99": [7,15,4,0,10,9,2,5,12,11,13,6,1,3,14,8],
    "korf100": [11,4,0,8,6,10,5,13,12,7,14,3,1,2,9,15],
}


class NPuzzleDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {"grid": v} for k, v in self._instances.items()}

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
        else:
            raise ValueError(f"Instance '{instance}' not found!")


    def build_problem(self, instance: str | None = None) -> "Problem":
        instance = self.get_instance(instance)
        l = len(instance) - 1
        n = math.isqrt(l+1)
        goal_blocks = [list(range(i * n, i * n + n)) for i in range(n)]
        initial_blocks = [instance[i:i + n] for i in range(0, l+1, n)]

        # --- Problem ---
        problem = Problem('npuzzle_problem')

        puzzle = Fluent('puzzle', ArrayType(n, ArrayType(n, IntType(0, l))))
        problem.add_fluent(puzzle, default_initial_value=0)
        problem.set_initial_value(puzzle, initial_blocks)

        # --- Actions ---
        move_up = InstantaneousAction('move_up', r=IntType(0, n - 1), c=IntType(0, n - 1))
        c = move_up.parameter('c')
        r = move_up.parameter('r')
        move_up.add_precondition(Equals(puzzle[r - 1][c], 0))
        move_up.add_precondition(Not(Equals(puzzle[r][c], 0)))
        move_up.add_effect(puzzle[r - 1][c], puzzle[r][c])
        move_up.add_effect(puzzle[r][c], 0)

        move_down = InstantaneousAction('move_down', r=IntType(0, n - 1), c=IntType(0, n - 1))
        c = move_down.parameter('c')
        r = move_down.parameter('r')
        move_down.add_precondition(Equals(puzzle[r + 1][c], 0))
        move_down.add_precondition(Not(Equals(puzzle[r][c], 0)))
        move_down.add_effect(puzzle[r + 1][c], puzzle[r][c])
        move_down.add_effect(puzzle[r][c], 0)

        move_left = InstantaneousAction('move_left', r=IntType(0, n - 1), c=IntType(0, n - 1))
        c = move_left.parameter('c')
        r = move_left.parameter('r')
        move_left.add_precondition(Equals(puzzle[r][c - 1], 0))
        move_left.add_precondition(Not(Equals(puzzle[r][c], 0)))
        move_left.add_effect(puzzle[r][c - 1], puzzle[r][c])
        move_left.add_effect(puzzle[r][c], 0)

        move_right = InstantaneousAction('move_right', r=IntType(0, n - 1), c=IntType(0, n - 1))
        c = move_right.parameter('c')
        r = move_right.parameter('r')
        move_right.add_precondition(Equals(puzzle[r][c + 1], 0))
        move_right.add_precondition(Not(Equals(puzzle[r][c], 0)))
        move_right.add_effect(puzzle[r][c + 1], puzzle[r][c])
        move_right.add_effect(puzzle[r][c], 0)

        problem.add_actions([move_up, move_down, move_left, move_right])

        # --- Goals ---
        problem.add_goal(Equals(puzzle, goal_blocks))

        # --- Costs ---
        costs: Dict[Action, Expression] = {
            move_up: Int(1), move_down: Int(1), move_left: Int(1), move_right: Int(1)
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem


DOMAIN = NPuzzleDomain()