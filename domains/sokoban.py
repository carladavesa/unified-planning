"""sokoban planning domain.

This module implements the sokoban benchmark as a `Domain`.
It is intended to be executed via `run.py`.

Example:
  python run.py --domain sokoban --compilation up --solving fast-downward
"""
import os
from typing import Dict, Optional
from unified_planning.model import Action, Expression, Object
from unified_planning.shortcuts import (
    ArrayType,
    Fluent,
    Int,
    IntType,
    InstantaneousAction,
    MinimizeActionCosts,
    Problem, UserType, Not,
)

from domains.base import Domain

PROBS_DIR = os.path.join(os.path.dirname(__file__), 'sokoban/probs')

INSTANCES: list[str] = [f"i_{i}" for i in range(1, 21)]

class SokobanDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_instance(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        matrix = [list(line.rstrip('\n')) for line in lines]
        initial_state = {}
        defined_positions = []
        goal_positions = []

        row_limits = {}
        col_limits = {}

        # Row limits
        for i, row in enumerate(matrix[1:-1]):
            if all(x == '#' for x in row) or '#' not in row:
                row_limits[i] = (0, len(row) - 3)
            else:
                start = row.index('#')
                while row[start] == '#':
                    start += 1
                end = len(row) - 1
                while row[end] == '#':
                    end -= 1
                row_limits[i] = (start - 1, end - 1)

        # Column limits
        num_cols = max(len(row) for row in matrix)
        padded_matrix = [list(''.join(row).ljust(num_cols)) for row in matrix]
        cols = []
        for col_idx in range(num_cols):
            cols.append([row[col_idx] for row in padded_matrix])
        for i, col in enumerate(cols[1:-1]):
            if all(x == '#' for x in col) or '#' not in col:
                col_limits[i] = (0, len(col) - 3)
            else:
                start = col.index('#')
                while col[start] == '#':
                    start += 1
                end = len(col) - 1 - col[::-1].index('#')
                while col[end] == '#':
                    end -= 1
                col_limits[i] = (start - 1, end - 1)

        # Parse cells
        for ri, row in enumerate(matrix[1:-1]):
            for ci, cell in enumerate(row[1:-1]):
                if (cell != '#'
                        and ci in col_limits and ri in row_limits
                        and col_limits[ci][0] <= ri <= col_limits[ci][1]
                        and row_limits[ri][0] <= ci <= row_limits[ri][1]):
                    defined_positions.append((ri, ci))
                    if cell == '@':
                        initial_state[(ri, ci)] = 'P'
                    elif cell == '$':
                        initial_state[(ri, ci)] = 'B'
                    elif cell == '.':
                        goal_positions.append((ri, ci))
                    elif cell == '*':
                        goal_positions.append((ri, ci))
                        initial_state[(ri, ci)] = 'B'

        rows = max(x[0] for x in defined_positions) + 1
        columns = max(x[1] for x in defined_positions) + 1
        all_positions = {(r, c) for r in range(rows) for c in range(columns)}
        undefined_positions = sorted(all_positions - set(defined_positions))

        return {
            'initial_state': initial_state,
            'undefined_positions': undefined_positions,
            'goal_positions': goal_positions,
            'rows': rows,
            'columns': columns,
        }

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if instance and instance in self._instances:
            path = os.path.join(PROBS_DIR, f"{instance}.txt")
            return self._parse_instance(path)
        raise ValueError(f"Instance '{instance}' not found!")

    def build_problem(self, instance: str | None = None) -> "Problem":
        data = self.get_instance(instance)

        initial_state = data['initial_state']
        undefined_positions = data['undefined_positions']
        goal_positions = data['goal_positions']
        rows = data['rows']
        columns = data['columns']

        # --- Problem ---
        problem = Problem('sokoban_problem')

        Pattern = UserType('Pattern')
        P = Object('P', Pattern)  # Player
        B = Object('B', Pattern)  # Box
        pattern_by_symbol = {'P': P, 'B': B}
        problem.add_objects([P, B])

        grid = Fluent('grid', ArrayType(rows, ArrayType(columns)), p=Pattern, undefined_positions=undefined_positions)
        problem.add_fluent(grid, default_initial_value=False)
        for (r, c), v in initial_state.items():
            problem.set_initial_value(grid[r][c](pattern_by_symbol[v]), True)

        # --- Actions ---
        move_right = InstantaneousAction('move_right', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        r = move_right.parameter('r')
        c = move_right.parameter('c')
        move_right.add_precondition(grid[r][c](P))
        move_right.add_precondition(Not(grid[r][c + 1](P)))
        move_right.add_precondition(Not(grid[r][c + 1](B)))
        move_right.add_effect(grid[r][c + 1](P), True)
        move_right.add_effect(grid[r][c](P), False)

        move_left = InstantaneousAction('move_left', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        r = move_left.parameter('r')
        c = move_left.parameter('c')
        move_left.add_precondition(grid[r][c](P))
        move_left.add_precondition(Not(grid[r][c - 1](P)))
        move_left.add_precondition(Not(grid[r][c - 1](B)))
        move_left.add_effect(grid[r][c - 1](P), True)
        move_left.add_effect(grid[r][c](P), False)

        move_up = InstantaneousAction('move_up', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        r = move_up.parameter('r')
        c = move_up.parameter('c')
        move_up.add_precondition(grid[r][c](P))
        move_up.add_precondition(Not(grid[r - 1][c](P)))
        move_up.add_precondition(Not(grid[r - 1][c](B)))
        move_up.add_effect(grid[r - 1][c](P), True)
        move_up.add_effect(grid[r][c](P), False)

        move_down = InstantaneousAction('move_down', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        r = move_down.parameter('r')
        c = move_down.parameter('c')
        move_down.add_precondition(grid[r][c](P))
        move_down.add_precondition(Not(grid[r + 1][c](P)))
        move_down.add_precondition(Not(grid[r + 1][c](B)))
        move_down.add_effect(grid[r + 1][c](P), True)
        move_down.add_effect(grid[r][c](P), False)

        push_box_right = InstantaneousAction('push_box_right', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        r = push_box_right.parameter('r')
        c = push_box_right.parameter('c')
        push_box_right.add_precondition(grid[r][c](P))
        push_box_right.add_precondition(grid[r][c + 1](B))
        push_box_right.add_precondition(Not(grid[r][c + 2](P)))
        push_box_right.add_precondition(Not(grid[r][c + 2](B)))
        push_box_right.add_effect(grid[r][c + 1](P), True)
        push_box_right.add_effect(grid[r][c + 2](B), True)
        push_box_right.add_effect(grid[r][c](P), False)
        push_box_right.add_effect(grid[r][c + 1](B), False)

        push_box_left = InstantaneousAction('push_box_left', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        r = push_box_left.parameter('r')
        c = push_box_left.parameter('c')
        push_box_left.add_precondition(grid[r][c](P))
        push_box_left.add_precondition(grid[r][c - 1](B))
        push_box_left.add_precondition(Not(grid[r][c - 2](P)))
        push_box_left.add_precondition(Not(grid[r][c - 2](B)))
        push_box_left.add_effect(grid[r][c - 1](P), True)
        push_box_left.add_effect(grid[r][c - 2](B), True)
        push_box_left.add_effect(grid[r][c](P), False)
        push_box_left.add_effect(grid[r][c - 1](B), False)

        push_box_up = InstantaneousAction('push_box_up', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        r = push_box_up.parameter('r')
        c = push_box_up.parameter('c')
        push_box_up.add_precondition(grid[r][c](P))
        push_box_up.add_precondition(grid[r - 1][c](B))
        push_box_up.add_precondition(Not(grid[r - 2][c](P)))
        push_box_up.add_precondition(Not(grid[r - 2][c](B)))
        push_box_up.add_effect(grid[r - 1][c](P), True)
        push_box_up.add_effect(grid[r - 2][c](B), True)
        push_box_up.add_effect(grid[r][c](P), False)
        push_box_up.add_effect(grid[r - 1][c](B), False)

        push_box_down = InstantaneousAction('push_box_down', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        r = push_box_down.parameter('r')
        c = push_box_down.parameter('c')
        push_box_down.add_precondition(grid[r][c](P))
        push_box_down.add_precondition(grid[r + 1][c](B))
        push_box_down.add_precondition(Not(grid[r + 2][c](P)))
        push_box_down.add_precondition(Not(grid[r + 2][c](B)))
        push_box_down.add_effect(grid[r + 1][c](P), True)
        push_box_down.add_effect(grid[r + 2][c](B), True)
        push_box_down.add_effect(grid[r][c](P), False)
        push_box_down.add_effect(grid[r + 1][c](B), False)

        problem.add_actions([
            move_right, move_left, move_up, move_down, push_box_right, push_box_left, push_box_up, push_box_down
        ])

        # --- Goals ---
        for r, c in goal_positions:
            problem.add_goal(grid[r][c](B))

        # --- Metric ---
        costs: Dict = {
            move_right: Int(0), move_left: Int(0), move_up: Int(0), move_down: Int(0),
            push_box_right: Int(1), push_box_left: Int(1), push_box_up: Int(1), push_box_down: Int(1),
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem


DOMAIN = SokobanDomain()