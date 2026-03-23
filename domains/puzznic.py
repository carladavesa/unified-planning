"""puzznic planning domain.

Example:
  python run.py --domain puzznic --compilation up --solving fast-downward
"""

import os
from typing import Dict, Optional
from unified_planning.model import Action, Expression, Object, RangeVariable, Variable, Axiom
from unified_planning.shortcuts import (
    ArrayType,
    Equals,
    Fluent,
    Int,
    IntType,
    InstantaneousAction,
    MinimizeActionCosts,
    Problem, UserType, Exists, And, Not, Or, DerivedBoolType,
)

from domains.base import Domain

PROBS_DIR = os.path.join(os.path.dirname(__file__), 'puzznic/probs')

INSTANCES: list[str] = [f"puzznic{i}" for i in range(1, 21)]


class PuzznicDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict[str, list]]:
        return {k: {} for k in self._instances}

    def _parse_instance(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            inlines = f.readlines()

        lines = []
        for l in inlines:
            if l == '\n':
                break
            lines.append(l)

        matrix = [list(line.rstrip('\n')) for line in lines]

        initial_state = {}
        undefined = []
        row_index = 0
        for row in matrix[1:-1]:
            if row[0] == '#' and row[1] != '#':
                first_wall_index = -1
            else:
                first_wall_index = row.index('#') - 1
            if row[-1] == '#' and row[-2] != '#':
                last_wall_index = len(row[1:-1])
            else:
                last_wall_index = len(row) - 1 - row[::-1].index('#')
            for col_index, cell in enumerate(row[1:-1]):
                if col_index <= first_wall_index or col_index >= last_wall_index or cell == '#':
                    undefined.append((row_index, col_index))
                elif cell != ' ':
                    initial_state[(row_index, col_index)] = cell
            row_index += 1

        num_rows = row_index
        num_cols = len(matrix[0]) - 2

        return {
            'initial_state': initial_state,
            'undefined': undefined,
            'rows': num_rows,
            'columns': num_cols,
        }

    def get_instance(self, instance: Optional[str] = None) -> list[int]:
        if instance and instance in self._instances:
            path = os.path.join(PROBS_DIR, f"{instance}.prob")
            return self._parse_instance(path)
        raise ValueError(f"Instance '{instance}' not found!")



    def build_problem(self, instance: str | None = None) -> "Problem":
        data = self.get_instance(instance)

        initial_state = data['initial_state']
        undefined = data['undefined']
        rows = data['rows']
        columns = data['columns']

        # --- Problem ---
        problem = Problem('puzznic_problem')

        Pattern = UserType('Pattern')
        F = Object('F', Pattern)
        B = Object('B', Pattern)
        Y = Object('Y', Pattern)
        G = Object('G', Pattern)
        R = Object('R', Pattern)
        L = Object('L', Pattern)
        O = Object('O', Pattern)
        V = Object('V', Pattern)
        P = Object('P', Pattern)
        C = Object('C', Pattern)
        pattern_by_symbol = {'F': F, 'B': B, 'Y': Y, 'G': G, 'R': R,
                             'L': L, 'O': O, 'V': V, 'P': P, 'C': C}
        problem.add_object(F)

        patterned = Fluent('patterned', ArrayType(rows, ArrayType(columns)),
                           p=Pattern, undefined_positions=undefined)
        problem.add_fluent(patterned, default_initial_value=False)

        for (r, c), p in initial_state.items():
            pattern_obj = pattern_by_symbol[p]
            if not problem.has_object(p):
                problem.add_object(pattern_obj)
            problem.set_initial_value(patterned[r][c](pattern_obj), True)

        for r in range(rows):
            for c in range(columns):
                if (r, c) not in initial_state and (r, c) not in undefined:
                    problem.set_initial_value(patterned[r][c](F), True)

        falling_flag = Fluent('falling_flag', DerivedBoolType())
        matching_flag = Fluent('matching_flag', DerivedBoolType())
        problem.add_fluent(falling_flag, default_initial_value=False)
        problem.add_fluent(matching_flag, default_initial_value=False)

        # --- Axioms ---
        axiom_falling = Axiom('axiom_falling')
        axiom_falling.set_head(falling_flag)
        i = RangeVariable('i', 1, rows - 1)
        j = RangeVariable('j', 0, columns - 1)
        axiom_falling.add_body_condition(
            Exists(And(Not(patterned[i - 1][j](F)), patterned[i][j](F)), i, j)
        )
        problem.add_axiom(axiom_falling)

        axiom_matching = Axiom('axiom_matching')
        axiom_matching.set_head(matching_flag)
        i = RangeVariable('i', 0, rows - 1)
        j = RangeVariable('j', 0, columns - 2)
        p = Variable('p', Pattern)
        matching_horizontal = Exists(
            And(patterned[i][j](p), patterned[i][j + 1](p), Not(Equals(p, F))), i, j, p
        )
        i = RangeVariable('i', 0, rows - 2)
        j = RangeVariable('j', 0, columns - 1)
        p = Variable('p', Pattern)
        matching_vertical = Exists(
            And(patterned[i][j](p), patterned[i + 1][j](p), Not(patterned[i][j](F))), i, j, p
        )
        axiom_matching.add_body_condition(Or(matching_horizontal, matching_vertical))
        problem.add_axiom(axiom_matching)

        # --- Actions ---
        move_block_right = InstantaneousAction('move_block_right', p=Pattern,
                                               r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        p, r, c = move_block_right.parameter('p'), move_block_right.parameter('r'), move_block_right.parameter('c')
        move_block_right.add_precondition(Not(falling_flag))
        move_block_right.add_precondition(Not(matching_flag))
        move_block_right.add_precondition(patterned[r][c](p))
        move_block_right.add_precondition(Not(Equals(p, F)))
        move_block_right.add_precondition(patterned[r][c + 1](F))
        move_block_right.add_effect(patterned[r][c](F), True)
        move_block_right.add_effect(patterned[r][c + 1](p), True)
        move_block_right.add_effect(patterned[r][c](p), False)
        move_block_right.add_effect(patterned[r][c + 1](F), Or(patterned[r][c](p), patterned[r][c + 1](p)))

        move_block_left = InstantaneousAction('move_block_left', p=Pattern,
                                              r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        p, r, c = move_block_left.parameter('p'), move_block_left.parameter('r'), move_block_left.parameter('c')
        move_block_left.add_precondition(Not(falling_flag))
        move_block_left.add_precondition(Not(matching_flag))
        move_block_left.add_precondition(patterned[r][c](p))
        move_block_left.add_precondition(Not(Equals(p, F)))
        move_block_left.add_precondition(patterned[r][c - 1](F))
        move_block_left.add_effect(patterned[r][c](F), True)
        move_block_left.add_effect(patterned[r][c - 1](p), True)
        move_block_left.add_effect(patterned[r][c](p), False)
        move_block_left.add_effect(patterned[r][c - 1](F), False)

        fall_block = InstantaneousAction('fall_block', p=Pattern,
                                         r=IntType(0, rows - 2), c=IntType(0, columns - 1))
        p, r, c = fall_block.parameter('p'), fall_block.parameter('r'), fall_block.parameter('c')
        fall_block.add_precondition(falling_flag)
        fall_block.add_precondition(patterned[r][c](p))
        fall_block.add_precondition(Not(Equals(p, F)))
        fall_block.add_precondition(patterned[r + 1][c](F))
        fall_block.add_effect(patterned[r][c](F), True)
        fall_block.add_effect(patterned[r + 1][c](p), True)
        fall_block.add_effect(patterned[r][c](p), False)
        fall_block.add_effect(patterned[r + 1][c](F), False)

        matching_blocks = InstantaneousAction('matching_blocks')
        matching_blocks.add_precondition(Not(falling_flag))
        matching_blocks.add_precondition(matching_flag)
        i = RangeVariable('i', 0, rows - 1)
        j = RangeVariable('j', 0, columns - 1)
        p = Variable('p', Pattern)
        matching_blocks.add_effect(patterned[i][j](F), True, condition=And(
            Not(Equals(p, F)),
            patterned[i][j](p),
            Or(patterned[i + 1][j](p), patterned[i - 1][j](p),
               patterned[i][j + 1](p), patterned[i][j - 1](p))
        ), forall=[i, j, p])

        problem.add_actions([move_block_right, move_block_left, fall_block, matching_blocks])

        # --- Goals ---
        for i in range(rows):
            for j in range(columns):
                if (i, j) not in undefined:
                    problem.add_goal(patterned[i][j](F))

        # --- Metric ---
        costs: Dict = {
            move_block_right: Int(1),
            move_block_left: Int(1),
            matching_blocks: Int(0),
            fall_block: Int(0),
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem


DOMAIN = PuzznicDomain()