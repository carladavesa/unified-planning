"""farmland planning domain."""
import os
import re
from typing import Optional, Dict

from unified_planning.model import Action, Expression, MinimizeActionCosts
from unified_planning.shortcuts import (
    Fluent, IntType, InstantaneousAction, Problem,
    GE, Plus, Minus, Times, Int, Not, Equals, UserType,
)
from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'farmland', 'handcrafted')
INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 21)]


class FarmlandDomain(Domain):
    def __init__(self):
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()
        content = re.sub(r';.*', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Parse farm objects
        obj_m = re.search(r'\(:objects\s*(.*?)\)', content, re.DOTALL)
        farms = []
        if obj_m:
            obj_str = obj_m.group(1)
            farms = re.findall(r'(farm\d+)', obj_str)

        # Parse initial x values: (= (x farmN) VAL)
        x_init = {}
        for m in re.finditer(r'\(=\s*\(x\s+(farm\d+)\)\s*([\d.]+)\)', content):
            x_init[m.group(1)] = int(float(m.group(2)))

        # Parse cost init
        cost_m = re.search(r'\(=\s*\(cost\)\s*([\d.]+)\)', content)
        cost_init = int(float(cost_m.group(1))) if cost_m else 0

        # Parse adj predicates
        adj = []
        for m in re.finditer(r'\(adj\s+(farm\d+)\s+(farm\d+)\)', content):
            adj.append((m.group(1), m.group(2)))

        # Parse goal: collect (>= (x farmN) VAL) and the benefit constraint
        goal_str = ''
        goal_m = re.search(r'\(:goal\s*(.*)', content, re.DOTALL)
        if goal_m:
            goal_str = goal_m.group(1)

        # Min x constraints per farm: (>= (x farmN) VAL)
        min_x = {}
        for m in re.finditer(r'\(>=\s*\(x\s+(farm\d+)\)\s*([\d.]+)\)', goal_str):
            min_x[m.group(1)] = int(float(m.group(2)))

        # Benefit constraint: (>= (+ (* c0 (x farm0)) (+ (* c1 (x farm1)) ...)) THRESHOLD)
        # Parse all (* coeff (x farmN)) pairs and threshold
        coeffs = {}
        for m in re.finditer(r'\(\*\s*([\d.]+)\s*\(x\s+(farm\d+)\)\)', goal_str):
            coeffs[m.group(2)] = int(float(m.group(1)))

        benefit_threshold = None
        for segment in goal_str.split('(>='):
            if '(*' in segment and '(x ' in segment:
                nums = re.findall(r'\b(\d+)\b', segment)
                if nums:
                    benefit_threshold = int(nums[-1])
                break

        return {
            'farms': farms,
            'x_init': x_init,
            'cost_init': cost_init,
            'adj': adj,
            'min_x': min_x,
            'coeffs': coeffs,
            'benefit_threshold': benefit_threshold,
        }

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if not instance or instance not in self._instances:
            raise ValueError(f"Instance '{instance}' not found!")
        problem_path = os.path.join(PDDL_DIR, f"{instance}.pddl")
        return self._parse_pddl(problem_path)

    def build_problem(self, instance: str | None = None) -> Problem:
        data = self.get_instance(instance)
        farms = data['farms']
        x_init = data['x_init']
        cost_init = data['cost_init']
        adjacent = data['adj']
        min_x = data['min_x']
        coeffs = data['coeffs']
        benefit_threshold = data['benefit_threshold']

        problem = Problem('farmland_problem')
        Farm = UserType('Farm')

        for f in farms:
            problem.add_object(f, Farm)

        # Bounds
        total_x = sum(x_init.values())
        max_goal = max(min_x.values(), default=0)
        max_initial = max(x_init.values(), default=0)
        x_ub = max(max_goal * 2, max_initial, total_x // len(farms) * 3)
        x_ub = min(x_ub, total_x)
        cost_ub = total_x // 4

        # Fluents
        adj = Fluent('adj', f1=Farm, f2=Farm)
        x = Fluent('x', IntType(0, x_ub), f=Farm)
        cost = Fluent('cost', IntType(0, cost_ub))

        problem.add_fluent(adj, default_initial_value=False)
        problem.add_fluent(x, default_initial_value=0)
        problem.add_fluent(cost, default_initial_value=0)

          # Initial values
        for f in farms:
            if f in x_init:
                o_f = problem.object(f)
                problem.set_initial_value(x(o_f), Int(x_init[f]))
        problem.set_initial_value(cost, Int(cost_init))

        for f1, f2 in adjacent:
            o_f1 = problem.object(f1)
            o_f2 = problem.object(f2)
            problem.set_initial_value(adj(o_f1, o_f2), True)

        # Actions
        movefast = InstantaneousAction('movefast', f1=Farm, f2=Farm)
        f1, f2 = movefast.parameter('f1'), movefast.parameter('f2')
        movefast.add_precondition(Not(Equals(f1, f2)))
        movefast.add_precondition(adj(f1, f2))
        movefast.add_precondition(GE(x(f1), 4))
        movefast.add_decrease_effect(x(f1), 4)
        movefast.add_increase_effect(x(f2), 2)
        problem.add_action(movefast)

        moveslow = InstantaneousAction('moveslow', f1=Farm, f2=Farm)
        f1, f2 = moveslow.parameter('f1'), moveslow.parameter('f2')
        moveslow.add_precondition(Not(Equals(f1, f2)))
        moveslow.add_precondition(adj(f1, f2))
        moveslow.add_precondition(GE(x(f1), 1))
        moveslow.add_decrease_effect(x(f1), 1)
        moveslow.add_increase_effect(x(f2), 1)
        problem.add_action(moveslow)

        # Goals: min x per farm
        for f, val in min_x.items():
            o_f = problem.object(f)
            problem.add_goal(GE(x(o_f), val))

        # Goal: benefit constraint
        if coeffs and benefit_threshold is not None:
            terms = [Times(c, x(problem.object(f))) for f, c in coeffs.items()]
            if len(terms) == 1:
                benefit_expr = terms[0]
            else:
                benefit_expr = terms[0]
                for t in terms[1:]:
                    benefit_expr = Plus(benefit_expr, t)
            problem.add_goal(GE(benefit_expr, benefit_threshold))

        costs: Dict[Action, Expression] = {
            movefast: Int(1),
            moveslow: Int(0),
        }

        problem.add_quality_metric(MinimizeActionCosts(costs))
        return problem


DOMAIN = FarmlandDomain()