"""farmland planning domain."""
import os
import re
from typing import Optional
from unified_planning.shortcuts import (
    Fluent, IntType, InstantaneousAction, Problem,
    GE, Plus, Minus, Times, Int, Not, Equals,
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
        adj = data['adj']
        min_x = data['min_x']
        coeffs = data['coeffs']
        benefit_threshold = data['benefit_threshold']

        problem = Problem('farmland_problem')

        # Bounds correctes
        total_x = sum(x_init.values())
        x_ub = total_x
        cost_ub = total_x

        x = {f: Fluent(f'x_{f}', IntType(0, x_ub)) for f in farms}
        cost = Fluent('cost', IntType(0, cost_ub))

        for f, fl in x.items():
            problem.add_fluent(fl, default_initial_value=Int(0))
        problem.add_fluent(cost, default_initial_value=Int(0))

        # Initial values
        for f in farms:
            if f in x_init:
                problem.set_initial_value(x[f](), Int(x_init[f]))
        problem.set_initial_value(cost(), Int(cost_init))

        # adj as a Python set for fast lookup
        adj_set = set(adj)

        # Actions: one movefast and one moveslow per (f1, f2) adj pair
        for (f1, f2) in adj_set:
            if f1 not in x or f2 not in x:
                continue

            # movefast: >= 4 from f1, -4 f1, +2 f2, cost+1
            a_fast = InstantaneousAction(f'movefast__{f1}__{f2}')
            a_fast.add_precondition(GE(x[f1](), Int(4)))
            a_fast.add_decrease_effect(x[f1](), Int(4))
            a_fast.add_increase_effect(x[f2](), Int(2))
            a_fast.add_increase_effect(cost(), Int(1))
            problem.add_action(a_fast)

            # moveslow: >= 1 from f1, -1 f1, +1 f2
            a_slow = InstantaneousAction(f'moveslow__{f1}__{f2}')
            a_slow.add_precondition(GE(x[f1](), Int(1)))
            a_slow.add_decrease_effect(x[f1](), Int(1))
            a_slow.add_increase_effect(x[f2](), Int(1))
            problem.add_action(a_slow)

        # Goal: min x per farm
        for f, val in min_x.items():
            if f in x:
                problem.add_goal(GE(x[f](), Int(val)))

        # Goal: benefit constraint sum(coeff_i * x_i) >= threshold
        if coeffs and benefit_threshold is not None:
            terms = [Times(Int(c), x[f]()) for f, c in coeffs.items() if f in x]
            if len(terms) == 1:
                benefit_expr = terms[0]
            else:
                benefit_expr = terms[0]
                for t in terms[1:]:
                    benefit_expr = Plus(benefit_expr, t)
            problem.add_goal(GE(benefit_expr, Int(benefit_threshold)))

        return problem


DOMAIN = FarmlandDomain()