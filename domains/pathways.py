"""pathways planning domain."""
import os
import re
from typing import Optional
from unified_planning.model import Object
from unified_planning.shortcuts import (
    Fluent, IntType, BoolType, UserType,
    InstantaneousAction, Problem,
    GE, LE, And, Plus, Minus, Times, Int,
)
from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'pathways', 'handcrafted')
INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 21)]


class PathwaysDomain(Domain):
    def __init__(self):
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()
        content = re.sub(r';.*', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Parse possible_* predicates from init
        possible = [s.lower() for s in re.findall(r'\(possible_(\w+)\)', content)]

        # Parse numeric values
        numeric = {}
        for m in re.finditer(r'\(=\s*\((\w+)\)\s*([\d.]+)\)', content):
            name = m.group(1).lower()
            val = float(m.group(2))
            numeric[name] = int(val) if val == int(val) else val

        # Parse goal string
        goal_match = re.search(r'\(:goal\s*(.*?)\)\s*\)', content, re.DOTALL)
        goal_str = goal_match.group(1).strip() if goal_match else ""

        return {
            'possible': possible,
            'numeric': numeric,
            'goal_str': goal_str,
        }

    def _parse_domain(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()
        content = re.sub(r';.*', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Extreu totes les substàncies que apareixen com available_*
        substances = set()
        for m in re.finditer(r'\(available_(\w+)\)', content):
            substances.add(m.group(1).lower())

        choosable = set()
        for m in re.finditer(r'\(chosen_(\w+)\)', content):
            choosable.add(m.group(1).lower())
        for m in re.finditer(r'\(possible_(\w+)\)', content):
            choosable.add(m.group(1).lower())

        actions = {}

        i = 0
        while True:
            idx = content.find('(:action', i)
            if idx == -1:
                break
            depth = 0
            start = idx
            for j in range(idx, len(content)):
                if content[j] == '(':
                    depth += 1
                elif content[j] == ')':
                    depth -= 1
                    if depth == 0:
                        block = content[start:j + 1]
                        break
            i = j + 1

            name_m = re.match(r'\(:action\s+(\S+)', block)
            if not name_m:
                continue
            name = name_m.group(1).lower()

            prec_idx = block.find(':precondition')
            eff_idx = block.find(':effect')
            if prec_idx == -1 or eff_idx == -1:
                actions[name] = {
                    'needs': {}, 'increases': {}, 'decreases': {},
                    'numsubs_inc': 0, 'chosen_true': [], 'possible_false': []
                }
                continue

            prec_str = block[prec_idx:eff_idx]
            eff_str = block[eff_idx:]

            needs = {}
            for pm in re.finditer(
                r'\(>=\s*\(\+\s*\(\*\s+\(available_(\w+)\)\s*[\d.]+\)\s*-([\d.]+)\s*\)\s*[\d.]+\)',
                prec_str
            ):
                subst = pm.group(1).lower()
                val = int(float(pm.group(2)))
                needs[subst] = val

            # Parse effects
            increases = {}
            decreases = {}
            for em in re.finditer(r'\(increase\s+\(available_(\w+)\)\s*([\d.]+)\)', eff_str):
                increases[em.group(1).lower()] = int(float(em.group(2)))
            for em in re.finditer(r'\(decrease\s+\(available_(\w+)\)\s*([\d.]+)\)', eff_str):
                decreases[em.group(1).lower()] = int(float(em.group(2)))

            numsubs_inc = 1 if re.search(r'\(increase\s+\(numsubs\)', eff_str) else 0
            chosen_true = [c.lower() for c in re.findall(r'\(chosen_(\w+)\)', eff_str.split('(not')[0])]
            possible_false = [p.lower() for p in re.findall(r'\(not\s*\(possible_(\w+)\)\)', eff_str)]

            actions[name] = {
                'needs': needs,
                'increases': increases,
                'decreases': decreases,
                'numsubs_inc': numsubs_inc,
                'chosen_true': chosen_true,
                'possible_false': possible_false,
            }

        return {
            'actions': actions,
            'substances': list(substances),
            'choosable': list(choosable),
        }

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if not instance or instance not in self._instances:
            raise ValueError(f"Instance '{instance}' not found!")
        domain_path = os.path.join(PDDL_DIR, f"domain_{instance}.pddl")
        problem_path = os.path.join(PDDL_DIR, f"{instance}.pddl")
        data = self._parse_pddl(problem_path)
        domain_data = self._parse_domain(domain_path)
        data['actions'] = domain_data['actions']
        data['substances'] = domain_data['substances']
        data['choosable'] = domain_data['choosable']
        return data

    def build_problem(self, instance: str | None = None) -> Problem:
        data = self.get_instance(instance)
        numeric = data['numeric']
        possible = data['possible']
        substances = data['substances']
        choosable = data['choosable']
        action_data = data['actions']

        problem = Problem('pathways_problem')

        # Bounds
        numsubs_ub = len(choosable)
        avail_ub = numsubs_ub

        # Fluents
        avail = {s: Fluent(f'available_{s}', IntType(0, avail_ub)) for s in substances}
        numsubs = Fluent('numsubs', IntType(0, numsubs_ub))
        chosen = {s: Fluent(f'chosen_{s}', BoolType()) for s in choosable}
        possible_f = {s: Fluent(f'possible_{s}', BoolType()) for s in choosable}

        for f in list(avail.values()) + [numsubs]:
            problem.add_fluent(f, default_initial_value=Int(0))
        for f in list(chosen.values()) + list(possible_f.values()):
            problem.add_fluent(f, default_initial_value=False)

        # Initial values: possible_*
        for s in choosable:
            if s in possible:
                problem.set_initial_value(possible_f[s](), True)

        # Initial values: available_*
        for name, val in numeric.items():
            if name.startswith('available_'):
                s = name[len('available_'):]
                if s in avail:
                    problem.set_initial_value(avail[s](), int(val))

        # Actions

        for action_name, adata in action_data.items():
            a = InstantaneousAction(action_name)

            # Precondicions numèriques
            for subst, val in adata['needs'].items():
                if subst in avail:
                    a.add_precondition(GE(avail[subst](), Int(val)))

            # Precondicions booleanes
            if action_name.startswith('choose__'):
                s = action_name[len('choose__'):]
                if s in possible_f:
                    a.add_precondition(possible_f[s]())
            elif action_name.startswith('initialize__'):
                s = action_name[len('initialize__'):]
                if s in chosen:
                    a.add_precondition(chosen[s]())

            # Efectes
            if adata['numsubs_inc']:
                a.add_increase_effect(numsubs(), 1)
            for s in adata['chosen_true']:
                if s in chosen:
                    a.add_effect(chosen[s](), True)
            for s in adata['possible_false']:
                if s in possible_f:
                    a.add_effect(possible_f[s](), False)
            for subst, val in adata['increases'].items():
                if subst in avail:
                    a.add_increase_effect(avail[subst](), Int(val))
            for subst, val in adata['decreases'].items():
                if subst in avail:
                    a.add_decrease_effect(avail[subst](), Int(val))

            problem.add_action(a)

        # Goal
        m = re.search(
            r'\(\+\s*\(\*\s*\(available_(\w+)\)[^)]*\)\s*\(\*\s*\(available_(\w+)\)',
            data['goal_str']
        )
        if m:
            s1, s2 = m.group(1).lower(), m.group(2).lower()
            t = re.search(r'-([\d.]+)\s*\)\s*[\d.]+\s*\)', data['goal_str'])
            threshold = int(float(t.group(1))) if t else 4
            if s1 in avail and s2 in avail:
                problem.add_goal(GE(Plus(avail[s1](), avail[s2]()), Int(threshold)))
            elif s1 in avail:
                problem.add_goal(GE(avail[s1](), Int(threshold)))
        else:
            goal_substs = [s.lower() for s in re.findall(r'\(available_(\w+)\)', data['goal_str'])]
            t = re.search(r'-([\d.]+)', data['goal_str'])
            threshold = int(float(t.group(1))) if t else 4
            if len(goal_substs) >= 2 and goal_substs[0] in avail and goal_substs[1] in avail:
                problem.add_goal(GE(Plus(avail[goal_substs[0]](), avail[goal_substs[1]]()), Int(threshold)))
            elif len(goal_substs) == 1 and goal_substs[0] in avail:
                problem.add_goal(GE(avail[goal_substs[0]](), Int(threshold)))

        return problem


DOMAIN = PathwaysDomain()