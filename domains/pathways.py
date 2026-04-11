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

        # Parse goal
        goal_match = re.search(r'\(:goal\s*(.*?)\)\s*\)', content, re.DOTALL)
        goal_str = goal_match.group(1).strip() if goal_match else ""

        print(f"possible raw: {re.findall(r'possible_(\w+)', content[:500])}")

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

        actions = {}

        # Troba cada (:action ...) extraient el bloc complet comptant parèntesis
        i = 0
        while True:
            idx = content.find('(:action', i)
            if idx == -1:
                break
            # Extreu el bloc complet
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

            # Nom de l'acció
            name_m = re.match(r'\(:action\s+(\S+)', block)
            if not name_m:
                continue
            name = name_m.group(1).lower()

            # Separa precondition i effect
            prec_idx = block.find(':precondition')
            eff_idx = block.find(':effect')
            if prec_idx == -1 or eff_idx == -1:
                actions[name] = {'needs': {}, 'increases': {}, 'decreases': {},
                                 'numsubs_inc': 0, 'chosen_true': [], 'possible_false': []}
                continue

            prec_str = block[prec_idx:eff_idx]
            eff_str = block[eff_idx:]

            # Parse preconditions numèriques: (>= (+ (* (available_X) 1.0) -N.0) 0.0)
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
            chosen_true = re.findall(r'\(chosen_(\w+)\)', eff_str.split('(not')[0])
            possible_false = re.findall(r'\(not\s*\(possible_(\w+)\)\)', eff_str)

            actions[name] = {
                'needs': needs,
                'increases': increases,
                'decreases': decreases,
                'numsubs_inc': numsubs_inc,
                'chosen_true': [c.lower() for c in chosen_true],
                'possible_false': [p.lower() for p in possible_false],
            }


        return actions

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if not instance or instance not in self._instances:
            raise ValueError(f"Instance '{instance}' not found!")
        domain_path = os.path.join(PDDL_DIR, f"domain_{instance}.pddl")
        problem_path = os.path.join(PDDL_DIR, f"{instance}.pddl")
        data = self._parse_pddl(problem_path)
        data['actions'] = self._parse_domain(domain_path)
        return data

    def build_problem(self, instance: str | None = None) -> Problem:
        data = self.get_instance(instance)
        numeric = data['numeric']
        possible = data['possible']

        problem = Problem('pathways_problem')

        # Compute bounds from instance values
        max_val = max((v for v in numeric.values() if isinstance(v, int) and v > 0), default=10)
        avail_ub = max_val * 5  # conservative upper bound

        # Fluents: available_* as integers
        substances = [
            'sp1', 'raf1', 'prbp2', 'prbe2f4p1dp12', 'pcaf', 'p300', 'p16',
            'p130e2f5p1dp12', 'e2f13', 'dmp1', 'chk1', 'cdk7', 'cdk46p3cycdp1',
            'cdk46p3cycd', 'cdc25c', 'ap2',
            'dmp1p1', 'cdc25cp2', 'p16cdk7', 'pcafp300', 'prbp1p2ap2',
            'prbp2ap2', 'prbp1p2', 'raf1p130e2f5p1dp12', 'raf1prbe2f4p1dp12', 'sp1e2f13'
        ]
        choosable = [
            'sp1', 'raf1', 'prbp2', 'prbe2f4p1dp12', 'pcaf', 'p300', 'p16',
            'p130e2f5p1dp12', 'e2f13', 'dmp1', 'chk1', 'cdk7', 'cdk46p3cycdp1',
            'cdk46p3cycd', 'cdc25c', 'ap2'
        ]

        avail = {s: Fluent(f'available_{s}', IntType(0, avail_ub)) for s in substances}
        numsubs = Fluent('numsubs', IntType(0, len(choosable)))
        chosen = {s: Fluent(f'chosen_{s}', BoolType()) for s in choosable}
        possible_f = {s: Fluent(f'possible_{s}', BoolType()) for s in choosable}

        for f in list(avail.values()) + [numsubs]:
            problem.add_fluent(f, default_initial_value=Int(0))
        for f in list(chosen.values()) + list(possible_f.values()):
            problem.add_fluent(f, default_initial_value=False)

        # Initial values
        for s in choosable:
            if s in possible:
                problem.set_initial_value(possible_f[s](), True)

        for name, val in numeric.items():
            if name.startswith('available_'):
                s = name[len('available_'):]
                if s in avail:
                    problem.set_initial_value(avail[s](), val)

        # Helper to get numeric constant
        def nc(name):
            return Int(int(numeric.get(name, 0)))

        # ===== Actions =====
        action_data = data['actions']

        for action_name, adata in action_data.items():
            a = InstantaneousAction(action_name)
            print(
                f"Action {action_name}: needs keys = {list(adata['needs'].keys())}, avail keys sample = {list(avail.keys())[:3]}")
            # Preconditions
            for subst, val in adata['needs'].items():
                if subst in avail:
                    a.add_precondition(GE(avail[subst](), Int(val)))

            # Bool preconditions (chosen_*)
            # Check if action name starts with initialize__ or choose__
            if action_name.startswith('choose__'):
                s = action_name[len('choose__'):]
                if s in possible_f:
                    a.add_precondition(possible_f[s]())
            elif action_name.startswith('initialize__'):
                s = action_name[len('initialize__'):]
                if s in chosen:
                    a.add_precondition(chosen[s]())

            # Effects
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

        # ===== Goal =====
        # Goal: available_prbp1p2ap2 + available_pcafp300 >= 4
        goal_threshold = int(numeric.get('goal_threshold', 4))
        # Parse from goal_str if possible
        m = re.search(r'\(\+\s*\(\*\s*\(available_(\w+)\)[^)]*\)\s*\(\*\s*\(available_(\w+)\)', data['goal_str'])
        if m:
            s1, s2 = m.group(1).lower(), m.group(2).lower()
            # extract threshold
            t = re.search(r'-(\d+)', data['goal_str'])
            threshold = int(t.group(1)) if t else 4
            problem.add_goal(GE(Plus(avail[s1](), avail[s2]()), threshold))
        else:
            problem.add_goal(GE(Plus(avail['prbp1p2ap2'](), avail['pcafp300']()), 4))

        return problem


DOMAIN = PathwaysDomain()