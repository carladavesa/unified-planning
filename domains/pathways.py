"""pathways planning domain."""
import os
import re
from typing import Optional
from unified_planning.model import Object, MinimizeActionCosts
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

        # Parse possible_*
        possible = [s.lower() for s in re.findall(r'\(possible_(\w+)\)', content)]

        # Parse numeric values
        numeric = {}
        for m in re.finditer(r'\(=\s*\((\w+)\)\s*([\d.]+)\)', content):
            name = m.group(1).lower()
            val = float(m.group(2))
            numeric[name] = int(val) if val == int(val) else val

        # Parse goal block correctly with paren counting
        goal_idx = content.find('(:goal')
        goal_str = ""
        if goal_idx >= 0:
            i = goal_idx + len('(:goal')
            while i < len(content) and content[i] != '(':
                i += 1
            depth = 0
            start = i
            while i < len(content):
                if content[i] == '(':
                    depth += 1
                elif content[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            goal_str = content[start:i + 1]

        # Now parse all individual goals from the (and ...) block
        goals = []
        # Each goal has form: (>= (+ (+ (* (available_X) 1.0) (* (available_Y) 1.0)) -T) 0.0)
        # or single substance: (>= (+ (* (available_X) 1.0) -T) 0.0)
        for gm in re.finditer(
                r'\(>=\s*\(\+\s*\(\+\s*\(\*\s*\(available_(\w+)\)[^)]*\)\s*\(\*\s*\(available_(\w+)\)[^)]*\)\s*\)\s*-([\d.]+)',
                goal_str
        ):
            s1, s2, t = gm.group(1).lower(), gm.group(2).lower(), int(float(gm.group(3)))
            goals.append((s1, s2, t))

        # Single-substance goals: (>= (+ (* (available_X) 1.0) -T) 0.0)
        # Match these only if not already captured above
        for gm in re.finditer(
                r'\(>=\s*\(\+\s*\(\*\s*\(available_(\w+)\)[^)]*\)\s*-([\d.]+)\s*\)\s*[\d.]+',
                goal_str
        ):
            s1, t = gm.group(1).lower(), int(float(gm.group(2)))
            # Only add if not part of a 2-substance goal
            if not any(s1 in (g[0], g[1]) for g in goals if len(g) == 3):
                goals.append((s1, None, t))

        return {
            'possible': possible,
            'numeric': numeric,
            'goal_str': goal_str,
            'goals': goals,
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
        # For each substance, the maximum it can increase is the sum of all possible increments
        max_possible_avail = {}
        for s in substances:
            initial = numeric.get(f'available_{s}', 0)
            max_inc = sum(
                adata['increases'].get(s, 0)
                for adata in action_data.values()
            )
            # Calcula el threshold màxim per cada substància segons els goals reals
            max_threshold_per_subst = {}
            for goal_tuple in data['goals']:
                s1, s2, t = goal_tuple
                for s in [s1, s2]:
                    if s is not None:
                        max_threshold_per_subst[s] = max(max_threshold_per_subst.get(s, 0), t)

            max_possible_avail = {}
            for s in substances:
                initial = numeric.get(f'available_{s}', 0)

                max_inc = sum(
                    adata['increases'].get(s, 0)
                    for adata in action_data.values()
                )

                # Bound mínim per acceptar tots els operadors
                max_op_val = 0
                for adata in action_data.values():
                    if s in adata['increases']:
                        max_op_val = max(max_op_val, adata['increases'][s])
                    if s in adata['decreases']:
                        max_op_val = max(max_op_val, adata['decreases'][s])
                    if s in adata['needs']:
                        max_op_val = max(max_op_val, adata['needs'][s])

                goal_t = max_threshold_per_subst.get(s, 0)

                # Multipliquem per un factor per tenir en compte cicles
                n_goals = len(data['goals'])
                max_possible_avail[s] = max(
                    (initial + max_inc) * (n_goals + 2),  # marge per cicles
                    goal_t,
                    max_op_val,
                    1
                )

        avail = {s: Fluent(f'available_{s}', IntType(0, max_possible_avail[s])) for s in substances}
        chosen = {s: Fluent(f'chosen_{s}', BoolType()) for s in choosable}
        possible_f = {s: Fluent(f'possible_{s}', BoolType()) for s in choosable}

        for f in list(avail.values()):
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

            for subst, val in adata['needs'].items():
                if subst in avail:
                    a.add_precondition(GE(avail[subst](), Int(val)))

            if action_name.startswith('choose__'):
                s = action_name[len('choose__'):]
                if s in possible_f:
                    a.add_precondition(possible_f[s]())
            elif action_name.startswith('initialize__'):
                s = action_name[len('initialize__'):]
                if s in chosen:
                    a.add_precondition(chosen[s]())

            # Effects
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

        # Goal — itera sobre TOTS els goals parsejats
        print(f"DEBUG goals parsed: {data['goals']}")  # opcional, per debugar
        for goal_tuple in data['goals']:
            s1, s2, threshold = goal_tuple
            if s2 is not None:
                # Goal de 2 substàncies: available_s1 + available_s2 >= threshold
                if s1 in avail and s2 in avail:
                    problem.add_goal(GE(Plus(avail[s1](), avail[s2]()), Int(threshold)))
                elif s1 in avail:
                    problem.add_goal(GE(avail[s1](), Int(threshold)))
                elif s2 in avail:
                    problem.add_goal(GE(avail[s2](), Int(threshold)))
            else:
                # Goal d'1 substància: available_s1 >= threshold
                if s1 in avail:
                    problem.add_goal(GE(avail[s1](), Int(threshold)))

        costs = {}
        for action_name, adata in action_data.items():
            if adata['numsubs_inc']:
                costs[problem.action(action_name)] = Int(1)
            else:
                costs[problem.action(action_name)] = Int(0)
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem


DOMAIN = PathwaysDomain()