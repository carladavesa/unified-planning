"""mystery prime typed planning domain."""
import os
import re
from typing import Optional
from unified_planning.model import Object
from unified_planning.shortcuts import (
    Fluent, IntType, InstantaneousAction,
    Problem, UserType, BoolType, GE, Plus, Times,
)
from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'mystery_prime/handcrafted')
INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 31)]


class MysteryPrimeDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()

        locale = {}
        harmony = {}
        for match in re.finditer(r'\(=\s*\(locale_(\w+)\)\s+([\d.]+)\s*\)', content):
            locale[match.group(1)] = int(float(match.group(2)))
        for match in re.finditer(r'\(=\s*\(harmony_(\w+)\)\s+([\d.]+)\s*\)', content):
            harmony[match.group(1)] = int(float(match.group(2)))

        init_section = re.search(r':init(.*?):goal', content, re.DOTALL)
        init_str = init_section.group(1) if init_section else ''

        craves = []
        for match in re.finditer(r'\(craves_(\w+)_(\w+)\)', init_str):
            craves.append((match.group(1), match.group(2)))

        fears = []
        for match in re.finditer(r'\(fears_(\w+)_(\w+)\)', init_str):
            fears.append((match.group(1), match.group(2)))

        goal_craves = []
        goal_section = re.search(r':goal\s*(.*?)(?=\s*\)?\s*$)', content, re.DOTALL)
        if goal_section:
            for match in re.finditer(r'\(craves_(\w+)_(\w+)\)', goal_section.group(1)):
                goal_craves.append((match.group(1), match.group(2)))

        return {
            'locale': locale,
            'harmony': harmony,
            'craves': craves,
            'fears': fears,
            'goal_craves': goal_craves,
        }

    def _parse_domain(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()

        # Foods from drink actions (més complet que functions)
        food_names = []
        for match in re.finditer(r':action drink__(\w+)__\w+', content):
            name = match.group(1)
            if name not in food_names:
                food_names.append(name)

        # Persons from overcome/succumb actions
        person_names = []
        for match in re.finditer(r':action overcome__(\w+)__(\w+)__\w+', content):
            for name in [match.group(1), match.group(2)]:
                if name not in person_names:
                    person_names.append(name)
        for match in re.finditer(r':action succumb__(\w+)__(\w+)__\w+', content):
            for name in [match.group(1), match.group(2)]:
                if name not in person_names:
                    person_names.append(name)
        # Also from feast
        for match in re.finditer(r':action feast__(\w+)__\w+__\w+', content):
            name = match.group(1)
            if name not in person_names and name not in food_names:
                person_names.append(name)

        drink_combos = []
        for match in re.finditer(r':action drink__(\w+)__(\w+)', content):
            f1, f2 = match.group(1), match.group(2)
            if f1 != f2:
                drink_combos.append((f1, f2))

        feast_combos = []
        for match in re.finditer(r':action feast__(\w+)__(\w+)__(\w+)', content):
            p, f1, f2 = match.group(1), match.group(2), match.group(3)
            if f1 != f2:
                feast_combos.append((p, f1, f2))

        overcome_combos = []
        for match in re.finditer(r':action overcome__(\w+)__(\w+)__(\w+)', content):
            overcome_combos.append((match.group(1), match.group(2), match.group(3)))

        succumb_combos = []
        for match in re.finditer(r':action succumb__(\w+)__(\w+)__(\w+)', content):
            succumb_combos.append((match.group(1), match.group(2), match.group(3)))

        return {
            'food_names': food_names,
            'person_names': person_names,
            'drink': drink_combos,
            'feast': feast_combos,
            'overcome': overcome_combos,
            'succumb': succumb_combos,
        }

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if instance and instance in self._instances:
            path = os.path.join(PDDL_DIR, f"{instance}.pddl")
            return self._parse_pddl(path)
        raise ValueError(f"Instance '{instance}' not found!")

    def get_domain(self, instance: str) -> dict:
        path = os.path.join(PDDL_DIR, f"domain_{instance}.pddl")
        return self._parse_domain(path)

    def build_problem(self, instance: str | None = None) -> "Problem":
        data = self.get_instance(instance)
        domain_data = self.get_domain(instance)

        problem = Problem("mprime_problem")

        Food = UserType('Food')
        Person = UserType('Person')

        food_names = domain_data['food_names']
        person_names = domain_data['person_names']

        foods = [Object(name, Food) for name in food_names]
        persons = [Object(name, Person) for name in person_names]
        problem.add_objects(foods + persons)

        food_objs = {o.name: o for o in foods}
        person_objs = {o.name: o for o in persons}

        max_locale = max(data['locale'].values(), default=10)
        max_harmony = max(data['harmony'].values(), default=10)

        locale = Fluent('locale', IntType(0, max_locale + 10), food=Food)
        harmony = Fluent('harmony', IntType(0, max_harmony + 10), person=Person)
        craves = Fluent('craves', BoolType(), person=Person, food=Food)
        fears = Fluent('fears', BoolType(), person=Person, other=Person)

        problem.add_fluent(locale, default_initial_value=0)
        problem.add_fluent(harmony, default_initial_value=0)
        problem.add_fluent(craves, default_initial_value=False)
        problem.add_fluent(fears, default_initial_value=False)

        for fname, val in data['locale'].items():
            problem.set_initial_value(locale(food_objs[fname]), val)
        for pname, val in data['harmony'].items():
            problem.set_initial_value(harmony(person_objs[pname]), val)
        for (p, f) in data['craves']:
            problem.set_initial_value(craves(person_objs[p], food_objs[f]), True)
        for (p1, p2) in data['fears']:
            problem.set_initial_value(fears(person_objs[p1], person_objs[p2]), True)

        # drink actions
        for (f1n, f2n) in domain_data['drink']:
            if f1n == f2n:
                continue
            a = InstantaneousAction(f'drink_{f1n}_{f2n}')
            f1o = food_objs[f1n]
            f2o = food_objs[f2n]
            a.add_precondition(GE(Plus(Times(locale(f1o), 1), 1), 0))
            a.add_decrease_effect(locale(f1o), 1)
            a.add_increase_effect(locale(f2o), 1)
            problem.add_action(a)

        # feast actions
        for (pn, f1n, f2n) in domain_data['feast']:
            if f1n == f2n:
                continue
            a = InstantaneousAction(f'feast_{pn}_{f1n}_{f2n}')
            po = person_objs[pn]
            f1o = food_objs[f1n]
            f2o = food_objs[f2n]
            a.add_precondition(craves(po, f1o))
            a.add_precondition(GE(Plus(Times(locale(f1o), 1), 1), 0))
            a.add_decrease_effect(locale(f1o), 1)
            a.add_effect(craves(po, f1o), False)
            a.add_effect(craves(po, f2o), True)
            problem.add_action(a)

        # overcome actions
        for (dn, pn, fn) in domain_data['overcome']:
            a = InstantaneousAction(f'overcome_{dn}_{pn}_{fn}')
            do = person_objs[dn]
            po = person_objs[pn]
            fo = food_objs[fn]
            a.add_precondition(GE(Plus(Times(harmony(po), 1), 1), 0))
            a.add_precondition(craves(do, fo))
            a.add_precondition(craves(po, fo))
            a.add_decrease_effect(harmony(po), 1)
            a.add_effect(craves(do, fo), False)
            a.add_effect(fears(do, po), True)
            problem.add_action(a)

        # succumb actions
        for (dn, pn, fn) in domain_data['succumb']:
            a = InstantaneousAction(f'succumb_{dn}_{pn}_{fn}')
            do = person_objs[dn]
            po = person_objs[pn]
            fo = food_objs[fn]
            a.add_precondition(craves(po, fo))
            a.add_precondition(fears(do, po))
            a.add_increase_effect(harmony(po), 1)
            a.add_effect(fears(do, po), False)
            a.add_effect(craves(do, fo), True)
            problem.add_action(a)

        for (p, f) in data['goal_craves']:
            problem.add_goal(craves(person_objs[p], food_objs[f]))

        return problem


DOMAIN = MysteryPrimeDomain()