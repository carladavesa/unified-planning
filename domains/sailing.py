"""sailing planning domain.

This module implements the sailing benchmark as a `Domain`.
It is intended to be executed via `run.py`.

Example:
  python run.py --domain sailing --compilation uti --solving fast-downward
"""
import os
import re
from typing import Dict, Optional
from unified_planning.model import Action, Expression, Object
from unified_planning.shortcuts import (
    ArrayType,
    Equals,
    Fluent,
    Int,
    IntType,
    InstantaneousAction,
    MinimizeActionCosts,
    Problem, UserType, Minus, Plus, LE, GE, BoolType,
)

from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'sailing/handcrafted')

INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 21)]


class SailingDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()

        # Parse objects
        boats, persons = [], []
        objects_match = re.search(r':objects(.*?)\)', content, re.DOTALL)
        if objects_match:
            objects_str = objects_match.group(1)
            for line in objects_str.strip().split('\n'):
                line = line.strip()
                if '- boat' in line:
                    boats.extend(line.replace('- boat', '').split())
                elif '- person' in line:
                    persons.extend(line.replace('- person', '').split())

        # Parse init values
        x_vals, y_vals, d_vals = {}, {}, {}
        for match in re.finditer(r'\(=\s*\(x\s+(\w+)\)\s+(-?\d+)\)', content):
            x_vals[match.group(1)] = int(match.group(2))
        for match in re.finditer(r'\(=\s*\(y\s+(\w+)\)\s+(-?\d+)\)', content):
            y_vals[match.group(1)] = int(match.group(2))
        for match in re.finditer(r'\(=\s*\(d\s+(\w+)\)\s+(-?\d+)\)', content):
            d_vals[match.group(1)] = int(match.group(2))

        # Parse goals (saved persons)
        saved_goals = re.findall(r'\(saved\s+(\w+)\)', content)

        return {
            'boats': boats,
            'persons': persons,
            'x': x_vals,
            'y': y_vals,
            'd': d_vals,
            'saved_goals': saved_goals,
        }

    def get_instance(self, instance: Optional[str] = None) -> list[int]:
        if instance and instance in self._instances:
            path = os.path.join(PDDL_DIR, f"{instance}.pddl")
            return self._parse_pddl(path)
        raise ValueError(f"Instance '{instance}' not found!")

    def build_problem(self, instance: str | None = None) -> "Problem":
        data = self.get_instance(instance)
        problem = Problem("sailing_problem")

        Boat = UserType('Boat')
        Person = UserType('Person')

        # Objects
        boats = [Object(name, Boat) for name in data['boats']]
        persons = [Object(name, Person) for name in data['persons']]
        problem.add_objects(boats + persons)

        # Compute bounds for int types
        all_x = list(data['x'].values())
        all_y = list(data['y'].values())
        all_d = list(data['d'].values())

        d_min = min(data['d'].values()) if data['d'] else 0
        d_max = max(data['d'].values()) if data['d'] else 0

        x_lb = min(all_x) - abs(d_min) // 2
        x_ub = max(all_x) + abs(d_min) // 2
        y_lb = d_min - 50
        y_ub = max(all_y) + 50
        d_lb, d_ub = d_min, d_max

        saved = Fluent('saved', BoolType(), person=Person)
        x = Fluent('x', IntType(x_lb, x_ub), boat=Boat)
        y = Fluent('y', IntType(y_lb, y_ub), boat=Boat)
        d = Fluent('d', IntType(d_lb, d_ub), person=Person)

        problem.add_fluent(saved, default_initial_value=False)
        problem.add_fluent(x)
        problem.add_fluent(y)
        problem.add_fluent(d)

        # Initial values
        boat_objs = {o.name: o for o in boats}
        person_objs = {o.name: o for o in persons}

        for bname, val in data['x'].items():
            problem.set_initial_value(x(boat_objs[bname]), val)
        for bname, val in data['y'].items():
            problem.set_initial_value(y(boat_objs[bname]), val)
        for pname, val in data['d'].items():
            problem.set_initial_value(d(person_objs[pname]), val)

        # Actions
        gonortheast = InstantaneousAction('gonortheast', b=Boat)
        b = gonortheast.parameter('b')
        gonortheast.add_increase_effect(x(b), 3)
        gonortheast.add_increase_effect(y(b), 3)

        gonorthwest = InstantaneousAction('gonorthwest', b=Boat)
        b = gonorthwest.parameter('b')
        gonorthwest.add_decrease_effect(x(b), 3)
        gonorthwest.add_increase_effect(y(b), 3)

        goest = InstantaneousAction('goest', b=Boat)
        b = goest.parameter('b')
        goest.add_increase_effect(x(b), 6)

        gowest = InstantaneousAction('gowest', b=Boat)
        b = gowest.parameter('b')
        gowest.add_decrease_effect(x(b), 6)

        gosouthwest = InstantaneousAction('gosouthwest', b=Boat)
        b = gosouthwest.parameter('b')
        gosouthwest.add_increase_effect(x(b), 4)
        gosouthwest.add_decrease_effect(y(b), 4)

        gosoutheast = InstantaneousAction('gosoutheast', b=Boat)
        b = gosoutheast.parameter('b')
        gosoutheast.add_decrease_effect(x(b), 4)
        gosoutheast.add_decrease_effect(y(b), 4)

        gosouth = InstantaneousAction('gosouth', b=Boat)
        b = gosouth.parameter('b')
        gosouth.add_decrease_effect(y(b), 4)

        saveperson = InstantaneousAction('saveperson', b=Boat, t=Person)
        b = saveperson.parameter('b')
        t = saveperson.parameter('t')
        saveperson.add_precondition(GE(Plus(x(b), y(b)), d(t)))
        saveperson.add_precondition(GE(Minus(y(b), x(b)), d(t)))
        saveperson.add_precondition(LE(Plus(x(b), y(b)), Plus(d(t), 50)))
        saveperson.add_precondition(LE(Minus(y(b), x(b)), Plus(d(t), 50)))
        saveperson.add_effect(saved(t), True)

        problem.add_actions([gonortheast, gonorthwest, goest, gowest, gosouthwest, gosoutheast, gosouth, saveperson])

        # Goals
        for pname in data['saved_goals']:
            problem.add_goal(saved(person_objs[pname]))

        problem.add_quality_metric(MinimizeActionCosts({
            gonortheast: Int(1), gonorthwest: Int(1),
            goest: Int(1), gowest: Int(1),
            gosouthwest: Int(1), gosoutheast: Int(1),
            gosouth: Int(1), saveperson: Int(0),
        }))

        return problem


DOMAIN = SailingDomain()