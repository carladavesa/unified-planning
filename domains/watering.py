"""watering planning domain.

This module implements the watering benchmark as a `Domain`.
It is intended to be executed via `run.py`.

Example:
  python run.py --domain watering --compilation uti --solving fast-downward
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
    Problem, UserType, Minus, Plus, LE, GE,
)

from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'watering/handcrafted')

INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 21)]


class WateringDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()

        content = re.sub(r';.*', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Objects
        objects = {'plant': [], 'tap': [], 'agent': []}
        obj_match = re.search(r'\(:objects(.*?)\)', content, re.IGNORECASE)
        if obj_match:
            obj_str = obj_match.group(1).strip()
            segments = re.split(r'-\s*(\w+)', obj_str)
            i = 0
            while i + 1 < len(segments):
                names = segments[i].split()
                type_name = segments[i + 1].strip().lower()
                if type_name in objects:
                    objects[type_name].extend(names)
                i += 2

        # Init
        init_match = re.search(r'\(:init(.*?)\)\s*\(:goal', content, re.IGNORECASE | re.DOTALL)
        init_str = init_match.group(1).strip() if init_match else ""

        numeric_global = {}
        numeric_obj = {}
        for m in re.finditer(r'\(=\s*\((\w+)\s*(\w*)\)\s*([\d.]+)\)', init_str):
            fluent = m.group(1).lower()
            arg = m.group(2).strip()
            value = int(m.group(3)) if '.' not in m.group(3) else float(m.group(3))
            if arg:
                numeric_obj[(fluent, arg)] = value
            else:
                numeric_global[fluent] = value

        # Goal
        goal_match = re.search(r'\(:goal\s*\(and(.*)', content, re.IGNORECASE | re.DOTALL)
        goal_str = goal_match.group(1) if goal_match else ""

        poured_goals = {}
        for m in re.finditer(r'\(=\s*\(poured\s+(\w+)\)\s*([\d]+)\)', goal_str):
            poured_goals[m.group(1)] = int(m.group(2))

        balance_goal = bool(re.search(r'\(=\s*\(total_poured\)\s*\(total_loaded\)\)', goal_str))

        return {
            'objects': objects,
            'numeric_global': numeric_global,
            'numeric_obj': numeric_obj,
            'poured_goals': poured_goals,
            'balance_goal': balance_goal,
        }

    def get_instance(self, instance: Optional[str] = None) -> list[int]:
        if instance and instance in self._instances:
            path = os.path.join(PDDL_DIR, f"{instance}.pddl")
            return self._parse_pddl(path)
        raise ValueError(f"Instance '{instance}' not found!")

    def build_problem(self, instance: str | None = None) -> "Problem":
        data = self.get_instance(instance)

        plants = data['objects']['plant']
        taps = data['objects']['tap']
        agents = data['objects']['agent']
        numeric_global = data['numeric_global']
        numeric_obj = data['numeric_obj']
        poured_goals = data['poured_goals']
        balance_goal = data['balance_goal']

        max_x_val = numeric_global.get('maxx', 0)
        min_x_val = numeric_global.get('minx', 0)
        max_y_val = numeric_global.get('maxy', 0)
        min_y_val = numeric_global.get('miny', 0)
        water_reserve_val = numeric_global.get('water_reserve', 0)

        max_carry_vals = {obj: v for (f, obj), v in numeric_obj.items() if f == 'max_carry'}
        carrying_vals  = {obj: v for (f, obj), v in numeric_obj.items() if f == 'carrying'}
        poured_vals    = {obj: v for (f, obj), v in numeric_obj.items() if f == 'poured'}
        x_vals         = {obj: v for (f, obj), v in numeric_obj.items() if f == 'x'}
        y_vals         = {obj: v for (f, obj), v in numeric_obj.items() if f == 'y'}

        max_carry_val = max(max_carry_vals.values()) if max_carry_vals else 1

        # --- Problem ---
        problem = Problem('watering_problem')

        Thing = UserType('Thing')
        Agent = UserType('Agent', Thing)
        Plant = UserType('Plant', Thing)
        Tap = UserType('Tap', Thing)

        max_x = Fluent('max_x', IntType(max_x_val, max_x_val))
        max_y = Fluent('max_y', IntType(max_y_val, max_y_val))
        min_x = Fluent('min_x', IntType(min_x_val, min_x_val))
        min_y = Fluent('min_y', IntType(min_y_val, min_y_val))
        max_carry = Fluent('max_carry', IntType(0, max_carry_val), a=Agent)
        water_reserve = Fluent('water_reserve', IntType(0, water_reserve_val))
        x = Fluent('x', IntType(min_x_val, max_x_val), t=Thing)
        y = Fluent('y', IntType(min_y_val, max_y_val), t=Thing)
        carrying = Fluent('carrying', IntType(0, max_carry_val), a=Agent)
        poured = Fluent('poured', IntType(0, water_reserve_val), p=Plant)
        total_poured = Fluent('total_poured', IntType(0, water_reserve_val))
        total_loaded = Fluent('total_loaded', IntType(0, water_reserve_val))

        problem.add_fluent(max_x, default_initial_value=0)
        problem.add_fluent(max_y, default_initial_value=0)
        problem.add_fluent(min_x, default_initial_value=0)
        problem.add_fluent(min_y, default_initial_value=0)
        problem.add_fluent(max_carry, default_initial_value=0)
        problem.add_fluent(water_reserve, default_initial_value=0)
        problem.add_fluent(x, default_initial_value=0)
        problem.add_fluent(y, default_initial_value=0)
        problem.add_fluent(carrying, default_initial_value=0)
        problem.add_fluent(poured, default_initial_value=0)
        problem.add_fluent(total_poured, default_initial_value=0)
        problem.add_fluent(total_loaded, default_initial_value=0)

        # --- Objects ---
        for p in plants: problem.add_object(Object(p, Plant))
        for t in taps: problem.add_object(Object(t, Tap))
        for a in agents: problem.add_object(Object(a, Agent))

        def obj(name): return problem.object(name)

        # --- Init ---
        problem.set_initial_value(max_x, max_x_val)
        problem.set_initial_value(min_x, min_x_val)
        problem.set_initial_value(max_y, max_y_val)
        problem.set_initial_value(min_y, min_y_val)
        problem.set_initial_value(water_reserve, water_reserve_val)
        problem.set_initial_value(total_poured, 0)
        problem.set_initial_value(total_loaded, 0)

        for agent, val in max_carry_vals.items():
            problem.set_initial_value(max_carry(obj(agent)), val)
        for agent, val in carrying_vals.items():
            problem.set_initial_value(carrying(obj(agent)), val)
        for o, val in x_vals.items():
            problem.set_initial_value(x(obj(o)), val)
        for o, val in y_vals.items():
            problem.set_initial_value(y(obj(o)), val)
        for plant, val in poured_vals.items():
            problem.set_initial_value(poured(obj(plant)), val)

        # --- Actions ---
        move_up = InstantaneousAction('move_up', a=Agent)
        a = move_up.parameter('a')
        move_up.add_precondition(LE(Plus(y(a), 1), max_y))
        move_up.add_increase_effect(y(a), 1)

        move_down = InstantaneousAction('move_down', a=Agent)
        a = move_down.parameter('a')
        move_down.add_precondition(GE(Minus(y(a), 1), min_y))
        move_down.add_decrease_effect(y(a), 1)

        move_right = InstantaneousAction('move_right', a=Agent)
        a = move_right.parameter('a')
        move_right.add_precondition(LE(Plus(x(a), 1), max_x))
        move_right.add_increase_effect(x(a), 1)

        move_left = InstantaneousAction('move_left', a=Agent)
        a = move_left.parameter('a')
        move_left.add_precondition(GE(Minus(x(a), 1), min_x))
        move_left.add_decrease_effect(x(a), 1)

        move_up_left = InstantaneousAction('move_up_left', a=Agent)
        a = move_up_left.parameter('a')
        move_up_left.add_precondition(GE(Minus(x(a), 1), min_x))
        move_up_left.add_precondition(LE(Plus(y(a), 1), max_y))
        move_up_left.add_decrease_effect(x(a), 1)
        move_up_left.add_increase_effect(y(a), 1)

        move_up_right = InstantaneousAction('move_up_right', a=Agent)
        a = move_up_right.parameter('a')
        move_up_right.add_precondition(LE(Plus(x(a), 1), max_x))
        move_up_right.add_precondition(LE(Plus(y(a), 1), max_y))
        move_up_right.add_increase_effect(x(a), 1)
        move_up_right.add_increase_effect(y(a), 1)

        move_down_left = InstantaneousAction('move_down_left', a=Agent)
        a = move_down_left.parameter('a')
        move_down_left.add_precondition(GE(Minus(x(a), 1), min_x))
        move_down_left.add_precondition(GE(Minus(y(a), 1), min_y))
        move_down_left.add_decrease_effect(x(a), 1)
        move_down_left.add_decrease_effect(y(a), 1)

        move_down_right = InstantaneousAction('move_down_right', a=Agent)
        a = move_down_right.parameter('a')
        move_down_right.add_precondition(LE(Plus(x(a), 1), max_x))
        move_down_right.add_precondition(GE(Minus(y(a), 1), min_y))
        move_down_right.add_increase_effect(x(a), 1)
        move_down_right.add_decrease_effect(y(a), 1)

        load = InstantaneousAction('load', a=Agent, t=Tap)
        a = load.parameter('a')
        t = load.parameter('t')
        load.add_precondition(Equals(x(a), x(t)))
        load.add_precondition(Equals(y(a), y(t)))
        load.add_precondition(LE(Plus(carrying(a), 1), max_carry(a)))
        load.add_precondition(GE(Minus(water_reserve, 1), 0))
        load.add_increase_effect(carrying(a), 1)
        load.add_increase_effect(total_loaded, 1)
        load.add_decrease_effect(water_reserve, 1)

        pour = InstantaneousAction('pour', a=Agent, p=Plant)
        a = pour.parameter('a')
        p = pour.parameter('p')
        pour.add_precondition(Equals(x(a), x(p)))
        pour.add_precondition(Equals(y(a), y(p)))
        pour.add_precondition(GE(carrying(a), 1))
        pour.add_increase_effect(poured(p), 1)
        pour.add_increase_effect(total_poured, 1)
        pour.add_decrease_effect(carrying(a), 1)

        problem.add_actions([
            move_up, move_down, move_right, move_left, move_up_left, move_up_right, move_down_left, move_down_right,
            load, pour
        ])

        # --- Goals ---
        for plant, val in poured_goals.items():
            problem.add_goal(Equals(poured(obj(plant)), val))
        if balance_goal:
            problem.add_goal(Equals(total_poured, total_loaded))

        # --- Metric ---
        costs: Dict[Action, Expression] = {
            move_up: Int(1), move_down: Int(1),
            move_right: Int(1), move_left: Int(1),
            move_up_left: Int(2), move_up_right: Int(2),
            move_down_left: Int(2), move_down_right: Int(2),
            load: Int(2), pour: Int(2),
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem


DOMAIN = WateringDomain()