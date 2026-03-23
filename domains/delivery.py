"""delivery planning domain.

This module implements the delivery benchmark as a `Domain`.
It is intended to be executed via `run.py`.

Example:
  python run.py --domain delivery --compilation uti --solving fast-downward
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

INSTANCES: list[str] = [
    "pfile1", "pfile2", "pfile3", "pfile4", "pfile5", "pfile6", "pfile7", "pfile8", "pfile9", "pfile10", "pfile11", "pfile12", "pfile13", "pfile14", "pfile15", "pfile16", "pfile17", "pfile18", "pfile19", "pfile20",
]

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'delivery/handcrafted')

class DeliveryDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {"stack": v} for k, v in self._instances.items()}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()

        content = re.sub(r';.*', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Objects
        objects = {'room': [], 'item': [], 'bot': [], 'arm': []}
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

        # Numeric fluents
        numeric = {}
        for m in re.finditer(r'\(=\s*\((\w[\w-]*)\s+([^)]*?)\)\s+([\d.]+)\)', init_str):
            fluent = m.group(1).replace('-', '_')
            args = tuple(m.group(2).strip().split())
            value = int(m.group(3)) if '.' not in m.group(3) else float(m.group(3))
            numeric[(fluent, args)] = value

        # Boolean predicates
        bool_facts = []
        init_no_numeric = re.sub(r'\(=\s*\([^)]*\)\s+[\d.]+\)', '', init_str)
        for m in re.finditer(r'\((\w[\w-]*)((?:\s+\w+)*)\)', init_no_numeric):
            pred = m.group(1).replace('-', '_')
            args = m.group(2).strip().split()
            bool_facts.append((pred, args))

        at_bot_init = {a[0]: a[1] for p, a in bool_facts if p == 'at_bot' and len(a) == 2}
        at_init = {a[0]: a[1] for p, a in bool_facts if p == 'at' and len(a) == 2}
        free_init = [a[0] for p, a in bool_facts if p == 'free' and len(a) == 1]
        mount_init = {a[0]: a[1] for p, a in bool_facts if p == 'mount' and len(a) == 2}
        door_init = [(a[0], a[1]) for p, a in bool_facts if p == 'door' and len(a) == 2]

        weights = {a[0]: v for (f, a), v in numeric.items() if f == 'weight'}
        load_limits = {a[0]: v for (f, a), v in numeric.items() if f == 'load_limit'}
        current_loads = {a[0]: v for (f, a), v in numeric.items() if f == 'current_load'}

        # Goal
        goal_match = re.search(r'\(:goal\s*\(and(.*?)\)\s*\)', content, re.IGNORECASE | re.DOTALL)
        if not goal_match:
            goal_match = re.search(r'\(:goal\s*(.*?)\)\s*\(:metric', content, re.IGNORECASE | re.DOTALL)
        goals_at = []
        if goal_match:
            for m in re.finditer(r'\((\w[\w-]*)((?:\s+\w+)*)\)', goal_match.group(1)):
                pred = m.group(1).replace('-', '_')
                args = m.group(2).strip().split()
                if pred == 'at' and len(args) == 2:
                    goals_at.append((args[0], args[1]))

        return {
            'objects': objects,
            'at_bot': at_bot_init,
            'at': at_init,
            'free': free_init,
            'mount': mount_init,
            'door': door_init,
            'weights': weights,
            'load_limits': load_limits,
            'current_load': current_loads,
            'goals_at': goals_at,
        }

    def get_instance(self, instance: Optional[str] = None):
        """Return a stack instance."""

        # Use pre-defined instances if available.
        if instance and instance in self._instances:
            path = os.path.join(PDDL_DIR, f"{instance}.pddl")
            return self._parse_pddl(path)
        raise ValueError(f"Instance '{instance}' not found!")


    def build_problem(self, instance: str | None = None) -> "Problem":
        data = self.get_instance(instance)
        rooms = data['objects']['room']
        items = data['objects']['item']
        bots = data['objects']['bot']
        arms = data['objects']['arm']
        at_bot_init = data['at_bot']
        at_init = data['at']
        free_init = data['free']
        mount_init = data['mount']
        door_init = data['door']
        weights = data['weights']
        load_limits = data['load_limits']
        current_loads = data['current_load']
        goals_at = data['goals_at']

        max_weight = max(weights.values())
        max_load_limit = max(load_limits.values())

        # --- Problem ---
        problem = Problem('delivery_problem')

        Room = UserType('Room')
        Item = UserType('Item')
        Arm = UserType('Arm')
        Bot = UserType('Bot')

        at_bot = Fluent('at_bot', b=Bot, x=Room)
        at = Fluent('at', i=Item, x=Room)
        door = Fluent('door', x=Room, y=Room)
        free = Fluent('free', a=Arm)
        in_arm = Fluent('in_arm', i=Item, a=Arm)
        in_tray = Fluent('in_tray', i=Item, b=Bot)
        mount = Fluent('mount', a=Arm, b=Bot)
        load_limit = Fluent('load_limit', IntType(0, max_load_limit), b=Bot)
        current_load = Fluent('current_load', IntType(0, max_load_limit), b=Bot)
        weight = Fluent('weight', IntType(0, max_weight), i=Item)

        problem.add_fluent(at_bot, default_initial_value=False)
        problem.add_fluent(at, default_initial_value=False)
        problem.add_fluent(door, default_initial_value=False)
        problem.add_fluent(free, default_initial_value=False)
        problem.add_fluent(in_arm, default_initial_value=False)
        problem.add_fluent(in_tray, default_initial_value=False)
        problem.add_fluent(mount, default_initial_value=False)
        problem.add_fluent(load_limit, default_initial_value=0)
        problem.add_fluent(current_load, default_initial_value=0)
        problem.add_fluent(weight, default_initial_value=0)

        # --- Objects ---
        for r in rooms: problem.add_object(Object(r, Room))
        for i in items: problem.add_object(Object(i, Item))
        for b in bots:  problem.add_object(Object(b, Bot))
        for a in arms:  problem.add_object(Object(a, Arm))

        def obj(name): return problem.object(name)

        # --- Init ---
        for bot, room in at_bot_init.items():
            problem.set_initial_value(at_bot(obj(bot), obj(room)), True)
        for item, room in at_init.items():
            problem.set_initial_value(at(obj(item), obj(room)), True)
        for arm in free_init:
            problem.set_initial_value(free(obj(arm)), True)
        for arm, bot in mount_init.items():
            problem.set_initial_value(mount(obj(arm), obj(bot)), True)
        for r1, r2 in door_init:
            problem.set_initial_value(door(obj(r1), obj(r2)), True)
        for item, w in weights.items():
            problem.set_initial_value(weight(obj(item)), w)
        for bot, ll in load_limits.items():
            problem.set_initial_value(load_limit(obj(bot)), ll)
        for bot, cl in current_loads.items():
            problem.set_initial_value(current_load(obj(bot)), cl)

        # --- Actions ---
        move = InstantaneousAction('move', b=Bot, x=Room, y=Room)
        b = move.parameter('b')
        x = move.parameter('x')
        y = move.parameter('y')
        move.add_precondition(at_bot(b, x))
        move.add_precondition(door(x, y))
        move.add_effect(at_bot(b, y), True)
        move.add_effect(at_bot(b, x), False)

        pick = InstantaneousAction('pick', i=Item, x=Room, a=Arm, b=Bot)
        i = pick.parameter('i')
        x = pick.parameter('x')
        a = pick.parameter('a')
        b = pick.parameter('b')
        pick.add_precondition(at(i, x))
        pick.add_precondition(at_bot(b, x))
        pick.add_precondition(free(a))
        pick.add_precondition(mount(a, b))
        pick.add_precondition(LE(Plus(current_load(b), weight(i)), load_limit(b)))
        pick.add_effect(in_arm(i, a), True)
        pick.add_effect(at(i, x), False)
        pick.add_effect(free(a), False)
        pick.add_increase_effect(current_load(b), weight(i))

        drop = InstantaneousAction('drop', i=Item, x=Room, a=Arm, b=Bot)
        i = drop.parameter('i')
        x = drop.parameter('x')
        a = drop.parameter('a')
        b = drop.parameter('b')
        drop.add_precondition(in_arm(i, a))
        drop.add_precondition(at_bot(b, x))
        drop.add_precondition(mount(a, b))
        drop.add_effect(free(a), True)
        drop.add_effect(at(i, x), True)
        drop.add_effect(in_arm(i, a), False)
        drop.add_decrease_effect(current_load(b), weight(i))

        to_tray = InstantaneousAction('to_tray', i=Item, a=Arm, b=Bot)
        i = to_tray.parameter('i')
        a = to_tray.parameter('a')
        b = to_tray.parameter('b')
        to_tray.add_precondition(in_arm(i, a))
        to_tray.add_precondition(mount(a, b))
        to_tray.add_effect(free(a), True)
        to_tray.add_effect(in_arm(i, a), False)
        to_tray.add_effect(in_tray(i, b), True)

        from_tray = InstantaneousAction('from_tray', i=Item, a=Arm, b=Bot)
        i = from_tray.parameter('i')
        a = from_tray.parameter('a')
        b = from_tray.parameter('b')
        from_tray.add_precondition(in_tray(i, b))
        from_tray.add_precondition(mount(a, b))
        from_tray.add_precondition(free(a))
        from_tray.add_effect(free(a), False)
        from_tray.add_effect(in_arm(i, a), True)
        from_tray.add_effect(in_tray(i, b), False)

        problem.add_actions([move, pick, drop, to_tray, from_tray])

        # --- Goal ---
        for item, room in goals_at:
            problem.add_goal(at(obj(item), obj(room)))

        costs: Dict[Action, Expression] = {
            move: Int(3), pick: Int(2), drop: Int(2), to_tray: Int(1), from_tray: Int(1),
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))
        return problem


DOMAIN = DeliveryDomain()