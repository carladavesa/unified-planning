"""expedition planning domain.

Example:
  python run.py --domain expedition --instance pfile1 --compilation uti --solving fast-downward
"""
import os
import re
from typing import Optional
from unified_planning.shortcuts import (
    Fluent, IntType, BoolType, UserType,
    InstantaneousAction, Problem,
    GE, LE, GT, And, Int,
)
from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'expedition', 'handcrafted')
INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 21)]


class ExpeditionDomain(Domain):
    def __init__(self):
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()
        content = re.sub(r';.*', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Objects
        objects_match = re.search(r'\(:objects\s+(.*?)\s*\)', content, re.DOTALL)
        sleds = []
        waypoints = []
        if objects_match:
            objects_str = objects_match.group(1)
            for group_match in re.finditer(r'((?:\S+\s+)+?)-\s*(\S+)', objects_str):
                names = group_match.group(1).strip().split()
                type_name = group_match.group(2).lower()
                if type_name == 'sled':
                    sleds.extend(names)
                elif type_name == 'waypoint':
                    waypoints.extend(names)

        # Extract :init and :goal sections separately to avoid cross-contamination
        init_match = re.search(r'\(:init\s+(.*?)\)\s*\(:goal', content, re.DOTALL)
        init_str = init_match.group(1) if init_match else ""

        goal_match = re.search(r'\(:goal\s*(.*?)\)\s*\)\s*$', content, re.DOTALL)
        goal_str = goal_match.group(1) if goal_match else ""

        # === Parse :init ===
        # at(s, w) initial positions
        at_init = {}
        for m in re.finditer(r'\(at\s+(\S+)\s+(\S+)\)', init_str):
            at_init[m.group(1).lower()] = m.group(2).lower()

        # isnext(w1, w2)
        isnext = []
        for m in re.finditer(r'\(isnext\s+(\S+)\s+(\S+)\)', init_str):
            isnext.append((m.group(1).lower(), m.group(2).lower()))

        # Numeric values
        sledcapacity = {}
        sledsupplies = {}
        waypointsupplies = {}
        for m in re.finditer(r'\(=\s*\(sledcapacity\s+(\S+)\)\s+([\d.]+)\)', init_str):
            sledcapacity[m.group(1).lower()] = int(float(m.group(2)))
        for m in re.finditer(r'\(=\s*\(sledsupplies\s+(\S+)\)\s+([\d.]+)\)', init_str):
            sledsupplies[m.group(1).lower()] = int(float(m.group(2)))
        for m in re.finditer(r'\(=\s*\(waypointsupplies\s+(\S+)\)\s+([\d.]+)\)', init_str):
            waypointsupplies[m.group(1).lower()] = int(float(m.group(2)))

        # === Parse :goal ===
        goal_at = {}
        for m in re.finditer(r'\(at\s+(\S+)\s+(\S+)\)', goal_str):
            goal_at[m.group(1).lower()] = m.group(2).lower()

        return {
            'sleds': sleds,
            'waypoints': waypoints,
            'at_init': at_init,
            'isnext': isnext,
            'sledcapacity': sledcapacity,
            'sledsupplies': sledsupplies,
            'waypointsupplies': waypointsupplies,
            'goal_at': goal_at,
        }

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if not instance or instance not in self._instances:
            raise ValueError(f"Instance '{instance}' not found!")
        problem_path = os.path.join(PDDL_DIR, f"{instance}.pddl")
        if not os.path.exists(problem_path):
            raise ValueError(f"Problem file not found: {problem_path}")
        return self._parse_pddl(problem_path)

    def build_problem(self, instance: str | None = None) -> Problem:
        data = self.get_instance(instance)
        sleds = data['sleds']
        waypoints = data['waypoints']
        at_init = data['at_init']
        isnext = data['isnext']
        sledcapacity_init = data['sledcapacity']
        sledsupplies_init = data['sledsupplies']
        waypointsupplies_init = data['waypointsupplies']
        goal_at = data['goal_at']

        # ==== Bounds ====
        # sledcapacity is static (no effect modifies it)
        max_capacity = max(sledcapacity_init.values(), default=1)
        # sledsupplies is bounded by sledcapacity
        sledsupplies_ub = max_capacity
        # waypointsupplies initial max bounds it (plus accumulation from stores)
        # Total supplies are conserved (retrieve ↔ store are inverse, move consumes)
        # Upper bound: initial total supplies
        reasonable_supply = len(sleds) * len(waypoints) * 2
        waypointsupplies_ub = reasonable_supply

        problem = Problem('expedition_problem')

        # Types
        Sled = UserType('Sled')
        Waypoint = UserType('Waypoint')

        # Objects
        sled_objs = {s: problem.add_object(s, Sled) for s in sleds}
        waypoint_objs = {w: problem.add_object(w, Waypoint) for w in waypoints}

        # Fluents
        at = Fluent('at', BoolType(), s=Sled, w=Waypoint)
        isnext_f = Fluent('isnext', BoolType(), x=Waypoint, y=Waypoint)
        sledsupplies = Fluent('sledsupplies', IntType(0, sledsupplies_ub), s=Sled)
        sledcapacity = Fluent('sledcapacity', IntType(0, max_capacity), s=Sled)
        waypointsupplies = Fluent('waypointsupplies', IntType(0, waypointsupplies_ub), w=Waypoint)

        problem.add_fluent(at, default_initial_value=False)
        problem.add_fluent(isnext_f, default_initial_value=False)
        problem.add_fluent(sledsupplies, default_initial_value=Int(0))
        problem.add_fluent(sledcapacity, default_initial_value=Int(0))
        problem.add_fluent(waypointsupplies, default_initial_value=Int(0))

        # Initial values
        for s, w in at_init.items():
            if s in sled_objs and w in waypoint_objs:
                problem.set_initial_value(at(sled_objs[s], waypoint_objs[w]), True)
        for w1, w2 in isnext:
            if w1 in waypoint_objs and w2 in waypoint_objs:
                problem.set_initial_value(isnext_f(waypoint_objs[w1], waypoint_objs[w2]), True)
        for s, v in sledcapacity_init.items():
            if s in sled_objs:
                problem.set_initial_value(sledcapacity(sled_objs[s]), Int(v))
        for s, v in sledsupplies_init.items():
            if s in sled_objs:
                problem.set_initial_value(sledsupplies(sled_objs[s]), Int(v))
        for w, v in waypointsupplies_init.items():
            if w in waypoint_objs:
                capped_v = min(v, waypointsupplies_ub)
                problem.set_initial_value(waypointsupplies(waypoint_objs[w]), Int(capped_v))

        # ==== Actions ====

        # moveforwards(?s, ?w1, ?w2)
        moveforwards = InstantaneousAction('moveforwards', s=Sled, w1=Waypoint, w2=Waypoint)
        s, w1, w2 = moveforwards.parameter('s'), moveforwards.parameter('w1'), moveforwards.parameter('w2')
        moveforwards.add_precondition(at(s, w1))
        moveforwards.add_precondition(GE(sledsupplies(s), Int(1)))
        moveforwards.add_precondition(isnext_f(w1, w2))
        moveforwards.add_effect(at(s, w1), False)
        moveforwards.add_effect(at(s, w2), True)
        moveforwards.add_decrease_effect(sledsupplies(s), Int(1))
        problem.add_action(moveforwards)

        # movebackwards(?s, ?w1, ?w2)
        movebackwards = InstantaneousAction('movebackwards', s=Sled, w1=Waypoint, w2=Waypoint)
        s, w1, w2 = movebackwards.parameter('s'), movebackwards.parameter('w1'), movebackwards.parameter('w2')
        movebackwards.add_precondition(at(s, w1))
        movebackwards.add_precondition(GE(sledsupplies(s), Int(1)))
        movebackwards.add_precondition(isnext_f(w2, w1))
        movebackwards.add_effect(at(s, w1), False)
        movebackwards.add_effect(at(s, w2), True)
        movebackwards.add_decrease_effect(sledsupplies(s), Int(1))
        problem.add_action(movebackwards)

        # storesupplies(?s, ?w)
        storesupplies = InstantaneousAction('storesupplies', s=Sled, w=Waypoint)
        s, w = storesupplies.parameter('s'), storesupplies.parameter('w')
        storesupplies.add_precondition(at(s, w))
        storesupplies.add_precondition(GE(sledsupplies(s), Int(1)))
        storesupplies.add_increase_effect(waypointsupplies(w), Int(1))
        storesupplies.add_decrease_effect(sledsupplies(s), Int(1))
        problem.add_action(storesupplies)

        # retrievesupplies(?s, ?w)
        retrievesupplies = InstantaneousAction('retrievesupplies', s=Sled, w=Waypoint)
        s, w = retrievesupplies.parameter('s'), retrievesupplies.parameter('w')
        retrievesupplies.add_precondition(at(s, w))
        retrievesupplies.add_precondition(GE(waypointsupplies(w), Int(1)))
        retrievesupplies.add_precondition(GT(sledcapacity(s), sledsupplies(s)))
        retrievesupplies.add_decrease_effect(waypointsupplies(w), Int(1))
        retrievesupplies.add_increase_effect(sledsupplies(s), Int(1))
        problem.add_action(retrievesupplies)

        # Goals
        for s, w in goal_at.items():
            if s in sled_objs and w in waypoint_objs:
                problem.add_goal(at(sled_objs[s], waypoint_objs[w]))

        return problem


DOMAIN = ExpeditionDomain()