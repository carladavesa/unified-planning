"""hydro_power planning domain.

Example:
  python run.py --domain hydro_power --instance pfile1 --compilation uti --solving fast-downward
"""
import os
import re
from typing import Optional
from unified_planning.shortcuts import (
    Fluent, IntType, BoolType, UserType,
    InstantaneousAction, Problem,
    GE, LE, And, Int, Times,
)
from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'hydro_power', 'handcrafted')
INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 21)]


class HydroPowerDomain(Domain):
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
        times = []
        turnvalues = []
        if objects_match:
            objects_str = objects_match.group(1)
            for group_match in re.finditer(r'((?:\S+\s+)+?)-\s*(\S+)', objects_str):
                names = group_match.group(1).strip().split()
                type_name = group_match.group(2).lower()
                if type_name == 'time':
                    times.extend(names)
                elif type_name == 'turnvalue':
                    turnvalues.extend(names)

        # Extract :init and :goal sections separately
        init_match = re.search(r'\(:init\s+(.*?)\)\s*\(:goal', content, re.DOTALL)
        init_str = init_match.group(1) if init_match else ""

        goal_match = re.search(r'\(:goal\s*(.*?)\)\s*\)\s*$', content, re.DOTALL)
        goal_str = goal_match.group(1) if goal_match else ""

        # === Parse :init ===
        # value(?n) = v
        values = {}
        for m in re.finditer(r'\(=\s*\(value\s+(\S+)\)\s+([\d.]+)\)', init_str):
            values[m.group(1).lower()] = int(float(m.group(2)))

        # demand(?t, ?n)
        demand = []
        for m in re.finditer(r'\(demand\s+(\S+)\s+(\S+)\)', init_str):
            demand.append((m.group(1).lower(), m.group(2).lower()))

        # before(?t1, ?t2)
        before = []
        for m in re.finditer(r'\(before\s+(\S+)\s+(\S+)\)', init_str):
            before.append((m.group(1).lower(), m.group(2).lower()))

        # timenow(?t)
        timenow_init = None
        m = re.search(r'\(timenow\s+(\S+)\)', init_str)
        if m:
            timenow_init = m.group(1).lower()

        # Numeric initial values
        stored_units = 0
        stored_capacity = 0
        funds = 0
        m = re.search(r'\(=\s*\(stored_units\)\s+([\d.]+)\)', init_str)
        if m:
            stored_units = int(float(m.group(1)))
        m = re.search(r'\(=\s*\(stored_capacity\)\s+([\d.]+)\)', init_str)
        if m:
            stored_capacity = int(float(m.group(1)))
        m = re.search(r'\(=\s*\(funds\)\s+([\d.]+)\)', init_str)
        if m:
            funds = int(float(m.group(1)))

        # === Parse :goal ===
        # (<= VAL (funds)) or (>= (funds) VAL)
        goal_funds = 0
        m = re.search(r'\(<=\s+([\d.]+)\s+\(funds\)\)', goal_str)
        if m:
            goal_funds = int(float(m.group(1)))
        else:
            m = re.search(r'\(>=\s*\(funds\)\s+([\d.]+)\)', goal_str)
            if m:
                goal_funds = int(float(m.group(1)))

        return {
            'times': times,
            'turnvalues': turnvalues,
            'values': values,
            'demand': demand,
            'before': before,
            'timenow_init': timenow_init,
            'stored_units': stored_units,
            'stored_capacity': stored_capacity,
            'funds': funds,
            'goal_funds': goal_funds,
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
        times = data['times']
        turnvalues = data['turnvalues']
        values_init = data['values']
        demand = data['demand']
        before = data['before']
        timenow_init = data['timenow_init']
        initial_stored_units = data['stored_units']
        initial_stored_capacity = data['stored_capacity']
        initial_funds = data['funds']
        goal_funds = data['goal_funds']

        # ==== Bounds ====
        # value: static, max del init
        max_value = max(values_init.values(), default=1)
        value_ub = max_value

        # stored_capacity + stored_units are conserved (what's generated/pumped)
        total_storage = initial_stored_units + initial_stored_capacity
        stored_units_ub = total_storage
        stored_capacity_ub = total_storage

        funds_ub = goal_funds + 20 * max_value

        problem = Problem('hydro_power_problem')

        # Types
        Time = UserType('Time')
        TurnValue = UserType('TurnValue')

        # Objects
        time_objs = {t: problem.add_object(t, Time) for t in times}
        turnvalue_objs = {n: problem.add_object(f'tv_{n}', TurnValue) for n in turnvalues}

        # Fluents
        timenow = Fluent('timenow', BoolType(), t=Time)
        before_f = Fluent('before', BoolType(), t1=Time, t2=Time)
        demand_f = Fluent('demand', BoolType(), t=Time, n=TurnValue)
        value_f = Fluent('value', IntType(0, value_ub), n=TurnValue)
        funds_f = Fluent('funds', IntType(0, funds_ub))
        stored_units_f = Fluent('stored_units', IntType(0, stored_units_ub))
        stored_capacity_f = Fluent('stored_capacity', IntType(0, stored_capacity_ub))

        problem.add_fluent(timenow, default_initial_value=False)
        problem.add_fluent(before_f, default_initial_value=False)
        problem.add_fluent(demand_f, default_initial_value=False)
        problem.add_fluent(value_f, default_initial_value=Int(0))
        problem.add_fluent(funds_f, default_initial_value=Int(0))
        problem.add_fluent(stored_units_f, default_initial_value=Int(0))
        problem.add_fluent(stored_capacity_f, default_initial_value=Int(0))

        # Initial values
        for n, v in values_init.items():
            if n in turnvalue_objs:
                problem.set_initial_value(value_f(turnvalue_objs[n]), Int(v))
        for t, n in demand:
            if t in time_objs and n in turnvalue_objs:
                problem.set_initial_value(demand_f(time_objs[t], turnvalue_objs[n]), True)
        for t1, t2 in before:
            if t1 in time_objs and t2 in time_objs:
                problem.set_initial_value(before_f(time_objs[t1], time_objs[t2]), True)
        if timenow_init and timenow_init in time_objs:
            problem.set_initial_value(timenow(time_objs[timenow_init]), True)

        problem.set_initial_value(funds_f(), Int(min(initial_funds, funds_ub)))
        problem.set_initial_value(stored_units_f(), Int(initial_stored_units))
        problem.set_initial_value(stored_capacity_f(), Int(initial_stored_capacity))

        # ==== Actions ====

        # advancetime(?t1, ?t2)
        advancetime = InstantaneousAction('advancetime', t1=Time, t2=Time)
        t1, t2 = advancetime.parameter('t1'), advancetime.parameter('t2')
        advancetime.add_precondition(timenow(t1))
        advancetime.add_precondition(before_f(t1, t2))
        advancetime.add_effect(timenow(t2), True)
        advancetime.add_effect(timenow(t1), False)
        problem.add_action(advancetime)

        # pumpwaterup(?t1, ?n1)
        pumpwaterup = InstantaneousAction('pumpwaterup', t=Time, tv=TurnValue)
        t, tv = pumpwaterup.parameter('t'), pumpwaterup.parameter('tv')
        pumpwaterup.add_precondition(timenow(t1))
        pumpwaterup.add_precondition(GE(funds_f(), Times(Int(21), value_f(tv))))
        pumpwaterup.add_precondition(GE(stored_capacity_f(), Int(1)))
        pumpwaterup.add_precondition(demand_f(t, tv))
        pumpwaterup.add_increase_effect(stored_units_f(), Int(1))
        pumpwaterup.add_decrease_effect(stored_capacity_f(), Int(1))
        pumpwaterup.add_decrease_effect(funds_f(), Times(Int(21), value_f(tv)))
        problem.add_action(pumpwaterup)

        generate = InstantaneousAction('generate', t=Time, tv=TurnValue)
        t, tv = generate.parameter('t'), generate.parameter('tv')
        generate.add_precondition(timenow(t))
        generate.add_precondition(GE(stored_units_f(), Int(1)))
        generate.add_precondition(demand_f(t, tv))
        generate.add_decrease_effect(stored_units_f(), Int(1))
        generate.add_increase_effect(stored_capacity_f(), Int(1))
        generate.add_increase_effect(funds_f(), Times(Int(20), value_f(tv)))
        problem.add_action(generate)

        # Goal
        problem.add_goal(GE(funds_f(), Int(goal_funds)))

        return problem


DOMAIN = HydroPowerDomain()