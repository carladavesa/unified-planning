"""
Extended Plant Watering
A set of agents on a map aims to water some plants by carrying water from a central reservoir to the plants. 
All agents can move along the eigth compass directions, and have specific actions to load the water and pouring
the plants. This version adds a contraint over the total amount of water and the maximum amount of water
that can be carried by an agent
"""

import subprocess
import argparse
from docs.extensions.domains import compilation_solving
from unified_planning.shortcuts import *

# Parser
parser = argparse.ArgumentParser(description="Solve Counters")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Read instance
instance_path = f'/Users/cds26/PycharmProjects/unified-planning/docs/extensions/domains/watering/read_instance.py'
instance = subprocess.run(['python3', instance_path, 'pfile1'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

output = instance.stdout.split("---")

# ---- Objects ----
obj_lines = output[0].splitlines()
plants = next(l.split('=')[1].split(',') for l in obj_lines if l.startswith('plants='))
taps = next(l.split('=')[1].split(',') for l in obj_lines if l.startswith('taps='))
agents = next(l.split('=')[1].split(',') for l in obj_lines if l.startswith('agents='))

# ---- Globals ----
max_x_val = int(next(l.split('=')[1] for l in output[1].splitlines() if l.startswith('maxx=')))
min_x_val = int(next(l.split('=')[1] for l in output[1].splitlines() if l.startswith('minx=')))
max_y_val = int(next(l.split('=')[1] for l in output[1].splitlines() if l.startswith('maxy=')))
min_y_val = int(next(l.split('=')[1] for l in output[1].splitlines() if l.startswith('miny=')))
water_reserve_val = int(next(l.split('=')[1] for l in output[1].splitlines() if l.startswith('water_reserve=')))


# ---- Per object ----
max_carry_vals = {}
carrying_vals  = {}
poured_vals    = {}
x_vals         = {}
y_vals         = {}

for line in output[2].splitlines():
    if line.startswith('max_carry('):
        agent, val = line[10:].split(')=')
        max_carry_vals[agent] = int(val)
    elif line.startswith('carrying('):
        agent, val = line[9:].split(')=')
        carrying_vals[agent] = int(val)
    elif line.startswith('poured('):
        plant, val = line[7:].split(')=')
        poured_vals[plant] = int(val)
    elif line.startswith('x('):
        obj, val = line[2:].split(')=')
        x_vals[obj] = int(val)
    elif line.startswith('y('):
        obj, val = line[2:].split(')=')
        y_vals[obj] = int(val)

# ---- Goals ----
poured_goals = {}
balance_goal = False
for line in output[3].splitlines():
    if line.startswith('goal_poured('):
        plant, val = line[12:].split(')=')
        poured_goals[plant] = int(val)
    elif line.startswith('goal_balance=true'):
        balance_goal = True

# ---- Bounds ----
max_carry_val  = max(max_carry_vals.values())

# ------------------------------------------------ Problem ------------------------------------------------------

watering_problem = unified_planning.model.Problem('watering_problem')

Thing = UserType('Thing')
Location = UserType('Location')
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

watering_problem.add_fluent(max_x, default_initial_value=0)
watering_problem.add_fluent(max_y, default_initial_value=0)
watering_problem.add_fluent(min_x, default_initial_value=0)
watering_problem.add_fluent(min_y, default_initial_value=0)
watering_problem.add_fluent(max_carry, default_initial_value=0)
watering_problem.add_fluent(water_reserve, default_initial_value=0)
watering_problem.add_fluent(x, default_initial_value=0)
watering_problem.add_fluent(y, default_initial_value=0)
watering_problem.add_fluent(carrying, default_initial_value=0)
watering_problem.add_fluent(poured, default_initial_value=0)
watering_problem.add_fluent(total_poured, default_initial_value=0)
watering_problem.add_fluent(total_loaded, default_initial_value=0)

# ------------------------------------------------ Initialisation ------------------------------------------------------
def get_object(name):
    return watering_problem.object(name)

for p in plants:  watering_problem.add_object(Object(p, Plant))
for t in taps:    watering_problem.add_object(Object(t, Tap))
for a in agents:  watering_problem.add_object(Object(a, Agent))

watering_problem.set_initial_value(max_x, max_x_val)
watering_problem.set_initial_value(min_x, min_x_val)
watering_problem.set_initial_value(max_y, max_y_val)
watering_problem.set_initial_value(min_y, min_y_val)
watering_problem.set_initial_value(water_reserve, water_reserve_val)
watering_problem.set_initial_value(total_poured, 0)
watering_problem.set_initial_value(total_loaded, 0)

for agent, val in max_carry_vals.items():
    watering_problem.set_initial_value(max_carry(get_object(agent)), val)
for agent, val in carrying_vals.items():
    watering_problem.set_initial_value(carrying(get_object(agent)), val)
for obj, val in x_vals.items():
    watering_problem.set_initial_value(x(get_object(obj)), val)
for obj, val in y_vals.items():
    watering_problem.set_initial_value(y(get_object(obj)), val)
for plant, val in poured_vals.items():
    watering_problem.set_initial_value(poured(get_object(plant)), val)

# ---------------------------------------------------- Actions ---------------------------------------------------------

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

watering_problem.add_actions([move_up, move_down, move_right, move_left, move_up_left, move_up_right,
                              move_down_left, move_down_right, load, pour])

# ---------------------------------------------------- Goals ---------------------------------------------------------

for plant, val in poured_goals.items():
    watering_problem.add_goal(Equals(poured(get_object(plant)), val))
if balance_goal:
    watering_problem.add_goal(Equals(total_poured, total_loaded))

costs: Dict[Action, Expression] = {
    move_up: Int(1),
    move_down: Int(1),
    move_right: Int(1),
    move_left: Int(1),
    move_up_left: Int(2),
    move_up_right: Int(2),
    move_down_left: Int(2),
    move_down_right: Int(2),
    load: Int(2),
    pour: Int(2)
}
watering_problem.add_quality_metric(MinimizeActionCosts(costs))

compilation_solving.compile_and_solve(watering_problem, solving, compilation)