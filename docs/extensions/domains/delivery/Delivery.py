import subprocess

from docs.extensions.domains import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Run: python -m docs.extensions.domains.delivery.Delivery --compilation uti --solving fast-downward

# Parser
parser = argparse.ArgumentParser(description="Solve Delivery")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Read instance
instance_path = f'/Users/cds26/PycharmProjects/unified-planning/docs/extensions/domains/delivery/read_instance.py'
instance = subprocess.run(['python3', instance_path, 'pfile1'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")

# ---- Objectes ----
obj_lines = output[0].splitlines()
rooms = next(l.split('=')[1].split(',') for l in obj_lines if l.startswith('rooms='))
items = next(l.split('=')[1].split(',') for l in obj_lines if l.startswith('items='))
bots  = next(l.split('=')[1].split(',') for l in obj_lines if l.startswith('bots='))
arms  = next(l.split('=')[1].split(',') for l in obj_lines if l.startswith('arms='))

# ---- at_bot ----
at_bot_init = {}
for line in output[1].splitlines():
    if line.startswith('at_bot('):
        bot, room = line[7:-1].split(',')
        at_bot_init[bot] = room

# ---- at ----
at_init = {}
for line in output[2].splitlines():
    if line.startswith('at('):
        item, room = line[3:-1].split(',')
        at_init[item] = room

# ---- free ----
free_init = []
for line in output[3].splitlines():
    if line.startswith('free('):
        free_init.append(line[5:-1])

# ---- mount ----
mount_init = {}
for line in output[4].splitlines():
    if line.startswith('mount('):
        arm, bot = line[6:-1].split(',')
        mount_init[arm] = bot

# ---- door ----
door_init = []
for line in output[5].splitlines():
    if line.startswith('door('):
        r1, r2 = line[5:-1].split(',')
        door_init.append((r1, r2))

# ---- weight ----
weights = {}
for line in output[6].splitlines():
    if line.startswith('weight('):
        item, val = line[7:].split(')=')
        weights[item] = int(val)

# ---- load_limit / current_load ----
load_limits   = {}
current_loads = {}
for line in output[7].splitlines():
    if line.startswith('load_limit('):
        bot, val = line[11:].split(')=')
        load_limits[bot] = int(val)
    elif line.startswith('current_load('):
        bot, val = line[13:].split(')=')
        current_loads[bot] = int(val)

# ---- goals ----
goals_at = []
for line in output[8].splitlines():
    if line.startswith('goal_at('):
        item, room = line[8:-1].split(',')
        goals_at.append((item, room))

# Bounds
max_weight     = max(weights.values())
max_load_limit = max(load_limits.values())

# ---------------------------------------------------- Problem ---------------------------------------------------------

delivery_problem = unified_planning.model.Problem('delivery_problem')

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
delivery_problem.add_fluent(at_bot, default_initial_value=False)
delivery_problem.add_fluent(at, default_initial_value=False)
delivery_problem.add_fluent(door, default_initial_value=False)
delivery_problem.add_fluent(free, default_initial_value=False)
delivery_problem.add_fluent(in_arm, default_initial_value=False)
delivery_problem.add_fluent(in_tray, default_initial_value=False)
delivery_problem.add_fluent(mount, default_initial_value=False)
load_limit = Fluent('load_limit', IntType(0, max_load_limit), b=Bot)
current_load = Fluent('current_load', IntType(0, max_load_limit), b=Bot)
weight = Fluent('weight', IntType(0, max_weight), i=Item)
delivery_problem.add_fluent(load_limit, default_initial_value=0)
delivery_problem.add_fluent(current_load, default_initial_value=0)
delivery_problem.add_fluent(weight, default_initial_value=0)

# ------------------------------------------------ Initialisation ------------------------------------------------------

def get_object(name):
    return delivery_problem.object(name)

# Objects
for r in rooms:
    delivery_problem.add_object(Object(r, Room))
for i in items:
    delivery_problem.add_object(Object(i, Item))
for b in bots:
    delivery_problem.add_object(Object(b, Bot))
for a in arms:
    delivery_problem.add_object(Object(a, Arm))

# Boolean initial values
for bot, room in at_bot_init.items():
    delivery_problem.set_initial_value(at_bot(get_object(bot), get_object(room)), True)
for item, room in at_init.items():
    delivery_problem.set_initial_value(at(get_object(item), get_object(room)), True)
for arm in free_init:
    delivery_problem.set_initial_value(free(get_object(arm)), True)
for arm, bot in mount_init.items():
    delivery_problem.set_initial_value(mount(get_object(arm), get_object(bot)), True)
for r1, r2 in door_init:
    delivery_problem.set_initial_value(door(get_object(r1), get_object(r2)), True)

# Numeric initial values
for item, w in weights.items():
    delivery_problem.set_initial_value(weight(get_object(item)), w)
for bot, ll in load_limits.items():
    delivery_problem.set_initial_value(load_limit(get_object(bot)), ll)
for bot, cl in current_loads.items():
    delivery_problem.set_initial_value(current_load(get_object(bot)), cl)

# ---------------------------------------------------- Actions ---------------------------------------------------------

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
drop.add_decrease_effect(current_load(b), weight(i))

to_tray = InstantaneousAction('to_tray', i=Item, a=Arm, b=Bot)
i = to_tray.parameter('i')
a = to_tray.parameter('a')
b = to_tray.parameter('b')
to_tray.add_precondition(in_arm(i, a))
to_tray.add_precondition(at_bot(b, x))
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

delivery_problem.add_actions([move, pick, drop, to_tray, from_tray])

for item, room in goals_at:
    delivery_problem.add_goal(at(get_object(item), get_object(room)))

costs: Dict[Action, Expression] = {
    move: Int(3),
    pick: Int(2),
    drop: Int(2),
    to_tray: Int(1),
    from_tray: Int(1),
}
delivery_problem.add_quality_metric(MinimizeActionCosts(costs))

compilation_solving.compile_and_solve(delivery_problem, solving, compilation)