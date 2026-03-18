import subprocess

from docs.extensions.domains import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Parser
parser = argparse.ArgumentParser(description="Solve Counters")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Read instance
instance_path = f'/Users/cds26/PycharmProjects/unified-planning/docs/extensions/domains/sailing/read_instance.py'
instance = subprocess.run(['python3', instance_path, 'pfile4'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
max_int_n = eval(output[0].strip())
values = eval(output[1].strip())
counters = len(values)

# ---------------------------------------------------- Problem ---------------------------------------------------------
"""
(define (domain delivery)
  (:types room item arm bot - object)
   (:predicates (at-bot ?b - bot ?x - room)
		 (at ?i - item ?x - room)
		 (door ?x - room ?y - room)
		 (free ?a - arm)
		 (in-arm ?i - item ?a - arm)
		 (in-tray ?i - item ?b - bot) 
		 (mount ?a - arm ?b - bot))
		
   (:functions 
	(load_limit ?b - bot) 
	(current_load ?b - bot) 
	(weight ?i - item)
	(total-cost))

"""
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
load_limit = Fluent('load_limit', IntType(), b=Bot)
current_load = Fluent('current_load', IntType(), b=Bot)
weight = Fluent('current_load', IntType(), b=Bot)
delivery_problem.add_fluent(load_limit, default_initial_value=0)
delivery_problem.add_fluent(current_load, default_initial_value=0)
delivery_problem.add_fluent(weight, default_initial_value=0)

# ------------------------------------------------ Initialisation ------------------------------------------------------

counters_problem.set_initial_value(max_int, max_int_n)
for c in range(counters):
    c_object = Object(f'c{c}', Counter)
    counters_problem.add_object(c_object)
    counters_problem.set_initial_value(value(c_object), values[c])

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
pick.add_precondition(at_bot(i, x))
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

delivery_problem.add_goal()

costs: Dict[Action, Expression] = {
    move: Int(3),
    pick: Int(2),
    drop: Int(2),
    to_tray: Int(1),
    from_tray: Int(1),
}
delivery_problem.add_quality_metric(MinimizeActionCosts(costs))

compilation_solving.compile_and_solve(delivery_problem, solving, compilation)