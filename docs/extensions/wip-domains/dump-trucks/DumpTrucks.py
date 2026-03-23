import math
import argparse
from unified_planning.shortcuts import *
from docs.extensions.domains import compilation_solving

# Run: python -m docs.extensions.domains.dump-trucks.DumpTrucks --compilation sc --solving fast-downward

# Trucks deliver packages between locations. Packages are loaded one by one,
# and each unload operation dumps the full truck content.

# --- Parser ---
parser = argparse.ArgumentParser(description='Solve Dump Trucks')
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')
args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# --- Instance ---
n_packages = 10

# --- Problem ---
dump_trucks_problem = Problem('dump_trucks_problem')

Location = UserType('Location')
l1 = Object('l1', Location)
l2 = Object('l2', Location)

Truck = UserType('Truck')
t1 = Object('t1', Truck)
t2 = Object('t2', Truck)

Package = UserType('Package')
packages = []
for i in range(n_packages):
    packages.append(Object(f'p{i+1}', Package))

dump_trucks_problem.add_objects([l1, l2, t1, t2])
dump_trucks_problem.add_objects(packages)

loc_of_truck = Fluent('loc_of_truck', Location, t=Truck)  # where a truck is
pat = Fluent('pat', SetType(Package), l=Location)  # packages at a location
pin = Fluent('pin', SetType(Package), T=Truck)  # packages in a truck
connects = Fluent('connects', SetType(Location), l=Location)  # locations connected from a location

dump_trucks_problem.add_fluent(loc_of_truck, default_initial_value=l1)
dump_trucks_problem.add_fluent(pat, default_initial_value=set())
dump_trucks_problem.add_fluent(pin, default_initial_value=set())
dump_trucks_problem.add_fluent(connects, default_initial_value=set())

dump_trucks_problem.set_initial_value(loc_of_truck(t1), l1)
dump_trucks_problem.set_initial_value(loc_of_truck(t2), l2)
dump_trucks_problem.set_initial_value(pat(l1), {*packages})
dump_trucks_problem.set_initial_value(connects(l1), {l2})
dump_trucks_problem.set_initial_value(connects(l2), {l1})

# --- Actions ---

move_truck = InstantaneousAction('move_truck', t=Truck, lfrom=Location, lto=Location)
t = move_truck.parameter('t')
lfrom = move_truck.parameter('lfrom')
lto = move_truck.parameter('lto')
move_truck.add_precondition(SetMember(lto, connects(lfrom)))
move_truck.add_precondition(Equals(loc_of_truck(t), lfrom))
move_truck.add_effect(loc_of_truck(t), lto)

load_truck = InstantaneousAction('load_truck', p=Package, t=Truck, l=Location)
p = load_truck.parameter('p')
t = load_truck.parameter('t')
l = load_truck.parameter('l')
load_truck.add_precondition(Equals(l, loc_of_truck(t)))
load_truck.add_precondition(SetMember(p, pat(l)))
load_truck.add_precondition(LT(SetCardinality(pin(t)), math.ceil(n_packages / 2)))
load_truck.add_effect(pat(l), SetRemove(p, pat(l)))
load_truck.add_effect(pin(t), SetAdd(p, pin(t)))

unload_truck = InstantaneousAction('unload_truck', t=Truck, l=Location)
t = unload_truck.parameter('t')
l = unload_truck.parameter('l')
unload_truck.add_precondition(Equals(l, loc_of_truck(t)))
unload_truck.add_effect(pat(l), SetUnion(pat(l), pin(t)))
unload_truck.add_effect(pin(t), set())

dump_trucks_problem.add_actions([move_truck, load_truck, unload_truck])

# --- Goals ---
dump_trucks_problem.add_goal(
    And(
        GT(SetCardinality(SetUnion(pin(t1), pin(t2))), 5),
        LT(SetCardinality(pin(t1)), SetCardinality(pin(t2)))
    )
)

# --- Compile and Solve ---
assert compilation in ['sc', 'sci', 'scin'], f"Unsupported compilation type: {compilation} for this domain!"

compilation_solving.compile_and_solve(dump_trucks_problem, solving, compilation)
