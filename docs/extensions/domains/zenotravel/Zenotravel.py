from unified_planning.shortcuts import *
from experiments import compilation_solving
import argparse
import math

# Run: python -m experiments.zenotravel.Zenotravel --compilation ut-integers --solving fast-downward

# Parser
parser = argparse.ArgumentParser(description="Solve Zenotravel")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# ------------------------------------------------ Problem -------------------------------------------------------------
zenotravel_problem = unified_planning.model.Problem('zenotravel_problem')

locatable = UserType('locatable')
aircraft = UserType('aircraft', locatable)
person = UserType('person', locatable)
city = UserType('city')

located = Fluent('located', l=locatable, c=city)
person_in = Fluent('in', p=person, a=aircraft)

fuel = Fluent('fuel', IntType(0,6000), a=aircraft)
distance = Fluent('distance', IntType(0, 810), c1=city, c2=city)
slow_burn = Fluent('slow_burn', IntType(4,4), a=aircraft)
fast_burn = Fluent('fast_burn', IntType(15,15), a=aircraft)
capacity = Fluent('capacity', IntType(6000,6000), a=aircraft)
total_fuel_used = Fluent('total_fuel_used', IntType(0,1000))
onboard = Fluent('onboard', IntType(0,6000), a=aircraft)
zoom_limit = Fluent('zoom_limit', IntType(8,8), a=aircraft)

zenotravel_problem.add_fluent(located, default_initial_value=False)
zenotravel_problem.add_fluent(person_in, default_initial_value=False)
zenotravel_problem.add_fluent(fuel)
zenotravel_problem.add_fluent(distance)
zenotravel_problem.add_fluent(slow_burn)
zenotravel_problem.add_fluent(fast_burn)
zenotravel_problem.add_fluent(capacity)
zenotravel_problem.add_fluent(total_fuel_used)
zenotravel_problem.add_fluent(onboard)
zenotravel_problem.add_fluent(zoom_limit)

# -------------------------------------------- Inisialisation ----------------------------------------------------------
plane1 = Object('plane1', aircraft)
person1 = Object('person1', person)
person2 = Object('person2', person)
person3 = Object('person3', person)
city0 = Object('city0', city)
city1 = Object('city1', city)
city2 = Object('city2', city)
zenotravel_problem.add_objects([plane1, person1, person2, person3, city0, city1, city2])

zenotravel_problem.set_initial_value(located(plane1, city0), True)
zenotravel_problem.set_initial_value(capacity(plane1), 6000)
zenotravel_problem.set_initial_value(fuel(plane1), 4000)
zenotravel_problem.set_initial_value(slow_burn(plane1), 4)
zenotravel_problem.set_initial_value(fast_burn(plane1), 15)
zenotravel_problem.set_initial_value(onboard(plane1), 0)
zenotravel_problem.set_initial_value(zoom_limit(plane1), 8)
zenotravel_problem.set_initial_value(located(person1, city0), True)
zenotravel_problem.set_initial_value(located(person2, city0), True)
zenotravel_problem.set_initial_value(located(person3, city1), True)
zenotravel_problem.set_initial_value(distance(city0, city0), 0)
zenotravel_problem.set_initial_value(distance(city0, city1), 678)
zenotravel_problem.set_initial_value(distance(city0, city2), 775)
zenotravel_problem.set_initial_value(distance(city1, city0), 678)
zenotravel_problem.set_initial_value(distance(city1, city1), 0)
zenotravel_problem.set_initial_value(distance(city1, city2), 810)
zenotravel_problem.set_initial_value(distance(city2, city0), 775)
zenotravel_problem.set_initial_value(distance(city2, city1), 810)
zenotravel_problem.set_initial_value(distance(city2, city2), 0)
zenotravel_problem.set_initial_value(total_fuel_used, 0)

# ------------------------------------------------ Actions -------------------------------------------------------------

board = InstantaneousAction('board', p=person, a=aircraft, c=city)
p = board.parameter('p')
a = board.parameter('a')
c = board.parameter('c')
board.add_precondition(located(p, c))
board.add_precondition(located(a, c))
board.add_effect(located(p, c), False)
board.add_effect(person_in(p, a), True)
board.add_increase_effect(onboard(a), 1)
zenotravel_problem.add_action(board)

debark = InstantaneousAction('debark', p=person, a=aircraft, c=city)
p = debark.parameter('p')
a = debark.parameter('a')
c = debark.parameter('c')
debark.add_precondition(person_in(p, a))
debark.add_precondition(located(a, c))
debark.add_effect(person_in(p, a), False)
debark.add_effect(located(p, c), True)
debark.add_decrease_effect(onboard(a), 1)
zenotravel_problem.add_action(debark)

fly_slow = InstantaneousAction('fly_slow', a=aircraft, c1=city, c2=city)
a = fly_slow.parameter('a')
c1 = fly_slow.parameter('c1')
c2 = fly_slow.parameter('c2')
fly_slow.add_precondition(located(a, c1))
fly_slow.add_precondition(GE(fuel(a), Times(distance(c1, c2), slow_burn(a))))
fly_slow.add_effect(located(a, c1), False)
fly_slow.add_effect(located(a, c2), True)
fly_slow.add_increase_effect(total_fuel_used, Times(distance(c1, c2), slow_burn(a)))
fly_slow.add_decrease_effect(fuel(a), Times(distance(c1, c2), slow_burn(a)))
zenotravel_problem.add_action(fly_slow)

fly_fast = InstantaneousAction('fly_fast', a=aircraft, c1=city, c2=city)
a = fly_fast.parameter('a')
c1 = fly_fast.parameter('c1')
c2 = fly_fast.parameter('c2')
fly_fast.add_precondition(located(a, c1))
fly_fast.add_precondition(GE(fuel(a), Times(distance(c1, c2), fast_burn(a))))
fly_fast.add_precondition(LE(onboard(a), zoom_limit(a)))
fly_fast.add_effect(located(a, c1), False)
fly_fast.add_effect(located(a, c2), True)
fly_fast.add_increase_effect(total_fuel_used, Times(distance(c1, c2), fast_burn(a)))
fly_fast.add_decrease_effect(fuel(a), Times(distance(c1, c2), fast_burn(a)))
zenotravel_problem.add_action(fly_fast)

refuel = InstantaneousAction('refuel', a=aircraft)
a = refuel.parameter('a')
refuel.add_precondition(GT(capacity(a), fuel(a)))
refuel.add_effect(fuel(a), capacity(a))
zenotravel_problem.add_action(refuel)

zenotravel_problem.add_goal(And(
    located(person1, city2),
    located(person2, city1),
    located(person3, city2)
))
compilation_solving.compile_and_solve(zenotravel_problem, solving, compilation)