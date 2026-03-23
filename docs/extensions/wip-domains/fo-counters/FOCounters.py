import subprocess

from experiments import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Parser
parser = argparse.ArgumentParser(description="Solve First-Order Counters")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Read instance
instance_path = f'/Users/cds26/PycharmProjects/unified-planning/experiments/counters/FOread_instance.py'
instance = subprocess.run(['python3', instance_path, 'pfile5'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
max_int_n = eval(output[0].strip())
values = eval(output[1].strip())
rate_values = eval(output[2].strip())
counters = len(values)

# ---------------------------------------------------- Problem ---------------------------------------------------------

counters_problem = unified_planning.model.Problem('counters_problem')

Counter = UserType('Counter')

value = Fluent('value', IntType(0, max_int_n), c=Counter)
rate_value = Fluent('rate_value', IntType(0, max_int_n), c=Counter)
max_int = Fluent('max_int', IntType(max_int_n, max_int_n))
counters_problem.add_fluent(value)
counters_problem.add_fluent(rate_value)
counters_problem.add_fluent(max_int)

# ------------------------------------------------ Initialisation ------------------------------------------------------

counters_problem.set_initial_value(max_int, max_int_n)
for c in range(counters):
    c_object = Object(f'c{c}', Counter)
    counters_problem.add_object(c_object)
    print(value(c_object), values[c], value(c_object).type, type(values[c]))
    counters_problem.set_initial_value(value(c_object), values[c])
    counters_problem.set_initial_value(rate_value(c_object), rate_values[c])


# ---------------------------------------------------- Actions ---------------------------------------------------------

increment = unified_planning.model.InstantaneousAction('increment', c=Counter)
c = increment.parameter('c')
increment.add_precondition(LE(Plus(value(c), rate_value(c)), max_int))
increment.add_increase_effect(value(c), rate_value(c))

decrement = unified_planning.model.InstantaneousAction('decrement', c=Counter)
c = decrement.parameter('c')
decrement.add_precondition(GE(Minus(value(c), rate_value(c)), 0))
decrement.add_decrease_effect(value(c), rate_value(c))

increase_rate = unified_planning.model.InstantaneousAction('increase_rate', c=Counter)
c = increase_rate.parameter('c')
increase_rate.add_precondition(LE(Plus(rate_value(c), 1), 10))
increase_rate.add_increase_effect(rate_value(c), 1)

decrement_rate = unified_planning.model.InstantaneousAction('decrement_rate', c=Counter)
c = decrement_rate.parameter('c')
decrement_rate.add_precondition(GE(rate_value(c), 1))
decrement_rate.add_decrease_effect(rate_value(c), 1)

counters_problem.add_actions([increment, decrement, increase_rate, decrement_rate])

for c in range(counters-1):
    counters_problem.add_goal(
        LE(Plus(value(counters_problem.object(f'c{c}')), 1), value(counters_problem.object(f'c{c + 1}')))
    )

costs: Dict[Action, Expression] = {
    increment: Int(1),
    decrement: Int(1),
    increase_rate: Int(1),
    decrement_rate: Int(1)
}
counters_problem.add_quality_metric(MinimizeActionCosts(costs))
print( counters_problem)
compilation_solving.compile_and_solve(counters_problem, solving, compilation)