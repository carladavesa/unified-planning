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
instance_path = f'/Users/cds26/PycharmProjects/unified-planning/docs/extensions/domains/counters/read_instance.py'
instance = subprocess.run(['python3', instance_path, 'pfile1'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
max_int_n = eval(output[0].strip())
values = eval(output[1].strip())
counters = len(values)

# ---------------------------------------------------- Problem ---------------------------------------------------------

counters_problem = unified_planning.model.Problem('counters_problem')

Counter = UserType('Counter')

value = Fluent('value', IntType(0, max_int_n), c=Counter)
max_int = Fluent('max_int', IntType(max_int_n, max_int_n))
counters_problem.add_fluent(value)
counters_problem.add_fluent(max_int)

# ------------------------------------------------ Initialisation ------------------------------------------------------

counters_problem.set_initial_value(max_int, max_int_n)
for c in range(counters):
    c_object = Object(f'c{c}', Counter)
    counters_problem.add_object(c_object)
    counters_problem.set_initial_value(value(c_object), values[c])

# ---------------------------------------------------- Actions ---------------------------------------------------------

increment = unified_planning.model.InstantaneousAction('increment', c=Counter)
c = increment.parameter('c')
increment.add_precondition(LE(Plus(value(c), 1), max_int_n))
increment.add_increase_effect(value(c), 2)

decrement = unified_planning.model.InstantaneousAction('decrement', c=Counter)
c = decrement.parameter('c')
decrement.add_precondition(GE(Minus(value(c), 1), 0))
decrement.add_decrease_effect(value(c), 1)

counters_problem.add_actions([increment, decrement])

for c in range(counters-1):
    counters_problem.add_goal(
        LE(Plus(value(counters_problem.object(f'c{c}')), 1), value(counters_problem.object(f'c{c + 1}')))
    )

#for c in range(counters-1):
#    counters_problem.add_goal(
#        Equals(value(counters_problem.object(f'c{c}')), c)
#    )

costs: Dict[Action, Expression] = {
    increment: Int(1),
    decrement: Int(1)
}
counters_problem.add_quality_metric(MinimizeActionCosts(costs))

compilation_solving.compile_and_solve(counters_problem, solving, compilation)