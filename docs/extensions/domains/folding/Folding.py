import argparse
import subprocess

from docs.extensions.domains import compilation_solving
from unified_planning.shortcuts import *

# Run: python -m experiments.folding.Folding --compilation ut-integers --solving fast-downward

# Parser
parser = argparse.ArgumentParser(description="Solve Folding")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')
args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Read instance
instance_path = f'/Users/cds26/PycharmProjects/unified-planning/docs/extensions/domains/folding/read_instance.py'
instance = subprocess.run(['python3', instance_path, 'o01'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
r = eval(eval(output[0].strip())[0].strip())
c = eval(eval(output[0].strip())[1].strip())
nodes = len(eval(output[1].strip()))
initial_state = eval(output[1].strip())
goal_state = eval(output[2].strip())

# ---------------------------------------------------- Problem ---------------------------------------------------------
folding_problem = Problem('folding_problem')

rows = Fluent('rows', ArrayType(nodes, IntType(0, r-1)))
cols = Fluent('cols', ArrayType(nodes, IntType(0, c-1)))

folding_problem.add_fluent(rows, default_initial_value=0)
folding_problem.add_fluent(cols, default_initial_value=0)

for i, s in enumerate(initial_state):
    folding_problem.set_initial_value(rows[i], s[0])
    folding_problem.set_initial_value(cols[i], s[1])

rotate_clockwise = InstantaneousAction('rotate_clockwise', n=IntType(0, nodes-2))
n = rotate_clockwise.parameter('n')
g = RangeVariable('g', n+1, nodes-1)
b = RangeVariable('b', 0, n)
for i in range(nodes):
    for j in range(i + 1, nodes):
        if j == i + 1:
            # Consecutive nodes must be adjacent
            rotate_clockwise.add_precondition(
                Or(
                    And(
                        Equals(rows[j], rows[i]),
                        Or(Equals(cols[j], cols[i] + 1), Equals(cols[j] + 1, cols[i]))
                    ),
                    And(
                        Equals(cols[j], cols[i]),
                        Or(Equals(rows[j], rows[i] + 1), Equals(rows[j] + 1, rows[i]))
                    )
                )
            )
        else:
            # No consecutive nodes: no collision
            rotate_clockwise.add_precondition(
                Or(Not(Equals(rows[i], rows[j])), Not(Equals(cols[i], cols[j])))
            )
# All nodes that are being moved (g), are not in the same position as the ones still (b)
next_row = rows[n] - cols[n] + cols[g]
next_col = cols[n] + rows[n] - rows[g]
rotate_clockwise.add_precondition(Forall(
    Or(Not(Equals(rows[b], next_row)),
       Not(Equals(cols[b], next_col))), g,b))

rotate_clockwise.add_effect(rows[g], next_row, forall=[g])
rotate_clockwise.add_effect(cols[g], next_col, forall=[g])
folding_problem.add_action(rotate_clockwise)

rotate_counter_clockwise = InstantaneousAction('rotate_counter_clockwise', n=IntType(0, nodes-1))
n = rotate_counter_clockwise.parameter('n')
g = RangeVariable('g', n+1, nodes-1)
b = RangeVariable('b', 0, n)
for i in range(nodes):
    for j in range(i + 1, nodes):
        if j == i + 1:
            # Consecutive nodes must be adjacent
            rotate_counter_clockwise.add_precondition(
                Or(
                    And(
                        Equals(rows[j], rows[i]),
                        Or(Equals(cols[j], cols[i] + 1), Equals(cols[j] + 1, cols[i]))
                    ),
                    And(
                        Equals(cols[j], cols[i]),
                        Or(Equals(rows[j], rows[i] + 1), Equals(rows[j] + 1, rows[i]))
                    )
                )
            )
        else:
            # No consecutive nodes: no collision
            rotate_counter_clockwise.add_precondition(
                Or(Not(Equals(rows[i], rows[j])), Not(Equals(cols[i], cols[j])))
            )
# All nodes that are being moved (g), are not in the same position as the ones still (b)
next_row = rows[n] + cols[n] - cols[g]
next_col = cols[n] - rows[n] + rows[g]
rotate_counter_clockwise.add_precondition(Forall(
    Or(Not(Equals(rows[b], next_row)),
       Not(Equals(cols[b], next_col))), g,b))

rotate_counter_clockwise.add_effect(rows[g], next_row, forall=[g])
rotate_counter_clockwise.add_effect(cols[g], next_col, forall=[g])
folding_problem.add_action(rotate_counter_clockwise)

for i, g in enumerate(goal_state):
    folding_problem.add_goal(Equals(rows[i], g[0]))
    folding_problem.add_goal(Equals(cols[i], g[1]))

costs: Dict[Action, Expression] = {
    rotate_clockwise: Int(1),
    rotate_counter_clockwise: Int(1),
}

folding_problem.add_quality_metric(MinimizeActionCosts(costs))
#assert compilation in ['integers', 'ut-integers', 'logarithmic'], f"Unsupported compilation type: {compilation}"
compilation_solving.compile_and_solve(folding_problem, solving, compilation)