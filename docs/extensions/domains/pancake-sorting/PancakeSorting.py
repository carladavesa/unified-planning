from unified_planning.shortcuts import *
from docs.extensions.domains import compilation_solving
import argparse

# Run: python -m docs.extensions.domains.pancake-sorting.PancakeSorting --compilation int --solving fast-downward

# --- Parser ---
parser = argparse.ArgumentParser(description="Solve Pancake Sorting Numeric")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# --- Instance ---
# Benchmark instance
instance = [3,4,2,1,0]
n = len(instance)
lower_bound = 0
upper_bound = n-1

# --- Problem ---
pancake_problem = Problem('pancake_problem')

pancake = Fluent('pancake', ArrayType(n, IntType(lower_bound, upper_bound)))
pancake_problem.add_fluent(pancake, default_initial_value=lower_bound)
pancake_problem.set_initial_value(pancake, instance)

# --- Action ---
flip = InstantaneousAction('flip', f=IntType(1, n-1))
f = flip.parameter('f')
b = RangeVariable('b', 0, f)
flip.add_effect(pancake[b], pancake[f - b], forall=[b])
pancake_problem.add_action(flip)

# --- Goals ---
for i in range(n):
    pancake_problem.add_goal(Equals(pancake[i], i))

# --- Costs ---
costs: Dict[Action, Expression] = {
    flip: Int(1),
}
pancake_problem.add_quality_metric(MinimizeActionCosts(costs))

# --- Compile and Solve ---
assert compilation in ['uti', 'log', 'int'], f"Unsupported compilation type: {compilation} for this domain!"

compilation_solving.compile_and_solve(pancake_problem, solving, compilation)