import argparse
import subprocess

from experiments import compilation_solving
from unified_planning.shortcuts import *

# Run: python -m experiments.ricochet-robots.RicochetRobots --compilation ut-integers --solving fast-downward

# Parser
parser = argparse.ArgumentParser(description="Solve Folding")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')
args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Read instance
instance_path = f'/Users/cds26/PycharmProjects/unified-planning/experiments/ricochet-robots/read_instance.py'
instance = subprocess.run(['python3', instance_path, 'p01'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
r = eval(eval(output[0].strip())[0].strip())
c = eval(eval(output[0].strip())[1].strip())
nodes = len(eval(output[1].strip()))
initial_state = eval(output[1].strip())
goal_state = eval(output[2].strip())

# ---------------------------------------------------- Problem ---------------------------------------------------------
ricochet_robots_problem = Problem('ricochet_robots_problem')

rows = Fluent('rows', ArrayType(nodes, IntType(0, r-1)))
cols = Fluent('cols', ArrayType(nodes, IntType(0, c-1)))

ricochet_robots_problem.add_fluent(rows, default_initial_value=0)
ricochet_robots_problem.add_fluent(cols, default_initial_value=0)

for i, s in enumerate(initial_state):
    ricochet_robots_problem.set_initial_value(rows[i], s[0])
    ricochet_robots_problem.set_initial_value(cols[i], s[1])



costs: Dict[Action, Expression] = {

}
ricochet_robots_problem.add_quality_metric(MinimizeActionCosts(costs))
#assert compilation in ['integers', 'ut-integers', 'logarithmic'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(ricochet_robots_problem, solving, compilation)