import subprocess
from experiments import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Parser
parser = argparse.ArgumentParser(description="Solve Sokoban")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

instance = subprocess.run(['python3', '/Users/cds26/PycharmProjects/unified-planning/experiments/sokoban/read_instance.py', 'i_2'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
print(output)
initial_state = eval(output[0].strip())
undefined_positions = eval(output[1].strip())
goal_positions = eval(output[2].strip())
rows = eval(output[3].strip())
columns = eval(output[4].strip())

# ---------------------------------------------------- Problem ---------------------------------------------------------
# Sokoban model with a predicate indicating the direction of movement, using Implies to define the resulting positions
# based on the chosen direction.
sokoban_problem = Problem('sokoban_problem')

Pattern = UserType('Pattern')
P = Object('P', Pattern)  # Player
B = Object('B', Pattern)  # Box
E = Object('E', Pattern)  # Empty
sokoban_problem.add_objects([P,B,E])

Direction = UserType('Direction')
right = Object('right', Direction)
left = Object('left', Direction)
up = Object('up', Direction)
down = Object('down', Direction)
sokoban_problem.add_objects([right, left, up, down])

grid = Fluent('grid', ArrayType(rows, ArrayType(columns, Pattern)), undefined_positions=undefined_positions)
sokoban_problem.add_fluent(grid, default_initial_value=E)
for (r, c), v in initial_state.items():
    sokoban_problem.set_initial_value(grid[r][c], eval(v))

move = InstantaneousAction('move', d=Direction, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move.parameter('r')
c = move.parameter('c')
d = move.parameter('d')
move.add_precondition(Equals(grid[r][c], P))
move.add_precondition(And(
    Implies(Equals(d, right), Equals(grid[r][c+1], E)),
    Implies(Equals(d, left), Equals(grid[r][c-1], E)),
    Implies(Equals(d, up), Equals(grid[r-1][c], E)),
    Implies(Equals(d, down), Equals(grid[r+1][c], E)),
))
move.add_effect(grid[r][c], E)
move.add_effect(grid[r][c+1], P, Equals(d, right))
move.add_effect(grid[r][c-1], P, Equals(d, left))
move.add_effect(grid[r-1][c], P, Equals(d, up))
move.add_effect(grid[r+1][c], P, Equals(d, down))

push_box = InstantaneousAction('push_box', d=Direction, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = push_box.parameter('r')
c = push_box.parameter('c')
d = push_box.parameter('d')
push_box.add_precondition(Equals(grid[r][c], P))
push_box.add_precondition(And(
    Implies(Equals(d, right), And(Equals(grid[r][c+1], B), Equals(grid[r][c+2], E))),
    Implies(Equals(d, left), And(Equals(grid[r][c-1], B), Equals(grid[r][c-2], E))),
    Implies(Equals(d, up), And(Equals(grid[r-1][c], B), Equals(grid[r-2][c], E))),
    Implies(Equals(d, down), And(Equals(grid[r+1][c], B), Equals(grid[r+2][c], E)))
))
push_box.add_effect(grid[r][c], E)
push_box.add_effect(grid[r][c+1], P, Equals(d, right))
push_box.add_effect(grid[r][c+2], B, Equals(d, right))
push_box.add_effect(grid[r][c-1], P, Equals(d, left))
push_box.add_effect(grid[r][c-2], B, Equals(d, left))
push_box.add_effect(grid[r-1][c], P, Equals(d, up))
push_box.add_effect(grid[r-2][c], B, Equals(d, up))
push_box.add_effect(grid[r+1][c], P, Equals(d, down))
push_box.add_effect(grid[r+2][c], B, Equals(d, down))

sokoban_problem.add_actions([move, push_box])

# Goal
for r,c in goal_positions:
    sokoban_problem.add_goal(Equals(grid[r][c], B))

costs: Dict[Action, Expression] = {
    move: Int(0),
    push_box: Int(1),
}
sokoban_problem.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(sokoban_problem, solving, compilation)