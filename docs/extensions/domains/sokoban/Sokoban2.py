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

instance = subprocess.run(['python3', '/Users/cds26/PycharmProjects/unified-planning/experiments/sokoban/read_instance.py', 'i_1'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
print(instance)
initial_state = eval(output[0].strip())
undefined_positions = eval(output[1].strip())
goal_positions = eval(output[2].strip())
rows = eval(output[3].strip())
columns = eval(output[4].strip())

# ---------------------------------------------------- Problem ---------------------------------------------------------
# Sokoban model with an action per direction of movement.
sokoban_problem = Problem('sokoban_problem')

Pattern = UserType('Pattern')
P = Object('P', Pattern)  # Player
E = Object('E', Pattern)  # Empty
B = Object('B', Pattern)  # Box
sokoban_problem.add_objects([P,E, B])

grid = Fluent('grid', ArrayType(rows, ArrayType(columns, Pattern)), undefined_positions=undefined_positions)
sokoban_problem.add_fluent(grid, default_initial_value=E)
for (r, c), v in initial_state.items():
    sokoban_problem.set_initial_value(grid[r][c], eval(v))

move_right = InstantaneousAction('move_right', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_right.parameter('r')
c = move_right.parameter('c')
move_right.add_precondition(Equals(grid[r][c], P))
move_right.add_precondition(Equals(grid[r][c+1], E))
move_right.add_effect(grid[r][c], E)
move_right.add_effect(grid[r][c+1], P)

move_left = InstantaneousAction('move_left', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_left.parameter('r')
c = move_left.parameter('c')
move_left.add_precondition(Equals(grid[r][c], P))
move_left.add_precondition(Equals(grid[r][c-1], E))
move_left.add_effect(grid[r][c], E)
move_left.add_effect(grid[r][c-1], P)

move_up = InstantaneousAction('move_up', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_up.parameter('r')
c = move_up.parameter('c')
move_up.add_precondition(Equals(grid[r][c], P))
move_up.add_precondition(Equals(grid[r-1][c], E))
move_up.add_effect(grid[r][c], E)
move_up.add_effect(grid[r-1][c], P)

move_down = InstantaneousAction('move_down', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_down.parameter('r')
c = move_down.parameter('c')
move_down.add_precondition(Equals(grid[r][c], P))
move_down.add_precondition(Equals(grid[r+1][c], E))
move_down.add_effect(grid[r][c], E)
move_down.add_effect(grid[r+1][c], P)

push_box_right = InstantaneousAction('push_box_right', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = push_box_right.parameter('r')
c = push_box_right.parameter('c')
push_box_right.add_precondition(Equals(grid[r][c], P))
push_box_right.add_precondition(Equals(grid[r][c+1], B))
push_box_right.add_precondition(Equals(grid[r][c+2], E))
push_box_right.add_effect(grid[r][c], E)
push_box_right.add_effect(grid[r][c+1], P)
push_box_right.add_effect(grid[r][c+2], B)

push_box_left = InstantaneousAction('push_box_left', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = push_box_left.parameter('r')
c = push_box_left.parameter('c')
push_box_left.add_precondition(Equals(grid[r][c], P))
push_box_left.add_precondition(Equals(grid[r][c-1], B))
push_box_left.add_precondition(Equals(grid[r][c-2], E))
push_box_left.add_effect(grid[r][c], E)
push_box_left.add_effect(grid[r][c-1], P)
push_box_left.add_effect(grid[r][c-2], B)

push_box_up = InstantaneousAction('push_box_up', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = push_box_up.parameter('r')
c = push_box_up.parameter('c')
push_box_up.add_precondition(Equals(grid[r][c], P))
push_box_up.add_precondition(Equals(grid[r-1][c], B))
push_box_up.add_precondition(Equals(grid[r-2][c], E))
push_box_up.add_effect(grid[r][c], E)
push_box_up.add_effect(grid[r-1][c], P)
push_box_up.add_effect(grid[r-2][c], B)

push_box_down = InstantaneousAction('push_box_down', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = push_box_down.parameter('r')
c = push_box_down.parameter('c')
push_box_down.add_precondition(Equals(grid[r][c], P))
push_box_down.add_precondition(Equals(grid[r+1][c], B))
push_box_down.add_precondition(Equals(grid[r+2][c], E))
push_box_down.add_effect(grid[r][c], E)
push_box_down.add_effect(grid[r+1][c], P)
push_box_down.add_effect(grid[r+2][c], B)

sokoban_problem.add_actions([move_right, move_left, move_up, move_down, push_box_right, push_box_left, push_box_up, push_box_down])

# Goal
for r,c in goal_positions:
    sokoban_problem.add_goal(Equals(grid[r][c], B))

costs: Dict[Action, Expression] = {
    move_right: Int(0),
    move_left: Int(0),
    move_up: Int(0),
    move_down: Int(0),
    push_box_right: Int(1),
    push_box_left: Int(1),
    push_box_up: Int(1),
    push_box_down: Int(1),
}
sokoban_problem.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(sokoban_problem, solving, compilation)