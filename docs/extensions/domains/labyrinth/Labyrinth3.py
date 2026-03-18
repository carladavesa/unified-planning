from experiments import compilation_solving
from unified_planning.shortcuts import *
import argparse
import random

# python3 -m experiments.gh207.Labyrinth --n 2 --compilation up --solving fast-downward

# Parser
parser = argparse.ArgumentParser(description="Solve Labyrinth Problem")
parser.add_argument('--n', type=str, help='Size of the puzzle')
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# ------------------------------------ Instance ---------------------------------------------------------
# 4_5_4
n = 4
n_cards = n*n
instance = [[0, 6, 7, 3], [5, 9, 10, 4], [8, 13, 14, 11], [12, 1, 2, 15]]
paths = [[{'W', 'E'}, {'S', 'W'}, {'N', 'W', 'S', 'E'}, {'N', 'W', 'E'}], [{'N', 'W', 'E'}, {'S', 'W'}, {'N', 'S'}, {'S', 'E'}], [{'N', 'S', 'E'}, {'S', 'W'}, {'N', 'S', 'E'}, {'W', 'E'}], [{'N', 'W'}, {'N', 'W', 'S'}, {'N', 'S'}, {'S', 'W'}]]

# ------------------------------------ Problem ---------------------------------------------------------

labyrinth = Problem('labyrinth')

Card = UserType("Card")
Direction = UserType("Direction")
N = Object("N", Direction)
S = Object("S", Direction)
E = Object("E", Direction)
W = Object("W", Direction)
labyrinth.add_objects([N, S, E, W])
labyrinth.add_objects([Object(f"card_{i}", Card) for i in range(n_cards)])
card_0 = labyrinth.object('card_0')
card_15 = labyrinth.object('card_15')

card_at = Fluent('card_at', ArrayType(n, ArrayType(n)), c=Card)
labyrinth.add_fluent(card_at, default_initial_value=False)

robot_at = Fluent('robot_at', c=Card)
labyrinth.add_fluent(robot_at, default_initial_value=False)
labyrinth.set_initial_value(robot_at(card_0), True)

connections = Fluent('connections', c=Card, d=Direction)
labyrinth.add_fluent(connections, default_initial_value=False)
for r in range(n):
    for c in range(n):
        card_object = labyrinth.object(f'card_{str(instance[r][c])}')
        labyrinth.set_initial_value(card_at[r][c](card_object), True)
        for i in paths[r][c]:
            labyrinth.set_initial_value(connections(card_object,eval(i)), True)

# ---------------------------------------- ACTIONS ----------------------------------------

move_north = InstantaneousAction('move_north', x1=Card, x2=Card, r=IntType(0, n-1), c=IntType(0, n-1))
x1 = move_north.parameter('x1')
x2 = move_north.parameter('x2')
r = move_north.parameter('r')
c = move_north.parameter('c')
move_north.add_precondition(robot_at(x1))
move_north.add_precondition(card_at[r][c](x1))
move_north.add_precondition(connections(x1, N))
move_north.add_precondition(card_at[r-1][c](x2))
move_north.add_precondition(connections(x2, S))
move_north.add_effect(robot_at(x2), True)
move_north.add_effect(robot_at(x1), False)
labyrinth.add_action(move_north)

move_south = InstantaneousAction('move_south', x1=Card, x2=Card, r=IntType(0, n-1), c=IntType(0, n-1))
x1 = move_south.parameter('x1')
x2 = move_south.parameter('x2')
r = move_south.parameter('r')
c = move_south.parameter('c')
move_south.add_precondition(robot_at(x1))
move_south.add_precondition(card_at[r][c](x1))
move_south.add_precondition(connections(x1, S))
move_south.add_precondition(card_at[r+1][c](x2))
move_south.add_precondition(connections(x2, N))
move_south.add_effect(robot_at(x2), True)
move_south.add_effect(robot_at(x1), False)
labyrinth.add_action(move_south)

move_east = InstantaneousAction('move_east', x1=Card, x2=Card, r=IntType(0, n-1), c=IntType(0, n-1))
x1 = move_east.parameter('x1')
x2 = move_east.parameter('x2')
r = move_east.parameter('r')
c = move_east.parameter('c')
move_east.add_precondition(robot_at(x1))
move_east.add_precondition(card_at[r][c](x1))
move_east.add_precondition(connections(x1, E))
move_east.add_precondition(card_at[r][c+1](x2))
move_east.add_precondition(connections(x2, W))
move_east.add_effect(robot_at(x2), True)
move_east.add_effect(robot_at(x1), False)
labyrinth.add_action(move_east)

move_west = InstantaneousAction('move_west', x1=Card, x2=Card, r=IntType(0, n-1), c=IntType(0, n-1))
x1 = move_west.parameter('x1')
x2 = move_west.parameter('x2')
r = move_west.parameter('r')
c = move_west.parameter('c')
move_west.add_precondition(robot_at(x1))
move_west.add_precondition(card_at[r][c](x1))
move_west.add_precondition(connections(x1, W))
move_west.add_precondition(card_at[r][c-1](x2))
move_west.add_precondition(connections(x2, E))
move_west.add_effect(robot_at(x2), True)
move_west.add_effect(robot_at(x1), False)
labyrinth.add_action(move_west)

rotate_col_up = InstantaneousAction('rotate_col_up', c=IntType(0, n-1))
c = rotate_col_up.parameter('c')
# a cap fila hi ha el robot
all_rows = RangeVariable('all_rows', 0, n - 1)
x = Variable('x', Card)
rotate_col_up.add_precondition(Forall(Implies(card_at[all_rows][c](x), Not(robot_at(x))), all_rows, x))
rotated_rows = RangeVariable("rotated_rows", 1, n - 1)
rotate_col_up.add_effect(card_at[rotated_rows-1][c](x), True, condition=card_at[rotated_rows][c](x), forall=[rotated_rows, x])
rotate_col_up.add_effect(card_at[rotated_rows-1][c](x), False, condition=card_at[rotated_rows-1][c](x), forall=[rotated_rows, x])
rotate_col_up.add_effect(card_at[n-1][c](x), True, condition=card_at[0][c](x), forall=[x])
rotate_col_up.add_effect(card_at[n-1][c](x), False, condition=card_at[n-1][c](x), forall=[x])
labyrinth.add_action(rotate_col_up)

rotate_col_down = InstantaneousAction('rotate_col_down', c=IntType(0, n-1))
c = rotate_col_down.parameter('c')
# a cap fila hi ha el robot
all_rows = RangeVariable('all_rows', 0, n - 1)
x = Variable('x', Card)
rotate_col_down.add_precondition(Forall(Implies(card_at[all_rows][c](x), Not(robot_at(x))), all_rows, x))
rotated_rows = RangeVariable("rotated_rows", 1, n - 1)
rotate_col_down.add_effect(card_at[rotated_rows][c](x), True, condition=card_at[rotated_rows-1][c](x), forall=[rotated_rows, x])
rotate_col_down.add_effect(card_at[rotated_rows][c](x), False, condition=card_at[rotated_rows][c](x), forall=[rotated_rows, x])
rotate_col_down.add_effect(card_at[0][c](x), True, condition=card_at[n-1][c](x), forall=[x])
rotate_col_down.add_effect(card_at[0][c](x), False, condition=card_at[0][c](x), forall=[x])
labyrinth.add_action(rotate_col_down)

rotate_row_left = InstantaneousAction('rotate_row_left', r=IntType(0, n-1))
r = rotate_row_left.parameter('r')
# a cap fila hi ha el robot
all_cols = RangeVariable("all_cols", 0, n - 1)
x = Variable('x', Card)
rotate_row_left.add_precondition(Forall(Implies(card_at[r][all_cols](x), Not(robot_at(x))), all_cols, x))
rotated_cols = RangeVariable("rotated_cols", 0, n - 2)
rotate_row_left.add_effect(card_at[r][rotated_cols](x), True, condition=card_at[r][rotated_cols+1](x), forall=[rotated_cols, x])
rotate_row_left.add_effect(card_at[r][rotated_cols](x), False, condition=card_at[r][rotated_cols](x), forall=[rotated_cols, x])
rotate_row_left.add_effect(card_at[r][n-1](x), True, condition=card_at[r][0](x), forall=[x])
rotate_row_left.add_effect(card_at[r][n-1](x), False, condition=card_at[r][n-1](x), forall=[x])
labyrinth.add_action(rotate_row_left)

rotate_row_right = InstantaneousAction('rotate_row_right', r=IntType(0, n-1))
r = rotate_row_right.parameter('r')
# a cap fila hi ha el robot
all_cols = RangeVariable("all_cols", 0, n - 1)
x = Variable('x', Card)
rotate_row_right.add_precondition(Forall(Implies(card_at[r][all_cols](x), Not(robot_at(x))), all_cols, x))
rotated_cols = RangeVariable("rotated_cols", 1, n - 1)
rotate_row_right.add_effect(card_at[r][rotated_cols](x), True, condition=card_at[r][rotated_cols-1](x), forall=[rotated_cols, x])
rotate_row_right.add_effect(card_at[r][rotated_cols](x), False, condition=card_at[r][rotated_cols](x), forall=[rotated_cols, x])
rotate_row_right.add_effect(card_at[r][0](x), True, condition=card_at[r][n-1](x), forall=[x])
rotate_row_right.add_effect(card_at[r][0](x), False, condition=card_at[r][0](x), forall=[x])
labyrinth.add_action(rotate_row_right)

labyrinth.add_goal(
    And(robot_at(card_15), card_at[n-1][n-1](card_15), connections(card_15, S))
)


costs: Dict[Action, Expression] = {
    move_west: Int(1),
    move_north: Int(1),
    move_south: Int(1),
    move_east: Int(1),
    rotate_col_up: Int(1),
    rotate_col_down: Int(1),
    rotate_row_left: Int(1),
    rotate_row_right: Int(1),
}
labyrinth.add_quality_metric(MinimizeActionCosts(costs))

compilation_solving.compile_and_solve(labyrinth, solving, compilation)