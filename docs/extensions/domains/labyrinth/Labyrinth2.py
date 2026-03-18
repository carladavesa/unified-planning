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

# ------------------------------------ Problem ---------------------------------------------------------
n = 4
n_cards = 16
instance = [[0,1,2,3],[9,5,7,4],[10,6,8,13],[12,11,14,15]]

# ------------------------------------ Problem ---------------------------------------------------------

labyrinth = Problem('labyrinth')

Direction = UserType("Direction")
N = Object("N", Direction)
S = Object("S", Direction)
E = Object("E", Direction)
W = Object("W", Direction)
labyrinth.add_objects([N, S, E, W])

card_at = Fluent('card_at', ArrayType(n, ArrayType(n, IntType(0, n_cards-1))))
labyrinth.add_fluent(card_at, default_initial_value=0)
labyrinth.set_initial_value(card_at, instance)

robot_at = Fluent('robot_at', IntType(0, n_cards-1))
labyrinth.add_fluent(robot_at, default_initial_value=0)
labyrinth.set_initial_value(robot_at, 0)

connections = Fluent("connections", ArrayType(n_cards, BoolType()), d=Direction)
labyrinth.add_fluent(connections, default_initial_value=True)

# ---------------------------------------- ACTIONS ----------------------------------------

move_north = InstantaneousAction('move_north', nfrom=IntType(0,n_cards), nto=IntType(0,n_cards),
                                 r=IntType(0, n-1), c=IntType(0, n-1))
nfrom = move_north.parameter('nfrom')
nto = move_north.parameter('nto')
r = move_north.parameter('r')
c = move_north.parameter('c')
move_north.add_precondition(Equals(nfrom, card_at[r][c]))
move_north.add_precondition(Equals(nto, card_at[r-1][c]))
move_north.add_precondition(Equals(robot_at, nfrom))
move_north.add_precondition(connections[nfrom](N))
move_north.add_precondition(connections[nto](S))
move_north.add_effect(robot_at, nto)
labyrinth.add_action(move_north)

move_south = InstantaneousAction('move_south', nfrom=IntType(0,n_cards), nto=IntType(0,n_cards),
                                 r=IntType(0, n-1), c=IntType(0, n-1))
nfrom = move_south.parameter('nfrom')
nto = move_south.parameter('nto')
r = move_south.parameter('r')
c = move_south.parameter('c')
move_south.add_precondition(Equals(nfrom, card_at[r][c]))
move_south.add_precondition(Equals(nto, card_at[r+1][c]))
move_south.add_precondition(Equals(robot_at, nfrom))
move_south.add_precondition(connections[nfrom](N))
move_south.add_precondition(connections[nto](S))
move_south.add_effect(robot_at, nto)
labyrinth.add_action(move_south)

move_east = InstantaneousAction('move_east', nfrom=IntType(0,n_cards), nto=IntType(0,n_cards),
                                 r=IntType(0, n-1), c=IntType(0, n-1))
nfrom = move_east.parameter('nfrom')
nto = move_east.parameter('nto')
r = move_east.parameter('r')
c = move_east.parameter('c')
move_east.add_precondition(Equals(nfrom, card_at[r][c]))
move_east.add_precondition(Equals(nto, card_at[r][c+1]))
move_east.add_precondition(Equals(robot_at, nfrom))
move_east.add_precondition(connections[nfrom](N))
move_east.add_precondition(connections[nto](S))
move_east.add_effect(robot_at, nto)
labyrinth.add_action(move_east)

move_west = InstantaneousAction('move_west', nfrom=IntType(0,n_cards), nto=IntType(0,n_cards),
                                 r=IntType(0, n-1), c=IntType(0, n-1))
nfrom = move_west.parameter('nfrom')
nto = move_west.parameter('nto')
r = move_west.parameter('r')
c = move_west.parameter('c')
move_west.add_precondition(Equals(nfrom, card_at[r][c]))
move_west.add_precondition(Equals(nto, card_at[r][c-1]))
move_west.add_precondition(Equals(robot_at, nfrom))
move_west.add_precondition(connections[nfrom](N))
move_west.add_precondition(connections[nto](S))
move_west.add_effect(robot_at, nto)
labyrinth.add_action(move_west)

rotate_col_up = InstantaneousAction('rotate_col_up', c=IntType(0, n-1), n_card=IntType(0,n_cards))
c = rotate_col_up.parameter('c')
n_card = rotate_col_up.parameter('n_card')
# a cap fila hi ha el robot
all_rows = RangeVariable('all_rows', 0, n - 1)
rotate_col_up.add_precondition(Equals(robot_at, n_card))
rotate_col_up.add_precondition(Forall(Not(Equals(n_card, card_at[all_rows][c])), all_rows))
rotated_rows = RangeVariable("rotated_rows", 1, n - 1)
rotate_col_up.add_effect(card_at[rotated_rows-1][c], card_at[rotated_rows][c], forall=[rotated_rows])
rotate_col_up.add_effect(card_at[n-1][c], card_at[0][c])
labyrinth.add_action(rotate_col_up)

rotate_col_down = InstantaneousAction('rotate_col_down', c=IntType(0, n-1), n_card=IntType(0,n_cards))
c = rotate_col_down.parameter('c')
n_card = rotate_col_down.parameter('n_card')
rotate_col_down.add_precondition(Equals(robot_at, n_card))
all_rows = RangeVariable("all_rows", 0, n - 1)
rotate_col_down.add_precondition(Forall(Not(Equals(n_card, card_at[all_rows][c])), all_rows))
rotated_rows = RangeVariable("rotated_rows", 1, n - 1)
rotate_col_down.add_effect(card_at[rotated_rows][c], card_at[rotated_rows-1][c], forall=[rotated_rows])
rotate_col_down.add_effect(card_at[0][c], card_at[n-1][c])
labyrinth.add_action(rotate_col_down)

rotate_row_left = InstantaneousAction('rotate_row_left', r=IntType(0, n-1), n_card=IntType(0,n_cards))
r = rotate_row_left.parameter('r')
n_card = rotate_row_left.parameter('n_card')
rotate_row_left.add_precondition(Equals(robot_at, n_card))
all_cols = RangeVariable("all_cols", 0, n - 1)
rotate_row_left.add_precondition(Forall(Not(Equals(n_card, card_at[r][all_cols])), all_cols))
rotated_cols = RangeVariable("rotated_cols", 0, n - 2)
rotate_row_left.add_effect(card_at[r][rotated_cols], card_at[r][rotated_cols+1], forall=[rotated_cols])
rotate_row_left.add_effect(card_at[r][n-1], card_at[r][0])
labyrinth.add_action(rotate_row_left)

rotate_row_right = InstantaneousAction('rotate_row_right', r=IntType(0, n-1), n_card=IntType(0,n_cards))
r = rotate_row_right.parameter('r')
n_card = rotate_row_right.parameter('n_card')
rotate_row_right.add_precondition(Equals(robot_at, n_card))
all_cols = RangeVariable("all_cols", 0, n - 1)
rotate_row_right.add_precondition(Forall(Not(Equals(n_card, card_at[r][all_cols])), all_cols))
rotated_cols = RangeVariable("rotated_cols", 1, n - 1)
rotate_row_right.add_effect(card_at[r][rotated_cols], card_at[r][rotated_cols-1], forall=[rotated_cols])
rotate_row_right.add_effect(card_at[r][0], card_at[r][n-1])
labyrinth.add_action(rotate_row_right)

labyrinth.add_goal(Equals(robot_at, card_at[n-1][n-1]))

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
