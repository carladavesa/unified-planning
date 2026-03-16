from docs.extensions.domains import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Run: python -m docs.extensions.domains.15-puzzle.15Puzzle --compilation uti --solving fast-downward

# --- Parser ---
parser = argparse.ArgumentParser(description="Solve 15-Puzzle")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# --- Instance ---
# 15-puzzle benchmark instance
initial_blocks = [[6,0,14,12],[1,15,9,10],[11,4,7,2],[8,3,5,13]]
goal_blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
n = 4
l = 15

# --- Problem ---
npuzzle_problem = Problem('npuzzle_problem')

puzzle = Fluent('puzzle', ArrayType(n, ArrayType(n, IntType(0,l))))
npuzzle_problem.add_fluent(puzzle, default_initial_value=0)
npuzzle_problem.set_initial_value(puzzle, initial_blocks)

# --- Actions ---
move_up = InstantaneousAction('move_up', r=IntType(0,n-1), c=IntType(0,n-1))
c = move_up.parameter('c')
r = move_up.parameter('r')
move_up.add_precondition(Equals(puzzle[r-1][c], 0))
move_up.add_precondition(Not(Equals(puzzle[r][c], 0)))
move_up.add_effect(puzzle[r-1][c], puzzle[r][c])
move_up.add_effect(puzzle[r][c], 0)

move_down = InstantaneousAction('move_down', r=IntType(0,n-1), c=IntType(0,n-1))
c = move_down.parameter('c')
r = move_down.parameter('r')
move_down.add_precondition(Equals(puzzle[r+1][c], 0))
move_down.add_precondition(Not(Equals(puzzle[r][c], 0)))
move_down.add_effect(puzzle[r+1][c], puzzle[r][c])
move_down.add_effect(puzzle[r][c], 0)

move_left = InstantaneousAction('move_left', r=IntType(0,n-1), c=IntType(0,n-1))
c = move_left.parameter('c')
r = move_left.parameter('r')
move_left.add_precondition(Equals(puzzle[r][c-1], 0))
move_left.add_precondition(Not(Equals(puzzle[r][c], 0)))
move_left.add_effect(puzzle[r][c-1], puzzle[r][c])
move_left.add_effect(puzzle[r][c], 0)

move_right = InstantaneousAction('move_right', r=IntType(0,n-1), c=IntType(0,n-1))
c = move_right.parameter('c')
r = move_right.parameter('r')
move_right.add_precondition(Equals(puzzle[r][c+1], 0))
move_right.add_precondition(Not(Equals(puzzle[r][c], 0)))
move_right.add_effect(puzzle[r][c+1], puzzle[r][c])
move_right.add_effect(puzzle[r][c], 0)

npuzzle_problem.add_actions([move_up, move_down, move_left, move_right])

# --- Goals ---
npuzzle_problem.add_goal(Equals(puzzle, goal_blocks))

# --- Costs ---
costs: Dict[Action, Expression] = {
    move_up: Int(1),
    move_down: Int(1),
    move_left: Int(1),
    move_right: Int(1)
}
npuzzle_problem.add_quality_metric(MinimizeActionCosts(costs))

# --- Compile and Solve ---
assert compilation in ['int', 'uti', 'log'], f"Unsupported compilation type: {compilation} for this domain!"

compilation_solving.compile_and_solve(npuzzle_problem, solving, compilation)
