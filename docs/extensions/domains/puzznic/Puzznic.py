import subprocess
from ast import literal_eval
from unified_planning.shortcuts import *
from docs.extensions.domains import compilation_solving
import argparse
import sys

# Run: python -m docs.extensions.domains.puzznic.Puzznic --compilation up --solving fast-downward

# --- Parser ---
parser = argparse.ArgumentParser(description="Solve Puzznic")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# --- Instance ---
repo_root = Path(__file__).resolve().parents[4]
instance_path = repo_root / 'experiments' / 'puzznic' / 'read_instance.py'
instance = subprocess.run([sys.executable, str(instance_path), 'puzznic1'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
initial_state = literal_eval(output[0].strip())
undefined  = literal_eval(output[1].strip())
rows = literal_eval(output[2].strip())
columns = literal_eval(output[3].strip())

n_blocks = rows*columns - len(undefined)

# --- Problem ---
puzznic_problem = Problem('puzznic_problem')

Pattern = UserType('Pattern')

F = Object('F', Pattern) # Free
B = Object('B', Pattern) # Blue
Y = Object('Y', Pattern) # Yellow
G = Object('G', Pattern) # Green
R = Object('R', Pattern) # Red
L = Object('L', Pattern) # Lightblue
O = Object('O', Pattern) # Orange
V = Object('V', Pattern) # Violet
P = Object('P', Pattern) # Pink
C = Object('C', Pattern) # Coal
pattern_by_symbol = {'F': F, 'B': B, 'Y': Y, 'G': G, 'R': R, 'L': L, 'O': O, 'V': V, 'P': P, 'C': C}

puzznic_problem.add_object(F)

patterned = Fluent('patterned', ArrayType(rows, ArrayType(columns)), p=Pattern, undefined_positions=undefined)
puzznic_problem.add_fluent(patterned, default_initial_value=False)

for (r, c), p in initial_state.items():
    pattern_obj = pattern_by_symbol[p]
    if not puzznic_problem.has_object(p):
        puzznic_problem.add_object(pattern_obj)
    puzznic_problem.set_initial_value(patterned[r][c](pattern_obj), True)

for r in range(rows):
    for c in range(columns):
        if (r,c) not in initial_state.keys() and (r,c) not in undefined:
            puzznic_problem.set_initial_value(patterned[r][c](F), True)

falling_flag = Fluent('falling_flag', DerivedBoolType())
puzznic_problem.add_fluent(falling_flag, default_initial_value=False)
matching_flag = Fluent('matching_flag', DerivedBoolType())
puzznic_problem.add_fluent(matching_flag, default_initial_value=False)

# --- Axioms ---
# Falling
axiom_falling = Axiom('axiom_falling')
axiom_falling.set_head(falling_flag)
i = RangeVariable('i', 1, rows-1)
j = RangeVariable('j', 0, columns-1)
axiom_falling.add_body_condition(
    Exists(And(Not(patterned[i-1][j](F)), patterned[i][j](F)), i,j)
)
puzznic_problem.add_axiom(axiom_falling)

# Matching
axiom_matching = Axiom('axiom_matching')
axiom_matching.set_head(matching_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-2)
p = Variable('p', Pattern)
matching_horizontal = Exists(
    And(patterned[i][j](p), patterned[i][j + 1](p), Not(Equals(p,F))), i,j,p
)
i = RangeVariable('i', 0, rows-2)
j = RangeVariable('j', 0, columns-1)
p = Variable('p', Pattern)
matching_vertical = Exists(
    And(patterned[i][j](p), patterned[i + 1][j](p), Not(patterned[i][j](F))), i,j,p
)
axiom_matching.add_body_condition(
    Or(matching_horizontal, matching_vertical)
)
puzznic_problem.add_axiom(axiom_matching)

# --- Actions ---
# Move Block
move_block_right = InstantaneousAction('move_block_right', p=Pattern, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
p = move_block_right.parameter('p')
r = move_block_right.parameter('r')
c = move_block_right.parameter('c')
move_block_right.add_precondition(Not(falling_flag))
move_block_right.add_precondition(Not(matching_flag))
move_block_right.add_precondition(patterned[r][c](p))
move_block_right.add_precondition(Not(Equals(p, F)))
move_block_right.add_precondition(patterned[r][c + 1](F))
move_block_right.add_effect(patterned[r][c](F), True)
move_block_right.add_effect(patterned[r][c + 1](p), True)
move_block_right.add_effect(patterned[r][c](p), False)
move_block_right.add_effect(patterned[r][c + 1](F), Or(patterned[r][c](p), patterned[r][c+1](p)))
puzznic_problem.add_action(move_block_right)

move_block_left = InstantaneousAction('move_block_left', p=Pattern, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
p = move_block_left.parameter('p')
r = move_block_left.parameter('r')
c = move_block_left.parameter('c')
move_block_left.add_precondition(Not(falling_flag))
move_block_left.add_precondition(Not(matching_flag))
move_block_left.add_precondition(patterned[r][c](p))
move_block_left.add_precondition(Not(Equals(p, F)))
move_block_left.add_precondition(patterned[r][c - 1](F))
move_block_left.add_effect(patterned[r][c](F), True)
move_block_left.add_effect(patterned[r][c - 1](p), True)
move_block_left.add_effect(patterned[r][c](p), False)
move_block_left.add_effect(patterned[r][c - 1](F), False)
puzznic_problem.add_action(move_block_left)

# Fall Block
fall_block = InstantaneousAction('fall_block', p=Pattern, r=IntType(0, rows - 2), c=IntType(0, columns - 1))
p = fall_block.parameter('p')
r = fall_block.parameter('r')
c = fall_block.parameter('c')
fall_block.add_precondition(falling_flag)
fall_block.add_precondition(patterned[r][c](p))
fall_block.add_precondition(Not(Equals(p, F)))
fall_block.add_precondition(patterned[r + 1][c](F))
fall_block.add_effect(patterned[r][c](F), True)
fall_block.add_effect(patterned[r + 1][c](p), True)
fall_block.add_effect(patterned[r][c](p), False)
fall_block.add_effect(patterned[r + 1][c](F), False)
puzznic_problem.add_action(fall_block)

# Match Blocks
matching_blocks = InstantaneousAction('matching_blocks')
matching_blocks.add_precondition(Not(falling_flag))
matching_blocks.add_precondition(matching_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-1)
p = Variable('p', Pattern)
matching_blocks.add_effect(patterned[i][j](F), True, condition=And(
    Not(Equals(p, F)),
    patterned[i][j](p),
    Or(patterned[i + 1][j](p), patterned[i - 1][j](p), patterned[i][j + 1](p), patterned[i][j - 1](p))
), forall=[i,j,p])
puzznic_problem.add_action(matching_blocks)

# --- Goals ---
for i in range(rows):
    for j in range(columns):
        if (i,j) not in undefined:
            puzznic_problem.add_goal(patterned[i][j](F))

# --- Costs ---
costs: Dict[Action, Expression] = {
    move_block_right: Int(1),
    move_block_left: Int(1),
    matching_blocks: Int(0),
    fall_block: Int(0)
}
puzznic_problem.add_quality_metric(MinimizeActionCosts(costs))

# --- Compile and Solve ---
assert compilation in ['up'], f"Unsupported compilation type: {compilation} for this domain!"

compilation_solving.compile_and_solve(puzznic_problem, solving, compilation)