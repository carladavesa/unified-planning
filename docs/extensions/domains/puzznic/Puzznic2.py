import subprocess
from unified_planning.shortcuts import *
from experiments import compilation_solving
import argparse

# python -m experiments.puzznic.Puzznic --compilation integers --solving symk

# Parser
parser = argparse.ArgumentParser(description="Solve Puzznic")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

instance_path = f'/Users/cds26/PycharmProjects/unified-planning/experiments/puzznic/read_instance.py'
instance = subprocess.run(['python3', instance_path, 'puzznic20'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
initial_state = eval(output[0].strip())
undefined  = eval(output[1].strip())
rows = eval(output[2].strip())
columns = eval(output[3].strip())

n_blocks = rows*columns - len(undefined)
# ------------------------------------------------ Problem -------------------------------------------------------------

puzznic_problem = unified_planning.model.Problem('puzznic_problem')

Pattern = UserType('Pattern')

B = Object('B', Pattern) # Blue
Y = Object('Y', Pattern) # Yellow
G = Object('G', Pattern) # Green
R = Object('R', Pattern) # Red
L = Object('L', Pattern) # Lightblue
O = Object('O', Pattern) # Orange
V = Object('V', Pattern) # Violet
P = Object('P', Pattern) # Pink
C = Object('C', Pattern) # Coal

patterned = Fluent('patterned', ArrayType(rows, ArrayType(columns)), p=Pattern, undefined_positions=undefined)
puzznic_problem.add_fluent(patterned, default_initial_value=False)

for (r, c), p in initial_state.items():
    if not puzznic_problem.has_object(str(p)):
        puzznic_problem.add_object(eval(p))
    puzznic_problem.set_initial_value(patterned[r][c](eval(p)), True)

falling_flag = Fluent('falling_flag', DerivedBoolType())
puzznic_problem.add_fluent(falling_flag, default_initial_value=False)
matching_flag = Fluent('matching_flag', DerivedBoolType())
puzznic_problem.add_fluent(matching_flag, default_initial_value=False)

# ------------------------------------------------ Axioms -------------------------------------------------------------
# Falling
axiom_falling = Axiom('axiom_falling')
axiom_falling.set_head(falling_flag)
i = RangeVariable('i', 1, rows-1)
j = RangeVariable('j', 0, columns-1)
v = Variable('v', Pattern)
v_falling = Variable('v_falling', Pattern)
axiom_falling.add_body_condition(
    Exists(And(patterned[i-1][j](v_falling), Forall(Not(patterned[i][j](v)), v)), i,j, v_falling)
)
puzznic_problem.add_axiom(axiom_falling)

# Matching
axiom_matching = Axiom('axiom_matching')
axiom_matching.set_head(matching_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-2)
p = Variable('p', Pattern)
matching_horizontal = Exists(
    And(patterned[i][j](p), patterned[i][j + 1](p)), i,j,p
)
i = RangeVariable('i', 0, rows-2)
j = RangeVariable('j', 0, columns-1)
matching_vertical = Exists(
    And(patterned[i][j](p), patterned[i + 1][j](p)), i,j,p
)
axiom_matching.add_body_condition(
    Or(matching_horizontal, matching_vertical)
)
puzznic_problem.add_axiom(axiom_matching)

# ------------------------------------------------ Actions -------------------------------------------------------------
# Move Block
move_block = InstantaneousAction('move_block', r=IntType(0, rows - 1), c=IntType(0, columns - 1),
                                 m=IntType(-1, 1), p=Pattern)
r = move_block.parameter('r')
c = move_block.parameter('c')
m = move_block.parameter('m')
p = move_block.parameter('p')
move_block.add_precondition(Not(falling_flag))
move_block.add_precondition(Not(matching_flag))
move_block.add_precondition(Not(Equals(m, 0)))
move_block.add_precondition(patterned[r][c](p))
v = Variable('v', Pattern)
move_block.add_precondition(Forall(Not(patterned[r][c + m](v)), v))
move_block.add_effect(patterned[r][c](p), False)
move_block.add_effect(patterned[r][c + m](p), True)
puzznic_problem.add_action(move_block)

# Fall Block
fall_block = InstantaneousAction('fall_block', r=IntType(0, rows - 2), c=IntType(0, columns - 1),
                                 p=Pattern)
r = fall_block.parameter('r')
c = fall_block.parameter('c')
p = fall_block.parameter('p')
fall_block.add_precondition(falling_flag)
fall_block.add_precondition(patterned[r][c](p))
fall_block.add_precondition(Forall(Not(patterned[r+1][c](v)), v)) # ?
fall_block.add_effect(patterned[r][c](p), False)
fall_block.add_effect(patterned[r + 1][c](p), True)
puzznic_problem.add_action(fall_block)

# Match Blocks
matching_blocks = InstantaneousAction('matching_blocks')
matching_blocks.add_precondition(Not(falling_flag))
matching_blocks.add_precondition(matching_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-1)
matching_colour = Variable('matching_colour', Pattern)
v = Variable('v', Pattern)
matching_blocks.add_effect(patterned[i][j](v), False, condition=And(
    patterned[i][j](matching_colour),
    Or(patterned[i + 1][j](matching_colour),
       patterned[i - 1][j](matching_colour),
       patterned[i][j + 1](matching_colour),
       patterned[i][j - 1](matching_colour))
), forall=[i,j,matching_colour,v])
puzznic_problem.add_action(matching_blocks)

# ------------------------------------------------ Goal -------------------------------------------------------------
v = Variable('v', Pattern)
for i in range(rows):
    for j in range(columns):
        if (i,j) not in undefined:
            puzznic_problem.add_goal(Forall(Not(patterned[i][j](v)), v))

# ------------------------------------------------ Costs -------------------------------------------------------------
costs: Dict[Action, Expression] = {
    move_block: Int(1),
    matching_blocks: Int(0),
    fall_block: Int(0)
}
puzznic_problem.add_quality_metric(MinimizeActionCosts(costs))

problem = puzznic_problem

assert compilation in ['up', 'integers', 'ut-integers'], f"Unsupported compilation type: {compilation}"
compilation_solving.compile_and_solve(puzznic_problem, solving, compilation)