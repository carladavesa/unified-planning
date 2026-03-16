from unified_planning.shortcuts import *
from docs.extensions.domains import compilation_solving
import argparse

# Run: python -m docs.extensions.domains.plotting.Plotting --compilation c --solving fast-downward

# --- Parser ---
parser = argparse.ArgumentParser(description="Solve Plotting")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# --- Instance (plt0_4_4_3_2) ---
instance = ['RRRR','RRRR','RRRR','RGGB']
remaining_blocks = 2

# --- Problem ---
plotting_problem = Problem('plotting_problem')

Colour = UserType('Colour')
R = Object('R', Colour) #RED
B = Object('B', Colour) #BLUE
G = Object('G', Colour) #GREEN
Y = Object('Y', Colour) #BLACK
O = Object('O', Colour) #ORANGE
V = Object('V', Colour) #VIOLET
W = Object('W', Colour) #WILDCARD
N = Object('N', Colour) #NONE
colour_by_symbol = {'R': R, 'B': B, 'G': G, 'Y': Y, 'O': O, 'V': V, 'W': W, 'N': N}
plotting_problem.add_objects([W, N])

initial_blocks = []
for i in instance:
    inside = []
    for j in i:
        colour_obj = colour_by_symbol[j]
        if not plotting_problem.has_object(j):
            plotting_problem.add_object(colour_obj)
        inside.append(colour_obj)
    initial_blocks.append(inside)

rows = len(initial_blocks)
columns = len(initial_blocks[0])
lr = rows-1
lc = columns-1

blocks = Fluent('blocks', ArrayType(rows, ArrayType(columns, Colour)))
hand = Fluent('hand', Colour)
plotting_problem.add_fluent(blocks, default_initial_value=N)
plotting_problem.add_fluent(hand, default_initial_value=W)
plotting_problem.set_initial_value(blocks, initial_blocks)

# --- Actions ---
shoot_partial_row = InstantaneousAction('shoot_partial_row', p=Colour, r=IntType(0, rows-1),
                                        l=IntType(0, columns-2))
p = shoot_partial_row.parameter('p')
r = shoot_partial_row.parameter('r')
l = shoot_partial_row.parameter('l')

shoot_partial_row.add_precondition(Not(Equals(p, N)))
shoot_partial_row.add_precondition(Not(Equals(p, W)))
shoot_partial_row.add_precondition(Or(Equals(hand, p), Equals(hand, W)))
# next block is never p, N nor W
shoot_partial_row.add_precondition(And(
    Not(Equals(blocks[r][l+1], p)), Not(Equals(blocks[r][l+1], N)), Not(Equals(blocks[r][l+1], W))))
b = RangeVariable('b', 0, l)
shoot_partial_row.add_precondition(Forall(Or(Equals(blocks[r][b], p), Equals(blocks[r][b], N)), b))
shoot_partial_row.add_precondition(Exists(Equals(blocks[r][b], p), b))

shoot_partial_row.add_effect(hand, blocks[r][l+1])
shoot_partial_row.add_effect(blocks[r][l+1], p)
shoot_partial_row.add_effect(blocks[0][b], N, forall=[b])
a = RangeVariable('a', 1, r)
shoot_partial_row.add_effect(blocks[a][b], blocks[a-1][b], forall=[a,b])

plotting_problem.add_action(shoot_partial_row)


shoot_column = InstantaneousAction('shoot_column', p=Colour, c=IntType(0, columns-1),
                                   l=IntType(0, rows))
p = shoot_column.parameter('p')
c = shoot_column.parameter('c')
l = shoot_column.parameter('l')
shoot_column.add_precondition(Not(Equals(p, N)))
shoot_column.add_precondition(Not(Equals(p, W)))
shoot_column.add_precondition(Or(Equals(hand, p), Equals(hand, W)))
shoot_column.add_precondition(Or(Equals(l, lr), And(
    Not(Equals(blocks[l+1][c], p)), Not(Equals(blocks[l+1][c], N)), Not(Equals(blocks[l+1][c], W)))))
b = RangeVariable('b', 0, l)
shoot_column.add_precondition(Forall(Or(Equals(blocks[b][c], p), Equals(blocks[b][c], N)), b))
shoot_column.add_precondition(Exists(Equals(blocks[b][c], p), b))

shoot_column.add_effect(hand, blocks[l+1][c], LT(l, lr))
shoot_column.add_effect(hand, p, Equals(l, lr))
shoot_column.add_effect(blocks[l+1][c], p, LT(l, lr))
shoot_column.add_effect(blocks[b][c], N, forall=[b])

plotting_problem.add_action(shoot_column)


shoot_row_and_column = InstantaneousAction('shoot_row_and_column', p=Colour, r=IntType(0, rows-2),
                                           l=IntType(1, rows-1))
p = shoot_row_and_column.parameter('p')
r = shoot_row_and_column.parameter('r')
l = shoot_row_and_column.parameter('l')
shoot_row_and_column.add_precondition(GT(l, r))
shoot_row_and_column.add_precondition(Not(Equals(p, N)))
shoot_row_and_column.add_precondition(Not(Equals(p, W)))
shoot_row_and_column.add_precondition(Or(Equals(hand, p), Equals(hand, W)))
c = RangeVariable('c', 0, lc)
shoot_row_and_column.add_precondition(Forall(Or(Equals(blocks[r][c], p), Equals(blocks[r][c], N)), c))
b = RangeVariable('b', r+1, l)
shoot_row_and_column.add_precondition(Forall(Or(Equals(blocks[b][lc], p), Equals(blocks[b][lc], N)), b))
shoot_row_and_column.add_precondition(Or(
    Exists(Equals(blocks[r][c], p), c),
    Exists(Equals(blocks[b][lc], p), b)
))
shoot_row_and_column.add_precondition(Or(Equals(l, lr), And(Not(Equals(blocks[l+1][lc], p)),
                                                            Not(Equals(blocks[l+1][lc], N)))))

shoot_row_and_column.add_effect(blocks[l+1][lc], p, LT(l,lr))
shoot_row_and_column.add_effect(hand, blocks[l+1][lc], LT(l,lr))
shoot_row_and_column.add_effect(hand, p, Equals(l,lr))
a = RangeVariable('a', 1, r)
c = RangeVariable('c', 0, lc-1)
shoot_row_and_column.add_effect(blocks[0][c], N, forall=[c])
shoot_row_and_column.add_effect(blocks[a][c], blocks[a-1][c], forall=[a,c])
b = RangeVariable('b', 0, r-1)
shoot_row_and_column.add_effect(blocks[l-b][lc], blocks[b][lc], forall=[b])
x = RangeVariable('x', 0, l-r)
shoot_row_and_column.add_effect(blocks[x][lc], N, forall=[x])

plotting_problem.add_action(shoot_row_and_column)


shoot_only_full_row = InstantaneousAction('shoot_only_full_row', p=Colour, r=IntType(0, rows-1))
p = shoot_only_full_row.parameter('p')
r = shoot_only_full_row.parameter('r')
shoot_only_full_row.add_precondition(Not(Equals(p, N)))
shoot_only_full_row.add_precondition(Not(Equals(p, W)))
shoot_only_full_row.add_precondition(Or(Equals(hand, p), Equals(hand, W)))
c = RangeVariable('c', 0, lc)
shoot_only_full_row.add_precondition(Forall(Or(Equals(blocks[r][c], p), Equals(blocks[r][c], N)), c))
shoot_only_full_row.add_precondition(Exists(Equals(blocks[r][c], p), c))
shoot_only_full_row.add_precondition(Or(Equals(r, lr), And(
    Not(Equals(blocks[r+1][lc], p)), Not(Equals(blocks[r+1][lc], N)), Not(Equals(blocks[r+1][lc], W))
)))

shoot_only_full_row.add_effect(blocks[r+1][lc], p, LT(r,lr))
shoot_only_full_row.add_effect(hand, blocks[r+1][lc], LT(r,lr))
shoot_only_full_row.add_effect(hand, p, Equals(r,lr))
a = RangeVariable('a', 1, r)
c = RangeVariable('c', 0, lc)
shoot_only_full_row.add_effect(blocks[0][c], N, forall=[c])
shoot_only_full_row.add_effect(blocks[a][c], blocks[a-1][c], forall=[a,c])

plotting_problem.add_action(shoot_only_full_row)

# --- Goals ---
rb = [Not(Equals(blocks[i][j], N)) for i in range(rows) for j in range(columns)]
plotting_problem.add_goal(LE(Count(rb), remaining_blocks))

# --- Costs ---
costs: Dict[Action, Expression] = {
    shoot_partial_row: Int(1),
    shoot_column: Int(1),
    shoot_only_full_row: Int(1),
    shoot_row_and_column: Int(1)
}
plotting_problem.add_quality_metric(MinimizeActionCosts(costs))

# --- Compile and Solve ---
assert compilation in ['c', 'ci', 'cin'], f"Unsupported compilation type: {compilation} for this domain!"

compilation_solving.compile_and_solve(plotting_problem, solving, compilation)