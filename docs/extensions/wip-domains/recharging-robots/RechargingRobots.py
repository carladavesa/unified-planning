from experiments import compilation_solving
from unified_planning.shortcuts import *

compilation = 'up'
solving = 'fast-downward'

dimension = 15
# ---------------------------------------------------- Problem ---------------------------------------------------------
recharging_robots = Problem('recharging_robots')

Location = UserType('Location')
Robot = UserType('Robot')

n1 = Object('n1', Node)

at = Fluent('at', ArrayType(dimension, ArrayType(dimension)), l=Location, r=Robot)
rotating = Fluent('rotating')

grid = Fluent('grid', ArrayType(dimension, ArrayType(columns, Tile)))

floor_tile.add_fluent(robot_has, default_initial_value=False)
floor_tile.add_fluent(available_colour, default_initial_value=False)
floor_tile.add_fluent(grid, default_initial_value=clear)


rotate = InstantaneousAction('rotate', n=Node, r=Rotation, fd=Direction, td=Direction)
n = rotate.parameter('n')
r = rotate.parameter('r')
fd = rotate.parameter('fd')
td = rotate.parameter('td')
rotate.add_precondition(Not(rotating))
rotate.add_precondition(available_colour(nc))
rotate.add_effect(robot_has(r, c), False)
rotate.add_effect(robot_has(r, nc), False)
folding.add_action(rotate)

paint_up = InstantaneousAction('paint_up', r=Robot, c=Colour, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = paint_up.parameter('r')
c = paint_up.parameter('c')
x = paint_up.parameter('x')
y = paint_up.parameter('y')
paint_up.add_precondition(robot_has(r, c))
paint_up.add_precondition(Equals(grid(x, y), r))
paint_up.add_precondition(Equals(grid(x-1, y), clear))
paint_up.add_effect(grid(x-1, y), c)
floor_tile.add_action(paint_up)

paint_down = InstantaneousAction('paint_down', r=Robot, c=Colour, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = paint_down.parameter('r')
c = paint_down.parameter('c')
x = paint_down.parameter('x')
y = paint_down.parameter('y')
paint_down.add_precondition(robot_has(r, c))
paint_down.add_precondition(Equals(grid(x, y), r))
paint_down.add_precondition(Equals(grid(x+1, y), clear))
paint_down.add_effect(grid(x+1, y), c)
floor_tile.add_action(paint_down)

up = InstantaneousAction('up', r=Robot, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = up.parameter('r')
x = up.parameter('x')
y = up.parameter('y')
up.add_precondition(Equals(grid(x, y), r))
up.add_precondition(Equals(grid(x-1, y), clear))
up.add_effect(grid(x-1, y), c)
up.add_effect(grid(x, y), clear)
floor_tile.add_action(up)

down = InstantaneousAction('down', r=Robot, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = down.parameter('r')
x = down.parameter('x')
y = down.parameter('y')
down.add_precondition(Equals(grid(x, y), r))
down.add_precondition(Equals(grid(x+1, y), clear))
down.add_effect(grid(x+1, y), c)
down.add_effect(grid(x, y), clear)
floor_tile.add_action(down)

left = InstantaneousAction('left', r=Robot, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = left.parameter('r')
x = left.parameter('x')
y = left.parameter('y')
left.add_precondition(Equals(grid(x, y), r))
left.add_precondition(Equals(grid(x, y-1), clear))
left.add_effect(grid(x, y-1), c)
left.add_effect(grid(x, y), clear)
floor_tile.add_action(left)

right = InstantaneousAction('right', r=Robot, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = right.parameter('r')
x = right.parameter('x')
y = right.parameter('y')
right.add_precondition(Equals(grid(x, y), r))
right.add_precondition(Equals(grid(x, y+1), clear))
right.add_effect(grid(x, y+1), c)
right.add_effect(grid(x, y), clear)
floor_tile.add_action(right)

for r in range(rows):
    for c in range(columns):
        floor_tile.add_goal(Not(Equals(grid[r][c], clear)))

costs: Dict[Action, Expression] = {
    change_colour: Int(5),
    paint_up: Int(2),
    paint_down: Int(2),
    up: Int(1),
    down: Int(1),
    left: Int(1),
    right: Int(1),
}
floor_tile.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(floor_tile, solving, compilation)