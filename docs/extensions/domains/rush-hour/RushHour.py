import math
from docs.extensions.domains import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Run: python -m docs.extensions.domains.rush-hour.RushHour --compilation up --solving fast-downward

# --- Parser ---
parser = argparse.ArgumentParser(description="Solve Rush Hour")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# --- Instance ---
# Benchmark instance
instance = 'ABBCooOAooCooOXXDoooOEFDGGUUEFHHPUUIIJJPKoooooPKo'
rows = columns = int(math.sqrt(len(instance)))
undefined = []
for i, char in enumerate(instance):
    r, c = divmod(i, columns)
    if char == 'x':
        undefined.append((r,c))
idx = instance.index('X')
row_goal = idx // rows

# --- Problem ---
# Vehicles can be cars (length 2), trucks (length 3), or 2x2 blocks.
rush_hour_problem = Problem('rush_hour_problem')

Vehicle = UserType('Vehicle')
none = Object('none', Vehicle)
X = Object('X', Vehicle)
rush_hour_problem.add_objects([none, X])
occupied = Fluent('occupied', ArrayType(rows, ArrayType(columns, Vehicle)), undefined_positions=undefined)
is_car = Fluent('is_car', v=Vehicle)
is_truck = Fluent('is_truck', v=Vehicle)
rush_hour_problem.add_fluent(occupied, default_initial_value=none)
rush_hour_problem.add_fluent(is_car, default_initial_value=False)
rush_hour_problem.add_fluent(is_truck, default_initial_value=False)
rush_hour_problem.set_initial_value(is_car(X), True)

for i, char in enumerate(instance):
    r, c = divmod(i, columns)
    if char == 'o':
        rush_hour_problem.set_initial_value(occupied[r][c], none)
    elif char != 'x':
        obj = Object(f'{char}', Vehicle)
        if not rush_hour_problem.has_object(char):
            rush_hour_problem.add_object(obj)
            rush_hour_problem.set_initial_value(is_car(obj), instance.count(char) == 2)
            rush_hour_problem.set_initial_value(is_truck(obj), instance.count(char) == 3)
        rush_hour_problem.set_initial_value(occupied[r][c], obj)

# --- Actions ---
move_car_right = InstantaneousAction('move_car_right', v=Vehicle, r=IntType(0,rows-1),
                                     c=IntType(0,columns-1))
v = move_car_right.parameter('v')
r = move_car_right.parameter('r')
c = move_car_right.parameter('c')
move_car_right.add_precondition(Not(Equals(v,none)))
move_car_right.add_precondition(is_car(v))
move_car_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v)))
move_car_right.add_precondition(Equals(occupied[r][c+2], none))

move_car_right.add_effect(occupied[r][c], none)
move_car_right.add_effect(occupied[r][c+2], v)


move_car_left = InstantaneousAction('move_car_left', v=Vehicle, r=IntType(0,rows-1),
                                    c=IntType(0,columns-1))
v = move_car_left.parameter('v')
r = move_car_left.parameter('r')
c = move_car_left.parameter('c')
move_car_left.add_precondition(Not(Equals(v,none)))
move_car_left.add_precondition(is_car(v))
move_car_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v)))
move_car_left.add_precondition(Equals(occupied[r][c-1], none))

move_car_left.add_effect(occupied[r][c-1], v)
move_car_left.add_effect(occupied[r][c+1], none)


move_car_up = InstantaneousAction('move_car_up', v=Vehicle, r=IntType(0,rows-1),
                                  c=IntType(0,columns-1))
v = move_car_up.parameter('v')
r = move_car_up.parameter('r')
c = move_car_up.parameter('c')
move_car_up.add_precondition(Not(Equals(v,none)))
move_car_up.add_precondition(is_car(v))
move_car_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v)))
move_car_up.add_precondition(Equals(occupied[r-1][c], none))

move_car_up.add_effect(occupied[r-1][c], v)
move_car_up.add_effect(occupied[r+1][c], none)


move_car_down = InstantaneousAction('move_car_down', v=Vehicle, r=IntType(0,rows-1),
                                    c=IntType(0,columns-1))
v = move_car_down.parameter('v')
r = move_car_down.parameter('r')
c = move_car_down.parameter('c')
move_car_down.add_precondition(Not(Equals(v,none)))
move_car_down.add_precondition(is_car(v))
move_car_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v)))
move_car_down.add_precondition(Equals(occupied[r+2][c], none))

move_car_down.add_effect(occupied[r][c], none)
move_car_down.add_effect(occupied[r+2][c], v)


move_truck_right = InstantaneousAction('move_truck_right', v=Vehicle, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_truck_right.parameter('v')
r = move_truck_right.parameter('r')
c = move_truck_right.parameter('c')
move_truck_right.add_precondition(Not(Equals(v,none)))
move_truck_right.add_precondition(is_truck(v))
move_truck_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v), Equals(occupied[r][c+2], v)))
move_truck_right.add_precondition(Equals(occupied[r][c+3], none))

move_truck_right.add_effect(occupied[r][c], none)
move_truck_right.add_effect(occupied[r][c+3], v)


move_truck_left = InstantaneousAction('move_truck_left', v=Vehicle, r=IntType(0,rows-1),
                                      c=IntType(0,columns-1))
v = move_truck_left.parameter('v')
r = move_truck_left.parameter('r')
c = move_truck_left.parameter('c')
move_truck_left.add_precondition(Not(Equals(v,none)))
move_truck_right.add_precondition(is_truck(v))
move_truck_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v), Equals(occupied[r][c+2], v)))
move_truck_left.add_precondition(Equals(occupied[r][c-1], none))

move_truck_left.add_effect(occupied[r][c+2], none)
move_truck_left.add_effect(occupied[r][c-1], v)


move_truck_up = InstantaneousAction('move_truck_up', v=Vehicle, r=IntType(0,rows-1),
                                    c=IntType(0,columns-1))
v = move_truck_up.parameter('v')
r = move_truck_up.parameter('r')
c = move_truck_up.parameter('c')
move_truck_up.add_precondition(Not(Equals(v,none)))
move_truck_right.add_precondition(is_truck(v))
move_truck_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v),
                                   Equals(occupied[r+2][c], v)))
move_truck_up.add_precondition(Equals(occupied[r-1][c], none))

move_truck_up.add_effect(occupied[r+2][c], none)
move_truck_up.add_effect(occupied[r-1][c], v)


move_truck_down = InstantaneousAction('move_truck_down', v=Vehicle, r=IntType(0,rows-1),
                                      c=IntType(0,columns-1))
v = move_truck_down.parameter('v')
r = move_truck_down.parameter('r')
c = move_truck_down.parameter('c')
move_truck_down.add_precondition(Not(Equals(v,none)))
move_truck_right.add_precondition(is_truck(v))
move_truck_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v),
                                     Equals(occupied[r+2][c], v)))
move_truck_down.add_precondition(Equals(occupied[r+3][c], none))

move_truck_down.add_effect(occupied[r][c], none)
move_truck_down.add_effect(occupied[r+3][c], v)


move_quad_right = InstantaneousAction('move_quad_right', v=Vehicle, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_quad_right.parameter('v')
r = move_quad_right.parameter('r')
c = move_quad_right.parameter('c')
move_quad_right.add_precondition(Not(Equals(v,none)))
move_truck_right.add_precondition(And(Not(is_truck(v)), Not(is_car(v))))
move_quad_right.add_precondition(And(
    Equals(occupied[r][c], v), Equals(occupied[r][c+1], v),
    Equals(occupied[r+1][c], v), Equals(occupied[r+1][c+1], v)
))
move_quad_right.add_precondition(Equals(occupied[r][c+2], none))
move_quad_right.add_precondition(Equals(occupied[r+1][c+2], none))

move_quad_right.add_effect(occupied[r][c], none)
move_quad_right.add_effect(occupied[r+1][c], none)
move_quad_right.add_effect(occupied[r][c+2], v)
move_quad_right.add_effect(occupied[r+1][c+2], v)


move_quad_left = InstantaneousAction('move_quad_left', v=Vehicle, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_quad_left.parameter('v')
r = move_quad_left.parameter('r')
c = move_quad_left.parameter('c')
move_quad_left.add_precondition(Not(Equals(v,none)))
move_quad_left.add_precondition(And(Not(is_truck(v)), Not(is_car(v))))
move_quad_left.add_precondition(And(
    Equals(occupied[r][c], v), Equals(occupied[r][c+1], v),
    Equals(occupied[r+1][c], v), Equals(occupied[r+1][c+1], v)
))
move_quad_left.add_precondition(Equals(occupied[r][c-1], none))
move_quad_left.add_precondition(Equals(occupied[r+1][c-1], none))

move_quad_left.add_effect(occupied[r][c+1], none)
move_quad_left.add_effect(occupied[r+1][c+1], none)
move_quad_left.add_effect(occupied[r][c-1], v)
move_quad_left.add_effect(occupied[r+1][c-1], v)

move_quad_up = InstantaneousAction('move_quad_up', v=Vehicle, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_quad_up.parameter('v')
r = move_quad_up.parameter('r')
c = move_quad_up.parameter('c')
move_quad_up.add_precondition(Not(Equals(v,none)))
move_quad_up.add_precondition(And(Not(is_truck(v)), Not(is_car(v))))
move_quad_up.add_precondition(And(
    Equals(occupied[r][c], v), Equals(occupied[r][c+1], v),
    Equals(occupied[r+1][c], v), Equals(occupied[r+1][c+1], v)
))
move_quad_up.add_precondition(Equals(occupied[r-1][c], none))
move_quad_up.add_precondition(Equals(occupied[r-1][c+1], none))

move_quad_up.add_effect(occupied[r+1][c], none)
move_quad_up.add_effect(occupied[r+1][c+1], none)
move_quad_up.add_effect(occupied[r-1][c], v)
move_quad_up.add_effect(occupied[r-1][c+1], v)


move_quad_down = InstantaneousAction('move_quad_down', v=Vehicle, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_quad_down.parameter('v')
r = move_quad_down.parameter('r')
c = move_quad_down.parameter('c')
move_quad_down.add_precondition(Not(Equals(v,none)))
move_quad_down.add_precondition(And(Not(is_truck(v)), Not(is_car(v))))
move_quad_down.add_precondition(And(

    Equals(occupied[r][c], v), Equals(occupied[r][c+1], v),
    Equals(occupied[r+1][c], v), Equals(occupied[r+1][c+1], v)
))
move_quad_down.add_precondition(Equals(occupied[r][c+2], none))
move_quad_down.add_precondition(Equals(occupied[r+1][c+2], none))

move_quad_down.add_effect(occupied[r][c], none)
move_quad_down.add_effect(occupied[r+1][c], none)
move_quad_down.add_effect(occupied[r][c+2], v)
move_quad_down.add_effect(occupied[r+1][c+2], v)


rush_hour_problem.add_actions([move_car_right, move_car_left, move_car_down, move_car_up,
                               move_truck_right, move_truck_left, move_truck_down, move_truck_up,
                               move_quad_right, move_quad_left, move_quad_down, move_quad_up])

# --- Goals ---
rush_hour_problem.add_goal(Equals(occupied[row_goal][columns-1], X))
rush_hour_problem.add_goal(Equals(occupied[row_goal][columns-2], X))

# --- Costs ---
costs: Dict[Action, Expression] = {
    move_car_left: Int(1),
    move_car_right: Int(1),
    move_car_up: Int(1),
    move_car_down: Int(1),
    move_truck_right: Int(1),
    move_truck_left: Int(1),
    move_truck_up: Int(1),
    move_truck_down: Int(1),
    move_quad_right: Int(1),
    move_quad_left: Int(1),
    move_quad_up: Int(1),
    move_quad_down: Int(1),
}
rush_hour_problem.add_quality_metric(MinimizeActionCosts(costs))

# --- Compile and Solve ---
assert compilation in ['up'], f"Unsupported compilation type: {compilation} for this domain!"

compilation_solving.compile_and_solve(rush_hour_problem, solving, compilation)