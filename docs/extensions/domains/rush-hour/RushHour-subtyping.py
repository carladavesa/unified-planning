import math
from experiments import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Parser
parser = argparse.ArgumentParser(description="Solve Rush Hour")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Example
instance = 'AoooBooAoooBCCDoooOOODXXoEFoGGHHEFooIJKKLooIJooLo'
rows = columns = int(math.sqrt(len(instance)))
undefined = []
for i, char in enumerate(instance):
    r, c = divmod(i, columns)
    if char == 'x':
        undefined.append((r,c))
idx = instance.index('X')
row_goal = idx // rows

# ---------------------------------------------------- Problem ---------------------------------------------------------

rush_hour_problem = unified_planning.model.Problem('rush_hour_problem')

Vehicle = UserType('Vehicle')
Car = UserType('Car', Vehicle)
Truck = UserType('Truck', Vehicle)
Quad = UserType('Quad', Vehicle)
none = Object('none', Vehicle)
X = Object('X', Car)
rush_hour_problem.add_objects([none, X])
occupied = Fluent('occupied', ArrayType(rows, ArrayType(columns, Vehicle)), undefined_positions=undefined)
rush_hour_problem.add_fluent(occupied, default_initial_value=none)

for i, char in enumerate(instance):
    r, c = divmod(i, columns)
    if char == 'o':
        rush_hour_problem.set_initial_value(occupied[r][c], none)
    elif char != 'x':
        if instance.count(char) == 2:
            object_type = Car
        elif instance.count(char) == 3:
            object_type = Truck
        else:
            object_type = Quad
        obj = Object(f'{char}', object_type)
        if not rush_hour_problem.has_object(char):
            rush_hour_problem.add_object(obj)
        rush_hour_problem.set_initial_value(occupied[r][c], obj)

# ------------------------------------------------ Move Car Right ------------------------------------------------------
move_car_right = InstantaneousAction('move_car_right', v=Car, r=IntType(0,rows-1),
                                     c=IntType(0,columns-1))
v = move_car_right.parameter('v')
r = move_car_right.parameter('r')
c = move_car_right.parameter('c')
# vehicle is not none
move_car_right.add_precondition(Not(Equals(v,none)))
# it is located in the position
move_car_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v)))
# next position must be empty
move_car_right.add_precondition(Equals(occupied[r][c+2], none))

# control effects occupied positions
move_car_right.add_effect(occupied[r][c], none)
move_car_right.add_effect(occupied[r][c+2], v)

# ------------------------------------------------- Move Car Left ------------------------------------------------------
move_car_left = InstantaneousAction('move_car_left', v=Car, r=IntType(0,rows-1),
                                    c=IntType(0,columns-2))
v = move_car_left.parameter('v')
r = move_car_left.parameter('r')
c = move_car_left.parameter('c')
# vehicle is not none
move_car_left.add_precondition(Not(Equals(v,none)))
# is horizontal and it is located in the position
move_car_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v)))
# next position must be empty
move_car_left.add_precondition(Equals(occupied[r][c-1], none))

# control effects occupied positions
move_car_left.add_effect(occupied[r][c-1], v)
move_car_left.add_effect(occupied[r][c+1], none)

# ------------------------------------------------- Move Car Up ------------------------------------------------------
move_car_up = InstantaneousAction('move_car_up', v=Car, r=IntType(0,rows-1),
                                  c=IntType(0,columns-1))
v = move_car_up.parameter('v')
r = move_car_up.parameter('r')
c = move_car_up.parameter('c')
# vehicle is not none
move_car_up.add_precondition(Not(Equals(v,none)))
# is horizontal and it is located in the position
move_car_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v)))
# next position must be empty
move_car_up.add_precondition(Equals(occupied[r-1][c], none))

# control effects occupied positions
move_car_up.add_effect(occupied[r-1][c], v)
move_car_up.add_effect(occupied[r+1][c], none)

# ------------------------------------------------- Move Car Down ------------------------------------------------------
move_car_down = InstantaneousAction('move_car_down', v=Car, r=IntType(0,rows-1),
                                    c=IntType(0,columns-1))
v = move_car_down.parameter('v')
r = move_car_down.parameter('r')
c = move_car_down.parameter('c')
# vehicle is not none
move_car_down.add_precondition(Not(Equals(v,none)))
# is horizontal and it is located in the position
move_car_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v)))
# next position must be empty
move_car_down.add_precondition(Equals(occupied[r+2][c], none))

# control effects occupied positions
move_car_down.add_effect(occupied[r][c], none)
move_car_down.add_effect(occupied[r+2][c], v)

# ------------------------------------------------ Move Truck Right ------------------------------------------------------
move_truck_right = InstantaneousAction('move_truck_right', v=Truck, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_truck_right.parameter('v')
r = move_truck_right.parameter('r')
c = move_truck_right.parameter('c')
# vehicle is not none
move_truck_right.add_precondition(Not(Equals(v,none)))
# it is located in the position
move_truck_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v), Equals(occupied[r][c+2], v)))
# next position must be empty
move_truck_right.add_precondition(Equals(occupied[r][c+3], none))

# control effects occupied positions
move_truck_right.add_effect(occupied[r][c], none)
move_truck_right.add_effect(occupied[r][c+3], v)

# ------------------------------------------------ Move Truck Left ------------------------------------------------------
move_truck_left = InstantaneousAction('move_truck_left', v=Truck, r=IntType(0,rows-1),
                                      c=IntType(0,columns-1))
v = move_truck_left.parameter('v')
r = move_truck_left.parameter('r')
c = move_truck_left.parameter('c')
# vehicle is not none
move_truck_left.add_precondition(Not(Equals(v,none)))
# it is located in the position
move_truck_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v), Equals(occupied[r][c+2], v)))
# next position must be empty
move_truck_left.add_precondition(Equals(occupied[r][c-1], none))

# control effects occupied positions
move_truck_left.add_effect(occupied[r][c+2], none)
move_truck_left.add_effect(occupied[r][c-1], v)

# ------------------------------------------------- Move Truck Up ------------------------------------------------------
move_truck_up = InstantaneousAction('move_truck_up', v=Truck, r=IntType(0,rows-1),
                                    c=IntType(0,columns-1))
v = move_truck_up.parameter('v')
r = move_truck_up.parameter('r')
c = move_truck_up.parameter('c')
# vehicle is not none
move_truck_up.add_precondition(Not(Equals(v,none)))
# is horizontal and it is located in the position
move_truck_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v),
                                   Equals(occupied[r+2][c], v)))
# next position must be empty
move_truck_up.add_precondition(Equals(occupied[r-1][c], none))

# control effects occupied positions
move_truck_up.add_effect(occupied[r+2][c], none)
move_truck_up.add_effect(occupied[r-1][c], v)

# ------------------------------------------------- Move Truck Down ------------------------------------------------------
move_truck_down = InstantaneousAction('move_truck_down', v=Truck, r=IntType(0,rows-1),
                                      c=IntType(0,columns-1))
v = move_truck_down.parameter('v')
r = move_truck_down.parameter('r')
c = move_truck_down.parameter('c')
# vehicle is not none
move_truck_down.add_precondition(Not(Equals(v,none)))
# is horizontal and it is located in the position
move_truck_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v),
                                     Equals(occupied[r+2][c], v)))
# next position must be empty
move_truck_down.add_precondition(Equals(occupied[r+3][c], none))

# control effects occupied positions
move_truck_down.add_effect(occupied[r][c], none)
move_truck_down.add_effect(occupied[r+3][c], v)

# ------------------------------------------------ Move Quad Right ------------------------------------------------------
move_quad_right = InstantaneousAction('move_quad_right', v=Quad, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_quad_right.parameter('v')
r = move_quad_right.parameter('r')
c = move_quad_right.parameter('c')
# vehicle is not none
move_quad_right.add_precondition(Not(Equals(v,none)))
# it is located in the position
move_quad_right.add_precondition(And(
    Equals(occupied[r][c], v), Equals(occupied[r][c+1], v),
    Equals(occupied[r+1][c], v), Equals(occupied[r+1][c+1], v)
))
# next position must be empty
move_quad_right.add_precondition(Equals(occupied[r][c+2], none))
move_quad_right.add_precondition(Equals(occupied[r+1][c+2], none))

# control effects occupied positions
move_quad_right.add_effect(occupied[r][c], none)
move_quad_right.add_effect(occupied[r+1][c], none)
move_quad_right.add_effect(occupied[r][c+2], v)
move_quad_right.add_effect(occupied[r+1][c+2], v)

# ------------------------------------------------ Move Quad Left ------------------------------------------------------
move_quad_left = InstantaneousAction('move_quad_left', v=Quad, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_quad_left.parameter('v')
r = move_quad_left.parameter('r')
c = move_quad_left.parameter('c')
# vehicle is not none
move_quad_left.add_precondition(Not(Equals(v,none)))
# it is located in the position
move_quad_left.add_precondition(And(
    Equals(occupied[r][c], v), Equals(occupied[r][c+1], v),
    Equals(occupied[r+1][c], v), Equals(occupied[r+1][c+1], v)
))
# next position must be empty
move_quad_left.add_precondition(Equals(occupied[r][c-1], none))
move_quad_left.add_precondition(Equals(occupied[r+1][c-1], none))

# control effects occupied positions
move_quad_left.add_effect(occupied[r][c+1], none)
move_quad_left.add_effect(occupied[r+1][c+1], none)
move_quad_left.add_effect(occupied[r][c-1], v)
move_quad_left.add_effect(occupied[r+1][c-1], v)

# ------------------------------------------------- Move Quad Up ------------------------------------------------------
move_quad_up = InstantaneousAction('move_quad_up', v=Quad, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_quad_up.parameter('v')
r = move_quad_up.parameter('r')
c = move_quad_up.parameter('c')
# vehicle is not none
move_quad_up.add_precondition(Not(Equals(v,none)))
# it is located in the position
move_quad_up.add_precondition(And(
    Equals(occupied[r][c], v), Equals(occupied[r][c+1], v),
    Equals(occupied[r+1][c], v), Equals(occupied[r+1][c+1], v)
))
# next position must be empty
move_quad_up.add_precondition(Equals(occupied[r-1][c], none))
move_quad_up.add_precondition(Equals(occupied[r-1][c+1], none))

# control effects occupied positions
move_quad_up.add_effect(occupied[r+1][c], none)
move_quad_up.add_effect(occupied[r+1][c+1], none)
move_quad_up.add_effect(occupied[r-1][c], v)
move_quad_up.add_effect(occupied[r-1][c+1], v)

# ------------------------------------------------- Move Quad Down ------------------------------------------------------
move_quad_down = InstantaneousAction('move_quad_down', v=Quad, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_quad_down.parameter('v')
r = move_quad_down.parameter('r')
c = move_quad_down.parameter('c')
# vehicle is not none
move_quad_down.add_precondition(Not(Equals(v,none)))
# it is located in the position
move_quad_down.add_precondition(And(

    Equals(occupied[r][c], v), Equals(occupied[r][c+1], v),
    Equals(occupied[r+1][c], v), Equals(occupied[r+1][c+1], v)
))
# next position must be empty
move_quad_down.add_precondition(Equals(occupied[r][c+2], none))
move_quad_down.add_precondition(Equals(occupied[r+1][c+2], none))

# control effects occupied positions
move_quad_down.add_effect(occupied[r][c], none)
move_quad_down.add_effect(occupied[r+1][c], none)
move_quad_down.add_effect(occupied[r][c+2], v)
move_quad_down.add_effect(occupied[r+1][c+2], v)


# Add actions
rush_hour_problem.add_actions([move_car_right, move_car_left, move_car_down, move_car_up,
                               move_truck_right, move_truck_left, move_truck_down, move_truck_up,
                               move_quad_right, move_quad_left, move_quad_down, move_quad_up])

# ----------------------------------------------------- Goal -----------------------------------------------------------
rush_hour_problem.add_goal(Equals(occupied[row_goal][columns-1], X))
rush_hour_problem.add_goal(Equals(occupied[row_goal][columns-2], X))

# ------------------------------------------------- Action Costs -------------------------------------------------------
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

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(rush_hour_problem, solving, compilation)