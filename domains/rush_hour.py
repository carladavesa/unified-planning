"""Rush Hour planning domain.

Example:
  python run.py --domain rush_hour --compilation up --solving fast-downward
"""
import math
from typing import Dict, Optional
from unified_planning.model import Action, Expression, Object
from unified_planning.shortcuts import (
    ArrayType,
    Equals,
    Fluent,
    Int,
    IntType,
    InstantaneousAction,
    MinimizeActionCosts,
    Problem, UserType, Minus, Plus, LE, GE, And, Not,
)

from domains.base import Domain

INSTANCES: Dict[str, str] = {
    "r1": "AoooBooAoooBCCDoooOOODXXoEFoGGHHEFooIJKKLooIJooLo",
    "r2": "ABOOOoCABPPPoCooQDDUUXXQooUUooQooGGooHEFIJooHEFIJ",
    "r3": "oABBCCOoADDPoOoXXEPoOQQQEPooooFGGHHooFIIJooKKLLJo",
    "r4": "ooooooPoBBoooPooOoXXPooOooUUCCOoQUUADEEQFoADooQFo",
    "r5": "ABBCooOAooCooOXXDoooOEFDGGUUEFHHPUUIIJJPKoooooPKo",
    "r6": "oooOOOCoooPPPCoQQQUUGXXBoUUGoEBRADDoEFRAooooFRooo",
    "r7": "ooAoBCCooAoBooDDEoFGGXXEoFoHUUVVIIHUUVVooJoKKLLoJ",
    "r8": "oooooUUAAOBBUUXXOVVoooDOVVCooDEEPCoooooPFFooooPoo",
    "r9": "CCFFPoQABooPoQABXXPoQooEDOOOooEDoooGGHHoUUoooooUU",
    "r10": "AUUooooAUUoOOOoCXXPoQoCoBPoQoDDBPoQEVVoRRREVVoooo"
}


class RushHourDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {"grid": v} for k, v in self._instances.items()}

    def get_instance(self, instance: Optional[str] = None) -> list[int]:
        if instance and instance in self._instances:
            return self._instances[instance]
        raise ValueError(f"Instance '{instance}' not found!")

    def build_problem(self, instance: str | None = None) -> "Problem":
        instance_str = self.get_instance(instance)

        rows = columns = int(math.sqrt(len(instance_str)))
        undefined = []
        for i, char in enumerate(instance_str):
            r, c = divmod(i, columns)
            if char == 'x':
                undefined.append((r, c))
        idx = instance_str.index('X')
        row_goal = idx // rows
        # --- Problem ---
        problem = Problem('rush_hour_problem')

        Vehicle = UserType('Vehicle')
        none = Object('none', Vehicle)
        X_obj = Object('X', Vehicle)
        problem.add_objects([none, X_obj])

        occupied = Fluent('occupied', ArrayType(rows, ArrayType(columns, Vehicle)), undefined_positions=undefined)
        is_car = Fluent('is_car', v=Vehicle)
        is_truck = Fluent('is_truck', v=Vehicle)
        problem.add_fluent(occupied, default_initial_value=none)
        problem.add_fluent(is_car, default_initial_value=False)
        problem.add_fluent(is_truck, default_initial_value=False)
        problem.set_initial_value(is_car(X_obj), True)

        for i, char in enumerate(instance_str):
            r, c = divmod(i, columns)
            if char == 'o':
                problem.set_initial_value(occupied[r][c], none)
            elif char != 'x':
                obj = Object(char, Vehicle)
                if not problem.has_object(char):
                    problem.add_object(obj)
                    problem.set_initial_value(is_car(obj), instance_str.count(char) == 2)
                    problem.set_initial_value(is_truck(obj), instance_str.count(char) == 3)
                problem.set_initial_value(occupied[r][c], problem.object(char))

        # --- Actions ---
        move_car_right = InstantaneousAction('move_car_right', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_car_right.parameter('v'), move_car_right.parameter('r'), move_car_right.parameter('c')
        move_car_right.add_precondition(Not(Equals(v, none)))
        move_car_right.add_precondition(is_car(v))
        move_car_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c + 1], v)))
        move_car_right.add_precondition(Equals(occupied[r][c + 2], none))
        move_car_right.add_effect(occupied[r][c], none)
        move_car_right.add_effect(occupied[r][c + 2], v)

        move_car_left = InstantaneousAction('move_car_left', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_car_left.parameter('v'), move_car_left.parameter('r'), move_car_left.parameter('c')
        move_car_left.add_precondition(Not(Equals(v, none)))
        move_car_left.add_precondition(is_car(v))
        move_car_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c + 1], v)))
        move_car_left.add_precondition(Equals(occupied[r][c - 1], none))
        move_car_left.add_effect(occupied[r][c - 1], v)
        move_car_left.add_effect(occupied[r][c + 1], none)

        move_car_up = InstantaneousAction('move_car_up', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_car_up.parameter('v'), move_car_up.parameter('r'), move_car_up.parameter('c')
        move_car_up.add_precondition(Not(Equals(v, none)))
        move_car_up.add_precondition(is_car(v))
        move_car_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r + 1][c], v)))
        move_car_up.add_precondition(Equals(occupied[r - 1][c], none))
        move_car_up.add_effect(occupied[r - 1][c], v)
        move_car_up.add_effect(occupied[r + 1][c], none)

        move_car_down = InstantaneousAction('move_car_down', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_car_down.parameter('v'), move_car_down.parameter('r'), move_car_down.parameter('c')
        move_car_down.add_precondition(Not(Equals(v, none)))
        move_car_down.add_precondition(is_car(v))
        move_car_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r + 1][c], v)))
        move_car_down.add_precondition(Equals(occupied[r + 2][c], none))
        move_car_down.add_effect(occupied[r][c], none)
        move_car_down.add_effect(occupied[r + 2][c], v)

        move_truck_right = InstantaneousAction('move_truck_right', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_truck_right.parameter('v'), move_truck_right.parameter('r'), move_truck_right.parameter('c')
        move_truck_right.add_precondition(Not(Equals(v, none)))
        move_truck_right.add_precondition(is_truck(v))
        move_truck_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c + 1], v), Equals(occupied[r][c + 2], v)))
        move_truck_right.add_precondition(Equals(occupied[r][c + 3], none))
        move_truck_right.add_effect(occupied[r][c], none)
        move_truck_right.add_effect(occupied[r][c + 3], v)

        move_truck_left = InstantaneousAction('move_truck_left', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_truck_left.parameter('v'), move_truck_left.parameter('r'), move_truck_left.parameter('c')
        move_truck_left.add_precondition(Not(Equals(v, none)))
        move_truck_left.add_precondition(is_truck(v))
        move_truck_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c + 1], v), Equals(occupied[r][c + 2], v)))
        move_truck_left.add_precondition(Equals(occupied[r][c - 1], none))
        move_truck_left.add_effect(occupied[r][c + 2], none)
        move_truck_left.add_effect(occupied[r][c - 1], v)

        move_truck_up = InstantaneousAction('move_truck_up', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_truck_up.parameter('v'), move_truck_up.parameter('r'), move_truck_up.parameter('c')
        move_truck_up.add_precondition(Not(Equals(v, none)))
        move_truck_up.add_precondition(is_truck(v))
        move_truck_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r + 1][c], v), Equals(occupied[r + 2][c], v)))
        move_truck_up.add_precondition(Equals(occupied[r - 1][c], none))
        move_truck_up.add_effect(occupied[r + 2][c], none)
        move_truck_up.add_effect(occupied[r - 1][c], v)

        move_truck_down = InstantaneousAction('move_truck_down', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_truck_down.parameter('v'), move_truck_down.parameter('r'), move_truck_down.parameter('c')
        move_truck_down.add_precondition(Not(Equals(v, none)))
        move_truck_down.add_precondition(is_truck(v))
        move_truck_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r + 1][c], v), Equals(occupied[r + 2][c], v)))
        move_truck_down.add_precondition(Equals(occupied[r + 3][c], none))
        move_truck_down.add_effect(occupied[r][c], none)
        move_truck_down.add_effect(occupied[r + 3][c], v)

        move_quad_right = InstantaneousAction('move_quad_right', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_quad_right.parameter('v'), move_quad_right.parameter('r'), move_quad_right.parameter('c')
        move_quad_right.add_precondition(Not(Equals(v, none)))
        move_quad_right.add_precondition(And(Not(is_truck(v)), Not(is_car(v))))
        move_quad_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c + 1], v), Equals(occupied[r + 1][c], v), Equals(occupied[r + 1][c + 1], v)))
        move_quad_right.add_precondition(Equals(occupied[r][c + 2], none))
        move_quad_right.add_precondition(Equals(occupied[r + 1][c + 2], none))
        move_quad_right.add_effect(occupied[r][c], none)
        move_quad_right.add_effect(occupied[r + 1][c], none)
        move_quad_right.add_effect(occupied[r][c + 2], v)
        move_quad_right.add_effect(occupied[r + 1][c + 2], v)

        move_quad_left = InstantaneousAction('move_quad_left', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_quad_left.parameter('v'), move_quad_left.parameter('r'), move_quad_left.parameter('c')
        move_quad_left.add_precondition(Not(Equals(v, none)))
        move_quad_left.add_precondition(And(Not(is_truck(v)), Not(is_car(v))))
        move_quad_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c + 1], v), Equals(occupied[r + 1][c], v), Equals(occupied[r + 1][c + 1], v)))
        move_quad_left.add_precondition(Equals(occupied[r][c - 1], none))
        move_quad_left.add_precondition(Equals(occupied[r + 1][c - 1], none))
        move_quad_left.add_effect(occupied[r][c + 1], none)
        move_quad_left.add_effect(occupied[r + 1][c + 1], none)
        move_quad_left.add_effect(occupied[r][c - 1], v)
        move_quad_left.add_effect(occupied[r + 1][c - 1], v)

        move_quad_up = InstantaneousAction('move_quad_up', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_quad_up.parameter('v'), move_quad_up.parameter('r'), move_quad_up.parameter('c')
        move_quad_up.add_precondition(Not(Equals(v, none)))
        move_quad_up.add_precondition(And(Not(is_truck(v)), Not(is_car(v))))
        move_quad_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c + 1], v), Equals(occupied[r + 1][c], v), Equals(occupied[r + 1][c + 1], v)))
        move_quad_up.add_precondition(Equals(occupied[r - 1][c], none))
        move_quad_up.add_precondition(Equals(occupied[r - 1][c + 1], none))
        move_quad_up.add_effect(occupied[r + 1][c], none)
        move_quad_up.add_effect(occupied[r + 1][c + 1], none)
        move_quad_up.add_effect(occupied[r - 1][c], v)
        move_quad_up.add_effect(occupied[r - 1][c + 1], v)

        move_quad_down = InstantaneousAction('move_quad_down', v=Vehicle, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
        v, r, c = move_quad_down.parameter('v'), move_quad_down.parameter('r'), move_quad_down.parameter('c')
        move_quad_down.add_precondition(Not(Equals(v, none)))
        move_quad_down.add_precondition(And(Not(is_truck(v)), Not(is_car(v))))
        move_quad_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c + 1], v), Equals(occupied[r + 1][c], v), Equals(occupied[r + 1][c + 1], v)))
        move_quad_down.add_precondition(Equals(occupied[r + 2][c], none))
        move_quad_down.add_precondition(Equals(occupied[r + 2][c + 1], none))
        move_quad_down.add_effect(occupied[r][c], none)
        move_quad_down.add_effect(occupied[r][c + 1], none)
        move_quad_down.add_effect(occupied[r + 2][c], v)
        move_quad_down.add_effect(occupied[r + 2][c + 1], v)

        problem.add_actions([
            move_car_right, move_car_left, move_car_up, move_car_down,
            move_truck_right, move_truck_left, move_truck_up, move_truck_down,
            move_quad_right, move_quad_left, move_quad_up, move_quad_down,
        ])

        # --- Goals ---
        problem.add_goal(Equals(occupied[row_goal][columns - 1], X_obj))
        problem.add_goal(Equals(occupied[row_goal][columns - 2], X_obj))

        # --- Metric ---
        costs: Dict[Action, Expression] = {
            move_car_right: Int(1), move_car_left: Int(1), move_car_up: Int(1), move_car_down: Int(1),
            move_truck_right: Int(1), move_truck_left: Int(1), move_truck_up: Int(1), move_truck_down: Int(1),
            move_quad_right: Int(1), move_quad_left: Int(1), move_quad_up: Int(1), move_quad_down: Int(1),
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem

DOMAIN = RushHourDomain()