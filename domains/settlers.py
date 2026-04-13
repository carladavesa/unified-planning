"""settlers planning domain.

Example:
  python run.py --domain settlers --instance pfile1 --compilation uti --solving fast-downward
"""
import os
import re
import subprocess
import ast
from typing import Optional
from unified_planning.model import Object
from unified_planning.shortcuts import (
    Fluent, IntType, BoolType, UserType,
    InstantaneousAction, Problem,
    GE, And, Not, Int,
    MinimizeActionCosts,
)
from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'settlers', 'handcrafted')
READ_INSTANCE_SCRIPT = os.path.join(os.path.dirname(__file__), 'settlers', 'read_instance.py')

INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 21)]


class SettlersDomain(Domain):
    def __init__(self):
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if not instance or instance not in self._instances:
            raise ValueError(f"Instance '{instance}' not found!")
        proc = subprocess.run(
            ['python3', READ_INSTANCE_SCRIPT, instance],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            raise ValueError(f"Error parsing '{instance}': {proc.stderr}")
        parts = proc.stdout.split('---')
        if len(parts) < 3:
            raise ValueError(f"Unexpected parser output for '{instance}'")
        places = ast.literal_eval(parts[0].strip())
        vehicles = ast.literal_eval(parts[1].strip()) if parts[1].strip() else []
        data = ast.literal_eval(parts[2].strip())
        return {'places': places, 'vehicles': vehicles, 'data': data}

    def _add_goals(self, problem: Problem, goals: list, fluent_map: dict, place_objs: dict) -> None:
        """Translate PDDL goal strings to UP goals."""
        from unified_planning.shortcuts import GE
        import re

        for goal_str in goals:
            goal_str = goal_str.strip()

            # (>= (housing location0) 2)
            m = re.match(r'^\(>=\s+\(housing\s+(\w+)\)\s+(\d+)\)$', goal_str)
            if m:
                loc, val = m.group(1), int(m.group(2))
                if loc in place_objs:
                    problem.add_goal(GE(fluent_map['housing'](place_objs[loc]), val))
                continue

            # (has-cabin location0), (has-coal-stack location0), etc.
            m = re.match(r'^\((has-[\w-]+)\s+(\w+)\)$', goal_str)
            if m:
                pred, loc = m.group(1), m.group(2)
                has_map = {
                    'has-cabin': 'has_cabin',
                    'has-coal-stack': 'has_coal_stack',
                    'has-quarry': 'has_quarry',
                    'has-mine': 'has_mine',
                    'has-sawmill': 'has_sawmill',
                    'has-ironworks': 'has_ironworks',
                    'has-docks': 'has_docks',
                    'has-wharf': 'has_wharf',
                }
                f_name = has_map.get(pred)
                if f_name and loc in place_objs:
                    problem.add_goal(fluent_map[f_name](place_objs[loc]))
                continue

            # (connected-by-rail location1 location2)
            m = re.match(r'^\(connected-by-rail\s+(\w+)\s+(\w+)\)$', goal_str)
            if m:
                p1, p2 = m.group(1), m.group(2)
                if p1 in place_objs and p2 in place_objs:
                    problem.add_goal(fluent_map['connected_by_rail'](place_objs[p1], place_objs[p2]))
                continue

            # (connected-by-land location1 location2)
            m = re.match(r'^\(connected-by-land\s+(\w+)\s+(\w+)\)$', goal_str)
            if m:
                p1, p2 = m.group(1), m.group(2)
                if p1 in place_objs and p2 in place_objs:
                    problem.add_goal(fluent_map['connected_by_land'](place_objs[p1], place_objs[p2]))
                continue

            print(f"[WARNING] Goal not parsed: {goal_str}")

    def build_problem(self, instance: str | None = None) -> Problem:
        parsed = self.get_instance(instance)
        places = parsed['places']
        vehicles = parsed['vehicles']
        pd = parsed['data']

        # Calculate bounds
        n_places = len(places)
        n_vehicles = len(vehicles)
        n_actions_estimate = (n_places + n_vehicles) * 20

        max_housing_goal = 0
        for goal_str in pd.get('goals', []):
            m = re.match(r'^\(>=\s+\(housing\s+\w+\)\s+(\d+)\)$', goal_str)
            if m:
                max_housing_goal = max(max_housing_goal, int(m.group(1)))

        available_ub = n_places * 5 + 5
        space_ub = 10
        housing_ub = max(max_housing_goal + 2, 5)
        labour_ub = n_actions_estimate
        pollution_ub = n_actions_estimate // 2
        resource_use_ub = n_actions_estimate // 3

        problem = Problem('settlers_problem')

        # ===== Types =====
        Place = UserType('Place')
        Vehicle = UserType('Vehicle')
        Resource = UserType('Resource')

        # ===== Objects =====
        place_objs = {p: Object(p, Place) for p in places}
        vehicle_objs = {v: Object(v, Vehicle) for v in vehicles}
        for obj in place_objs.values():
            problem.add_object(obj)
        for obj in vehicle_objs.values():
            problem.add_object(obj)

        known_resources = {'ore', 'iron', 'stone', 'wood', 'timber', 'coal'}
        resources_found = {res for (_, res) in pd.get('resources', {}).keys()}
        all_resources = sorted(resources_found | known_resources)
        resource_objs = {r: Object(r, Resource) for r in all_resources}
        for obj in resource_objs.values():
            problem.add_object(obj)

        # ===== Fluents =====
        available = Fluent('available', IntType(0, available_ub), r=Resource, s=Place)
        available_v = Fluent('available_v', IntType(0, available_ub), r=Resource, v=Vehicle)
        space_in = Fluent('space_in', IntType(0, space_ub), v=Vehicle)
        housing = Fluent('housing', IntType(0, housing_ub), p=Place)
        labour = Fluent('labour', IntType(0, labour_ub))
        pollution = Fluent('pollution', IntType(0, pollution_ub))
        resource_use = Fluent('resource_use', IntType(0, resource_use_ub))

        woodland = Fluent('woodland', BoolType(), p=Place)
        mountain = Fluent('mountain', BoolType(), p=Place)
        metalliferous = Fluent('metalliferous', BoolType(), p=Place)
        by_coast = Fluent('by_coast', BoolType(), p=Place)
        has_cabin = Fluent('has_cabin', BoolType(), p=Place)
        has_coal_stack = Fluent('has_coal_stack', BoolType(), p=Place)
        has_quarry = Fluent('has_quarry', BoolType(), p=Place)
        has_mine = Fluent('has_mine', BoolType(), p=Place)
        has_sawmill = Fluent('has_sawmill', BoolType(), p=Place)
        has_ironworks = Fluent('has_ironworks', BoolType(), p=Place)
        has_docks = Fluent('has_docks', BoolType(), p=Place)
        has_wharf = Fluent('has_wharf', BoolType(), p=Place)
        is_cart = Fluent('is_cart', BoolType(), v=Vehicle)
        is_train = Fluent('is_train', BoolType(), v=Vehicle)
        is_ship = Fluent('is_ship', BoolType(), v=Vehicle)
        potential = Fluent('potential', BoolType(), v=Vehicle)
        connected_by_land = Fluent('connected_by_land', BoolType(), p1=Place, p2=Place)
        connected_by_rail = Fluent('connected_by_rail', BoolType(), p1=Place, p2=Place)
        connected_by_sea = Fluent('connected_by_sea', BoolType(), p1=Place, p2=Place)
        is_at = Fluent('is_at', BoolType(), v=Vehicle, p=Place)

        for f in [available, available_v, space_in, housing, labour, pollution, resource_use]:
            problem.add_fluent(f, default_initial_value=Int(0))
        for f in [woodland, mountain, metalliferous, by_coast, has_cabin, has_coal_stack,
                  has_quarry, has_mine, has_sawmill, has_ironworks, has_docks, has_wharf,
                  is_cart, is_train, is_ship, potential,
                  connected_by_land, connected_by_rail, connected_by_sea, is_at]:
            problem.add_fluent(f, default_initial_value=False)

        # ===== Initial values =====
        gv = pd.get('global_values', {})
        if 'labour' in gv:
            problem.set_initial_value(labour(), gv['labour'])
        if 'pollution' in gv:
            problem.set_initial_value(pollution(), gv['pollution'])
        if 'resource-use' in gv:
            problem.set_initial_value(resource_use(), gv['resource-use'])

        for loc, val in pd.get('housing', {}).items():
            problem.set_initial_value(housing(place_objs[loc]), val)

        for (store, res), val in pd.get('resources', {}).items():
            r_obj = resource_objs.get(res)
            if not r_obj or val == 0:
                continue
            if store in place_objs:
                problem.set_initial_value(available(r_obj, place_objs[store]), val)
            elif store in vehicle_objs:
                problem.set_initial_value(available_v(r_obj, vehicle_objs[store]), val)

        for v, val in pd.get('space_in', {}).items():
            if val != 0:
                problem.set_initial_value(space_in(vehicle_objs[v]), val)

        for v in pd.get('potential_vehicles', []):
            problem.set_initial_value(potential(vehicle_objs[v]), True)

        for v, ptype in pd.get('vehicle_properties', {}).items():
            if ptype == 'cart':
                problem.set_initial_value(is_cart(vehicle_objs[v]), True)
            elif ptype == 'train':
                problem.set_initial_value(is_train(vehicle_objs[v]), True)
            elif ptype == 'ship':
                problem.set_initial_value(is_ship(vehicle_objs[v]), True)

        for (a, b) in pd.get('connections', {}).get('by_land', []):
            problem.set_initial_value(connected_by_land(place_objs[a], place_objs[b]), True)
        for (a, b) in pd.get('connections', {}).get('by_rail', []):
            problem.set_initial_value(connected_by_rail(place_objs[a], place_objs[b]), True)
        for (a, b) in pd.get('connections', {}).get('by_sea', []):
            problem.set_initial_value(connected_by_sea(place_objs[a], place_objs[b]), True)

        for (v, l) in pd.get('is_at', []):
            if v in vehicle_objs and l in place_objs:
                problem.set_initial_value(is_at(vehicle_objs[v], place_objs[l]), True)

        HAS_MAP = {
            'has-cabin': has_cabin, 'has-coal-stack': has_coal_stack,
            'has-quarry': has_quarry, 'has-mine': has_mine,
            'has-sawmill': has_sawmill, 'has-ironworks': has_ironworks,
            'has-docks': has_docks, 'has-wharf': has_wharf,
        }
        for pred, locs in pd.get('has_predicates', {}).items():
            f = HAS_MAP.get(pred)
            if f:
                for loc in locs:
                    problem.set_initial_value(f(place_objs[loc]), True)

        for prop, locs in pd.get('place_properties', {}).items():
            f_map = {'woodland': woodland, 'mountain': mountain,
                     'metalliferous': metalliferous, 'by-coast': by_coast}
            for prop_name, f in f_map.items():
                if prop_name in locs:
                    problem.set_initial_value(f(place_objs[prop]), True)

        # ===== Actions =====
        wood = resource_objs['wood']
        timber = resource_objs['timber']
        stone = resource_objs['stone']
        ore = resource_objs['ore']
        iron = resource_objs['iron']
        coal = resource_objs['coal']

        # load1
        load1 = InstantaneousAction('load1', v=Vehicle, p=Place, r=Resource)
        v, p, r = load1.parameter('v'), load1.parameter('p'), load1.parameter('r')
        load1.add_precondition(is_at(v, p))
        load1.add_precondition(GE(available(r, p), 1))
        load1.add_precondition(GE(space_in(v), 1))
        load1.add_decrease_effect(space_in(v), 1)
        load1.add_increase_effect(available_v(r, v), 1)
        load1.add_decrease_effect(available(r, p), 1)
        load1.add_increase_effect(labour(), 1)

        # unload1
        unload1 = InstantaneousAction('unload1', v=Vehicle, p=Place, r=Resource)
        v, p, r = unload1.parameter('v'), unload1.parameter('p'), unload1.parameter('r')
        unload1.add_precondition(is_at(v, p))
        unload1.add_precondition(GE(available_v(r, v), 1))
        unload1.add_increase_effect(space_in(v), 1)
        unload1.add_decrease_effect(available_v(r, v), 1)
        unload1.add_increase_effect(available(r, p), 1)
        unload1.add_increase_effect(labour(), 1)

        # movecart1
        movecart1 = InstantaneousAction('movecart1', v=Vehicle, p1=Place, p2=Place)
        v, p1, p2 = movecart1.parameter('v'), movecart1.parameter('p1'), movecart1.parameter('p2')
        movecart1.add_precondition(is_cart(v))
        movecart1.add_precondition(connected_by_land(p1, p2))
        movecart1.add_precondition(is_at(v, p1))
        movecart1.add_effect(is_at(v, p1), False)
        movecart1.add_effect(is_at(v, p2), True)
        movecart1.add_increase_effect(labour(), 2)

        # movetrain1
        movetrain1 = InstantaneousAction('movetrain1', v=Vehicle, p1=Place, p2=Place)
        v, p1, p2 = movetrain1.parameter('v'), movetrain1.parameter('p1'), movetrain1.parameter('p2')
        movetrain1.add_precondition(is_train(v))
        movetrain1.add_precondition(connected_by_rail(p1, p2))
        movetrain1.add_precondition(is_at(v, p1))
        movetrain1.add_precondition(GE(available_v(coal, v), 1))
        movetrain1.add_effect(is_at(v, p1), False)
        movetrain1.add_effect(is_at(v, p2), True)
        movetrain1.add_decrease_effect(available_v(coal, v), 1)
        movetrain1.add_increase_effect(pollution(), 1)

        # moveship1
        moveship1 = InstantaneousAction('moveship1', v=Vehicle, p1=Place, p2=Place)
        v, p1, p2 = moveship1.parameter('v'), moveship1.parameter('p1'), moveship1.parameter('p2')
        moveship1.add_precondition(is_ship(v))
        moveship1.add_precondition(connected_by_sea(p1, p2))
        moveship1.add_precondition(is_at(v, p1))
        moveship1.add_precondition(GE(available_v(coal, v), 2))
        moveship1.add_effect(is_at(v, p1), False)
        moveship1.add_effect(is_at(v, p2), True)
        moveship1.add_decrease_effect(available_v(coal, v), 2)
        moveship1.add_increase_effect(pollution(), 2)

        # buildcabin1
        buildcabin1 = InstantaneousAction('buildcabin1', p=Place)
        p = buildcabin1.parameter('p')
        buildcabin1.add_precondition(woodland(p))
        buildcabin1.add_effect(has_cabin(p), True)
        buildcabin1.add_increase_effect(labour(), 1)

        # buildquarry1
        buildquarry1 = InstantaneousAction('buildquarry1', p=Place)
        p = buildquarry1.parameter('p')
        buildquarry1.add_precondition(mountain(p))
        buildquarry1.add_effect(has_quarry(p), True)
        buildquarry1.add_increase_effect(labour(), 2)

        # buildcoalstack1
        buildcoalstack1 = InstantaneousAction('buildcoalstack1', p=Place)
        p = buildcoalstack1.parameter('p')
        buildcoalstack1.add_precondition(GE(available(timber, p), 1))
        buildcoalstack1.add_effect(has_coal_stack(p), True)
        buildcoalstack1.add_decrease_effect(available(timber, p), 1)
        buildcoalstack1.add_increase_effect(labour(), 2)

        # buildsawmill1
        buildsawmill1 = InstantaneousAction('buildsawmill1', p=Place)
        p = buildsawmill1.parameter('p')
        buildsawmill1.add_precondition(GE(available(timber, p), 2))
        buildsawmill1.add_effect(has_sawmill(p), True)
        buildsawmill1.add_decrease_effect(available(timber, p), 2)
        buildsawmill1.add_increase_effect(labour(), 2)

        # buildmine1
        buildmine1 = InstantaneousAction('buildmine1', p=Place)
        p = buildmine1.parameter('p')
        buildmine1.add_precondition(metalliferous(p))
        buildmine1.add_precondition(GE(available(wood, p), 2))
        buildmine1.add_effect(has_mine(p), True)
        buildmine1.add_decrease_effect(available(wood, p), 2)
        buildmine1.add_increase_effect(labour(), 3)

        # buildironworks1
        buildironworks1 = InstantaneousAction('buildironworks1', p=Place)
        p = buildironworks1.parameter('p')
        buildironworks1.add_precondition(GE(available(stone, p), 2))
        buildironworks1.add_precondition(GE(available(wood, p), 2))
        buildironworks1.add_effect(has_ironworks(p), True)
        buildironworks1.add_decrease_effect(available(stone, p), 2)
        buildironworks1.add_decrease_effect(available(wood, p), 2)
        buildironworks1.add_increase_effect(labour(), 3)

        # builddocks1
        builddocks1 = InstantaneousAction('builddocks1', p=Place)
        p = builddocks1.parameter('p')
        builddocks1.add_precondition(by_coast(p))
        builddocks1.add_precondition(GE(available(stone, p), 2))
        builddocks1.add_precondition(GE(available(wood, p), 2))
        builddocks1.add_effect(has_docks(p), True)
        builddocks1.add_decrease_effect(available(stone, p), 2)
        builddocks1.add_decrease_effect(available(wood, p), 2)
        builddocks1.add_increase_effect(labour(), 2)

        # buildwharf1
        buildwharf1 = InstantaneousAction('buildwharf1', p=Place)
        p = buildwharf1.parameter('p')
        buildwharf1.add_precondition(has_docks(p))
        buildwharf1.add_precondition(GE(available(stone, p), 2))
        buildwharf1.add_precondition(GE(available(iron, p), 2))
        buildwharf1.add_effect(has_wharf(p), True)
        buildwharf1.add_decrease_effect(available(stone, p), 2)
        buildwharf1.add_decrease_effect(available(iron, p), 2)
        buildwharf1.add_increase_effect(labour(), 2)

        # buildrail1
        buildrail1 = InstantaneousAction('buildrail1', p1=Place, p2=Place)
        p1, p2 = buildrail1.parameter('p1'), buildrail1.parameter('p2')
        buildrail1.add_precondition(connected_by_land(p1, p2))
        buildrail1.add_precondition(GE(available(wood, p1), 1))
        buildrail1.add_precondition(GE(available(iron, p1), 1))
        buildrail1.add_effect(connected_by_rail(p1, p2), True)
        buildrail1.add_decrease_effect(available(wood, p1), 1)
        buildrail1.add_decrease_effect(available(iron, p1), 1)
        buildrail1.add_increase_effect(labour(), 2)

        # buildhouse1
        buildhouse1 = InstantaneousAction('buildhouse1', p=Place)
        p = buildhouse1.parameter('p')
        buildhouse1.add_precondition(GE(available(wood, p), 1))
        buildhouse1.add_precondition(GE(available(stone, p), 1))
        buildhouse1.add_increase_effect(housing(p), 1)
        buildhouse1.add_decrease_effect(available(wood, p), 1)
        buildhouse1.add_decrease_effect(available(stone, p), 1)

        # buildcart1
        buildcart1 = InstantaneousAction('buildcart1', p=Place, v=Vehicle)
        p, v = buildcart1.parameter('p'), buildcart1.parameter('v')
        buildcart1.add_precondition(GE(available(timber, p), 1))
        buildcart1.add_precondition(potential(v))
        buildcart1.add_effect(is_at(v, p), True)
        buildcart1.add_effect(is_cart(v), True)
        buildcart1.add_effect(potential(v), False)
        buildcart1.add_decrease_effect(available(timber, p), 1)
        buildcart1.add_increase_effect(space_in(v), 1)
        buildcart1.add_increase_effect(labour(), 1)

        # buildtrain1
        buildtrain1 = InstantaneousAction('buildtrain1', p=Place, v=Vehicle)
        p, v = buildtrain1.parameter('p'), buildtrain1.parameter('v')
        buildtrain1.add_precondition(potential(v))
        buildtrain1.add_precondition(GE(available(iron, p), 2))
        buildtrain1.add_effect(is_at(v, p), True)
        buildtrain1.add_effect(is_train(v), True)
        buildtrain1.add_effect(potential(v), False)
        buildtrain1.add_decrease_effect(available(iron, p), 2)
        buildtrain1.add_increase_effect(space_in(v), 5)
        buildtrain1.add_increase_effect(labour(), 2)

        # buildship1
        buildship1 = InstantaneousAction('buildship1', p=Place, v=Vehicle)
        p, v = buildship1.parameter('p'), buildship1.parameter('v')
        buildship1.add_precondition(potential(v))
        buildship1.add_precondition(GE(available(iron, p), 4))
        buildship1.add_effect(has_wharf(p), True)
        buildship1.add_effect(is_at(v, p), True)
        buildship1.add_effect(is_ship(v), True)
        buildship1.add_effect(potential(v), False)
        buildship1.add_decrease_effect(available(iron, p), 4)
        buildship1.add_increase_effect(space_in(v), 10)
        buildship1.add_increase_effect(labour(), 3)

        # felltimber1
        felltimber1 = InstantaneousAction('felltimber1', p=Place)
        p = felltimber1.parameter('p')
        felltimber1.add_precondition(has_cabin(p))
        felltimber1.add_increase_effect(available(timber, p), 1)
        felltimber1.add_increase_effect(labour(), 1)

        # breakstone1
        breakstone1 = InstantaneousAction('breakstone1', p=Place)
        p = breakstone1.parameter('p')
        breakstone1.add_precondition(has_quarry(p))
        breakstone1.add_increase_effect(available(stone, p), 1)
        breakstone1.add_increase_effect(labour(), 1)
        breakstone1.add_increase_effect(resource_use(), 1)

        # mineore1
        mineore1 = InstantaneousAction('mineore1', p=Place)
        p = mineore1.parameter('p')
        mineore1.add_precondition(has_mine(p))
        mineore1.add_increase_effect(available(ore, p), 1)
        mineore1.add_increase_effect(resource_use(), 2)

        # burncoal1
        burncoal1 = InstantaneousAction('burncoal1', p=Place)
        p = burncoal1.parameter('p')
        burncoal1.add_precondition(has_coal_stack(p))
        burncoal1.add_precondition(GE(available(timber, p), 1))
        burncoal1.add_decrease_effect(available(timber, p), 1)
        burncoal1.add_increase_effect(available(coal, p), 1)
        burncoal1.add_increase_effect(pollution(), 1)

        # sawwood1
        sawwood1 = InstantaneousAction('sawwood1', p=Place)
        p = sawwood1.parameter('p')
        sawwood1.add_precondition(has_sawmill(p))
        sawwood1.add_precondition(GE(available(timber, p), 1))
        sawwood1.add_decrease_effect(available(timber, p), 1)
        sawwood1.add_increase_effect(available(wood, p), 1)

        # makeiron1
        makeiron1 = InstantaneousAction('makeiron1', p=Place)
        p = makeiron1.parameter('p')
        makeiron1.add_precondition(has_ironworks(p))
        makeiron1.add_precondition(GE(available(ore, p), 1))
        makeiron1.add_precondition(GE(available(coal, p), 2))
        makeiron1.add_decrease_effect(available(ore, p), 1)
        makeiron1.add_decrease_effect(available(coal, p), 2)
        makeiron1.add_increase_effect(available(iron, p), 1)
        makeiron1.add_increase_effect(pollution(), 2)

        problem.add_actions([
            load1, unload1, movecart1, movetrain1, moveship1,
            buildcabin1, buildquarry1, buildcoalstack1, buildsawmill1,
            buildmine1, buildironworks1, builddocks1, buildwharf1,
            buildrail1, buildhouse1, buildcart1, buildtrain1, buildship1,
            felltimber1, breakstone1, mineore1, burncoal1, sawwood1, makeiron1,
        ])

        # ===== Goals =====
        fluent_map = {
            'housing': housing,
            'has_cabin': has_cabin,
            'has_coal_stack': has_coal_stack,
            'has_quarry': has_quarry,
            'has_mine': has_mine,
            'has_sawmill': has_sawmill,
            'has_ironworks': has_ironworks,
            'has_docks': has_docks,
            'has_wharf': has_wharf,
            'connected_by_rail': connected_by_rail,
            'connected_by_land': connected_by_land,
        }
        self._add_goals(problem, pd.get('goals', []), fluent_map, place_objs)

        return problem


DOMAIN = SettlersDomain()