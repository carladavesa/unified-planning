"""rover planning domain."""
import os
import re
from typing import Optional
from unified_planning.shortcuts import (
    Fluent, Int, IntType, BoolType, UserType, Object,
    InstantaneousAction, Problem,
    GE, LE, And, Plus, Minus, Int,
    MinimizeActionCosts, MinimizeExpressionOnFinalState,
)
from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'rovers', 'handcrafted')
INSTANCES: list[str] = [f"pfile{i}" for i in range(1, 21)]


class RoverDomain(Domain):
    def __init__(self):
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()
        content = re.sub(r';.*', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Objects per tipus
        def parse_objects(s):
            result = {}
            segments = re.split(r'-\s*(\w+)', s)
            i = 0
            while i + 1 < len(segments):
                names = segments[i].split()
                t = segments[i+1].strip().lower()
                result.setdefault(t, []).extend(names)
                i += 2
            return result

        obj_m = re.search(r'\(:objects(.*?)\)(?=\s*\(:init)', content, re.DOTALL)
        objects = parse_objects(obj_m.group(1)) if obj_m else {}

        # Init
        init_m = re.search(r'\(:init(.*?)\)(?=\s*\(:goal)', content, re.DOTALL)
        init_str = init_m.group(1) if init_m else ""

        # Numeric init: (= (fluent args...) val)
        numeric = {}
        for m in re.finditer(r'\(=\s*\(([\w-]+)(?:\s+([\w\s]+?))?\)\s*([\d.]+)\s*\)', init_str):
            fname = m.group(1).lower().replace('-', '_')
            args = m.group(2).strip().split() if m.group(2) else []
            val = int(float(m.group(3)))
            numeric[(fname, tuple(args))] = val

        # Boolean init: predicats
        bool_facts = set()
        for m in re.finditer(r'\((\w[\w-]*(?:\s+\w+)*)\)', init_str):
            token = m.group(1).strip()
            if not token.startswith('='):
                parts = token.split()
                pred = parts[0].lower().replace('-', '_')
                args = tuple(p.lower() for p in parts[1:])
                bool_facts.add((pred, args))

        # Goal
        goal_m = re.search(r'\(:goal\s*\(and(.*?)\)\s*\)', content, re.DOTALL)
        goal_str = goal_m.group(1) if goal_m else ""
        goals = []
        for m in re.finditer(r'\(([\w-]+(?:\s+\w+)*)\)', goal_str):
            parts = m.group(1).strip().split()
            pred = parts[0].lower().replace('-', '_')
            args = tuple(p.lower() for p in parts[1:])
            goals.append((pred, args))

        # Metric
        metric = None
        metric_m = re.search(r'\(:metric\s+(\w+)\s+\(([\w-]+)\)', content)
        if metric_m:
            metric = (metric_m.group(1), metric_m.group(2).lower().replace('-', '_'))

        return {
            'objects': objects,
            'numeric': numeric,
            'bool_facts': bool_facts,
            'goals': goals,
            'metric': metric,
        }

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if not instance or instance not in self._instances:
            raise ValueError(f"Instance '{instance}' not found!")
        path = os.path.join(PDDL_DIR, f"{instance}.pddl")
        return self._parse_pddl(path)

    def build_problem(self, instance: str | None = None) -> Problem:
        data = self.get_instance(instance)
        objects = data['objects']
        numeric = data['numeric']
        bool_facts = data['bool_facts']
        goals = data['goals']
        metric = data['metric']

        problem = Problem('rover_problem')

        # Tipus
        Rover = UserType('rover')
        Waypoint = UserType('waypoint')
        Store = UserType('store')
        Camera = UserType('camera')
        Mode = UserType('mode')
        Lander = UserType('lander')
        Objective = UserType('objective')

        type_map = {
            'rover': Rover, 'waypoint': Waypoint, 'store': Store,
            'camera': Camera, 'mode': Mode, 'lander': Lander, 'objective': Objective,
        }

        # Objectes
        obj_map = {}
        for type_name, names in objects.items():
            ut = type_map.get(type_name)
            if ut is None:
                continue
            for name in names:
                o = Object(name, ut)
                problem.add_object(o)
                obj_map[name] = o

        def o(name): return obj_map[name]

        # Bounds per energy
        max_energy = 100
        energy_ub = max_energy

        # Fluents enters
        energy = Fluent('energy', IntType(0, energy_ub), r=Rover)
        max_recharges = 50
        total_cost = Fluent('total-cost', IntType(0, max_recharges))
        problem.add_fluent(energy, default_initial_value=Int(0))
        problem.add_fluent(total_cost, default_initial_value=Int(0))

        # Fluents booleans
        def bf(name, *types):
            sig = {f'p{i}': t for i, t in enumerate(types)}
            return Fluent(name, BoolType(), **sig)

        in_f = bf('in_f', Rover, Waypoint)
        at_lander = bf('at_lander', Lander, Waypoint)
        can_traverse = bf('can_traverse', Rover, Waypoint, Waypoint)
        equipped_soil = bf('equipped_for_soil_analysis', Rover)
        equipped_rock = bf('equipped_for_rock_analysis', Rover)
        equipped_img = bf('equipped_for_imaging', Rover)
        empty = bf('empty', Store)
        have_rock = bf('have_rock_analysis', Rover, Waypoint)
        have_soil = bf('have_soil_analysis', Rover, Waypoint)
        full = bf('full', Store)
        calibrated = bf('calibrated', Camera, Rover)
        supports = bf('supports', Camera, Mode)
        available = bf('available', Rover)
        visible = bf('visible', Waypoint, Waypoint)
        have_image = bf('have_image', Rover, Objective, Mode)
        comm_soil = bf('communicated_soil_data', Waypoint)
        comm_rock = bf('communicated_rock_data', Waypoint)
        comm_image = bf('communicated_image_data', Objective, Mode)
        at_soil_sample = bf('at_soil_sample', Waypoint)
        at_rock_sample = bf('at_rock_sample', Waypoint)
        visible_from = bf('visible_from', Objective, Waypoint)
        store_of = bf('store_of', Store, Rover)
        calib_target = bf('calibration_target', Camera, Objective)
        on_board = bf('on_board', Camera, Rover)
        channel_free = bf('channel_free', Lander)
        in_sun = bf('in_sun', Waypoint)

        bool_fluents = {
            'in': in_f, 'at_lander': at_lander, 'can_traverse': can_traverse,
            'equipped_for_soil_analysis': equipped_soil,
            'equipped_for_rock_analysis': equipped_rock,
            'equipped_for_imaging': equipped_img,
            'empty': empty, 'have_rock_analysis': have_rock,
            'have_soil_analysis': have_soil, 'full': full,
            'calibrated': calibrated, 'supports': supports,
            'available': available, 'visible': visible,
            'have_image': have_image,
            'communicated_soil_data': comm_soil,
            'communicated_rock_data': comm_rock,
            'communicated_image_data': comm_image,
            'at_soil_sample': at_soil_sample, 'at_rock_sample': at_rock_sample,
            'visible_from': visible_from, 'store_of': store_of,
            'calibration_target': calib_target, 'on_board': on_board,
            'channel_free': channel_free, 'in_sun': in_sun,
        }

        for f in bool_fluents.values():
            problem.add_fluent(f, default_initial_value=False)

        # Valors inicials numerics
        for (fname, args), val in numeric.items():
            if fname == 'energy' and args:
                r = obj_map.get(args[0])
                if r:
                    problem.set_initial_value(energy(r), Int(val))
            elif fname == 'total_cost':
                problem.set_initial_value(total_cost(), Int(val))

        # Valors inicials booleans
        pred_name_map = {
            'in': 'in', 'at_lander': 'at_lander', 'can_traverse': 'can_traverse',
            'equipped_for_soil_analysis': 'equipped_for_soil_analysis',
            'equipped_for_rock_analysis': 'equipped_for_rock_analysis',
            'equipped_for_imaging': 'equipped_for_imaging',
            'empty': 'empty', 'have_rock_analysis': 'have_rock_analysis',
            'have_soil_analysis': 'have_soil_analysis', 'full': 'full',
            'calibrated': 'calibrated', 'supports': 'supports',
            'available': 'available', 'visible': 'visible',
            'have_image': 'have_image',
            'communicated_soil_data': 'communicated_soil_data',
            'communicated_rock_data': 'communicated_rock_data',
            'communicated_image_data': 'communicated_image_data',
            'at_soil_sample': 'at_soil_sample', 'at_rock_sample': 'at_rock_sample',
            'visible_from': 'visible_from', 'store_of': 'store_of',
            'calibration_target': 'calibration_target', 'on_board': 'on_board',
            'channel_free': 'channel_free', 'in_sun': 'in_sun',
        }

        for (pred, args) in bool_facts:
            f = bool_fluents.get(pred)
            if f is None:
                continue
            arg_objs = [obj_map[a] for a in args if a in obj_map]
            if len(arg_objs) == len(args):
                problem.set_initial_value(f(*arg_objs), True)

        # ===== ACTIONS =====

        # navigate(?x rover, ?y waypoint, ?z waypoint)
        navigate = InstantaneousAction('navigate', x=Rover, y=Waypoint, z=Waypoint)
        x, y, z = navigate.parameter('x'), navigate.parameter('y'), navigate.parameter('z')
        navigate.add_precondition(can_traverse(x, y, z))
        navigate.add_precondition(available(x))
        navigate.add_precondition(in_f(x, y))
        navigate.add_precondition(visible(y, z))
        navigate.add_precondition(GE(energy(x), Int(8)))
        navigate.add_decrease_effect(energy(x), Int(8))
        navigate.add_effect(in_f(x, y), False)
        navigate.add_effect(in_f(x, z), True)
        problem.add_action(navigate)

        # recharge(?x rover, ?w waypoint)
        recharge = InstantaneousAction('recharge', x=Rover, w=Waypoint)
        x, w = recharge.parameter('x'), recharge.parameter('w')
        recharge.add_precondition(in_f(x, w))
        recharge.add_precondition(in_sun(w))
        recharge.add_precondition(LE(energy(x), Int(80)))
        recharge.add_increase_effect(energy(x), Int(20))
        recharge.add_increase_effect(total_cost(), Int(1))
        problem.add_action(recharge)

        # samplesoil(?x rover, ?s store, ?p waypoint)
        samplesoil = InstantaneousAction('samplesoil', x=Rover, s=Store, p=Waypoint)
        x, s, p = samplesoil.parameter('x'), samplesoil.parameter('s'), samplesoil.parameter('p')
        samplesoil.add_precondition(in_f(x, p))
        samplesoil.add_precondition(GE(energy(x), Int(3)))
        samplesoil.add_precondition(at_soil_sample(p))
        samplesoil.add_precondition(equipped_soil(x))
        samplesoil.add_precondition(store_of(s, x))
        samplesoil.add_precondition(empty(s))
        samplesoil.add_decrease_effect(energy(x), Int(3))
        samplesoil.add_effect(empty(s), False)
        samplesoil.add_effect(full(s), True)
        samplesoil.add_effect(have_soil(x, p), True)
        samplesoil.add_effect(at_soil_sample(p), False)
        problem.add_action(samplesoil)

        # samplerock(?x rover, ?s store, ?p waypoint)
        samplerock = InstantaneousAction('samplerock', x=Rover, s=Store, p=Waypoint)
        x, s, p = samplerock.parameter('x'), samplerock.parameter('s'), samplerock.parameter('p')
        samplerock.add_precondition(in_f(x, p))
        samplerock.add_precondition(GE(energy(x), Int(5)))
        samplerock.add_precondition(at_rock_sample(p))
        samplerock.add_precondition(equipped_rock(x))
        samplerock.add_precondition(store_of(s, x))
        samplerock.add_precondition(empty(s))
        samplerock.add_decrease_effect(energy(x), Int(5))
        samplerock.add_effect(empty(s), False)
        samplerock.add_effect(full(s), True)
        samplerock.add_effect(have_rock(x, p), True)
        samplerock.add_effect(at_rock_sample(p), False)
        problem.add_action(samplerock)

        # drop(?x rover, ?y store)
        drop = InstantaneousAction('drop', x=Rover, y=Store)
        x, y = drop.parameter('x'), drop.parameter('y')
        drop.add_precondition(store_of(y, x))
        drop.add_precondition(full(y))
        drop.add_effect(full(y), False)
        drop.add_effect(empty(y), True)
        problem.add_action(drop)

        # calibrate(?r rover, ?i camera, ?t objective, ?w waypoint)
        calibrate = InstantaneousAction('calibrate', r=Rover, i=Camera, t=Objective, w=Waypoint)
        r, i, t, w = calibrate.parameter('r'), calibrate.parameter('i'), calibrate.parameter('t'), calibrate.parameter('w')
        calibrate.add_precondition(equipped_img(r))
        calibrate.add_precondition(GE(energy(r), Int(2)))
        calibrate.add_precondition(calib_target(i, t))
        calibrate.add_precondition(in_f(r, w))
        calibrate.add_precondition(visible_from(t, w))
        calibrate.add_precondition(on_board(i, r))
        calibrate.add_decrease_effect(energy(r), Int(2))
        calibrate.add_effect(calibrated(i, r), True)
        problem.add_action(calibrate)

        # takeimage(?r rover, ?p waypoint, ?o objective, ?i camera, ?m mode)
        takeimage = InstantaneousAction('takeimage', r=Rover, p=Waypoint, o=Objective, i=Camera, m=Mode)
        r, p, o, i, m = (takeimage.parameter('r'), takeimage.parameter('p'),
                         takeimage.parameter('o'), takeimage.parameter('i'), takeimage.parameter('m'))
        takeimage.add_precondition(calibrated(i, r))
        takeimage.add_precondition(on_board(i, r))
        takeimage.add_precondition(equipped_img(r))
        takeimage.add_precondition(supports(i, m))
        takeimage.add_precondition(visible_from(o, p))
        takeimage.add_precondition(in_f(r, p))
        takeimage.add_precondition(GE(energy(r), Int(1)))
        takeimage.add_decrease_effect(energy(r), Int(1))
        takeimage.add_effect(have_image(r, o, m), True)
        takeimage.add_effect(calibrated(i, r), False)
        problem.add_action(takeimage)

        # communicatesoildata(?r rover, ?l lander, ?p waypoint, ?x waypoint, ?y waypoint)
        commsoil = InstantaneousAction('communicatesoildata', r=Rover, l=Lander, p=Waypoint, x=Waypoint, y=Waypoint)
        r, l, p, x, y = (commsoil.parameter('r'), commsoil.parameter('l'), commsoil.parameter('p'),
                          commsoil.parameter('x'), commsoil.parameter('y'))
        commsoil.add_precondition(in_f(r, x))
        commsoil.add_precondition(at_lander(l, y))
        commsoil.add_precondition(have_soil(r, p))
        commsoil.add_precondition(visible(x, y))
        commsoil.add_precondition(available(r))
        commsoil.add_precondition(channel_free(l))
        commsoil.add_precondition(GE(energy(r), Int(4)))
        commsoil.add_decrease_effect(energy(r), Int(4))
        commsoil.add_effect(comm_soil(p), True)
        commsoil.add_effect(available(r), True)
        problem.add_action(commsoil)

        # communicaterockdata(?r rover, ?l lander, ?p waypoint, ?x waypoint, ?y waypoint)
        commrock = InstantaneousAction('communicaterockdata', r=Rover, l=Lander, p=Waypoint, x=Waypoint, y=Waypoint)
        r, l, p, x, y = (commrock.parameter('r'), commrock.parameter('l'), commrock.parameter('p'),
                          commrock.parameter('x'), commrock.parameter('y'))
        commrock.add_precondition(in_f(r, x))
        commrock.add_precondition(at_lander(l, y))
        commrock.add_precondition(have_rock(r, p))
        commrock.add_precondition(GE(energy(r), Int(4)))
        commrock.add_precondition(visible(x, y))
        commrock.add_precondition(available(r))
        commrock.add_precondition(channel_free(l))
        commrock.add_decrease_effect(energy(r), Int(4))
        commrock.add_effect(comm_rock(p), True)
        commrock.add_effect(available(r), True)
        problem.add_action(commrock)

        # communicateimagedata(?r rover, ?l lander, ?o objective, ?m mode, ?x waypoint, ?y waypoint)
        commimg = InstantaneousAction('communicateimagedata', r=Rover, l=Lander, o=Objective, m=Mode, x=Waypoint, y=Waypoint)
        r, l, o, m, x, y = (commimg.parameter('r'), commimg.parameter('l'), commimg.parameter('o'),
                              commimg.parameter('m'), commimg.parameter('x'), commimg.parameter('y'))
        commimg.add_precondition(in_f(r, x))
        commimg.add_precondition(at_lander(l, y))
        commimg.add_precondition(have_image(r, o, m))
        commimg.add_precondition(visible(x, y))
        commimg.add_precondition(available(r))
        commimg.add_precondition(channel_free(l))
        commimg.add_precondition(GE(energy(r), Int(6)))
        commimg.add_decrease_effect(energy(r), Int(6))
        commimg.add_effect(comm_image(o, m), True)
        commimg.add_effect(available(r), True)
        problem.add_action(commimg)

        # ===== GOALS =====
        goal_fluent_map = {
            'communicated_soil_data': comm_soil,
            'communicated_rock_data': comm_rock,
            'communicated_image_data': comm_image,
        }
        for pred, args in goals:
            f = goal_fluent_map.get(pred)
            if f:
                arg_objs = [obj_map[a] for a in args if a in obj_map]
                if len(arg_objs) == len(args):
                    problem.add_goal(f(*arg_objs))

        # ===== METRIC =====
        if metric and metric[0] == 'minimize':
            if metric[1] == 'total_cost':
                problem.add_quality_metric(
                    MinimizeExpressionOnFinalState(total_cost())
                )

        return problem


DOMAIN = RoverDomain()