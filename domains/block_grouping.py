import os
import re
from typing import Dict, Optional, List, Tuple
from unified_planning.model import Action, Expression, Object
from unified_planning.shortcuts import (
    Fluent, Int, IntType, InstantaneousAction, MinimizeActionCosts,
    Problem, UserType, Minus, Plus, LE, GE, Equals, Not, Or, And,
)
from domains.base import Domain

PDDL_DIR = os.path.join(os.path.dirname(__file__), 'block_grouping/handcrafted')

class BlockGroupingDomain(Domain):
    def __init__(self) -> None:
        self._instances = self._discover_instances()

    def _discover_instances(self) -> List[str]:
        if not os.path.exists(PDDL_DIR):
            return []
        return sorted([
            f[:-5] for f in os.listdir(PDDL_DIR) if f.endswith('.pddl')
        ])

    def list_instances(self) -> dict[str, dict]:
        return {k: {} for k in self._instances}

    def _parse_pddl(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            content = f.read()

        content = re.sub(r';.*', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Objects
        blocks = []
        obj_match = re.search(r'\(:objects(.*?)\)', content, re.IGNORECASE)
        if obj_match:
            obj_str = obj_match.group(1).strip()
            segments = re.split(r'-\s*(\w+)', obj_str)
            i = 0
            while i + 1 < len(segments):
                names = segments[i].split()
                type_name = segments[i + 1].strip().lower()
                if type_name == 'block':
                    blocks.extend(names)
                i += 2

        # Init
        init_match = re.search(r'\(:init(.*?)\)\s*\(:goal', content, re.IGNORECASE | re.DOTALL)
        init_str = init_match.group(1).strip() if init_match else ""

        numeric = {}
        for m in re.finditer(r'\(=\s*\((\w+)(?:\s+(\w+))?\)\s+([\d.]+)\)', init_str):
            fluent = m.group(1)
            arg = m.group(2)  # None if no argument (maxx, minx, etc.)
            value = int(m.group(3))
            numeric[(fluent, arg)] = value

        xy = {}  # {block: {'x': val, 'y': val}}
        for (fluent, arg), val in numeric.items():
            if fluent in ('x', 'y') and arg:
                xy.setdefault(arg, {})[fluent] = val

        maxx = numeric.get(('maxx', None), 20)
        minx = numeric.get(('minx', None), 1)
        maxy = numeric.get(('maxy', None), 20)
        miny = numeric.get(('miny', None), 1)

        # Goals - parse raw goal string
        goal_match = re.search(r'\(:goal\s*\(and(.*)\)\s*\)', content, re.IGNORECASE | re.DOTALL)
        goal_str = goal_match.group(1).strip() if goal_match else ""

        # Equal pairs: (= (x bi) (x bj)) and (= (y bi) (y bj))
        equal_x = []
        equal_y = []
        for m in re.finditer(r'\(=\s*\(x\s+(\w+)\)\s*\(x\s+(\w+)\)\)', goal_str):
            equal_x.append((m.group(1), m.group(2)))
        for m in re.finditer(r'\(=\s*\(y\s+(\w+)\)\s*\(y\s+(\w+)\)\)', goal_str):
            equal_y.append((m.group(1), m.group(2)))

        # Not-equal pairs: (or (not (= (x bi) (x bj))) (not (= (y bi) (y bj))))
        not_equal = []
        for m in re.finditer(
            r'\(or\s*\(not\s*\(=\s*\(x\s+(\w+)\)\s*\(x\s+(\w+)\)\)\)\s*\(not\s*\(=\s*\(y\s+(\w+)\)\s*\(y\s+(\w+)\)\)\)\)',
            goal_str
        ):
            not_equal.append((m.group(1), m.group(2)))

        return {
            'blocks': blocks,
            'xy': xy,
            'maxx': maxx, 'minx': minx,
            'maxy': maxy, 'miny': miny,
            'equal_x': equal_x,
            'equal_y': equal_y,
            'not_equal': not_equal,
        }

    def get_instance(self, instance: Optional[str] = None) -> dict:
        if instance and instance in self._instances:
            path = os.path.join(PDDL_DIR, f"{instance}.pddl")
            return self._parse_pddl(path)
        raise ValueError(f"Instance '{instance}' not found!")

    def build_problem(self, instance: str | None = None) -> "Problem":
        data = self.get_instance(instance)
        blocks = data['blocks']
        xy = data['xy']
        maxx, minx = data['maxx'], data['minx']
        maxy, miny = data['maxy'], data['miny']
        max_int_n = max(maxx, maxy)

        problem = Problem('block_grouping_problem')
        Block = UserType('Block')

        x = Fluent('x', IntType(0, max_int_n), b=Block)
        y = Fluent('y', IntType(0, max_int_n), b=Block)
        problem.add_fluent(x)
        problem.add_fluent(y)

        # Objects
        for b in blocks:
            problem.add_object(Object(b, Block))

        def obj(name): return problem.object(name)

        # Init
        for b in blocks:
            problem.set_initial_value(x(obj(b)), xy[b]['x'])
            problem.set_initial_value(y(obj(b)), xy[b]['y'])

        # Actions
        move_up = InstantaneousAction('move_up', b=Block)
        b = move_up.parameter('b')
        move_up.add_precondition(LE(Plus(y(b), 1), maxy))
        move_up.add_increase_effect(y(b), 1)

        move_down = InstantaneousAction('move_down', b=Block)
        b = move_down.parameter('b')
        move_down.add_precondition(GE(Minus(y(b), 1), miny))
        move_down.add_decrease_effect(y(b), 1)

        move_right = InstantaneousAction('move_right', b=Block)
        b = move_right.parameter('b')
        move_right.add_precondition(LE(Plus(x(b), 1), maxx))
        move_right.add_increase_effect(x(b), 1)

        move_left = InstantaneousAction('move_left', b=Block)
        b = move_left.parameter('b')
        move_left.add_precondition(GE(Minus(x(b), 1), minx))
        move_left.add_decrease_effect(x(b), 1)

        problem.add_actions([move_up, move_down, move_right, move_left])

        # Goals
        for b1, b2 in data['equal_x']:
            problem.add_goal(Equals(x(obj(b1)), x(obj(b2))))
        for b1, b2 in data['equal_y']:
            problem.add_goal(Equals(y(obj(b1)), y(obj(b2))))
        for b1, b2 in data['not_equal']:
            problem.add_goal(Or(Not(Equals(x(obj(b1)), x(obj(b2)))), Not(Equals(y(obj(b1)), y(obj(b2))))))

        costs: Dict[Action, Expression] = {
            move_up: Int(1), move_down: Int(1),
            move_right: Int(1), move_left: Int(1),
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem


DOMAIN = BlockGroupingDomain()