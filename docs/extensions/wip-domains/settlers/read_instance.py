#!/usr/bin/env python3
"""
Parser for Settlers PDDL instances.
Extracts the problem structure from PDDL files.
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set


def parse_init_facts(init_block: str) -> Dict:
    """Parse the :init block to extract facts and fluent values.

    Uses a simple atom extraction (all parenthesis-balanced atoms without nested
    parenthesis) so we correctly handle tokens that are split across lines.
    """
    facts = {
        'places': set(),
        'vehicles': set(),
        'place_properties': {},  # {place: [properties]}
        'resources': {},  # {(store, resource): value}
        'space_in': {},  # {vehicle: value}
        'global_values': {},  # {'labour', 'pollution', 'resource-use'}
        'housing': {},  # {place: value}
        'connections': {
            'by_land': set(),
            'by_rail': set(),
            'by_sea': set()
        },
        'vehicle_properties': {},  # {vehicle: type}
        'potential_vehicles': set(),
        'is_at': [],  # list of (vehicle, place)
        'has_predicates': {}  # predicate -> set(locations)
    }

    # Normalize whitespace
    s = init_block
    # Extract all simple atoms like '(woodland location0)' or '(= (housing location0) 0)'
    atoms = re.findall(r"\([^()]*\)", s)

    for atom in atoms:
        atom = atom.strip()
        # skip empty and comments
        if atom.startswith(';') or atom == '()':
            continue

        # handle equality/assignment atoms: "(= (housing location0) 0)"
        # or "(= (available wood location0) 0)" or "(= (labour) 0)"
        m_assign = re.match(r"^\(= \(([^)]+)\)\s+(-?\d+)\)$", atom)
        if m_assign:
            inner = m_assign.group(1).strip()  # like 'housing location0' or 'available wood location0' or 'labour'
            value = int(m_assign.group(2))
            parts = inner.split()
            if parts[0] == 'housing' and len(parts) == 2:
                place = parts[1]
                facts['places'].add(place)
                facts['housing'][place] = value
            elif parts[0] == 'available':
                # available <resource> <store>
                if len(parts) == 3:
                    resource = parts[1]
                    store = parts[2]
                    if store.startswith('location'):
                        facts['places'].add(store)
                    elif store.startswith('vehicle'):
                        facts['vehicles'].add(store)
                    facts['resources'][(store, resource)] = value
            elif parts[0] in ('labour', 'pollution', 'resource-use'):
                facts['global_values'][parts[0]] = value
            elif parts[0] == 'space-in' and len(parts) == 2:
                # assignment like '(= (space-in vehicle0) 0)'
                v = parts[1]
                facts['vehicles'].add(v)
                facts['space_in'][v] = value
            else:
                # fallback: ignore other assignments for now
                pass
            continue

        # handle facts like '(woodland location0)' or '(mountain location0)'
        m_prop = re.match(r"^\((woodland|mountain|metalliferous|by-coast)\s+(location\d+)\)$", atom)
        if m_prop:
            prop = m_prop.group(1)
            location = m_prop.group(2)
            facts['places'].add(location)
            facts['place_properties'].setdefault(location, []).append(prop)
            continue

        # boolean has-* predicates e.g. '(has-cabin location0)'
        m_has = re.match(r"^\((has-[a-z-]+)\s+(location\d+)\)$", atom)
        if m_has:
            pred = m_has.group(1)
            loc = m_has.group(2)
            facts['places'].add(loc)
            facts['has_predicates'].setdefault(pred, set()).add(loc)
            continue

        # is-at: '(is-at vehicle0 location0)'
        m_isat = re.match(r"^\(is-at\s+(vehicle\d+)\s+(location\d+)\)$", atom)
        if m_isat:
            v = m_isat.group(1)
            l = m_isat.group(2)
            facts['vehicles'].add(v)
            facts['places'].add(l)
            facts['is_at'].append((v, l))
            continue

        # space-in vehicle: fallback if simple atom appears like '(space-in vehicle0)'
        m_space = re.match(r"^\(space-in\s+(vehicle\d+)\)$", atom)
        if m_space:
            v = m_space.group(1)
            facts['vehicles'].add(v)
            facts['space_in'].setdefault(v, 0)
            continue

        # vehicle properties like '(is-cart vehicle0)'
        m_vehicle_type = re.match(r"^\(is-(cart|train|ship)\s+(vehicle\d+)\)$", atom)
        if m_vehicle_type:
            vtype = m_vehicle_type.group(1)
            vehicle = m_vehicle_type.group(2)
            facts['vehicles'].add(vehicle)
            facts['vehicle_properties'][vehicle] = vtype
            continue

        # potential vehicles: '(potential vehicle0)'
        m_pot = re.match(r"^\(potential\s+(vehicle\d+)\)$", atom)
        if m_pot:
            facts['potential_vehicles'].add(m_pot.group(1))
            facts['vehicles'].add(m_pot.group(1))
            continue

        # connected-by-* pairs
        m_land = re.match(r"^\(connected-by-land\s+(location\d+)\s+(location\d+)\)$", atom)
        if m_land:
            facts['places'].add(m_land.group(1))
            facts['places'].add(m_land.group(2))
            facts['connections']['by_land'].add((m_land.group(1), m_land.group(2)))
            continue

        m_rail = re.match(r"^\(connected-by-rail\s+(location\d+)\s+(location\d+)\)$", atom)
        if m_rail:
            facts['places'].add(m_rail.group(1))
            facts['places'].add(m_rail.group(2))
            facts['connections']['by_rail'].add((m_rail.group(1), m_rail.group(2)))
            continue

        m_sea = re.match(r"^\(connected-by-sea\s+(location\d+)\s+(location\d+)\)$", atom)
        if m_sea:
            facts['places'].add(m_sea.group(1))
            facts['places'].add(m_sea.group(2))
            facts['connections']['by_sea'].add((m_sea.group(1), m_sea.group(2)))
            continue

        # other cases ignored for now

    return facts


def parse_goal(goal_block: str) -> List[str]:
    """Parse the :goal block to extract goal conditions.

    This function extracts the top-level subexpressions inside the (:goal ...) form.
    It returns each child as a string (including its parentheses), e.g.
    ['(>= (housing location0) 2)', '(has-coal-stack location0)']
    """
    goals = []
    s = goal_block.strip()
    # remove wrapping whitespace and any leading '(and' if present
    # If the block starts with '(and', we want to extract children inside it.
    if s.startswith('(') and s.endswith(')'):
        # keep inner content only
        s = s[1:-1].strip()
    if s.startswith('and'):
        s = s[3:].strip()

    # now scan s and extract top-level parenthesized expressions
    depth = 0
    token = ''
    for ch in s:
        if ch == '(':
            depth += 1
            token += ch
        elif ch == ')':
            token += ch
            depth -= 1
            if depth == 0:
                goals.append(token.strip())
                token = ''
        else:
            if depth > 0:
                token += ch
            else:
                # skip whitespace between top-level tokens
                pass

    # fallback: if no parentheses were found, try to find atoms
    if not goals:
        goals = re.findall(r"\([^()]+\)", goal_block)

    return goals


def parse_pddl_file(filepath: str) -> Dict:
    """Parse a PDDL problem file and extract all relevant information."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract init block - match from (:init to the closing paren
    # We need to be careful with nested parentheses
    init_start = content.find('(:init')
    if init_start != -1:
        # Find the matching closing paren
        depth = 0
        init_end = init_start
        for i in range(init_start, len(content)):
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                depth -= 1
                if depth == 0:
                    init_end = i
                    break
        init_block = content[init_start + 6:init_end]
    else:
        init_block = ""

    # Extract goal block
    goal_start = content.find('(:goal')
    if goal_start != -1:
        # Find the matching closing paren
        depth = 0
        goal_end = goal_start
        for i in range(goal_start, len(content)):
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                depth -= 1
                if depth == 0:
                    goal_end = i
                    break
        goal_block = content[goal_start + 6:goal_end]
    else:
        goal_block = ""

    # Parse init facts
    facts = parse_init_facts(init_block)

    # Parse goals
    goals = parse_goal(goal_block)

    return {
        'facts': facts,
        'goals': goals
    }


def serialize_data(data: Dict) -> Tuple[str, str, str]:
    """
    Serialize the parsed data into a format suitable for Unified Planning.
    Returns three strings: places, vehicles, problem_data
    """
    facts = data['facts']
    goals = data['goals']

    # Serialize places
    places_list = sorted(facts['places'])
    places_str = repr(places_list)

    # Serialize vehicles
    vehicles_list = sorted(facts['vehicles'])
    vehicles_str = repr(vehicles_list)

    # Serialize all problem data
    problem_data = {
        'places': places_list,
        'vehicles': vehicles_list,
        'place_properties': {k: facts['place_properties'].get(k, []) for k in places_list},
        'vehicle_properties': {k: facts['vehicle_properties'].get(k, None) for k in vehicles_list},
        'resources': dict(facts['resources']),
        'space_in': dict(facts['space_in']),
        'global_values': dict(facts['global_values']),
        'housing': {k: facts['housing'].get(k, 0) for k in places_list},
        'connections': {
            'by_land': [list(conn) for conn in facts['connections']['by_land']],
            'by_rail': [list(conn) for conn in facts['connections']['by_rail']],
            'by_sea': [list(conn) for conn in facts['connections']['by_sea']]
        },
        'potential_vehicles': sorted(facts['potential_vehicles']),
        'is_at': facts['is_at'],
        'has_predicates': {k: sorted(list(v)) for k, v in facts['has_predicates'].items()},
        'goals': goals
    }

    return places_str, vehicles_str, repr(problem_data)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: read_instance.py <instance_name>")
        sys.exit(1)

    instance_name = sys.argv[1]
    filepath = f'/Users/cds26/PycharmProjects/BitBlast/benchmark/Settlers/instances/{instance_name}.pddl'

    try:
        data = parse_pddl_file(filepath)
        places_str, vehicles_str, problem_data_str = serialize_data(data)

        # Output in the format expected by Settlers.py
        print(places_str)
        print("---")
        print(vehicles_str)
        print("---")
        print(problem_data_str)
    except Exception as e:
        print(f"Error parsing {instance_name}: {e}", file=sys.stderr)
        sys.exit(1)

