import sys
import os
import re


def parse_pddl_instance(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    content = re.sub(r';.*', '', content)
    content = re.sub(r'\s+', ' ', content).strip()

    # ------------------------------------------------------------------ objects
    objects = {'plant': [], 'tap': [], 'agent': []}
    obj_match = re.search(r'\(:objects(.*?)\)', content, re.IGNORECASE)
    if obj_match:
        obj_str = obj_match.group(1).strip()
        segments = re.split(r'-\s*(\w+)', obj_str)
        i = 0
        while i + 1 < len(segments):
            names = segments[i].split()
            type_name = segments[i + 1].strip().lower()
            if type_name in objects:
                objects[type_name].extend(names)
            i += 2

    # ------------------------------------------------------------------ init
    init_match = re.search(r'\(:init(.*?)\)\s*\(:goal', content, re.IGNORECASE | re.DOTALL)
    init_str = init_match.group(1).strip() if init_match else ""

    # Numeric assignments: (= (fluent) value) i (= (fluent arg) value)
    numeric_global = {}   # fluent -> value           ex: (= (maxx) 10)
    numeric_obj    = {}   # (fluent, obj) -> value    ex: (= (x plant1) 5)

    for m in re.finditer(r'\(=\s*\((\w+)\s*(\w*)\)\s*([\d.]+)\)', init_str):
        fluent = m.group(1).lower()
        arg    = m.group(2).strip()
        value  = int(m.group(3)) if '.' not in m.group(3) else float(m.group(3))
        if arg:
            numeric_obj[(fluent, arg)] = value
        else:
            numeric_global[fluent] = value

    # ------------------------------------------------------------------ goal
    goal_match = re.search(r'\(:goal\s*\(and(.*)', content, re.IGNORECASE | re.DOTALL)
    goal_str = goal_match.group(1) if goal_match else ""

    # (= (poured plantX) N)
    poured_goals = {}
    for m in re.finditer(r'\(=\s*\(poured\s+(\w+)\)\s*([\d]+)\)', goal_str):
        poured_goals[m.group(1)] = int(m.group(2))

    # (= (total_poured) (total_loaded))
    balance_goal = bool(re.search(r'\(=\s*\(total_poured\)\s*\(total_loaded\)\)', goal_str))

    return {
        'objects':        objects,
        'numeric_global': numeric_global,
        'numeric_obj':    numeric_obj,
        'poured_goals':   poured_goals,
        'balance_goal':   balance_goal,
    }


def print_result(result):
    sep = "---"

    obj = result['objects']
    print(f"plants={','.join(obj['plant'])}")
    print(f"taps={','.join(obj['tap'])}")
    print(f"agents={','.join(obj['agent'])}")
    print(sep)

    g = result['numeric_global']
    for key in ['maxx', 'minx', 'maxy', 'miny', 'water_reserve', 'total_poured', 'total_loaded']:
        if key in g:
            print(f"{key}={g[key]}")
    print(sep)

    for (fluent, obj_name), val in result['numeric_obj'].items():
        print(f"{fluent}({obj_name})={val}")
    print(sep)

    for plant, val in result['poured_goals'].items():
        print(f"goal_poured({plant})={val}")
    if result['balance_goal']:
        print("goal_balance=true")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 read_instance.py <instance_name_or_path>", file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    script_dir = f'/Users/cds26/PycharmProjects/unified-planning/docs/extensions/domains/watering/instances-pddl'

    for candidate in [
        os.path.join(script_dir, f"{name}.pddl"),
        os.path.join(script_dir, name),
    ]:
        if os.path.isfile(candidate):
            path = candidate
            break
    else:
        print(f"File not found: {name}", file=sys.stderr)
        sys.exit(1)

    result = parse_pddl_instance(path)
    print_result(result)