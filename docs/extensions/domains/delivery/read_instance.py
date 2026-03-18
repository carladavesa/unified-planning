import sys
import os
import re

def parse_pddl_instance(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove comments
    content = re.sub(r';.*', '', content)
    # Normalise whitespace
    content = re.sub(r'\s+', ' ', content).strip()

    # ------------------------------------------------------------------ objects
    objects = {'room': [], 'item': [], 'bot': [], 'arm': []}
    obj_match = re.search(r'\(:objects(.*?)\)', content, re.IGNORECASE)
    if obj_match:
        obj_str = obj_match.group(1).strip()
        # Split on '-' separators: "a b c - type d e - type2 ..."
        segments = re.split(r'-\s*(\w+)', obj_str)
        # segments = [names_chunk, type, names_chunk, type, ...]
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

    # Numeric assignments  (= (fluent args) value)
    numeric = {}
    for m in re.finditer(r'\(=\s*\((\w[\w-]*)\s+([^)]*?)\)\s+([\d.]+)\)', init_str):
        fluent = m.group(1).replace('-', '_')
        args   = m.group(2).strip().split()
        value  = int(m.group(3)) if '.' not in m.group(3) else float(m.group(3))
        key    = (fluent, tuple(args))
        numeric[key] = value

    def get_numeric(fluent, *args):
        return numeric.get((fluent, args), None)

    # Boolean predicates  (pred arg1 arg2 ...)
    bool_facts = []
    # Remove numeric assignments first to avoid false positives
    init_no_numeric = re.sub(r'\(=\s*\([^)]*\)\s+[\d.]+\)', '', init_str)
    for m in re.finditer(r'\((\w[\w-]*)((?:\s+\w+)*)\)', init_no_numeric):
        pred = m.group(1).replace('-', '_')
        args = m.group(2).strip().split()
        bool_facts.append((pred, args))

    # Organise boolean facts into named structures
    at_bot  = {args[0]: args[1] for pred, args in bool_facts if pred == 'at_bot'  and len(args) == 2}
    at      = {args[0]: args[1] for pred, args in bool_facts if pred == 'at'      and len(args) == 2}
    free    = [args[0]          for pred, args in bool_facts if pred == 'free'     and len(args) == 1]
    mount   = {args[0]: args[1] for pred, args in bool_facts if pred == 'mount'    and len(args) == 2}
    door    = [(args[0], args[1]) for pred, args in bool_facts if pred == 'door'   and len(args) == 2]

    # Numeric fluents per object
    weight       = {args[0]: get_numeric('weight', args[0])
                    for args in [a for p, a in bool_facts if False]  # placeholder
                   }
    # Re-extract weight / load from numeric dict
    weights      = {a[0]: v for (f, a), v in numeric.items() if f == 'weight'}
    load_limits  = {a[0]: v for (f, a), v in numeric.items() if f == 'load_limit'}
    current_load = {a[0]: v for (f, a), v in numeric.items() if f == 'current_load'}

    # ------------------------------------------------------------------ goal
    goal_match = re.search(r'\(:goal\s*\(and(.*?)\)\s*\)', content, re.IGNORECASE | re.DOTALL)
    if not goal_match:
        goal_match = re.search(r'\(:goal\s*(.*?)\)\s*\(:metric', content, re.IGNORECASE | re.DOTALL)
    goals = []
    if goal_match:
        goal_str = goal_match.group(1)
        for m in re.finditer(r'\((\w[\w-]*)((?:\s+\w+)*)\)', goal_str):
            pred = m.group(1).replace('-', '_')
            args = m.group(2).strip().split()
            goals.append((pred, args))

    at_goals = [(args[0], args[1]) for pred, args in goals if pred == 'at' and len(args) == 2]

    # ------------------------------------------------------------------ summary
    result = {
        'objects':       objects,
        'at_bot':        at_bot,
        'at':            at,
        'free':          free,
        'mount':         mount,
        'door':          door,
        'weights':       weights,
        'load_limits':   load_limits,
        'current_load':  current_load,
        'goals_at':      at_goals,
    }
    return result


def print_result(result):
    sep = "---"

    obj = result['objects']
    print(f"rooms={','.join(obj['room'])}")
    print(f"items={','.join(obj['item'])}")
    print(f"bots={','.join(obj['bot'])}")
    print(f"arms={','.join(obj['arm'])}")
    print(sep)

    for bot, room in result['at_bot'].items():
        print(f"at_bot({bot},{room})")
    print(sep)

    for item, room in result['at'].items():
        print(f"at({item},{room})")
    print(sep)

    for arm in result['free']:
        print(f"free({arm})")
    print(sep)

    for arm, bot in result['mount'].items():
        print(f"mount({arm},{bot})")
    print(sep)

    for r1, r2 in result['door']:
        print(f"door({r1},{r2})")
    print(sep)

    for item, w in result['weights'].items():
        print(f"weight({item})={w}")
    print(sep)

    for bot, ll in result['load_limits'].items():
        print(f"load_limit({bot})={ll}")
    for bot, cl in result['current_load'].items():
        print(f"current_load({bot})={cl}")
    print(sep)

    for item, room in result['goals_at']:
        print(f"goal_at({item},{room})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 read_instance.py <instance_name_or_path>", file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    print(f"Reading {name}")
    script_dir = f'/Users/cds26/PycharmProjects/unified-planning/docs/extensions/domains/delivery/instances-pddl'
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