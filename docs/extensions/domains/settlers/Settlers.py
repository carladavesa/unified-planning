import subprocess
from unified_planning.shortcuts import *
import argparse
import ast
import re

# Parser
parser = argparse.ArgumentParser(description="Solve Settlers")
parser.add_argument('--instance', type=str, default='pfile1', help='Instance name (pfileX)')
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
instance_name = args.instance

# Call the read_instance parser
instance_path = f'/Users/cds26/PycharmProjects/BitBlast/benchmark/Settlers/read_instance.py'
proc = subprocess.run(['python3', instance_path, instance_name], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if proc.returncode != 0:
    print('Error calling read_instance:', proc.stderr)
    raise SystemExit(1)

parts = proc.stdout.split('---')
if len(parts) < 3:
    print('Unexpected parser output')
    print(proc.stdout)
    raise SystemExit(1)

places = ast.literal_eval(parts[0].strip())
vehicles = ast.literal_eval(parts[1].strip()) if parts[1].strip() else []
problem_data = ast.literal_eval(parts[2].strip())

# ---------------------------------------------------- Problem ---------------------------------------------------------
problem = Problem('settlers_problem')

Store = UserType('Store')
Resource = UserType('Resource')

# Create objects: both places and vehicles are Stores (for fluents like available/space-in)
for p in places:
    problem.add_object(Object(p, Store))
for v in vehicles:
    problem.add_object(Object(v, Store))

# Create resource objects (collect from problem_data resources and a fallback known list)
known_resources = {'ore', 'iron', 'stone', 'wood', 'timber', 'coal'}
resources_found = set()
for (store, res) in problem_data.get('resources', {}).keys():
    resources_found.add(res)
all_resources = sorted(list(resources_found.union(known_resources)))
for r in all_resources:
    problem.add_object(Object(r, Resource))

# Numeric fluents
space_in = Fluent('space_in', IntType(0, 100), l=Store)
housing = Fluent('housing', IntType(0, 100), l=Store)
labour = Fluent('labour', IntType(0, 1000))
pollution = Fluent('pollution', IntType(0, 1000))
resource_use = Fluent('resource_use', IntType(0, 1000))

# Single available fluent parameterized by resource and store
available = Fluent('available', IntType(0, 100), r=Resource, s=Store)

# Boolean fluents (modeled as fluent with bool semantics)
has_cabin = Fluent('has_cabin', BoolType(), l=Store)
has_coal_stack = Fluent('has_coal_stack', BoolType(), l=Store)
has_quarry = Fluent('has_quarry', BoolType(), l=Store)
has_mine = Fluent('has_mine', BoolType(), l=Store)
has_sawmill = Fluent('has_sawmill', BoolType(), l=Store)
has_ironworks = Fluent('has_ironworks', BoolType(), l=Store)
has_docks = Fluent('has_docks', BoolType(), l=Store)
has_wharf = Fluent('has_wharf', BoolType(), l=Store)

is_cart = Fluent('is_cart', BoolType(), v=Store)
is_train = Fluent('is_train', BoolType(), v=Store)
is_ship = Fluent('is_ship', BoolType(), v=Store)
potential = Fluent('potential', BoolType(), v=Store)

# Connections and position
connected_by_land = Fluent('connected_by_land', BoolType(), a=Store, b=Store)
connected_by_rail = Fluent('connected_by_rail', BoolType(), a=Store, b=Store)
connected_by_sea = Fluent('connected_by_sea', BoolType(), a=Store, b=Store)

is_at = Fluent('is_at', BoolType(), v=Store, l=Store)

# Register fluents
for f in [space_in, housing, labour, pollution, resource_use,
          available,
          has_cabin, has_coal_stack, has_quarry, has_mine, has_sawmill, has_ironworks, has_docks, has_wharf,
          is_cart, is_train, is_ship, potential,
          connected_by_land, connected_by_rail, connected_by_sea, is_at]:
    problem.add_fluent(f)

# ------------------------------------------------ Initialisation ------------------------------------------------------
# Set initial numeric and boolean values from problem_data
pd = problem_data
# global values
if 'global_values' in pd:
    gv = pd['global_values']
    if 'labour' in gv:
        problem.set_initial_value(labour, gv['labour'])
    if 'pollution' in gv:
        problem.set_initial_value(pollution, gv['pollution'])
    if 'resource-use' in gv:
        problem.set_initial_value(resource_use, gv['resource-use'])

# housing
for loc, val in pd.get('housing', {}).items():
    problem.set_initial_value(housing(problem.object(loc)), val)

# resources: pd stores keys as (store, resource)
for (store, res), val in pd.get('resources', {}).items():
    try:
        r_obj = problem.object(res)
        s_obj = problem.object(store)
        problem.set_initial_value(available(r_obj, s_obj), val)
    except Exception:
        # skip if some object missing
        pass

# space_in
for v, val in pd.get('space_in', {}).items():
    problem.set_initial_value(space_in(problem.object(v)), val)

# potentials
for v in pd.get('potential_vehicles', []):
    problem.set_initial_value(potential(problem.object(v)), True)

# vehicle properties (is-cart/is-train/is-ship)
for v, ptype in pd.get('vehicle_properties', {}).items():
    if ptype == 'cart':
        problem.set_initial_value(is_cart(problem.object(v)), True)
    elif ptype == 'train':
        problem.set_initial_value(is_train(problem.object(v)), True)
    elif ptype == 'ship':
        problem.set_initial_value(is_ship(problem.object(v)), True)

# place properties -> set connections and simple properties
for (a, b) in pd.get('connections', {}).get('by_land', []):
    problem.set_initial_value(connected_by_land(problem.object(a), problem.object(b)), True)
for (a, b) in pd.get('connections', {}).get('by_rail', []):
    problem.set_initial_value(connected_by_rail(problem.object(a), problem.object(b)), True)
for (a, b) in pd.get('connections', {}).get('by_sea', []):
    problem.set_initial_value(connected_by_sea(problem.object(a), problem.object(b)), True)

# is_at from parsed init
for (v, l) in pd.get('is_at', []):
    try:
        problem.set_initial_value(is_at(problem.object(v), problem.object(l)), True)
    except Exception:
        # ignore missing objects
        pass

# has-* predicates
HAS_MAP = {
    'has-cabin': has_cabin,
    'has-coal-stack': has_coal_stack,
    'has-quarry': has_quarry,
    'has-mine': has_mine,
    'has-sawmill': has_sawmill,
    'has-ironworks': has_ironworks,
    'has-docks': has_docks,
    'has-wharf': has_wharf,
}
for pred, locs in pd.get('has_predicates', {}).items():
    fluent = HAS_MAP.get(pred)
    if fluent:
        for loc in locs:
            problem.set_initial_value(fluent(problem.object(loc)), True)

# ---------------------------------------------------- Actions ---------------------------------------------------------
# Instead of hand-coding actions, parse the PDDL domain to create actions that
# mirror the PDDL definitions exactly (parameters, preconditions, effects).

domain_path = '/Users/cds26/PycharmProjects/BitBlast/benchmark/Settlers/instances/domain.pddl'
try:
    with open(domain_path, 'r') as f:
        domain_txt = f.read()
except FileNotFoundError:
    domain_txt = ''

# helper: map pddl types to UP types
TYPE_MAP = {
    'vehicle': Store,
    'place': Store,
    'store': Store,
    'resource': Resource,
    'object': Store
}

# mapping predicate names to UP fluents
PRED_MAP = {
    'is-at': is_at,
    'is-cart': is_cart,
    'is-train': is_train,
    'is-ship': is_ship,
    'potential': potential,
    'has-cabin': has_cabin,
    'has-coal-stack': has_coal_stack,
    'has-quarry': has_quarry,
    'has-mine': has_mine,
    'has-sawmill': has_sawmill,
    'has-ironworks': has_ironworks,
    'has-docks': has_docks,
    'has-wharf': has_wharf,
    'connected-by-land': connected_by_land,
    'connected-by-rail': connected_by_rail,
    'connected-by-sea': connected_by_sea,
}

# parse parameter list: '?v - vehicle ?p - place' -> [('v','vehicle'), ...]
def parse_params(params_str):
    pairs = re.findall(r'\?([^\s]+)\s*-\s*([^\s)]+)', params_str)
    return [(name, ptype) for name, ptype in pairs]

# extract actions using regex (simple approximation expects parameters, precondition and effect blocks)
actions = re.findall(r"\(:action\s+([^\s]+)\s*:parameters\s*\(([^)]*)\)\s*:precondition\s*(\(.*?\))\s*:effect\s*(\(.*?\))", domain_txt, re.DOTALL)

for act_name, params_str, precond_str, effect_str in actions:
    # prepare parameter dict for InstantaneousAction
    parsed_params = parse_params(params_str)
    # build kwargs for types: name without ? mapped to TYPE_MAP
    kw = {}
    for pname, ptype in parsed_params:
        py_name = pname
        up_type = TYPE_MAP.get(ptype, Store)
        kw[py_name] = up_type
    # create the action
    try:
        action = InstantaneousAction(act_name, **kw)
    except Exception:
        # fallback: skip action if something fails
        continue

    # get parameter references
    param_refs = {name: action.parameter(name) for name, _ in parsed_params}

    # handle preconditions
    # handle comparisons like (>= (housing ?p) 2) and simple atoms
    # GE comparisons
    for inner, val in re.findall(r"\(>=\s*\(([^)]+)\)\s*([\-\d]+)\)", precond_str):
        parts = inner.strip().split()
        if parts[0] == 'housing' and len(parts) == 2:
            p = parts[1].lstrip('?')
            action.add_precondition(GE(housing(resolve_arg(p, param_refs)), int(val)))
        elif parts[0] == 'available' and len(parts) == 3:
            r = parts[1].lstrip('?')
            s = parts[2].lstrip('?')
            action.add_precondition(GE(available(resolve_arg(r, param_refs), resolve_arg(s, param_refs)), int(val)))
        elif parts[0] in ('labour','pollution','resource-use'):
            # global comparison
            if parts[0] == 'labour':
                action.add_precondition(GE(labour, int(val)))
            elif parts[0] == 'pollution':
                action.add_precondition(GE(pollution, int(val)))
            elif parts[0] == 'resource-use':
                action.add_precondition(GE(resource_use, int(val)))

            action.add_precondition(GE(housing(param_refs[p]), int(val)))
    simple_atoms = re.findall(r"\(([a-z-]+)(?:\s+[^()]+)?\)", precond_str)
    # but the above will capture predicate names; better extract atomic expressions
            action.add_precondition(GE(available(param_refs[r], param_refs[s]), int(val)))
    for atom in atoms:
        atom = atom.strip()
        if atom.startswith('(and') or atom.startswith('(>='):
            continue
        # match predicate with args
        m = re.match(r"^\(([a-z-]+)\s+(.+)\)$", atom)
        if m:
            pred = m.group(1)
            args = m.group(2).split()
            args = [a.lstrip('?') for a in args]
            if pred in PRED_MAP:
                # build call using param_refs or concrete objects
                try:
                    expr = PRED_MAP[pred]
                    # call with parameters in order, resolving each argument
                    call_args = [resolve_arg(a, param_refs) for a in args]
                    action.add_precondition(expr(*call_args))
                except Exception:
                    pass
            else:
                # unknown predicate - try matching 'is-cart' style boolean with single arg
                pass

    # handle effects: increases/decreases and boolean add/delete
    # increases
    for inner, val in re.findall(r"\(increase\s*\(([^)]+)\)\s*([\-\d]+)\)", effect_str):
                # build call using param_refs
        if parts[0] == 'labour' and len(parts) == 1:
            action.add_increase_effect(labour, int(val))
                    # call with parameters in order
                    call_args = [param_refs[a] for a in args]
        elif parts[0] == 'resource-use' and len(parts) == 1:
            action.add_increase_effect(resource_use, int(val))
        elif parts[0] == 'available' and len(parts) == 3:
            r = parts[1].lstrip('?')
            s = parts[2].lstrip('?')
            action.add_increase_effect(available(resolve_arg(r, param_refs), resolve_arg(s, param_refs)), int(val))
    # decreases
    for inner, val in re.findall(r"\(decrease\s*\(([^)]+)\)\s*([\-\d]+)\)", effect_str):
        parts = inner.strip().split()
        if parts[0] == 'available' and len(parts) == 3:
            r = parts[1].lstrip('?')
            s = parts[2].lstrip('?')
            action.add_decrease_effect(available(resolve_arg(r, param_refs), resolve_arg(s, param_refs)), int(val))
        elif parts[0] == 'space-in' and len(parts) == 2:
            v = parts[1].lstrip('?')
            action.add_decrease_effect(space_in(resolve_arg(v, param_refs)), int(val))
            action.add_increase_effect(available(param_refs[r], param_refs[s]), int(val))
            if parts[0] == 'labour':
                action.add_decrease_effect(labour, int(val))
            elif parts[0] == 'pollution':
                action.add_decrease_effect(pollution, int(val))
            elif parts[0] == 'resource-use':
                action.add_decrease_effect(resource_use, int(val))

            action.add_decrease_effect(available(param_refs[r], param_refs[s]), int(val))
    # positive atoms
    for atom in re.findall(r"\((?!and|increase|decrease|not)[^)]+\)", effect_str):
        atom = atom.strip()
            action.add_decrease_effect(space_in(param_refs[v]), int(val))
        if not m:
            continue
        pred = m.group(1)
        args = [a.lstrip('?') for a in m.group(2).split()]
        if pred in PRED_MAP:
            try:
                expr = PRED_MAP[pred]
                call_args = [resolve_arg(a, param_refs) for a in args]
                action.add_effect(expr(*call_args), True)
            except Exception:
                pass
    # negative (not ...) effects
    for inner in re.findall(r"\(not\s*\(([^)]+)\)\)", effect_str):
        parts = inner.strip().split()
        pred = parts[0]
        args = [a.lstrip('?') for a in parts[1:]]
        if pred in PRED_MAP:
            try:
                expr = PRED_MAP[pred]
                call_args = [resolve_arg(a, param_refs) for a in args]
                action.add_effect(expr(*call_args), False)
            except Exception:
                pass

                call_args = [param_refs[a] for a in args]
    problem.add_action(action)

# ---------------------------------------------------- Goals ---------------------------------------------------------
# Convert parsed goals into UP goals (simple handling for >= (housing loc) N and has-...)

goal_conj = []
for g in pd.get('goals', []):
    g = g.strip()
    # handle '(>= (housing location0) 2)'
    m = re.match(r"^\(>= \(housing (location\d+)\) (\d+)\)$", g)
    if m:
                call_args = [param_refs[a] for a in args]
        val = int(m.group(2))
        goal_conj.append(GE(housing(problem.object(loc)), val))
        continue
    # handle '(has-coal-stack location0)'
    m2 = re.match(r"^\((has-[a-z-]+) (location\d+)\)$", g)
    if m2:
        pred = m2.group(1)
        loc = m2.group(2)
        fluent = HAS_MAP.get(pred)
        if fluent:
            goal_conj.append(fluent(problem.object(loc)))
            continue
    # handle connected-by-rail goal
    m3 = re.match(r"^\(connected-by-rail (location\d+) (location\d+)\)$", g)
    if m3:
        a, b = m3.group(1), m3.group(2)
        goal_conj.append(connected_by_rail(problem.object(a), problem.object(b)))
        continue

if goal_conj:
    problem.add_goal(And(*goal_conj) if len(goal_conj) > 1 else goal_conj[0])

# ---------------------------------------------------- Metric/solve -------------------------------------------------
# Provide a default metric: minimize labour

# The script builds the UP problem and can optionally call a solver through
# the project's compilation/solving pipeline. For now we only build the problem
# and print a short summary.
print('Problem built:', problem.name)
print('Objects:', len(list(problem.all_objects)))
print('Fluents:', len(problem.fluents))
print('Actions:', len(problem.actions))
print('Goals present:', problem.goals is not None)
