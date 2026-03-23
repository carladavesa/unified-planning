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
