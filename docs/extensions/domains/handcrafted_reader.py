from docs.extensions.domains import compilation_solving
from unified_planning.io import PDDLReader

'''domain = '15-puzzle'
instance = 'korf1'
solving = 'fast-downward'

reader = PDDLReader()
domain_filename = f'{domain}/handcrafted/domain.pddl'
problem_filename = f'{domain}/handcrafted/{instance}.pddl'

problem = reader.parse_problem(domain_filename, problem_filename)

compilation_solving.compile_and_solve(problem, solving, compilation_kinds_to_apply=[])'''

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from docs.extensions.domains import compilation_solving

# Paths to your files
domain_file = "/docs/extensions/domains/hanoi.pddl"
problem_file = "/docs/extensions/domains/hanoi-3.pddl"

# Read PDDL files
reader = PDDLReader()
problem = reader.parse_problem(domain_file, problem_file)
print(problem.kind)


#compilation_solving.compile_and_solve(problem=problem, solver="fast-downward", compilation="ut-integers",timeout=1800)


# Solve the problem
#with OneshotPlanner(name="fast-downward",problem_kind=problem.kind) as planner:
#    result = planner.solve(problem)
#
# Print results
#if result.plan is not None:
#    print("Plan found:")
#    print(result.plan)
#else:
#    print("No plan found.")"""