from docs.extensions.domains import compilation_solving
from unified_planning.io.up_pddl_reader import UPPDDLReader

'''domain = '15-puzzle'
instance = 'korf1'
solving = 'fast-downward'

reader = PDDLReader()
domain_filename = f'docs/extensions/domains/{domain}/handcrafted/domain.pddl'
problem_filename = f'docs/extensions/domains/{domain}/handcrafted/{instance}.pddl'

problem = reader.parse_problem(domain_filename, problem_filename)

compilation_solving.compile_and_solve(problem, solving, compilation='up') #,compilation_kinds_to_apply=[]
'''

domain = 'dump-trucks'
tests = False
instance = 'i0'
solving = 'fast-downward'      # requires numeric planner: sci/INTEGERS_REMOVING breaks on cardinality fluents


ext = ''

if tests:
    name_test = f"_{domain}"
    domain = "tests"
    instance = f"problem{name_test}"
else: name_test = ""

reader = UPPDDLReader()
domain_filename = f'docs/extensions/domains/{domain}/pddl-extension/domain{name_test}.pddl'
problem_filename = f'docs/extensions/domains/{domain}/pddl-extension/{instance}.pddl'

problem = reader.parse_problem(domain_filename, problem_filename)

print(f" -->> {problem}")

#['up', 'int', 'uti', 'log', 'c', 'ci', 'cin', 'sc', 'sci', 'scin', 'None']
# sci breaks: INTEGERS_REMOVING encodes card fluents as object types → OBJECT_FLUENTS → PDDL writer crash
# scin keeps card fluents as PDDL numeric functions, requires a numeric planner (enhsp)
compilation_solving.compile_and_solve(problem, solving, compilation='sci', verbose=True)
