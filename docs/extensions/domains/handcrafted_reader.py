from docs.code_snippets.pddl_interop import domain_filename
from docs.extensions.domains import compilation_solving
from unified_planning.io import PDDLReader

'''domain = '15-puzzle'
instance = 'korf1'
solving = 'fast-downward'

reader = PDDLReader()
domain_filename = f'docs/extensions/domains/{domain}/handcrafted/domain.pddl'
problem_filename = f'docs/extensions/domains/{domain}/handcrafted/{instance}.pddl'

problem = reader.parse_problem(domain_filename, problem_filename)

compilation_solving.compile_and_solve(problem, solving, compilation='up') #,compilation_kinds_to_apply=[]
'''

domain = '15-puzzle'
instance = 'i0'
solving = 'fast-downward'

reader = PDDLReader()
domain_filename = f'docs/extensions/domains/{domain}/pddl-extension/domain.pddl'
problem_filename = f'docs/extensions/domains/{domain}/pddl-extension/{instance}.pddl'

problem = reader.parse_problem(domain_filename, problem_filename)

#['up', 'int', 'uti', 'log', 'c', 'ci', 'cin', 'sc', 'sci', 'scin', 'None']
compilation_solving.compile_and_solve(problem, solving, compilation='uti')
