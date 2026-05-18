from docs.code_snippets.pddl_interop import domain_filename
from docs.extensions.domains import compilation_solving
from unified_planning.io import PDDLReader

domain = 'dump-trucks'
instance = 'korf1'
solving = 'fast-downward'

reader = PDDLReader()
domain_filename = f'docs/extensions/domains/{domain}/handcrafted/domain.pddl'
problem_filename = f'docs/extensions/domains/{domain}/handcrafted/{instance}.pddl'

problem = reader.parse_problem(domain_filename, problem_filename)

compilation_solving.compile_and_solve(problem, solving, compilation='scin') #,compilation_kinds_to_apply=[]

