"""Quick test for set operations in PDDL parser."""
import sys
sys.path.insert(0, '/home/isaac/unified-planning')

from unified_planning.io.up_pddl_reader import UPPDDLReader

reader = UPPDDLReader()
problem = reader.parse_problem(
    '/home/isaac/unified-planning/docs/extensions/domains/tests/domain_sets.pddl',
    '/home/isaac/unified-planning/docs/extensions/domains/tests/problem_sets.pddl',
)

print("Problem parsed successfully!")
print(problem)
