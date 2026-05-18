"""Test dump trucks domain: parse PDDL and compile through 'sc' pipeline."""
import sys
sys.path.insert(0, '/home/isaac/unified-planning')

from unified_planning.io.up_pddl_reader import UPPDDLReader
import docs.extensions.domains.compilation_solving as cs

reader = UPPDDLReader()
problem = reader.parse_problem(
    '/home/isaac/unified-planning/docs/extensions/domains/dump-trucks/pddl-extension/domain.pddl',
    '/home/isaac/unified-planning/docs/extensions/domains/dump-trucks/pddl-extension/i0.pddl',
)

print("Parsed successfully!")
print(f"Problem kind: {problem.kind}")

compiled, _, _ = cs.compile_problem(problem, 'sc')
print("\nCompiled successfully!")
