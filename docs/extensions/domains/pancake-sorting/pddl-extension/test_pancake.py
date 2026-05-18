"""Parse and compile pancake-sorting instances using the arrays extension."""
import sys
sys.path.insert(0, '/home/isaac/unified-planning')

from unified_planning.io.up_pddl_reader import UPPDDLReader
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
DOMAIN = os.path.join(BASE, 'domain.pddl')
PROBLEMS = [f'i{i}.pddl' for i in range(1, 5)]

reader = UPPDDLReader()

for pfile in PROBLEMS:
    ppath = os.path.join(BASE, pfile)
    print(f"\n{'='*50}")
    print(f"Parsing {pfile} ...")
    problem = reader.parse_problem(DOMAIN, ppath)
    print("  OK — parsed successfully")
    print(f"  Actions: {[a.name for a in problem.actions]}")

    # Compile with IPAR -> ARRAYS
    sys.path.insert(0, '/home/isaac/unified-planning/docs/extensions/domains')
    from compilation_solving import compile_problem
    compiled, results, t = compile_problem(problem, 'up')
    print(f"  Compiled in {t:.2f}s — {len(compiled.actions)} grounded actions")
