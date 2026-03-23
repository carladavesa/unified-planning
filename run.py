"""Run a domain planning problem via a unified interface.

This script provides a single entry point for executing planning problems defined in
`domains/`. Each domain is implemented as a module providing a `Domain` instance
(or a `build_problem()` function).

Example:
  python run.py --domain pancake-sorting --compilation int --solving fast-downward

Available commands:
  --list-domains        List all available domains.
  --list-instances      List instances for a given domain.

This runner is a small standalone helper that compiles and solves problems using
Unified Planning APIs. It avoids importing helper code from `docs/`.
"""

from __future__ import annotations

import argparse
import os.path
import signal
import time

from domains import DOMAINS
from domains.base import Domain, FunctionDomain
from unified_planning.io import PDDLReader

try:
    import rantanplan
except ImportError:
    pass

from unified_planning.shortcuts import *
from unified_planning.engines import CompilationKind

DEFAULT_TIMEOUT = 1800  # 30 minutes

# Compilation pipelines (same as the docs helper used to provide)
COMPILATION_PIPELINES = {
    "up": [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    "int": [  # numeric
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
    ],
    "uti": [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.GROUNDING,
        CompilationKind.INTEGERS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    "log": [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_LOGARITHMIC_REMOVING,
    ],
    "c": [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    "ci": [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    "cin": [  # numeric
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
    ],
    "sc": [
        CompilationKind.SETS_REMOVING,
        CompilationKind.COUNT_REMOVING,
        # CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    "sci": [
        CompilationKind.SETS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        # CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    "scin": [  # numeric
        CompilationKind.SETS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
        # CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    "None": [],
}


class TimeoutException(Exception):
    """Raised when operation exceeds time limit."""


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException()


def print_up_problem_size(problem, name: str = "") -> None:
    problem_str = str(problem)
    size = len(problem_str)

    print("--- Problem Size --")
    print(f"Number of actions: {len(problem.actions)}")
    print(f"Characters: {size}")
    print(f"Lines: {len(problem_str.splitlines())}")


def compile_problem(
    problem: "Problem", compilation_pipeline: str, timeout: int = DEFAULT_TIMEOUT
) -> tuple["Problem", List, float]:
    """Compile problem through specified pipeline."""

    if compilation_pipeline not in COMPILATION_PIPELINES:
        raise ValueError(
            f"Unknown compilation pipeline: '{compilation_pipeline}'. "
            f"Available: {list(COMPILATION_PIPELINES.keys())}"
        )

    compilation_kinds = COMPILATION_PIPELINES[compilation_pipeline]
    if not compilation_kinds:
        print("No compilation steps specified")
        return problem, [], 0.0

    print(f"\n{'=' * 60}")
    print(f"Compilation Pipeline: {compilation_pipeline}")
    print(f"{'=' * 60}")

    start_time = time.time()
    results = []

    # Set timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        for i, ck in enumerate(compilation_kinds, 1):
            print(f"Compiling {ck}")
            params = {}
            if ck == CompilationKind.ARRAYS_REMOVING:
                params = {"mode": "permissive"}

            with Compiler(problem_kind=problem.kind, compilation_kind=ck, params=params) as compiler:
                result = compiler.compile(problem, ck)
                results.append(result)
                problem = result.problem

        print_up_problem_size(problem)
        signal.alarm(0)
        compilation_time = time.time() - start_time
        return problem, results, compilation_time

    except TimeoutException:
        signal.alarm(0)
        print(f"\nCompilation timeout ({timeout}s)")
        raise
    except Exception as e:
        signal.alarm(0)
        print(f"\nCompilation error: {e}")
        raise


def solve_problem(
    problem: "Problem",
    solver_name: str,
    compilation_results: List,
    timeout: int = DEFAULT_TIMEOUT,
    planner_params: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Solve a compiled problem with a specified planner."""

    print(f"\n{'=' * 60}")
    print(f"Solver: {solver_name}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    params = dict(planner_params or {})
    base_solver = solver_name
    if solver_name.startswith("any_"):
        base_solver = solver_name.split("_", 1)[1]
    elif solver_name in ["enhsp-any"]:
        base_solver = solver_name

    try:
        if solver_name.startswith("any_") or solver_name in ["enhsp-any"]:
            if solver_name.startswith("any_"):
                planner_name = solver_name.split("_", 1)[1]
            else:
                planner_name = solver_name
            with AnytimePlanner(name=planner_name, params=params) as planner:
                solution_count = 0
                for res in planner.get_solutions(problem, timeout=timeout):
                    solution_count += 1
                    print(f"\n--- Solution {solution_count} ---")

                    plan = res.plan
                    for result in reversed(compilation_results):
                        plan = plan.replace_action_instances(result.map_back_action_instance)
                    print(plan)
                    print(f"Actions: {len(plan.actions)}")

            signal.alarm(0)
            solving_time = time.time() - start_time
            print(f"\nFound {solution_count} solution(s)")
            return solving_time

        else:
            with OneshotPlanner(name=solver_name, params=params) as planner:
                if not planner.supports(problem.kind):
                    print("Warning: Problem has unsupported features:")
                    unsupported = [
                        f"  - {feature}"
                        for feature in problem.kind.features
                        if feature not in planner.supported_kind().features
                    ]
                    print("\n".join(unsupported))

                file = open("file.txt", "w", encoding="utf-8")
                result = planner.solve(problem, output_stream=file)
                if result.plan is not None:
                    print("Solution found!\n")
                    plan = result.plan
                    for comp_result in reversed(compilation_results):
                        plan = plan.replace_action_instances(
                            comp_result.map_back_action_instance
                        )
                    print(plan)
                    print(f"\nActions: {len(plan.actions)}")
                else:
                    print("No solution found")
                    print(f"Status: {result.status}")

                signal.alarm(0)
                solving_time = time.time() - start_time
                return solving_time

    except TimeoutException:
        signal.alarm(0)
        print(f"\nSolving timeout ({timeout}s)")
        raise
    except Exception as e:
        signal.alarm(0)
        print(f"\nSolving error: {e}")
        raise


def compile_and_solve(
    problem: "Problem",
    solver: str,
    compilation: str = "none",
    timeout: int = DEFAULT_TIMEOUT,
    planner_params: Optional[Dict[str, Any]] = None,
):
    """Compile and solve a planning problem."""

    # Suppress UP credits
    get_environment().credits_stream = None

    total_start = time.time()

    try:
        compiled_problem, comp_results, comp_time = compile_problem(problem, compilation, timeout)

        print(f"\n{'=' * 60}")
        print("Compiled Problem:")
        print(f"{'=' * 60}\n")
        print(compiled_problem)

        print(f"  Compilation: {comp_time:.2f}s")

        remaining_timeout = max(1, timeout - int(comp_time))
        solve_time = solve_problem(
            compiled_problem,
            solver,
            comp_results,
            remaining_timeout,
            planner_params=planner_params,
        )

        total_time = time.time() - total_start
        print(f"\n{'=' * 60}")
        print("Summary:")
        print(f"  Compilation: {comp_time:.2f}s")
        if solve_time is not None:
            print(f"  Solving:     {solve_time:.2f}s")
        print(f"  Total:       {total_time:.2f}s")
        print(f"{'=' * 60}\n")

    except TimeoutException:
        print(f"\nOverall timeout ({timeout}s)")
    except Exception as e:
        print(f"\nError: {e}")
        raise


def _get_domain(module) -> Domain:
    """Return a Domain instance from the loaded domain module."""
    # 1) Prefer an explicit DOMAIN object
    if hasattr(module, "DOMAIN"):
        dom = getattr(module, "DOMAIN")
        if isinstance(dom, Domain):
            return dom

    # 2) Allow a factory function returning a Domain
    if hasattr(module, "get_domain"):
        dom = getattr(module, "get_domain")()
        if isinstance(dom, Domain):
            return dom

    # 3) Fallback: wrap a build_problem() function in a Domain adapter
    if hasattr(module, "build_problem"):
        build_fn = getattr(module, "build_problem")
        instances_fn = getattr(module, "default_instances", None)
        return FunctionDomain(build_fn, instances_fn)

    raise AttributeError(
        "Domain module must provide a `DOMAIN` object, `get_domain()`, or `build_problem()` function."
    )


def list_domains() -> None:
    print("Available domains:")
    for d in sorted(DOMAINS.keys()):
        print(f"  - {d}")


def list_instances(domain: str) -> None:
    module = DOMAINS.get(domain)

    if module is None:
        print(f"Domain '{domain}' not found.")
        return

    try:
        instances = module.list_instances()
    except Exception as e:
        print(f"Error reading instances from domain '{domain}': {e}")
        return

    if not instances:
        print("No instances defined.")
        return

    print(f"Instances for domain '{domain}':")
    for name, params in instances.items():
        print(f"  - {name}: {params}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python run.py")
    parser.add_argument("--domain", help="Domain name (e.g. pancake-sorting) or path to PDDL domain file")
    parser.add_argument("--instance", default="default", help="Instance name or path to PDDL problem file")
    parser.add_argument("--compilation", default="up", help="Compilation pipeline")
    parser.add_argument("--solving", default="fast-downward", help="Solver to use")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds")
    parser.add_argument("--list-domains", action="store_true", help="List available domains")
    parser.add_argument("--list-instances", help="List instances for a domain")
    parser.add_argument("--param", action="append", default=[], help="Extra param key=value")

    args = parser.parse_args(argv)

    if args.list_domains:
        list_domains()
        return

    if args.list_instances:
        list_instances(args.list_instances)
        return

    if not args.domain:
        parser.error("--domain is required")

    planner_params: Dict[str, Any] = {}
    for p in args.param:
        if "=" not in p:
            raise ValueError(f"Invalid --param value: {p} (expected key=value)")
        k, v = p.split("=", 1)
        planner_params[k] = v

    if os.path.exists(args.domain) and os.path.exists(args.instance):
        # crea problema llegint fitxers PDDL directament (domini i instància)
        reader = PDDLReader()
        problem = reader.parse_problem(args.domain, args.instance)

    else:
        # creem problema a través del codi Python de cada domini
        module = DOMAINS.get(args.domain)
        if module is None:
            parser.error(f"Domain '{args.domain}' not found")

        dom = _get_domain(module)

        # Build the problem without CLI-driven parameter overrides.
        problem = dom.build_problem(instance=args.instance)

    compile_and_solve(
        problem,
        args.solving,
        args.compilation,
        args.timeout,
        planner_params=planner_params,
    )


if __name__ == "__main__":
    main()