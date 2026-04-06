import signal
import time
from unified_planning.shortcuts import *
from unified_planning.engines import CompilationKind

COMPILATION_PIPELINES = {
    'up': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        #CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    'int': [ # numeric
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
    ],
    'uti': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    'log': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_LOGARITHMIC_REMOVING,
    ],
    'c': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_REMOVING,
    ],
    'ci': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        #CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    'cin': [ # numeric
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
    ],
    'sc' : [
        CompilationKind.SETS_REMOVING,
        CompilationKind.COUNT_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    'sci' : [
        CompilationKind.SETS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        #CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    'scin' : [ # numeric
        CompilationKind.SETS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
        #CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    'all' : [
        CompilationKind.SETS_REMOVING,
        CompilationKind.COUNT_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
    ],
    'None': []
}

# ==================== Timeout Handling ====================

DEFAULT_TIMEOUT = 1800  # 30 minutes

class TimeoutException(Exception):
    """Raised when operation exceeds time limit."""
    pass

def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException()

# ==================== Compilation ====================

def print_up_problem_size(problem, name=""):
    problem_str = str(problem)
    size = len(problem_str)

    print(f"--- Problem Size --")
    print(f"Number of actions: {len(problem.actions)}")
    print(f"Characters: {size}")
    print(f"Lines: {len(problem_str.splitlines())}")

def compile_problem(
        problem: "Problem", compilation_pipeline: str, timeout: int = DEFAULT_TIMEOUT
) -> tuple["Problem", List, float]:
    """
    Compile problem through specified pipeline.

    Args:
        problem: Original planning problem
        compilation_pipeline: Name of pipeline from COMPILATION_PIPELINES
        timeout: Timeout in seconds

    Returns:
        Tuple of (compiled_problem, compilation_results, compilation_time)

    Raises:
        ValueError: If compilation pipeline name is invalid
        TimeoutException: If compilation exceeds timeout
    """
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
            # Compilation parameters
            params = {}
            if ck == CompilationKind.ARRAYS_REMOVING:
                params = {'mode': 'permissive'}

            # Compile
            with Compiler(problem_kind=problem.kind, compilation_kind=ck, params=params) as compiler:
                result = compiler.compile(problem, ck)
                results.append(result)
                problem = result.problem

        #print(problem)
        print_up_problem_size(problem)
        signal.alarm(0)  # Cancel timeout
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


# ==================== Solving ====================

def solve_problem(
        problem: "Problem",
        solver_name: str,
        compilation_results: List,
        timeout: int = DEFAULT_TIMEOUT
) -> Optional[float]:
    """
    Solve problem with specified planner.

    Args:
        problem: Compiled problem to solve
        solver_name: Name of solver/planner
        compilation_results: Results from compilation for plan mapping
        timeout: Timeout in seconds

    Returns:
        Solving time in seconds, or None if no solution found

    Raises:
        TimeoutException: If solving exceeds timeout
    """
    print(f"\n{'=' * 60}")
    print(f"Solver: {solver_name}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Set timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        # Anytime planners
        if solver_name.startswith('any_') or solver_name in ['enhsp-any']:
            if solver_name.startswith('any_'):
                planner_name = solver_name.split('_', 1)[1]
            else:
                planner_name = solver_name
            with AnytimePlanner(name=planner_name) as planner:
                solution_count = 0
                for res in planner.get_solutions(problem, timeout=timeout):
                    solution_count += 1
                    print(f"\n--- Solution {solution_count} ---")

                    # Map back through compilation
                    plan = res.plan
                    for result in reversed(compilation_results):
                        plan = plan.replace_action_instances(result.map_back_action_instance)
                    print(plan)
                    print(f"Actions: {len(plan.actions)}")

            signal.alarm(0)
            solving_time = time.time() - start_time
            print(f"\nFound {solution_count} solution(s)")
            return solving_time

        # Oneshot planners
        else:
            with OneshotPlanner(name=solver_name) as planner:
                if not planner.supports(problem.kind):
                    print("Warning: Problem has unsupported features:")
                    unsupported = [
                        f"  - {feature}" for feature in problem.kind.features
                        if feature not in planner.supported_kind().features
                    ]
                    print("\n".join(unsupported))

                # Solve
                #file = open("path/to/file", "w", encoding="utf-8")
                #result = planner.solve(problem, output_stream=file)
                result = planner.solve(problem)
                if result.plan is not None:
                    print("Solution found!\n")
                    # Map back through compilation
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


# ==================== Compile and Solve ====================

def compile_and_solve(
        problem: "Problem",
        solver: str,
        compilation: str = 'none',
        timeout: int = DEFAULT_TIMEOUT,
):
    """
    Compile and solve a planning problem.

    Args:
        problem: Planning problem to solve
        solver: Name of solver/planner to use
        compilation: Name of compilation pipeline (default: 'none')
        timeout: Timeout in seconds (default: 1800)

    Raises:
        ValueError: If compilation pipeline is invalid
        TimeoutException: If compilation or solving exceeds timeout
    """
    # Suppress UP credits
    get_environment().credits_stream = None

    total_start = time.time()

    try:
        # Compilation
        compiled_problem, comp_results, comp_time = compile_problem(problem, compilation, timeout)

        # Print compiled problem
        print(f"\n{'=' * 60}")
        print("Compiled Problem:")
        print(f"{'=' * 60}\n")
        print(compiled_problem)

        print(f"  Compilation: {comp_time:.2f}s")

        # Solving
        remaining_timeout = max(1, timeout - int(comp_time))
        solve_time = solve_problem(compiled_problem, solver, comp_results, remaining_timeout)

        # Summary
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