# Copyright 2021-2023 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""This module defines the preconditions to table simplifier class."""
from itertools import product

from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model.operators import OperatorKind
from bidict import bidict
from ortools.sat.python import cp_model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import Problem, Action, ProblemKind, Object, MinimizeActionCosts, FNode, Effect, \
    InstantaneousAction, EffectKind, Fluent, Parameter, Expression
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action
from typing import Dict, List, Optional, OrderedDict, Iterable, Tuple, Iterator, Union, Any, cast
from functools import partial

from unified_planning.model.walkers import Simplifier, simplifier
from unified_planning.shortcuts import And, Not, Iff, Or, FALSE, TRUE, Int, Equals, ObjectExp, UserType, BoolType
import math
import unified_planning as up
import unified_planning.engines as engines
import re

class CPSolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collects all unique solutions from CP-SAT solver."""

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solutions = []
        self.__seen = set()  # To detect duplicates

    def on_solution_callback(self):
        solution = {str(v): self.Value(v) for v in self.__variables}
        sol_tuple = tuple(sorted(solution.items()))

        if sol_tuple not in self.__seen:
            self.__seen.add(sol_tuple)
            self.__solutions.append(solution)

    @property
    def solutions(self) -> list[dict[str, int]]:
        return self.__solutions

class PreconditionsToTableSimplifier(engines.engine.Engine, CompilerMixin):
    """
    Preconditions To Static Predicate Compiler:
    Uses CP-SAT to enumerate all valid value combinations satisfying the action preconditions
    (including bounds of unconditional numeric effects). Creates a new boolean static predicate
    per action, instantiated with all valid combinations as true in the initial state.
    The original action is replaced by a single action whose precondition checks this static predicate.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.PRECONDITIONS_TO_TABLE_SIMPLIFIER)
        self._domains: Dict[str, Tuple[int, int]] = {}
        self._number_objects_cache: Dict[int, FNode] = {}

    @property
    def name(self):
        return "ptts"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_parameters("REAL_ACTION_PARAMETERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("ARRAY_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_fluents_type("DERIVED_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        supported_kind.set_time("CONTINUOUS_TIME")
        supported_kind.set_time("DISCRETE_TIME")
        supported_kind.set_time("INTERMEDIATE_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("EXTERNAL_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("TIMED_EFFECTS")
        supported_kind.set_time("TIMED_GOALS")
        supported_kind.set_time("DURATION_INEQUALITIES")
        supported_kind.set_time("SELF_OVERLAPPING")
        supported_kind.set_expression_duration("STATIC_FLUENTS_IN_DURATIONS")
        supported_kind.set_expression_duration("FLUENTS_IN_DURATIONS")
        supported_kind.set_expression_duration("INT_TYPE_DURATIONS")
        supported_kind.set_expression_duration("REAL_TYPE_DURATIONS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_constraints_kind("STATE_INVARIANTS")
        supported_kind.set_constraints_kind("TRAJECTORY_CONSTRAINTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_actions_cost_kind("STATIC_FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_quality_metrics("OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("TEMPORAL_OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_oversubscription_kind("INT_NUMBERS_IN_OVERSUBSCRIPTION")
        supported_kind.set_oversubscription_kind("REAL_NUMBERS_IN_OVERSUBSCRIPTION")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= PreconditionsToTableSimplifier.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.PRECONDITIONS_TO_TABLE_SIMPLIFIER

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("INT_FLUENTS")
        new_kind.unset_conditions_kind("ARRAY_FLUENTS")
        return new_kind

    # Operators that can appear inside arithmetic expressions
    ARITHMETIC_OPS = {
        OperatorKind.PLUS: 'plus',
        OperatorKind.MINUS: 'minus',
        OperatorKind.DIV: 'div',
        OperatorKind.TIMES: 'mult',
    }

    # ==================== AUXILIARY METHODS ====================

    def _get_number_object(self, value: int) -> FNode:
        if value in self._number_objects_cache:
            return self._number_objects_cache[value]
        new_object = Object(f'n{value}', UserType('Number'))
        self._new_problem.add_object(new_object)
        new_object_expression = ObjectExp(new_object)
        self._number_objects_cache[value] = new_object_expression
        return new_object_expression

    def _project_and_group_by_result(self, value: FNode, condition: FNode, solutions: List[dict]) -> dict:
        """
        Project solutions over effect dependencies and groups by solution
        Returns: {result_key: {'value': int, 'combinations': [tuple]}}
        """
        dependencies = self._get_effect_dependencies(value)
        if condition and condition != TRUE():
            dependencies |= self._get_effect_dependencies(condition)

        dep_values_to_result = {}
        for solution in solutions:
            sol_cond = self._evaluate_with_solution(condition, solution)
            if sol_cond is False or sol_cond == FALSE():
                continue
            dep_values = tuple(sorted(
                (var, solution[var]) for var in dependencies if var in solution
            ))
            value_result = self._evaluate_with_solution(value, solution)
            if value_result is None:
                continue
            result_key = str(value_result)
            if dep_values not in dep_values_to_result:
                dep_values_to_result[dep_values] = {'value': value_result, 'result_key': result_key}

        result_to_deps = {}
        for dep_values, data in dep_values_to_result.items():
            rk = data['result_key']
            if rk not in result_to_deps:
                result_to_deps[rk] = {'value': data['value'], 'combinations': []}
            result_to_deps[rk]['combinations'].append(dep_values)
        return result_to_deps

    def _get_effect_dependencies(self, node: FNode) -> set:
        """
        Get all fluent/parameter variable strings that appear in an expression.
        These are the variables that affect the result of the expression.
        """
        deps = set()
        if node.is_fluent_exp() or node.is_parameter_exp():
            deps.add(str(node))
        for arg in node.args:
            deps.update(self._get_effect_dependencies(arg))
        return deps

    def _build_condition_from_combination(self, new_problem, var_str_to_fnode, dep_dict):
        """Builds conditions from the dependant variable values."""
        conditions = []
        for var_str, value in dep_dict.items():
            fnode = var_str_to_fnode.get(var_str)
            if not fnode:
                continue
            if fnode.is_fluent_exp():
                fluent = fnode.fluent()
                if fluent.type.is_int_type():
                    # Converteix el valor a bits
                    new_fluents, new_values = self._convert_fluent_and_value(
                        new_problem, fnode, Int(value)
                    )
                    for f, v in zip(new_fluents, new_values):
                        conditions.append(f if v else Not(f))
                # altres tipus si cal...
        if not conditions:
            return None
        return And(conditions) if len(conditions) > 1 else conditions[0]

    def _create_precondition_from_variable(
            self,
            fnode: FNode,
            value: int
    ) -> Optional[FNode]:
        """
        Create a precondition from a variable and its value.
        Handles int fluents, fluents, bool fluents, and parameters.
        """
        if fnode.is_fluent_exp():
            fluent = fnode.fluent()
            new_fluent = self._new_problem.fluent(fluent.name)
            if fluent.type.is_int_type():
                num_obj = self._get_number_object(value)
                return new_fluent(*fnode.args, num_obj)
            elif fluent.type.is_user_type():
                obj = self._get_object_from_index(fluent.type, value)
                if obj:
                    return Equals(new_fluent, ObjectExp(obj))
            elif fluent.type.is_bool_type():
                if value == 1:
                    return new_fluent
                else:
                    return Not(new_fluent)

        elif fnode.is_parameter_exp():
            param = fnode.parameter()
            assert param.type.is_user_type(), "param type is not UserType"
            obj = self._get_object_from_index(param.type, value)
            if obj:
                return Equals(fnode, ObjectExp(obj))

        return None

    def _build_pattern_condition(
            self,
            var_str_to_fnode: dict,
            pattern: dict
    ) -> Optional[FNode]:
        """Build a condition from a pattern (dict of var_str -> value)."""
        conditions = []
        for var_str, value in pattern.items():
            fnode = var_str_to_fnode.get(var_str)
            if not fnode:
                continue
            cond = self._create_precondition_from_variable(fnode, value)
            if cond:
                conditions.append(cond)

        if not conditions:
            return None
        return And(conditions) if len(conditions) > 1 else conditions[0]

    def _set_fluent_bits(self, problem, fluent, k_args, new_value, n_bits, object_ref: Optional[Object] = None):
        for bit_index in range(n_bits):
            this_fluent = problem.fluent(f"{fluent.name}_{bit_index}")(*k_args, *(object_ref,) if object_ref is not None else ())
            problem.set_initial_value(this_fluent, new_value[bit_index])

    def _evaluate_with_solution(
            self,
            expr: FNode,
            solution: dict,
    ) -> Optional[FNode]:
        """
        Evaluate an expression using values from solution.
        """
        def evaluate_recursive(node: FNode):
            # Constant
            if node.is_constant():
                return node.constant_value()

            # Object - get index
            if node.is_object_exp():
                obj = node.object()
                idx = self._object_to_index.get((obj.type, obj))
                if idx is not None:
                    return idx
                return None

            # Fluent/Parameter - look up in solution
            if node.is_fluent_exp() or node.is_parameter_exp():
                var_str = str(node)
                if var_str not in solution:
                    return None
                val = solution[var_str]
                if isinstance(val, set):
                    # mirar si esta en el set, si hi es
                    return val  # Retorna el set directament
                # Valor concret
                if node.type.is_user_type():
                    node_type = node.fluent().type if node.is_fluent_exp() else node.parameter().type
                    return self._get_object_from_index(node_type, val)
                return val

            # TRUE/FALSE
            if node.is_true():
                return True
            if node.is_false():
                return False

            # Equals
            if node.is_equals():
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))

                # Both concrete
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left == right
                    return None

                # Left is set, right is concrete
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if right not in left:
                        return False
                    if len(left) == 1:
                        return True
                    return None

                # Right is set, left is concrete
                if isinstance(right, set) and not isinstance(left, set) and left is not None:
                    if left not in right:
                        return False
                    if len(right) == 1:
                        return True
                    return None
                return None

            # Not
            if node.is_not():
                inner = evaluate_recursive(node.arg(0))
                if inner is None or isinstance(inner, set):
                    return None
                if isinstance(inner, bool):
                    return not inner
                return None

            # And
            if node.is_and():
                for arg in node.args:
                    result = evaluate_recursive(arg)
                    if result is False:
                        return False
                all_true = all(evaluate_recursive(arg) is True for arg in node.args)
                if all_true:
                    return True
                return None

            # Or
            if node.is_or():
                for arg in node.args:
                    result = evaluate_recursive(arg)
                    if result is True:
                        return True
                all_false = all(evaluate_recursive(arg) is False for arg in node.args)
                if all_false:
                    return False
                return None

            # Plus
            if node.is_plus():
                values = [evaluate_recursive(arg) for arg in node.args]
                if any(isinstance(v, set) or v is None for v in values):
                    return None
                return sum(values)

            # Minus
            if node.is_minus():
                values = [evaluate_recursive(arg) for arg in node.args]
                if any(isinstance(v, set) or v is None for v in values):
                    return None
                if len(values) == 1:
                    return -values[0]
                return values[0] - sum(values[1:])

            # Times
            if node.is_times():
                values = [evaluate_recursive(arg) for arg in node.args]
                if any(isinstance(v, set) or v is None for v in values):
                    return None
                result = 1
                for v in values:
                    result *= v
                return result

            # Lt
            if node.is_lt():
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left < right
                    return None
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if all(v < right for v in left):
                        return True
                    if all(v >= right for v in left):
                        return False
                    return None
                if isinstance(right, set) and not isinstance(left, set) and left is not None:
                    if all(left < v for v in right):
                        return True
                    if all(left >= v for v in right):
                        return False
                    return None
                return None

            # Le
            if node.is_le():
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left <= right
                    return None
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if all(v <= right for v in left):
                        return True
                    if all(v > right for v in left):
                        return False
                    return None
                if isinstance(right, set) and not isinstance(left, set) and left is not None:
                    if all(left <= v for v in right):
                        return True
                    if all(left > v for v in right):
                        return False
                    return None
                return None

            # Gt
            if node.node_type == OperatorKind.GT:
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left > right
                    return None
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if all(v > right for v in left):
                        return True
                    if all(v <= right for v in left):
                        return False
                    return None
                return None

            # Ge
            if node.node_type == OperatorKind.GE:
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left >= right
                    return None
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if all(v >= right for v in left):
                        return True
                    if all(v < right for v in left):
                        return False
                    return None
                return None
            return None

        return evaluate_recursive(expr)
        #if result is None or isinstance(result, set):
        #    return expr
        #elif isinstance(result, bool):
        #    return TRUE() if result else FALSE()
        #elif isinstance(result, int):
        #    if new_problem is not None:
        #        return self._get_number_object(new_problem, result)
        #    return Int(result)
        #elif isinstance(result, Object):
        #    return ObjectExp(result)
        #return expr

    # =================== COMPRESSION METHODS ===================

    def _compute_maximal_itemsets_apriori(
            self,
            solutions: list[dict[str, int]],
            min_support: int = 2,
            max_itemset_size: int = None
    ) -> dict[frozenset, int]:
        """
        Compute ONLY maximal frequent itemsets directly.
        """
        if not solutions:
            return {}

        variables = list(solutions[0].keys())
        num_vars = len(variables)

        if max_itemset_size is None:
            max_itemset_size = num_vars - 1

        # Guardem itemsets per nivell
        levels = {}

        # Level 1
        level_1 = {}
        for var in variables:
            value_counts = {}
            for sol in solutions:
                val = sol[var]
                item = frozenset([(var, val)])
                value_counts[item] = value_counts.get(item, 0) + 1
            for item, count in value_counts.items():
                if count >= min_support:
                    level_1[item] = count

        if not level_1:
            return {}

        levels[1] = level_1
        current_level = level_1
        k = 2

        while current_level and k <= max_itemset_size:
            candidates = self._apriori_generate_candidates(current_level, k)

            if not candidates:
                break

            next_level = {}
            for candidate in candidates:
                count = sum(1 for sol in solutions
                            if all(sol.get(var) == val for var, val in candidate))
                if count >= min_support:
                    next_level[candidate] = count

            if not next_level:
                break

            levels[k] = next_level
            current_level = next_level
            k += 1

        # Ara trobem maximals: un itemset és maximal si cap superset és freqüent
        max_level = max(levels.keys())
        maximal = dict(levels[max_level])  # El nivell més alt sempre és maximal

        # Per cada nivell inferior
        for level in range(max_level - 1, 0, -1):
            for itemset, freq in levels[level].items():
                # Comprovar si té algun superset freqüent al nivell superior
                has_frequent_superset = False

                if level + 1 in levels:
                    for larger in levels[level + 1]:
                        if itemset < larger:
                            has_frequent_superset = True
                            break

                if not has_frequent_superset:
                    maximal[itemset] = freq
        return maximal

    def _compute_min_support(self, num_solutions: int, num_variables: int) -> int:
        """
        Adaptive min_support based on problem size.

        Intuition: With more variables, we need higher support to find
        meaningful patterns (not just noise).
        """
        # Base: sqrt del nombre de solucions
        base = max(2, int(num_solutions ** 0.5))

        # Ajustar segons variables: més variables → més suport
        factor = 1 + (num_variables / 20)  # Factor creix amb variables

        min_support = max(2, int(base / factor))
        return min_support

    def _apriori_generate_candidates(
            self,
            prev_level: dict[frozenset, int],
            k: int
    ) -> set[frozenset]:
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets.

        Uses the Apriori principle: only combine itemsets that share k-2 items,
        and prune candidates where any (k-1)-subset is not frequent.
        """
        candidates = set()
        prev_itemsets = list(prev_level.keys())

        # Sort items within each itemset for consistent comparison
        # Convert to sorted tuples for easier joining
        sorted_itemsets = [tuple(sorted(itemset)) for itemset in prev_itemsets]

        for i in range(len(sorted_itemsets)):
            for j in range(i + 1, len(sorted_itemsets)):
                itemset1 = sorted_itemsets[i]
                itemset2 = sorted_itemsets[j]

                # Join condition: first k-2 items must be the same
                if k > 2 and itemset1[:-1] != itemset2[:-1]:
                    continue

                # Create candidate by union
                candidate = frozenset(itemset1) | frozenset(itemset2)

                if len(candidate) != k:
                    continue

                # Pruning: check that all (k-1)-subsets are frequent
                is_valid = True
                for item in candidate:
                    subset = candidate - {item}
                    if subset not in prev_level:
                        is_valid = False
                        break

                if is_valid:
                    candidates.add(candidate)

        return candidates

    def _sort_by_area(self, maximal: dict[frozenset, int]) -> list[tuple[frozenset, int, int]]:
        """
        Sort maximal itemsets by area (descending).
        """
        with_area = []
        for itemset, freq in maximal.items():
            area = len(itemset) * freq
            with_area.append((itemset, freq, area))

        # Sort by area descending, then by frequency descending (tie-breaker)
        with_area.sort(key=lambda x: (-x[2], -x[1]))
        return with_area

    def _select_non_overlapping_mfi(
            self,
            solutions: list[dict[str, int]],
            sorted_mfi: list[tuple[frozenset, int, int]]
    ) -> tuple[list[tuple[frozenset, set[int]]], list[int]]:
        """
        Greedily select MFI with highest area, removing overlapping ones.
        Optimized with pre-indexed solutions.
        """
        if not solutions or not sorted_mfi:
            return [], list(range(len(solutions)))

        # Pre-indexar: per cada (var, val), quines solucions el contenen?
        item_to_solutions = {}
        for idx, solution in enumerate(solutions):
            for var, val in solution.items():
                item = (var, val)
                if item not in item_to_solutions:
                    item_to_solutions[item] = set()
                item_to_solutions[item].add(idx)

        def get_coverage_fast(itemset: frozenset) -> set[int]:
            """Get coverage using pre-built index."""
            if not itemset:
                return set(range(len(solutions)))

            items = list(itemset)
            result = item_to_solutions.get(items[0], set()).copy()
            for item in items[1:]:
                result &= item_to_solutions.get(item, set())
            return result

        uncovered = set(range(len(solutions)))
        selected = []

        for itemset, freq, area in sorted_mfi:
            if not uncovered:
                break
            coverage = get_coverage_fast(itemset) & uncovered
            if len(coverage) >= 2:
                selected.append((itemset, coverage))
                uncovered -= coverage
        return selected, list(uncovered)

    def _entries_to_formula(
            self,
            new_problem: Problem,
            variables: bidict,
            entries: list[dict],
            default_tuples: list[dict]
    ) -> Optional[FNode]:
        """
        Convert compressed entries to a DNF formula (bit encoding).
        """
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}
        or_clauses = []

        for entry in entries:
            pattern = entry['pattern']
            subtable_vars = entry['subtable_vars']
            subtable_tuples = entry['subtable_tuples']

            # Build pattern constraints (bits)
            pattern_constraints = []
            for var_str, value in pattern.items():
                cond = self._build_condition_from_combination(new_problem, var_str_to_fnode, {var_str: value})
                if cond:
                    pattern_constraints.append(cond)

            if not subtable_vars or not subtable_tuples:
                # No subtable - just the pattern
                if pattern_constraints:
                    or_clauses.append(
                        And(pattern_constraints) if len(pattern_constraints) > 1 else pattern_constraints[0])
                continue

            # Has subtable
            if len(subtable_vars) == 1:
                var_str = subtable_vars[0]
                values = {t[var_str] for t in subtable_tuples}
                fnode = var_str_to_fnode.get(var_str)

                if fnode and fnode.is_fluent_exp():
                    fluent = fnode.fluent()

                    # Check full domain
                    full_domain = None
                    if fluent.type.is_int_type():
                        full_domain = set(range(fluent.type.lower_bound, fluent.type.upper_bound + 1))
                    elif fluent.type.is_user_type():
                        full_domain = set(range(len(list(new_problem.objects(fluent.type)))))
                    elif fluent.type.is_bool_type():
                        full_domain = {0, 1}

                    if full_domain is not None and values == full_domain:
                        # Full domain - no constraint needed for subtable var
                        if pattern_constraints:
                            or_clauses.append(
                                And(pattern_constraints) if len(pattern_constraints) > 1 else pattern_constraints[0])
                        continue

                    # Not full domain - add bit constraints for subtable var
                    if len(values) == 1:
                        val = next(iter(values))
                        cond = self._build_condition_from_combination(new_problem, var_str_to_fnode, {var_str: val})
                        if cond:
                            pattern_constraints.append(cond)
                    else:
                        # OR de valors possibles
                        value_clauses = []
                        for val in sorted(values):
                            cond = self._build_condition_from_combination(new_problem, var_str_to_fnode, {var_str: val})
                            if cond:
                                value_clauses.append(cond)
                        if value_clauses:
                            pattern_constraints.append(
                                Or(value_clauses) if len(value_clauses) > 1 else value_clauses[0]
                            )

                if pattern_constraints:
                    or_clauses.append(
                        And(pattern_constraints) if len(pattern_constraints) > 1 else pattern_constraints[0])
                continue

            # Multiple variables in subtable - OR de combinacions
            subtable_or_clauses = []
            for subtable_tuple in subtable_tuples:
                cond = self._build_condition_from_combination(new_problem, var_str_to_fnode, subtable_tuple)
                if cond:
                    subtable_or_clauses.append(cond)

            if subtable_or_clauses:
                subtable_or = Or(subtable_or_clauses) if len(subtable_or_clauses) > 1 else subtable_or_clauses[0]
                if pattern_constraints:
                    or_clauses.append(And(pattern_constraints + [subtable_or]))
                else:
                    or_clauses.append(subtable_or)
            elif pattern_constraints:
                or_clauses.append(And(pattern_constraints) if len(pattern_constraints) > 1 else pattern_constraints[0])

        # Default tuples
        for solution in default_tuples:
            cond = self._build_condition_from_combination(new_problem, var_str_to_fnode, solution)
            if cond:
                or_clauses.append(cond)

        if not or_clauses:
            return None
        return Or(or_clauses).simplify() if len(or_clauses) > 1 else or_clauses[0].simplify()

    def _create_entries(
            self,
            solutions: list[dict[str, int]],
            selected_mfi: list[tuple[frozenset, set[int]]],
            remaining_indices: list[int]
    ) -> tuple[list[dict], list[dict]]:
        """
        Create compressed entries from selected MFI.
        Subtable is stored as a list of partial tuples (valid combinations only).
        """
        if not solutions:
            return [], []

        all_variables = set(solutions[0].keys())
        entries = []

        for itemset, covered_indices in selected_mfi:
            # Get the solutions covered by this itemset
            covered_solutions = [solutions[idx] for idx in covered_indices]

            #  Find ALL variables that are constant across covered solutions
            pattern = {}
            subtable_vars = []
            #pattern_vars = set(pattern.keys())

            for var in all_variables:
                values = {sol[var] for sol in covered_solutions}
                if len(values) == 1:
                    # Constant across all covered solutions -> add to pattern
                    pattern[var] = next(iter(values))
                else:
                    # Varies -> add to subtable
                    subtable_vars.append(var)

            # Subtable variables
            subtable_vars = sorted(subtable_vars)

            # Collect unique subtable tuples
            seen = set()
            subtable_tuples = []
            for sol in covered_solutions:
                subtable_tuple = tuple((var, sol[var]) for var in subtable_vars)
                if subtable_tuple not in seen:
                    seen.add(subtable_tuple)
                    subtable_tuples.append({var: sol[var] for var in subtable_vars})

            entries.append({
                'pattern': pattern,
                'subtable_vars': subtable_vars,
                'subtable_tuples': subtable_tuples,
                'num_tuples': len(covered_indices)
            })

        # Default tuples
        default_tuples = [solutions[idx] for idx in remaining_indices]
        return entries, default_tuples

    # ========================= GETTERS =========================

    def _get_object_from_index(self, user_type, index):
        """Get object corresponding to an index for a UserType."""
        if hasattr(self, '_index_to_object'):
            return self._index_to_object.get((user_type, index))
        return None

    def _get_static_fluents(self, problem: Problem, fluents: dict[FNode, FNode]) -> Dict[FNode, FNode]:
        """
        Get all static fluent instances throughout the problem.
        Two cases:
        1. Effect has parameters (e.g., onboard(a)) -> ALL instances of that fluent are modifiable
        2. Effect has concrete indices/objects -> only THAT specific instance is modifiable
        """
        # Base fluents base that are modifiable with parameters (cover all instances)
        modifiable_fluent_names = set()
        # Concrete instances that are modifiable
        modifiable_instances = set()
        for action in problem.actions:
            for effect in action.effects:
                effect_fluent = effect.fluent
                fluent_base = effect_fluent.fluent()
                # Check if the effect has parameters (no concrete instance)
                has_parameter = any(
                    arg.is_parameter_exp() or
                    (hasattr(arg, 'is_variable_exp') and arg.is_variable_exp())
                    for arg in effect_fluent.args
                )
                if has_parameter:
                    modifiable_fluent_names.add(fluent_base.name)
                else:
                    modifiable_instances.add(effect_fluent)

        static = {}
        for f, v in fluents.items():
            if f.is_fluent_exp():
                if f.fluent().type.is_array_type():
                    continue
            if f.fluent().name in modifiable_fluent_names:
                continue
            if f in modifiable_instances:
                continue
            static[f] = v
        return static

    def _get_uniform_static_fluents(self, problem: Problem) -> Dict[str, FNode]:
        """
        Get static fluents that have the same value for ALL instantiations.
        Returns: {fluent_name: constant_FNode}
        """
        fluent_values = {}
        fluent_fnodes = {}
        fluent_objects = {}

        for fnode, value in self._static_fluents.items():
            fluent = fnode.fluent()
            name = fluent.name
            if name not in fluent_values:
                fluent_values[name] = set()
                fluent_objects[name] = fluent

            # Extract value
            if value.is_bool_constant():
                val = value.bool_constant_value()
            elif value.is_int_constant():
                val = value.int_constant_value()
            elif value.is_real_constant():
                val = value.real_constant_value()
            else:
                val = str(value)
            fluent_values[name].add(val)
            fluent_fnodes[name] = value

        uniform = {}
        for name, values in fluent_values.items():
            if len(values) != 1: # More than one different value
                continue
            fluent = fluent_objects[name]
            # Verify that all instances are statics
            if fluent.signature:
                domains = [list(problem.objects(param.type)) for param in fluent.signature]
                expected_count = 1
                for d in domains:
                    expected_count *= len(d)
                actual_count = sum(1 for f in self._static_fluents.keys() if f.fluent().name == name)
                if actual_count == expected_count:
                    uniform[name] = fluent_fnodes[name]
            else:
                uniform[name] = fluent_fnodes[name]
        return uniform

    # ==================== QUERY METHODS ====================

    def _has_arithmetic(self, node: FNode) -> bool:
        """Check if expression contains arithmetic operations."""
        if (node.node_type in self.ARITHMETIC_OPS or
                node.is_le() or node.is_lt()):
            return True
        return any(self._has_arithmetic(arg) for arg in node.args)

    # ==================== CP-SAT Constraint Solving ====================

    def _register_usertype_mapping(self, user_type, objects):
        """Register object to type indexes mapping"""
        if not hasattr(self, '_object_to_index'):
            self._object_to_index = {}
        if not hasattr(self, '_index_to_object'):
            self._index_to_object = {}

        for idx, obj in enumerate(objects):
            self._object_to_index[(user_type, obj)] = idx
            self._index_to_object[(user_type, idx)] = obj

    def _add_cp_constraints(self, problem: Problem, node: FNode, variables: bidict, model: cp_model.CpModel):
        """Recursively build CP-SAT constraints from expression tree."""
        # Constants
        if node.is_constant():
            return node.constant_value()

        # Fluents
        if node.is_fluent_exp():
            if node in variables:
                return variables[node]
            fluent = node.fluent()
            # Static fluent - constant
            if fluent.name in self._uniform_fluents:
                return self._uniform_fluents[fluent.name].constant_value()

            if fluent.type.is_int_type():
                var = model.NewIntVar(
                    fluent.type.lower_bound,
                    fluent.type.upper_bound,
                    str(node)
                )
            elif fluent.type.is_user_type():
                # Obtenir tots els objectes d'aquest tipus
                objects = list(problem.objects(fluent.type))
                # Crear variable entera amb domini [0, len(objects)-1]
                var = model.NewIntVar(0, len(objects) - 1, str(node))
                # Guardar mapping objecte -> índex
                if not hasattr(self, '_object_to_index'):
                    self._object_to_index = {}
                for idx, obj in enumerate(objects):
                    self._object_to_index[(fluent.type, obj)] = idx
            else:
                var = model.NewBoolVar(str(node))

            variables[node] = var
            return var

        # Parameters
        if node.is_parameter_exp():
            if node in variables:
                return variables[node]

            param = node.parameter()
            assert param.type.is_user_type(), f"Parameter type {param.type} not supported"
            # Obtenir tots els objectes d'aquest tipus
            objects = list(problem.objects(param.type))
            if not objects:
                # Si no hi ha objectes, pot ser un tipus sense instàncies
                var = model.NewIntVar(0, 0, str(node))
            else:
                var = model.NewIntVar(0, len(objects) - 1, str(node))
                # Guardar mapping
                self._register_usertype_mapping(param.type, objects)

            variables[node] = var
            return var

        # Equality
        if node.is_equals():
            left = node.arg(0)
            right = node.arg(1)
            if left.type.is_user_type():
                left_var = self._add_cp_constraints(problem, left, variables, model)
                # Si right és un objecte/paràmetre constant
                if right.is_object_exp():
                    obj = right.object()
                    idx = self._object_to_index.get((left.type, obj))
                    if idx is not None:
                        eq_var = model.NewBoolVar(f"eq_{id(node)}")
                        model.Add(left_var == idx).OnlyEnforceIf(eq_var)
                        model.Add(left_var != idx).OnlyEnforceIf(eq_var.Not())
                        return eq_var

                # Si right és un altre fluent/variable
                else:
                    right_var = self._add_cp_constraints(problem, right, variables, model)
                    eq_var = model.NewBoolVar(f"eq_{id(node)}")
                    model.Add(left_var == right_var).OnlyEnforceIf(eq_var)
                    model.Add(left_var != right_var).OnlyEnforceIf(eq_var.Not())
                    return eq_var
            else:
                left = self._add_cp_constraints(problem, node.arg(0), variables, model)
                right = self._add_cp_constraints(problem, node.arg(1), variables, model)
                eq_var = model.NewBoolVar(f"eq_{id(node)}")
                model.Add(left == right).OnlyEnforceIf(eq_var)
                model.Add(left != right).OnlyEnforceIf(eq_var.Not())
                return eq_var

        # AND
        if node.is_and():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            and_var = model.NewBoolVar(f"and_{id(node)}")
            # Això crea la implicació: and_var == true <=> tots els child_vars == true
            model.AddBoolAnd(*child_vars).OnlyEnforceIf(and_var)
            # També necessitem: si algun child és false, and_var és false
            for child in child_vars:
                model.AddImplication(and_var, child)
            return and_var

        # OR
        if node.is_or():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            or_var = model.NewBoolVar(f"or_{id(node)}")
            # or_var == true <=> almenys un child_var == true
            model.AddBoolOr(*child_vars).OnlyEnforceIf(or_var)
            # Si or_var és false, tots els children són false
            for child in child_vars:
                model.AddImplication(child, or_var)
            return or_var

        # Implies: A -> B  equivalent to  (not A) or B
        if node.is_implies():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)

            impl_var = model.NewBoolVar(f"impl_{id(node)}")

            # impl_var == true <=> (not left) or right
            # Equivalent: impl_var == true <=> left == false OR right == true

            # Si impl_var és true: not(left) or right ha de ser true
            model.AddBoolOr(left.Not(), right).OnlyEnforceIf(impl_var)

            # Si impl_var és false: left ha de ser true AND right ha de ser false
            model.Add(left == 1).OnlyEnforceIf(impl_var.Not())
            model.Add(right == 0).OnlyEnforceIf(impl_var.Not())
            return impl_var

        # Not
        if node.is_not():
            inner_var = self._add_cp_constraints(problem, node.arg(0), variables, model)
            not_var = model.NewBoolVar(f"not_{id(node)}")
            # not_var és la negació d'inner_var
            model.Add(not_var == (1 - inner_var))
            return not_var

        # Comparisons and arithmetic
        if node.is_lt():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)

            lt_var = model.NewBoolVar(f"lt_{id(node)}")
            model.Add(left < right).OnlyEnforceIf(lt_var)
            model.Add(left >= right).OnlyEnforceIf(lt_var.Not())

            return lt_var

        if node.is_le():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)

            le_var = model.NewBoolVar(f"le_{id(node)}")
            model.Add(left <= right).OnlyEnforceIf(le_var)
            model.Add(left > right).OnlyEnforceIf(le_var.Not())

            return le_var

        # Arithmetic - retorna expressions lineals
        if node.is_plus():
            args = [self._add_cp_constraints(problem, arg, variables, model)
                    for arg in node.args]
            return sum(args)

        if node.is_minus():
            args = [self._add_cp_constraints(problem, arg, variables, model)
                    for arg in node.args]
            if len(args) == 1:
                return -args[0]
            return args[0] - sum(args[1:])

        if node.is_times():
            args = [self._add_cp_constraints(problem, arg, variables, model)
                    for arg in node.args]
            return math.prod(args)

        raise NotImplementedError(f"Node type {node.node_type} not implemented in CP-SAT")

    def _add_effect_bounds_constraints(
            self,
            problem: Problem,
            effects: List[Effect],
            variables: bidict,
            model: cp_model.CpModel
    ) -> None:
        """
        Add constraints to ensure all arithmetic effects produce in-bounds values.

        For unconditional effects: lb <= expr <= ub
        For conditional effects: condition => (lb <= expr <= ub)
        """
        for effect in effects:
            # Only handle arithmetic effects
            if effect.value.node_type not in self.ARITHMETIC_OPS:
                continue

            # Get target fluent bounds
            target_fluent = effect.fluent.fluent()
            if not target_fluent.type.is_int_type():
                continue

            lb = target_fluent.type.lower_bound
            ub = target_fluent.type.upper_bound

            # Build CP expression for the effect value
            effect_expr = self._add_cp_constraints(problem, effect.value, variables, model)

            # Check if effect is conditional
            is_conditional = (effect.condition is not None and
                              not effect.condition.is_true() and
                              effect.condition != TRUE())

            if not is_conditional:
                # Unconditional effect: directly add bounds
                model.Add(effect_expr >= lb)
                model.Add(effect_expr <= ub)
            else:
                # Conditional effect: Implies(condition, bounds)
                condition_var = self._add_cp_constraints(problem, effect.condition, variables, model)

                # condition => (effect_expr >= lb)
                model.Add(effect_expr >= lb).OnlyEnforceIf(condition_var)
                # condition => (effect_expr <= ub)
                model.Add(effect_expr <= ub).OnlyEnforceIf(condition_var)

    def _solve_with_cp_sat(self, variables, cp_model_obj):
        """Use CP-SAT solver to enumerate valid value assignments."""
        # Solve
        solver = cp_model.CpSolver()
        collector = CPSolutionCollector(list(variables.values()))
        solver.parameters.enumerate_all_solutions = True
        status = solver.Solve(cp_model_obj, collector)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        solutions = collector.solutions
        return solutions

    def _solutions_to_dnf(self, new_problem: Problem, solutions: List[dict], variables: bidict) -> Optional[FNode]:
        """Convert CP-SAT solutions to DNF formula."""
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}
        or_clauses = []
        for solution in solutions:
            and_clauses = []

            for var_str, value in solution.items():
                fnode = var_str_to_fnode.get(var_str)
                if not fnode:
                    continue
                if fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)
                    if fluent.type.is_int_type():
                        and_clauses.append(new_fluent(*fnode.args, self._get_number_object(value)))
                    elif fluent.type.is_user_type():
                        obj = self._get_object_from_index(fluent.type, value)
                        if obj:
                            and_clauses.append(Equals(new_fluent, ObjectExp(obj)))
                    elif fluent.type.is_bool_type():
                        if value == 1:
                            and_clauses.append(new_fluent)
                        else:
                            and_clauses.append(Not(new_fluent))
                elif fnode.is_parameter_exp():
                    param = fnode.parameter()
                    if param.type.is_user_type():
                        obj = self._get_object_from_index(param.type, value)
                        if obj:
                            and_clauses.append(Equals(fnode, ObjectExp(obj)))

            if and_clauses:
                or_clauses.append(And(and_clauses) if len(and_clauses) > 1 else and_clauses[0])
        if not or_clauses:
            return None
        return Or(or_clauses).simplify() if len(or_clauses) > 1 else or_clauses[0].simplify()

    # =================== EXPRESSION TRANSFORMATION ===================

    def _convert_new_expression(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode"
    ) -> up.model.fnode.FNode:
        # faltara afegir que quan un UNIFORM fluent que conte parametre d'una accio (clar no esta instanciat,
        # thus no sabem el valor de l'static fluent, pero sabem que mai canvia per tant el substituim per el seu corresponent value
        operation_map = {
            OperatorKind.EQUALS: 'equals',  # The only operator that works within arrays
            OperatorKind.LT: 'lt',
            OperatorKind.LE: 'le',
            #OperatorKind.PLUS: 'plus',
            #OperatorKind.MINUS: 'minus',
            #OperatorKind.DIV: 'div',
            #OperatorKind.TIMES: 'mult',
        }
        operation = operation_map.get(node.node_type)
        # de mom suporta equals i lt
        if operation is not None:
            fluent, value = (node.arg(0), node.arg(1)) if node.arg(0).is_fluent_exp() else (node.arg(1), node.arg(0))
            if fluent.type.is_int_type():
                new_fluents, new_values = self._convert_fluent_and_value(new_problem, fluent, value)
                if operation == 'equals':
                    and_node = []
                    for f, v in zip(new_fluents, new_values):
                        if value.is_fluent_exp():
                            and_node.append(Iff(f, v))
                        else:
                            and_node.append(f if v else Not(f))
                    return And(and_node)

                elif operation == 'lt':
                    if value.is_fluent_exp():
                        or_node = []
                        iff_node = []
                        for f, v in zip(new_fluents, new_values):
                            new_and_node = And(Not(f), v)
                            or_node.append(And(iff_node, new_and_node))
                            iff_node.append(Iff(f, v))
                        return Or(or_node)
                    else:
                        # Comparació fluent < constant
                        value_bits = self._convert_value(value.constant_value(), len(new_fluents))
                        # Lexicographic comparison: f < v
                        # Find first position where f has 0 and v has 1
                        # and all previous bits are equal
                        or_terms = []
                        equals_prefix = []  # Acumula igualtat dels bits anteriors

                        for i in range(len(new_fluents) - 1, -1, -1):
                            f_bit = new_fluents[i]
                            v_bit = value_bits[i]

                            if v_bit:  # v té 1 en aquesta posició
                                # f < v si: (bits anteriors iguals) AND (f té 0 aquí) AND (v té 1)
                                term = And(equals_prefix + [Not(f_bit)])
                                or_terms.append(term)
                            # Si v té 0 en aquesta posició, no pot ser f < v amb aquest prefix
                            # Afegeix igualtat d'aquest bit per al següent prefix
                            if v_bit:
                                equals_prefix.append(f_bit)
                            else:
                                equals_prefix.append(Not(f_bit))

                        return Or(or_terms) if or_terms else FALSE()

                elif operation == 'le':
                    if value.is_fluent_exp():
                        # Fluent <= fluent (equals OR lt)
                        and_node = []
                        for f, v in zip(new_fluents, new_values):
                            and_node.append(Iff(f, v))
                        equals_node = And(and_node)
                        or_node = []
                        iff_node = []
                        for f, v in zip(new_fluents, new_values):
                            new_and_node = And(Not(f), v)
                            or_node.append(And(iff_node, new_and_node))
                            iff_node.append(Iff(f, v))
                        lt_node = Or(or_node)
                        return Or(equals_node, lt_node)
                    else:
                        value_bits = self._convert_value(value.constant_value(), len(new_fluents))
                        # Equals part
                        equals_and = []
                        for f, v_bit in zip(new_fluents, value_bits):
                            equals_and.append(f if v_bit else Not(f))
                        equals_node = And(equals_and)
                        # LT part (optimized)
                        or_terms = []
                        equals_prefix = []
                        for i in range(len(new_fluents) - 1, -1, -1):
                            f_bit = new_fluents[i]
                            v_bit = value_bits[i]
                            if v_bit:
                                term = And(equals_prefix + [Not(f_bit)])
                                or_terms.append(term)
                            if v_bit:
                                equals_prefix.append(f_bit)
                            else:
                                equals_prefix.append(Not(f_bit))
                        lt_node = Or(or_terms) if or_terms else FALSE()

                        return Or(equals_node, lt_node)

                elif operation == 'plus' or operation == 'minus' or operation == 'div' or operation == 'mult':
                    raise NotImplementedError(f"Operation {operation} not supported")
            elif fluent.type.is_array_type() and operation == 'equals':
                new_fluents, new_values = self._convert_fluent_and_value(new_problem, fluent, value)
                and_node = []
                for i in range(len(new_fluents)):
                    for f, v in zip(new_fluents[i], new_values[i]):
                        if value.is_fluent_exp():
                            and_node.append(Iff(f, v))
                        else:
                            and_node.append(f if v else Not(f))
                return And(and_node)
        elif node.is_constant() or node.is_variable_exp() or node.is_fluent_exp() or node.is_timing_exp() or node.is_parameter_exp():
            return node
        elif node.args:
            new_args = [
                self._convert_new_expression(new_problem, arg)
                for arg in node.args
            ]
            em = new_problem.environment.expression_manager
            return em.create_node(node.node_type, tuple(new_args)).simplify()
        return node

    def _convert_value(self, value, n_bits):
        """Convert integer value to binary list of n_bits."""
        return [b == '1' for b in bin(value)[2:].zfill(n_bits)]

    def _replace_static(self, node: FNode) -> FNode:
        """Substitueix static fluents"""
        if node.is_fluent_exp() and node in self._static_fluents:
            return self._static_fluents[node]
        if node.is_constant() or node.is_parameter_exp():
            return node
        if not node.args:
            return node
        new_args = [self._replace_static(arg) for arg in node.args]
        if all(n is o for n, o in zip(new_args, node.args)):
            return node  # Sense canvis
        em = node.environment.expression_manager
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _transform_node(
            self, old_problem: Problem, node: FNode
    ) -> Union[Union[str, FNode], Any]:
        """Transform expression node to use Number objects instead of integers."""
        em = self._new_problem.environment.expression_manager
        # Integer constants become Number objects
        if node.is_int_constant():
            return self._get_number_object(node.constant_value())

        # Other terminals
        if node.is_object_exp() or node.is_constant() or node.is_parameter_exp() or node.is_fluent_exp():
            return node

        if node.is_equals():
            fluent, constant = (node.arg(0), node.arg(1)) if node.arg(0).is_fluent_exp() else (node.arg(1), node.arg(0))
            # Integer fluents
            if fluent.type.is_int_type():
                return self._new_problem.fluent(fluent.fluent().name)(*fluent.args, self._get_number_object(constant.constant_value()))
            return node

        # Recursively transform children
        new_args = []
        for arg in node.args:
            transformed = self._transform_node(old_problem, arg)
            new_args.append(transformed)

        return em.create_node(node.node_type, tuple(new_args)).simplify()

    # ==================== ACTION TRANSFORMATION ====================

    def _get_assign_effect(
            self, problem, fluent, value, condition, forall, solutions, variables: bidict
    ) -> Iterator[Effect]:
        """Convert arithmetic assign effects to multiple assignments via conditional effects. """
        var_str_to_fnode = {str(k): k for k in variables.keys()}
        result_to_deps = self._project_and_group_by_result(value, condition, solutions)
        returned = set()
        for rk, data in result_to_deps.items():
            new_fluent = self._transform_node(problem, Equals(fluent, data['value']))
            new_value = TRUE()

            if len(result_to_deps) == 1:
                cond = TRUE()
            else:
                conditions = [
                    self._build_pattern_condition(var_str_to_fnode, dict(dv))
                    for dv in data['combinations']
                ]
                conditions = [c for c in conditions if c]
                cond = Or(conditions).simplify() if len(conditions) > 1 else conditions[0] if conditions else TRUE()

            ek = (str(new_fluent), str(new_value), str(cond))
            if ek not in returned:
                yield Effect(new_fluent, new_value, cond, EffectKind.ASSIGN, forall)
                returned.add(ek)

    def _add_effects_dnf_mode(
            self,
            new_action: InstantaneousAction,
            old_problem: Problem,
            variables: bidict,
            solutions: List[dict],
            unstatic_effects: List[Effect]
    ) -> None:
        """Add effects for DNF mode, handling arithmetic with conditional effects if needed."""
        for effect in unstatic_effects:
            if effect.is_increase():
                value = effect.fluent + 1
            elif effect.is_decrease():
                value = effect.fluent - 1
            else:
                value = effect.value

            for new_effect in self._get_assign_effect(
                    old_problem, effect.fluent, value, effect.condition, effect.forall, solutions, variables
            ):
                new_action.add_effect(new_effect.fluent, new_effect.value, new_effect.condition, new_effect.forall)

    def _add_effects_for_entry_bits(self, new_action, new_problem, old_problem, variables, entry, normalized_effects):
        pattern = entry['pattern']
        subtable_tuples = entry.get('subtable_tuples', [])

        entry_solutions = [dict(pattern)]
        if subtable_tuples:
            entry_solutions = [{**pattern, **t} for t in subtable_tuples]

        filtered_effects = []
        for effect in normalized_effects:
            if effect.condition != TRUE():
                cond_result = self._evaluate_with_solution(effect.condition, pattern)
                if cond_result is False or cond_result == FALSE():
                    continue
            filtered_effects.append(effect)

        self._add_effects_dnf_mode(new_action, new_problem, old_problem, variables, entry_solutions, filtered_effects)

    def _create_instantiated_actions(
            self,
            old_action: Action,
            problem: Problem,
            new_problem: Problem,
            params: OrderedDict,
            solutions: List[dict],
            variables: bidict,
            unstatic_effects: List[Effect],
    ) -> List[Action]:
        var_str_to_fnode = {str(k): k for k in variables.keys()}
        new_actions = []

        # ===== COMPRESSION =====
        if self.COMPRESSION_MODE == "mfi" and len(solutions) > 1:
            min_support = self._compute_min_support(len(solutions), len(variables))
            maximal = self._compute_maximal_itemsets_apriori(
                solutions, min_support=min_support, max_itemset_size=len(variables) - 1
            )
            if maximal:
                sorted_mfi = self._sort_by_area(maximal)
                selected, remaining = self._select_non_overlapping_mfi(solutions, sorted_mfi)
                entries, default_tuples = self._create_entries(solutions, selected, remaining)
            else:
                entries, default_tuples = [], solutions
        else:
            entries, default_tuples = [], solutions

        # ===== ACTIONS FOR EACH ENTRY (MFI) =====
        for idx, entry in enumerate(entries):
            action_name = f"{old_action.name}_e{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)

            # Preconditions of the pattern (constant variables)
            for var_str, value in entry['pattern'].items():
                cond = self._build_condition_from_combination(new_problem, var_str_to_fnode, {var_str: value})
                if cond:
                    new_action.add_precondition(cond)

            # Precondicions del subtable (variables que varien → OR de tuples)
            if entry['subtable_tuples']:
                if len(entry['subtable_tuples']) == 1:
                    subtable_cond = self._build_condition_from_combination(
                        new_problem, var_str_to_fnode, entry['subtable_tuples'][0]
                    )
                else:
                    clauses = [
                        self._build_condition_from_combination(new_problem, var_str_to_fnode, t)
                        for t in entry['subtable_tuples']
                    ]
                    clauses = [c for c in clauses if c]
                    subtable_cond = Or(clauses).simplify() if len(clauses) > 1 else clauses[0] if clauses else None
                if subtable_cond:
                    new_action.add_precondition(subtable_cond)

            # Effects
            self._add_effects_for_entry_bits(
                new_action, new_problem, problem, variables, entry, unstatic_effects
            )
            new_actions.append(new_action)

        # ===== AN ACTION FOR EACH SOLUTION (NAIVE) =====
        for idx, solution in enumerate(default_tuples):
            action_name = f"{old_action.name}_d{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)
            # Preconditions
            for var_str, value in solution.items():
                cond = self._build_condition_from_combination(new_problem, var_str_to_fnode, {var_str: value})
                if cond:
                    new_action.add_precondition(cond)

            # Effects: entry sense subtable (solució completa)
            entry = {'pattern': solution, 'subtable_vars': [], 'subtable_tuples': []}
            self._add_effects_for_entry_bits(
                new_action, new_problem, problem, variables, entry, unstatic_effects
            )
            new_actions.append(new_action)

        return new_actions

    def _get_param_combinations(self, signature):
        """Get all combinations of parameter values."""
        param_values = [self._new_problem.objects(param.type) for param in signature]
        return list(product(*param_values))

    def _create_fluent_and_instantiate(self, old_action, solutions, variables):
        # Els paràmetres del fluent són NOMÉS els TIPUS dels fluents (Number), NO els objectes (Counter)
        new_action_parameters = []
        fluent_parameters = []
        i = 1
        for var in variables.keys():
            var_type = self._number_ut if var.fluent().type.is_int_type() else var.fluent().type
            param_name = f'p{i}' # new action parameter
            new_action_parameters.append(Parameter(param_name, var_type))
            fluent_parameters.append(Parameter(param_name, var_type))
            i +=1

        # Creating the new action fluent
        fluent_name = old_action.name + "_fluent"
        new_fluent = Fluent(fluent_name, BoolType(), fluent_parameters)

        # Afegir el fluent al problema
        self._new_problem.add_fluent(new_fluent, default_initial_value=False)

        # Instanciar el fluent NOMÉS amb els valors de les solucions (n0, n1, n2, n3)
        for solution in solutions:
            fluent_args = []
            for var in variables.keys():
                var_key = str(var)
                if var_key in solution:
                    solution_value = solution[var_key]
                    if var.fluent().type.is_int_type():
                        fluent_args.append(self._get_number_object(solution_value))
                    else:
                        fluent_args.append(solution_value)

            # Establir el valor inicial del fluent a True per aquesta combinació
            self._new_problem.set_initial_value(new_fluent(*fluent_args), True)

        # Create new action
        action_name = f"{old_action.name}"
        print("new_action_parameters", new_action_parameters)
        params = OrderedDict(((p.name, p.type) for p in old_action.parameters + new_action_parameters))

        new_action = InstantaneousAction(action_name, _parameters=params, _env=self._new_problem.environment)

        print("newww action_parameters", new_action.parameters)
        print("new_fluent", new_fluent)

        # Retornar l'expressió del fluent cridat
        i = 0
        new_args_fluent = []
        for var in variables.keys():
            # Add variable as precondition
            fluent_args = [*var.args]
            updated_fluent = self._new_problem.fluent(var.fluent().name)
            fluent_args.append(new_action_parameters[i])

            new_action.add_precondition(updated_fluent(*fluent_args))
            new_args_fluent.append(new_action_parameters[i])
            i += 1

        # Add new action fluent as precondition
        new_action.add_precondition(new_fluent(*new_args_fluent))

        return new_action, new_action_parameters

    def _transform_preconditions_action(
            self, problem: Problem, old_action: Action
    ) -> Tuple[Action, List[Parameter]]:
        """
        Transform an action preconditions with an instantiated predicate
        which is satisfied for all possible combination values of the expression constraints.
        """
        # Replace static fluents in preconditions -- what about UNIFORM FLUENTS ?
        unstatic_preconditions = []
        for precondition in old_action.preconditions:
            np = self._replace_static(precondition)
            if np.is_and():
                for new_precondition in np.args:
                    if new_precondition is not TRUE():
                        unstatic_preconditions.append(new_precondition)
            else:
                if np is not TRUE():
                    unstatic_preconditions.append(np)

        # Replace static fluents in effects
        unstatic_effects = []
        for effect in old_action.effects:
            new_value = self._replace_static(effect.value)
            new_condition = self._replace_static(effect.condition)
            unstatic_effect = Effect(effect.fluent, new_value, new_condition, effect.kind, effect.forall)
            unstatic_effects.append(unstatic_effect)

        # ===== CP-SAT =====
        self._object_to_index = {}
        self._index_to_object = {}
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        # Add preconditions as constraints
        result_var = self._add_cp_constraints(problem, And(unstatic_preconditions), variables, cp_model_obj)
        cp_model_obj.Add(result_var == 1)

        # Add effect bounds constraints
        self._add_effect_bounds_constraints(problem, unstatic_effects, variables, cp_model_obj)

        # Solve
        solutions = self._solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            return None

        # Create and instantiate new fluent with the possible combinations & add pre
        new_action, new_parameters = self._create_fluent_and_instantiate(old_action, solutions, variables)

        print("NEWWWWW", new_action)

        # Add effects with conditional effects if needed
        self._add_effects_dnf_mode(new_action, problem, variables, solutions, unstatic_effects)
        return new_action, new_parameters

    def _transform_actions(self, problem: Problem):
        """Transform all actions by replacing all preconditions with a new instantiated predicate."""
        new_to_old = {}
        for action in problem.actions:
            new_action, new_parameters = self._transform_preconditions_action(problem, action)
            self._new_problem.add_action(new_action)
            new_to_old[new_action] = (action, new_parameters) # (action, instantiation)
        return new_to_old

    # ==================== FLUENT TRANSFORMATION ====================

    def _transform_fluents(self, problem: Problem):
        """Transform integer fluents -> user-type fluents."""

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            if fluent.type.is_int_type():
                # Integer fluent -> Boolean fluent with user-type parameter

                new_fluent = Fluent(fluent.name, BoolType(), fluent.signature + [Parameter('n', self._number_ut)],
                                    self._new_problem.environment)
                lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
                assert lb is not None and ub is not None
                self._domains[fluent.name] = (lb, ub)

                if default_value is not None:
                    default_obj = self._get_number_object(default_value.constant_value())
                    print(new_fluent, default_obj)
                    self._new_problem.add_fluent(new_fluent, default_initial_value=False)
                else:
                    self._new_problem.add_fluent(new_fluent)

                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        self._new_problem.set_initial_value(
                            self._new_problem.fluent(fluent.name)(*f.args, self._get_number_object(v.constant_value())),
                            True
                        )
            else:
                self._new_problem.add_fluent(fluent, default_initial_value=default_value)
                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        self._new_problem.set_initial_value(f, v)

    # ==================== GOALS TRANSFORMATION ====================

    def _transform_goals(self, problem: Problem, new_problem: Problem) -> None:
        """
        Transform goals, using CP-SAT only if there's arithmetic.
        """
        # Replace static fluents
        non_static_goals = []
        for goal in problem.goals:
            ng = self._replace_static(goal)
            if ng.is_and():
                for g in ng.args:
                    if g is not TRUE():
                        non_static_goals.append(g)
            else:
                if ng is not TRUE():
                    non_static_goals.append(ng)
        if not non_static_goals:
            return

        has_arithmetic = any(self._has_arithmetic(g) for g in non_static_goals)

        # ===== NO ARITHMETIC: Direct transformation =====
        if not has_arithmetic:
            for goal in non_static_goals:
                transformed = self._transform_node(problem, goal)
                if transformed and transformed != TRUE():
                    new_problem.add_goal(transformed)
            return
        # ===== HAS ARITHMETIC: Use CP-SAT =====
        self._object_to_index = {}
        self._index_to_object = {}
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        result_var = self._add_cp_constraints(new_problem, And(non_static_goals), variables, cp_model_obj)
        cp_model_obj.Add(result_var == 1)

        solutions = self._solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            raise UPProblemDefinitionError("No possible goal!")

        dnf_formula = self._solutions_to_dnf(new_problem, solutions, variables)

        if dnf_formula:
            new_problem.add_goal(dnf_formula)
        else:
            raise UPProblemDefinitionError("No possible goal!")

    def _transform_action_costs(
            self,
            qm: MinimizeActionCosts,
            new_to_old: Dict[Action, Tuple[Action, Tuple[Action, FNode]]]
    ) -> Dict[Action, Expression]:
        """Transform action costs, substituting integer parameters."""
        new_costs = {}
        for new_action, (old_action, instantiation) in new_to_old.items():
            if old_action is None:
                continue
            old_cost = qm.get_action_cost(old_action)
            # If cost is a parameter, substitute its value
            if old_cost.is_parameter_exp():
                # Find which integer parameter this is
                param_idx = 0
                for param in old_action.parameters:
                    if param.name == str(old_cost):
                        break
                    if param.type.is_int_type():
                        param_idx += 1
                new_costs[new_action] = Int(instantiation[param_idx])
            else:
                new_costs[new_action] = old_cost
        return new_costs

    def _ground_minimize_action_costs_metric(
            self, metric: MinimizeActionCosts,
            trace_back_map: Dict[Action, Tuple[Action, List[FNode]]]
    ) -> MinimizeActionCosts:
        """
        Support method for a grounder to handle the MinimizeActionCosts metric.

        :param metric: The metric to convert.
        :param trace_back_map: The grounding map from a grounded Action to the Action
            and parameters that created the grounded action.
        :param simplifier: The simplifier used to evaluate the cost; if this simplifier
            has the Instance of the problem at construction time, it will also substitute
            the static fluents in the action cost with their value.
        :return: The equivalent MinimizeActionCosts metric for the grounded problem.
        """
        new_costs: Dict[Action, Expression] = {}
        for new_action, (old_action, params) in trace_back_map.items():
            old_cost = metric.get_action_cost(old_action)
            print("old cost", old_cost)
            if old_cost is not None:
                new_costs[new_action] = old_cost
        return MinimizeActionCosts(new_costs)

    def _compile(
            self,
            problem: "up.model.AbstractProblem",
            compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """
        """
        assert isinstance(problem, Problem)

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_quality_metrics()
        new_problem.initial_values.clear()

        self._new_problem = new_problem
        self._number_ut = UserType('Number')

        # === Find Static and Uniform Fluents ===
        self._static_fluents = self._get_static_fluents(problem, problem.initial_values)
        self._uniform_fluents = self._get_uniform_static_fluents(problem)
        print("STATIC FLUENTS", self._static_fluents)
        print("UNIFORM FLUENTS", self._uniform_fluents)

        # ========== Transform Fluents ==========
        self._transform_fluents(problem)

        # ========== Transform Actions ==========
        new_to_old: Dict[Action, Tuple[Action, List[FNode]]] = self._transform_actions(problem)

        # =========== Transform Goals ===========
        self._transform_goals(problem, new_problem)

        # ====== Transform Quality Metrics ======
        for qm in problem.quality_metrics:
            if qm.is_minimize_action_costs():
                assert isinstance(qm, MinimizeActionCosts)
                new_metric = self._ground_minimize_action_costs_metric(
                    qm, new_to_old
                )
                new_problem.add_quality_metric(new_metric)
            else:
                new_problem.add_quality_metric(qm)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
