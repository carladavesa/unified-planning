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
"""This module defines the arrays logarithmic remover class."""
import math
from itertools import product
from bidict import bidict
from unified_planning.model.operators import OperatorKind
from ortools.sat.python import cp_model

import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import (
    Problem,
    Action,
    ProblemKind, Object, MinimizeActionCosts, FNode, AbstractProblem, InstantaneousAction, Effect, EffectKind,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action,
)
from unified_planning.exceptions import UPProblemDefinitionError
from typing import Dict, List, Optional, OrderedDict, Iterable, Tuple, Iterator
from functools import partial
from unified_planning.shortcuts import And, Not, Iff, Equals, Or, FALSE, TRUE
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


class ArraysLogarithmicRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes integer arrays by encoding each integer value as Boolean bits.

    Integer and integer-array fluents are transformed into bit-level Boolean fluents,
    preserving semantics through bitwise precondition/effect rewrites.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.ARRAYS_LOGARITHMIC_REMOVING)
        self.n_bits = OrderedDict()
        self._static_fluents: Dict[FNode, FNode] = {}
        self._object_to_index = {}
        self._index_to_object = {}

    @property
    def name(self):
        return "irm"

    # Operators that can appear inside arithmetic expressions
    ARITHMETIC_OPS = {
        OperatorKind.PLUS: 'plus',
        OperatorKind.MINUS: 'minus',
        OperatorKind.DIV: 'div',
        OperatorKind.TIMES: 'mult',
    }

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
        # supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
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
        return problem_kind <= ArraysLogarithmicRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.ARRAYS_LOGARITHMIC_REMOVING

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("INT_FLUENTS")
        new_kind.unset_conditions_kind("ARRAY_FLUENTS")
        return new_kind

    # ==================== HELPER METHODS FOR ARITHMETIC ====================

    def _replace_static(self, node: FNode) -> FNode:
        """Replace static fluents with their constant values."""
        if node.is_fluent_exp() and node in self._static_fluents:
            return self._static_fluents[node]
        if node.is_constant() or node.is_parameter_exp():
            return node
        if not node.args:
            return node
        new_args = [self._replace_static(arg) for arg in node.args]
        if all(n is o for n, o in zip(new_args, node.args)):
            return node
        em = node.environment.expression_manager
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _has_arithmetic(self, node: FNode) -> bool:
        """Check if expression contains arithmetic operations or comparisons."""
        if (node.node_type in self.ARITHMETIC_OPS or
                node.is_le() or node.is_lt()):
            return True
        return any(self._has_arithmetic(arg) for arg in node.args)

    def _requires_cp_in_condition(self, node: FNode) -> bool:
        """Determine if a condition requires CP-SAT solver."""
        if node.node_type in self.ARITHMETIC_OPS:
            return True
        if node.is_lt() or node.is_le():
            return True
        if node.is_equals():
            left, right = node.arg(0), node.arg(1)
            if left.type.is_int_type() or right.type.is_int_type():
                if self._has_arithmetic(left) or self._has_arithmetic(right):
                    return True
        return any(self._requires_cp_in_condition(arg) for arg in node.args if arg.args)

    def _find_static_fluents(self, problem: Problem, fluents: dict[FNode, FNode]) -> Dict[FNode, FNode]:
        """Find fluents that are never modified by any action."""
        static_fluents = {}
        modified_fluents = set()

        for action in problem.actions:
            for effect in action.effects:
                modified_fluents.add(effect.fluent.fluent().name)

        for fluent_exp, value in fluents.items():
            if fluent_exp.fluent().name not in modified_fluents:
                static_fluents[fluent_exp] = value

        return static_fluents

    def _get_object_from_index(self, user_type, index):
        """Get object from index mapping."""
        return self._index_to_object.get((user_type, index))

    # ==================== CP-SAT CONSTRAINT SOLVING ====================

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

            if fluent.type.is_int_type():
                var = model.NewIntVar(
                    fluent.type.lower_bound,
                    fluent.type.upper_bound,
                    str(node)
                )
            elif fluent.type.is_user_type():
                objects = list(problem.objects(fluent.type))
                var = model.NewIntVar(0, len(objects) - 1, str(node))
                if not hasattr(self, '_object_to_index'):
                    self._object_to_index = {}
                for idx, obj in enumerate(objects):
                    self._object_to_index[(fluent.type, obj)] = idx
                    self._index_to_object[(fluent.type, idx)] = obj
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
            objects = list(problem.objects(param.type))
            if not objects:
                raise UPProblemDefinitionError(
                    f"User type {param.type} has no objects, cannot create variable for parameter {param}")
            var = model.NewIntVar(0, len(objects) - 1, str(node))
            variables[node] = var
            return var

        # Equality
        if node.is_equals():
            left = node.arg(0)
            right = node.arg(1)
            if left.type.is_user_type():
                left_var = self._add_cp_constraints(problem, left, variables, model)
                if right.is_object_exp():
                    obj = right.object()
                    idx = self._object_to_index.get((left.type, obj))
                    if idx is not None:
                        eq_var = model.NewBoolVar(f"eq_{id(node)}")
                        model.Add(left_var == idx).OnlyEnforceIf(eq_var)
                        model.Add(left_var != idx).OnlyEnforceIf(eq_var.Not())
                        return eq_var
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
            model.AddBoolAnd(*child_vars).OnlyEnforceIf(and_var)
            for child in child_vars:
                model.AddImplication(and_var, child)
            return and_var

        # OR
        if node.is_or():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            or_var = model.NewBoolVar(f"or_{id(node)}")
            model.AddBoolOr(*child_vars).OnlyEnforceIf(or_var)
            for child in child_vars:
                model.AddImplication(child, or_var)
            return or_var

        # Implies
        if node.is_implies():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)
            impl_var = model.NewBoolVar(f"impl_{id(node)}")
            model.AddBoolOr(left.Not(), right).OnlyEnforceIf(impl_var)
            model.Add(left == 1).OnlyEnforceIf(impl_var.Not())
            model.Add(right == 0).OnlyEnforceIf(impl_var.Not())
            return impl_var

        # Not
        if node.is_not():
            inner_var = self._add_cp_constraints(problem, node.arg(0), variables, model)
            not_var = model.NewBoolVar(f"not_{id(node)}")
            model.Add(not_var == (1 - inner_var))
            return not_var

        # Comparisons
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

        # Arithmetic - returns linear expressions
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
            result = args[0]
            for arg in args[1:]:
                # Get bounds for multiplication result
                if hasattr(arg, 'type'):
                    lb, ub = arg.type.lower_bound, arg.type.upper_bound
                else:
                    lb, ub = -10000, 10000  # Default bounds
                temp = model.NewIntVar(lb, ub, f"mult_{id(node)}")
                model.AddMultiplicationEquality(temp, result, arg)
                result = temp
            return result

        if node.is_div():
            args = [self._add_cp_constraints(problem, arg, variables, model)
                    for arg in node.args]
            result = args[0]
            for arg in args[1:]:
                if hasattr(arg, 'type'):
                    lb, ub = arg.type.lower_bound, arg.type.upper_bound
                else:
                    lb, ub = -10000, 10000
                temp = model.NewIntVar(lb, ub, f"div_{id(node)}")
                model.AddDivisionEquality(temp, result, arg)
                result = temp
            return result

        raise NotImplementedError(f"Node type {node.node_type} not implemented in CP-SAT")

    def _add_effect_bounds_constraints(self, problem, variables, cp_model_obj, unstatic_effects):
        """
        Add constraints to ensure all arithmetic effects produce in-bounds values.

        For unconditional effects: lb <= expr <= ub
        For conditional effects: condition => (lb <= expr <= ub)
        """
        for effect in unstatic_effects:
            # Only handle arithmetic effects
            if effect.value.node_type not in self.ARITHMETIC_OPS:
                continue

            # Get target fluent bounds
            target_fluent = effect.fluent.fluent()
            if not target_fluent.type.is_int_type():
                continue

            lb, ub = target_fluent.type.lower_bound, target_fluent.type.upper_bound
            # Registers the fluent variable
            fluent_var = self._add_cp_constraints(problem, effect.fluent, variables, cp_model_obj)

            if effect.is_increase() or effect.is_decrease():
                try:
                    delta = effect.value.constant_value()
                except:
                    continue
                if effect.is_increase():
                    result_expr = fluent_var + delta
                else:
                    result_expr = fluent_var - delta

                if effect.condition is not None and not effect.condition.is_true():
                    cond_var = self._add_cp_constraints(problem, effect.condition, variables, cp_model_obj)
                    cp_model_obj.Add(result_expr >= lb).OnlyEnforceIf(cond_var)
                    cp_model_obj.Add(result_expr <= ub).OnlyEnforceIf(cond_var)
                else:
                    cp_model_obj.Add(result_expr >= lb)
                    cp_model_obj.Add(result_expr <= ub)
            else:
                # Assignment effect: bounds apply iff the effect is active.
                expr = self._add_cp_constraints(problem, effect.value, variables, cp_model_obj)
                if effect.condition is not None and not effect.condition.is_true():
                    if effect.condition.is_false():
                        continue
                    cond_var = self._add_cp_constraints(problem, effect.condition, variables, cp_model_obj)
                    cp_model_obj.Add(expr >= lb).OnlyEnforceIf(cond_var)
                    cp_model_obj.Add(expr <= ub).OnlyEnforceIf(cond_var)
                else:
                    cp_model_obj.Add(expr >= lb)
                    cp_model_obj.Add(expr <= ub)

    def _solve_with_cp_sat(self, variables, cp_model_obj):
        """Use CP-SAT solver to enumerate all valid value assignments."""
        solver = cp_model.CpSolver()
        collector = CPSolutionCollector(list(variables.values()))
        solver.parameters.enumerate_all_solutions = True
        status = solver.Solve(cp_model_obj, collector)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        solutions = collector.solutions
        return solutions

    def _evaluate_with_solution(
            self,
            new_problem: Problem,
            expr: FNode,
            solution: dict,
    ) -> Optional[FNode]:
        """Evaluate arithmetic expression with a specific variable assignment."""
        if expr.is_constant():
            return expr

        if expr.is_fluent_exp():
            var_name = str(expr)
            if var_name in solution:
                value = solution[var_name]
                return new_problem.environment.expression_manager.Int(value)
            return None

        if expr.is_plus():
            args = [self._evaluate_with_solution(new_problem, arg, solution) for arg in expr.args]
            if all(a and a.is_int_constant() for a in args):
                result = sum(a.constant_value() for a in args)
                return new_problem.environment.expression_manager.Int(result)

        if expr.is_minus():
            args = [self._evaluate_with_solution(new_problem, arg, solution) for arg in expr.args]
            if all(a and a.is_int_constant() for a in args):
                if len(args) == 1:
                    result = -args[0].constant_value()
                else:
                    result = args[0].constant_value() - sum(a.constant_value() for a in args[1:])
                return new_problem.environment.expression_manager.Int(result)

        if expr.is_times():
            args = [self._evaluate_with_solution(new_problem, arg, solution) for arg in expr.args]
            if all(a and a.is_int_constant() for a in args):
                result = 1
                for a in args:
                    result *= a.constant_value()
                return new_problem.environment.expression_manager.Int(result)

        if expr.is_div():
            args = [self._evaluate_with_solution(new_problem, arg, solution) for arg in expr.args]
            if all(a and a.is_int_constant() for a in args):
                result = args[0].constant_value()
                for a in args[1:]:
                    if a.constant_value() != 0:
                        result //= a.constant_value()
                return new_problem.environment.expression_manager.Int(result)

        return None

    def _solutions_to_dnf_bits(self, problem: Problem, new_problem: Problem, solutions: List[dict], variables: bidict) -> Optional[FNode]:
        """Convert CP-SAT solutions to DNF formula using bit representations."""
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
                    if fluent.type.is_int_type():
                        # Convert integer value to bit representation
                        name_fluent = fluent.name.split('[')[0]
                        n_bits = self.n_bits[name_fluent]
                        value_bits = self._convert_value(value, n_bits)

                        # Get bit fluents
                        new_fluents = self._get_new_fluent(new_problem, fnode)

                        # Create equality constraints for each bit
                        for f, bit_val in zip(new_fluents, value_bits):
                            and_clauses.append(f if bit_val else Not(f))

            if and_clauses:
                or_clauses.append(And(and_clauses) if len(and_clauses) > 1 else and_clauses[0])

        if not or_clauses:
            return None
        return Or(or_clauses).simplify() if len(or_clauses) > 1 else or_clauses[0].simplify()


    def _add_effects_with_cp(
            self,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            variables: bidict,
            solutions: List[dict],
            unstatic_effects: List[Effect]
    ) -> None:
        """
        Add effects - EXACTLY like integers_remover._add_effects_for_solution
        but adapted for bit encoding.

        For each effect:
        1. Evaluate condition with solution → if FALSE, skip; if TRUE, unconditional
        2. Transform to bits what remains
        """
        # For MODE="multiple", typically one solution per action
        if len(solutions) != 1:
            return

        solution = solutions[0]

        for effect in unstatic_effects:
            if effect.is_increase() or effect.is_decrease():
                # ✅ For arrays_remover: work with bit fluents directly
                fluent = effect.fluent.fluent()
                try:
                    delta = effect.value.constant_value()
                except:
                    for new_eff in self._transform_increase_decrease_effect(effect, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                fluent_var_str = str(effect.fluent)
                cur_val = solution.get(fluent_var_str)
                if cur_val is None:
                    # No value in solution → fallback
                    for new_eff in self._transform_increase_decrease_effect(effect, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                next_val = (cur_val + delta) if effect.is_increase() else (cur_val - delta)

                # ✅ Get bit fluents for current and next values
                name_fluent = fluent.name.split('[')[0]
                if name_fluent not in self.n_bits:
                    # Not a transformed fluent → skip
                    continue

                n_bits = self.n_bits[name_fluent]
                cur_bits = self._convert_value(cur_val, n_bits)
                next_bits = self._convert_value(next_val, n_bits)
                new_fluents = self._get_new_fluent(new_problem, effect.fluent)

                # ✅ Evaluate effect condition
                if effect.condition is not None and not effect.condition.is_true():
                    cond_result = self._evaluate_with_solution(new_problem, effect.condition, solution)
                    if cond_result == FALSE():
                        continue  # Condition always false → skip
                    elif cond_result == TRUE():
                        cond = TRUE()
                    else:
                        cond = self._get_new_expression(new_problem, effect.condition)
                        if cond is None:
                            cond = TRUE()
                else:
                    cond = TRUE()

                # Add bit effects
                for f, next_bit in zip(new_fluents, next_bits):
                    new_action.add_effect(f, TRUE() if next_bit else FALSE(), cond, effect.forall)

            elif effect.value.node_type in self.ARITHMETIC_OPS:
                new_fluent = self._get_new_fluent(new_problem, effect.fluent) if effect.fluent.is_fluent_exp() else effect.fluent
                base_cond = self._get_new_expression(new_problem, effect.condition) or TRUE()
                result_fnode = self._evaluate_with_solution(new_problem, effect.value, solution)
                if result_fnode and result_fnode != effect.value:
                    # Convert result to bits if needed
                    if effect.fluent.is_fluent_exp():
                        fluent_name = effect.fluent.fluent().name.split('[')[0]
                        if fluent_name in self.n_bits and result_fnode.is_int_constant():
                            n_bits = self.n_bits[fluent_name]
                            result_val = result_fnode.constant_value()
                            result_bits = self._convert_value(result_val, n_bits)
                            new_fluents = self._get_new_fluent(new_problem, effect.fluent)
                            for f, bit_val in zip(new_fluents, result_bits):
                                new_action.add_effect(f, TRUE() if bit_val else FALSE(), base_cond)
                        else:
                            new_action.add_effect(new_fluent, result_fnode, base_cond)
                    else:
                        new_action.add_effect(new_fluent, result_fnode, base_cond)

            else:
                # Simple assignment
                new_fluent = self._get_new_fluent(new_problem, effect.fluent) if effect.fluent.is_fluent_exp() else effect.fluent
                new_value = self._get_new_expression(new_problem, effect.value) if effect.value.is_fluent_exp() or not effect.value.is_constant() else effect.value

                # ✅ Evaluate condition - THIS IS THE KEY!
                cond_result = self._evaluate_with_solution(new_problem, effect.condition, solution)
                if cond_result == FALSE():
                    continue  # Skip if always false
                elif cond_result == TRUE():
                    new_cond = TRUE()
                else:
                    new_cond = self._get_new_expression(new_problem, cond_result if cond_result else effect.condition)
                    if new_cond is None:
                        new_cond = TRUE()

                # Convert to bits
                if effect.fluent.is_fluent_exp():
                    fluent_name = effect.fluent.fluent().name.split('[')[0]
                    if fluent_name in self.n_bits:
                        if effect.value.is_constant():
                            n_bits = self.n_bits[fluent_name]
                            value_bits = self._convert_value(effect.value.constant_value(), n_bits)
                            bit_fluents = self._get_new_fluent(new_problem, effect.fluent)
                            for f, bit_val in zip(bit_fluents, value_bits):
                                new_action.add_effect(f, TRUE() if bit_val else FALSE(), new_cond, effect.forall)
                        elif effect.value.is_fluent_exp():
                            new_fluents, new_values = self._convert_fluent_and_value(new_problem, effect.fluent, effect.value)
                            if effect.fluent.fluent().type.is_int_type():
                                for f, v in zip(new_fluents, new_values):
                                    new_action.add_effect(f, v, new_cond, effect.forall)
                        else:
                            pass  # Arithmetic in simple assignment (shouldn't reach here)
                    else:
                        new_action.add_effect(effect.fluent, effect.value, new_cond, effect.forall)
                else:
                    new_action.add_effect(effect.fluent, effect.value, new_cond, effect.forall)

    def _transform_increase_decrease_effect(
            self,
            effect: Effect,
            problem: Problem,
            new_problem: Problem,
    ) -> List[Effect]:
        """Convert increase/decrease effects to conditional bit assignments."""
        fluent = effect.fluent.fluent()
        lb, ub = fluent.type.lower_bound, fluent.type.upper_bound

        name_fluent = fluent.name.split('[')[0]
        n_bits = self.n_bits[name_fluent]

        # Calculate the valid bounds
        try:
            int_value = effect.value.constant_value()
        except:
            int_value = effect.value

        if effect.is_increase():
            valid_range = range(max(lb, lb), min(ub - int_value, ub) + 1) if isinstance(int_value, int) else range(lb, ub + 1)
        else:
            valid_range = range(max(lb + int_value, lb), min(ub, ub) + 1) if isinstance(int_value, int) else range(lb, ub + 1)

        returned = []

        for i in valid_range:
            next_val = (i + int_value) if effect.is_increase() else (i - int_value)
            try:
                next_val_int = next_val.simplify().constant_value() if hasattr(next_val, 'simplify') else next_val
            except:
                continue

            # Convert current and next values to bits
            current_bits = self._convert_value(i, n_bits)
            next_bits = self._convert_value(next_val_int, n_bits)

            # Get bit fluents
            new_fluents = self._get_new_fluent(new_problem, effect.fluent)

            # Build condition: fluent equals current value
            cond_clauses = []
            for f, bit_val in zip(new_fluents, current_bits):
                cond_clauses.append(f if bit_val else Not(f))
            value_condition = And(cond_clauses) if len(cond_clauses) > 1 else cond_clauses[0]

            # Combine with effect condition
            full_condition = And(value_condition, effect.condition).simplify() if effect.condition != TRUE() else value_condition

            # Create effects for each bit
            for f, next_bit in zip(new_fluents, next_bits):
                new_effect = Effect(
                    f,
                    TRUE() if next_bit else FALSE(),
                    full_condition,
                    EffectKind.ASSIGN,
                    effect.forall
                )
                returned.append(new_effect)

        return returned

    def _transform_goals(self, problem: Problem, new_problem: Problem) -> None:
        """
        Transform goals: separate arithmetic and non-arithmetic.
        Only use CP-SAT for arithmetic goals.
        Based on ALR latest implementation.
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

        # Check if there's any arithmetic
        has_arithmetic = any(self._has_arithmetic(g) for g in non_static_goals)

        # ===== NO ARITHMETIC: Direct transformation =====
        if not has_arithmetic:
            for goal in non_static_goals:
                transformed = self._get_new_expression(new_problem, goal)
                if transformed and transformed != TRUE():
                    new_problem.add_goal(transformed)
            return

        # ===== HAS ARITHMETIC: Use CP-SAT =====
        # Clear mappings
        self._object_to_index = {}
        self._index_to_object = {}

        # Build CP-SAT model with ALL goals
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        result_var = self._add_cp_constraints(problem, And(non_static_goals), variables, cp_model_obj)
        cp_model_obj.Add(result_var == 1)

        # Solve
        solutions = self._solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            raise UPProblemDefinitionError("No possible goal satisfies arithmetic constraints!")

        # Convert solutions to DNF and add as goal
        dnf_formula = self._solutions_to_dnf_bits(problem, new_problem, solutions, variables)

        if dnf_formula:
            new_problem.add_goal(dnf_formula)
        else:
            raise UPProblemDefinitionError("No possible goal!")

    def _transform_action_with_cp(
            self, problem: Problem, new_problem: Problem, old_action: Action
    ) -> List[InstantaneousAction]:
        """
        Transform action with integer arithmetic using CP-SAT, then encode to bits.

        Strategy:
        1. Replace static fluents
        2. Add ALL preconditions and effects to CP-SAT for filtering
        3. Solve to get all valid variable assignments
        4. Two modes (MODE selectable):
           - MODE="dnf": Single action with DNF precondition
           - MODE="multiple": Multiple actions (one per solution)
        """
        params = OrderedDict(((p.name, p.type) for p in old_action.parameters))

        # Replace static fluents in preconditions
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

        # Check if we have arithmetic
        has_arithmetic = any(self._requires_cp_in_condition(p) for p in unstatic_preconditions) or \
                        any(effect.value.node_type in self.ARITHMETIC_OPS or
                            effect.is_increase() or effect.is_decrease()
                            for effect in unstatic_effects)

        if not has_arithmetic:
            return []  # No arithmetic, shouldn't reach here

        # ===== Arithmetic path: use CP-SAT =====
        # Clear mappings
        self._object_to_index = {}
        self._index_to_object = {}

        # Build CP-SAT model
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        # Add ALL preconditions as constraints
        result_var = self._add_cp_constraints(problem, And(unstatic_preconditions), variables, cp_model_obj)
        cp_model_obj.Add(result_var == 1)

        # Add effect bounds as constraints
        self._add_effect_bounds_constraints(problem, variables, cp_model_obj, unstatic_effects)

        # Solve to get all valid variable assignments
        solutions = self._solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            return []

        # ===== MULTIPLE ACTIONS: One action per solution =====
        new_actions = []
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        for idx, solution in enumerate(solutions):
            action_name = f"{old_action.name}_s{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)

            # Add preconditions: conjunció dels valors concrets d'aquesta solució
            and_clauses = []
            for var_str, value in solution.items():
                fnode = var_str_to_fnode.get(var_str)
                if not fnode or not fnode.is_fluent_exp():
                    continue
                fluent = fnode.fluent()
                if fluent.type.is_int_type():
                    name_fluent = fluent.name.split('[')[0]
                    if name_fluent in self.n_bits:
                        n_bits = self.n_bits[name_fluent]
                        value_bits = self._convert_value(value, n_bits)
                        new_fluents = self._get_new_fluent(new_problem, fnode)
                        for f, bit_val in zip(new_fluents, value_bits):
                            and_clauses.append(f if bit_val else Not(f))

            if and_clauses:
                new_action.add_precondition(And(and_clauses) if len(and_clauses) > 1 else and_clauses[0])

            # Add effects per aquesta solució
            self._add_effects_with_cp(new_action, problem, new_problem, variables, [solution], unstatic_effects)

            new_actions.append(new_action)

        return new_actions

    # ==================== EXISTING METHODS ====================

    def _get_new_fluent(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
            indexes: Optional["up.model.fnode.FNode"] = None,
    ) -> List["up.model.fnode.FNode"]:
        """Return the bit-fluent expansion for an integer/int-array fluent expression."""
        assert node.is_fluent_exp()

        name_fluent = node.fluent().name.split('[')[0]
        n_bits = self.n_bits[name_fluent]

        if node.fluent().type.is_int_type():
            indexes = [int(i) for i in re.findall(r'\[([0-9]+)]', node.fluent().name)]
        if not indexes:
            return [
                new_problem.fluent(f"{name_fluent}_{i}")(*node.args)
                for i in range(n_bits)
            ]
        else:
            position_object_fluent = new_problem.object(f'p_{"_".join(map(str, indexes))}')
            return [
                new_problem.fluent(f"{name_fluent}_{i}")(*node.args, position_object_fluent)
                for i in range(n_bits)
            ]

    def _get_fluent_domain(
            self,
            fluent: "up.model.Fluent",
            save: bool = False
    ) -> Iterable[int]:
        """Return all index tuples for an array fluent; optionally cache required bit width."""
        domain = []
        inner_fluent = fluent.type

        while inner_fluent.is_array_type():
            domain.append(range(inner_fluent.size))
            inner_fluent = inner_fluent.elements_type

        assert inner_fluent.is_int_type(), f"Fluent {fluent.name} has not type int. Only integer arrays supported."

        # Save number of bits required to encode the integer domain
        if save:
            self.n_bits[fluent.name] = math.ceil(math.log2(inner_fluent.upper_bound + 1))

        return tuple(product(*domain))

    def _get_element_value(self, v, combination) -> "up.model.fnode.FNode":
        """Obtain the value of the element for a given combination of access."""
        element_value = v
        for c in combination:
            element_value = element_value.constant_value()[c]
        return element_value

    def _get_new_expression(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode"
    ) -> up.model.fnode.FNode:
        """Rewrite expressions over encoded fluents into equivalent Boolean formulas."""

        # Check if this node contains arithmetic that needs CP-SAT
        if self._has_arithmetic(node):
            # Use CP-SAT to solve arithmetic expressions
            variables = bidict({})
            cp_model_obj = cp_model.CpModel()

            # Build constraints
            result_var = self._add_cp_constraints(new_problem, node, variables, cp_model_obj)

            # Add constraint that result must be true
            if hasattr(result_var, 'Not'):  # It's a boolean variable
                cp_model_obj.Add(result_var == 1)

            # Solve
            solutions = self._solve_with_cp_sat(variables, cp_model_obj)

            if not solutions:
                return FALSE()

            # Convert solutions to DNF using bit representation
            return self._solutions_to_dnf_bits(new_problem, new_problem, solutions, variables)

        operation_map = {
            OperatorKind.EQUALS: 'equals',  # Supported arithmetic comparisons on encoded integers
            OperatorKind.LT: 'lt',
            OperatorKind.LE: 'le',
        }
        operation = operation_map.get(node.node_type)
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
                        # Compare encoded fluent < integer constant lexicographically
                        value_bits = self._convert_value(value.constant_value(), len(new_fluents))
                        or_terms = []
                        equals_prefix = []  # Equality constraints accumulated for more-significant bits

                        for i in range(len(new_fluents) - 1, -1, -1):
                            f_bit = new_fluents[i]
                            v_bit = value_bits[i]

                            if v_bit:  # v has 1 at this position
                                # f < v when higher bits are equal and f has 0 at this position
                                term = And(equals_prefix + [Not(f_bit)])
                                or_terms.append(term)
                            # If v has 0 here, this position cannot witness f < v
                            # Extend prefix equality for the next bit
                            if v_bit:
                                equals_prefix.append(f_bit)
                            else:
                                equals_prefix.append(Not(f_bit))

                        return Or(or_terms) if or_terms else FALSE()

                elif operation == 'le':
                    if value.is_fluent_exp():
                        # Fluent <= fluent is encoded as (equals OR less-than).
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
                self._get_new_expression(new_problem, arg)
                for arg in node.args
            ]
            em = new_problem.environment.expression_manager
            return em.create_node(node.node_type, tuple(new_args)).simplify()
        return node

    def _convert_value(self, value, n_bits):
        """Convert integer value to binary list of n_bits."""
        return [b == '1' for b in bin(value)[2:].zfill(n_bits)]

    def _set_fluent_bits(self, problem, fluent, k_args, new_value, n_bits, object_ref: Optional[Object] = None):
        """Set initial values for all bit fluents representing one encoded integer value."""
        for bit_index in range(n_bits):
            this_fluent = problem.fluent(f"{fluent.name}_{bit_index}")(*k_args, *(object_ref,) if object_ref is not None else ())
            problem.set_initial_value(this_fluent, new_value[bit_index])

    def _convert_fluent_and_value(
            self,
            new_problem: AbstractProblem,
            fluent: FNode,
            value: FNode,
    ) -> Tuple[List[FNode], List[FNode]]:
        """Convert a fluent/value pair into aligned bit-level representations."""
        name_fluent = fluent.fluent().name.split('[')[0]
        n_bits = self.n_bits[name_fluent]
        if fluent.type.is_int_type():
            new_fluents = self._get_new_fluent(new_problem, fluent)
            if value.is_fluent_exp():
                new_values = self._get_new_fluent(new_problem, value)
            else:
                assert value.is_constant(), "Value must be a constant!"
                new_values = self._convert_value(value.constant_value(), n_bits)
        else:
            assert fluent.type.is_array_type()
            indices = tuple(int(i) for i in re.findall(r'\[([0-9]+)]', fluent.fluent().name))
            new_fluents = []
            new_values = []
            if value.is_fluent_exp():
                value_fluent_domain = tuple(int(i) for i in re.findall(r'\[([0-9]+)]', value.fluent().name))
                for combination in self._get_fluent_domain(fluent.fluent()):
                    new_fluents.append(self._get_new_fluent(new_problem, fluent, indices + combination))
                    new_values.append(self._get_new_fluent(new_problem, value, value_fluent_domain + combination))
            else:
                for combination in self._get_fluent_domain(fluent.fluent()):
                    new_fluents.append(self._get_new_fluent(new_problem, fluent, indices + combination))
                    element_value = self._get_element_value(value, combination)
                    new_values.append(self._convert_value(element_value.constant_value(), n_bits))
        return new_fluents, new_values

    def _compile(
            self,
            problem: "up.model.AbstractProblem",
            compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)

        new_to_old: Dict[Action, Action] = {}

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_quality_metrics()
        new_problem.initial_values.clear()

        for fluent in problem.fluents:
            # Transform integer-array fluents into bit fluents indexed by position objects.
            internal_type = fluent.type
            while internal_type.is_array_type():
                internal_type = internal_type.elements_type
            if fluent.type.is_array_type() and internal_type.is_int_type():
                Position = new_problem.environment.type_manager.UserType('Position')
                new_parameter = up.model.Parameter('p', Position)

                combination = self._get_fluent_domain(fluent, True)
                for c in combination:
                    new_object = model.Object(f'p_{"_".join(map(str, c))}', Position)
                    if not new_problem.has_object(new_object.name):
                        new_problem.add_object(new_object)
                n_bits = self.n_bits[fluent.name]
                # Default initial values
                default_value = problem.fluents_defaults.get(fluent)
                default_bits = self._convert_value(default_value.constant_value(), n_bits) if default_value else [None] * n_bits
                for i in range(n_bits):
                    new_fluent = model.Fluent(f"{fluent.name}_{i}", _signature=fluent.signature + [new_parameter],
                                              environment=new_problem.environment)
                    new_problem.add_fluent(new_fluent, default_initial_value=default_bits[i])

                # Initial values
                for k, v in problem.explicit_initial_values.items():
                    fluent_name = k.fluent().name.split('[')[0]
                    # For entire arrays (e.g., puzzle = [[8,7,6], [0,4,1], [2,5,3]])
                    if k.fluent() == fluent:
                        for positions in combination:
                            element_value = self._get_element_value(v, positions)
                            new_value = self._convert_value(element_value.constant_value(), n_bits)
                            object_ref = new_problem.object(f'p_{"_".join(map(str, positions))}')
                            self._set_fluent_bits(new_problem, fluent, k.args, new_value, n_bits, object_ref)

                    # For sub-arrays (e.g., puzzle[0] = [8,7,6]) or specific elements (e.g., puzzle[0][0] = 8)
                    elif fluent_name == fluent.name:
                        explicit_domain = tuple(int(i) for i in re.findall(r'\[([0-9]+)]', k.fluent().name))
                        domain = self._get_fluent_domain(k.fluent()) if k.fluent().type.is_array_type() else [()]

                        for c in domain:
                            element_value = self._get_element_value(v, c)
                            new_value = self._convert_value(element_value.constant_value(), n_bits)

                            if new_value != default_bits:
                                combined_domain = explicit_domain + c if c else explicit_domain
                                object_ref = new_problem.object(f'p_{"_".join(map(str, combined_domain))}')
                                self._set_fluent_bits(new_problem, fluent, k.args, new_value, n_bits, object_ref)
            elif fluent.type.is_int_type():
                self._get_fluent_domain(fluent, True)
                n_bits = self.n_bits[fluent.name]
                # Default initial values
                default_value = problem.fluents_defaults.get(fluent)
                default_bits = self._convert_value(default_value.constant_value(), n_bits) if default_value \
                    else [None] * n_bits
                for i in range(n_bits):
                    new_fluent = model.Fluent(f"{fluent.name}_{i}", _signature=fluent.signature,
                                              environment=new_problem.environment)
                    new_problem.add_fluent(new_fluent, default_initial_value=default_bits[i])
                # Initial values
                for k, v in problem.explicit_initial_values.items():
                    if k.fluent() == fluent:
                        new_value = self._convert_value(v.constant_value(), n_bits)
                        self._set_fluent_bits(new_problem, fluent, k.args, new_value, n_bits)
            else:
                default_value = problem.fluents_defaults.get(fluent)
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for k, v in problem.explicit_initial_values.items():
                    if k.fluent().name == fluent.name and v != default_value:
                        new_problem.set_initial_value(k, v)

        # Find static fluents before transforming actions
        self._static_fluents = self._find_static_fluents(problem, problem.initial_values)

        print("_static_fluents", self._static_fluents)

        for action in problem.actions:
            # Check if action has arithmetic operations
            has_arithmetic_in_preconditions = any(
                self._requires_cp_in_condition(self._replace_static(p))
                for p in action.preconditions
            )
            has_arithmetic_in_effects = any(
                effect.value.node_type in self.ARITHMETIC_OPS or effect.is_increase() or effect.is_decrease()
                for effect in action.effects
            )

            if has_arithmetic_in_preconditions or has_arithmetic_in_effects:
                # Use CP-SAT path for actions with arithmetic
                new_actions = self._transform_action_with_cp(problem, new_problem, action)
                for new_action in new_actions:
                    new_problem.add_action(new_action)
                    new_to_old[new_action] = action
            else:
                # Standard transformation for non-arithmetic actions
                new_action = action.clone()
                new_action.name = get_fresh_name(new_problem, action.name)
                new_action.clear_preconditions()
                new_action.clear_effects()

                for precondition in action.preconditions:
                    new_precondition = self._get_new_expression(new_problem, precondition)
                    new_action.add_precondition(new_precondition)

                for effect in action.effects:
                    fluent = effect.fluent
                    value = effect.value
                    new_condition = self._get_new_expression(new_problem, effect.condition)

                    fluent_name = fluent.fluent().name.split('[')[0]
                    if fluent_name in self.n_bits:
                        new_fluents, new_values = self._convert_fluent_and_value(new_problem, fluent, value)
                        # Single encoded integer assignment
                        if fluent.fluent().type.is_int_type():
                            for f, v in zip(new_fluents, new_values):
                                new_action.add_effect(f, v, new_condition, effect.forall)
                        # Array/sub-array assignment expanded element-wise
                        else:
                            for i in range(len(new_fluents)):
                                for f, v in zip(new_fluents[i], new_values[i]):
                                    new_action.add_effect(f, v, new_condition, effect.forall)
                    else:
                        new_action.add_effect(fluent, value, new_condition, effect.forall)

                new_problem.add_action(new_action)
                new_to_old[new_action] = action

        # Transform goals properly (like ALR latest, not just direct transformation)
        self._transform_goals(problem, new_problem)

        for qm in problem.quality_metrics:
            if qm.is_minimize_sequential_plan_length() or qm.is_minimize_makespan():
                new_problem.add_quality_metric(qm)
            elif qm.is_minimize_action_costs():
                assert isinstance(qm, MinimizeActionCosts)
                new_costs: Dict["up.model.Action", "up.model.Expression"] = {}
                for new_act, old_act in new_to_old.items():
                    if old_act is None:
                        continue
                    new_costs[new_act] = qm.get_action_cost(old_act)
                new_problem.add_quality_metric(
                    MinimizeActionCosts(new_costs, environment=new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(qm)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
