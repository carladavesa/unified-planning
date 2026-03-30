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
"""This module defines the integers logarithmic remover class."""
import math
import re
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
    ProblemKind,
    FNode,
    AbstractProblem,
    InstantaneousAction,
    Effect,
    EffectKind,
    Object,
    MinimizeActionCosts, Fluent, Expression, Axiom
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    add_cp_constraints,
    add_effect_bounds_constraints,
    solve_with_cp_sat,
    get_fresh_name,
    replace_action,
    updated_minimize_action_costs, requires_arithmetic, substitute_modified_fluents, evaluate_goal_in_initial_state,
)
from unified_planning.exceptions import UPProblemDefinitionError
from typing import Dict, List, Optional, OrderedDict, Iterable, Tuple
from functools import partial
from unified_planning.shortcuts import And, Not, Iff, Or, FALSE, TRUE


class LogarithmicRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes bounded integers by encoding each integer value as Boolean bits.

    Integer fluents are transformed into bit-level Boolean fluents (using binary representation),
    preserving semantics through bitwise precondition/effect rewrites.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.LOGARITHMIC_REMOVING)
        self.n_bits = OrderedDict()
        self._object_to_index = {}
        self._index_to_object = {}
        self._goal_registry = []

    @property
    def name(self):
        return "alrm"

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
        #supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        # supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_parameters("REAL_ACTION_PARAMETERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
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
        return problem_kind <= LogarithmicRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.LOGARITHMIC_REMOVING

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("INT_FLUENTS")
        return new_kind

    # ==================== METHODS ====================

    def _convert_value(self, value: int, n_bits: int) -> List[bool]:
        """Convert integer value to binary list of n_bits."""
        return [b == '1' for b in bin(value)[2:].zfill(n_bits)]

    def _get_bit_fluents(self, new_problem: Problem, fluent_exp: FNode) -> List[FNode]:
        """Get the bit fluents representing an integer fluent."""
        fluent = fluent_exp.fluent()

        # Only integer fluents are transformed to bits
        if not fluent.type.is_int_type():
            return [fluent_exp]

        name = fluent.name
        if name not in self.n_bits:
            # Not a transformed fluent, return as-is
            return [fluent_exp]

        n_bits = self.n_bits[name]
        return [
            new_problem.fluent(f"{name}_{i}")(*fluent_exp.args)
            for i in range(n_bits)
        ]

    # ==================== NODE TRANSFORMATION ====================

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

        if expr.is_le():
            args = [self._evaluate_with_solution(new_problem, arg, solution) for arg in expr.args]
            if all(a and a.is_int_constant() for a in args):
                return TRUE() if args[0].constant_value() <= args[1].constant_value() else FALSE()

        if expr.is_lt():
            args = [self._evaluate_with_solution(new_problem, arg, solution) for arg in expr.args]
            if all(a and a.is_int_constant() for a in args):
                return TRUE() if args[0].constant_value() < args[1].constant_value() else FALSE()

        if expr.is_not():
            v = self._evaluate_with_solution(new_problem, expr.arg(0), solution)
            if v == TRUE(): return FALSE()
            if v == FALSE(): return TRUE()

        if expr.is_equals():
            args = [self._evaluate_with_solution(new_problem, arg, solution) for arg in expr.args]
            if all(a and a.is_int_constant() for a in args):
                return TRUE() if args[0].constant_value() == args[1].constant_value() else FALSE()

        return expr

    # ==================== ACTION TRANSFORMATION ====================

    def _add_effects_for_solution(
            self,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            variables: bidict,
            solution: dict,
            old_effects: List[Effect]
    ) -> None:
        """
        Add effects for bit encoding.
        Transform all effects: integer fluents to bits.
        Transform conditions properly.
        """
        for old_effect in old_effects:
            if old_effect.is_increase() or old_effect.is_decrease():
                # Increase/decrease effect
                fluent = old_effect.fluent.fluent()
                try:
                    delta = old_effect.value.constant_value()
                except:
                    for new_eff in self._transform_increase_decrease_effect(old_effect, problem, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                fluent_var_str = str(old_effect.fluent)
                cur_val = solution.get(fluent_var_str)
                if cur_val is None:
                    # No value in solution
                    for new_eff in self._transform_increase_decrease_effect(old_effect, problem, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                next_val = (cur_val + delta) if old_effect.is_increase() else (cur_val - delta)

                if fluent.type.is_int_type() and fluent.name in self.n_bits:
                    # Integer fluent: transform to bits
                    n_bits = self.n_bits[fluent.name]
                    next_bits = self._convert_value(next_val, n_bits)
                    bit_fluents = self._get_bit_fluents(new_problem, old_effect.fluent)

                    if old_effect.condition is not None and not old_effect.condition.is_true():
                        cond_result = self._evaluate_with_solution(new_problem, old_effect.condition, solution)
                        if cond_result == FALSE():
                            continue
                        elif cond_result == TRUE():
                            cond = TRUE()
                        else:
                            cond = self._get_new_expression(new_problem, old_effect.condition)
                    else:
                        cond = TRUE()

                    for f, bit_val in zip(bit_fluents, next_bits):
                        new_action.add_effect(f, TRUE() if bit_val else FALSE(), cond, old_effect.forall)
                else:
                    # Non-integer fluent: add as-is with transformed condition
                    new_fluent = self._get_new_expression(new_problem, old_effect.fluent)
                    if old_effect.condition is not None and not old_effect.condition.is_true():
                        cond_result = self._evaluate_with_solution(new_problem, old_effect.condition, solution)
                        if cond_result == FALSE():
                            continue
                        elif cond_result == TRUE():
                            cond = TRUE()
                        else:
                            cond = self._get_new_expression(new_problem, old_effect.condition)
                    else:
                        cond = TRUE()
                    # For non-integer, need to compute next value
                    next_obj = self._get_new_expression(new_problem, old_effect.value)
                    if new_fluent and next_obj:
                        new_action.add_effect(new_fluent, next_obj, cond, old_effect.forall)

            elif old_effect.value.node_type in self.ARITHMETIC_OPS:
                # Arithmetic assignment
                new_fluent = self._get_new_expression(new_problem, old_effect.fluent)
                new_value = self._evaluate_with_solution(new_problem, old_effect.value, solution)

                if old_effect.condition is not None and not old_effect.condition.is_true():
                    cond_result = self._evaluate_with_solution(new_problem, old_effect.condition, solution)
                    if cond_result == FALSE():
                        continue
                    elif cond_result == TRUE():
                        new_cond = TRUE()
                    else:
                        new_cond = self._get_new_expression(new_problem, old_effect.condition)
                else:
                    new_cond = self._get_new_expression(new_problem, old_effect.condition) or TRUE()
                new_value = self._evaluate_with_solution(new_problem, old_effect.value, solution)
                if new_value and new_cond != FALSE():
                    new_action.add_effect(new_fluent, new_value, new_cond)

            else:
                new_fluent = self._get_new_expression(new_problem, old_effect.fluent)
                new_value = self._get_new_expression(new_problem, old_effect.value)

                if old_effect.condition is not None and not old_effect.condition.is_true():
                    if requires_arithmetic(old_effect.condition):
                        result = self._evaluate_with_solution(new_problem, old_effect.condition, solution)
                        if result == TRUE() and new_fluent and new_value:
                            new_action.add_effect(new_fluent, new_value, TRUE(), old_effect.forall)
                        elif result == FALSE():
                            continue
                    else:
                        new_cond = self._get_new_expression(new_problem, old_effect.condition) or TRUE()
                        if new_fluent and new_value:
                            new_action.add_effect(new_fluent, new_value, new_cond, old_effect.forall)
                else:
                    if new_fluent and new_value:
                        new_action.add_effect(new_fluent, new_value, TRUE(), old_effect.forall)

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

    def _get_fluent_exps_in_expression(self, node: FNode) -> set:
        """Get all fluent expressions that appear in an expression."""
        result = set()
        if node.is_fluent_exp():
            result.add(node)
        for arg in node.args:
            result.update(self._get_fluent_exps_in_expression(arg))
        return result

    # ==================== GOALS TRANSFORMATION ====================

    def _add_goal_maintenance_effects(
            self,
            action: Action,
            goal_expr: FNode,
            goal_fluent_exp: FNode,
    ) -> None:
        """
        Add conditional effects to maintain the truth value of a goal fluent.
        This is called for each arithmetic goal to ensure the auxiliary fluent
        stays synchronized with the goal expression.
        """
        goal_fluent_exps = self._get_fluent_exps_in_expression(goal_expr)
        action_modifies_exps = {
            effect.fluent
            for effect in action.effects
            if effect.fluent.is_fluent_exp()
        }

        # Check if this action modifies any fluents in the goal
        goal_fluent_strs = {str(f) for f in goal_fluent_exps}
        action_modifies_strs = {str(f) for f in action_modifies_exps}

        if not goal_fluent_strs & action_modifies_strs:
            return

        if not goal_fluent_exps & action_modifies_exps:
            return

        # Replace modified fluents for its next expression
        substituted_goal = substitute_modified_fluents(action, goal_expr)

        # Add conditional effects with the goal expression as condition to maintain the goal fluent's value
        action.add_effect(goal_fluent_exp, TRUE(), substituted_goal)
        action.add_effect(goal_fluent_exp, FALSE(), Not(substituted_goal).simplify())

    def _transform_goals(self, problem: Problem, new_problem: Problem) -> None:
        """
        Transform goals: separate arithmetic and non-arithmetic.
        For arithmetic goals, create auxiliary boolean fluents.
        """
        non_arithmetic_goals = []
        arithmetic_goals = []

        for goal in problem.goals:
            if requires_arithmetic(goal):
                arithmetic_goals.append(goal)
            else:
                non_arithmetic_goals.append(goal)

        # Non-arithmetic goals - direct transformation
        for goal in non_arithmetic_goals:
            transformed = self._get_new_expression(new_problem, goal)
            if transformed and transformed != TRUE():
                new_problem.add_goal(transformed)

        # Create a predicate for each arithmetic goal
        self._goal_registry = []
        for i, goal in enumerate(arithmetic_goals):
            fluent_name = f"goal_{i}"
            from unified_planning.model import Fluent
            from unified_planning.shortcuts import BoolType
            goal_fluent = Fluent(fluent_name, BoolType())
            new_problem.add_fluent(goal_fluent, default_initial_value=FALSE())

            init_val = evaluate_goal_in_initial_state(problem, goal)
            new_problem.set_initial_value(goal_fluent(), TRUE() if init_val else FALSE())

            new_problem.add_goal(goal_fluent())
            self._goal_registry.append((goal, goal_fluent()))

    # ==================== AXIOMS TRANSFORMATION ====================

    def _transform_axioms(self, problem: Problem, new_problem: Problem, new_to_old: Dict):
        """Transform axioms"""
        for axiom in problem.axioms:
            params = OrderedDict((p.name, p.type) for p in axiom.parameters)
            # Clone and transform
            new_axiom_name = get_fresh_name(new_problem, axiom.name)
            new_axiom = Axiom(new_axiom_name, params, axiom.environment)

            skip_axiom = False
            new_axiom.set_head(axiom.head.fluent)
            for body in axiom.body:
                new_body = self._get_new_expression(new_problem, body)
                if new_body is None:
                    skip_axiom = True
                    break
                else:
                    new_axiom.add_body_condition(new_body)
            if skip_axiom:
                continue
            new_problem.add_axiom(new_axiom)
            new_to_old[new_axiom] = axiom

    def _create_multiple_actions(
            self,
            old_action: Action,
            problem: Problem,
            new_problem: Problem,
            params: OrderedDict,
            solutions: List[dict],
            variables: bidict,
            extra_preconditions: List[FNode] = [],
    ):
        """Create multiple actions for each original action, one per valid variable assignment."""
        # ===== MULTIPLE ACTIONS: One action per solution =====
        new_actions = []
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        for idx, solution in enumerate(solutions):
            action_name = f"{old_action.name}_s{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)

            # Non-arithmetic preconditions
            for prec in extra_preconditions:
                new_action.add_precondition(prec)

            # Add preconditions from solution
            for var_str, value in solution.items():
                fnode = var_str_to_fnode.get(var_str)
                if not fnode or not fnode.is_fluent_exp():
                    continue
                fluent = fnode.fluent()
                if fluent.type.is_int_type():
                    name_fluent = fluent.name
                    if name_fluent in self.n_bits:
                        n_bits = self.n_bits[name_fluent]
                        value_bits = self._convert_value(value, n_bits)
                        new_fluents = self._get_bit_fluents(new_problem, fnode)
                        for f, bit_val in zip(new_fluents, value_bits):
                            new_action.add_precondition(f if bit_val else Not(f))
                else:
                    new_action.add_precondition(fnode)

            # Effects
            self._add_effects_for_solution(new_action, problem, new_problem, variables, solution, old_action.effects)
            new_actions.append(new_action)

        return new_actions

    def _transform_action_integers(self, problem: Problem, new_problem: Problem, old_action: Action) -> List[Action]:
        params = OrderedDict(((p.name, p.type) for p in old_action.parameters))

        # Check if preconditions/effects require CP-SAT
        has_arithmetic_preconditions = any(requires_arithmetic(p) for p in old_action.preconditions)
        has_arithmetic_effects = any(
            effect.value.node_type in self.ARITHMETIC_OPS
            or effect.is_increase() or effect.is_decrease()
            or requires_arithmetic(effect.condition)
            for effect in old_action.effects
        )

        # No arithmetic: direct transformation, no CP-SAT needed
        if not has_arithmetic_preconditions and not has_arithmetic_effects:
            new_action = InstantaneousAction(old_action.name, _parameters=params, _env=problem.environment)

            for old_precondition in old_action.preconditions:
                new_precondition = self._get_new_expression(new_problem, old_precondition)
                if new_precondition and new_precondition != TRUE():
                    new_action.add_precondition(new_precondition)

            for old_effect in old_action.effects:
                new_condition = self._get_new_expression(new_problem, old_effect.condition)
                if old_effect.fluent.type.is_int_type():
                    new_fluents, new_values = self._convert_fluent_and_value(
                        new_problem, old_effect.fluent, old_effect.value
                    )
                    for f, v in zip(new_fluents, new_values):
                        new_action.add_effect(f, v, new_condition, old_effect.forall)
                else:
                    new_action.add_effect(old_effect.fluent, old_effect.value, new_condition, old_effect.forall)

            return [new_action]

        # Arithmetic: CP-SAT to enumerate all valid assignments
        self._object_to_index = {}
        self._index_to_object = {}

        variables = bidict({})
        cp_model_obj = cp_model.CpModel()
        non_arithmetic_precs = []

        if has_arithmetic_preconditions:
            # Add ALL preconditions as constraints
            result_var = add_cp_constraints(problem, And(old_action.preconditions), variables, cp_model_obj,
                                            self._object_to_index)
            cp_model_obj.Add(result_var == 1)
        else:
            for new_action in old_action.preconditions:
                transformed = self._get_new_expression(new_problem, new_action)
                if transformed and transformed != TRUE():
                    non_arithmetic_precs.append(transformed)

        # Add effect bounds as constraints
        add_effect_bounds_constraints(problem, variables, cp_model_obj, old_action.effects, self._object_to_index, True)
        # Solve to get all valid variable assignments
        solutions = solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            return []

        return self._create_multiple_actions(
            old_action, problem, new_problem, params, solutions, variables, non_arithmetic_precs
        )

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions by grounding integer parameters into binary representation."""
        new_to_old = {}
        for old_action in problem.actions:
            temporal_action = old_action.clone()
            for goal_expr, goal_fluent_exp in self._goal_registry:
                self._add_goal_maintenance_effects(
                    temporal_action, goal_expr, goal_fluent_exp
                )
            new_actions = self._transform_action_integers(problem, new_problem, temporal_action)
            for new_action in new_actions:
                new_problem.add_action(new_action)
                new_to_old[new_action] = old_action
        return new_to_old

    def _get_new_fluent(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
            indexes: Optional["up.model.fnode.FNode"] = None,
    ) -> List["up.model.fnode.FNode"]:
        """Return the bit-fluent expansion for an integer fluent expression."""
        assert node.is_fluent_exp()

        fluent = node.fluent()

        # Only integer fluents are transformed to bits
        if not fluent.type.is_int_type():
            return [node]

        name = fluent.name
        if name not in self.n_bits:
            # Not a transformed fluent, return as-is
            return [node]

        n_bits = self.n_bits[name]
        return [
            new_problem.fluent(f"{name}_{i}")(*node.args)
            for i in range(n_bits)
        ]

    def _get_fluent_domain(
            self,
            fluent: Fluent,
            save: bool = False
    ) -> Iterable[int]:
        """Calculate and cache the number of bits required for an integer fluent."""
        if not fluent.type.is_int_type():
            return []

        inner_fluent = fluent.type
        assert inner_fluent.is_int_type(), f"Fluent {fluent.name} must be integer type. Arrays should be removed beforehand."

        # Calculate and save number of bits required to encode the integer domain
        if save:
            self.n_bits[fluent.name] = math.ceil(math.log2(inner_fluent.upper_bound + 1))

        return [fluent.name]

    def _get_element_value(self, v, combination) -> "up.model.fnode.FNode":
        """Obtain the value of the element for a given combination of access."""
        element_value = v
        for c in combination:
            element_value = element_value.constant_value()[c]
        return element_value

    def _get_new_expression(
            self,
            new_problem: AbstractProblem,
            node: FNode
    ) -> FNode:
        """
        Transform expressions over encoded fluents into equivalent Boolean formulas.
        """
        # Handle equality on integers
        if node.node_type == OperatorKind.EQUALS:
            left, right = node.arg(0), node.arg(1)

            # Check if one side is an integer fluent
            if left.is_fluent_exp() and left.fluent().type.is_int_type():
                fluent, value = left, right
            elif right.is_fluent_exp() and right.fluent().type.is_int_type():
                fluent, value = right, left
            else:
                # Not an integer equality, just recurse
                new_args = [self._get_new_expression(new_problem, arg) for arg in node.args]
                em = new_problem.environment.expression_manager
                return em.create_node(node.node_type, tuple(new_args)).simplify()

            # Get bit fluents and values
            new_fluents, new_values = self._convert_fluent_and_value(new_problem, fluent, value)

            # All bits must match
            and_clauses = []
            for f, v in zip(new_fluents, new_values):
                if value.is_fluent_exp():
                    and_clauses.append(Iff(f, v))
                else:
                    and_clauses.append(f if v else Not(f))
            return And(and_clauses) if len(and_clauses) > 1 else and_clauses[0]

        # For non-comparison nodes: recurse and transform recursively
        if node.is_constant() or node.is_parameter_exp() or node.is_timing_exp():
            return node
        elif node.is_fluent_exp():
            return node
        elif node.args:
            new_args = [self._get_new_expression(new_problem, arg) for arg in node.args]
            em = new_problem.environment.expression_manager
            return em.create_node(node.node_type, tuple(new_args)).simplify()
        return node

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

    def _transform_fluents(self, problem: Problem, new_problem: Problem):
        """
        Transform integer fluents into bit-level boolean fluents.

        Each integer fluent becomes n_bits boolean fluents (one per bit).
        Non-integer fluents are copied unchanged.
        """
        for fluent in problem.fluents:
            if fluent.type.is_int_type():
                # Calculate bits needed for this integer fluent
                self._get_fluent_domain(fluent, save=True)
                n_bits = self.n_bits[fluent.name]

                # Default initial values
                default_value = problem.fluents_defaults.get(fluent)
                default_bits = self._convert_value(default_value.constant_value(), n_bits) if default_value else [False] * n_bits

                # Create bit fluents
                for i in range(n_bits):
                    bit_fluent = model.Fluent(f"{fluent.name}_{i}", _signature=fluent.signature, environment=new_problem.environment)
                    new_problem.add_fluent(bit_fluent, default_initial_value=default_bits[i])

                # Set initial values from the original problem
                for k, v in problem.explicit_initial_values.items():
                    if k.fluent() == fluent:
                        new_value = self._convert_value(v.constant_value(), n_bits)
                        self._set_fluent_bits(new_problem, fluent, k.args, new_value, n_bits)
            else:
                # Non-integer fluent: copy as-is
                default_value = problem.fluents_defaults.get(fluent)
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for k, v in problem.explicit_initial_values.items():
                    if k.fluent() == fluent:
                        new_problem.set_initial_value(k, v)


    def _compile(
            self,
            problem: AbstractProblem,
            compilation_kind: CompilationKind,
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

        # ========== Transform Fluents ==========
        self._transform_fluents(problem, new_problem)

        # ========== Transform Goals ==========
        self._transform_goals(problem, new_problem)

        # ========== Transform Actions ==========
        new_to_old = self._transform_actions(problem, new_problem)

        # ========== Transform Quality Metrics ==========
        for metric in problem.quality_metrics:
            if metric.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(
                        metric,
                        new_to_old,
                        new_problem.environment
                    )
                )
            else:
                new_problem.add_quality_metric(metric)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
