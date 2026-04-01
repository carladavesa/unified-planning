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
"""This module defines the integers remover class."""
import operator
import unified_planning as up
import unified_planning.engines as engines
from bidict import bidict
from ortools.sat.python import cp_model
from unified_planning.engines.compilers.utils import (
    add_cp_constraints, add_effect_bounds_constraints, compute_integer_range, solve_with_cp_sat, requires_arithmetic,
    substitute_modified_fluents, evaluate_goal_in_initial_state, get_fluent_exps_in_expression, evaluate_with_solution
)
from typing import Any
from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import (
    Problem, Action, ProblemKind, Effect, EffectKind, Object, FNode, InstantaneousAction, Axiom
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import get_fresh_name, replace_action, updated_minimize_action_costs
from typing import Optional, Iterator, OrderedDict, Union
from functools import partial
from unified_planning.shortcuts import And, Or, Equals, Not, FALSE, UserType, TRUE, ObjectExp
from typing import Dict

class IntegersRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes bounded integers from a planning problem.

    Converts integer fluents to object-typed fluents where objects represent numeric values (n0, n1, n2, ...).
    Integer arithmetic and comparisons are handled by enumerating possible value combinations.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGERS_REMOVING)

    @property
    def name(self):
        return "irm"

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
        #supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
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
        supported_kind.set_conditions_kind("COUNTING")
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
        return problem_kind <= IntegersRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INTEGERS_REMOVING

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("INT_FLUENTS")
        new_kind.unset_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        return new_kind

    # Operators that can appear inside arithmetic expressions
    ARITHMETIC_OPS = {
        OperatorKind.PLUS: 'plus',
        OperatorKind.MINUS: 'minus',
        OperatorKind.DIV: 'div',
        OperatorKind.TIMES: 'mult',
    }

    # ==================== METHODS ====================

    def _get_number_object(self, problem: Problem, value: int) -> FNode:
        """Get or create object representing numeric value (e.g., n5 for 5)."""
        try:
            return ObjectExp(problem.object(f'n{value}'))
        except UPProblemDefinitionError:
            raise UPProblemDefinitionError(
                f"Number object 'n{value}' not found. "
                f"Ensure _compute_number_range covers all needed values."
            )

    # ==================== NODE TRANSFORMATION ====================

    def _transform_node(
            self, old_problem: Problem, new_problem: Problem, node: FNode
    ) -> Union[Union[None, str, FNode], Any]:
        """Transform expression node to use Number objects instead of integers."""
        em = new_problem.environment.expression_manager

        # Integer constants become Number objects
        if node.is_int_constant():
            return self._get_number_object(new_problem, node.constant_value())

        # Integer fluents
        if node.is_fluent_exp():
            if node.fluent().type.is_int_type():
                return new_problem.fluent(node.fluent().name)(*node.args)
            return node

        # Other terminals
        if node.is_object_exp() or node.is_constant() or node.is_parameter_exp():
            return node

        # Check for arithmetic operations
        if node.node_type in self.ARITHMETIC_OPS:
            raise UPProblemDefinitionError(
                f"Arithmetic operation {self.ARITHMETIC_OPS[node.node_type]} "
                f"not supported as external expression"
            )

        # Recursively transform children
        new_args = []
        for arg in node.args:
            transformed = self._transform_node(old_problem, new_problem, arg)
            if transformed is None:
                return None
            new_args.append(transformed)

        return em.create_node(node.node_type, tuple(new_args)).simplify()

    # ==================== ACTION TRANSFORMATION ====================

    def _transform_increase_decrease_effect(
            self,
            effect,
            new_problem: Problem,
    ) -> Iterator[Effect]:
        """
        Convert increase/decrease effects to conditional assignments.

        For each valid value in the fluent's domain, creates a conditional effect
        that applies when the fluent has that value. The condition ensures the
        result stays within bounds.
        """
        fluent = effect.fluent.fluent()
        lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
        new_fluent = new_problem.fluent(fluent.name)(*effect.fluent.args)

        # Calculate the valid bounds
        try:
            int_value = effect.value.constant_value()
        except:
            int_value = effect.value

        if effect.is_increase():
            # For increase: final value = i + delta, so i must be in [lb, ub-delta]
            valid_range = range(max(lb, lb), min(ub - int_value, ub) + 1) if isinstance(int_value, int) else range(lb, ub + 1)
        else:
            # For decrease: final value = i - delta, so i must be in [lb+delta, ub]
            valid_range = range(max(lb + int_value, lb), min(ub, ub) + 1) if isinstance(int_value, int) else range(lb, ub + 1)

        returned = set()

        for i in valid_range:
            next_val = (i + int_value) if effect.is_increase() else (i - int_value)
            try:
                next_val_int = next_val.simplify().constant_value() if hasattr(next_val, 'simplify') else next_val
            except:
                continue

            old_obj = self._get_number_object(new_problem, i)
            new_obj = self._get_number_object(new_problem, next_val_int)
            new_effect = Effect(
                new_fluent,
                new_obj,
                And(Equals(new_fluent, old_obj), effect.condition).simplify(),
                EffectKind.ASSIGN,
                effect.forall
            )
            if new_effect not in returned:
                yield new_effect
                returned.add(new_effect)

    def _create_precondition_from_variable(
            self,
            fnode: FNode,
            value: int,
            new_problem: Problem
    ) -> Optional[FNode]:
        """
        Create a precondition from a variable and its value.
        Handles int fluents, UserType fluents, bool fluents, and parameters.
        """
        if fnode.is_fluent_exp():
            fluent = fnode.fluent()
            new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
            if fluent.type.is_int_type():
                num_obj = self._get_number_object(new_problem, value)
                return Equals(new_fluent, num_obj)
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

    def _create_multiple_actions(self, old_action, problem, new_problem, params, solutions, variables):
        new_actions = []

        prec_fluent_strs = set()
        for prec in old_action.preconditions:
            for f in get_fluent_exps_in_expression(prec):
                prec_fluent_strs.add(str(f))

        modified_fluent_strs = {
            str(effect.fluent) for effect in old_action.effects
            if str(effect.fluent) not in prec_fluent_strs
               and not effect.is_increase() and not effect.is_decrease()
               and effect.condition.is_true()
        }

        for idx, solution in enumerate(solutions):
            action_name = f"{old_action.name}_d{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)

            and_clauses = []
            for fnode, var in variables.items():
                var_str = str(fnode)
                if var_str in modified_fluent_strs:
                    continue
                if var_str not in solution:
                    continue
                cond = self._create_precondition_from_variable(fnode, solution[var_str], new_problem)
                if cond:
                    and_clauses.append(cond)
            if and_clauses:
                new_action.add_precondition(And(and_clauses) if len(and_clauses) > 1 else and_clauses[0])

            self._add_effects_for_solution(new_action, problem, new_problem, solution, old_action.effects)
            new_actions.append(new_action)

        return new_actions

    def _expand_condition_with_cp(self, problem, new_problem, condition, solution):
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()
        result_var = add_cp_constraints(problem, condition, variables, cp_model_obj, self._object_to_index)

        for fnode, var in list(variables.items()):
            if str(fnode) in solution:
                cp_model_obj.Add(var == solution[str(fnode)])

        cp_model_obj.Add(result_var == 1)
        true_solutions = solve_with_cp_sat(variables, cp_model_obj) or []

        unknown_vars = {str(fnode): fnode for fnode, var in variables.items()
                        if str(fnode) not in solution}

        clauses = []
        for sol in true_solutions:
            sol_conds = []
            for var_str, fnode in unknown_vars.items():
                if var_str not in sol:
                    continue
                cond = self._create_precondition_from_variable(fnode, sol[var_str], new_problem)
                if cond:
                    sol_conds.append(cond)
            if sol_conds:
                clauses.append(And(sol_conds).simplify() if len(sol_conds) > 1 else sol_conds[0])

        if not clauses:
            return []
        full_cond = Or(clauses).simplify() if len(clauses) > 1 else clauses[0]
        return [(TRUE(), full_cond)]

    def _add_effects_for_solution(self, new_action, problem, new_problem, solution, old_effects):
        for old_effect in old_effects:
            # Evaluate condition
            if old_effect.condition.is_true():
                new_condition = TRUE()
            else:
                new_condition = evaluate_with_solution(new_problem, old_effect.condition, solution)
                if new_condition == FALSE():
                    continue

            # Increase/decrease effect
            if old_effect.is_increase() or old_effect.is_decrease():
                fluent = old_effect.fluent.fluent()
                new_fluent = new_problem.fluent(fluent.name)(*old_effect.fluent.args)
                try:
                    delta = old_effect.value.constant_value()
                except:
                    for new_eff in self._transform_increase_decrease_effect(old_effect, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                cur_val = solution.get(str(old_effect.fluent))
                if cur_val is None:
                    for new_eff in self._transform_increase_decrease_effect(old_effect, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                next_val = (cur_val + delta) if old_effect.is_increase() else (cur_val - delta)
                new_obj = self._get_number_object(new_problem, next_val)
                if not new_condition.is_true() and requires_arithmetic(new_condition):
                    expansions = self._expand_condition_with_cp(problem, new_problem, new_condition, solution)
                    for _, cond in expansions:
                        new_action.add_effect(new_fluent, new_obj, cond, old_effect.forall)
                else:
                    new_action.add_effect(new_fluent, new_obj, new_condition, old_effect.forall)

            # Integer assignment
            elif old_effect.fluent.type.is_int_type():
                new_fluent = self._transform_node(problem, new_problem, old_effect.fluent)
                evaluated_val = evaluate_with_solution(new_problem, old_effect.value, solution)
                new_value = self._transform_node(problem, new_problem, evaluated_val)
                if not new_fluent or not new_value:
                    continue
                if not new_condition.is_true() and requires_arithmetic(new_condition):
                    expansions = self._expand_condition_with_cp(problem, new_problem, new_condition, solution)
                    for _, cond in expansions:
                        new_action.add_effect(new_fluent, new_value, cond, old_effect.forall)
                else:
                    new_action.add_effect(new_fluent, new_value, new_condition, old_effect.forall)

            # Non-integer assignment
            else:
                new_fluent = self._transform_node(problem, new_problem, old_effect.fluent)
                new_value = self._transform_node(problem, new_problem, old_effect.value)
                if not new_fluent or not new_value:
                    continue
                if not new_condition.is_true() and requires_arithmetic(new_condition):
                    expansions = self._expand_condition_with_cp(problem, new_problem, new_condition, solution)
                    for _, cond in expansions:
                        new_action.add_effect(new_fluent, new_value, cond, old_effect.forall)
                else:
                    new_action.add_effect(new_fluent, new_value, new_condition, old_effect.forall)

    def _transform_action_integers(self, problem, new_problem, old_action):
        params = OrderedDict(((p.name, p.type) for p in old_action.parameters))

        has_arithmetic_preconditions = any(requires_arithmetic(p) for p in old_action.preconditions)
        has_arithmetic_effects = any(
            effect.value.node_type in self.ARITHMETIC_OPS
            or effect.is_increase() or effect.is_decrease()
            or requires_arithmetic(effect.condition)
            for effect in old_action.effects
        )

        # No arithmetic: direct transformation
        if not has_arithmetic_preconditions and not has_arithmetic_effects:
            new_action = InstantaneousAction(old_action.name, _parameters=params, _env=problem.environment)
            for old_precondition in old_action.preconditions:
                new_precondition = self._transform_node(problem, new_problem, old_precondition)
                if new_precondition and new_precondition != TRUE():
                    new_action.add_precondition(new_precondition)
            for old_effect in old_action.effects:
                new_fluent = self._transform_node(problem, new_problem, old_effect.fluent)
                new_value = self._transform_node(problem, new_problem, old_effect.value)
                new_cond = self._transform_node(problem, new_problem, old_effect.condition)
                if new_cond is None:
                    new_cond = TRUE()
                if new_fluent and new_value:
                    new_action.add_effect(new_fluent, new_value, new_cond, old_effect.forall)
            return [new_action]

        # Arithmetic: CP-SAT
        self._object_to_index = {}
        self._index_to_object = {}
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        if has_arithmetic_preconditions:
            result_var = add_cp_constraints(problem, And(old_action.preconditions), variables, cp_model_obj,
                                            self._object_to_index)
            cp_model_obj.Add(result_var == 1)
        else:
            for prec in old_action.preconditions:
                if not requires_arithmetic(prec):
                    prec_var = add_cp_constraints(problem, prec, variables, cp_model_obj, self._object_to_index)
                    cp_model_obj.Add(prec_var == 1)

        add_effect_bounds_constraints(problem, variables, cp_model_obj, old_action.effects, self._object_to_index, False)
        solutions = solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            return []

        self._index_to_object = {(t, idx): obj for (t, obj), idx in self._object_to_index.items()}
        return self._create_multiple_actions(old_action, problem, new_problem, params, solutions, variables)

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions by grounding integer parameters into objects."""
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
                new_body = self._transform_node(problem, new_problem, body)
                if new_body is None:
                    skip_axiom = True
                    break
                else:
                    new_axiom.add_body_condition(new_body)
            if skip_axiom:
                continue
            new_problem.add_axiom(new_axiom)
            new_to_old[new_axiom] = axiom

    # ==================== GOALS TRANSFORMATION ====================

    def _add_goal_maintenance_effects(
            self,
            action: Action,
            goal_expr: FNode,
            goal_fluent_exp: FNode,
    ) -> None:
        """
        Adds conditional effects to maintain the truth value of a goal fluent based on the goal expression.
        """
        goal_fluent_exps = get_fluent_exps_in_expression(goal_expr)
        action_modifies_exps = {
            effect.fluent
            for effect in action.effects
            if effect.fluent.is_fluent_exp()
        }

        goal_fluent_strs = {str(f) for f in goal_fluent_exps}
        action_modifies_strs = {str(f) for f in action_modifies_exps}

        if not goal_fluent_strs & action_modifies_strs:
            return

        if not goal_fluent_exps & action_modifies_exps:
            return

        # Replace modified fluents for its next expression
        substituted_goal = substitute_modified_fluents(action, goal_expr).simplify()

        # Add conditional effects with the goal expression as condition to maintain the goal fluent's value
        if substituted_goal is TRUE():
            action.add_effect(goal_fluent_exp, TRUE())
        elif substituted_goal is FALSE():
            action.add_effect(goal_fluent_exp, FALSE())
        else:
            action.add_effect(goal_fluent_exp, TRUE(), substituted_goal)
            action.add_effect(goal_fluent_exp, FALSE(), Not(substituted_goal))

    def _transform_goals(self, problem: Problem, new_problem: Problem) -> None:
        arithmetic_goals = []
        direct_goals = []

        for goal in problem.goals:
            if requires_arithmetic(goal):
                arithmetic_goals.append(goal)
            else:
                direct_goals.append(goal)

        # Non-arithmetic goals
        for goal in direct_goals:
            transformed = self._transform_node(problem, new_problem, goal)
            if transformed and transformed != TRUE():
                new_problem.add_goal(transformed)

        # Create a predicate for each arithmetic goal
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

    def _get_object_from_index(self, user_type, index):
        """
        Get object corresponding to an index for a UserType.

        Uses the internal index-to-object mapping created during CP-SAT constraint building.
        """
        if hasattr(self, '_index_to_object'):
            return self._index_to_object.get((user_type, index))
        return None



    def _transform_fluents(self, problem: Problem, new_problem: Problem):
        """
        Transform integer fluents to object-typed fluents with Number type.

        Each integer fluent becomes an object fluent where objects represent
        numeric values (n0, n1, n2, ...). Non-integer fluents are copied unchanged.
        """
        number_ut = UserType('Number')

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            if fluent.type.is_int_type():
                # Integer fluent -> Object fluent with Number type
                from unified_planning.model import Fluent
                new_fluent = Fluent(fluent.name, number_ut, fluent.signature, new_problem.environment)
                lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
                assert lb is not None and ub is not None

                if default_value is not None:
                    default_obj = self._get_number_object(new_problem, default_value.constant_value())
                    new_problem.add_fluent(new_fluent, default_initial_value=default_obj)
                else:
                    new_problem.add_fluent(new_fluent)

                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        new_problem.set_initial_value(
                            new_problem.fluent(fluent.name)(*f.args),
                            self._get_number_object(new_problem, v.constant_value())
                        )
            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        new_problem.set_initial_value(f, v)

        #self._lt_fluent = setup_lt_predicate(new_problem)

    def _compile(
            self,
            problem: "up.model.AbstractProblem",
            compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_axioms()
        new_problem.initial_values.clear()
        new_problem.clear_quality_metrics()
        self._goal_registry = []

        # Compute the range of integer values needed across the entire problem
        global_lb, global_ub = compute_integer_range(problem)
        ut_number = UserType('Number')
        for v in range(global_lb, global_ub + 1):
            new_problem.add_object(Object(f'n{v}', ut_number))

        # ========== Transform Fluents ==========
        self._transform_fluents(problem, new_problem)

        # ========== Transform Goals ==========
        self._transform_goals(problem, new_problem)

        # ========== Transform Actions ==========
        new_to_old = self._transform_actions(problem, new_problem)

        # ========== Transform Axioms ==========
        self._transform_axioms(problem, new_problem, new_to_old)

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
