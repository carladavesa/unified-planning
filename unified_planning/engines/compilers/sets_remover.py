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
"""This module defines the sets remover class."""
import itertools
from itertools import product
import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import (
    Problem,
    Action,
    ProblemKind, Fluent, Effect, FNode, Parameter,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action, updated_minimize_action_costs,
)
from typing import Dict, Optional, Union, List
from functools import partial
from unified_planning.shortcuts import BoolType, EMPTY_SET, Or, Not, And, TRUE, Iff, Equals, FALSE, IntType


class SetsRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that transforms set fluents into Boolean-indexed fluents.
    Encoding:
    - Original: s(params): set{T}
    - Encoded: s(t, params): bool where t ranges over objects of type T

    The compiler also rewrites set predicates/operations and introduces cardinality helper fluents when needed.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.SETS_REMOVING)
        self._fluent_mapping = {}
        # Maps original set fluent names to their encoded Boolean fluents
        self._cardinality_registry: Dict[str, FNode] = {}

    @property
    def name(self):
        return "srm"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("REAL_ACTION_PARAMETERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("ARRAY_FLUENTS")
        supported_kind.set_fluents_type("SET_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_fluents_type("DERIVED_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_conditions_kind("COUNTING")
        supported_kind.set_conditions_kind("MEMBERING")
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
        return problem_kind <= SetsRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.SETS_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_fluents_type("SET_FLUENTS")
        new_kind.set_fluents_type("ARRAY_FLUENTS")
        return new_kind

    # ==================== FLUENT TRANSFORMATION ====================

    def _get_param_combinations(self, problem: Problem, signature):
        """Get all combinations of parameter values."""
        param_values = [problem.objects(param.type) for param in signature]
        return list(product(*param_values))

    def _add_set_as_boolean_fluent(self, problem: Problem, new_problem: Problem, fluent: Fluent, default_value):
        """Encode a set fluent as a Boolean fluent with an extra element parameter."""
        elements_type = fluent.type.elements_type
        assert elements_type.is_user_type(), "Only UserType types are supported"
        element_param = model.Parameter(
            str(elements_type)[0].lower(),
            elements_type
        )

        new_signature = [element_param] + list(fluent.signature)
        new_fluent = model.Fluent(
            name=fluent.name,
            typename=BoolType(),
            _signature=new_signature,
            environment=fluent.environment
        )

        new_problem.add_fluent(new_fluent, default_initial_value=False)
        self._fluent_mapping[fluent.name] = new_fluent

        param_combinations = self._get_param_combinations(problem, fluent.signature)
        for params in param_combinations:
            initial_value = problem.explicit_initial_values.get(fluent(*params))
            if initial_value:
                elements = initial_value.constant_value()
            elif default_value and default_value != EMPTY_SET():
                elements = default_value.constant_value()
            else:
                continue

            for element in elements:
                new_problem.set_initial_value(
                    new_fluent(element, *params),
                    True
                )

    def _transform_fluents(self, problem: Problem, new_problem: Problem):
        """Transform all fluents from old problem to new problem."""
        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)
            if fluent.type.is_set_type():
                self._add_set_as_boolean_fluent(problem, new_problem, fluent, default_value)
            else:
                self._add_regular_fluent(problem, new_problem, fluent, default_value)

    def _set_boolean_values(self, new_problem, new_fluent, base_params, elements):
        """Set boolean values for each element in the set."""
        for element in elements:
            params = [element] + list(base_params)
            new_problem.set_initial_value(new_fluent(*params), True)

    def _add_set_fluent_as_boolean_array(self, problem, new_problem, fluent, default_value):
        """Helper to encode set fluents as Boolean-indexed fluents."""
        # Create new parameter for set elements
        elements_type = fluent.type.elements_type
        element_param = model.Parameter(str(elements_type)[0].lower(), elements_type)

        # Build new signature with element parameter first
        new_signature = [element_param] + list(fluent.signature)

        # Create and add the new boolean fluent
        new_fluent = model.Fluent(
            name=fluent.name,
            typename=BoolType(),
            _signature=new_signature,
            environment=fluent.environment
        )
        new_problem.add_fluent(new_fluent, default_initial_value=False)

        # Get all parameter combinations
        parameter_combinations = self._get_param_combinations(problem, fluent.signature)

        # Validate default value
        if default_value is not None:
            assert default_value.type.is_set_type(), "Default value must be a set type"

        # Set initial values for each combination
        for combi in parameter_combinations:
            initial_value = problem.explicit_initial_values.get(fluent(*combi))
            if initial_value:
                self._set_boolean_values(new_problem, new_fluent, combi, initial_value.constant_value())
            elif default_value is not None and default_value != EMPTY_SET():
                self._set_boolean_values(new_problem, new_fluent, combi, default_value.constant_value())

    def _add_regular_fluent(self, problem: Problem, new_problem: Problem, fluent: Fluent, default_value):
        """Add non-set fluent unchanged."""
        new_problem.add_fluent(fluent, default_initial_value=default_value)
        self._fluent_mapping[fluent.name] = fluent

        for f, v in problem.explicit_initial_values.items():
            if f.fluent() == fluent:
                new_problem.set_initial_value(fluent(*f.args), v)

    # ==================== EXPRESSION TRANSFORMATION ====================

    def _transform_fluent_exp(self, old_problem, new_problem: Problem, node: FNode) -> FNode:
        """Transform fluent expression."""
        fluent = node.fluent()
        new_args = [self._transform_expression(old_problem, new_problem, arg) for arg in node.args]
        if new_problem.has_fluent(fluent.name):
            return new_problem.fluent(fluent.name)(*new_args)
        else:
            return None

    def _transform_member(self, node: FNode) -> FNode:
        """
        Transform: element in set_fluent(params)
        Into: set_fluent(element, params)
        """
        element = node.args[0]
        set_expr = node.args[1]
        assert set_expr.type.is_set_type(), "Second arg must be a set"
        assert set_expr.is_fluent_exp() or set_expr.is_constant(), "Set expression must be a fluent or a constant"

        if set_expr.is_fluent_exp():
            new_fluent = self._fluent_mapping[set_expr.fluent().name]
            new_args = [element] + list(set_expr.args)
            return new_fluent(*new_args)
        else:
            or_expr = []
            for element_set in list(set_expr.constant_value()):
                or_expr.append(Equals(element, element_set))
            return Or(*or_expr)

    def _transform_subseteq(self, node: FNode) -> FNode:
        """
        Transform: set in set_fluent(params)
        Into: set_fluent(element, params)
        """
        set_expr_1 = node.args[0]
        set_expr_2 = node.args[1]
        assert set_expr_1.type.is_set_type() and set_expr_2.type.is_set_type(), "Args must be sets"
        assert set_expr_1.is_fluent_exp() or set_expr_1.is_constant(), "First set expression must be a fluent or a constant"
        assert set_expr_2.is_fluent_exp() or set_expr_2.is_constant(), "Second set expression must be a fluent or a constant"

        if set_expr_2.is_fluent_exp():
            new_fluent = self._fluent_mapping[set_expr_2.fluent().name]
            if set_expr_1.constant_value():
                set_elements = set_expr_1.constant_value()
                and_expr = []
                for e in set_elements:
                    new_args = [e] + list(set_expr_2.args)
                    and_expr.append(new_fluent(*new_args))
                return And(*and_expr)
        raise NotImplementedError(f"Case of Subseteq not implemented yet")

    def _transform_disjoint(self, new_problem: Problem, node: FNode) -> FNode:
        """
        Transform: set_1 ∩ set_2 == ∅
        Into: And([Not(And(fluent1(obj1, ...), fluent2(obj1, ...))), ...])
        """
        set1 = node.args[0]
        set2 = node.args[1]
        assert set1.type.is_set_type() and set2.type.is_set_type(), "Both arguments must be sets"
        assert set1.is_fluent_exp() or set1.is_constant(), "Set expression must be a fluent or a constant"
        assert set2.is_fluent_exp() or set2.is_constant(), "Set expression must be a fluent or a constant"

        elements_type = set1.type.elements_type if set1.is_fluent_exp() else set2.type.elements_type
        elements = list(new_problem.objects(elements_type))
        and_expr = []

        if set1.is_fluent_exp() and set2.is_fluent_exp():
            fluent1 = self._fluent_mapping[set1.fluent().name]
            fluent2 = self._fluent_mapping[set2.fluent().name]
            for elem in elements:
                and_expr.append(Not(And(fluent1(elem, *set1.args), fluent2(elem, *set2.args))).simplify())
        else:
            fluent, constant = (set1, set2.constant_value()) if set1.is_fluent_exp() else (set2, set1.constant_value())
            new_fluent = self._fluent_mapping[fluent.fluent().name]
            for elem in constant:
                and_expr.append(Not(new_fluent(elem, *fluent.args)).simplify())
        return And(*and_expr)

    def _transform_cardinality(self, old_problem: Problem, new_problem: Problem, node: FNode) -> FNode:
        """
        Transform |set_expr| into an integer helper fluent representing the cardinality.
        """
        set_expr = node.args[0]
        elements_type = set_expr.type.elements_type if set_expr.is_fluent_exp() else set_expr.args[0].type.elements_type
        elements = list(new_problem.objects(elements_type))

        if set_expr.is_fluent_exp():
            # Create an integer helper fluent (same parameters as the source fluent)
            old_fluent = set_expr.fluent()
            card_parameters = [
                a.parameter() if a.is_parameter_exp() else a.variable()
                for a in set_expr.args
                if a.is_parameter_exp() or a.is_variable_exp()
            ]

            fluent_name = f'card_{old_fluent.name}'
            if new_problem.has_fluent(fluent_name):
                return new_problem.fluent(fluent_name)(*set_expr.args)

            # Contains action parameters
            if card_parameters:
                # New cardinality fluent contains a value from 0 to the number of objects of the type it contains in the problem
                new_fluent = Fluent(fluent_name, IntType(0, len(elements)), card_parameters)
                default_initial_value = len(old_problem.fluents_defaults[old_fluent].constant_value())

                self._cardinality_registry[fluent_name] = set_expr
                new_problem.add_fluent(new_fluent, default_initial_value=default_initial_value)

                # Initial values
                parameter_values = self._get_param_combinations(new_problem, card_parameters)
                for p in parameter_values:
                    try:
                        initial_value = len(
                            old_problem.explicit_initial_values[old_fluent(*p)].constant_value())
                        new_problem.set_initial_value(new_fluent(*p), initial_value)
                    except:
                        pass

                return new_fluent(*set_expr.args)

            # Doesn't contain action parameters
            else:
                fluent_name = f'card_{old_fluent.name}_{str(*set_expr.args)}' \
                    if set_expr.args else f'card_{old_fluent.name}'
                if new_problem.has_fluent(fluent_name):
                    return new_problem.fluent(fluent_name)(*set_expr.args)

                new_fluent = Fluent(fluent_name, IntType(0, len(elements)))
                default_initial_value = len(old_problem.fluents_defaults[old_fluent].constant_value())
                new_problem.add_fluent(new_fluent, default_initial_value=default_initial_value)

                # Initialize with arguments parameters (if so)
                try:
                    # If the instantiation has an initial value, add it
                    initial_value = len(
                        old_problem.explicit_initial_values[old_fluent(*set_expr.args)].constant_value())
                    new_problem.set_initial_value(new_fluent(*set_expr.args), initial_value)
                except:
                    pass

                self._cardinality_registry[fluent_name] = set_expr
                return new_fluent()

        elif set_expr.is_set_union():
            # Create an integer helper fluent for cardinality of a union expression.
            set1, set2 = set_expr.args
            parameters1 = [p.parameter() for p in set1.args if p.is_parameter_exp()]
            parameters2 = [p.parameter() for p in set2.args if p.is_parameter_exp()]
            old_fluent1 = set1.fluent()
            old_fluent2 = set2.fluent()

            fluent_name = f'card_{set1.fluent().name}_u_{set2.fluent().name}'

            if new_problem.has_fluent(fluent_name):
                return new_problem.fluent(fluent_name)(*set_expr.args)

            # Both set operands are parameterized in this branch
            if parameters1 and parameters2:
                new_fluent = Fluent(fluent_name, IntType(0, len(elements)), parameters1+parameters2)

                default_value = len(set(old_problem.fluents_defaults[old_fluent1].constant_value() + old_problem.fluents_defaults[old_fluent2].constant_value()))
                default_initial_value = default_value

                self._cardinality_registry[fluent_name] = set_expr
                new_problem.add_fluent(new_fluent, default_initial_value=default_initial_value)

                card_parameters = [
                    a.parameter() if a.is_parameter_exp() else a.variable()
                    for a in set_expr.args
                    if a.is_parameter_exp() or a.is_variable_exp()
                ]
                parameter_values = self._get_param_combinations(new_problem, card_parameters)
                for p in parameter_values:
                    try:
                        initial_value1 = old_problem.explicit_initial_values[old_fluent1(*p)].constant_value()
                        initial_value2 = old_problem.explicit_initial_values[old_fluent1(*p)].constant_value()
                        new_problem.set_initial_value(new_fluent(*p), len(initial_value1 | initial_value2))
                    except:
                        pass
                return new_fluent(*set_expr.args)

            # Only one operand is parameterized.
            elif parameters1 or parameters2:
                raise NotImplementedError(f"Cardinality of {set_expr.node_type} with only one expression "
                                          f"containing action parameters not supported yet")

            else:
                fluent_name = f'card_{set1.fluent().name}_{str(*set1.args)}_u_{str(set2.fluent().name)}_{str(*set2.args)}'

                if new_problem.has_fluent(fluent_name):
                    return new_problem.fluent(fluent_name)(*set_expr.args)

                new_fluent = Fluent(fluent_name, IntType(0, len(elements)))
                default_value = len(set(
                    old_problem.fluents_defaults[old_fluent1].constant_value() |
                    old_problem.fluents_defaults[old_fluent2].constant_value()))
                default_initial_value = default_value
                new_problem.add_fluent(new_fluent, default_initial_value=default_initial_value)

                # Initialize with arguments parameters (if so)
                try:
                    initial_value1 = old_problem.explicit_initial_values[old_fluent1(*set1.args)].constant_value()
                    initial_value2 = old_problem.explicit_initial_values[old_fluent1(*set2.args)].constant_value()
                    new_problem.set_initial_value(new_fluent(*set_expr.args), len(initial_value1 | initial_value2))
                except:
                    pass

                self._cardinality_registry[fluent_name] = set_expr
                return new_fluent()

        raise NotImplementedError(f"Cardinality of {set_expr.node_type} not supported yet.")

    def _transform_add_remove(self, node: FNode) -> FNode:
        """
        Transform: set.add(element) or set.remove(element)
        Into: the fluent(element, params) that should be set to True/False
        """
        element = node.arg(0)
        set_expr = node.arg(1)

        assert set_expr.is_fluent_exp(), "Add/Remove only works on fluent sets"

        new_fluent = self._fluent_mapping[set_expr.fluent().name]
        return new_fluent(element, *set_expr.args)

    def _transform_union(self, new_problem: Problem, node: FNode) -> FNode:
        """Union should not appear alone, only in comparison operations, cardinality, or effects."""
        raise NotImplementedError("Outermost union expressions are only accepted in effect context")

    def _transform_equality(self, old_problem: Problem, new_problem: Problem, node: FNode) -> FNode:
        """
        Transform equality between set expressions.
        Cases:
            1. `set_fluent == {constant_set}`
            2. `set_fluent1 == set_fluent2`
        """
        left = node.arg(0)
        right = node.arg(1)
        em = new_problem.environment.expression_manager

        # Get set fluent and constant set
        set_fluent = None
        constant_set = None
        other_set_fluent = None

        assert (left.is_fluent_exp() and (right.is_fluent_exp() or right.is_parameter_exp() or right.is_constant()) or
                (right.is_fluent_exp() and (left.is_fluent_exp() or left.is_parameter_exp() or left.is_constant()))), \
            f"Expression of the form {node} not supported"

        if left.is_fluent_exp() and left.fluent().type.is_set_type():
            set_fluent = left
            if right.is_set_constant():
                constant_set = right
            elif right.is_fluent_exp() and right.fluent().type.is_set_type():
                other_set_fluent = right
        elif right.is_fluent_exp() and right.fluent().type.is_set_type():
            set_fluent = right
            if left.is_set_constant():
                constant_set = left
            elif left.is_fluent_exp() and left.fluent().type.is_set_type():
                other_set_fluent = left

        # Case 1: set_fluent == {constant_set}
        if set_fluent and constant_set:
            fluent_name = set_fluent.fluent().name
            fluent_args = set_fluent.args
            constant_elements = list(o.object() for o in constant_set.constant_value())
            # Get all possible elements for this set type
            elements_type = set_fluent.fluent().type.elements_type
            all_elements = list(new_problem.objects(elements_type))
            clauses = []
            # All elements in constant must be in the set
            for elem in constant_elements:
                elem_exp = em.ObjectExp(elem)
                member_fluent = new_problem.fluent(fluent_name)(elem_exp, *fluent_args)
                clauses.append(member_fluent)
            # All elements NOT in constant must NOT be in the set
            for obj in all_elements:
                if obj not in constant_elements:
                    elem_exp = em.ObjectExp(obj)
                    member_fluent = new_problem.fluent(fluent_name)(elem_exp, *fluent_args)
                    clauses.append(Not(member_fluent))
            return And(clauses).simplify() if clauses else TRUE()

        # Case 2: set_fluent1 == set_fluent2
        elif set_fluent and other_set_fluent:
            fluent1_name = set_fluent.fluent().name
            fluent1_args = set_fluent.args
            fluent2_name = other_set_fluent.fluent().name
            fluent2_args = other_set_fluent.args

            # Get all possible elements
            elements_type = set_fluent.fluent().type.elements_type
            all_elements = new_problem.objects(elements_type)

            clauses = []

            # For each element: it's in set1 IFF it's in set2
            for obj in all_elements:
                elem_exp = em.ObjectExp(obj)
                member1 = new_problem.fluent(fluent1_name)(elem_exp, *fluent1_args)
                member2 = new_problem.fluent(fluent2_name)(elem_exp, *fluent2_args)

                # member1 <-> member2 (biconditional)
                clauses.append(Iff(member1, member2))

            return And(clauses).simplify() if clauses else TRUE()

        # Not a set equality - transform args recursively
        else:
            new_left = self._transform_expression(old_problem, new_problem, left)
            new_right = self._transform_expression(old_problem, new_problem, right)
            return Equals(new_left, new_right).simplify()

    def _transform_expression(self, old_problem: Problem, new_problem: Problem, node: FNode) -> FNode:
        """
        Transform expressions recursively.
        Delegates to specific handlers based on node type.
        """
        if node.is_fluent_exp():
            return self._transform_fluent_exp(old_problem, new_problem, node)
        elif node.is_parameter_exp() or node.is_variable_exp() or node.is_constant():
            return node
        elif node.is_set_member():
            return self._transform_member(node)
        elif node.is_set_subseteq():
            return self._transform_subseteq(node)
        elif node.is_set_disjoint():
            return self._transform_disjoint(new_problem, node)
        elif node.is_set_cardinality():
            return self._transform_cardinality(old_problem, new_problem, node)
        elif node.is_set_add() or node.is_set_remove():
            return self._transform_add_remove(node)
        elif node.is_set_union():
            return self._transform_union(new_problem, node)
        elif node.is_equals():
            return self._transform_equality(old_problem, new_problem, node)
        else:
            em = new_problem.environment.expression_manager
            new_args = [self._transform_expression(old_problem, new_problem, arg) for arg in node.args]
            if None in new_args:
                return None
            if node.is_exists() or node.is_forall():
                return em.create_node(node.node_type, tuple(new_args), tuple(node.variables())).simplify()
            return em.create_node(node.node_type, tuple(new_args)).simplify()

    # ==================== ACTION TRANSFORMATION ====================

    def _transform_effect(
        self,
        old_problem: Problem,
        new_problem: Problem,
        effect: Effect
    ) -> Union[Effect, List[Effect], None]:
        """Transform one effect, dispatching to the proper set-operation handler."""
        if effect.value.is_set_add() or effect.value.is_set_remove():
            return self._transform_add_remove_effect(old_problem, new_problem, effect)
        elif effect.value.is_set_union():
            return self._transform_union_effect(old_problem, new_problem, effect)
        elif effect.value.is_set_intersect():
            return self._transform_intersect_effect(new_problem, effect)
        elif effect.value.is_set_difference():
            return self._transform_difference_effect(new_problem, effect)
        elif effect.value.is_set_constant():
            return self._transform_set_constant_effect(new_problem, effect)
        else:
            # Non-set value: transform recursively.
            new_fluent = self._transform_expression(old_problem, new_problem, effect.fluent)
            new_value = self._transform_expression(old_problem, new_problem, effect.value)
            new_condition = self._transform_expression(old_problem, new_problem, effect.condition)

            if new_condition is None or new_condition.is_false() or new_fluent is None:
                return None

            return Effect(new_fluent, new_value, new_condition, effect.kind, effect.forall)

    def _transform_add_remove_effect(self, old_problem: Problem, new_problem: Problem, effect: Effect):
        """
        Transform: set_fluent := set_fluent.add(elem)
        Into: set_fluent(elem, ...) := True
        """
        set_expr = effect.value.arg(1)
        # No nested expressions allowed
        assert set_expr.is_fluent_exp() or set_expr.is_constant() or set_expr.is_parameter_exp(), \
            "Nesting of Set methods not supported!"

        assert effect.fluent == set_expr, "Assignment to different set not supported with Add/Remove"
        # aixo no es dificil de fer - potser?

        new_fluent = self._transform_expression(old_problem, new_problem, effect.value)
        new_value = TRUE() if effect.value.is_set_add() else FALSE()  # True for add, False for remove
        new_condition = self._transform_expression(old_problem, new_problem, effect.condition)

        return Effect(new_fluent, new_value, new_condition, effect.kind, effect.forall)

    def _transform_union_effect(self, old_problem: Problem, new_problem: Problem, effect: Effect) -> Union[Effect, List[Effect], None]:
        """
        Transform: result_set := set1 ∪ set2
        Into: for each object o: result_set(o) := set1(o) || set2(o)
        """
        new_effects = []
        set1, set2 = effect.value.args

        # No nested expressions allowed
        assert set1.is_fluent_exp() or set1.is_constant() or set1.is_parameter_exp(), \
            "Nesting of Set methods not supported!"
        assert set2.is_fluent_exp() or set2.is_constant() or set2.is_parameter_exp(), \
            "Nesting of Set methods not supported!"

        elements_type = set1.type.elements_type
        elements = list(new_problem.objects(elements_type))

        new_fluent = self._fluent_mapping[effect.fluent.fluent().name]
        fluent1 = self._fluent_mapping[set1.fluent().name]
        fluent2 = self._fluent_mapping[set2.fluent().name]

        for elem in elements:
            new_condition = Or(
                fluent1(elem, *set1.args),
                fluent2(elem, *set2.args)
            )
            new_fluent_expr = new_fluent(elem, *effect.fluent.args)
            new_effects.append(Effect(new_fluent_expr, TRUE(), new_condition, effect.kind, effect.forall))
        return new_effects

    def _transform_intersect_effect(self, new_problem: Problem, effect: Effect) -> List[Effect]:
        """
        Transform: result_set := set1 ∩ set2
        Into: for each object o: result_set(o) := set1(o) & set2(o)
        """
        new_effects = []
        set1, set2 = effect.value.args

        # No nested expressions allowed
        assert set1.is_fluent_exp() or set1.is_constant() or set1.is_parameter_exp(), \
            "Nesting of Set methods not supported!"
        assert set2.is_fluent_exp() or set2.is_constant() or set2.is_parameter_exp(), \
            "Nesting of Set methods not supported!"

        elements_type = set1.type.elements_type
        elements = list(new_problem.objects(elements_type))
        new_fluent = self._fluent_mapping[effect.fluent.fluent().name]

        if set1.is_fluent_exp() and set2.is_fluent_exp():
            fluent1 = self._fluent_mapping[set1.fluent().name]
            fluent2 = self._fluent_mapping[set2.fluent().name]
            for elem in elements:
                new_condition = And(fluent1(elem, *set1.args), fluent2(elem, *set2.args))
                new_fluent_expr = new_fluent(elem, *effect.fluent.args)
                new_effects.append(Effect(new_fluent_expr, True, new_condition, effect.kind, effect.forall))
        else:
            fluent, constant = (set1, set2.constant_value()) if set1.is_fluent_exp() else (set2, set1.constant_value())
            new_fluent_value = self._fluent_mapping[fluent.fluent().name]
            for elem in constant:
                new_condition = new_fluent_value(elem, *fluent.args)
                new_fluent_expr = new_fluent(elem, *effect.fluent.args)
                new_effects.append(Effect(new_fluent_expr, True, new_condition, effect.kind, effect.forall))
        return new_effects

    def _transform_difference_effect(self, new_problem: Problem, new_action: Action, effect: Effect):
        """
        Transform: result_set := set1 \ set2
        Into: for each object o: result_set(o) := set1(o) & ¬set2(o)
        """
        set1, set2 = effect.value.args

        # No nested expressions allowed
        assert set1.is_fluent_exp() or set1.is_constant() or set1.is_parameter_exp(), \
            "Nesting of Set methods not supported!"
        assert set2.is_fluent_exp() or set2.is_constant() or set2.is_parameter_exp(), \
            "Nesting of Set methods not supported!"

        elements_type = set1.type.elements_type
        elements = list(new_problem.objects(elements_type))
        new_fluent = self._fluent_mapping[effect.fluent.fluent().name]

        if set1.is_fluent_exp() and set2.is_fluent_exp():
            fluent1 = self._fluent_mapping[set1.fluent().name]
            fluent2 = self._fluent_mapping[set2.fluent().name]
            for elem in elements:
                new_condition = And(fluent1(elem, *set1.args), Not(fluent2(elem, *set2.args)))
                new_fluent_expr = new_fluent(elem, *effect.fluent.args)
                new_action.add_effect(new_fluent_expr, True, new_condition, effect.forall)

        elif set1.is_constant():
            # Constant-first case: include each constant element only if absent in the second operand.
            constant, fluent = set1.constant_value(), set2
            new_fluent_value = self._fluent_mapping[fluent.fluent().name]
            for elem in constant:
                new_condition = Not(new_fluent_value(elem, *fluent.args))
                new_fluent_expr = new_fluent(elem, *effect.fluent.args)
                new_action.add_effect(new_fluent_expr, True, new_condition, effect.forall)
        else:
            # Constant-second case: keep elements from the first operand that are not in the constant second set.
            fluent, constant = set1, [o.object() for o in list(set2.constant_value())]
            new_fluent_value = self._fluent_mapping[fluent.fluent().name]
            for elem in [e for e in elements if e not in constant]:
                new_condition = new_fluent_value(elem, *fluent.args)
                new_fluent_expr = new_fluent(elem, *effect.fluent.args)
                new_action.add_effect(new_fluent_expr, True, new_condition, effect.forall)

    def _transform_set_constant_effect(self, new_problem: Problem, effect: Effect):
        """
        Transform: set_fluent := {obj1, obj2, ...}
        Into: set_fluent(obj1) := True, set_fluent(obj2) := True,
              set_fluent(others) := False
        """
        new_effects = []
        elements_type = effect.fluent.type.elements_type
        all_elements = list(new_problem.objects(elements_type))

        new_fluent = self._fluent_mapping[effect.fluent.fluent().name]
        constant_elements = [e.object() for e in effect.value.constant_value()]

        for elem in all_elements:
            value = TRUE() if elem in constant_elements else FALSE()
            fluent_expr = new_fluent(elem, *effect.fluent.args)
            new_effects.append(Effect(fluent_expr, value, TRUE(), effect.kind, effect.forall))
        return new_effects

    def _evaluate_expression(
            self,
            problem: Problem,
            expression: FNode,
            fluent_to_update: Optional[FNode] = None,
            new_value: Optional[FNode] = None,
            effect_type: Optional[str] = None
    ) -> FNode:
        """
        Evaluate expression, used to compute initial values and to determine how effects change card fluents.
        """
        em = problem.environment.expression_manager
        # Base cases
        if expression.is_constant() or expression.is_parameter_exp() or expression.is_object_exp():
            return expression
        # Fluent expression
        if expression.is_fluent_exp():
            # Check if this is the fluent being updated
            if fluent_to_update is not None and fluent_to_update == expression:
                if effect_type == 'increase':
                    return em.Plus(expression, new_value).simplify()
                elif effect_type == 'decrease':
                    return em.Minus(expression, new_value).simplify()
                else:
                    return new_value

            # Regular fluent evaluation
            return problem.initial_value(expression)

        # Recursive case
        new_args = [
            self._evaluate_expression(problem, arg, fluent_to_update, new_value, effect_type)
            for arg in expression.args
        ]
        return em.create_node(expression.node_type, tuple(new_args)).simplify()

    def _find_affected_fluents(self, expression: FNode) -> List[FNode]:
        """Extract all fluent expressions from an expression tree."""
        if expression.is_fluent_exp():
            return [expression]
        fluents = []
        for arg in expression.args:
            fluents.extend(self._find_affected_fluents(arg))
        return fluents

    def _exactly_k_combinations(self, arguments: List[FNode], k: int) -> List[FNode]:
        """Generate all formulas where exactly k arguments are true."""
        n = len(arguments)
        if k > n or k < 0:
            return []
        combinations = []
        for true_indices in itertools.combinations(range(n), k):
            true_set = set(true_indices)
            literals = [arguments[i] if i in true_set else Not(arguments[i]) for i in range(n)]
            combinations.append(And(*literals))
        return combinations

    def _add_card_effect_to_action(
            self, new_problem, action: Action, card: FNode, old_value: FNode, new_effects: List[Effect], equality_conditions: List[FNode] = True
    ):
        """Add conditional effects to maintain cardinality helper fluents."""
        card_expr = self._cardinality_registry[card.fluent().name]
        # Assumes effects are unconditional in this branch.
        if old_value.is_constant():
            if card_expr.is_fluent_exp():
                # effect condition + equality conditions
                n_elements = len(old_value.constant_value())
                action.add_effect(card, n_elements, And(equality_conditions).simplify())
                return

            elif card_expr.is_set_union():
                # Enumerate combinations to compute updated union cardinality.
                for equality in equality_conditions:
                    # Identify the other union operand for this equality branch.
                    other_fluent = card_expr.arg(1) if card_expr.arg(0).arg(0) == equality.arg(1) else card_expr.arg(0)
                    new_other_fluent = new_problem.fluent(other_fluent.fluent().name)
                    all_elements = set(new_problem.objects(card_expr.arg(0).fluent().type.elements_type))
                    constant_set = set(o.object() for o in old_value.constant_value())
                    constant_len = len(constant_set)
                    remaining_elements = all_elements - constant_set
                    new_other_fluents = [new_other_fluent(e, *other_fluent.args) for e in remaining_elements]

                    # Enumerate possible counts from remaining elements.
                    # Elements already in the constant set are always counted.
                    if remaining_elements:
                        for i in range(constant_len, len(remaining_elements) + 1):
                            combinations = self._exactly_k_combinations(new_other_fluents, i)

                            new_condition = And(
                                equality,
                                Or(*combinations)
                            ).simplify()
                            action.add_effect(card, i + constant_len, new_condition)
                    else:
                        new_condition = And(equality).simplify()
                        action.add_effect(card, constant_len, new_condition)

                    return

        elif old_value.is_set_add():
            if len(new_effects) == 1:
                new_effect = new_effects[0]
                if card_expr.is_fluent_exp():
                    new_condition = And(
                        new_effect.condition,
                        equality_conditions,
                        Not(new_effect.fluent)
                    ).simplify()
                    action.add_increase_effect(card, 1, new_condition)
                    return

                elif card_expr.is_set_union():
                    if equality_conditions:
                        # Process each instantiated equality branch independently.
                        for equality in equality_conditions:
                            # Ensure the updated element is also absent from the other union operand.
                            other_fluent = card_expr.arg(1) if card_expr.arg(0).arg(0) == equality.arg(1) else card_expr.arg(0)
                            new_other_fluent = new_problem.fluent(other_fluent.fluent().name)
                            element = old_value.arg(0)

                            new_condition = And(
                                equality,
                                new_effect.condition,
                                Not(new_effect.fluent),
                                Not(new_other_fluent(element, *other_fluent.args))
                            ).simplify()
                            action.add_increase_effect(card, 1, new_condition)
                    return

        elif old_value.is_set_union():
            if card_expr.is_fluent_exp():
                # Find all combinations per possible number of objects
                for equality in equality_conditions:
                    condition_effects = [effect.condition for effect in new_effects]

                    for i in range(0, len(condition_effects) + 1):
                        combinations = self._exactly_k_combinations(condition_effects, i)
                        new_condition = And(
                            equality if not equality.arg(1).is_variable_exp() else True,
                            Or(*combinations)
                        ).simplify()
                        # Replace the variable for the parameter
                        new_args = []
                        for arg in card.args:
                            if arg.is_variable_exp() and arg == equality.arg(1):
                                arg = equality.arg(0)
                            new_args.append(arg)

                        action.add_effect(card.fluent()(*new_args), i, new_condition)
                return

        raise NotImplementedError(f"Not implemented yet")

    def _generate_card_effects(self, old_problem: Problem, new_problem: Problem, action: Action) -> Action:
        """
        Generate effects for cardinality fluents based on action effects.
        For each cardinality fluent tracking expression E:
        - If action effects change fluents in E, add conditional effects to update the count fluent accordingly
        """
        new_action = action.clone()
        new_action.clear_effects()
        # Process each effect
        for old_effect in action.effects:
            new_effects = self._transform_effect(old_problem, new_problem, old_effect)
            if new_effects is None:
                break
            if not isinstance(new_effects, list):
                new_effects = [new_effects]
            for new_effect in new_effects:
                new_action.add_effect(new_effect.fluent, new_effect.value, new_effect.condition, new_effect.forall)
            for card_name, card_expr in self._cardinality_registry.items():
                # Find which fluents in card_expr are affected by action
                affected_fluents = self._find_affected_fluents(card_expr)
                # Check if this effect can affect any tracked fluent
                tracked_fluents = []
                # Check if arguments match (build equality conditions)
                equality_conditions = []
                for tracked_fluent in affected_fluents:
                    # Must be same fluent name
                    if old_effect.fluent.fluent().name != tracked_fluent.fluent().name:
                        continue
                    all_match = True
                    for old_effect_arg, tracked_arg in zip(old_effect.fluent.args, tracked_fluent.args):
                        if old_effect_arg == tracked_arg:
                            # Exact match (same parameter or object)
                            continue
                        elif old_effect_arg.is_parameter_exp():
                            # Effect has parameter, tracked has concrete object
                            # Add condition: param = object
                            equality_conditions.append(Equals(old_effect_arg, tracked_arg))
                        else:
                            # Different concrete values - no match possible
                            all_match = False
                            break
                    if not all_match:
                        continue
                    tracked_fluents.append(tracked_fluent)

                if not tracked_fluents:
                    continue

                # Evaluate what happens to card_expr after the effect
                card_parameters = [
                    a.parameter() if a.is_parameter_exp() else a.variable()
                    for a in card_expr.args
                    if a.is_parameter_exp() or a.is_variable_exp()
                ]
                # Add effect to action
                card_fluent = new_problem.fluent(card_name)(*card_parameters)
                self._add_card_effect_to_action(new_problem, new_action, card_fluent, old_effect.value, new_effects, equality_conditions)
        return new_action

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

        self._fluent_mapping.clear()
        new_to_old: Dict[Action, Action] = {}

        # Transform set fluents
        self._transform_fluents(problem, new_problem)

        # Transform actions (preconditions only, effects later)
        temp_actions = []
        for action in problem.actions:
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()

            # Transform preconditions
            for precondition in action.preconditions:
                new_precondition = self._transform_expression(problem, new_problem, precondition)
                if new_precondition in [FALSE(), None]:
                    break
                new_action.add_precondition(new_precondition)
            temp_actions.append(new_action)

        # Transform goals
        for goal in problem.goals:
            new_goal = self._transform_expression(problem, new_problem, goal)
            if new_goal is None:
                raise UPProblemDefinitionError(f"The Goal expression cannot be evaluated!")
            new_problem.add_goal(new_goal)

        # Add effects that affect fluents within cardinality expressions
        final_actions = []
        for temp_action, old_action in zip(temp_actions, problem.actions):
            final_action = self._generate_card_effects(problem, new_problem, temp_action)
            final_actions.append(final_action)
            new_problem.add_action(final_action)
            new_to_old[final_action] = old_action

        # Transform quality metrics
        for qm in problem.quality_metrics:
            if qm.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(qm, new_to_old, new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(qm)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
