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
"""This module defines the integer parameter actions remover and range variable remover compiler."""
import re
from itertools import product
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model.fnode import FNode
import unified_planning.engines as engines
from collections import OrderedDict
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import Problem, InstantaneousAction, Action, ProblemKind, Fluent, MinimizeActionCosts, \
    RangeVariable, OperatorKind, Effect, Axiom, Expression
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import get_fresh_name, lift_action_instance
from typing import Dict, List, Optional, Tuple, OrderedDict, Union
from functools import partial
from unified_planning.shortcuts import Int, FALSE, TRUE, Exists, Forall


class IntParameterActionsRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes integer parameters from actions by instantiating them.

    Transforms:
    1. Integer action parameters -> grounded instantiated actions for each valid integer value
    2. Range variables -> expanded quantifiers (forall/exists) over instantiated ranges
    3. Array fluents -> indexed fluents with explicit array accesses

    Example:
        action(p: Int[1,3]) with precondition p < 2
    Becomes:
        action_p1 (where p=1), action_p2 (where p=2), action_p3 (where p=3)
        But only action_p1 is created since 1 < 2 is true, while 2 < 2 and 3 < 2 are false.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INT_PARAMETER_ACTIONS_REMOVING)
        self.domains: Dict[str, List[Tuple[int, ...]]] = {}
        self._expression_cache: Dict[Tuple[int, Tuple[int, ...]], FNode] = {}

    @property
    def name(self):
        return "ipar"

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
        supported_kind.set_fluents_type("SET_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_conditions_kind("COUNTING")
        supported_kind.set_conditions_kind("MEMBERING")
        supported_kind.set_conditions_kind("RANGE_VARIABLES")
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
        return problem_kind <= IntParameterActionsRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INT_PARAMETER_ACTIONS_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        new_kind.unset_conditions_kind("RANGE_VARIABLES")
        return new_kind

    # ======================== UTILITY METHODS ========================

    def _save_array_domain(self, fluent: Fluent):
        """
        Extract and cache valid array positions for a fluent.
        Stores all combinations of valid indices in self.domains, accounting for undefined positions if specified.
        """
        current_type = fluent.type
        domain_ranges = []
        dimensions = []

        while current_type.is_array_type():
            dimensions.append(current_type.size)
            domain_ranges.append(range(current_type.size))
            current_type = current_type.elements_type

        all_positions = list(product(*domain_ranges))

        if fluent.undefined_positions is not None:
            valid_positions = [
                pos for pos in all_positions
                if pos not in fluent.undefined_positions
            ]
        else:
            valid_positions = all_positions

        self.domains[fluent.name] = valid_positions

    def _extract_array_indices(
            self,
            fluent: Fluent,
            int_params: Optional[Dict[str, int]] = None,
            instantiations: Optional[Tuple[int, ...]] = None
    ) -> Union[List[int], None]:
        """
        Extract and evaluate array indices from fluent name.
        Parses numeric and parameter-based indices, substituting parameter values from the current instantiation.
        Returns None if out of bounds or undefined.
        """
        pattern = r'\[(.*?)\]'
        indices = []
        fluent_name = fluent.name
        for access_expr in re.findall(pattern, fluent_name):
            if access_expr.isdigit():
                # Constant index
                index_value = int(access_expr)
            else:
                # Expression with parameters - substitute and evaluate
                evaluated_expr = access_expr
                for param_name, param_idx in int_params.items():
                    evaluated_expr = re.sub(
                        r'\b' + re.escape(param_name) + r'\b',
                        str(instantiations[param_idx]),
                        evaluated_expr
                    )

                # Evaluate arithmetic expression
                try:
                    index_value = eval(evaluated_expr)
                except:
                    return None

            indices.append(index_value)

        # Check if access is within valid domain
        fluent_base_name = fluent_name.split('[')[0]
        valid_accesses = self.domains.get(fluent_base_name, [])
        if tuple(indices) not in valid_accesses:
            return None
        return indices

    # ==================== QUICK VALIDATION ====================

    def _quick_check_expression(
            self, problem: Problem, node: FNode, int_params: Dict[str, int], instantiation: Tuple[int, ...]
    ) -> Optional[bool]:
        """
        Quick check if expression is definitely false without full transformation.
        Used to prune invalid instantiations early before full action generation.
        Returns: True/False if definite, None if unknown.
        """
        # Check parameter equality
        if node.is_equals():
            left, right = node.arg(0), node.arg(1)

            # parameter == constant
            if left.is_parameter_exp() and right.is_int_constant():
                param_name = left.parameter().name
                if param_name in int_params:
                    return instantiation[int_params[param_name]] == right.constant_value()

            # constant == parameter
            if right.is_parameter_exp() and left.is_int_constant():
                param_name = right.parameter().name
                if param_name in int_params:
                    return instantiation[int_params[param_name]] == left.constant_value()

            # Check array access out of bounds
            if left.is_fluent_exp():
                fluent_base = left.fluent().name.split('[')[0]
                old_fluent = problem.fluent(fluent_base)
                if old_fluent.type.is_array_type():
                    indices = self._extract_array_indices(left.fluent(), int_params, instantiation)
                    if indices is None:
                        return None  # Out of bounds

        # Check AND
        elif node.is_and():
            for arg in node.args:
                result = self._quick_check_expression(problem, arg, int_params, instantiation)
                if result == None:
                    return None

        # Check OR
        elif node.is_or():
            all_false = True
            for arg in node.args:
                result = self._quick_check_expression(problem, arg, int_params, instantiation)
                if result == True:
                    return True
                if result != False:
                    all_false = False
            if all_false:
                return False

        return None  # Unknown

    def _is_instantiation_valid(
            self,
            problem: Problem,
            action: Action,
            int_param_map: Dict[str, int],
            instantiation: Tuple[int, ...]
    ) -> bool:
        """
        Check if an instantiation is valid before creating the full action.
        Quick validation on preconditions to prune impossible instantiations early.
        Returns False if definitely impossible (e.g., precondition always false).
        """
        # Quick check preconditions
        for precond in action.preconditions:
            result = self._quick_check_expression(problem, precond, int_param_map, instantiation)
            if result == False:
                return False  # Definitely false precondition

        return True

    # ==================== RANGE VARIABLE TRANSFORMATION ====================

    def _extract_variables(self, variables: List) -> Tuple[Tuple, Dict[str, Tuple[int, int]]]:
        """Separate regular variables from range variables."""
        regular_vars = []
        range_vars = {}
        for var in variables:
            if isinstance(var, RangeVariable):
                range_vars[var.name] = (var.initial, var.last)
            else:
                regular_vars.append(var)
        return tuple(regular_vars), range_vars

    def _update_range_vars(
            self, range_vars: Dict[str, Tuple[int, int]], int_params: Dict[str, int], instantiations: Tuple[int, ...]
    ) -> Dict[str, Tuple[int, int]]:
        """
        Evaluate range expressions with current parameter values.
        Substitutes integer parameter instantiations into range expressions so that quantified variables have concrete bounds.
        """
        updated = {}
        for var_name, (initial, last) in range_vars.items():
            initial_str = str(initial)
            last_str = str(last)
            for param_name, param_idx in int_params.items():
                param_value = str(instantiations[param_idx])
                initial_str = initial_str.replace(param_name, param_value)
                last_str = last_str.replace(param_name, param_value)

            updated[var_name] = (eval(initial_str), eval(last_str))
        return updated

    def _get_range_instantiations(self, ranges: Dict[str, Tuple[int, int]]) -> List[Tuple[int, ...]]:
        """Generate all combinations of values for range variables."""
        if not ranges:
            return [()]
        range_iterables = [
            range(start, end + 1)
            for start, end in ranges.values()
        ]
        return list(product(*range_iterables))

    # ==================== EXPRESSION TRANSFORMATION ====================

    def _transform_quantifier(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Dict[str, int],
            instantiations: Tuple[int, ...],
    ) -> FNode:
        """
        Transform forall/exists by expanding range variables.
        Replaces range variables with concrete instantiations, then expands the quantifier into a
        conjunction/disjunction over valid value ranges.
        """
        regular_vars, range_vars = self._extract_variables(node.variables())

        if not range_vars:
            # No range variables: keep quantifier
            new_args = [
                self._transform_expression(old_problem, new_problem, arg, int_params, instantiations)
                for arg in node.args
            ]
            new_args = self._handle_none_args(node.node_type, new_args)
            if new_args is None:
                return None
            em = old_problem.environment.expression_manager
            return em.create_node(node.node_type, tuple(new_args), regular_vars).simplify()

        # Update ranges with current parameter values
        updated_ranges = self._update_range_vars(range_vars, int_params, instantiations)

        # Expand range variables
        expanded_int_params = int_params.copy()
        for var_name in range_vars.keys():
            expanded_int_params[var_name] = len(expanded_int_params)

        # Get all instantiations for range variables
        range_instantiations = self._get_range_instantiations(updated_ranges)

        # Expand quantifier body for each instantiation
        expanded_args = []
        for range_inst in range_instantiations:
            full_inst = instantiations + range_inst
            for arg in node.args:
                transformed = self._transform_expression(
                    old_problem, new_problem, arg, expanded_int_params, full_inst
                )
                if transformed is not None:
                    expanded_args.append(transformed)
        if not expanded_args:
            return None

        # Combine with appropriate operator
        em = new_problem.environment.expression_manager
        new_op = OperatorKind.AND if node.is_forall() else OperatorKind.OR
        new_node = em.create_node(new_op, tuple(expanded_args)).simplify()
        if regular_vars:
            if node.is_exists():
                return Exists(new_node, *regular_vars)
            elif node.is_forall():
                return Forall(new_node, *regular_vars)
            else:
                raise UPProblemDefinitionError(f"Error handling quantifiers!")
        return new_node

    def _transform_fluent_exp(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Dict[str, int],
            instantiations: Tuple[int, ...]
    ) -> Union[FNode, None]:
        """
        Transform fluent expression by substituting integer parameters and array indices.
        For array fluents, extracts concrete indices; for regular fluents, substitutes parameter values into arguments.
        """
        # Transform arguments first
        new_args = []
        for arg in node.args:
            transformed = self._transform_expression(old_problem, new_problem, arg, int_params, instantiations)
            if transformed is None:
                return None
            new_args.append(transformed)

        fluent_base_name = node.fluent().name.split('[')[0]
        old_fluent = old_problem.fluent(fluent_base_name)
        # Array fluent: extract indices from name
        if old_fluent.type.is_array_type():
            index_params = self._extract_array_indices(node.fluent(), int_params, instantiations)
            if index_params is None:
                return None
            idx = "".join(f"[{i}]" for i in index_params)
            return Fluent(
                f"{fluent_base_name}{idx}",
                node.fluent().type,
                _signature=node.fluent().signature,
                environment=node.fluent().environment,
                undefined_positions=node.fluent().undefined_positions
            )(*new_args)

        # Regular fluent
        return node.fluent()(*new_args)

    def _handle_none_args(self, node_type: OperatorKind, args: List) -> Union[List[FNode], None]:
        """Handle undefined (None) values in arguments based on operator semantics."""
        if None not in args:
            return args
        if node_type in {OperatorKind.OR, OperatorKind.COUNT, OperatorKind.EXISTS}:
            filtered = [arg for arg in args if arg is not None]
            return filtered if filtered else None
        elif node_type == OperatorKind.IMPLIES:
            if args[1] is None and args[0] is not None:
                return [args[0], FALSE()]
            return [TRUE(), args[1]] if args[1] is not None else None
        else:
            return None

    def _transform_generic(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Dict[str, int],
            instantiations: Tuple[int, ...]
    ) -> Union[FNode, None]:
        """Generic recursive transformation."""
        em = old_problem.environment.expression_manager

        new_args = [
            self._transform_expression(old_problem, new_problem, arg, int_params, instantiations)
            for arg in node.args
        ]
        new_args = self._handle_none_args(node.node_type, new_args)
        if new_args is None or new_args == []:
            return None
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _transform_array_access(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Dict[str, int],
            instantiations: Tuple[int, ...]
    ) -> Union[FNode, None]:
        """
        Transform ARRAY_READ or ARRAY_WRITE into an indexed fluent expression.
        Supports N-dimensional arrays by unwinding the chain of ARRAY_READ/WRITE nodes.
        (read (board ?a) ?i ?j) with i=1,j=2 → board[1][2](?a)
        (write ((board ?a) ?i ?j) val) with i=1,j=2 → board[1][2](?a)  [effect target]
        """
        # Unwind the chain of ARRAY_READ/ARRAY_WRITE nodes to collect all indices
        indices = []
        current = node
        while current.is_array_read() or current.is_array_write():
            idx = self._transform_expression(
                old_problem, new_problem, current.arg(1), int_params, instantiations
            )
            if idx is None or not idx.is_int_constant():
                return None
            indices.insert(0, idx.constant_value())
            current = current.arg(0)

        if not current.is_fluent_exp():
            return None

        base_fluent = current.fluent()
        base_name = base_fluent.name.split('[')[0]

        # Check the full index tuple is within the valid domain
        if tuple(indices) not in self.domains.get(base_name, []):
            return None

        # Transform the fluent's own arguments (e.g., the object parameters)
        new_fluent_args = [
            self._transform_expression(old_problem, new_problem, a, int_params, instantiations)
            for a in current.args
        ]
        if any(a is None for a in new_fluent_args):
            return None

        # Determine element type after all index levels
        elem_type = base_fluent.type
        for _ in indices:
            elem_type = elem_type.elements_type

        indexed_name = base_name + "".join(f"[{k}]" for k in indices)
        indexed_fluent = Fluent(
            indexed_name,
            elem_type,
            _signature=base_fluent.signature,
            environment=base_fluent.environment,
            undefined_positions=base_fluent.undefined_positions,
        )
        return indexed_fluent(*new_fluent_args)

    def _transform_expression(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Optional[Dict[str, int]] = None,
            instantiations: Optional[Tuple[int, ...]] = None,
    ) -> Union[FNode, None]:
        """
        Transform expression by substituting integer parameters and expanding quantifiers, replacing:
        - Integer parameters with their instantiated constant values
        - Array accesses with indexed fluent names
        - Range variables with concrete bounds
        """
        if int_params is None:
            int_params = {}
        if instantiations is None:
            instantiations = ()
        cache_key = (id(node), instantiations)
        if cache_key in self._expression_cache:
            if self._expression_cache[cache_key] is None:
                return None
            return self._expression_cache[cache_key]

        # Base cases
        if node.is_constant() or node.is_timing_exp():
            return node

        # If experssion wraps a RangeVariable, substitute with integer
        if node.is_variable_exp():
            v = node.variable()
            if isinstance(v, RangeVariable) and v.name in int_params:
                param_index = int_params[v.name]
                result = Int(instantiations[param_index])
                self._expression_cache[cache_key] = result
                return result
            return node

        if node.is_parameter_exp():
            param_name = node.parameter().name
            if param_name in int_params:
                param_index = int_params[param_name]
                result = Int(instantiations[param_index])
                self._expression_cache[cache_key] = result
                if result is None:
                    return None
                return result
            return node

        if node.is_range_variable_exp():
            var_name = node.range_variable().name
            if var_name in int_params:
                param_index = int_params[var_name]
                result = Int(instantiations[param_index])
                self._expression_cache[cache_key] = result
                return result
            return node

        if node.is_fluent_exp():
            result = self._transform_fluent_exp(old_problem, new_problem, node, int_params, instantiations)
            self._expression_cache[cache_key] = result
            if result is None:
                return None
            return result

        if node.is_forall() or node.is_exists():
            result = self._transform_quantifier(old_problem, new_problem, node, int_params, instantiations)
            self._expression_cache[cache_key] = result
            if result is None:
                return None
            return result

        if node.is_array_read() or node.is_array_write():
            result = self._transform_array_access(old_problem, new_problem, node, int_params, instantiations)
            self._expression_cache[cache_key] = result
            if result is None:
                return None
            return result

        result = self._transform_generic(old_problem, new_problem, node, int_params, instantiations)
        self._expression_cache[cache_key] = result
        if result is None:
            return None
        return result

    # ==================== ACTION TRANSFORMATION ====================

    def _add_effect_to_action(
            self, action: Action, effect_type: str, fluent: FNode, value: FNode, condition: FNode, forall: Tuple
    ):
        """Add effect to action, dispatching by effect type (assign/increase/decrease)."""
        if effect_type == 'increase':
            action.add_increase_effect(fluent, value, condition, forall)
        elif effect_type == 'decrease':
            action.add_decrease_effect(fluent, value, condition, forall)
        else:
            action.add_effect(fluent, value, condition, forall)

    def _add_single_effect(
            self,
            action: Action,
            effect_type: str,
            fluent: FNode,
            value: FNode,
            condition: FNode,
            original_condition: FNode,
            forall: Tuple
    ) -> bool:
        """
        Add single effect to action with bounds checking.
        Prunes invalid effects (e.g., assignments violating integer bounds).
        Returns False if action should be pruned.
        """
        # Handle unconditional effects
        if original_condition == TRUE():
            if fluent is None or value is None:
                return False
            if fluent.type.is_int_type() and value.is_constant():
                if not fluent.type.lower_bound <= value.constant_value() <= fluent.type.upper_bound:
                    return False
            self._add_effect_to_action( action, effect_type, fluent, value, condition, forall)
        # Handle conditional effects
        else:
            if condition not in [None, FALSE()] and fluent is not None and value is not None:
                if (fluent.type.is_int_type() and value.is_constant() and
                        not fluent.type.lower_bound <= value.constant_value() <= fluent.type.upper_bound):
                    return True
                self._add_effect_to_action(action, effect_type, fluent, value, condition, forall)
        return True

    def _add_instantiated_effect(
            self,
            problem: Problem,
            new_problem: Problem,
            effect: Effect,
            new_action: Action,
            int_param_map: Dict[str, int],
            instantiation: Tuple[int, ...]
    ) -> bool:
        """
        Add single effect to action, handling forall with range variables.
        Expands forall effects over range variables into individual effects for each instantiation.
        """
        if effect.is_increase():
            effect_type = 'increase'
        elif effect.is_decrease():
            effect_type = 'decrease'
        else:
            effect_type = 'none'

        regular_forall, range_vars = self._extract_variables(effect.forall)
        if not range_vars:
            new_fluent = self._transform_expression(problem, new_problem, effect.fluent, int_param_map, instantiation)
            new_value = self._transform_expression(problem, new_problem, effect.value, int_param_map, instantiation)
            new_condition = self._transform_expression(problem, new_problem, effect.condition, int_param_map, instantiation)

            return self._add_single_effect(
                new_action, effect_type, new_fluent, new_value, new_condition, effect.condition, regular_forall
            )

        # Evaluate range variables with current instantiation
        updated_ranges = self._update_range_vars(range_vars, int_param_map, instantiation)

        # Expand forall with range variables
        expanded_int_params = int_param_map.copy()
        for var_name in range_vars.keys():
            expanded_int_params[var_name] = len(expanded_int_params)

        range_insts = self._get_range_instantiations(updated_ranges)
        for range_inst in range_insts:
            full_inst = instantiation + range_inst
            new_fluent = self._transform_expression(problem, new_problem, effect.fluent, expanded_int_params, full_inst)
            new_value = self._transform_expression(problem, new_problem, effect.value, expanded_int_params, full_inst)
            new_condition = self._transform_expression(problem, new_problem, effect.condition, expanded_int_params, full_inst)
            success = self._add_single_effect(
                new_action, effect_type, new_fluent, new_value, new_condition, effect.condition, regular_forall
            )
            if not success:
                return False
        return True

    def _add_instantiated_effects(
            self,
            problem: Problem,
            new_problem: Problem,
            old_action: Action,
            new_action: Action,
            int_param_map: Dict[str, int],
            instantiation: Tuple[int, ...]
    ) -> bool:
        """
        Add all effects to instantiated action.
        Returns True if any valid effects were added, False if action should be pruned.
        """
        for effect in old_action.effects:
            success = self._add_instantiated_effect(
                problem, new_problem, effect, new_action, int_param_map, instantiation
            )
            if not success:
                return False
        return len(new_action.effects) > 0

    def _create_instantiated_action(
            self,
            problem: Problem,
            new_problem: Problem,
            action: Action,
            regular_params: OrderedDict,
            int_param_map: Dict[str, int],
            instantiation: Tuple[int, ...]
    ) -> Union[Action, None]:
        """
        Create a single instantiated action for a specific integer parameter assignment.
        Transforms preconditions and effects, pruning the action if any become false/invalid.
        """
        # Generate unique name
        action_name = get_fresh_name(new_problem, action.name, list(map(str, instantiation)))
        # Create action with only regular (noninteger) parameters
        assert isinstance(action, InstantaneousAction), "Only InstantaneousActions are supported"
        new_action = InstantaneousAction(action_name, regular_params, action.environment)
        # Transform preconditions
        for precondition in action.preconditions:
            new_precondition = self._transform_expression(
                problem, new_problem, precondition, int_param_map, instantiation
            )
            if new_precondition in [FALSE(), None]:
                return None
            new_action.add_precondition(new_precondition)

        # Transform effects
        has_valid_effects = self._add_instantiated_effects(
            problem, new_problem, action, new_action, int_param_map, instantiation
        )
        if not has_valid_effects:
            return None
        return new_action

    def _instantiate_action(
            self, problem: Problem, new_problem: Problem, action: Action
    ) -> List[Tuple[Action, Tuple[int, ...]]]:
        """
        Create all valid instantiations of an action for integer parameters.
        Generates Cartesian product of integer parameter ranges, validates each, and returns list of pairs for valid instances.
        """
        # Separate regular and integer parameters
        regular_params = OrderedDict()
        int_param_map = {}
        int_param_ranges = []

        for param in action.parameters:
            if param.type.is_user_type():
                regular_params[param.name] = param.type
            elif param.type.is_int_type():
                int_param_map[param.name] = len(int_param_map)
                int_param_ranges.append((param.type.lower_bound, param.type.upper_bound))
            else:
                raise UPProblemDefinitionError(f"Parameter type {param.type} not supported")

        # Generate all instantiations
        instantiations = self._get_range_instantiations(
            {f"p{i}": r for i, r in enumerate(int_param_ranges)}
        ) if int_param_ranges else [()]
        result = []
        for inst in instantiations:
            new_action = self._create_instantiated_action(
                problem, new_problem, action, regular_params, int_param_map, inst
            )
            if new_action is not None:
                result.append((new_action, inst))
        return result

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Tuple[Action, Tuple[int, ...]]]:
        """Transform all actions by grounding integer parameters."""
        new_to_old = {}
        for action in problem.actions:
            instantiated_actions = self._instantiate_action(problem, new_problem, action)
            for new_action, instantiation in instantiated_actions:
                new_problem.add_action(new_action)
                new_to_old[new_action] = (action, instantiation)
        return new_to_old

    # ==================== AXIOMS TRANSFORMATION ====================

    def _transform_axioms(self, problem: Problem, new_problem: Problem, new_to_old: Dict):
        """
        Transform axioms by replacing integer parameter references.
        Raises an error if axioms contain integer parameters (not yet supported).
        """
        for axiom in problem.axioms:
            for param in axiom.parameters:
                if param.type.is_int_type():
                    raise NotImplementedError(
                        "Integer parameters in axioms are not supported!"
                    )
            params = OrderedDict((p.name, p.type) for p in axiom.parameters)
            new_axiom_name = get_fresh_name(new_problem, axiom.name)
            new_axiom = Axiom(new_axiom_name, params, axiom.environment)

            skip_axiom = False
            new_axiom.set_head(axiom.head.fluent)
            for body in axiom.body:
                new_body = self._transform_expression(problem, new_problem, body)
                if new_body is None:
                    skip_axiom = True
                    break
                else:
                    new_axiom.add_body_condition(new_body)
            if skip_axiom:
                continue
            new_problem.add_axiom(new_axiom)
            new_to_old[new_axiom] = axiom

    # ==================== QUALITY METRICS TRANSFORMATION ====================

    def _transform_quality_metrics(
            self, problem: Problem, new_problem: Problem, new_to_old: Dict[Action, Tuple[Action, Tuple[int, ...]]]
    ):
        """Transform quality metrics, handling action costs with integer parameter substitution."""
        for qm in problem.quality_metrics:
            if qm.is_minimize_sequential_plan_length() or qm.is_minimize_makespan():
                new_problem.add_quality_metric(qm)
            elif qm.is_minimize_action_costs():
                assert isinstance(qm, MinimizeActionCosts)
                new_costs = self._transform_action_costs(qm, new_to_old)
                new_problem.add_quality_metric(
                    MinimizeActionCosts(new_costs, environment=new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(qm)

    def _transform_action_costs(
            self,
            qm: MinimizeActionCosts,
            new_to_old: Dict[Action, Tuple[Action, Tuple[int, ...]]]
    ) -> Dict[Action, Expression]:
        """Transform action costs by substituting integer parameter values."""
        new_costs = {}
        for new_action, (old_action, instantiation) in new_to_old.items():
            if old_action is None:
                continue
            old_cost = qm.get_action_cost(old_action)
            # If cost is a parameter, substitute its instantiated value
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

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation."""
        assert isinstance(problem, Problem)
        self._expression_cache.clear()

        # Create new problem
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_actions()
        new_problem.clear_axioms()
        new_problem.clear_quality_metrics()

        # Transform components
        for fluent in new_problem.fluents:
            if fluent.type.is_array_type():
                self._save_array_domain(fluent)

        new_to_old = self._transform_actions(problem, new_problem)
        self._transform_quality_metrics(problem, new_problem, new_to_old)
        self._transform_axioms(problem, new_problem, new_to_old)

        return CompilerResult(
            new_problem, partial(lift_action_instance, map=new_to_old), self.name,
        )
