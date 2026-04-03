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
"""This module defines the quantifiers remover class."""


from itertools import product
import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import Problem, Action, ProblemKind, InstantaneousAction, FNode, Object, Parameter, Fluent, \
    Type, OperatorKind, Effect, Axiom
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action, updated_minimize_action_costs, get_fresh_name
from typing import Dict, List, Optional, Tuple, OrderedDict, Union
from functools import partial
from unified_planning.shortcuts import FALSE, UserType, And, TRUE
import re

class ArraysRemover(engines.engine.Engine, CompilerMixin):
    """
    Removes arrays  by instantiating them.
    Also transforms array fluents into indexed fluents.

    Transforms:

        1. Array fluents -> indexed fluents with Index type (UserType)

    """

    def __init__(self, mode: str = 'strict'):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.ARRAYS_REMOVING)
        self.mode = mode
        self.domains: Dict[str, List[Tuple[int, ...]]] = {}
        self._index_objects_cache: Dict[int, Object] = {}
        self._expression_cache: Dict[int, FNode] = {}

    @property
    def name(self):
        return "arm"

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
        return problem_kind <= ArraysRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.ARRAYS_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_fluents_type("ARRAY_FLUENTS")
        return new_kind

    # ==================== UTILITY METHODS ====================

    def _get_index_object(self, problem: Problem, n: int) -> Object:
        """Get or create index object i{n} with caching."""
        if n in self._index_objects_cache:
            return self._index_objects_cache[n]

        obj_name = f'i{n}'
        if not problem.has_object(obj_name):
            obj = Object(obj_name, UserType('Index'))
            problem.add_object(obj)
            self._index_objects_cache[n] = obj
            return obj

        obj = problem.object(obj_name)
        self._index_objects_cache[n] = obj
        return obj

    def _extract_array_indices(
            self,
            new_problem: Problem,
            fluent_name: str
    ) -> Union[List[Object], None]:
        """Extract and evaluate array indices from fluent name."""
        # Parse indices
        indices = tuple(int(i) for i in re.findall(r'\[([0-9]+)\]', fluent_name))

        # Check validity
        fluent_base_name = fluent_name.split('[')[0]
        valid_accesses = self.domains.get(fluent_base_name, [])

        if indices not in valid_accesses:
            return None

        # Convert to objects (with cache)
        return [self._get_index_object(new_problem, i) for i in indices]

    # ==================== EXPRESSION TRANSFORMATION ====================

    def _transform_fluent_exp(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode
    ) -> Union[FNode, None]:
        """Transform fluent expression, handling arrays."""
        # Check cache
        cache_key = id(node)
        if cache_key in self._expression_cache:
            return self._expression_cache[cache_key]

        # Transform arguments first
        new_args = []
        for arg in node.args:
            transformed = self._transform_expression(old_problem, new_problem, arg)
            if transformed is None:
                return None
            new_args.append(transformed)

        fluent_base_name = node.fluent().name.split('[')[0]
        old_fluent = old_problem.fluent(fluent_base_name)

        # Array fluent: extract indices from name
        if old_fluent.type.is_array_type():
            new_fluent = new_problem.fluent(fluent_base_name)
            index_params = self._extract_array_indices(new_problem, node.fluent().name)

            if index_params is None:
                return None

            result = new_fluent(*(index_params + new_args))
            self._expression_cache[cache_key] = result
            return result

        # Regular fluent
        result = node.fluent()(*new_args)
        self._expression_cache[cache_key] = result
        return result

    def _transform_array_comparison(
            self,
            new_problem: Problem,
            node: FNode,
    ) -> FNode:
        """
        Transform comparison with array fluent.
        Example: board == [[1,2],[3,4]] → multiple comparisons
        """
        # Identify fluent and value
        fluent_arg, value_arg = (
            (node.arg(0), node.arg(1))
            if node.arg(1).is_constant()
            else (node.arg(1), node.arg(0))
        )

        fluent_base_name = fluent_arg.fluent().name.split('[')[0]
        new_fluent = new_problem.fluent(fluent_base_name)

        # Generate comparison for each position
        em = new_problem.environment.expression_manager
        comparisons = []

        for pos, val in self._get_new_fluent_value(new_problem, new_fluent, fluent_arg, value_arg).items():
            comparison = em.create_node(node.node_type, (pos, val))
            comparisons.append(comparison.simplify())

        return And(comparisons).simplify()

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
            return[TRUE(), args[1]] if args[1] is not None else None
        else:
            return None

    def _transform_quantifier(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode
    ) -> FNode:
        """Transform quantifier expression."""
        new_args = [
            self._transform_expression(old_problem, new_problem, arg)
            for arg in node.args
        ]

        new_args = self._handle_none_args(node.node_type, new_args)

        if new_args is None:
            return None

        em = old_problem.environment.expression_manager
        return em.create_node(node.node_type, tuple(new_args), tuple(node.variables())).simplify()

    def _transform_array_access(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
    ) -> Union[FNode, None]:
        """
        Transform ARRAY_READ or ARRAY_WRITE into an indexed fluent call.
        Supports N-dimensional arrays by unwinding the chain of ARRAY_READ/WRITE nodes.
        Expects all indices to be constant integers (i.e., after int-param removal).
        (read (board ?a) Int(1) Int(2)) → board(i1, i2, ?a)
        """
        # Unwind the chain of ARRAY_READ/ARRAY_WRITE nodes to collect all indices
        indices = []
        current = node
        while current.is_array_read() or current.is_array_write():
            idx = self._transform_expression(old_problem, new_problem, current.arg(1))
            if idx is None or not idx.is_int_constant():
                return None
            indices.insert(0, idx.constant_value())
            current = current.arg(0)

        if not current.is_fluent_exp():
            return None

        base_name = current.fluent().name.split('[')[0]
        indexed_name = base_name + "".join(f"[{k}]" for k in indices)

        index_params = self._extract_array_indices(new_problem, indexed_name)
        if index_params is None:
            return None

        new_fluent = new_problem.fluent(base_name)
        new_args = [
            self._transform_expression(old_problem, new_problem, a)
            for a in current.args
        ]
        if any(a is None for a in new_args):
            return None
        return new_fluent(*(index_params + new_args))

    def _transform_expression(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode
    ) -> Union[FNode, None]:
        """Transform expression by substituting array accesses."""
        if node.is_constant() or node.is_variable_exp() or node.is_timing_exp() or node.is_parameter_exp():
            return node
        if node.is_fluent_exp():
            return self._transform_fluent_exp(old_problem, new_problem, node)
        if node.is_forall() or node.is_exists():
            return self._transform_quantifier(old_problem, new_problem, node)
        # ARRAY_READ/ARRAY_WRITE: resolve to indexed fluent — must come before the
        # array-comparison check below, which would misinterpret these nodes.
        if node.is_array_read() or node.is_array_write():
            return self._transform_array_access(old_problem, new_problem, node)
        # Special case: array fluent comparisons
        if node.args and node.arg(0).type.is_array_type():
            return self._transform_array_comparison(new_problem, node)

        # Generic recursive transformation
        em = old_problem.environment.expression_manager
        new_args = [
            self._transform_expression(old_problem, new_problem, arg)
            for arg in node.args
        ]
        new_args = self._handle_none_args(node.node_type, new_args)
        if new_args is None or not new_args:
            return None
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    # ==================== ACTION TRANSFORMATION ====================

    def _add_effect_to_action(
            self, action: Action, effect_type: str, fluent: FNode, value: FNode, condition: FNode, forall: Tuple
    ):
        """Add effect to action based on type."""
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
        """Add single effect to action. Returns False if action should be pruned."""
        # Handle unconditional effects
        if original_condition == TRUE():
            if fluent is None or value is None:
                return False  # Invalid unconditional effect
            self._add_effect_to_action(action, effect_type, fluent, value, condition, forall)
        # Handle conditional effects
        else:
            if condition not in (None, FALSE()) and fluent is not None and value is not None:
                self._add_effect_to_action(action, effect_type, fluent, value, condition, forall)
        return True

    def _add_instantiated_effect(
            self,
            problem: Problem,
            new_problem: Problem,
            effect: Effect,
            new_action: Action,
    ) -> bool:
        """Add single effect to action. Returns False if action should be pruned."""
        # Determine effect type
        if effect.is_increase():
            effect_type = 'increase'
        elif effect.is_decrease():
            effect_type = 'decrease'
        else:
            effect_type = 'none'

        new_fluent = self._transform_expression(problem, new_problem, effect.fluent)
        new_value = self._transform_expression(problem, new_problem, effect.value)
        new_condition = self._transform_expression(problem, new_problem, effect.condition)

        return self._add_single_effect(
            new_action, effect_type, new_fluent, new_value, new_condition, effect.condition, effect.forall
        )

    def _add_instantiated_effects(
            self,
            problem: Problem,
            new_problem: Problem,
            old_action: Action,
            new_action: Action
    ) -> bool:
        """Add all effects to instantiated action. Returns True if any effects added."""
        for effect in old_action.effects:
            success = self._add_instantiated_effect(problem, new_problem, effect, new_action)
            if not success:
                return False

        return len(new_action.effects) > 0

    def _transform_action_arrays(
            self, problem: Problem, new_problem: Problem, old_action: Action
    ) -> Union[Action, None]:
        """Transform array accesses in action."""
        params = OrderedDict((p.name, p.type) for p in old_action.parameters)
        new_action = InstantaneousAction(old_action.name, _parameters=params, _env=problem.environment)

        # Transform preconditions
        for precondition in old_action.preconditions:
            new_precondition = self._transform_expression(problem, new_problem, precondition)

            if new_precondition is None or new_precondition == FALSE():
                return None  # Impossible action

            new_action.add_precondition(new_precondition)

        # Transform effects
        has_valid_effects = self._add_instantiated_effects(problem, new_problem, old_action, new_action)

        if not has_valid_effects:
            return None

        return new_action

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions by substituting array accesses."""
        new_to_old = {}
        for action in problem.actions:
            new_action = self._transform_action_arrays(problem, new_problem, action)
            if new_action is not None:
                new_problem.add_action(new_action)
                new_to_old[new_action] = action
        return new_to_old

    # ==================== FLUENT TRANSFORMATION ====================

    def _get_array_domain_and_type(self, fluent: Fluent) -> Tuple[List[int], Type]:
        """Extract domain and element type from array fluent."""
        current_type = fluent.type
        domain_ranges = []
        dimensions = []

        while current_type.is_array_type():
            dimensions.append(current_type.size)
            domain_ranges.append(range(current_type.size))
            current_type = current_type.elements_type

        # All possible positions
        all_positions = list(product(*domain_ranges))

        # Filter out undefined positions
        if fluent.undefined_positions is not None:
            valid_positions = [
                pos for pos in all_positions
                if pos not in fluent.undefined_positions
            ]
        else:
            valid_positions = all_positions

        self.domains[fluent.name] = valid_positions
        return dimensions, current_type

    def _get_element_value(self, array_value: FNode, indices: Tuple[int, ...]) -> FNode:
        """Extract element value from nested array constant."""
        element = array_value
        for idx in indices:
            element = element.constant_value()[idx]
        return element

    def _get_new_fluent_value(
            self,
            new_problem: Problem,
            new_fluent: Fluent,
            f: FNode,
            v: FNode
    ) -> Dict[FNode, FNode]:
        """Convert array assignment to indexed fluent assignments."""
        # Extract pre-indices from fluent name
        pre_indices = tuple(int(i) for i in re.findall(r'\[([0-9]+)\]', f.fluent().name))
        new_equalities = {}
        old_params = list(f.args)

        if pre_indices:
            # Partial array assignment
            new_params = [self._get_index_object(new_problem, i) for i in pre_indices] + old_params

            if not f.fluent().type.is_array_type():
                # Single element
                new_equalities[new_fluent(*new_params)] = v
            else:
                # Sub-array
                post_indices = [
                    pos[len(pre_indices):]
                    for pos in self.domains[new_fluent.name]
                    if pos[:len(pre_indices)] == pre_indices
                ]

                for post_idx in post_indices:
                    full_params = new_params + [self._get_index_object(new_problem, i) for i in post_idx]
                    element_value = self._get_element_value(v, post_idx)
                    new_equalities[new_fluent(*full_params)] = element_value
        else:
            # Full array assignment
            for pos in self.domains[new_fluent.name]:
                element_value = self._get_element_value(v, pos)
                params = [self._get_index_object(new_problem, i) for i in pos] + old_params
                new_equalities[new_fluent(*params)] = element_value

        return new_equalities

    def _add_array_as_indexed_fluent(self, problem, new_problem, fluent, default_value, index_ut):
        """Transform array fluent into indexed fluent."""
        # Get domain and element type
        n_elements, element_type = self._get_array_domain_and_type(fluent)
        max_index = max(n_elements)

        # Add index objects (with cache)
        for i in range(max_index):
            self._get_index_object(new_problem, i)

        # Create new signature with Index parameters
        new_signature = [Parameter(f'i_{dim + 1}', index_ut)
                         for dim in range(len(n_elements))] + list(fluent.signature)

        new_fluent = Fluent(fluent.name, element_type, new_signature, fluent.environment)
        new_problem.add_fluent(new_fluent, default_initial_value=default_value)

        # Set initial values
        for f, v in problem.explicit_initial_values.items():
            fluent_name = f.fluent().name.split('[')[0]

            if f.fluent() == fluent or fluent_name == fluent.name:
                new_equalities = self._get_new_fluent_value(new_problem, new_fluent, f, v)
                for nf, nv in new_equalities.items():
                    new_problem.set_initial_value(nf, nv)

    def _transform_fluents(self, problem: Problem, new_problem: Problem):
        """Transform array fluents to indexed fluents."""
        index_ut = UserType('Index')

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            if fluent.type.is_array_type():
                self._add_array_as_indexed_fluent(problem, new_problem, fluent, default_value, index_ut)
            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        new_problem.set_initial_value(fluent(*f.args), v)

    # ==================== AXIOMS TRANSFORMATION ====================

    def _transform_axioms(self, problem: Problem, new_problem: Problem, new_to_old: Dict):
        """Transform axioms"""
        for axiom in problem.axioms:
            # Check for integer parameters
            for param in axiom.parameters:
                if param.type.is_int_type():
                    raise NotImplementedError(
                        "Integer parameters in axioms are not supported!"
                    )
            params = OrderedDict((p.name, p.type) for p in axiom.parameters)
            # Clone and transform
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

    # ==================== GOAL TRANSFORMATION ====================

    def _transform_goals(self, problem: Problem, new_problem: Problem):
        """Transform all goals."""
        for goal in problem.goals:
            new_goal = self._transform_expression(problem, new_problem, goal)
            if new_goal is not None:
                new_problem.add_goal(new_goal)

    # ==================== MAIN COMPILATION ====================

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation method."""
        assert isinstance(problem, Problem)

        # Create new problem
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_axioms()
        new_problem.initial_values.clear()
        new_problem.clear_quality_metrics()
        assert self.mode == 'strict' or self.mode == 'permissive'

        # Transform components
        self._transform_fluents(problem, new_problem)
        new_to_old = self._transform_actions(problem, new_problem)
        self._transform_goals(problem, new_problem)
        self._transform_axioms(problem, new_problem, new_to_old)

        # Transform quality metrics
        for metric in problem.quality_metrics:
            if metric.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(metric, new_to_old, new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(metric)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
