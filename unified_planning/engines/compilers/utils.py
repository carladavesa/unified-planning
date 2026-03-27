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
"""This module defines different utility functions for the compilers."""
import bidict
from ortools.sat.python import cp_model
from fractions import Fraction
import unified_planning as up
from unified_planning.exceptions import UPConflictingEffectsException, UPUsageError, UPProblemDefinitionError
from unified_planning.environment import Environment
from unified_planning.model import (
    FNode,
    TimeInterval,
    Action,
    InstantaneousAction,
    DurativeAction,
    Problem,
    Effect,
    Expression,
    BoolExpression,
    NumericConstant,
    SimulatedEffect,
    Parameter,
    DurationInterval,
    TimePointInterval,
    PlanQualityMetric,
    MinimizeActionCosts,
    MinimizeExpressionOnFinalState,
    MaximizeExpressionOnFinalState,
    Oversubscription,
    TemporalOversubscription,
    AbstractProblem, OperatorKind,
)
from unified_planning.plans import ActionInstance
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    Union,
    cast,
)


def check_and_simplify_conditions(
    problem: AbstractProblem, action: DurativeAction, simplifier
) -> Tuple[bool, List[Tuple[TimeInterval, FNode]]]:
    """
    Simplifies conditions and if it is False (a contraddiction)
    returns False, otherwise returns True.
    If the simplification is True (a tautology) removes all conditions at the given timing.
    If the simplification is still an AND rewrites back every "arg" of the AND
    in the conditions
    If the simplification is not an AND sets the simplification as the only
    condition at the given timing.
    Then, the new conditions are returned as a List[Tuple[Timing, FNode]] and the user can
    decide how to use the new conditions.
    """
    # new action conditions
    nac: List[Tuple[TimeInterval, FNode]] = []
    # i = interval, lc = list condition
    for i, lc in action.conditions.items():
        # conditions (as an And FNode)
        c = problem.environment.expression_manager.And(lc)
        # conditions simplified
        cs = simplifier.simplify(c)
        if cs.is_bool_constant():
            if not cs.bool_constant_value():
                return (
                    False,
                    [],
                )
        else:
            if cs.is_and():
                for new_cond in cs.args:
                    nac.append((i, new_cond))
            else:
                nac.append((i, cs))
    return (True, nac)


def check_and_simplify_preconditions(
    problem: AbstractProblem, action: InstantaneousAction, simplifier
) -> Tuple[bool, List[FNode]]:
    """
    Simplifies preconditions and if it is False (a contraddiction)
    returns False, otherwise returns True.
    If the simplification is True (a tautology) removes all preconditions.
    If the simplification is still an AND rewrites back every "arg" of the AND
    in the preconditions
    If the simplification is not an AND sets the simplification as the only
    precondition.
    Then, the new preconditions are returned as a List[FNode] and the user can
    decide how to use the new preconditions.
    """
    # action preconditions
    ap = action.preconditions
    if len(ap) == 0:
        return (True, [])
    # preconditions (as an And FNode)
    p = problem.environment.expression_manager.And(ap)
    # preconditions simplified
    ps = simplifier.simplify(p)
    # new action preconditions
    nap: List[FNode] = []
    if ps.is_bool_constant():
        if not ps.bool_constant_value():
            return (False, [])
    else:
        if ps.is_and():
            nap.extend(ps.args)
        else:
            nap.append(ps)
    action._set_preconditions(nap)
    return (True, nap)


def create_effect_with_given_subs(
    problem: Problem,
    old_effect: Effect,
    simplifier,
    subs: Dict[Expression, Expression],
) -> Optional[Effect]:
    new_fluent = old_effect.fluent.substitute(subs)
    new_value = simplifier.simplify(old_effect.value.substitute(subs))
    new_condition = simplifier.simplify(old_effect.condition.substitute(subs))
    if new_condition == problem.environment.expression_manager.FALSE():
        return None
    else:
        return Effect(
            new_fluent, new_value, new_condition, old_effect.kind, old_effect.forall
        )


def create_action_with_given_subs(
    problem: Problem,
    old_action: Action,
    simplifier,
    subs: Dict[Expression, Expression],
) -> Optional[Action]:
    """This method is used to instantiate the actions parameters to a constant."""
    naming_list: List[str] = []
    for param, value in subs.items():
        assert isinstance(param, Parameter)
        assert isinstance(value, FNode)
        naming_list.append(str(value))
    c_subs = cast(Dict[Parameter, FNode], subs)
    if isinstance(old_action, InstantaneousAction):
        new_action = InstantaneousAction(
            get_fresh_name(problem, old_action.name, naming_list),
            _env=old_action.environment,
        )
        for p in old_action.preconditions:
            new_action.add_precondition(p.substitute(subs))
        for e in old_action.effects:
            new_effect = create_effect_with_given_subs(problem, e, simplifier, subs)
            if new_effect is not None:
                # We try to add the new effect, but a compiler might generate conflicting effects,
                # so the action is just considered invalid
                try:
                    new_action._add_effect_instance(new_effect)
                except UPConflictingEffectsException:
                    return None
        se = old_action.simulated_effect
        if se is not None:
            new_fluents = []
            for f in se.fluents:
                new_fluents.append(f.substitute(subs))

            def fun(_problem, _state, _):
                assert se is not None
                return se.function(_problem, _state, c_subs)

            # We try to add the new simulated effect, but a compiler might generate conflicting effects,
            # so the action is just considered invalid
            try:
                new_action.set_simulated_effect(SimulatedEffect(new_fluents, fun))
            except UPConflictingEffectsException:
                return None
        is_feasible, new_preconditions = check_and_simplify_preconditions(
            problem, new_action, simplifier
        )
        if not is_feasible:
            return None
        new_action._set_preconditions(new_preconditions)
        return new_action
    elif isinstance(old_action, DurativeAction):
        new_durative_action = DurativeAction(
            get_fresh_name(problem, old_action.name, naming_list),
            _env=old_action.environment,
        )
        old_duration = old_action.duration
        new_duration = DurationInterval(
            old_duration.lower.substitute(subs),
            old_duration.upper.substitute(subs),
            old_duration.is_left_open(),
            old_duration.is_right_open(),
        )
        new_durative_action.set_duration_constraint(new_duration)
        for i, cl in old_action.conditions.items():
            for c in cl:
                new_durative_action.add_condition(i, c.substitute(subs))
        for t, el in old_action.effects.items():
            for e in el:
                new_effect = create_effect_with_given_subs(problem, e, simplifier, subs)
                if new_effect is not None:
                    # We try to add the new simulated effect, but a compiler might generate conflicting effects,
                    # so the action is just considered invalid
                    try:
                        new_durative_action._add_effect_instance(t, new_effect)
                    except UPConflictingEffectsException:
                        return None
        for t, se in old_action.simulated_effects.items():
            new_fluents = []
            for f in se.fluents:
                new_fluents.append(f.substitute(subs))

            def fun(_problem, _state, _):
                assert se is not None
                return se.function(_problem, _state, c_subs)

            # We try to add the new simulated effect, but a compiler might generate conflicting effects,
            # so the action is just considered invalid
            try:
                new_durative_action.set_simulated_effect(
                    t, SimulatedEffect(new_fluents, fun)
                )
            except UPConflictingEffectsException:
                return None
        is_feasible, new_conditions = check_and_simplify_conditions(
            problem, new_durative_action, simplifier
        )
        if not is_feasible:
            return None
        new_durative_action.clear_conditions()
        for interval, c in new_conditions:
            new_durative_action.add_condition(interval, c)
        return new_durative_action
    else:
        raise NotImplementedError


def get_fresh_name(
    problem: AbstractProblem,
    original_name: str,
    parameters_names: Sequence[str] = tuple(),
    trailing_info: Optional[str] = None,
) -> str:
    """This method returns a fresh name for the problem, given a name and an iterable of names in input."""
    name_list = [original_name]
    name_list.extend(parameters_names)
    if trailing_info:
        name_list.append(trailing_info)
    new_name = "_".join(name_list)
    base_name = new_name
    count = 0
    while problem.has_name(new_name):
        new_name = f"{base_name}_{str(count)}"
        count += 1
    return new_name


def lift_action_instance(
    action_instance: ActionInstance,
    map: Dict["up.model.Action", Tuple["up.model.Action", List["up.model.FNode"]]],
) -> ActionInstance:
    """ "map" is a map from every action in the "grounded_problem" to the tuple
    (original_action, parameters).

    Where the grounded action is obtained by grounding
    the "original_action" with the specific "parameters"."""
    lifted_action, parameters = map[action_instance.action]
    return ActionInstance(lifted_action, action_instance.actual_parameters + tuple(parameters))


def replace_action(
    action_instance: ActionInstance,
    map: Dict["up.model.Action", Optional["up.model.Action"]],
) -> Optional[ActionInstance]:
    try:
        replaced_action = map[action_instance.action]
    except KeyError:
        raise UPUsageError(
            "The Action of the given ActionInstance does not have a valid replacement."
        )
    if replaced_action is not None:
        if type(replaced_action) is tuple:
            non_number_param = [a for a in action_instance.actual_parameters if str(a.type) != 'Number']
            return ActionInstance(
                replaced_action[0],
                non_number_param,
                action_instance.agent,
                action_instance.motion_paths,
            )
        else:
            return ActionInstance(
                replaced_action,
                action_instance.actual_parameters,
                action_instance.agent,
                action_instance.motion_paths,
            )
    else:
        return None


def add_invariant_condition_apply_function_to_problem_expressions(
    original_problem: Problem,
    new_problem: Problem,
    condition: Optional[FNode] = None,
    function: Optional[Callable[[FNode], FNode]] = None,
) -> Dict[Action, Optional[Action]]:
    """
    This function takes the original problem, the new problem and adds to the new problem
    all the fields that involve an expression, applying the given function (the identity if it is None)
    to all expressions in the problem and adding the condition as an invariant for the whole problem;
    adding it as a final goal and as a precondition for every action. For the temporal case,
    whenever there is the possibility that a point in time is relevant, the condition is also added there.

    NOTE: The new_problem field will be modified!

    :param original_problem: The Problem acting as a base that will be modified in the new problem.
    :param new_problem: The problem created from the original problem; outside of this function the name,
        the fluents and the objects must be manually added.
    :param condition: Optionally, the condition to add in every relevant point of the Problem, making
        it de-facto an invariant.
    :param function: Optionally, the function that will be called and that creates every expression of the
        new problem.
    :return: The mapping from the actions of the new problem to the actions of the original problem;
        every action is mapped to the action it was generated from.
    """
    env = new_problem.environment
    em = env.expression_manager
    if condition is None:
        condition = em.TRUE()
    assert condition is not None
    if function is None:
        function = lambda x: x
    new_to_old: Dict[Action, Optional[Action]] = {}

    for constraint in original_problem.trajectory_constraints:
        new_problem.add_trajectory_constraint(function(constraint))

    for original_action in original_problem.actions:
        params = OrderedDict(((p.name, p.type) for p in original_action.parameters))
        if isinstance(original_action, InstantaneousAction):
            new_action: Union[
                InstantaneousAction, DurativeAction
            ] = InstantaneousAction(original_action.name, params, env)
            assert isinstance(new_action, InstantaneousAction)
            new_cond = em.And(
                *map(function, original_action.preconditions), condition
            ).simplify()
            if new_cond.is_false():
                continue
            elif new_cond.is_and():
                for arg in new_cond.args:
                    new_action.add_precondition(arg)
            else:
                new_action.add_precondition(new_cond)
            for effect in original_action.effects:
                new_action._add_effect_instance(
                    _apply_function_to_effect(effect, function)
                )
        elif isinstance(original_action, DurativeAction):
            new_action = DurativeAction(original_action.name, params, env)
            assert isinstance(new_action, DurativeAction)
            old_duration = original_action.duration
            new_duration = DurationInterval(
                function(old_duration.lower),
                function(old_duration.upper),
                old_duration.is_left_open(),
                old_duration.is_right_open(),
            )
            new_action.set_duration_constraint(new_duration)
            for interval, cond_list in original_action.conditions.items():
                new_cond = em.And(*map(function, cond_list), condition).simplify()
                if new_cond.is_false():
                    continue
                elif new_cond.is_and():
                    for arg in new_cond.args:
                        new_action.add_condition(interval, arg)
                else:
                    new_action.add_condition(interval, new_cond)
            for timing, effects in original_action.effects.items():
                for effect in effects:
                    new_action._add_effect_instance(
                        timing, _apply_function_to_effect(effect, function)
                    )
                interval = TimePointInterval(timing)
                if interval not in new_action.conditions:
                    new_action.add_condition(interval, condition)
        else:
            raise NotImplementedError
        new_problem.add_action(new_action)
        new_to_old[new_action] = original_action

    for interval, goal_list in original_problem.timed_goals.items():
        new_goal = em.And(*map(function, goal_list), condition).simplify()
        if new_goal.is_and():
            for arg in new_goal.args:
                new_problem.add_timed_goal(interval, arg)
        else:
            new_problem.add_timed_goal(interval, new_goal)
    for timing, effects in original_problem.timed_effects.items():
        for effect in effects:
            new_problem._add_effect_instance(
                timing, _apply_function_to_effect(effect, function)
            )
        interval = TimePointInterval(timing)
        if interval not in new_problem.timed_goals:
            new_problem.add_timed_goal(interval, condition)

    new_goal = em.And(*map(function, original_problem.goals), condition).simplify()
    if new_goal.is_and():
        for arg in new_goal.args:
            new_problem.add_goal(arg)
    else:
        new_problem.add_goal(new_goal)

    for qm in original_problem.quality_metrics:
        if qm.is_minimize_action_costs():
            assert isinstance(qm, MinimizeActionCosts)
            new_costs: Dict["up.model.Action", "up.model.Expression"] = {}
            for new_a, old_a in new_to_old.items():
                if old_a is None:
                    continue
                cost = qm.get_action_cost(old_a)
                if cost is not None:
                    cost = function(cost)
                    new_costs[new_a] = cost
            new_qm: PlanQualityMetric = MinimizeActionCosts(
                new_costs, environment=new_problem.environment
            )
        elif qm.is_minimize_expression_on_final_state():
            assert isinstance(qm, MinimizeExpressionOnFinalState)
            new_qm = MinimizeExpressionOnFinalState(
                function(qm.expression), environment=new_problem.environment
            )
        elif qm.is_maximize_expression_on_final_state():
            assert isinstance(qm, MaximizeExpressionOnFinalState)
            new_qm = MaximizeExpressionOnFinalState(
                function(qm.expression), environment=new_problem.environment
            )
        elif qm.is_oversubscription():
            assert isinstance(qm, Oversubscription)
            new_goals: Dict[BoolExpression, NumericConstant] = {}
            for goal, gain in qm.goals.items():
                new_goal = function(em.And(goal, condition).simplify())
                new_goals[new_goal] = (
                    cast(Union[int, Fraction], new_goals.get(new_goal, 0)) + gain
                )
            new_qm = Oversubscription(new_goals, environment=new_problem.environment)
        elif qm.is_temporal_oversubscription():
            assert isinstance(qm, TemporalOversubscription)
            new_temporal_goals: Dict[
                Tuple["up.model.timing.TimeInterval", "up.model.BoolExpression"],
                NumericConstant,
            ] = {}
            for (interval, goal), gain in qm.goals.items():
                new_goal = function(em.And(goal, condition).simplify())
                new_temporal_goals[(interval, new_goal)] = (
                    cast(
                        Union[int, Fraction],
                        new_temporal_goals.get((interval, new_goal), 0),
                    )
                    + gain
                )
            new_qm = TemporalOversubscription(
                new_temporal_goals, environment=new_problem.environment
            )
        else:
            new_qm = qm
        new_problem.add_quality_metric(new_qm)

    for fluent, value in original_problem.initial_values.items():
        new_problem.set_initial_value(function(fluent), function(value))

    return new_to_old


def _apply_function_to_effect(
    effect: Effect, function: Callable[[FNode], FNode]
) -> Effect:
    auto_promote = effect.environment.expression_manager.auto_promote
    return Effect(
        function(effect.fluent),
        function(effect.value),
        function(effect.condition),
        effect.kind,
        tuple((exp.variable() for exp in auto_promote(effect.forall))),
    )


def updated_minimize_action_costs(
    quality_metric: PlanQualityMetric,
    new_to_old: Union[Dict[Action, Action], Dict[Action, Optional[Action]]],
    environment: Environment,
):
    """
    This method takes a `MinimizeActionCosts` `PlanQualityMetric`, a mapping from the new
    action introduced by the compiler to the old action of the problem (None if the
    new action) does not have a counterpart in the original problem) and returns the
    updated equivalent metric for the new problem. This simply changes the costs keys
    and does not alter the cost expression, so it does not cover use-cases like grounding.

    :param quality_metric: The `MinimizeActionCosts`metric to update.
    :param new_to_old: The action's mapping from the compiled problem to the original problem.
    :param environment: The environment of the new problem (therefore, also of the new actions).
    """
    assert isinstance(quality_metric, MinimizeActionCosts)
    new_costs: Dict["up.model.Action", "up.model.Expression"] = {}
    for new_act, old_act in new_to_old.items():
        if old_act is not None:
            new_cost = quality_metric.get_action_cost(old_act)
            if new_cost is not None:
                new_costs[new_act] = new_cost
        else:
            new_costs[new_act] = environment.expression_manager.Int(0)
    return MinimizeActionCosts(new_costs, environment=environment)


def split_all_ands(exp_list: List[FNode]) -> List[FNode]:
    """
    Helper function. Takes in input a List of FNodes and returns a list of FNodes that do not contain any AND operator as the first operator.

    :param exp_list: The List of FNodes that we want to remove AND operators from.
    :return: A list of FNodes not containing AND as the first operators such that AND(e for e in in_exp_list) is equivalent to AND(e for e in returned_list).
    """
    end_list = []
    start_list = exp_list.copy()
    while len(start_list) > 0:
        temp_list = []
        for exp in start_list:
            if exp.is_and():
                for sub_exp in exp.args:
                    temp_list.append(sub_exp)
            else:
                end_list.append(exp)
        start_list = temp_list
    return end_list


# --- INTEGERS UTILS ---

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

def requires_arithmetic(node: FNode) -> bool:
    from unified_planning.model.operators import OperatorKind
    ARITHMETIC_OPS = {OperatorKind.PLUS, OperatorKind.MINUS, OperatorKind.TIMES, OperatorKind.DIV}
    if node.node_type in ARITHMETIC_OPS or node.is_lt() or node.is_le():
        return True
    return any(requires_arithmetic(arg) for arg in node.args)

def compute_integer_range(problem: Problem) -> tuple[int, int]:
    """
    Scan the entire problem to find the full range of integer values needed.

    The range is determined by:
    - Bounds of all integer fluents (lb, ub)
    - Integer constants appearing in expressions (e.g. the 6 in a+b <= 6),
      because these constants become Number objects in lt(val, n6).

    We do NOT consider the possible values of arithmetic sub-expressions
    (sums, differences, etc.) because those are always assigned to fluents
    whose domain already bounds the result.

    Returns (global_lb, global_ub).
    """
    global_lb = 0
    global_ub = 0

    # 1. Bounds from integer fluents
    for fluent in problem.fluents:
        if fluent.type.is_int_type():
            lb = fluent.type.lower_bound
            ub = fluent.type.upper_bound
            if lb is not None:
                global_lb = min(global_lb, lb)
            if ub is not None:
                global_ub = max(global_ub, ub)

    # 2. Integer constants in expressions (e.g. comparisons like count <= 1)
    all_expressions = []
    for action in problem.actions:
        all_expressions.extend(action.preconditions)
        for effect in action.effects:
            all_expressions.append(effect.value)
            all_expressions.append(effect.condition)
    all_expressions.extend(problem.goals)

    def scan(node: FNode):
        if node is None:
            return
        if node.is_int_constant():
            v = node.constant_value()
            nonlocal global_lb, global_ub
            global_lb = min(global_lb, v)
            global_ub = max(global_ub, v)
        for arg in node.args:
            scan(arg)

    for expr in all_expressions:
        if expr is not None:
            scan(expr)
    return global_lb, global_ub

def solve_with_cp_sat(variables, cp_model_obj):
    """
    Use CP-SAT solver to enumerate all valid value assignments.

    Returns a list of solutions, where each solution is a dictionary
    mapping variable names to their assigned values.
    """
    solver = cp_model.CpSolver()
    collector = CPSolutionCollector(list(variables.values()))
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(cp_model_obj, collector)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    solutions = collector.solutions
    return solutions

def add_cp_constraints(
    problem: Problem,
    node: FNode,
    variables: bidict,
    model: cp_model.CpModel,
    object_to_index: dict,
) -> any:
    # -- Constants --
    if node.is_constant():
        return node.constant_value()

    # -- Fluents --
    if node.is_fluent_exp():
        if node in variables:
            return variables[node]
        fluent = node.fluent()
        if fluent.type.is_int_type():
            var = model.NewIntVar(fluent.type.lower_bound, fluent.type.upper_bound, str(node))
        elif fluent.type.is_user_type():
            objects = list(problem.objects(fluent.type))
            if not objects:
                raise UPProblemDefinitionError(
                    f"User type {fluent.type} has no objects, cannot create variable for fluent {fluent}"
                )
            var = model.NewIntVar(0, len(objects) - 1, str(node))
            for idx, obj in enumerate(objects):
                object_to_index[(fluent.type, obj)] = idx
        else:
            var = model.NewBoolVar(str(node))
        variables[node] = var
        return var

    # -- Parameters --
    if node.is_parameter_exp():
        if node in variables:
            return variables[node]
        param = node.parameter()
        assert param.type.is_user_type(), f"Parameter type {param.type} not supported"
        objects = list(problem.objects(param.type))
        if not objects:
            raise UPProblemDefinitionError(
                f"User type {param.type} has no objects, cannot create variable for parameter {param}"
            )
        var = model.NewIntVar(0, len(objects) - 1, str(node))
        variables[node] = var
        return var

    # -- Equality --
    if node.is_equals():
        left_node, right_node = node.arg(0), node.arg(1)
        if left_node.type.is_user_type():
            left_var = add_cp_constraints(problem, left_node, variables, model, object_to_index)
            if right_node.is_object_exp():
                obj = right_node.object()
                idx = object_to_index.get((left_node.type, obj))
                if idx is not None:
                    eq_var = model.NewBoolVar(f"eq_{id(node)}")
                    model.Add(left_var == idx).OnlyEnforceIf(eq_var)
                    model.Add(left_var != idx).OnlyEnforceIf(eq_var.Not())
                    return eq_var
            else:
                right_var = add_cp_constraints(problem, right_node, variables, model, object_to_index)
                eq_var = model.NewBoolVar(f"eq_{id(node)}")
                model.Add(left_var == right_var).OnlyEnforceIf(eq_var)
                model.Add(left_var != right_var).OnlyEnforceIf(eq_var.Not())
                return eq_var
        else:
            left  = add_cp_constraints(problem, node.arg(0), variables, model, object_to_index)
            right = add_cp_constraints(problem, node.arg(1), variables, model, object_to_index)
            eq_var = model.NewBoolVar(f"eq_{id(node)}")
            model.Add(left == right).OnlyEnforceIf(eq_var)
            model.Add(left != right).OnlyEnforceIf(eq_var.Not())
            return eq_var

    # -- AND --
    if node.is_and():
        children = [add_cp_constraints(problem, a, variables, model, object_to_index) for a in node.args]
        and_var = model.NewBoolVar(f"and_{id(node)}")
        model.AddBoolAnd(*children).OnlyEnforceIf(and_var)
        for child in children:
            model.AddImplication(and_var, child)
        return and_var

    # -- OR --
    if node.is_or():
        children = [add_cp_constraints(problem, a, variables, model, object_to_index) for a in node.args]
        or_var = model.NewBoolVar(f"or_{id(node)}")
        model.AddBoolOr(*children).OnlyEnforceIf(or_var)
        for child in children:
            model.AddImplication(child, or_var)
        return or_var

    # -- IMPLIES --
    if node.is_implies():
        left  = add_cp_constraints(problem, node.arg(0), variables, model, object_to_index)
        right = add_cp_constraints(problem, node.arg(1), variables, model, object_to_index)
        impl_var = model.NewBoolVar(f"impl_{id(node)}")
        model.AddBoolOr(left.Not(), right).OnlyEnforceIf(impl_var)
        model.Add(left  == 1).OnlyEnforceIf(impl_var.Not())
        model.Add(right == 0).OnlyEnforceIf(impl_var.Not())
        return impl_var

    # -- NOT --
    if node.is_not():
        inner = add_cp_constraints(problem, node.arg(0), variables, model, object_to_index)
        not_var = model.NewBoolVar(f"not_{id(node)}")
        model.Add(not_var == (1 - inner))
        return not_var

    # -- LT --
    if node.is_lt():
        left  = add_cp_constraints(problem, node.arg(0), variables, model, object_to_index)
        right = add_cp_constraints(problem, node.arg(1), variables, model, object_to_index)
        lt_var = model.NewBoolVar(f"lt_{id(node)}")
        model.Add(left <  right).OnlyEnforceIf(lt_var)
        model.Add(left >= right).OnlyEnforceIf(lt_var.Not())
        return lt_var

    # -- LE --
    if node.is_le():
        left  = add_cp_constraints(problem, node.arg(0), variables, model, object_to_index)
        right = add_cp_constraints(problem, node.arg(1), variables, model, object_to_index)
        le_var = model.NewBoolVar(f"le_{id(node)}")
        model.Add(left <=  right).OnlyEnforceIf(le_var)
        model.Add(left  > right).OnlyEnforceIf(le_var.Not())
        return le_var

    # -- PLUS --
    if node.is_plus():
        return sum(
            add_cp_constraints(problem, a, variables, model, object_to_index)
            for a in node.args
        )

    # -- MINUS --
    if node.is_minus():
        args = [add_cp_constraints(problem, a, variables, model, object_to_index) for a in node.args]
        return args[0] if len(args) == 1 else args[0] - sum(args[1:])

    # -- TIMES --
    if node.is_times():
        args = [add_cp_constraints(problem, a, variables, model, object_to_index) for a in node.args]
        result = args[0]
        for arg in args[1:]:
            assert hasattr(arg, 'type'), \
                f"TIMES not supported between linear expressions, only IntVars. Got: {type(arg)}"
            temp = model.NewIntVar(arg.type.lower_bound, arg.type.upper_bound, f"mult_{id(node)}")
            model.AddMultiplicationEquality(temp, result, arg)
            result = temp
        return result

    raise NotImplementedError(f"Node type {node.node_type} not implemented in CP-SAT translation")

def add_effect_bounds_constraints(
        problem: Problem,
        variables: bidict,
        model: cp_model.CpModel,
        effects: List[Effect],
        object_to_index: dict,
):
    for effect in effects:
        if not effect.fluent.is_fluent_exp():
            continue
        fluent = effect.fluent.fluent()
        if not fluent.type.is_int_type():
            continue

        lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
        # Registers the fluent variable
        fluent_var = add_cp_constraints(problem, effect.fluent, variables, model, object_to_index)

        if effect.is_increase() or effect.is_decrease():

            try:
                delta = effect.value.constant_value()
                result_expr = fluent_var + delta if effect.is_increase() else fluent_var - delta
            except:
                delta_expr = add_cp_constraints(problem, effect.value, variables, model, object_to_index)
                result_expr = fluent_var + delta_expr if effect.is_increase() else fluent_var - delta_expr

            if effect.condition is not None and not effect.condition.is_true():
                cond_var = add_cp_constraints(problem, effect.condition, variables, model, object_to_index)
                model.Add(result_expr >= lb).OnlyEnforceIf(cond_var)
                model.Add(result_expr <= ub).OnlyEnforceIf(cond_var)
            else:
                model.Add(result_expr >= lb)
                model.Add(result_expr <= ub)

        else:
            if effect.value.node_type not in {OperatorKind.PLUS, OperatorKind.MINUS, OperatorKind.DIV, OperatorKind.TIMES}:
                continue
            expr = add_cp_constraints(problem, effect.value, variables, model, object_to_index)
            if effect.condition is not None and not effect.condition.is_true():
                if effect.condition.is_false():
                    continue
                cond_var = add_cp_constraints(problem, effect.condition, variables, model, object_to_index)
                model.Add(expr >= lb).OnlyEnforceIf(cond_var)
                model.Add(expr <= ub).OnlyEnforceIf(cond_var)
            else:
                model.Add(expr >= lb)
                model.Add(expr <= ub)

    for effect in effects:
        if requires_arithmetic(effect.condition):
            register_fluent_variables(
                problem, effect.condition, variables, model, object_to_index
            )

def register_fluent_variables(
    problem: Problem, node: FNode, variables: bidict, model: cp_model.CpModel, object_to_index: dict,
) -> None:
    """Register fluent variables from an expression without adding any constraints."""
    if node.is_fluent_exp():
        if node not in variables:
            fluent = node.fluent()
            if fluent.type.is_int_type():
                var = model.NewIntVar(
                    fluent.type.lower_bound, fluent.type.upper_bound, str(node)
                )
                variables[node] = var
        return
    for arg in node.args:
        register_fluent_variables(problem, arg, variables, model, object_to_index)

def substitute_modified_fluents(action: Action, expr: FNode) -> FNode:
    """
    Substitute fluents modified by the action with their 'after' expression.

    For increase(f, d): f → f + d
    For decrease(f, d): f → f - d
    For assign(f, v):   f → v
    """
    em = expr.environment.expression_manager

    # Build map: fluent → after expression
    after_map = {}
    for effect in action.effects:
        if not effect.fluent.is_fluent_exp():
            continue
        if effect.is_increase():
            after_map[effect.fluent] = em.Plus(effect.fluent, effect.value)
        elif effect.is_decrease():
            after_map[effect.fluent] = em.Minus(effect.fluent, effect.value)
        else:
            after_map[effect.fluent] = effect.value

    def substitute(node: FNode) -> FNode:
        if node.is_fluent_exp() and node in after_map:
            return after_map[node]
        if not node.args:
            return node
        new_args = [substitute(arg) for arg in node.args]
        if all(n is o for n, o in zip(new_args, node.args)):
            return node
        return em.create_node(node.node_type, tuple(new_args))

    return substitute(expr)

def evaluate_goal_in_initial_state(problem: Problem, goal: FNode) -> bool:
    """Evaluates the goal initial value."""

    def eval_node(node):
        if node.is_int_constant():
            return node.constant_value()
        if node.is_fluent_exp():
            val = problem.initial_values.get(node)
            if val is None:
                for f, v in problem.initial_values.items():
                    if str(f) == str(node):
                        return v.constant_value() if v.is_constant() else None
            return val.constant_value() if val is not None and val.is_constant() else None
        if node.is_lt():
            l, r = eval_node(node.arg(0)), eval_node(node.arg(1))
            return (l < r) if l is not None and r is not None else None
        if node.is_le():
            l, r = eval_node(node.arg(0)), eval_node(node.arg(1))
            return (l <= r) if l is not None and r is not None else None
        if node.is_plus():
            vals = [eval_node(a) for a in node.args]
            return sum(vals) if all(v is not None for v in vals) else None
        if node.is_minus():
            vals = [eval_node(a) for a in node.args]
            if all(v is not None for v in vals):
                return vals[0] - sum(vals[1:])
        if node.is_equals():
            l, r = eval_node(node.arg(0)), eval_node(node.arg(1))
            return (l == r) if l is not None and r is not None else None
        return None

    result = eval_node(goal)
    return bool(result) if result is not None else False
