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
"""The ExpressionManager is used to create expressions.

All objects are memoized so that two syntactically equivalent expressions
are represented by the same object.
"""


import unified_planning as up
import unified_planning.model.types
from unified_planning.model.operators import OperatorKind
from unified_planning.exceptions import (
    UPTypeError,
    UPExpressionDefinitionError,
    UPValueError,
)
from fractions import Fraction
from typing import Optional, Iterable, List, Union, Dict, Tuple, Iterator, Sequence, Set

BoolExpression = Union[
    "up.model.fnode.FNode",
    "up.model.fluent.Fluent",
    "up.model.parameter.Parameter",
    "up.model.variable.Variable",
    "up.model.presence.Presence",
    "up.model.range_variable.RangeVariable",
    bool,
]
NumericConstant = Union[int, float, Fraction, str]
NumericExpression = Union[NumericConstant, "up.model.fnode.FNode"]
ConstantExpression = Union[
    NumericExpression,
    "up.model.object.Object",
    bool,
]
TimeExpression = Union[
    "up.model.timing.Timing",
    "up.model.timing.Timepoint",
    int,
    float,
    Fraction,
]
ListExpression = Union[
    list,
    List["Expression"],
]
SetExpression = Union[
    set,
    Set["Expression"],
]
Expression = Union[
    TimeExpression,
    BoolExpression,
    ConstantExpression,
    ListExpression,
    SetExpression,
]


def uniform_numeric_constant(value: NumericConstant) -> Union[Fraction, int]:
    """Utility method to handle NumericConstant polymorphism."""
    if not isinstance(value, (float, Fraction)):
        try:
            return int(value)
        except ValueError:
            pass
    try:
        number = Fraction(value)
    except ValueError:
        raise UPValueError(f"Numeric constant {value} can't be converted to a number")
    assert isinstance(number, Fraction)
    if number.denominator == 1:
        return number.numerator
    return number


class ExpressionManager(object):
    """ExpressionManager is responsible for the creation of all expressions."""

    def __init__(self, environment: "up.environment.Environment"):
        self.environment = environment
        self.expressions: Dict[
            "up.model.fnode.FNodeContent", "up.model.fnode.FNode"
        ] = {}
        self._next_free_id = 1

        self.true_expression = self.create_node(
            node_type=OperatorKind.BOOL_CONSTANT, args=tuple(), payload=True
        )
        self.false_expression = self.create_node(
            node_type=OperatorKind.BOOL_CONSTANT, args=tuple(), payload=False
        )
        # empty set as a constant
        self.empty_set_expression = self.create_node(
            node_type=OperatorKind.SET_CONSTANT, args=tuple(), payload=tuple()
        )
        return

    def _polymorph_args_to_iterator(
        self, *args: Union[Expression, Iterable[Expression]]
    ) -> Iterator[Expression]:
        """
        Helper function to return an Iterator of arguments from args.
        This function is used to allow N-ary operators to express their arguments
        both as a list of arguments or as a tuple of arguments:
        e.g. And([a,b,c]) and And(a,b,c)
        are both valid, and they are converted into (a,b,c)
        """
        for a in args:
            if isinstance(a, (Set, set)):
                yield a
            elif isinstance(a, Iterable) and not isinstance(a, str):
                for p in a:
                    yield p
            else:
                yield a

    def auto_promote(
        self, *args: Union[Expression, Iterable[Expression]]
    ) -> List["up.model.fnode.FNode"]:
        """
        Method that takes an iterable of expressions and returns the list
        of these expressions casted to FNode.

        :param args: The iterable of expression that must be promoted to FNode.
        :return: The resulting list of FNode.
        """
        res = []
        for e in self._polymorph_args_to_iterator(*args):
            if isinstance(e, up.model.fluent.Fluent):
                assert (
                    e.environment == self.environment
                ), "Fluent has a different environment of the expression manager"
                res.append(self.FluentExp(e))
            elif isinstance(e, up.model.parameter.Parameter):
                assert (
                    e.environment == self.environment
                ), "Parameter has a different environment of the expression manager"
                res.append(self.ParameterExp(e))
            elif isinstance(e, up.model.variable.Variable):
                assert (
                    e.environment == self.environment
                ), "Variable has a different environment of the expression manager"
                res.append(self.VariableExp(e))
            elif isinstance(e, up.model.range_variable.RangeVariable):
                assert (
                    e.environment == self.environment
                ), "RangeVariable has a different environment of the expression manager"
                res.append(self.RangeVariableExp(e))
            elif isinstance(e, up.model.object.Object):
                assert (
                    e.environment == self.environment
                ), "Object has a different environment of the expression manager"
                res.append(self.ObjectExp(e))
            elif isinstance(e, up.model.timing.Timing):
                res.append(self.TimingExp(e))
            elif isinstance(e, up.model.timing.Timepoint):
                res.append(self.TimingExp(up.model.timing.Timing(delay=0, timepoint=e)))
            elif isinstance(e, up.model.presence.Presence):
                res.append(self.PresentExp(e))
            elif isinstance(e, bool):
                res.append(self.Bool(e))
            elif (
                isinstance(e, int)
                or isinstance(e, float)
                or isinstance(e, Fraction)
                or isinstance(e, str)
            ):
                number = uniform_numeric_constant(e)
                if isinstance(number, int):
                    res.append(self.Int(number))
                else:
                    assert isinstance(number, Fraction)
                    res.append(self.Real(number))
            elif isinstance(e, List):
                res.append(self.Array(e))
            elif isinstance(e, Set):
                res.append(self.Set(e))
            else:
                assert (
                    e.environment == self.environment
                ), "Expression has a different environment of the expression manager"
                res.append(e)
        return res

    def create_node(
        self,
        node_type: OperatorKind,
        args: Tuple["up.model.fnode.FNode", ...],
        payload: Optional[
            Union[
                "up.model.fluent.Fluent",
                "up.model.object.Object",
                "up.model.parameter.Parameter",
                "up.model.variable.Variable",
                "up.model.range_variable.RangeVariable",
                "up.model.timing.Timing",
                "up.model.presence.Presence",
                str,
                bool,
                int,
                Fraction,
                list,
                tuple,
                set,
                Tuple[Union["up.model.variable.Variable", "up.model.range_variable.RangeVariable"], ...],
            ]
        ] = None,
    ) -> "up.model.fnode.FNode":
        """
        Creates the unified_planning expressions if it hasn't been created yet in the environment. Otherwise
        returns the existing one.

        :param node_type: The OperationKind referring to this expression (like a PLUS, MINUS, FLUENT_EXP, etc.).
        :param args: The direct sons in this expression tree; a tuple of expressions.
        :param payload: In some OperationKind contains the information about the expression; for an INT_EXP
            contains the integer, for a FLUENT_EXP the fluent etc.
        :return: The created expression.
        """
        content = up.model.fnode.FNodeContent(node_type, args, payload)
        res = self.expressions.get(content, None)
        if res is not None:
            return res
        else:
            assert all(
                a.environment == self.environment for a in args
            ), "2 FNode in the same expression have different environments"
            n = up.model.fnode.FNode(content, self._next_free_id, self.environment)
            self._next_free_id += 1
            self.expressions[content] = n
            self.environment.type_checker.get_type(n)
            return n

    def Count(
        self, *args: Union[BoolExpression, Iterable[BoolExpression]]
    ) -> "up.model.fnode.FNode":
        """
        | Creates an expression of the form:

            * ``Count(a,b,c)``
            * ``Count([a,b,c])``

        | This function has polymorphic n-arguments.
        | Restriction: Arguments must be ``boolean``.

        :param \\*args: Either an ``Iterable`` of ``boolean expressions``, like ``[a, b, c]``, or an unpacked version
            of it, like ``a, b, c``.
        :return: The ``COUNT`` expression created.
        """
        tuple_args = tuple(self.auto_promote(*args))

        if len(tuple_args) == 0:
            return self.Int(0)
        else:
            return self.create_node(node_type=OperatorKind.COUNT, args=tuple_args)

    def ArrayIndex(
            self, array_expr: Expression, index: Expression
    ) -> "up.model.fnode.FNode":
        """
        | Creates an expression of the form:

            * ``array_expr[index]``

        | Restriction: ``array_expr`` must be of ``array type`` and ``index`` must be an integer expression.

        :param array_expr: The array expression (can be a fluent returning an array or a constant array).
        :param index: The index expression.
        :return: The ``ARRAY_INDEX`` expression created.
        """
        array_expr, index = self.auto_promote(array_expr, index)
        return self.create_node(node_type=OperatorKind.ARRAY_INDEX, args=(array_expr, index))

    def ArrayRead(self, array_exp, index_exp):
        return self.create_node(OperatorKind.ARRAY_READ, (array_exp, index_exp))

    def ArrayWrite(self, array_exp, index_exp):
        return self.create_node(OperatorKind.ARRAY_WRITE, (array_exp, index_exp))

    def SetMember(
        self, element: Expression, set_expr: SetExpression
    ) -> "up.model.fnode.FNode":
        """
        | Creates an expression of the form:

            * ``SetMember(element,set_expr)``

        | Restriction: ``set_expr`` must be of ``set type`` and ``element`` must be of the same type as the elements
        of the set.

        :param element: The element to check if it is a member of ``set_expr``.
        :param set_expr: The set expression (can be a fluent returning a set or a set constant).
        :return: The ``MEMBER`` expression created.
        """
        element, set_expr = self.auto_promote(element, set_expr)
        return self.create_node(node_type=OperatorKind.SET_MEMBER, args=(element, set_expr))

    def SetSubseteq(
        self, set_expr1: SetExpression, set_expr2: SetExpression
    ) -> "up.model.fnode.FNode":
        """
        | Creates an expression of the form:

            * ``SetSubseteq(set_expr1,set_expr2)``

        | Restriction: ``set_expr1`` and ``set_expr1`` must be of ``set type`` and the elements have to be of the same type.

        :param set_expr1: The set expression to check if it is a subseteq of ``set_expr2``.
        :param set_expr2: The set expression (can be a fluent returning a set or a set constant).
        :return: The ``SUBSETEQ`` expression created.
        """
        set_expr1, set_expr2 = self.auto_promote(set_expr1, set_expr2)
        return self.create_node(node_type=OperatorKind.SET_SUBSETEQ, args=(set_expr1, set_expr2))

    def SetDisjoint(
        self, set_expr1: SetExpression, set_expr2: SetExpression
    ) -> "up.model.fnode.FNode":
        """
        | Creates an expression of the form:

            * ``disjoint(set_expr1,set_expr2)``

        | Restriction: ``set_expr1`` and ``set_expr1`` must be of ``set type`` and the elements have to be of the same type.

        :param set_expr1: The element to check if it is a member of ``set_expr``.
        :param set_expr2: The set expression (can be a fluent returning a set or a set constant).
        :return: The ``DISJOINT`` expression created.
        """
        set_expr1, set_expr2 = self.auto_promote(set_expr1, set_expr2)
        return self.create_node(node_type=OperatorKind.SET_DISJOINT, args=(set_expr1, set_expr2))

    def SetCardinality(self, set_expr: SetExpression) -> "up.model.fnode.FNode":
        """
        | Creates an expression of the form:

            * ``Cardinality(set_expr)``

        | Returns the number of elements in the set.

        :param set_expr: A set expression (fluent or constant).
        :return: The ``CARDINALITY`` expression created (returns an integer).
        """
        (set_expr,) = self.auto_promote(set_expr)

        if set_expr.is_set_constant():
            return self.Int(len(set_expr.constant_value()))
        return self.create_node(node_type=OperatorKind.SET_CARDINALITY, args=(set_expr,))

    def SetAdd(
            self, element: Expression, set_expr: SetExpression
    ) -> "up.model.fnode.FNode":
        """
        | Creates an expression that adds an element to a set:

            * ``SetAdd(set_expr, element)`` equivalent to ``set_expr u {element}``

        | Restriction: ``set_expr`` must be of ``set type`` and ``element`` must be
          of the same type as the elements of the set.

        :param set_expr: The set expression.
        :param element: The element to add to the set.
        :return: The ``SET_ADD`` expression created (returns a new set).
        """
        set_expr, element = self.auto_promote(set_expr, element)
        if set_expr == self.EMPTY_SET():
            return self.Set({element})
        return self.create_node(node_type=OperatorKind.SET_ADD, args=(element, set_expr))

    def SetRemove(
            self, element: Expression, set_expr: SetExpression
    ) -> "up.model.fnode.FNode":
        """
        | Creates an expression that removes an element from a set:

            * ``SetRemove(set_expr, element)`` equivalent to ``set_expr \ {element}``

        | Restriction: ``set_expr`` must be of ``set type`` and ``element`` must be
          of the same type as the elements of the set.

        :param set_expr: The set expression.
        :param element: The element to remove from the set.
        :return: The ``SET_REMOVE`` expression created (returns a new set).
        """
        set_expr, element = self.auto_promote(set_expr, element)
        if set_expr == self.EMPTY_SET():
            return self.EMPTY_SET()
        return self.create_node(node_type=OperatorKind.SET_REMOVE, args=(element, set_expr))

    def SetUnion(
            self, set_expr1: SetExpression, set_expr2: SetExpression
    ) -> "up.model.fnode.FNode":
        """
        | Creates a union of sets:

            * ``SetUnion(set1, set2)`` equivalent to ``set1 ∪ set2`` FOR THE MOMENT ONLY 2

        | This function has polymorphic n-arguments.
        | Restriction: All arguments must be of ``set type`` with the same element type.

        :param \\*args: Either an ``Iterable`` of set expressions, like ``[s1, s2]``,
                        or an unpacked version of it, like ``s1, s2, s3``.
        :return: The ``SET_UNION`` expression created.
        """
        set_expr1, set_expr2 = tuple(self.auto_promote(set_expr1, set_expr2))
        return self.create_node(node_type=OperatorKind.SET_UNION, args=(set_expr1, set_expr2))

    def SetIntersection(
            self, set_expr1: SetExpression, set_expr2: SetExpression
    ) -> "up.model.fnode.FNode":
        """
        | Creates an intersection of sets:

            * ``SetIntersection(set_expr1, set_expr2)`` equivalent to ``set_expr1 ∩ set_expr2``

        | This function has polymorphic n-arguments.
        | Restriction: All arguments must be of ``set type`` with the same element type.

        :param set_expr1: The set to subtract from.
        :param set_expr2: The set to subtract.
        :return: The ``SET_INTERSECTION`` expression created.
        """
        set_expr1, set_expr2 = tuple(self.auto_promote(set_expr1, set_expr2))
        return self.create_node(node_type=OperatorKind.SET_INTERSECT, args=(set_expr1, set_expr2))

    def SetDifference(
            self, set_expr1: SetExpression, set_expr2: SetExpression
    ) -> "up.model.fnode.FNode":
        """
        | Creates a set difference:

            * ``SetDifference(set_expr1, set_expr2)`` equivalent to ``set_expr1 \ set_expr2``

        | Restriction: Both arguments must be of ``set type`` with the same element type.

        :param set_expr1: The set to subtract from.
        :param set_expr2: The set to subtract.
        :return: The ``SET_DIFFERENCE`` expression created.
        """
        set_expr1, set_expr2 = self.auto_promote(set_expr1, set_expr2)
        return self.create_node(node_type=OperatorKind.SET_DIFFERENCE, args=(set_expr1, set_expr2))

    def And(
        self, *args: Union[BoolExpression, Iterable[BoolExpression]]
    ) -> "up.model.fnode.FNode":
        """
        | Returns a conjunction of terms.
        | This function has polymorphic n-arguments:

            * ``And(a,b,c)``
            * ``And([a,b,c])``

        | Restriction: Arguments must be ``boolean``.

        :param \\*args: Either an ``Iterable`` of ``boolean`` expressions, like ``[a, b, c]``, or an unpacked version
            of it, like ``a, b, c``.
        :return: The ``AND`` expression created.
        """
        tuple_args = tuple(self.auto_promote(*args))

        if len(tuple_args) == 0:
            return self.TRUE()
        elif len(tuple_args) == 1:
            return tuple_args[0]
        else:
            return self.create_node(node_type=OperatorKind.AND, args=tuple_args)

    def Or(
        self, *args: Union[BoolExpression, Iterable[BoolExpression]]
    ) -> "up.model.fnode.FNode":
        """
        | Returns an disjunction of terms.
        | This function has polymorphic n-arguments:

            * ``Or(a,b,c)``
            * ``Or([a,b,c])``

        | Restriction: Arguments must be ``boolean``

        :param \\*args: Either an ``Iterable`` of ``boolean expressions``, like ``[a, b, c]``, or an unpacked version
            of it, like ``a, b, c``.
        :return: The ``OR`` expression created.
        """
        tuple_args = tuple(self.auto_promote(*args))

        if len(tuple_args) == 0:
            return self.FALSE()
        elif len(tuple_args) == 1:
            return tuple_args[0]
        else:
            return self.create_node(node_type=OperatorKind.OR, args=tuple_args)

    def XOr(
        self, *args: Union[BoolExpression, Iterable[BoolExpression]]
    ) -> "up.model.fnode.FNode":
        """
        | Returns an exclusive disjunction of terms in CNF form.
        | This function has polimorphic n-arguments:

            * XOr(a,b,c)
            * XOr([a,b,c])

        | Restriction: Arguments must be boolean

        :param \\*args: Either an ``Iterable`` of ``boolean expressions``, like ``[a, b, c]``, or an unpacked version
            of it, like ``a, b, c``.
        :return: The exclusive disjunction in CNF form.
        """
        tuple_args = tuple(self.auto_promote(*args))

        if len(tuple_args) == 0:
            return self.FALSE()
        elif len(tuple_args) == 1:
            return tuple_args[0]
        else:
            new_args = []
            for a in tuple_args:
                new_args.append(
                    self.And([a] + [self.Not(o) for o in tuple_args if o is not a])
                )
            return self.Or(new_args)

    def Not(self, expression: BoolExpression) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``not expression``

        Restriction: ``expression`` must be of ``boolean type``

        :param expression: The ``boolean`` expression of which the negation must be created.
        :return: The created ``Not`` expression.
        """
        (expression,) = self.auto_promote(expression)
        if expression.is_not():
            return expression.arg(0)
        return self.create_node(node_type=OperatorKind.NOT, args=(expression,))

    def Implies(
        self, left: BoolExpression, right: BoolExpression
    ) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``left -> right``

        Restriction: ``Left`` and ``Right`` must be of ``boolean type``

        :param left: The ``boolean`` expression acting as the premise of the ``Implies``.
        :param right: The ``boolean`` expression acting as the implied part of the ``Implies``.
        :return: The created ``Implication``.
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.IMPLIES, args=(left, right))

    def Iff(
        self, left: BoolExpression, right: BoolExpression
    ) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``left <-> right``

        Semantically, The expression is ``True`` only if ``left`` and ``right`` have the same value.
        Restriction: ``Left`` and ``Right`` must be of ``boolean type``

        :param left: The ``left`` member of the ``Iff expression``.
        :param right: The ``right`` member of the ``Iff expression``.
        :return: The created ``Iff`` expression.
        """
        if type(left) is list:
            left = [left]
        if type(right) is list:
            right = [right]
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.IFF, args=(left, right))

    def Exists(
        self, expression: BoolExpression, *vars: Union["unified_planning.model.Variable", "unified_planning.model.range_variable.RangeVariable"]
    ) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``Exists (var[0]... var[n]) | expression``

        Restriction: expression must be of ``boolean type`` and
        vars must be of ``Variable`` type

        :param expression: The main expression of the ``existential``. The expression should contain
            the given ``variables``.
        :param \\*vars: All the ``Variables`` appearing in the ``existential`` expression.
        :return: The created ``Existential`` expression.
        """
        expressions = tuple(self.auto_promote(expression))
        if len(vars) == 0:
            raise UPExpressionDefinitionError(
                f"Exists of expression: {str(expression)} must be created with at least one variable, otherwise it is not needed."
            )
        for v in vars:
            if not (isinstance(v, up.model.variable.Variable) or isinstance(v, unified_planning.model.range_variable.RangeVariable)):
                raise UPTypeError("Expecting 'up.Variable' or 'up.RangeVariable, got %s", type(v))
        return self.create_node(
            node_type=OperatorKind.EXISTS, args=expressions, payload=vars
        )

    def Forall(
        self, expression: BoolExpression, *vars: Union["unified_planning.model.Variable", "unified_planning.model.RangeVariable"]
    ) -> "up.model.fnode.FNode":
        """Creates an expression of the form:
            ``Forall (var[0]... var[n]) | expression``

        Restriction: expression must be of ``boolean type`` and
        vars must be of ``Variable`` type

        :param expression: The main expression of the ``universal`` quantifier. The expression should contain
            the given ``variables``.
        :param \\*vars: All the ``Variables`` appearing in the ``universal`` expression.
        :return: The created ``Forall`` expression.
        """
        expressions = tuple(self.auto_promote(expression))
        if len(vars) == 0:
            raise UPExpressionDefinitionError(
                f"Forall of expression: {str(expression)} must be created with at least one variable, otherwise it is not needed."
            )
        for v in vars:
            if not (isinstance(v, up.model.variable.Variable) or isinstance(v, unified_planning.model.range_variable.RangeVariable)):
                raise UPTypeError("Expecting 'up.Variable' or 'up.RangeVariable, got %s", type(v))
        return self.create_node(
            node_type=OperatorKind.FORALL, args=expressions, payload=vars
        )

    def Always(self, expression: BoolExpression) -> "up.model.fnode.FNode":
        """Creates an expression of the form:
            ``Always(a)``

        Restriction: expression must be of ``boolean type`` and with only one arg.

        :param expression: The ``boolean`` expression of the trajectory constraints.
        :return: The created ``Always`` expression.
        """
        expressions = tuple(self.auto_promote(expression))
        return self.create_node(node_type=OperatorKind.ALWAYS, args=expressions)

    def Sometime(self, expression: BoolExpression) -> "up.model.fnode.FNode":
        """Creates an expression of the form:
            ``Sometime(a)``

        Restriction: expression must be of ``boolean type`` and with only one arg.

        :param expression: The ``boolean`` expression of the trajectory constraints.
        :return: The created ``Sometime`` expression.
        """
        expressions = tuple(self.auto_promote(expression))
        return self.create_node(node_type=OperatorKind.SOMETIME, args=expressions)

    def AtMostOnce(self, expression: BoolExpression) -> "up.model.fnode.FNode":
        """Creates an expression of the form:
            ``At-Most-Once(a, b)``

        Restriction: expression must be of ``boolean type`` and with only two arg.

        :param expression: The ``boolean`` expression of the trajectory constraints.
        :return: The created ``At-Most-Once(a, b)`` expression.
        """
        expressions = tuple(self.auto_promote(expression))
        return self.create_node(node_type=OperatorKind.AT_MOST_ONCE, args=expressions)

    def SometimeBefore(
        self, phi: BoolExpression, psi: BoolExpression
    ) -> "up.model.fnode.FNode":
        """Creates an expression of the form:
            ``Sometime-Before(a, b)``

        Restriction: expression must be of ``boolean type`` and with only one args

        :param expression: The ``boolean`` expression of the trajectory constraints.
        :return: The created ``Sometime`` expression.
        """
        expressions = tuple(self.auto_promote(phi, psi))
        return self.create_node(
            node_type=OperatorKind.SOMETIME_BEFORE, args=expressions
        )

    def SometimeAfter(
        self, phi: BoolExpression, psi: BoolExpression
    ) -> "up.model.fnode.FNode":
        """Creates an expression of the form:
            ``Sometime-After(a, b)``

        Restriction: expression must be of ``boolean type`` and with only two arg.

        :param expression: The ``boolean`` expression of the trajectory constraints.
        :return: The created ``Sometime-After(a, b)`` expression.
        """
        expressions = tuple(self.auto_promote(phi, psi))
        return self.create_node(node_type=OperatorKind.SOMETIME_AFTER, args=expressions)

    def FluentExp(
        self, fluent: "up.model.fluent.Fluent", params: Sequence[Expression] = tuple()
    ) -> "up.model.fnode.FNode":
        """
        | Creates an expression for the given ``fluent`` and ``parameters``.
        | Restriction: ``parameters type`` must be compatible with the ``Fluent`` :func:``signature <unified_planning.model.Fluent.signature>``

        :param fluent: The ``Fluent`` that will be set as the ``payload`` of this expression.
        :param params: The Sequence of expressions acting as ``parameters`` for this ``Fluent``; mainly the parameters will
            be :class:``Objects <unified_planning.model.Object>`` (when the ``FluentExp`` is grounded) or
            :func:``Action parameters <unified_planning.model.Action.parameters>`` (when the ``FluentExp`` is lifted).
        :return: The created ``Fluent`` Expression.
        """
        assert fluent.environment == self.environment
        params_exp = self.auto_promote(params)
        if fluent.arity != len(params_exp):
            raise UPExpressionDefinitionError(
                f"In FluentExp, fluent: {fluent.name} has arity {fluent.arity} but {len(params_exp)} parameters were passed."
            )
        return self.create_node(
            node_type=OperatorKind.FLUENT_EXP, args=tuple(params_exp), payload=fluent
        )

    def Dot(
        self,
        agent: Union["up.model.multi_agent.Agent", str],
        fluent_exp: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
    ) -> "up.model.fnode.FNode":
        """
        Creates an expression for the given ``agent`` and ``fluent_exp``.
        Restriction: agent must be of ``agent type`` or the name of an agent and
        fluent_exp must be of ``fluentExp type``

        :param agent: The ``Agent`` that will be set as the ``payload`` of this expression.
        :param fluent_exp: The ``Fluent_exp`` that will be set as the ``args`` of this expression.
        :return: The created ``Dot`` Expression.
        """
        (fluent_exp,) = self.auto_promote(fluent_exp)
        assert fluent_exp.is_fluent_exp()
        if not isinstance(agent, str):
            assert isinstance(agent, up.model.multi_agent.Agent), "Typing not respected"
            assert agent.environment == self.environment
            if fluent_exp.fluent() not in agent.fluents:
                raise UPExpressionDefinitionError(
                    f"Fluent {fluent_exp.fluent()} does not belong to agent {agent.name}"
                )
            agent = agent.name
        assert isinstance(agent, str)
        return self.create_node(
            node_type=OperatorKind.DOT, args=(fluent_exp,), payload=agent
        )

    def ParameterExp(
        self, param: "up.model.parameter.Parameter"
    ) -> "up.model.fnode.FNode":
        """
        Returns an expression for the given :func:`Action parameter <unified_planning.model.Action.parameters>`.

        :param param: The ``Parameter`` that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``param`` as his payload.
        """
        return self.create_node(
            node_type=OperatorKind.PARAM_EXP, args=tuple(), payload=param
        )

    def VariableExp(self, var: "up.model.variable.Variable") -> "up.model.fnode.FNode":
        """
        Returns an expression for the given ``Variable``.

        :param var: The ``Variable`` that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``variable`` as his payload.
        """
        assert var.environment == self.environment
        return self.create_node(
            node_type=OperatorKind.VARIABLE_EXP, args=tuple(), payload=var
        )

    def RangeVariableExp(self, var: "up.model.range_variable.RangeVariable") -> "up.model.fnode.FNode":
        """
        Returns an expression for the given ``RangeVariable``.

        :param var: The ``Variable`` that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``variable`` as his payload.
        """
        assert var.environment == self.environment
        return self.create_node(
            node_type=OperatorKind.RANGE_VARIABLE_EXP, args=tuple(), payload=var
        )

    def ObjectExp(self, obj: "up.model.object.Object") -> "up.model.fnode.FNode":
        """
        Returns an expression for the given object.

        :param obj: The ``Object`` that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given object as his payload.
        """
        assert obj.environment == self.environment
        return self.create_node(
            node_type=OperatorKind.OBJECT_EXP, args=tuple(), payload=obj
        )

    def TimingExp(self, timing: "up.model.timing.Timing") -> "up.model.fnode.FNode":
        """
        Returns an expression for the given ``Timing``.

        :param timing: The ``Timing`` that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``timing`` as his payload.
        """
        return self.create_node(
            node_type=OperatorKind.TIMING_EXP, args=tuple(), payload=timing
        )

    def PresentExp(
        self, presence: "up.model.presence.Presence"
    ) -> "up.model.fnode.FNode":
        return self.create_node(
            node_type=OperatorKind.PRESENT_EXP, args=tuple(), payload=presence
        )

    def TRUE(self) -> "up.model.fnode.FNode":
        """Return the boolean constant ``True``."""
        return self.true_expression

    def FALSE(self) -> "up.model.fnode.FNode":
        """Return the boolean constant ``False``."""
        return self.false_expression

    def EMPTY_SET(self) -> "up.model.fnode.FNode":
        """Return the boolean constant ``False``."""
        return self.empty_set_expression

    def Bool(self, value: bool) -> "up.model.fnode.FNode":
        """
        Return a boolean constant.

        :param value: The boolean value that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``value`` as his payload.
        """
        if not isinstance(value, bool):
            raise UPTypeError("Expecting bool, got %s" % type(value))

        if value:
            return self.true_expression
        else:
            return self.false_expression

    def Int(self, value: int) -> "up.model.fnode.FNode":
        """
        Return an ``int`` constant.

        :param value: The integer that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``integer`` as his payload.
        """
        if not isinstance(value, int):
            raise UPTypeError("Expecting int, got %s" % type(value))
        return self.create_node(
            node_type=OperatorKind.INT_CONSTANT, args=tuple(), payload=value
        )

    def Real(self, value: Fraction) -> "up.model.fnode.FNode":
        """
        Return a ``real`` constant.

        :param value: The ``Fraction`` that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``value`` as his payload.
        """
        if not isinstance(value, Fraction):
            raise UPTypeError("Expecting Fraction, got %s" % type(value))
        return self.create_node(
            node_type=OperatorKind.REAL_CONSTANT, args=tuple(), payload=value
        )

    def Array(self, value: List) -> "up.model.fnode.FNode":
        """
        Return an ``list`` constant.
        :param value: The list that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``list`` as his payload.
        """
        # convert the list to tuple to resolve the hash problem
        adapted_value = tuple(self.auto_promote(value))
        if not isinstance(value, List):
            raise UPTypeError("Expecting List, got %s" % type(value))
        return self.create_node(
            node_type=OperatorKind.ARRAY_CONSTANT, args=tuple(), payload=adapted_value
        )

    def Set(self, value: Set) -> "up.model.fnode.FNode":
        """
        Return an ``set`` constant.
        :param value: The set that must be promoted to ``FNode``.
        :return: The ``FNode`` containing the given ``set`` as his payload.
        """
        # convert the list to tuple to resolve the hash problem
        if isinstance(value, set):
            if len(value) == 0:
                return self.empty_set_expression
            else:
                promoted_elements = []
                for elem in value:
                    # auto_promote retorna una llista, agafem el primer element
                    promoted = self.auto_promote(elem)
                    promoted_elements.extend(promoted)
                adapted_value = tuple(promoted_elements)
        else:
            adapted_value = tuple(self.auto_promote(value))

        return self.create_node(
            node_type=OperatorKind.SET_CONSTANT, args=tuple(), payload=adapted_value
        )

    def Plus(
        self, *args: Union[Expression, Iterable[Expression]]
    ) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``args[0] + ... + args[n]``

        :param \\*args: Either an ``Iterable`` of expressions, like ``[a, b, 3]``, or an unpacked version
            of it, like ``a, b, 3``.
        :return: The ``PLUS`` expression created. (like ``a + b + 3``)
        """
        tuple_args = tuple(self.auto_promote(*args))

        if len(tuple_args) == 0:
            return self.Int(0)
        elif len(tuple_args) == 1:
            return tuple_args[0]
        else:
            return self.create_node(node_type=OperatorKind.PLUS, args=tuple_args)

    def Minus(self, left: Expression, right: Expression) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form: ``left - right``.

        :param left: The ``Minus minuend``.
        :param right: The ``Minus subtrahend``.
        :return: The created ``Minus`` expression.
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.MINUS, args=(left, right))

    def Times(
        self, *args: Union[Expression, Iterable[Expression]]
    ) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``args[0] * ... * args[n]``

        :param \\*args: Either an ``Iterable`` of expressions, like ``[a, b, 3]``, or an unpacked version
            of it, like ``a, b, 3``.
        :return: The ``TIMES`` expression created. (like ``a * b * 3``)
        """
        tuple_args = tuple(self.auto_promote(*args))

        if len(tuple_args) == 0:
            return self.Int(1)
        elif len(tuple_args) == 1:
            return tuple_args[0]
        else:
            return self.create_node(node_type=OperatorKind.TIMES, args=tuple_args)

    def Div(self, left: Expression, right: Expression) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``left / right``

        :param left: The ``Div dividend``.
        :param right: The ``Div divisor``.
        :return: The created ``DIV`` expression.
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.DIV, args=(left, right))

    def LE(self, left: Expression, right: Expression) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``left <= right``.

        :param left: The left side of the ``<=``.
        :param right: The right side of the ``<=``.
        :return: The created ``LE`` expression.
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.LE, args=(left, right))

    def GE(self, left: Expression, right: Expression) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``left >= right``.

        :param left: The left side of the ``>=``.
        :param right: The right side of the ``>=``.
        :return: The created ``GE`` expression.
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.LE, args=(right, left))

    def LT(self, left: Expression, right: Expression) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``left < right``.

        :param left: The left side of the ``<``.
        :param right: The right side of the ``<``.
        :return: The created ``LT`` expression.
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.LT, args=(left, right))

    def GT(self, left: Expression, right: Expression) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``left > right``.

        :param left: The left side of the ``>``.
        :param right: The right side of the ``>``.
        :return: The created ``GT`` expression.
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.LT, args=(right, left))

    def Equals(self, left: Expression, right: Expression) -> "up.model.fnode.FNode":
        """
        Creates an expression of the form:
            ``left == right``.

        NOTE: Is not valid for boolean expression, for those use ``Iff``.

        :param left: The left side of the ``==``.
        :param right: The right side of the ``==``.
        :return: The created ``Equals`` expression.
        """
        if type(left) is list:
            left = [left]
        if type(right) is list:
            right = [right]
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=OperatorKind.EQUALS, args=(left, right))

    def EqualsOrIff(
        self, left: Expression, right: Expression
    ) -> "up.model.fnode.FNode":
        """
        Creates an expression of either one of the following forms:
        ``left <-> right``
        ``left == right``
        :param left: The ``left`` member of the ``EqualsOrIff expression``.
        :param right: The ``right`` member of the ``EqualsOrIff expression``.
        :return: The created ``Equals`` or ``Iff`` expression.
        """

        left, right = self.auto_promote(left, right)
        if left.type.is_bool_type() and right.type.is_bool_type():

            return self.create_node(node_type=OperatorKind.IFF, args=(left, right))
        else:
            return self.create_node(node_type=OperatorKind.EQUALS, args=(left, right))
