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
from ortools.sat.python import cp_model
from typing import Any
from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, Action, ProblemKind, Effect, EffectKind, Object, FNode, InstantaneousAction, \
    Axiom, Parameter
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import get_fresh_name, replace_action, updated_minimize_action_costs
from typing import Optional, Iterator, OrderedDict, Union
from functools import partial
from unified_planning.shortcuts import And, Or, Equals, Not, FALSE, UserType, TRUE, ObjectExp
from typing import List, Dict, Tuple

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

"""
Smart Table compression algorithm based on:
  Charlier et al. "Automatic Synthesis of Smart Table Constraints
  by Abstraction of Table Constraints" (IJCAI 2017)

Integrated with the IntegersRemover / CP-SAT pipeline:
  - Variables are FNodes (fluents, parameters) from unified_planning
  - Solutions come from CP-SAT as List[dict[str, int]] keyed by str(IntVar)
  - Domains are extracted from FNode types (int, bool, user type)
  - The bidict maps FNode → cp_model.IntVar

Supported column constraints: LT (<), GT (>), EQ (=), WILDCARD (*)
  GT(xi, xj) ≡ LT(xj, xi) — in PDDL both map to the lt() predicate,
  just with arguments swapped.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING
from bidict import bidict

if TYPE_CHECKING:
    from unified_planning.model import Problem, FNode
    from ortools.sat.python import cp_model


# ============================================================
# DATA STRUCTURES
# ============================================================

class Op(Enum):
    WILDCARD = "*"
    EQ       = "="
    LT       = "<"
    GT       = ">"   # xi > xj  ≡  lt(xj, xi) in PDDL (arguments swapped)


@dataclass(frozen=True)
class ColConstraint:
    """
    Column constraint for one variable in a smart tuple.

    Unary:    op=LT,  value=3    → xi < 3
    Binary:   op=GT,  col_ref=j  → xi > xj  (j is a column index into col_names)
    Wildcard: op=WILDCARD        → xi = *   (any value — no precondition needed)

    Note on PDDL translation:
      LT(xi, value=v)   → lt(val_xi, nv)
      GT(xi, col_ref=j) → lt(val_xj, val_xi)   <- arguments swapped!
      EQ(xi, col_ref=j) → equals(val_xi, val_xj)
      WILDCARD          → (no precondition)
    """
    op:      Op
    value:   Optional[int] = None   # unary: the constant
    col_ref: Optional[int] = None   # binary: index into col_names

    def satisfies(self, xi_val: int, xj_val: Optional[int] = None) -> bool:
        if self.op == Op.WILDCARD:
            return True
        b = xj_val if self.col_ref is not None else self.value
        if self.op == Op.EQ:  return xi_val == b
        if self.op == Op.LT:  return xi_val < b
        if self.op == Op.GT:  return xi_val > b
        raise ValueError(f"Unknown op: {self.op}")

    def __str__(self):
        if self.op == Op.WILDCARD:
            return "*"
        ref = f"x{self.col_ref}" if self.col_ref is not None else str(self.value)
        return f"{self.op.value}{ref}"


# SmartTuple: one ColConstraint per column (same order as col_names)
SmartTuple = list[ColConstraint]

# SmartTable: list of SmartTuples (disjunction of conjunctions)
SmartTable = list[SmartTuple]

# Internal concrete table: list of tuples of ints
ConcreteTable = list[tuple[int, ...]]


# ============================================================
# DOMAIN EXTRACTION FROM FNode
# ============================================================

def _get_fnode_domain(
    fnode:           "FNode",
    problem:         "Problem",
    object_to_index: dict,
) -> tuple[int, int]:
    """
    Extract integer domain (lb, ub) for a FNode variable.

    - Int fluent/parameter:   use type bounds directly
    - Bool fluent/parameter:  (0, 1)
    - User type:              (0, len(objects) - 1)
    """
    if fnode.is_fluent_exp():
        ftype = fnode.fluent().type
    elif fnode.is_parameter_exp():
        ftype = fnode.parameter().type
    else:
        raise ValueError(f"Unsupported FNode kind: {fnode}")

    if ftype.is_int_type():
        return ftype.lower_bound, ftype.upper_bound

    if ftype.is_bool_type():
        return 0, 1

    if ftype.is_user_type():
        objects = list(problem.objects(ftype))
        return 0, len(objects) - 1

    raise ValueError(f"Unsupported FNode type: {ftype}")


# ============================================================
# INPUT CONVERSION: CP-SAT solutions → ConcreteTable + col_names
# ============================================================

def _solutions_to_concrete_table(
    solutions: list[dict[str, int]],
    variables: bidict,
) -> tuple[ConcreteTable, list["FNode"]]:
    """
    Convert CP-SAT solutions to an internal ConcreteTable.

    CP-SAT solutions are keyed by str(IntVar). We recover the
    corresponding FNode via the bidict (FNode → IntVar).

    Returns:
        (table, col_names) where:
          table[i][j]  = integer value of col_names[j] in solution i
          col_names[j] = FNode for column j
    """
    if not solutions:
        return [], []

    # str(IntVar) → FNode
    var_str_to_fnode: dict[str, "FNode"] = {
        str(var): fnode
        for fnode, var in variables.items()
    }

    # Column order: sorted by var string name for determinism
    first = solutions[0]
    col_names: list["FNode"] = []
    for var_str in sorted(first.keys()):
        fnode = var_str_to_fnode.get(var_str)
        if fnode is not None:
            col_names.append(fnode)

    if not col_names:
        return [], []

    # FNode → str(IntVar) for row construction
    fnode_to_var_str: dict["FNode", str] = {
        fnode: str(var)
        for fnode, var in variables.items()
    }

    table: ConcreteTable = [
        tuple(sol[fnode_to_var_str[fnode]] for fnode in col_names)
        for sol in solutions
    ]

    return table, col_names


# ============================================================
# CORE ALGORITHM  (Charlier et al. IJCAI 2017)
# ============================================================

def _compress_to_smart_table(
    table:   ConcreteTable,
    domains: list[tuple[int, int]],
) -> SmartTable:
    if not table:
        return []

    n            = len(table[0])
    all_concrete = set(table)

    at: list[tuple[tuple, SmartTuple]] = [(row, []) for row in table]

    for step in range(n):
        col     = n - 1 - step
        is_last = (step == n - 1)
        at      = _abstraction_step(at, col, domains, all_concrete, is_last)

    # Reconstruct full SmartTuples: prefix elements become EQ constraints
    result = []
    for (prefix, abstract) in at:
        prefix_ccs = [ColConstraint(op=Op.EQ, value=v) for v in prefix]
        full = prefix_ccs + abstract
        result.append(full)

    assert all(len(r) == n for r in result), \
        f"Incomplete smart tuples after reconstruction: {[(len(r), n) for r in result if len(r) != n]}"

    return result

def _abstraction_step(at, col, domains, all_concrete, is_last):
    nat      = _generate_new_abstract_tuples(at, col, domains, all_concrete)
    combined = [e for e in at if _is_sound(e, all_concrete, domains)] + nat
    return _greedy_covering(combined, all_concrete, is_last)

def _generate_new_abstract_tuples(
    at:           list[tuple[tuple, SmartTuple]],
    col:          int,
    domains:      list[tuple[int, int]],
    all_concrete: set[tuple],
) -> list[tuple[tuple, SmartTuple]]:
    """
    For column `col`, generate new smart tuples by replacing the concrete
    value at `col` with unary or binary column constraints.

    SOUNDNESS: a candidate smart tuple is only added if every concrete tuple
    it covers (within the full domain) belongs to all_concrete. This ensures
    the resulting smart table is exactly equivalent to the input table.
    """
    lb, ub = domains[col]
    nat    = []

    # Group by (prefix[:col], abstract_key)
    groups: dict[tuple, dict] = {}
    for (concrete, abstract) in at:
        prefix  = concrete[:col]
        abs_key = tuple((c.op, c.value, c.col_ref) for c in abstract)
        key     = (prefix, abs_key)
        if key not in groups:
            groups[key] = {
                "prefix":   prefix,
                "abstract": abstract,
                "values":   set(),
                "rows":     [],
            }
        groups[key]["values"].add(concrete[col])
        groups[key]["rows"].append(concrete)

    for data in groups.values():
        prefix   = data["prefix"]
        abstract = data["abstract"]
        values   = data["values"]
        rows     = data["rows"]

        # --- Unary candidates ---
        for cc in _unary_candidates(values, lb, ub):
            candidate = (prefix, [cc] + abstract)
            if _is_sound(candidate, all_concrete, domains):
                nat.append(candidate)

        # --- Binary candidates: col <op> j, for j < col ---
        for op in [Op.LT, Op.GT, Op.EQ]:
            for j in range(col - 1, -1, -1):   # closest first
                cc = ColConstraint(op=op, col_ref=j)
                if all(cc.satisfies(row[col], row[j]) for row in rows):
                    candidate = (prefix, [cc] + abstract)
                    if _is_sound(candidate, all_concrete, domains):
                        nat.append(candidate)
                    break   # take closest valid j regardless of soundness

    return nat

def _unary_candidates(
    values: set[int],
    lb:     int,
    ub:     int,
) -> list[ColConstraint]:
    candidates: list[tuple[int, ColConstraint]] = []

    # WILDCARD covers everything
    candidates.append((ub - lb + 1, ColConstraint(op=Op.WILDCARD)))

    # EQ = v: covers only {v}
    for v in range(lb, ub + 1):
        if values <= {v}:
            candidates.append((1, ColConstraint(op=Op.EQ, value=v)))

    # LT < v: covers {lb, ..., v-1}
    # Only generate v up to ub (not ub+1) to avoid referencing n{ub+1}
    # which may not exist as a Number object.
    # Note: LT < ub+1 would be equivalent to WILDCARD anyway (covers all).
    for v in range(lb + 1, ub + 1):
        covered = set(range(lb, v))
        if values <= covered:
            candidates.append((len(covered), ColConstraint(op=Op.LT, value=v)))

    candidates.sort(key=lambda x: x[0])
    return [cc for (_, cc) in candidates]

def _specificity(abstract: SmartTuple) -> int:
    """
    Tiebreaker: prefer binary constraints > unary LT/EQ > wildcard.
    Binary constraints collapse better in subsequent abstraction steps.
    """
    score = 0
    for cc in abstract:
        if cc.op == Op.WILDCARD:
            score += 0
        elif cc.col_ref is not None:
            score += 10   # binary: best
        elif cc.op == Op.EQ:
            score += 5
        elif cc.op in (Op.LT, Op.GT):
            score += 3
    return score


def _greedy_covering(
    combined:     list[tuple[tuple, SmartTuple]],
    all_concrete: set[tuple],
    is_last:      bool,
) -> list[tuple[tuple, SmartTuple]]:
    """
    Greedy set cover: select minimum smart tuples to cover all concrete rows.
    Sorts by (coverage DESC, specificity DESC) so binary constraints
    are preferred over wildcards when coverage is equal.
    """
    scored: list[tuple[frozenset, int, tuple]] = []
    for entry in combined:
        prefix, abstract = entry
        covered = frozenset(
            c for c in all_concrete
            if _concrete_satisfies_entry(c, prefix, abstract)
        )
        spec = _specificity(abstract)
        scored.append((covered, spec, entry))

    scored.sort(key=lambda x: (-len(x[0]), -x[1]))

    selected  = []
    remaining = set(all_concrete)

    if is_last:
        # Last step: recompute greedily after each selection (paper §3.1)
        while remaining:
            best = max(scored, key=lambda x: (len(x[0] & remaining), x[1]))
            newly = best[0] & remaining
            if not newly:
                break
            selected.append(best[2])
            remaining -= newly
            scored = [
                (cov - newly, spec, entry)
                for (cov, spec, entry) in scored
                if entry is not best[2]
            ]
    else:
        for covered_set, spec, entry in scored:
            if not remaining:
                break
            newly = covered_set & remaining
            if newly:
                selected.append(entry)
                remaining -= newly

    return selected


def _is_sound(entry, all_concrete, domains):
    import itertools
    prefix, abstract = entry
    plen   = len(prefix)
    n_cols = plen + len(abstract)

    for cc in abstract:
        if cc.col_ref is not None:
            if cc.col_ref >= n_cols:
                return False
            # col_ref ha d'apuntar a una columna ANTERIOR a la posició actual
            # (no pot apuntar a sí mateixa ni a columnes posteriors no processades)

    col_ranges = [
        range(domains[plen + i][0], domains[plen + i][1] + 1)
        for i in range(len(abstract))
    ]

    found_any = False
    for vals in itertools.product(*col_ranges):
        def get_col(j):
            return prefix[j] if j < plen else vals[j - plen]

        satisfies = all(
            abstract[i].satisfies(
                vals[i],
                get_col(abstract[i].col_ref) if abstract[i].col_ref is not None else None
            )
            for i in range(len(abstract))
        )
        if satisfies:
            row = prefix + vals
            if row not in all_concrete:
                return False
            found_any = True

    return found_any

def _concrete_satisfies_entry(
    concrete: tuple,
    prefix:   tuple,
    abstract: SmartTuple,
) -> bool:
    """Check if a concrete row satisfies a smart tuple entry (prefix + abstract)."""
    for i, v in enumerate(prefix):
        if concrete[i] != v:
            return False
    prefix_len = len(prefix)
    for i, cc in enumerate(abstract):
        col    = prefix_len + i
        xi_val = concrete[col]
        xj_val = concrete[cc.col_ref] if cc.col_ref is not None else None
        if not cc.satisfies(xi_val, xj_val):
            return False
    return True


# ============================================================
# TOP-LEVEL: CP-SAT solutions → SmartTable (with FNode columns)
# ============================================================

def solutions_to_smart_table(
    solutions:       list[dict[str, int]],
    variables:       bidict,
    problem:         "Problem",
    object_to_index: dict,
) -> tuple[SmartTable, list["FNode"]]:
    """
    Full pipeline: CP-SAT solutions → compressed SmartTable.

    Args:
        solutions:       List[dict[str, int]] from CPSolutionCollector
        variables:       bidict[FNode, IntVar] built in _add_cp_constraints
        problem:         original UP Problem (for domain extraction)
        object_to_index: self._object_to_index from IntegersRemover

    Returns:
        (smart_table, col_names) where:
          smart_table[i] = SmartTuple (one ColConstraint per column)
          col_names[j]   = FNode corresponding to column j
    """
    table, col_names = _solutions_to_concrete_table(solutions, variables)
    if not table:
        return [], col_names

    domains = [
        _get_fnode_domain(fnode, problem, object_to_index)
        for fnode in col_names
    ]

    smart = _compress_to_smart_table(table, domains)
    return smart, col_names


# ============================================================
# PRINT
# ============================================================

def col_constraint_to_str(cc: ColConstraint, col_names: list) -> str:
    if cc.op == Op.WILDCARD:
        return "*"
    if cc.col_ref is not None:
        ref = str(col_names[cc.col_ref])
    else:
        ref = str(cc.value)
    return f"{cc.op.value}{ref}"

def print_smart_table(smart: SmartTable, col_names: list) -> None:
    labels = [str(f) for f in col_names]
    w      = max((len(l) for l in labels), default=6) + 2
    header = "  ".join(f"{name:>{w}}" for name in labels)
    print(header)
    print("-" * len(header))
    for row in smart:
        print("  ".join(
            f"{col_constraint_to_str(cc, col_names):>{w}}"
            for cc in row
        ))


# ============================================================
# PDDL TRANSLATION
# ============================================================

def setup_lt_predicate(new_problem: "Problem") -> "Fluent":
    """
    Create the static 'lt' predicate and populate its initial values.

    lt(Number, Number) is True iff the first argument is strictly less
    than the second. It is added once to new_problem and its initial
    state is fully populated for all pairs of Number objects.

    Call this after _transform_fluents (so Number objects already exist)
    and before translating any SmartTable.

    Returns the lt Fluent so it can be reused in smart_table_to_condition.
    """
    from unified_planning.model import Fluent
    from unified_planning.shortcuts import UserType, BoolType, TRUE, FALSE, ObjectExp
    number_ut = UserType("Number")

    # Create fluent if not already present
    lt_name = "lt"
    if any(f.name == lt_name for f in new_problem.fluents):
        return new_problem.fluent(lt_name)

    lt_fluent = Fluent(lt_name, BoolType(), [Parameter("a", number_ut), Parameter("b", number_ut)])
    new_problem.add_fluent(lt_fluent, default_initial_value=FALSE())

    # Populate: lt(ni, nj) = True iff i < j
    number_objects = list(new_problem.objects(number_ut))

    # Sort by numeric value embedded in object name (n0, n1, n2, ...)
    def obj_value(obj) -> int:
        name = obj.name  # "n0", "n1", ...
        return int(name[1:])

    number_objects.sort(key=obj_value)

    for obj_a in number_objects:
        for obj_b in number_objects:
            if obj_value(obj_a) < obj_value(obj_b):
                new_problem.set_initial_value(
                    lt_fluent(ObjectExp(obj_a), ObjectExp(obj_b)),
                    TRUE()
                )
            # False cases are already covered by default_initial_value=FALSE()

    return lt_fluent


def smart_row_to_conditions(
    smart_row:           list[ColConstraint],
    col_names:       list["FNode"],
    new_problem:     "Problem",
    get_number_obj,  # callable: int → FNode  (self._get_number_object)
) -> List[FNode]:
    """
    Translate a SmartTable into a UP condition (FNode).

    Each SmartTuple becomes a conjunction of PDDL atoms.
    Multiple SmartTuples are combined in a disjunction.

    Translation rules per ColConstraint:
      WILDCARD             → True (no constraint)
      EQ(xi, value=v)      → Equals(new_fluent_xi, number_obj(v))
      LT(xi, value=v)      → lt(new_fluent_xi, number_obj(v))
      EQ(xi, col_ref=j)    → Equals(new_fluent_xi, new_fluent_xj)
      GT(xi, col_ref=j)    → lt(new_fluent_xj, new_fluent_xi)   ← args swapped!
      LT(xi, col_ref=j)    → lt(new_fluent_xi, new_fluent_xj)

    Args:
        smart:           SmartTable from solutions_to_smart_table
        col_names:       FNode list (column i → FNode)
        new_problem:     the compiled UP problem (has Number objects + lt fluent)
        get_number_obj:  self._get_number_object(value) → FNode for nN object

    Returns:
        FNode condition (Or of Ands), ready to add as precondition/goal.
    """
    from unified_planning.shortcuts import And, Equals, TRUE

    lt_fluent = new_problem.fluent("lt")

    def _fnode_to_new_fluent(fnode: "FNode") -> "FNode":
        """Translate an original FNode to its compiled counterpart."""
        if fnode.is_fluent_exp():
            return new_problem.fluent(fnode.fluent().name)(*fnode.args)
        # parameter expressions stay as-is
        return fnode

    def _col_to_new_fluent(col: int) -> "FNode":
        return _fnode_to_new_fluent(col_names[col])

    def _translate_cc(
        cc:  ColConstraint,
        col: int,
    ) -> "FNode":
        """Translate one ColConstraint to a PDDL atom."""
        if cc.op == Op.WILDCARD:
            return TRUE()

        xi = _col_to_new_fluent(col)

        if cc.col_ref is not None:
            # Binary constraint
            xj = _col_to_new_fluent(cc.col_ref)
            if cc.op == Op.EQ:
                return Equals(xi, xj)
            if cc.op == Op.GT:
                # xi > xj  ≡  lt(xj, xi)
                return lt_fluent(xj, xi)
            if cc.op == Op.LT:
                # xi < xj  ≡  lt(xi, xj)
                return lt_fluent(xi, xj)
        else:
            # Unary constraint: compare against a Number object
            num_obj = get_number_obj(new_problem, cc.value)
            if cc.op == Op.EQ:
                return Equals(xi, num_obj)
            if cc.op == Op.LT:
                # xi < v  ≡  lt(xi, nv)
                return lt_fluent(xi, num_obj)
            if cc.op == Op.GT:
                # xi > v  ≡  lt(nv, xi)
                return lt_fluent(num_obj, xi)

        raise ValueError(f"Unhandled ColConstraint: {cc}")

    and_clauses = []
    for col, cc in enumerate(smart_row):
        atom = _translate_cc(cc, col)
        if atom != TRUE():
            and_clauses.append(atom)

    if and_clauses:
        return And(and_clauses) if len(and_clauses) > 1 else and_clauses[0]
    else:
        return TRUE()

class IntegersRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes bounded integers from a planning problem.

    Converts integer fluents to object-typed fluents where objects represent numeric values (n0, n1, n2, ...).
    Integer arithmetic and comparisons are handled by enumerating possible value combinations.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGERS_REMOVING)
        self._domains: Dict[str, Tuple[int, int]] = {}
        self._static_fluents: Dict[FNode, FNode] = {}

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
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
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

    def _replace_static(self, node: FNode) -> FNode:
        """Replace static fluents with their constant values.

        Strategy:
        1. Check if exact FNode is in _static_fluents (ground fluents)
        2. If not, check if the BASE FLUENT is static (parametrized fluents with constant values)
           - A fluent is static if NO action ever modifies it
           - If static, ALL instances have the same constant value
        3. Return the constant value directly (without evaluating parameters)
        """
        if node.is_fluent_exp():
            # First: exact match for ground fluents
            if node in self._static_fluents:
                return self._static_fluents[node]

            # Second: check if base fluent is static
            # Key insight: if a fluent is static, all its instances have the same value
            # because no action ever modifies it, and it's defined by the initial values
            base_fluent = node.fluent()

            # Collect all values for this base fluent from _static_fluents
            values_for_base = []
            for static_node, static_val in self._static_fluents.items():
                if static_node.is_fluent_exp() and static_node.fluent() == base_fluent:
                    values_for_base.append(static_val)

            if values_for_base:
                # Check if all instances have the same value
                # (they should if the fluent is truly static)
                first_val = values_for_base[0]
                if all(v == first_val for v in values_for_base):
                    # All instances of this base fluent have the same constant value
                    # Therefore, any instance can be replaced with that value
                    return first_val

        if node.is_constant() or node.is_parameter_exp():
            return node
        if not node.args:
            return node
        new_args = [self._replace_static(arg) for arg in node.args]
        if all(n is o for n, o in zip(new_args, node.args)):
            return node
        em = node.environment.expression_manager
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _get_number_object(self, problem: Problem, value: int) -> FNode:
        """Get or create object representing numeric value (e.g., n5 for 5)."""
        try:
            return ObjectExp(problem.object(f'n{value}'))
        except UPProblemDefinitionError:
            raise UPProblemDefinitionError(
                f"Number object 'n{value}' not found. "
                f"Ensure _compute_number_range covers all needed values."
            )

    def _has_arithmetic(self, node: FNode) -> bool:
        """
        Check if expression contains arithmetic operations or comparisons.

        Returns True if the node or any of its sub-expressions contains
        arithmetic operators (plus, minus, times, div) or comparisons (lt, le).
        """
        if (node.node_type in self.ARITHMETIC_OPS or
                node.is_le() or node.is_lt()):
            return True
        return any(self._has_arithmetic(arg) for arg in node.args)

    def _requires_cp_in_condition(self, node: FNode) -> bool:
        """
        Determine if a condition requires CP-SAT solver.

        Returns True if the condition contains arithmetic operations or
        integer comparisons that need to be solved via constraint programming.
        This includes arithmetic expressions and comparisons over integer domains.
        """
        if node.node_type in self.ARITHMETIC_OPS:
            return True
        if node.is_lt() or node.is_le():
            return True
        return any(self._requires_cp_in_condition(arg) for arg in node.args)

    # ==================== CP-SAT Constraint Solving ====================

    def _add_cp_constraints(self, problem: Problem, node: FNode, variables: bidict, model: cp_model.CpModel):
        """
        Recursively build CP-SAT constraints from expression tree.

        Converts planning expressions into OR-Tools CP-SAT constraints,
        creating integer variables for fluents and building constraint networks
        for logical and arithmetic operations.
        """
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
                # Get all objects of this type
                objects = list(problem.objects(fluent.type))
                if not objects:
                    raise UPProblemDefinitionError(
                        f"User type {fluent.type} has no objects, cannot create variable for fluent {fluent}"
                    )
                # Create integer variable with domain [0, len(objects)-1]
                var = model.NewIntVar(0, len(objects) - 1, str(node))
                # Store object to index mapping
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
            # Get all objects of this type
            objects = list(problem.objects(param.type))
            if not objects:
                raise UPProblemDefinitionError(
                    f"User type {param.type} has no objects, cannot create variable for parameter {param}"
                )
            var = model.NewIntVar(0, len(objects) - 1, str(node))
            variables[node] = var
            return var

        # Equality
        if node.is_equals():
            left = node.arg(0)
            right = node.arg(1)
            if left.type.is_user_type():
                left_var = self._add_cp_constraints(problem, left, variables, model)
                # If right is a constant object
                if right.is_object_exp():
                    obj = right.object()
                    idx = self._object_to_index.get((left.type, obj))
                    if idx is not None:
                        eq_var = model.NewBoolVar(f"eq_{id(node)}")
                        model.Add(left_var == idx).OnlyEnforceIf(eq_var)
                        model.Add(left_var != idx).OnlyEnforceIf(eq_var.Not())
                        return eq_var

                # If right is another fluent/variable
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
            # and_var == true <=> all child_vars == true
            model.AddBoolAnd(*child_vars).OnlyEnforceIf(and_var)
            # Also ensure: if any child is false, and_var is false
            for child in child_vars:
                model.AddImplication(and_var, child)
            return and_var

        # OR
        if node.is_or():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            or_var = model.NewBoolVar(f"or_{id(node)}")
            # or_var == true <=> at least one child_var == true
            model.AddBoolOr(*child_vars).OnlyEnforceIf(or_var)
            # If or_var is false, all children are false
            for child in child_vars:
                model.AddImplication(child, or_var)
            return or_var

        # Implies: A -> B  equivalent to  (not A) or B
        if node.is_implies():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)

            impl_var = model.NewBoolVar(f"impl_{id(node)}")

            # impl_var == true <=> (not left) or right
            # If impl_var is true: not(left) or right must be true
            model.AddBoolOr(left.Not(), right).OnlyEnforceIf(impl_var)

            # If impl_var is false: left must be true AND right must be false
            model.Add(left == 1).OnlyEnforceIf(impl_var.Not())
            model.Add(right == 0).OnlyEnforceIf(impl_var.Not())
            return impl_var

        # Not
        if node.is_not():
            inner_var = self._add_cp_constraints(problem, node.arg(0), variables, model)
            not_var = model.NewBoolVar(f"not_{id(node)}")
            # not_var is the negation of inner_var
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
                # CP-SAT requires explicit multiplication
                temp = model.NewIntVar(arg.type.lower_bound, arg.type.upper_bound, f"mult_{id(node)}")
                model.AddMultiplicationEquality(temp, result, arg)
                result = temp
            return result

        raise NotImplementedError(f"Node type {node.node_type} not implemented in CP-SAT")

    def _solutions_to_dnf(self, new_problem: Problem, solutions: List[dict], variables: bidict) -> Optional[FNode]:
        """
        Convert CP-SAT solutions to DNF (Disjunctive Normal Form) formula.

        Each solution becomes a conjunction of variable assignments,
        and all solutions are combined in a disjunction.
        """
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
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                    if fluent.type.is_int_type():
                        and_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
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

    def _get_row_solutions(
            self,
            problem: Problem,
            variables: bidict,
            row: List[ColConstraint],
            col_names: List[FNode],
    ) -> List[dict]:
        """
        Given a smart table row, enumerate all concrete solutions that satisfy its constraints using CP-SAT.
        """
        model = cp_model.CpModel()
        row_vars = bidict({})

        # Re-create variables with same domains
        for fnode, orig_var in variables.items():
            if fnode.is_fluent_exp():
                fluent = fnode.fluent()
                if fluent.type.is_int_type():
                    v = model.NewIntVar(fluent.type.lower_bound, fluent.type.upper_bound, str(fnode))
                elif fluent.type.is_user_type():
                    objects = list(problem.objects(fluent.type))
                    v = model.NewIntVar(0, len(objects) - 1, str(fnode))
                else:
                    v = model.NewBoolVar(str(fnode))
            elif fnode.is_parameter_exp():
                param = fnode.parameter()
                objects = list(problem.objects(param.type))
                v = model.NewIntVar(0, len(objects) - 1, str(fnode))
            row_vars[fnode] = v

        # Add constraints from the smart table row
        for i, cc in enumerate(row):
            xi_fnode = col_names[i]
            xi_var = row_vars[xi_fnode]
            if cc.op == Op.WILDCARD:
                pass
            elif cc.col_ref is not None:
                xj_var = row_vars[col_names[cc.col_ref]]
                if cc.op == Op.EQ:
                    model.Add(xi_var == xj_var)
                elif cc.op == Op.LT:
                    model.Add(xi_var < xj_var)
                elif cc.op == Op.GT:
                    model.Add(xi_var > xj_var)
            else:
                if cc.op == Op.EQ:
                    model.Add(xi_var == cc.value)
                elif cc.op == Op.LT:
                    model.Add(xi_var < cc.value)
                elif cc.op == Op.GT:
                    model.Add(xi_var > cc.value)

        return self._solve_with_cp_sat(row_vars, model) or []

    def _add_effects_dnf_mode(
            self,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            variables: bidict,
            solutions: List[dict],
            normalized_effects: List[Effect]
    ) -> None:
        """
        Add effects with conditional arithmetic expressions.

        For arithmetic effects, groups solutions by their result value and creates
        conditional effects where the condition specifies which variable assignments
        lead to each result.
        """
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        for effect in normalized_effects:
            if effect.is_increase() or effect.is_decrease():
                for new_effect in self._transform_increase_decrease_effect(effect, new_problem):
                    new_action.add_effect(new_effect.fluent, new_effect.value,
                                          new_effect.condition, new_effect.forall)

            elif effect.value.node_type in self.ARITHMETIC_OPS:
                # Arithmetic effect - need conditional effects per solution
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                base_cond = self._transform_node(problem, new_problem, effect.condition)
                if base_cond is None:
                    base_cond = TRUE()

                # Group solutions by result value
                result_to_conditions = {}
                for solution in solutions:
                    evaluated = self._evaluate_with_solution(
                        new_problem, effect.value, solution
                    )
                    if evaluated:
                        result_key = str(evaluated)
                        if result_key not in result_to_conditions:
                            result_to_conditions[result_key] = {'value': evaluated, 'solutions': []}
                        result_to_conditions[result_key]['solutions'].append(solution)
                # Create one conditional effect per unique result
                for result_key, data in result_to_conditions.items():
                    # Build condition: base_cond AND (solution1 OR solution2 OR ...)
                    solution_clauses = []
                    for sol in data['solutions']:
                        sol_clause = []
                        for var_str, value in sol.items():
                            fnode = var_str_to_fnode.get(var_str)
                            if fnode and fnode.is_fluent_exp():
                                fluent = fnode.fluent()
                                new_fl = new_problem.fluent(fluent.name)(*fnode.args)
                                if fluent.type.is_int_type():
                                    sol_clause.append(Equals(new_fl, self._get_number_object(new_problem, value)))
                                elif fluent.type.is_user_type():
                                    obj = self._get_object_from_index(fluent.type, value)
                                    if obj:
                                        sol_clause.append(Equals(new_fl, ObjectExp(obj)))
                                elif fluent.type.is_bool_type():
                                    sol_clause.append(new_fl if value == 1 else Not(new_fl))

                        if sol_clause:
                            solution_clauses.append(And(sol_clause) if len(sol_clause) > 1 else sol_clause[0])

                    if solution_clauses:
                        solutions_or = Or(solution_clauses) if len(solution_clauses) > 1 else solution_clauses[0]
                        full_cond = And(base_cond, solutions_or).simplify() if base_cond != TRUE() else solutions_or
                        new_action.add_effect(new_fluent, data['value'], full_cond)

            else:
                # Simple assignment
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_value = self._transform_node(problem, new_problem, effect.value)
                new_cond = self._transform_node(problem, new_problem, effect.condition)
                if new_cond is None:
                    new_cond = TRUE()
                if new_fluent and new_value:
                    new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)

    def _solve_with_cp_sat(self, variables, cp_model_obj):
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

    # ==================== EFFECT TRANSFORMATION ====================

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
            problem: Problem,
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

    def _create_dnf_actions(
            self,
            old_action: Action,
            problem: Problem,
            new_problem: Problem,
            params: OrderedDict,
            solutions: List[dict],
            variables: bidict,
            normalized_effects: List[Effect],
    ) -> List[Action]:
        print("action:", old_action.name)
        print("solutions:", len(solutions), solutions[0])
        new_actions = []
        for idx, solution in enumerate(solutions):
            action_name = f"{old_action.name}_d{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)

            # Precondition
            and_clauses = []
            for fnode, var in variables.items():
                var_name = str(var)
                if var_name not in solution:
                    continue
                cond = self._create_precondition_from_variable(
                    fnode, solution[var_name], problem, new_problem
                )
                if cond:
                    and_clauses.append(cond)
            if and_clauses:
                new_action.add_precondition(And(and_clauses) if len(and_clauses) > 1 else and_clauses[0])

            # Effects
            self._add_effects_for_solution(new_action, problem, new_problem, solution, normalized_effects)
            new_actions.append(new_action)

        return new_actions

    def _add_effects_for_solution(
            self,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            solution: dict,
            normalized_effects: List[Effect],
    ) -> None:
        for effect in normalized_effects:
            if effect.is_increase() or effect.is_decrease():
                fluent = effect.fluent.fluent()
                new_fluent = new_problem.fluent(fluent.name)(*effect.fluent.args)
                try:
                    delta = effect.value.constant_value()
                except:
                    for new_eff in self._transform_increase_decrease_effect(effect, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                fluent_var_str = str(effect.fluent)
                cur_val = solution.get(fluent_var_str)
                if cur_val is None:
                    # No tenim el valor concret → fallback al mètode genèric
                    for new_eff in self._transform_increase_decrease_effect(effect, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                next_val = (cur_val + delta) if effect.is_increase() else (cur_val - delta)
                new_obj = self._get_number_object(new_problem, next_val)

                # Condició de l'efecte original (pot ser condicional)
                if effect.condition is not None and not effect.condition.is_true():
                    cond_result = self._evaluate_with_solution(new_problem, effect.condition, solution)
                    if cond_result == FALSE():
                        continue  # Condició sempre falsa → saltem l'efecte
                    elif cond_result == TRUE():
                        cond = TRUE()
                    else:
                        cond = self._transform_node(problem, new_problem, effect.condition)
                else:
                    cond = TRUE()
                new_action.add_effect(new_fluent, new_obj, cond, effect.forall)

            elif effect.value.node_type in self.ARITHMETIC_OPS:
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                base_cond = self._transform_node(problem, new_problem, effect.condition) or TRUE()
                result_fnode = self._evaluate_with_solution(new_problem, effect.value, solution)
                if result_fnode and result_fnode != effect.value:
                    new_action.add_effect(new_fluent, result_fnode, base_cond)

            else:
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_value = self._transform_node(problem, new_problem, effect.value)
                new_cond = self._transform_node(problem, new_problem, effect.condition) or TRUE()
                if new_fluent and new_value:
                    new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)

    def _add_effects_for_smart_row(
            self,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            variables: bidict,
            col_names: List[FNode],
            row: List[ColConstraint],
            normalized_effects: List[Effect],
    ) -> None:
        # Build a new CP-SAT model with:
        # 1. Row constraints (from smart table)
        # 2. Effect assignment constraints (result variable per arithmetic effect)
        model = cp_model.CpModel()
        row_vars = bidict({})

        # Re-create variables with same domains
        for fnode, orig_var in variables.items():
            if fnode.is_fluent_exp():
                fluent = fnode.fluent()
                if fluent.type.is_int_type():
                    v = model.NewIntVar(fluent.type.lower_bound, fluent.type.upper_bound, str(fnode))
                elif fluent.type.is_user_type():
                    objects = list(problem.objects(fluent.type))
                    v = model.NewIntVar(0, len(objects) - 1, str(fnode))
                else:
                    v = model.NewBoolVar(str(fnode))
            elif fnode.is_parameter_exp():
                objects = list(problem.objects(fnode.parameter().type))
                v = model.NewIntVar(0, len(objects) - 1, str(fnode))
            row_vars[fnode] = v

        # 1. Add smart row constraints
        for i, cc in enumerate(row):
            xi_var = row_vars[col_names[i]]
            if cc.op == Op.WILDCARD:
                pass
            elif cc.col_ref is not None:
                xj_var = row_vars[col_names[cc.col_ref]]
                if cc.op == Op.EQ:
                    model.Add(xi_var == xj_var)
                elif cc.op == Op.LT:
                    model.Add(xi_var < xj_var)
                elif cc.op == Op.GT:
                    model.Add(xi_var > xj_var)
            else:
                if cc.op == Op.EQ:
                    model.Add(xi_var == cc.value)
                elif cc.op == Op.LT:
                    model.Add(xi_var < cc.value)
                elif cc.op == Op.GT:
                    model.Add(xi_var > cc.value)

        # 2. Add effect constraints and collect result variables
        effect_result_vars: dict[int, cp_model.IntVar] = {}
        for eff_idx, effect in enumerate(normalized_effects):
            if effect.value.node_type not in self.ARITHMETIC_OPS:
                continue
            fluent = effect.fluent.fluent()
            lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
            result_var = model.NewIntVar(lb, ub, f"effect_result_{eff_idx}")
            expr = self._add_cp_constraints(problem, effect.value, row_vars, model)
            model.Add(result_var == expr)
            effect_result_vars[eff_idx] = result_var

        # Afegir els result_vars al collector manualment
        all_vars = list(row_vars.values()) + list(effect_result_vars.values())

        solver = cp_model.CpSolver()
        collector = CPSolutionCollector(all_vars)
        solver.parameters.enumerate_all_solutions = True
        solver.Solve(model, collector)
        solutions = collector.solutions

        # 4. Generate effects
        for effect in normalized_effects:
            if effect.is_increase() or effect.is_decrease():
                fluent = effect.fluent.fluent()
                new_fluent = new_problem.fluent(fluent.name)(*effect.fluent.args)
                try:
                    delta = effect.value.constant_value()
                except:
                    for new_eff in self._transform_increase_decrease_effect(effect, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                fluent_var_str = str(effect.fluent)
                seen = set()
                for sol in solutions:
                    cur_val = sol.get(fluent_var_str)
                    if cur_val is None or cur_val in seen:
                        continue
                    seen.add(cur_val)
                    next_val = (cur_val + delta) if effect.is_increase() else (cur_val - delta)
                    new_obj = self._get_number_object(new_problem, next_val)

                    # Si totes les solucions de la row tenen el mateix valor → condició redundant
                    all_same = all(s.get(fluent_var_str) == cur_val for s in solutions)
                    if all_same:
                        cond = TRUE()
                    else:
                        old_obj = self._get_number_object(new_problem, cur_val)
                        cond = Equals(new_fluent, old_obj)

                    new_action.add_effect(new_fluent, new_obj, cond, effect.forall)

            elif effect.value.node_type in self.ARITHMETIC_OPS:
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                base_cond = self._transform_node(problem, new_problem, effect.condition) or TRUE()

                # Group solutions by result value
                result_groups: dict[int, list[dict]] = {}
                result_var_name = f"effect_result_{normalized_effects.index(effect)}"
                for sol in solutions:
                    val = sol.get(result_var_name)
                    if val is None:
                        continue
                    result_groups.setdefault(val, []).append(sol)

                for result_val, group_sols in result_groups.items():
                    new_val = self._get_number_object(new_problem, result_val)

                    if len(group_sols) == len(solutions):
                        # All solutions give same result → no condition needed
                        new_action.add_effect(new_fluent, new_val, base_cond)
                    else:
                        # Build condition: which variable assignments lead to this result
                        # Only include variables that appear in the effect expression
                        clauses = []
                        for sol in group_sols:
                            sol_conds = []
                            for fnode in row_vars:
                                var_name = str(fnode)
                                if var_name not in sol:
                                    continue
                                cond = self._create_precondition_from_variable(
                                    fnode, sol[var_name], problem, new_problem
                                )
                                if cond:
                                    sol_conds.append(cond)
                            if sol_conds:
                                clauses.append(And(sol_conds) if len(sol_conds) > 1 else sol_conds[0])

                        if clauses:
                            sols_or = Or(clauses) if len(clauses) > 1 else clauses[0]
                            full_cond = And(base_cond, sols_or).simplify() if base_cond != TRUE() else sols_or
                            new_action.add_effect(new_fluent, new_val, full_cond)

            else:
                # Simple assignment
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_value = self._transform_node(problem, new_problem, effect.value)
                new_cond = self._transform_node(problem, new_problem, effect.condition) or TRUE()
                if new_fluent and new_value:
                    new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)

    def _create_instantiated_actions(
            self,
            old_action: Action,
            problem: Problem,
            new_problem: Problem,
            params: OrderedDict,
            smart_table: list[list[ColConstraint]],
            col_names: list[FNode],
            variables: bidict,
            normalized_effects: List[Effect],
    ) -> List[Action]:
        """
        Create instantiated actions from solutions.
        """
        new_actions = []
        for idx, smart_row in enumerate(smart_table):
            action_name = f"{old_action.name}_d{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)
            # Add preconditions
            new_condition = smart_row_to_conditions(smart_row, col_names, new_problem, self._get_number_object)
            new_action.add_precondition(new_condition)

            # Add effects
            self._add_effects_for_smart_row(
                new_action, problem, new_problem, variables, col_names, smart_row, normalized_effects
            )
            new_actions.append(new_action)

        return new_actions

    def _add_effect_bounds_constraints(self, problem, variables, cp_model_obj, unstatic_effects):
        for effect in unstatic_effects:
            if not effect.fluent.is_fluent_exp():
                continue
            fluent = effect.fluent.fluent()
            if not fluent.type.is_int_type():
                continue

            lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
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

    def _transform_action_integers(
            self, problem: Problem, new_problem: Problem, old_action: Action
    ) -> List[Action]:
        """
        Transform action with integer arithmetic into compiled form.

        Strategy:
        1. Separate preconditions into arithmetic and non-arithmetic
        2. Transform non-arithmetic preconditions directly
        3. For arithmetic preconditions: use CP-SAT to generate DNF
        4. Create single action with combined preconditions
        5. Handle arithmetic effects with conditional assignments
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

        # Separate preconditions: those requiring CP-SAT vs. direct transformation
        has_cp_preconditions = any(self._requires_cp_in_condition(p) for p in unstatic_preconditions)

        # Check if we have arithmetic that requires CP-SAT
        has_arithmetic_effects = any(
            effect.value.node_type in self.ARITHMETIC_OPS or effect.is_increase() or effect.is_decrease()
            for effect in unstatic_effects
        )

        # No arithmetic at all
        if not has_cp_preconditions and not has_arithmetic_effects:
            action_name = f"{old_action.name}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)
            for dp in unstatic_preconditions:
                transformed = self._transform_node(problem, new_problem, dp)
                if transformed and transformed != TRUE():
                    new_action.add_precondition(transformed)
            for effect in unstatic_effects:
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_value = self._transform_node(problem, new_problem, effect.value)
                new_cond = self._transform_node(problem, new_problem, effect.condition)
                if new_cond is None:
                    new_cond = TRUE()
                if new_fluent and new_value:
                    new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)
            return [new_action]

        # Que passa quan te nomes algun efecte increase/decrease? No caldria passar a or-tools

        # ===== Arithmetic path: use CP-SAT =====
        # Clear mappings from previous action
        self._object_to_index = {}
        self._index_to_object = {}

        # Build CP-SAT model
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        # Add arithmetic preconditions as constraints
        if has_cp_preconditions:
            result_var = self._add_cp_constraints(problem, And(unstatic_preconditions), variables, cp_model_obj)
            cp_model_obj.Add(result_var == 1)

        # Add effect result bounds as constraints
        self._add_effect_bounds_constraints(problem, variables, cp_model_obj, unstatic_effects)


        # Solve CP-SAT to get all valid variable assignments
        solutions = self._solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            return []

        MODE = "direct" # smart_table / direct (without smart table)
        if MODE == "smart_table":
            smart_table, col_names = solutions_to_smart_table(solutions, variables, new_problem, self._object_to_index)
            #print_smart_table(smart_table, col_names)
            return self._create_instantiated_actions(
                old_action, problem, new_problem, params, smart_table, col_names, variables, unstatic_effects
            )
        else:
            return self._create_dnf_actions(
                old_action, problem, new_problem, params, solutions, variables, unstatic_effects
            )

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions by grounding integer parameters."""
        new_to_old = {}
        total_original = len(problem.actions)
        total_new = 0
        for action in problem.actions:
            new_actions = self._transform_action_integers(problem, new_problem, action)
            for new_action in new_actions:
                new_problem.add_action(new_action)
                new_to_old[new_action] = action
            total_new += len(new_actions)
        print(f"\n=== ACTION TRANSFORMATION SUMMARY ===")
        print(f"Original actions: {total_original}")
        print(f"New actions: {total_new}")
        return new_to_old

    def _find_static_fluents(self, problem: Problem, fluents: dict[FNode, FNode]) -> Dict[FNode, FNode]:
        """
        Find static fluents, including type-based static detection.

        A fluent instance is static if:
        1. No action ever modifies the base fluent, OR
        2. The base fluent is only modified for DIFFERENT object types than this instance

        Example: x(plant1) is static if only x(agent) is modified in actions,
        because plant1 is a Plant, not an Agent.
        """
        # First, use original logic: track which BASE fluents are modified
        modifiable_base_fluents = set()
        modifiable_fluent_exps = set()

        for action in problem.actions:
            has_parameters = len(action.parameters) > 0
            for effect in action.effects:
                if effect.fluent.is_fluent_exp():
                    if has_parameters:
                        modifiable_base_fluents.add(effect.fluent.fluent())
                    else:
                        modifiable_fluent_exps.add(effect.fluent)

        # Second, apply type-based detection for parametric fluents
        # Track which (base_fluent, param_index) is modified by which TYPES
        modified_by_param_type = {}  # {(base_fluent, param_idx): set(Type)}

        for action in problem.actions:
            for effect in action.effects:
                if not effect.fluent.is_fluent_exp():
                    continue

                base_fluent = effect.fluent.fluent()
                effect_exp = effect.fluent

                # Only do type-based detection if fluent has parameters
                if len(base_fluent.signature) == 0:
                    continue

                # Get signature (list of parameters) of the base fluent
                signature = base_fluent.signature

                # For each parameter of the base fluent
                for param_idx in range(len(signature)):
                    # Get the argument at this position in the effect
                    if param_idx >= len(effect_exp.args):
                        continue

                    arg = effect_exp.args[param_idx]

                    # Determine the TYPE of this argument
                    arg_type = None
                    if arg.is_parameter_exp():
                        # Parameter from action: find its type
                        param = arg.parameter()
                        for ap in action.parameters:
                            if ap.name == param.name:
                                arg_type = ap.type
                                break
                    elif arg.is_fluent_exp() or arg.is_object_exp():
                        # Object or fluent: get its type
                        arg_type = arg.type

                    if arg_type:
                        key = (base_fluent, param_idx)
                        if key not in modified_by_param_type:
                            modified_by_param_type[key] = set()
                        modified_by_param_type[key].add(arg_type)

        # Identify static fluents
        static_fluents = {}
        for fluent_exp, value in fluents.items():
            if not fluent_exp.is_fluent_exp():
                # Non-fluent expressions are always static
                static_fluents[fluent_exp] = value
                continue

            base_fluent = fluent_exp.fluent()

            # Original logic: check if base fluent is modified
            is_modifiable = (
                    base_fluent in modifiable_base_fluents or
                    fluent_exp in modifiable_fluent_exps
            )

            if is_modifiable:
                # Base fluent is modified. Check type-based detection for parametric instances
                if len(fluent_exp.args) == 0:
                    # No parameters: use original logic (not static)
                    continue

                # Has parameters: apply type-based detection
                is_static = True
                for param_idx, arg in enumerate(fluent_exp.args):
                    key = (base_fluent, param_idx)

                    # If this parameter position is never modified, continue
                    if key not in modified_by_param_type:
                        continue

                    # Get the type of this specific argument
                    arg_type = arg.type

                    # If this TYPE was modified at this position, it's not static
                    if arg_type in modified_by_param_type[key]:
                        is_static = False
                        break

                if is_static:
                    static_fluents[fluent_exp] = value
            else:
                # Base fluent is never modified: static
                static_fluents[fluent_exp] = value

        return static_fluents

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

    def _transform_goals(self, problem: Problem, new_problem: Problem) -> None:
        """
        Transform goals: separate arithmetic and non-arithmetic.
        Only use CP-SAT for arithmetic goals.
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

        # Separate goals: arithmetic vs non-arithmetic
        cp_goals = [g for g in non_static_goals if self._requires_cp_in_condition(g)]
        direct_goals = [g for g in non_static_goals if not self._requires_cp_in_condition(g)]
        print(f"Direct goals: {len(direct_goals)}")
        print(f"Arithmetic goals: {len(cp_goals)}")

        # ===== Add direct (non-arithmetic) goals =====
        for goal in direct_goals:
            transformed = self._transform_node(problem, new_problem, goal)
            if transformed and transformed != TRUE():
                new_problem.add_goal(transformed)

        # ===== If no arithmetic goals, done =====
        if not cp_goals:
            return

        # ===== HAS ARITHMETIC GOALS: Use CP-SAT only for them =====
        # Clear mappings
        self._object_to_index = {}
        self._index_to_object = {}

        # Build CP-SAT model
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        # Add only arithmetic goals as constraints
        result_var = self._add_cp_constraints(problem, And(cp_goals), variables, cp_model_obj)
        cp_model_obj.Add(result_var == 1)

        # Solve
        solutions = self._solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            raise UPProblemDefinitionError("No possible goal satisfies arithmetic constraints!")

        MODE_GOALS = "dnf" # smart_table / dnf
        if MODE_GOALS == "smart_table":
            print("creating smart table for goals....")
            smart_table, col_names = solutions_to_smart_table(solutions, variables, new_problem, self._object_to_index)
            #print_smart_table(smart_table, col_names)
            # Algorithm to smart table constraints
            new_goals = []
            for smart_row in smart_table:
                new_goals.append(smart_row_to_conditions(smart_row, col_names, new_problem, self._get_number_object))
            print(new_goals)
            if new_goals:
                new_problem.add_goal(And(new_goals))
            else:
                raise UPProblemDefinitionError("No possible goal!")
        else: # dnf
            dnf_formula = self._solutions_to_dnf(new_problem, solutions, variables)
            if dnf_formula:
                new_problem.add_goal(dnf_formula)
            else:
                raise UPProblemDefinitionError("No possible goal!")

    def _get_object_from_index(self, user_type, index):
        """
        Get object corresponding to an index for a UserType.

        Uses the internal index-to-object mapping created during CP-SAT constraint building.
        """
        if hasattr(self, '_index_to_object'):
            return self._index_to_object.get((user_type, index))
        return None

    def _evaluate_with_solution(
            self,
            new_problem: Problem,
            expr: FNode,
            solution: dict,
    ) -> Optional[FNode]:
        """
        Evaluate an expression using values from a CP-SAT solution.

        Recursively evaluates the expression, replacing variables with their
        assigned values from the solution. Returns a concrete value or the
        original expression if evaluation is not possible.
        """
        def evaluate_recursive(node: FNode):
            if node.is_constant():
                return node.constant_value()
            if node.is_object_exp():
                obj = node.object()
                if hasattr(self, '_object_to_index'):
                    idx = self._object_to_index.get((obj.type, obj))
                    if idx is not None:
                        return idx
                return None
            if node.is_fluent_exp() or node.is_parameter_exp():
                var_str = str(node)
                if var_str in solution:
                    val = solution[var_str]
                    if node.type.is_user_type():
                        node_type = node.fluent().type if node.is_fluent_exp() else node.parameter().type
                        return self._get_object_from_index(node_type, val)
                    return val
                return None
            if node.is_true():
                return True
            if node.is_false():
                return False
            if node.is_plus():
                values = [evaluate_recursive(arg) for arg in node.args]
                if all(v is not None for v in values):
                    return sum(values)
            if node.is_minus():
                values = [evaluate_recursive(arg) for arg in node.args]
                if all(v is not None for v in values):
                    if len(values) == 1:
                        return -values[0]
                    return values[0] - sum(values[1:])
            return None

        result = evaluate_recursive(expr)
        if result is None:
            return expr
        elif isinstance(result, bool):
            return TRUE() if result else FALSE()
        elif isinstance(result, int):
            return self._get_number_object(new_problem, result)
        elif isinstance(result, Object):
            return ObjectExp(result)
        return expr

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
                self._domains[fluent.name] = (lb, ub)

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

    def _compute_number_range(self, problem: Problem) -> tuple[int, int]:
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

    def _compile(
            self,
            problem: "up.model.AbstractProblem",
            compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)
        self._domains.clear()

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_axioms()
        new_problem.initial_values.clear()
        new_problem.clear_quality_metrics()

        # Compute the range of integer values needed across the entire problem
        global_lb, global_ub = self._compute_number_range(problem)
        ut_number = UserType('Number')
        for v in range(global_lb, global_ub + 1):
            new_problem.add_object(Object(f'n{v}', ut_number))

        # Transform components
        self._static_fluents = self._find_static_fluents(problem, problem.initial_values)
        print("self._static_fluents", self._static_fluents)
        self._transform_fluents(problem, new_problem)

        # ========== Transform Actions ==========
        new_to_old = self._transform_actions(problem, new_problem)

        # ========== Transform Axioms ==========
        self._transform_axioms(problem, new_problem, new_to_old)

        # ========== Transform Goals ==========
        self._transform_goals(problem, new_problem)

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
            new_problem,
            partial(replace_action, map=new_to_old),
            self.name
        )
