# Unified-Planning PDDL Extensions: Bounded Integers, Arrays, Sets, and Integer Quantifiers

This document describes every change made to the base unified-planning library to support four
PDDL extensions: **bounded integers**, **N-dimensional arrays**, **sets**, and **integer-range
quantifiers** (`forall`/`exists` over bounded integer types).
The changes span five layers: data model, type system, parser, compilation pipeline, and test domains.

---

## 1. PDDL Syntax Overview

### Bounded integers

```pddl
(:types
    size  - (number 0 3)    ; integer in [0, 3]
    range - (number 0 15)   ; integer in [0, 15]
)
(:functions (top ?p - peg) - size)
(:action move
    :parameters (?r - range)
    :precondition (= (top ?p) ?r)
    ...
)
```

### Arrays

```pddl
(:types
    puzzle15 - (array 4 4 range)   ; 4×4 2-D array of range integers
    stack    - (array 5 range)     ; 1-D array of 5 range integers
)
(:functions (puzzle) - puzzle15)

; read element (i, j)
(read (puzzle) ?i ?j)

; write element (i, j) to value v (as an effect)
(write ((puzzle) ?i ?j) v)

; constant literal (in :init or :goal)
(= (puzzle) (array.mk ((0 1 2 3)(4 5 6 7)(8 9 10 11)(12 13 14 15))))
```

### Sets

```pddl
(:types
    item    - object
    itemset - (set item)
)
(:functions (basket) - itemset)

; membership test (precondition / goal)
(member ?x (basket))
(not (member ?x (basket)))

; set-valued tests
(subset  (bag1) (bag2))     ; bag1 ⊆ bag2
(disjoint (bag1) (bag2))    ; bag1 ∩ bag2 = ∅

; set-valued expressions (used with assign)
(union      (bag1) (bag2))
(intersect  (bag1) (bag2))
(difference (bag2) (bag1))

; mutation effects
(add    ?x (basket))        ; basket := basket ∪ {x}
(remove ?x (basket))        ; basket := basket \ {x}

; constant literals (in :init or :goal)
(= (basket) (set.mk (apple banana)))
(= (basket) (set.mk ()))    ; empty set
```

### Forall / exists over bounded integer ranges

When the quantified variable has a bounded integer type, the quantifier iterates
over every integer in `[lower, upper]` rather than over a set of objects.
This works in both preconditions/goals and in effect `forall` blocks.

```pddl
(:types
    pancakes - (number 0 4)   ; integer range 0..4
    stack    - (array 5 pancakes)
)
(:functions (pancake_stack) - stack)

; Effect forall: reverse the prefix 0..?f of the stack (simultaneous assignment)
(:action flip
    :parameters (?f - pancakes)
    :precondition ()
    :effect (and
        (forall (?i - pancakes)
            (when (<= ?i ?f)
                (write (pancake_stack) (?i) (read (pancake_stack) (- ?f ?i)))
            )
        )
    )
)

; Precondition exists: check whether any position holds value 0
(exists (?i - pancakes) (= (read (pancake_stack) ?i) 0))
```

After IPAR, `forall (?i - pancakes)` with the `when` guard expands to individual
conditional effects for each value of `?i` in `[0, 4]`, and `exists` expands to a
disjunction.

### Compilation pipelines (in `docs/extensions/domains/compilation_solving.py`)

| Name | Stages | Use case |
|------|--------|----------|
| `up`  | IPAR → ARRAYS | Arrays + bounded-int params → int-typed fluents |
| `uti` | IPAR → ARRAYS → INTEGERS → USERTYPE | Full grounding to classical |
| `sc`  | SETS → COUNT → USERTYPE | Sets → bool fluents → classical |
| `sci` | SETS → COUNT_INT → INTEGERS | Sets with cardinality → integer fluents |

`IPAR` = `IntParameterActionsRemover`.

---

## 2. Data Model Layer

### 2.1 New operator kinds — `unified_planning/model/operators.py`

Added to `OperatorKind`:

```python
# Arrays
ARRAY_CONSTANT = auto()   # constant array literal
ARRAY_READ     = auto()   # (read array i)   → element value
ARRAY_WRITE    = auto()   # (write array i)   → new array value (effect target)

# Sets
SET_CONSTANT   = auto()   # constant set literal  {e1, e2, …}
SET_MEMBER     = auto()   # element ∈ set          → bool
SET_SUBSETEQ   = auto()   # set1 ⊆ set2            → bool
SET_DISJOINT   = auto()   # set1 ∩ set2 = ∅        → bool
SET_CARDINALITY = auto()  # |set|                   → int
SET_ADD        = auto()   # set ∪ {elem}            → set
SET_REMOVE     = auto()   # set \ {elem}            → set
SET_INTERSECT  = auto()   # set1 ∩ set2             → set
SET_UNION      = auto()   # set1 ∪ set2             → set
SET_DIFFERENCE = auto()   # set1 \ set2             → set
```

Updated groupings:

```python
BOOL_OPERATORS  += {SET_MEMBER, SET_SUBSETEQ}      # these return bool
CONSTANTS       += {ARRAY_CONSTANT, SET_CONSTANT}  # these are literals
IRA_OPERATORS   += {SET_CARDINALITY}               # this returns int
```

### 2.2 New FNode predicate methods — `unified_planning/model/fnode.py`

Predicate methods follow the pattern `is_<operator_name>()`:

```python
# Arrays
is_array_constant() -> bool
is_array_read()     -> bool
is_array_write()    -> bool

# Sets
is_set_constant()   -> bool
is_set_member()     -> bool
is_set_subseteq()   -> bool
is_set_disjoint()   -> bool
is_set_cardinality() -> bool
is_set_add()        -> bool
is_set_remove()     -> bool
is_set_union()      -> bool
is_set_intersect()  -> bool
is_set_difference() -> bool
```

### 2.3 Expression manager builder methods — `unified_planning/model/expression.py`

```python
# Constants
Array(value: list)  -> FNode   # wraps a Python list (auto-promotes elements)
Set(value: set)     -> FNode   # wraps a Python set  (auto-promotes elements)
EMPTY_SET()         -> FNode   # cached empty set constant

# Array operations
ArrayRead(array_exp, index_exp)           -> FNode
ArrayWrite(array_exp, index_exp)          -> FNode

# Set predicate operations (return bool)
SetMember(element, set_expr)              -> FNode
SetSubseteq(set_expr1, set_expr2)         -> FNode
SetDisjoint(set_expr1, set_expr2)         -> FNode

# Set numeric operation (returns int)
SetCardinality(set_expr)                  -> FNode  # folds to Int(n) if constant

# Set-valued operations (return set)
SetAdd(element, set_expr)                 -> FNode
SetRemove(element, set_expr)              -> FNode
SetUnion(set_expr1, set_expr2)            -> FNode
SetIntersection(set_expr1, set_expr2)     -> FNode
SetDifference(set_expr1, set_expr2)       -> FNode
```

`auto_promote` was extended to handle Python `list` → `Array(…)` and Python `set` → `Set(…)`.

---

## 3. Type System

### 3.1 `_IntType(lower_bound, upper_bound)` — bounded integer

Already present in the base library, but now surfaced via the parser and used
throughout the compilers. Key property:

```python
t.is_int_type()         # True
t.lower_bound           # int or None
t.upper_bound           # int or None
```

### 3.2 `_ArrayType(size, elements_type)` — fixed-size array

```python
t.is_array_type()       # True
t.size                  # int — number of elements at this dimension
t.elements_type         # Type — element type (may itself be _ArrayType for N-D)
```

An N-D array `(array d1 d2 … dn T)` is represented as nested `_ArrayType`:
`_ArrayType(d1, _ArrayType(d2, … _ArrayType(dn, T) …))`.

### 3.3 `_SetType(elements_type)` — set

```python
t.is_set_type()         # True
t.elements_type         # Type or None (None = empty set type)
```

---

## 4. Type Checker — `unified_planning/model/walkers/type_checker.py`

New walker handlers (registered with `@walkers.handles`):

| Handler | Input operator | Returns |
|---------|---------------|---------|
| `walk_identity_list` | `ARRAY_CONSTANT` | `_ArrayType(size, elements_type)` |
| `walk_identity_set`  | `SET_CONSTANT`   | `_SetType(elements_type)` or `SetType(None)` for empty |
| `walk_array_read`    | `ARRAY_READ`     | `expression.arg(0).type.elements_type` |
| `walk_array_write`   | `ARRAY_WRITE`    | `expression.arg(0).type.elements_type` |
| `walk_member`        | `SET_MEMBER`     | `BOOL` (checks element_type == set.elements_type) |
| `walk_subseteq`      | `SET_SUBSETEQ`   | `BOOL` |
| `walk_disjoint`      | `SET_DISJOINT`   | `BOOL` |
| `walk_cardinality`   | `SET_CARDINALITY` | `IntType(0, None)` |
| `walk_set_to_set`    | `SET_ADD/REMOVE/UNION/INTERSECT/DIFFERENCE` | `SetType(args[1].elements_type)` |

`combine_types` was extended to merge `_ArrayType` and `_IntType` values
(taking the union of bounds for int, and recursing on element types for arrays).

---

## 5. Simplifier — `unified_planning/model/walkers/simplifier.py`

New constant-folding rules for all new operators:

| Walker method | Key simplifications |
|---------------|---------------------|
| `walk_array_read/write` | Pass-through (no folding; index evaluation happens in compilers) |
| `walk_set_member` | `member(_, ∅)` → `False`; both constants → evaluate statically |
| `walk_set_subseteq` | `subset(∅, _)` → `False`; both constants → evaluate |
| `walk_set_disjoint` | either arg `∅` → `True` |
| `walk_set_cardinality` | `∅` → `Int(0)`; constant set → `Int(len)` |
| `walk_set_add` | add to `∅` → singleton; constant set → evaluate |
| `walk_set_remove` | remove from `∅` → `∅`; constant set → evaluate |
| `walk_set_union` | union with `∅` → other operand |
| `walk_set_intersect` | either arg `∅` → `∅` |
| `walk_set_difference` | subtract `∅` → first operand; first is `∅` → `∅` |

> **Known bug:** `walk_set_difference` currently calls `SetIntersection` instead of
> `SetDifference` (copy-paste error). This only affects constant-folding of set
> difference in the simplifier; runtime compilation is unaffected.

---

## 6. Parser — `unified_planning/io/up_pddl_reader.py`

### 6.1 Grammar — type constructors (line ~131)

```python
type_constructor = (
    Group(Keyword("number") + Word(nums) + Word(nums))   # (number lo hi)
    | Group(Keyword("array") + OneOrMore(name | Word(nums)))  # (array d1 … dn T)
    | Group(Keyword("set") + name)                        # (set T)
)
```

Processing in `_parse_types_map`:
- `number lo hi` → `IntType(lo, hi)`
- `array d1 [d2 …] T` → nested `ArrayType(d1, ArrayType(d2, … T))`
- `set T` → `SetType(types_map[T])`

Requirements `:arrays` and `:sets` are recognised alongside standard ones.

### 6.2 Expression stack machine

The parser uses an explicit stack `[(var_bindings, exp, is_True_branch)]`.
`False` branch pushes children; `True` branch pops results and builds an FNode.

#### 6.2.1 False branch — new handlers

```python
# Array constant (terminal — no children)
elif exp[0].value == "array.mk":
    def _parse_array_content(group):     # recursive for N-D
        if isinstance(group[0].value, ParseResults):
            return [_parse_array_content(group[k]) …]
        return [int(group[k].value) …]
    elements = _parse_array_content(exp[1])   # 1-group form
    # OR elements = [_parse_array_content(exp[k]) …]  # multi-group form
    solved.append(self._em.Array(elements))

# Set constant (terminal — no children)
elif exp[0].value == "set.mk":
    for token in elems_group:
        if problem.has_object(token):   elements.add(ObjectExp(…))
        else:                           elements.add(int(token))
    solved.append(self._em.Set(elements))

# Array read (non-terminal)
elif exp[0].value == "read":
    stack.append((var, exp, True))
    for i in range(1, len(exp)):
        stack.append((var, exp[i], False))

# Set predicate / value operations (non-terminal)
elif exp[0].value in ("member","subset","disjoint","cardinality",
                       "union","intersect","difference"):
    stack.append((var, exp, True))
    for i in range(1, len(exp)):
        stack.append((var, exp[i], False))
```

#### 6.2.2 True branch — new handlers

```python
# Array read: chain one ArrayRead per index
elif exp[0].value == "read":
    args = [solved.pop() for _ in range(len(exp)-1)]
    result = args[0]                  # base fluent expression
    for idx in args[1:]:
        result = self._em.ArrayRead(result, idx)
    solved.append(result)
    # args[0] is the LAST pushed = first pushed child = fluent
    # (stack pops LIFO, so push order exp[1]…exp[N] → pop order exp[N]…exp[1]
    #  → `[solved.pop() for …]` collects them back as exp[1], exp[2], …, exp[N])

# Set operations
elif exp[0].value == "member":
    element, set_expr = solved.pop(), solved.pop()
    solved.append(self._em.SetMember(element, set_expr))

elif exp[0].value == "subset":
    set1, set2 = solved.pop(), solved.pop()
    solved.append(self._em.SetSubseteq(set1, set2))

elif exp[0].value == "disjoint":
    set1, set2 = solved.pop(), solved.pop()
    solved.append(self._em.SetDisjoint(set1, set2))

elif exp[0].value == "cardinality":
    solved.append(self._em.SetCardinality(solved.pop()))

elif exp[0].value == "union":
    set1, set2 = solved.pop(), solved.pop()
    solved.append(self._em.SetUnion(set1, set2))

elif exp[0].value == "intersect":
    set1, set2 = solved.pop(), solved.pop()
    solved.append(self._em.SetIntersection(set1, set2))

elif exp[0].value == "difference":
    set1, set2 = solved.pop(), solved.pop()
    solved.append(self._em.SetDifference(set1, set2))
```

### 6.3 Effect parsing — new operators

#### `write` — three syntactic forms

```python
# 1-D: (write (fluent args) (index) value)   — 4 tokens
array_node  = parse(exp[1])
index_node  = parse(exp[2])
value_node  = parse(exp[3])
target = ArrayWrite(array_node, index_node)

# N-D: (write ((fluent args) i1 … iN) value) — 3 tokens, nested target
node = parse(target_seq[0])                  # base fluent
for k in 1 … N-1:
    node = ArrayRead(node, parse(target_seq[k]))   # peel outer dimensions
target = ArrayWrite(node, parse(target_seq[N]))    # write at final index

# Scalar: (write (fluent) value)             — 3 tokens, flat target
target = parse(exp[1]);  value_node = parse(exp[2])
```

All forms produce `act.add_effect(target, value_node, cond)`.

#### `add` / `remove` — set mutation effects

```python
elif op == "add":
    element_node = parse(exp[1])
    set_node     = parse(exp[2])          # must be a FluentExp
    value_node   = SetAdd(element_node, set_node)
    act.add_effect(set_node, value_node, cond)

elif op == "remove":
    element_node = parse(exp[1])
    set_node     = parse(exp[2])
    value_node   = SetRemove(element_node, set_node)
    act.add_effect(set_node, value_node, cond)
```

### 6.4 Quantifier variable creation — `forall` / `exists`

At three sites in the parser, quantifier variables are created from a type lookup:

| Site | Location |
|------|----------|
| Precondition/goal stack machine (`exists`/`forall` False branch) | line ~578 |
| Effect `forall` for instantaneous actions | line ~1095 |
| Effect `forall` for durative actions | line ~1337 |

**Before:** all quantifier variables were unconditionally created as `Variable`:
```python
new_vars[o] = up.model.Variable(o, t)
```

**After:** if the type is a bounded integer, a `RangeVariable` is created instead:
```python
if t.is_int_type() and t.lower_bound is not None and t.upper_bound is not None:
    new_vars[o] = up.model.RangeVariable(o, t.lower_bound, t.upper_bound)
else:
    new_vars[o] = up.model.Variable(o, t)
```

`RangeVariable` is the signal IPAR uses to distinguish integer-range quantifiers
from object-type quantifiers.  Both `Forall` and `Exists` in the expression
manager accept `RangeVariable` alongside `Variable`.

### 6.5 `:init` section — `array.mk` and `set.mk`

When the RHS of an `=` assignment is `array.mk` or `set.mk`, a special path is taken
instead of general expression parsing:

```python
if constructor == "array.mk":
    def _parse_array_mk(group):   # recursive helper for N-D
        if isinstance(group[0].value, ParseResults):
            return [_parse_array_mk(group[k]) …]
        return [int(group[k].value) …]
    value = _parse_array_mk(rhs[1])           # or multi-group form

else:  # set.mk
    for token in rhs[1]:
        if problem.has_object(token):   elements.add(problem.object(token))
        else:                           elements.add(int(token))
    value = elements   # Python set

problem.set_initial_value(lhs, value)
# set_initial_value calls auto_promote(value) which:
#   list → Array(…)   set → Set(…)
```

---

## 7. `problem.py` — effect fluent helper

`_get_static_and_unused_fluents` used `e.fluent.fluent()` which fails when
`e.fluent` is an `ARRAY_WRITE` node (not a `FLUENT_EXP`). A helper was added:

```python
def _effect_fluent(f_node):
    """Unwind ARRAY_WRITE / ARRAY_READ chains to reach the base FluentExp."""
    current = f_node
    while current.is_array_write() or current.is_array_read():
        current = current.arg(0)
    return current.fluent()
```

`add_effect` was also patched to allow `ARRAY_WRITE` nodes as the effect target
(previously guarded to `FLUENT_EXP` only).

---

## 8. Compiler: `IntParameterActionsRemover` (IPAR)

**File:** `unified_planning/engines/compilers/int_parameter_actions_remover.py`
**CompilationKind:** `INT_PARAMETER_ACTIONS_REMOVING`

### What it does

Grounds every action parameter of bounded-integer type by instantiating one
concrete action per valid integer value.  Arithmetic over the parameter
(e.g., `?r - 1`) is simplified before comparison, so only feasible groundings
are emitted.

### Key changes from base

**`_transform_array_access`** was rewritten to handle N-D arrays:

```python
def _transform_array_access(self, …, node, int_params, instantiations):
    indices = []
    current = node
    # Unwind the chain: ARRAY_READ(ARRAY_READ(fluent, i), j) → [i, j]
    while current.is_array_read() or current.is_array_write():
        idx = transform(current.arg(1), …).simplify()
        if not idx.is_int_constant(): return None
        indices.insert(0, idx.constant_value())
        current = current.arg(0)
    if not current.is_fluent_exp(): return None

    base_name = current.fluent().name.split('[')[0]
    if tuple(indices) not in self.domains.get(base_name, set()):
        return None   # out-of-bounds index — skip this instantiation

    # Build indexed fluent name: base[i0][i1]…
    indexed_name = base_name + "".join(f"[{k}]" for k in indices)
    …
    return indexed_fluent(*new_fluent_args)
```

`self.domains` maps each array fluent name to the set of valid index tuples,
populated during problem transformation.

### Forall / exists expansion over integer ranges

**`_transform_quantifier`** — already existed for object quantifiers; it now also
handles bounded integer variables.  When `effect.forall` (or a precondition
quantifier) contains a `RangeVariable`, IPAR:

1. Calls `_get_range_instantiations` to enumerate all integers in `[lower, upper]`.
2. For each value, appends it to the instantiation tuple and recursively transforms
   the body, substituting the range variable with its concrete integer.
3. Combines the results: `forall` → `And(…)`, `exists` → `Or(…)`.

**`_transform_expression`** — two new cases added to substitute integer values for
range variables that appear inside expressions:

```python
# VARIABLE_EXP wrapping a RangeVariable (the parser always uses VariableExp)
if node.is_variable_exp():
    v = node.variable()
    if isinstance(v, RangeVariable) and v.name in int_params:
        return Int(instantiations[int_params[v.name]])
    return node

# RANGE_VARIABLE_EXP nodes (direct RangeVariableExp usage)
if node.is_range_variable_exp():
    var_name = node.range_variable().name
    if var_name in int_params:
        return Int(instantiations[int_params[var_name]])
    return node
```

> **Why two cases?** The parser calls `VariableExp(range_var)` (creating a
> `VARIABLE_EXP` node) regardless of whether the variable is a `RangeVariable`.
> The old code passed all `VARIABLE_EXP` nodes through unchanged (treating them as
> "free variables that stay symbolic").  Without the first case the range variable
> would never be substituted, so all array indices would evaluate to `None` and
> every grounded action would be pruned.

### Compiler corrections

**`effect.py` — `free_vars_without_duplicates` assertion** — the original assertion
in `Effect.__init__` rejected any forall variable that was not a plain `Variable`:

```python
# Before
assert isinstance(v, up.model.variable.Variable), "Typing not respected"

# After
assert isinstance(v, (up.model.variable.Variable, up.model.range_variable.RangeVariable)), "Typing not respected"
```

When a forall variable is a `RangeVariable` and appears free in the effect
expression (which is always the case, e.g. `?i` appears in `write … (?i) …`), the
iterator must yield it.  Without this fix, `Effect.__init__` raises `AssertionError`
the moment any `forall (?i - int_type)` effect is parsed.

**`_add_single_effect` — missing `is_constant()` guard** — the conditional effects
branch called `value.constant_value()` unconditionally.  When the effect value is a
fluent expression (e.g. `pancake_stack[k]`, the value being read from the array),
this raises `AssertionError` inside `constant_value()`.  Fixed by guarding:

```python
# Before
if (fluent.type.is_int_type() and
        not fluent.type.lower_bound <= value.constant_value() <= fluent.type.upper_bound):

# After
if (fluent.type.is_int_type() and value.is_constant() and
        not fluent.type.lower_bound <= value.constant_value() <= fluent.type.upper_bound):
```

---

## 9. Compiler: `ArraysRemover`

**File:** `unified_planning/engines/compilers/arrays_remover.py`
**CompilationKind:** `ARRAYS_REMOVING`

### What it does

Replaces each array fluent with a family of scalar fluents, one per index position.
A shared `Index` user type with objects `i0, i1, …` is introduced; the indexed
fluents take the index as an extra leading parameter.

```
tower(?p) : array[5, range]
→
tower(i0, ?p) : range
tower(i1, ?p) : range
tower(i2, ?p) : range
tower(i3, ?p) : range
tower(i4, ?p) : range
```

For N-D arrays the index tuple becomes multiple leading parameters.

### Key changes

**`_transform_array_access`** — same N-D unwind logic as IPAR:

```python
indices = []
current = node
while current.is_array_read() or current.is_array_write():
    idx = transform(current.arg(1)).simplify()
    if not idx.is_int_constant(): return None
    indices.insert(0, idx.constant_value())
    current = current.arg(0)
# look up indexed fluent: name[i0][i1]…
indexed_name = base_name + "".join(f"[{k}]" for k in indices)
index_params = _extract_array_indices(new_problem, indexed_name)
return new_fluent(*(index_params + original_fluent_args))
```

**`_add_array_as_indexed_fluent`** — creates `Fluent("name[k]", elem_type, …)` for
every valid index position.

**`_transform_array_comparison`** — expands `(= arr1 arr2)` into a conjunction
`And(= arr1[k] arr2[k] for k in all_positions)`.

**`_get_new_fluent_value`** — converts an `ARRAY_CONSTANT` value in `:init` into
per-index scalar assignments.

---

## 10. Compiler: `IntegersRemover`

**File:** `unified_planning/engines/compilers/integers_remover.py`
**CompilationKind:** `INTEGERS_REMOVING`

### What it does

Converts bounded-integer fluents to object-typed fluents, creating one object
per integer value in the range (e.g., `n0, n1, …, n10` for `Int[0,10]`).
Integer arithmetic constraints (comparisons, arithmetic effects) are enumerated
using OR-Tools CP-SAT.  This yields a purely classical planning problem with
object fluents.

No structural changes were required to this compiler; it consumed the
`_IntType` bounds produced by the parser and IPAR output correctly.

---

## 11. Compiler: `SetsRemover`

**File:** `unified_planning/engines/compilers/sets_remover.py`
**CompilationKind:** `SETS_REMOVING`

### What it does

Encodes every set fluent `s(?params) : set{T}` as a boolean-indexed fluent
`s(?t, ?params) : bool`, where `?t` ranges over all objects of type `T`.
`s(?t, ?params) = True` iff `?t ∈ s(?params)`.

### Transformation table

| Original expression | Compiled form |
|---------------------|---------------|
| `member(o, s(p))` | `s_bool(o, p)` |
| `not member(o, s(p))` | `not s_bool(o, p)` |
| `subset(s1, s2)` | `And(¬s1_bool(o) ∨ s2_bool(o)  for o)` (constant⊆fluent only) |
| `disjoint(s1, s2)` | `And(¬(s1_bool(o) ∧ s2_bool(o)) for o)` |
| `cardinality(s)` | integer helper fluent (maintained by conditional effects) |
| `assign s := union(s1,s2)` | for each `o` in s1∨s2: `s(o):=T` |
| `assign s := intersect(s1,s2)` | for each `o` in s1∧s2: `s(o):=T` |
| `assign s := difference(s1,s2)` | for each `o` in s1∧¬s2: `s(o):=T` |
| `assign s := {o1,o2,…}` | `s(o1):=T, s(o2):=T, s(others):=F` |
| `add elem s` effect | `s_bool(elem, …) := True` |
| `remove elem s` effect | `s_bool(elem, …) := False` |
---
```
