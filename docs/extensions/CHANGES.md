# Branch `pddl-extension` — Full Change Log vs `master`

This document describes **every change** introduced in the `pddl-extension` branch
relative to `master`, explains *why* each change was needed, and describes the
consequence of omitting it. Changes are grouped by subsystem, following the
natural dependency order: grammar → parser → IR model → tree walkers → compilers
→ domain files → tooling.

---

## 0. Introduction: Syntax, Data Model, and Type System


Before diving into the individual changes, this section gives a broad mental map of
the three foundational layers that the extension touches: the **PDDL surface syntax**
(what the domain author writes), the **IR data model** (how the library represents
expressions in memory), and the **type system** (how the library classifies values
and checks consistency). Every code change in the branch can be understood as an
extension to one or more of these three layers.

---

### 1.1. PDDL Surface Syntax

#### Type declarations

Standard PDDL declares types by naming a parent type:

```pddl
(:types location truck - object)
```

This extension introduces *compound type constructors* as parents. Instead of a bare
name, the parent can be a parenthesised constructor expression:

```pddl
(:types
    size     - (number 0 3)          ; integer restricted to [0, 3]
    tile     - (number 0 15)         ; integer restricted to [0, 15]
    temp     - (number 10 40)        ; integer restricted to [10, 40] — non-zero lower bound
    floor    - (number 2 8)          ; integer restricted to [2, 8]   — non-zero lower bound
    row      - (array 4 tile)        ; 1-D fixed-size array of 4 tile integers
    board    - (array 4 4 tile)      ; 2-D array: 4 rows × 4 columns of tile
    itemset  - (set item)            ; unordered collection of item objects
)
```

Three constructors are available:

| Constructor | Syntax | Meaning |
|---|---|---|
| Bounded integer | `(number lo hi)` | Finite integer range `[lo, hi]` (inclusive); `lo` may be any non-negative integer |
| Array | `(array d₁ d₂ … dₙ T)` | N-dimensional fixed-size array; last token is element type |
| Set | `(set T)` | Unbounded unordered collection of elements of type `T` |

> **Range bounds:** Both `lo` and `hi` must be non-negative integers (the grammar
> uses `Word(nums)` which matches decimal digits only). Negative lower bounds such
> as `(number -5 5)` are not currently supported and will raise a parse error.

Types declared with these constructors can be used as fluent return types, as
parameter types for actions, and as quantifier variable types.

---

#### Array expressions and effects

Array fluents hold an entire array as their value. Individual elements are accessed
and mutated with `read` and `write`:

```pddl
; Read one element (usable anywhere an expression is expected)
(read (board ?a) ?i ?j)          ; element at position [?i][?j] of board(?a)
(read (stack) ?k)                 ; element at position ?k of a 1-D stack

; Write one element (as an effect)
(write (stack) (?k) new_val)      ; 1-D: set stack[?k] := new_val
(write ((board ?a) ?i ?j) val)   ; N-D: set board(?a)[?i][?j] := val
```

The 1-D and N-D write syntaxes differ in their grouping. For 1-D, the index is a
separate parenthesised group after the fluent. For N-D, the entire target (fluent +
all indices) is wrapped in an outer group, because a bare `(write (f) i j v)` would
be ambiguous — the parser cannot tell where indices end and the value begins without
the extra grouping.

Array constants appear in the `:init` section using `array.mk`, or with the
`array.empty` shorthand for the zero-element case:

```pddl
(= (stack) (array.mk (3 0 2 1 4)))                    ; 1-D: 5 elements
(= (board) (array.mk ((0 1 2 3)(4 5 6 7)(8 9 10 11))) ; 2-D: 3 rows of 4
(= (stack) array.empty)                                ; empty array (shorthand)
```

Both shorthands (`array.empty`, `set.empty`) are also valid inside preconditions,
goals, and effects anywhere the full constructor form is accepted.

---

#### Set expressions and effects

Set fluents hold an unordered collection of objects or integers. The set operators
cover membership testing, structural comparisons, arithmetic on cardinality, and
value-returning operations:

```pddl
; Membership and structural tests (preconditions / goals)
(member ?x (basket))              ; ?x ∈ basket
(subset (bag1) (bag2))            ; bag1 ⊆ bag2
(disjoint (bag1) (bag2))          ; bag1 ∩ bag2 = ∅

; Numeric function on a set
(cardinality (basket))            ; |basket| — usable in arithmetic comparisons

; Set-valued expressions (right-hand side of assign, or passed to other operators)
(union      (bag1) (bag2))        ; bag1 ∪ bag2
(intersect  (bag1) (bag2))        ; bag1 ∩ bag2
(difference (bag1) (bag2))        ; bag1 \ bag2

; Mutation shorthand effects (equivalent to assign + set expression)
(add    ?x (basket))              ; basket := basket ∪ {?x}
(remove ?x (basket))              ; basket := basket \ {?x}

; Constant literals
(= (basket) (set.mk (apple banana)))
(= (basket) (set.mk ()))          ; empty set (long form)
(= (basket) set.empty)            ; empty set (shorthand)
```

---

#### Count expressions

`count` takes a list of boolean sub-expressions and returns how many of them are
true. It appears wherever a numeric expression is expected, typically inside an
arithmetic comparison:

```pddl
; At least 2 of the 3 lights are on
(>= (count (lit ?a) (lit ?b) (lit ?c)) 2)

; Exactly 1 sensor is active
(= (count (active s1) (active s2) (active s3)) 1)

; Symmetric: 3 must be true on both sides
(= (count (flag ?a) (flag ?b) (flag ?c)) (count (done d1) (done d2) (done d3)))
```

Two compilers are available for eliminating `count`:

| Compiler | `CompilationKind` | Strategy |
|---|---|---|
| `CountRemover` | `COUNT_REMOVING` | Expands `count op k` into a DNF boolean formula over all k-combinations of the arguments |
| `CountIntRemover` | `COUNT_INT_REMOVING` | Introduces one `IntType(0,1)` helper fluent per count term; rewrites the comparison as an arithmetic sum |

The DNF expansion is exact but produces formulas that grow as C(n,k), making it
better suited for small counts or small argument lists. The integer-fluent approach
keeps the formula compact but adds state variables and maintenance effects to every
action that can change a tracked fluent.

---

#### Quantifiers over bounded integer types

When a `forall` or `exists` variable is typed with a bounded integer type, the
quantifier iterates over every integer in `[lower, upper]` rather than over objects.
The lower bound can be any non-negative integer — it does not have to be 0:

```pddl
(:types idx  - (number 0 3))   ; iterates 0, 1, 2, 3
(:types midx - (number 2 5))   ; iterates 2, 3, 4, 5  (non-zero lower bound)

; Precondition: check that no cell in column ?c holds the robot
(forall (?r - idx)
    (not (= (robot_at) (read (card_at) ?r ?c)))
)

; Effect: shift every cell of column ?c upward by one
(forall (?r - idx)
    (when (>= ?r 1)
        (write ((card_at) (- ?r 1) ?c) (read (card_at) ?r ?c))
    )
)
```

The `when` guard is essential when the quantifier's body contains arithmetic that
can produce out-of-bounds indices (e.g., `(- ?r 1)` at `?r = 0`). The guard ensures
the conditional effect is inactive for the problematic cases, while the compiler
silently prunes them when expanding the quantifier.

---

## 1. `unified_planning/io/up_pddl_reader.py`

This is the largest change in the branch. It touches the grammar definition, the
expression parser (the expression stack machine), and the type-declaration resolver.

### 1.1 Grammar: allow `.` in names

```diff
- name = Word(alphas, alphanums + "_" + "-")
+ name = Word(alphas, alphanums + "_" + "-" + ".")
```

pyparsing's `Word(init_chars, body_chars)` matches a token that starts with one
character from `init_chars` and continues with zero or more characters from
`body_chars`. The original rule accepted letters, digits, `_`, and `-`. Adding `.`
to `body_chars` lets a token like `array.mk` be matched as a single name rather than
being split on the `.`.

Without this change, the lexer sees `array.mk` as `array` (a name), `.` (unknown
character → parse error) and `mk` (another name). pyparsing raises an error before
the grammar even tries to match the higher-level rule.

**Risk of omitting:** Every problem file that uses `array.mk` or `set.mk` in the
`:init` section fails to parse.

---

### 1.2 Grammar: new requirements tokens

```diff
- ":strips :typing ... :continuous-effects"
+ ":strips :typing ... :continuous-effects :arrays :sets :bounded-integers"
```

The `one_of(string)` call splits the string on whitespace and builds a
`MatchFirst` of `Literal` matchers, one per token. Any requirement keyword not in
the list causes a parse error when the grammar tries to match the `require_def`
rule. Adding the three new keywords makes the grammar accept them. Note that
**requirements are not acted upon** — they are parsed and discarded. No feature
is gated behind them at the library level; the parser always accepts the extended
syntax regardless of which requirements are declared.

---

### 1.3 Grammar: `type_constructor` and `type_parent` rules

```python
type_constructor = (
    Group(Keyword("number") + Word(pyparsing.nums) + Word(pyparsing.nums))
    | Group(Keyword("array") + OneOrMore((name | Word(pyparsing.nums))))
    | Group(Keyword("set") + name)
)
type_parent = Group(Suppress("(") + type_constructor + Suppress(")")) | name
```

**pyparsing combinators used:**

- `Keyword("number")` matches the exact string `"number"` only when it appears as a
  complete token (not as a prefix of a longer word). This is different from
  `Literal("number")`, which would match `"number"` at the start of `"number0"`.
- `Word(pyparsing.nums)` matches a non-empty sequence of decimal digit characters,
  producing a single string token like `"0"`, `"10"`, or `"15"`. Both bounds are
  matched with this rule, so `lo` can be any non-negative integer — `(number 2 8)`,
  `(number 10 40)`, etc. are all valid. Negative lower bounds are **not** supported
  (the `-` sign is not part of `pyparsing.nums`).
- `Group(...)` wraps all the matched tokens of its content into a **nested**
  `ParseResults` object. Without `Group`, all tokens would be flattened into the
  parent list. With it, `(number 0 4)` produces `[['number', '0', '4']]` instead
  of `['number', '0', '4']`. This nesting is the mechanism by which the type-
  declaration resolver (§1.9) distinguishes a compound parent from a plain name.
- `OneOrMore(name | Word(pyparsing.nums))` greedily matches one or more tokens that
  are either names or digit sequences. For `(array 4 4 card)` this matches the three
  tokens `['4', '4', 'card']`. The last one is always the element type name; all
  preceding ones are dimension sizes.
- `Suppress("(")` / `Suppress(")")` consume the parentheses without adding them to
  the match output. This is why the resolver only sees the content, not the brackets.

`type_parent` is the union (`|`) of a parenthesised compound constructor and a
plain `name`. pyparsing tries alternatives left-to-right and commits to the first
that succeeds. Since a plain name never starts with `(`, there is no ambiguity.

The outer `Group` in `type_parent` wraps the already-grouped `type_constructor`
result in **one more level** of nesting. After parsing `pancakes - (array 5 stack)`,
`type_line[1]` (the parent field) becomes `[[ ['array', '5', 'stack'] ]]`. The outer
list is the `type_parent` group; its single element is the `type_constructor` group.
The resolver accesses `raw_parent[0]` to strip the outer level and reach the
constructor group directly.

**Risk of omitting:** Any domain using `(number ...)`, `(array ...)`, or `(set ...)`
type parents fails to parse; the `:types` block raises a grammar error.

---

### 1.4 Grammar: propagate `type_parent` into `:types`

```diff
- Group(Group(OneOrMore(name)) + Optional(Suppress("-") + name))
+ Group(Group(OneOrMore(name)) + Optional(Suppress("-") + type_parent))
```

The outer `Group` captures one type declaration line. The inner `Group` captures the
list of names being declared (e.g., `['pancakes', 'stack']`). The `Optional(-
type_parent)` captures the parent. Previously `name` only accepted a plain string;
replacing it with `type_parent` makes the grammar call the new rule when the parser
encounters the parent position.

---

### 1.5 Expression stack machine: `read` operator

The expression parser in `UPPDDLReader` uses an explicit **stack machine** rather
than recursive descent. Understanding the mechanism is prerequisite to understanding
all the changes in §1.6–§1.8.

**How the stack machine works:**

The machine maintains two stacks:
- `stack`: tuples `(var_dict, parse_node, solved: bool)` — expressions waiting to be
  processed.
- `solved`: a list of resolved `FNode` objects, used as a LIFO stack.

Each iteration pops from `stack`. If `solved=False` (first encounter), the node is
pushed back with `solved=True` and all its children are pushed with `solved=False`.
Because the stack is LIFO, the children will be processed before the parent is seen
again. When the parent is popped a second time with `solved=True`, all its children
have already been resolved and are sitting on top of `solved`. The parent handler
pops N items from `solved` (N = number of children) and builds its FNode.

**Crucially, children are pushed left-to-right onto `stack`, so they are processed
right-to-left (LIFO). This means when the parent pops from `solved`, it pops
them back in left-to-right order:**

For `(read (board ?a) ?i ?j)` with three children:

```
Push order onto stack:   exp[1]=(board ?a), exp[2]=?i, exp[3]=?j
Processing order (LIFO): ?j first, ?i second, (board ?a) third
Append order to solved:  solved = [..., ?j_node, ?i_node, board_a_node]
                                                            ↑ top of stack
Pop order (LIFO × 3):    args[0]=board_a_node, args[1]=?i_node, args[2]=?j_node
```

The code for the solved pass:

```python
elif exp[0].value == "read":
    n_children = len(exp) - 1          # 3 for (read arr i j)
    args = [solved.pop() for _ in range(n_children)]
    base_exp = args[0]                  # board(?a) — the array fluent
    if len(args) == 1:
        solved.append(base_exp)         # bare (read fluent) → identity
    else:
        result = base_exp
        for idx in args[1:]:            # [?i_node, ?j_node]
            if not result.type.is_array_type():
                raise SyntaxError(...)
            result = self._em.ArrayRead(result, idx)
        solved.append(result)
```

Tracing the loop for `(read (board ?a) ?i ?j)`:

```
Iteration 1: idx = ?i_node
    result = ArrayRead(board(?a), ?i)        ← type: ArrayType(4, card) → card_row
Iteration 2: idx = ?j_node
    result = ArrayRead(ArrayRead(board(?a), ?i), ?j)  ← type: card
solved.append(result)
```

The type check `result.type.is_array_type()` runs on the **intermediate result**
before each index is applied. For a 2-D array it succeeds twice; for a 3-D array
it would succeed three times. If more indices are provided than there are dimensions,
the check fires after the last dimension is stripped, giving a clear error.

**Risk of omitting:** The `read` token hits the final `else` branch and raises a
`NotImplementedError`, crashing on any domain that reads from an array in a
precondition or goal.

---

### 1.6 Expression stack machine: set operators

```python
elif exp[0].value == "member":
    element = solved.pop()
    set_expr = solved.pop()
    solved.append(self._em.SetMember(element, set_expr))
elif exp[0].value == "subset":
    set1 = solved.pop()
    set2 = solved.pop()
    solved.append(self._em.SetSubseteq(set1, set2))
# ... union, intersect, difference, disjoint, cardinality similarly
```

The `ExpressionManager` already had Python methods for all set operators (they were
added when the Python-API-level set support was built). What was missing was the
PDDL-parser bridge that maps the textual token to the corresponding `_em` call.

**Argument order and LIFO:** For a binary operator like `(member ?p (package_at ?l))`
with two children `exp[1]=?p` and `exp[2]=(package_at ?l)`:

```
Push onto stack:   exp[1]=?p, exp[2]=(package_at ?l)
Process order:     (package_at ?l) first → solved.append(pkg_at_node)
                   ?p second              → solved.append(p_node)
Pop order:         element = solved.pop() = p_node
                   set_expr = solved.pop() = pkg_at_node
```

The LIFO reversal means the last-pushed child (`(package_at ?l)`) is popped first,
so naming conventions matter: `element = solved.pop()` retrieves `?p` (the PDDL-
first argument) because it was pushed last. This is consistent across all binary set
operators.

**Risk of omitting:** Any domain using set predicates (like `dump-trucks`) fails with
`NotImplementedError` when the parser encounters `member`, `subset`, etc.

---

### 1.7 Expression stack machine: `array.mk` and `set.mk` constructors (unsolved pass)

These constructors appear in `:init` assignments: `(= (arr) (array.mk (1 2 3)))`.
Unlike regular sub-expressions, their arguments are **raw literal lists** — not
sub-expressions that need recursive resolution through the stack. They are therefore
handled in the *unsolved* pass rather than being re-queued for a solved pass.

```python
elif exp[0].value == "array.mk":
    def _parse_array_content(group):
        if len(group) == 0:
            return []
        if isinstance(group[0].value, ParseResults):
            # Nested group → recurse (N-D array row)
            return [_parse_array_content(group[k]) for k in range(len(group))]
        # Leaf group → convert each token to int
        return [int(group[k].value) for k in range(len(group))]

    if len(exp) == 2:
        elements = _parse_array_content(exp[1])        # 1-D: single content group
    else:
        elements = [_parse_array_content(exp[k])       # N-D: one group per row
                    for k in range(1, len(exp))]
    solved.append(self._em.Array(elements))
```

`exp[0]` is the `array.mk` token; `exp[1]` (and onwards) are the content groups.
For `(array.mk (3 0 2 1 4))`, pyparsing has already produced `exp[1]` as a nested
`ParseResults` containing five integer tokens. `_parse_array_content` checks whether
the first element of the group is itself a `ParseResults` (meaning "this is a row of
rows") or a plain string (meaning "this is a row of values"). The recursion bottoms
out when every element is a plain integer string.

For a 2-D `(array.mk (row0_tokens) (row1_tokens) ...)`, `exp` has `len > 2`, so
each `exp[k]` for `k ≥ 1` is a row group, and we collect them into a list of lists.

The `set.mk` handler is simpler: it iterates over the group, checks if each token
names a declared object (→ wrap in `ObjectExp`) or is a numeral (→ convert to `int`),
and builds a Python `set`. The `problem.has_object(token)` call is a live lookup into
the problem's object table, which has already been populated before `:init` is parsed.

**Risk of omitting:** Initial array or set values cannot be parsed; problems fail
with `NotImplementedError` at `(:init ...)` time.

---

### 1.8 Expression stack machine: set and `read` operators (unsolved pass redirect)

```python
elif exp[0].value in ("member", "subset", "disjoint",
                      "union", "intersect", "difference", "cardinality"):
    stack.append((var, exp, True))              # re-queue self for solved pass
    for i in range(1, len(exp)):
        stack.append((var, exp[i], False))      # queue children for unsolved pass
elif exp[0].value == "read":
    stack.append((var, exp, True))
    for i in range(1, len(exp)):
        stack.append((var, exp[i], False))
```

This is the **unsolved pass** code — the first time one of these tokens is seen.
The two lines do the same thing as the general `else` branch at the bottom of the
unsolved pass, but they must be written explicitly because the general `else` branch
raises `SyntaxError("Unknown operator")` for any token it does not recognise.

The pattern `stack.append(self, True); for child: stack.append(child, False)` means:
1. Push `self` with `solved=True` so it will be processed *after* its children.
2. Push all children with `solved=False` so each child goes through resolution first.

Because children are pushed in forward order and the stack is LIFO, child `exp[-1]`
is processed first and child `exp[1]` is processed last — which is exactly the right
order to leave them on `solved` for the LIFO pops described in §1.5 and §1.6.

**Risk of omitting:** These operators fall through to the default `else` branch which
raises `SyntaxError("Unknown operator")` regardless of context.

---

### 1.9 Expression stack machine: `count` operator

`count` follows exactly the same two-pass pattern described in §1.8.

**Unsolved pass** (first encounter — queue children):

```python
elif exp[0].value == "count":
    stack.append((var, exp, True))
    for i in range(1, len(exp)):
        stack.append((var, exp[i], False))
```

**Solved pass** (second encounter — combine resolved children):

```python
elif exp[0].value == "count":
    args = [solved.pop() for _ in range(1, len(exp))]
    solved.append(self._em.Count(args))
```

`ExpressionManager.Count` accepts either an unpacked list of boolean expressions or
a single iterable; passing `args` (a Python list) directly uses the iterable form.
The LIFO pop order matches the push order (see §1.5 for the reversal argument), so
`args[0]` corresponds to `exp[1]`, `args[1]` to `exp[2]`, and so on — the same
left-to-right order in which the count arguments appear in the PDDL source.

`count` differs from `exists`/`forall` (§1.5 and 1.8) in that it introduces no
variable bindings. Its arguments are plain boolean sub-expressions (predicates,
fluent comparisons, etc.) that are already valid in the current variable scope.

**Risk of omitting:** `count` falls through to the default `else` branch in both
the unsolved and solved passes, raising `SyntaxError("Unknown operator")` or
`UPUnreachableCodeError`, respectively, on any domain that uses `(count ...)` in a
precondition or goal.

---

### 1.10 Expression stack machine: inline range types in `forall`/`exists`

The quantifier-variable parser previously rebuilt a flat string from the raw tokens and
re-ran `_pp_parameters.parseString` on it. That trick worked for named types
(`?i - pancakes`) but silently broke when the type was a nested group such as
`(number 0 ?f)` — the `ParseResults` object is not a string, so the join raised a
`TypeError` at runtime.

The fix replaces all four `vars_string / _pp_parameters.parseString` blocks with a
single helper `_parse_quantifier_vars` that walks the `CustomParseResults` token list
directly:

```python
def _parse_quantifier_vars(self, exp_vars, act, types_map):
    ...
    if isinstance(type_tok.value, str):
        # named type — look up in types_map (unchanged behaviour)
        t = types_map[type_tok.value]
        ...
    else:
        # inline (number lo hi) — bounds can be literals or ?param references
        lo = act.parameter(s[1:]) if s.startswith("?") else int(s)
        hi = ...
        result[o] = up.model.RangeVariable(o, lo, hi, self._env)
```

This enables the **inline range-bounded quantifier** syntax:

```pddl
; Static bounds
(forall (?i - (number 0 4)) ...)

; Dynamic upper bound — range shrinks to [0, ?f] at ground time
(forall (?i - (number 0 ?f))
    (write (stack) (?i) (read (stack) (- ?f ?i)))
)
```

`RangeVariable` already accepted `Union[int, Parameter]` bounds; the compiler
(`IntParameterActionsRemover._update_range_vars`) already substituted parameter names
into the range strings at instantiation time. The only missing piece was the parser
creating the `RangeVariable` from an inline type declaration.

**Covered call sites:** precondition/goal quantifiers (`_parse_exp` unsolved pass),
instantaneous-action effect `forall` (`_add_effect`), durative-action condition
`forall` (`_add_condition`), and durative-action effect `forall` (`_add_durative_effect`).

**Risk of omitting:** Any domain using `(forall (?x - (number lo hi)) ...)` with an
inline bounded-integer type (rather than a named type defined in `:types`) raises
`TypeError: sequence item N: expected str instance, ParseResults found` at parse time.

---

### 1.11 Type declaration resolver: compound type parents

The resolver runs after the grammar has parsed the entire `:types` block into a
`domain_res["types"]` list. Each element of that list is a `ParseResults` from the
grammar rule:

```
Group(Group(OneOrMore(name)) + Optional(Suppress("-") + type_parent))
```

For a simple declaration like `pancakes - object`, the parse result is:
```
type_line = [['pancakes'], 'object']
type_line[0] = ['pancakes']    ← names being declared
type_line[1] = 'object'        ← plain string parent
isinstance(type_line[1], str)  → True
```

For a compound declaration like `stack - (array 5 pancakes)`, the parse result is:
```
type_line = [['stack'], [[ ['array', '5', 'pancakes'] ]]]
type_line[0] = ['stack']
type_line[1] = [[ ['array', '5', 'pancakes'] ]]   ← ParseResults (two levels deep)
isinstance(type_line[1], str) → False
```

The two layers of nesting come from the two `Group` wrappers in `type_parent`:
the outer `Group` (in `type_parent` itself) and the inner `Group` (in
`type_constructor`). The resolver peels the outer layer with `raw_parent[0]` to
reach the constructor group `['array', '5', 'pancakes']`.

```python
compound_declarations: list = []

for type_line in domain_res.get("types", []):
    raw_parent = type_line[1] if len(type_line) > 1 else None
    for declared_type in type_line[0]:
        if isinstance(raw_parent, str) or raw_parent is None:
            # Original path
            type_declarations[declared_type] = father_name
        else:
            constructor = raw_parent[0]      # strip outer Group level
            compound_declarations.append((declared_type, constructor))
```

After all simple types are resolved and added to `types_map`, the compound
declarations are processed:

```python
for declared_type, constructor in compound_declarations:
    kind = str(constructor[0])           # 'number', 'array', or 'set'
    if kind == "number":
        lo = int(str(constructor[1]))    # lower bound, e.g. 0
        hi = int(str(constructor[2]))    # upper bound, e.g. 4
        types_map[declared_type] = self._tm.IntType(lo, hi)
    elif kind == "array":
        elem_name = str(constructor[-1]) # last token is element type name
        sizes = [int(str(constructor[i])) for i in range(1, len(constructor) - 1)]
        # Build array type from inside out
        array_type = types_map[elem_name]
        for size in reversed(sizes):
            array_type = self._tm.ArrayType(size, array_type)
        types_map[declared_type] = array_type
    elif kind == "set":
        elem_name = str(constructor[1])
        types_map[declared_type] = self._tm.SetType(types_map[elem_name])
```

**N-D array construction (inside-out loop):**

For `grid - (array 4 4 card)`:
- `elem_name = 'card'`
- `sizes = [4, 4]`
- `reversed(sizes) = [4, 4]` (same in this case, but matters for non-square)
- Iteration 1: `array_type = ArrayType(4, card)` — a 1-D array of 4 cards (a column)
- Iteration 2: `array_type = ArrayType(4, ArrayType(4, card))` — a 4-row 2-D grid

The `ArrayType(n, T)` constructor means "an array of `n` elements of type `T`". The
first `size` in `reversed(sizes)` is the **last declared size** (innermost dimension).
Wrapping progressively: the innermost `ArrayType` represents columns, the outermost
represents rows.

For a 3-D array `(array 2 3 4 T)` with `sizes = [2, 3, 4]`:
```
reversed: [4, 3, 2]
step 1: ArrayType(4, T)
step 2: ArrayType(3, ArrayType(4, T))
step 3: ArrayType(2, ArrayType(3, ArrayType(4, T)))
```

Reading `arr[i][j][k]` first strips the outermost type (size 2, by `i`), then the
next (size 3, by `j`), then the innermost (size 4, by `k`), reaching element `T`.

**Why process compound types after simple types:** `ArrayType` needs `types_map[elem_name]`
to already exist. If `card` is declared as a simple `object`-subtype earlier in the
`:types` block, it will be in `types_map` by the time the compound pass runs.

**Risk of omitting:** All compound type declarations silently produce wrong type
names, crashing downstream when the type name is looked up.

---

### 1.11 Effect parser: `write` operator

The effect parser iterates over the children of the `:effect` block. When it
encounters a `write` token (stored in the variable `op`), it dispatches based on the
token count and shape of the expression:

```python
elif op == "write":
    if len(exp) == 4:
        # 1-D array write: (write (array-fluent) (index) value)
        array_node = self._parse_exp(problem, act, types_map, forall_variables, exp[1], complete_str)
        index_node = self._parse_exp(problem, act, types_map, forall_variables, exp[2], complete_str)
        value_node = self._parse_exp(problem, act, types_map, forall_variables, exp[3], complete_str)
        target_node = self._em.ArrayWrite(array_node, index_node)
    elif len(exp) == 3 and isinstance(exp[1][0].value, ParseResults):
        # N-D array write: (write ((array-fluent args) i1 ... iN) value)
        target_seq = exp[1]       # e.g. ((board ?a) ?i ?j)
        n_indices = len(target_seq) - 1
        value_node = self._parse_exp(..., exp[2], ...)
        node = self._parse_exp(..., target_seq[0], ...)   # board(?a)
        for k in range(1, n_indices):
            idx = self._parse_exp(..., target_seq[k], ...)
            node = self._em.ArrayRead(node, idx)           # chain reads
        last_idx = self._parse_exp(..., target_seq[n_indices], ...)
        target_node = self._em.ArrayWrite(node, last_idx)
    else:
        # Scalar write: (write (fluent) val)
        target_node = self._parse_exp(..., exp[1], ...)
        value_node  = self._parse_exp(..., exp[2], ...)
    eff = (target_node, value_node, cond)
    act.add_effect(...)
```

**Distinguishing 1-D vs N-D via `exp[1][0].value`:**

In `(write ((card_at) ?r ?c) val)`, the token structure is:
- `exp[0]` = `write`
- `exp[1]` = `((card_at) ?r ?c)` — a group containing a nested group as its first element
- `exp[2]` = `val`

`exp[1][0]` is the first element of the outer group, which is `(card_at)` — itself a
`ParseResults`. Calling `.value` on a `ParseResults` returns the `ParseResults`
itself. `isinstance(..., ParseResults)` is True, triggering the N-D path.

For the 1-D syntax `(write (tower ?p) (?r) val)`:
- `exp[0]` = `write`, `exp[1]` = `(tower ?p)`, `exp[2]` = `(?r)`, `exp[3]` = `val`
- `len(exp) == 4` triggers the 1-D path before the N-D check is reached.

**Chain construction for N-D:**

For `(write ((card_at) ?r ?c) val)` with `n_indices = 2`:
```
target_seq = [(card_at), ?r, ?c]
node = card_at_fluent_exp        ← parse target_seq[0]
Loop k=1 to n_indices-1=1:
    idx = ?r_node                ← parse target_seq[1]
    node = ArrayRead(card_at_fluent_exp, ?r_node)
last_idx = ?c_node               ← parse target_seq[2]
target_node = ArrayWrite(ArrayRead(card_at, ?r), ?c)
```

The chain `ArrayWrite(ArrayRead(card_at, ?r), ?c)` means: "in the 1-D sub-array that
is row `?r` of `card_at`, write at position `?c`". The IPAR compiler (§11) later
substitutes concrete values for `?r` and `?c` and resolves this to a named fluent.

**Why `ArrayWrite` as the effect target node:** `Effect` stores its target in
`effect.fluent`. Normally this must be a `FLUENT_EXP`. An `ARRAY_WRITE` node is
used instead to encode "the indexed slot of a fluent". The model and compiler layers
know to unwrap `ARRAY_WRITE` chains when they need the base `Fluent` object.

**Risk of omitting:** Any domain using `(write ...)` in effects raises
`NotImplementedError` in the effect parser.

---

### 1.12 Effect parser: `add` and `remove` set mutation operators

```python
elif op in ("add", "remove"):
    element_node = self._parse_exp(..., exp[1], ...)    # e.g. ?p
    set_node     = self._parse_exp(..., exp[2], ...)    # e.g. (package_at ?l)
    if op == "add":
        value_node = self._em.SetAdd(element_node, set_node)
    else:
        value_node = self._em.SetRemove(element_node, set_node)
    eff = (set_node, value_node, cond)
    act.add_effect(...)
```

`(add ?p (package_at ?l))` is parsed as: target fluent = `(package_at ?l)`, new
value = `SetAdd(?p, (package_at ?l))`. This is a standard fluent-assignment effect
with a functional value — no new effect *kind* is needed because `SetAdd` is just an
`ExpressionManager` method that creates a `SET_ADD` FNode. The `SetRemover` compiler
will later expand this into a per-object boolean update.

**Risk of omitting:** Any domain using `add`/`remove` effect operators raises
`NotImplementedError`.

---

### 1.13 Forall/exists variable creation: `RangeVariable` for bounded integer types

This change appears at **three sites** inside the parser:

- Precondition/goal stack machine (quantifier variable binding)
- Effect forall for instantaneous actions
- Effect forall for durative actions

At each site:

```python
# Before
new_vars[o] = up.model.Variable(o, t)

# After
if t.is_int_type() and t.lower_bound is not None and t.upper_bound is not None:
    new_vars[o] = up.model.RangeVariable(o, t.lower_bound, t.upper_bound)
else:
    new_vars[o] = up.model.Variable(o, t)
```

`new_vars` is a `dict[str, Variable | RangeVariable]` mapping PDDL variable names
(without the `?` prefix) to their variable objects. When the parser later encounters
`?i` in an expression, it calls `self._em.VariableExp(new_vars['i'])`, which creates
a `VARIABLE_EXP` FNode with the variable object as its payload.

`IPAR` inspects each variable with `isinstance(var, RangeVariable)` to decide how
to expand a `forall`. If the variable is a plain `Variable`, IPAR treats it as an
object-type quantifier and passes it through unchanged. If it is a `RangeVariable`,
IPAR reads `var.lower_bound` and `var.upper_bound` to generate one action copy per
integer in the range. This is the same mechanism used for bounded-integer action
parameters.

**Why three sites and not one:** The parser has separate code paths for
preconditions/goals (which handle `forall` and `exists` in expressions), effect
`forall` for instantaneous actions, and effect `forall` for durative actions. All
three independently build a local `new_vars` dict.

**Risk of omitting:** `forall` effects over bounded integer types (e.g., the pancake
`flip` action and all labyrinth `rotate` actions) are silently passed through
unexpanded. The planner receives a problem with unresolved quantifier variables,
which may crash or produce wrong plans.

---

### 1.14 Initial-state parser: `array.mk` and `set.mk` in `:init`

```python
if isinstance(rhs.value, ParseResults) and rhs[0].value in ("array.mk", "set.mk"):
    lhs = self._parse_exp(problem, None, types_map, {}, init[1], problem_str)
    constructor = rhs[0].value
    if constructor == "array.mk":
        def _parse_array_mk(group): ...   # mirrors §1.7 _parse_array_content
        value = _parse_array_mk(...)
    else:  # set.mk
        elements = set()
        for i in range(len(elems_group)):
            token = elems_group[i].value
            if isinstance(token, str) and problem.has_object(token):
                elements.add(problem.object(token))
            else:
                elements.add(int(token))
        value = elements
    problem.set_initial_value(lhs, value)
else:
    # Original path
    problem.set_initial_value(
        self._parse_exp(..., init[1], ...),
        self._parse_exp(..., init[2], ...),
    )
```

The detection `isinstance(rhs.value, ParseResults)` checks whether the right-hand
side of `=` is a group (i.e., a parenthesised expression) rather than a bare token.
If it is, `rhs[0].value` reads the first token inside the group — which is the
constructor name `"array.mk"` or `"set.mk"`.

The original code would call `_parse_exp` on the RHS, which would attempt to resolve
`array.mk` as an expression operator. `array.mk` is not a known operator in the
solved-pass dispatch (§1.5), so it falls through to `_transform_generic`, which
recurses into its children and tries to build an FNode with them — but the children
are integer literals inside a group, not sub-expressions with FNode structure. The
result is either a type error or a `None` return that propagates into a crash in
`set_initial_value`.

The short-circuit path builds a plain Python `list` or `set` (not an FNode) and
passes it directly to `set_initial_value`. The `Problem` class already accepts
Python lists and sets for array/set fluent initialisation.

**Risk of omitting:** All problem instances that initialise an array or set fluent
fail to parse with a mysterious error deep inside the expression resolver.

---

### 1.15 Empty-collection shorthands: `array.empty` and `set.empty`

Both keywords are bare atoms (no parentheses required) and are equivalent to their
long forms:

| Shorthand | Equivalent |
|-----------|------------|
| `array.empty` | `(array.mk ())` |
| `set.empty` | `(set.mk ())` |

They are valid in every position that accepts a collection literal:

```pddl
; In :init
(= (stack) array.empty)
(= (basket) set.empty)

; In a goal or precondition
(= (basket) set.empty)

; As an effect value
(assign (basket) set.empty)
(assign (stack) array.empty)
```

**Implementation — two changes in `up_pddl_reader.py`:**

*Expression context (`_parse_exp`, string-handler branch):*

```python
elif exp.value == "array.empty":
    solved.append(self._em.Array([]))
elif exp.value == "set.empty":
    solved.append(self._em.EMPTY_SET())
```

This covers all expression positions (preconditions, goals, effect values). The
tokeniser already accepts the dot-separated name because `.` was added to
`body_chars` in §1.1.

*Initial-state context (`:init` special-case path):*

```python
_rhs_is_empty = isinstance(rhs.value, str) and rhs.value in ("array.empty", "set.empty")
if _rhs_is_constructor or _rhs_is_empty:
    ...
    if _rhs_is_empty:
        value = [] if rhs.value == "array.empty" else set()
```

The `:init` path is kept separate from `_parse_exp` because `set_initial_value`
accepts raw Python `list`/`set` values for collection fluents (they bypass the
FNode type-compatibility check that would otherwise reject an empty constant whose
element type cannot be inferred).

**Risk of omitting:** Writing `(= (basket) set.empty)` raises a `SyntaxError`
("Not able to handle…") because `set.empty` is not a registered operator, fluent,
or object name, so the expression resolver reaches its error fallback.

---

## 2. `unified_planning/model/operators.py`

```diff
+ ARRAY_READ = auto()
+ ARRAY_WRITE = auto()
```

`OperatorKind` is a Python `enum.Enum` where every member represents one kind of
node in the FNode expression tree. `auto()` assigns sequential integer values; the
specific values do not matter as long as they are unique.

Every FNode stores its kind as `self._content.node_type: OperatorKind`. All tree-
walking code (walkers, compilers, printers) dispatches on this field. Adding two new
members here is the minimal change that allows new FNode kinds to exist without
colliding with any existing kind.

**Why separate from `ARRAY_INDEX`:** `ARRAY_INDEX` already existed. It is used by
`ArraysRemover` internally to construct indexed-fluent names like `board[1][2]`. It
was never meant to appear in user-level expression trees. Reusing it for `read`/`write`
would require every downstream pass to decide by context whether an `ARRAY_INDEX`
node represents a user-level access (needs compilation) or a compiler-internal name
(already compiled) — an untenable design.

**Risk of omitting:** Creating `ArrayRead`/`ArrayWrite` FNodes (§3) would fail with
`KeyError` inside `create_node` when it looks up `OperatorKind.ARRAY_READ`.

---

## 3. `unified_planning/model/expression.py`

```python
def ArrayRead(self, array_exp, index_exp):
    return self.create_node(OperatorKind.ARRAY_READ, (array_exp, index_exp))

def ArrayWrite(self, array_exp, index_exp):
    return self.create_node(OperatorKind.ARRAY_WRITE, (array_exp, index_exp))
```

`ExpressionManager.create_node(node_type, args, payload=None)` is the central FNode
factory. It hashes the `(node_type, args, payload)` triple and returns a cached node
if one already exists with the same structure — this is the **hash-consing** (flyweight)
pattern used throughout the library to ensure that structurally identical expressions
are the same object. For example, two calls to `ArrayRead(board(?a), Int(1))` with
the same inputs return the exact same FNode object.

The `(array_exp, index_exp)` tuple becomes `node.args`, accessible as `node.arg(0)`
and `node.arg(1)` by all downstream code.

**Risk of omitting:** Every call to `self._em.ArrayRead(...)` raises `AttributeError`.

---

## 4. `unified_planning/model/fnode.py`

### 4.1 String representation

`FNode.__str__` dispatches on `self.node_type` through a dictionary of lambdas. Each
lambda closes over `self` and formats the node. Adding entries for `ARRAY_READ` and
`ARRAY_WRITE` follows the existing pattern:

```python
OperatorKind.ARRAY_READ:  lambda: f"(read {self.arg(0)} {self.arg(1)})",
OperatorKind.ARRAY_WRITE: lambda: f"(write {self.arg(0)} {self.arg(1)})",
```

`self.arg(0)` and `self.arg(1)` recursively call `__str__` on the children, so the
full expression is printed in nested PDDL-like syntax. For example:

```
ArrayRead(ArrayRead(board(?a), ?i), ?j)  → "(read (read board(?a) ?i) ?j)"
```

Without these entries, `self._node_type_to_str[self.node_type]()` raises `KeyError`.

### 4.2 Predicate methods

```python
def is_array_read(self) -> bool:
    return self.node_type == OperatorKind.ARRAY_READ

def is_array_write(self) -> bool:
    return self.node_type == OperatorKind.ARRAY_WRITE
```

`self.node_type` is stored as `self._content.node_type` where `_content` is the
immutable `FNodeContent` namedtuple. The comparison is an `==` on enum members —
O(1) and cheap.

These predicates are used at multiple downstream sites. Having them as methods rather
than inline `node_type == OperatorKind.X` comparisons:
1. Makes the calling code readable.
2. Avoids importing `OperatorKind` in every file that inspects nodes.
3. Mirrors the existing API (`is_fluent_exp()`, `is_array_constant()`, etc.).

**Risk of omitting:** All downstream code that calls `is_array_read()` or
`is_array_write()` raises `AttributeError`.

---

## 5. `unified_planning/model/effect.py`

### 5.1 Skip nested-fluent check for `ARRAY_WRITE` targets

```python
# Before — always runs
fluents_in_fluent = set(fve.get(fluent))
fluents_in_fluent.remove(fluent)
if fluents_in_fluent:
    raise UPProblemDefinitionError(...)

# After — skipped for ARRAY_WRITE
if not fluent.is_array_write():
    fluents_in_fluent = set(fve.get(fluent))
    fluents_in_fluent.remove(fluent)
    if fluents_in_fluent:
        raise UPProblemDefinitionError(...)
```

`fve.get(node)` is a call to `FreeVarsExtractor.get`, which walks the subtree rooted
at `node` and returns all `FLUENT_EXP` nodes found anywhere inside it. For a plain
effect like `(assign (position ?r) 5)`, `fluent = FluentExp(position, [?r])`, and
`fve.get(fluent)` returns `{FluentExp(position, [?r])}`. After removing `fluent`
itself, the set is empty — no other fluents are referenced in the target, as required.

For an array-write target like `ArrayWrite(board(?a), ?i)`, `fve.get(...)` walks
into `board(?a)` (the first argument of `ARRAY_WRITE`) and finds the `FLUENT_EXP`
for `board`. The result of `fve.get(target)` includes `board`, but `target` itself
is an `ARRAY_WRITE` node — not a `FLUENT_EXP`. The line `fluents_in_fluent.remove(fluent)`
tries to remove the `ARRAY_WRITE` node from a set that contains only `FLUENT_EXP`
nodes, so removal fails (KeyError) or the set is non-empty after removal, triggering
the `UPProblemDefinitionError`.

Skipping the check for `ARRAY_WRITE` is correct: an array-write target necessarily
contains its fluent as a sub-expression by construction, and this is the intended
semantics.

**Risk of omitting:** Every `(write ...)` effect raises `UPProblemDefinitionError`
at construction time.

### 5.2 Widen the `forall` variable type assertion

```python
# Before
assert isinstance(v, up.model.variable.Variable), "Typing not respected"

# After
assert isinstance(v, (up.model.variable.Variable, up.model.range_variable.RangeVariable))
```

`Effect.__init__` has an inner generator function `free_vars_without_duplicates` that
iterates over `self._forall` and for each variable `v` checks whether `v` appears
free in the effect (via `fve.get(fluent) | fve.get(value) | fve.get(condition)`).
If `v` is free, the generator `yield`s it (making it available to callers that need
the set of bound variables) and asserts it is a `Variable` instance.

After §1.12, `self._forall` for a bounded-integer quantifier contains a
`RangeVariable`. `RangeVariable` is a separate class with a parallel interface to
`Variable` but is **not a subclass** of `Variable`. Python's `isinstance` check is
strict: `isinstance(range_var, Variable)` is `False`. The assert fires on the first
bounded-integer forall variable encountered.

Widening to `isinstance(v, (Variable, RangeVariable))` accepts both.

**Risk of omitting:** Every effect with a `forall` over a bounded integer type raises
`AssertionError` when the effect is added to the action.

---

## 6. `unified_planning/model/problem.py`

### Helper `_effect_fluent` and its use in static-fluent analysis

```python
def _effect_fluent(f_node: "up.model.fnode.FNode") -> "up.model.fluent.Fluent":
    current = f_node
    while current.is_array_write() or current.is_array_read():
        current = current.arg(0)
    return current.fluent()
```

`_get_static_and_unused_fluents` builds two sets: `static_fluents` (fluents never
written) and `unused_fluents` (fluents never read). For every action effect `e`, it
calls:

```python
static_fluents.discard(e.fluent.fluent())   # original code
static_fluents.discard(_effect_fluent(e.fluent))  # new code
```

`FNode.fluent()` is only defined for `FLUENT_EXP` nodes — it returns the `Fluent`
object stored in the node's payload. Calling `.fluent()` on any other node type
raises `AttributeError`.

For an array-write effect, `e.fluent` is an `ARRAY_WRITE` node. Its structure is:

```
ARRAY_WRITE
├── arg(0): ARRAY_READ (for N-D) or FLUENT_EXP (for 1-D)
│           ├── arg(0): FLUENT_EXP  ← the base fluent
│           └── arg(1): index
└── arg(1): index
```

The `while` loop descends through the chain by always following `arg(0)` (the
"array" side, not the "index" side) until it reaches a `FLUENT_EXP`. Then
`.fluent()` succeeds.

This helper is called at **six sites** because the containing method iterates over
multiple collections of effects: instantaneous action effects, durative action
effects, durative action continuous effects, event effects, process effects, and
timed effects.

**Risk of omitting:** `_get_static_and_unused_fluents` raises `AttributeError` for
any problem with array-write effects, crashing validation and some compilers.

---

## 7. `unified_planning/model/transition.py`

```python
# Before
if not fluent_exp.is_fluent_exp() and not fluent_exp.is_dot():
    raise UPUsageError(...)

# After
if not fluent_exp.is_fluent_exp() and not fluent_exp.is_dot() and not fluent_exp.is_array_write():
    raise UPUsageError(...)
```

`UntimedEffectMixin.add_effect(fluent, value, condition)` first calls
`auto_promote(fluent, ...)` which converts a plain `Fluent` object into a
`FLUENT_EXP` FNode if needed. It then validates that the result is one of the
accepted target forms.

When the parser calls `act.add_effect(target_node, value_node, cond)` with an
`ARRAY_WRITE` target, `auto_promote` does nothing (the input is already an FNode).
The condition `not is_fluent_exp() and not is_dot()` evaluates to `True`, triggering
the error.

Adding `not fluent_exp.is_array_write()` short-circuits the check for array-write
targets, allowing them through to the `Effect` constructor.

**Risk of omitting:** `act.add_effect(array_write_target, ...)` raises `UPUsageError`
before the effect is even constructed.

---

## 8. `unified_planning/model/walkers/simplifier.py`

```python
def walk_array_read(self, expression, args, **kwargs):
    return self.manager.ArrayRead(args[0], args[1])

def walk_array_write(self, expression, args, **kwargs):
    return self.manager.ArrayWrite(args[0], args[1])
```

`Simplifier` is a `DagWalker`: it visits each unique FNode once (DAG, not tree —
shared subexpressions are not re-visited). For each node it calls the method
registered for its `OperatorKind`. The registration is automatic: method names
matching `walk_<lower_operator_kind>` are discovered by the metaclass. `ARRAY_READ`
→ `walk_array_read`, `ARRAY_WRITE` → `walk_array_write`.

The `args` parameter contains the **already-simplified** versions of the node's
children (this is the bottom-up DAG traversal). The implementation reconstructs the
node from simplified children. For `ArrayRead(arr, idx)`, if `arr` simplifies to
`arr'` and `idx` simplifies to `Int(3)`, the result is `ArrayRead(arr', Int(3))`.

No deeper simplification is possible: unlike `Plus(Int(2), Int(3)) → Int(5)`, array
access cannot be statically evaluated without knowing the array's concrete values
(which are a runtime state).

**Risk of omitting:** Calling `.simplify()` on any expression containing `ARRAY_READ`
or `ARRAY_WRITE` falls to `walk_error`, raising `NotImplementedError`.

---

## 9. `unified_planning/model/walkers/type_checker.py`

```python
@walkers.handles(OperatorKind.ARRAY_READ)
def walk_array_read(self, expression, args, **kwargs):
    return expression.arg(0).type.elements_type

@walkers.handles(OperatorKind.ARRAY_WRITE)
def walk_array_write(self, expression, args, **kwargs):
    return expression.arg(0).type.elements_type
```

Here the `@walkers.handles` decorator is used instead of method-name discovery,
which is an equivalent but more explicit registration mechanism.

The `TypeChecker` walker returns the **type** of each node (not the node itself).
`expression.arg(0)` is the array-typed subexpression. `.type` retrieves its
pre-computed type (set by the type checker when the array fluent was declared).
For `ArrayType(n, T)`, `.elements_type` returns `T`.

For a 2-D case: if `board(?a)` has type `ArrayType(4, ArrayType(4, card))`, then
`ArrayRead(board(?a), ?r)` has type `ArrayType(4, card)` (a row), and
`ArrayRead(ArrayRead(board(?a), ?r), ?c)` has type `card` (a single element).

For `ARRAY_WRITE`, the "type" is defined as the element type because `ArrayWrite(arr, i)`
represents a write handle to a slot of element type `T` — the value written must
have type `T`.

**Risk of omitting:** The type checker raises `NotImplementedError` for any
expression containing `ARRAY_READ`/`ARRAY_WRITE`, breaking effect type validation
and compiler passes that type-check intermediate expressions.

---

## Why the compilers had to be changed for `read`/`write`

Sections §10 and §11 each add a `_transform_array_access` method to an existing
compiler. A natural question is: could this resolution have been placed elsewhere —
in a shared walker, in the parser, or in a single compiler rather than two?

**Why the parser cannot do it.**  
At parse time, array indices are still symbolic action parameters or quantifier
variables (e.g., `?r`, `?c`). The parser produces `ArrayRead(ArrayRead(board(?a), ?r), ?c)`,
which is the correct IR representation of the intent "read element `?r, ?c` of
`board(?a)`". Resolving this to a flat fluent like `board[2][3](?a)` requires concrete
integer values for `?r` and `?c`, which are not yet available. The parser cannot
perform the resolution.

**Why a shared stateless walker cannot do it.**  
`ArraysRemover`'s `_transform_array_access` calls `new_problem.fluent(indexed_name)`
to look up the pre-declared per-slot fluent. Walkers (`Simplifier`, `TypeChecker`,
`IdentityDAGWalker`) operate on FNode trees in isolation — they are not passed a
`Problem` object. They cannot look up whether `board[2][3]` exists, and they cannot
create new fluents if it does not. The operation is fundamentally problem-scoped, not
expression-scoped.

**Why the same `_transform_array_access` cannot be shared between IPAR and
`ArraysRemover`.**  
The two compilers encounter `ARRAY_READ`/`ARRAY_WRITE` at different levels of
concreteness:

- **IPAR** runs first. Its indices are still symbolic range variables. Its method
  must substitute `?r` → `Int(2)` via `int_params`/`instantiations` before it can
  read the index value. It also does bounds checking (`tuple(indices) not in
  self.domains[base_name]`) to prune out-of-range action copies silently.
- **`ArraysRemover`** runs after IPAR. Its indices are already concrete `Int` constants.
  It skips the substitution step entirely and goes straight to the indexed-fluent
  lookup in the new problem's fluent table.

The signatures and semantics of the two methods differ enough that sharing them
would require threading IPAR-specific state (`int_params`, `instantiations`,
`self.domains`) into a method that `ArraysRemover` never needs. Keeping them
separate keeps each compiler self-contained and correct for its stage in the
pipeline.

**Why the generic `_transform_expression` fallback would silently produce wrong output.**  
Both compilers share a generic fallback that calls `create_node(node_type, new_args)`:
it rebuilds the current node from its (recursively transformed) children. For a node
type it does not recognise, it keeps the node kind unchanged and only substitutes
values in the children.

Applied to `ArrayRead(board(?a), Int(2))` in `ArraysRemover`, the generic path would
produce another `ARRAY_READ` node with the same children (already concrete) — an
`ARRAY_READ` FNode, not a `FLUENT_EXP` for `board[2](?a)`. The indexed-fluent
representation that the rest of the pipeline expects never materialises, and no error
is raised. The planner receives a problem containing unresolved `ARRAY_READ` nodes,
which crash or silently misfire at evaluation time.

In summary: `ARRAY_READ`/`ARRAY_WRITE` resolution is a *problem-scoped, stage-
specific* operation. It cannot be moved to parse time (indices are symbolic), cannot
be shared across compilers (different concreteness), and cannot be handled by the
generic fallback (which would silently pass the nodes through unresolved).

---

## 10. `unified_planning/engines/compilers/arrays_remover.py`

### 10.1 New `_transform_array_access` method

```python
def _transform_array_access(self, old_problem, new_problem, node):
    indices = []
    current = node
    while current.is_array_read() or current.is_array_write():
        idx = self._transform_expression(old_problem, new_problem, current.arg(1))
        if idx is None or not idx.is_int_constant():
            return None
        indices.insert(0, idx.constant_value())   # prepend to maintain left-to-right order
        current = current.arg(0)

    if not current.is_fluent_exp():
        return None

    base_name = current.fluent().name.split('[')[0]
    indexed_name = base_name + "".join(f"[{k}]" for k in indices)
    ...
    return new_fluent(*(index_params + new_args))
```

`ArraysRemover` runs after `IntParameterActionsRemover` (IPAR). By this point, all
integer range variables have been replaced with concrete integers. The chain
`ArrayRead(ArrayRead(board(?a), 1), 2)` can therefore be fully resolved.

**Chain unwinding (`indices.insert(0, ...)`):**

The `while` loop unwinds the chain from the **outside in** (each iteration processes
the outermost node). For `ArrayRead(ArrayRead(board(?a), 1), 2)`:

```
Iteration 1: current = ArrayRead(ArrayRead(board(?a), 1), 2)
    idx = transform(arg(1)) = Int(2) → .constant_value() = 2
    indices.insert(0, 2)  →  indices = [2]
    current = arg(0) = ArrayRead(board(?a), 1)

Iteration 2: current = ArrayRead(board(?a), 1)
    idx = transform(arg(1)) = Int(1) → .constant_value() = 1
    indices.insert(0, 1)  →  indices = [1, 2]
    current = arg(0) = board(?a)  ← FLUENT_EXP, loop exits
```

`indices.insert(0, ...)` prepends each newly discovered index so that the final list
is in left-to-right (outermost-first) order: `[1, 2]` means "row 1, column 2".

**Indexed fluent name construction:**

```python
base_name = current.fluent().name.split('[')[0]   # 'board' (strips any prior indexing)
indexed_name = 'board' + '[1]' + '[2]'            # 'board[1][2]'
```

`ArraysRemover` previously expanded array fluent `board` into per-slot fluents named
`board[0][0]`, `board[0][1]`, ..., `board[3][3]`. These exist in `new_problem`. The
lookup `new_problem.fluent(indexed_name)` retrieves the pre-declared per-slot fluent.

**Risk of omitting:** `ARRAY_READ`/`ARRAY_WRITE` nodes reach the array-comparison
check below (§10.2), which misroutes them, producing wrong or crashing output.

### 10.2 Guard in `_transform_expression`

```python
if node.is_array_read() or node.is_array_write():
    return self._transform_array_access(old_problem, new_problem, node)
# (existing) Special case: array fluent comparisons
if node.args and node.arg(0).type.is_array_type():
    return self._transform_array_comparison(new_problem, node)
```

`_transform_array_comparison` handles expressions of the form `(= arr1 arr2)` where
both sides are full array fluents being compared for equality. Its first argument is
`node.arg(0)`, which it expects to be an array-typed fluent expression (not an
`ARRAY_READ` wrapping an array).

An `ARRAY_READ` node has `arg(0)` equal to an array-typed expression (the array
being read), so the condition `node.arg(0).type.is_array_type()` is also true for
`ARRAY_READ`. Without the earlier guard, `ARRAY_READ` would be passed to
`_transform_array_comparison`, which would interpret it as an equality comparison
and produce nonsense.

The guard must appear **before** the array-comparison check because the two
conditions overlap.

---

## 11. `unified_planning/engines/compilers/int_parameter_actions_remover.py`

### 11.1 New `_transform_array_access` method

```python
def _transform_array_access(self, old_problem, new_problem, node, int_params, instantiations):
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
    ...
    if tuple(indices) not in self.domains.get(base_name, []):
        return None
    ...
    return indexed_fluent(*new_fluent_args)
```

IPAR runs first in the pipeline. Its job is to take an action with bounded-integer
parameters (e.g., `move_north(?r ?c - idx)`) and produce one copy per valid
instantiation of those parameters. It stores the expansion state in:

- `int_params: Dict[str, int]` — maps range-variable name to its position in the
  instantiation tuple. E.g., for parameters `(?r, ?c)`: `int_params = {'r': 0, 'c': 1}`.
- `instantiations: Tuple[int, ...]` — the concrete values for this copy. E.g.,
  `(2, 3)` means `?r=2, ?c=3`.

When the method encounters `(read (board ?a) ?r ?c)` under instantiation `?r=2, ?c=3`,
the chain unwinding produces:

```
Iteration 1: current = ArrayRead(ArrayRead(board(?a), ?r), ?c)
    idx = _transform_expression(?c, int_params={'c':1}, instantiations=(2,3))
        → looks up 'c' in int_params → param_index=1
        → returns Int(instantiations[1]) = Int(3)
    indices.insert(0, 3)  →  indices = [3]
    current = ArrayRead(board(?a), ?r)

Iteration 2: current = ArrayRead(board(?a), ?r)
    idx = _transform_expression(?r, ...) = Int(2)
    indices.insert(0, 2)  →  indices = [2, 3]
    current = board(?a)  ← FLUENT_EXP
```

**Domain bounds check:**

```python
if tuple(indices) not in self.domains.get(base_name, []):
    return None
```

`self.domains['board']` is the set of valid index tuples for `board` (e.g.,
`{(0,0),(0,1),...,(3,3)}`). If `(2,3)` is valid, the method proceeds; if not (e.g.,
out-of-bounds arithmetic like `?r - 1 = -1`), it returns `None`, and the calling
code skips this instantiation. This is how IPAR silently prunes invalid index
combinations.

**Why not shared with §10.1:** IPAR's version must substitute symbolic variables
into integers using `int_params`/`instantiations` before checking. `ArraysRemover`'s
version assumes all indices are already concrete integers — the `int_params`
parameter does not even exist in that version's signature.

### 11.2 Guard in `_transform_expression`

Same rationale as §10.2. Without the guard, `ARRAY_READ` falls through to
`_transform_generic`, which calls `create_node(node_type, tuple(new_args))` with
recursively transformed children but without performing domain-bounds checking or
indexed-fluent lookup. The result is a reconstructed `ARRAY_READ` FNode with
substituted children — structurally wrong (it should have been replaced with an
indexed fluent) and not caught by any downstream check.

### 11.3 Handle `VARIABLE_EXP` wrapping a `RangeVariable`

```python
if node.is_variable_exp():
    v = node.variable()
    if isinstance(v, RangeVariable) and v.name in int_params:
        param_index = int_params[v.name]
        result = Int(instantiations[param_index])
        self._expression_cache[cache_key] = result
        return result
    return node
```

The original blanket base case was:

```python
if node.is_constant() or node.is_variable_exp() or node.is_timing_exp():
    return node
```

This returned all `VARIABLE_EXP` nodes unchanged, which is correct for object-type
variables (which remain symbolic until the classical planner grounds them). But after
§1.12, a `forall (?i - idx)` produces a `VARIABLE_EXP` whose payload is a
`RangeVariable`. This node should be substituted with `Int(k)` just like a
`PARAMETER_EXP` for a bounded-integer parameter.

The check `isinstance(v, RangeVariable) and v.name in int_params` distinguishes:
- `v` is a `RangeVariable` → substitutable, look it up in `int_params`
- `v` is a plain `Variable` → leave unchanged

`int_params[v.name]` gives the index into `instantiations`. For `?i` mapped to
position 2 and `instantiations = (1, 3, 2)`, the substitution is `Int(2)`.

### 11.4 Handle `RANGE_VARIABLE_EXP` directly

```python
if node.is_range_variable_exp():
    var_name = node.range_variable().name
    if var_name in int_params:
        return Int(instantiations[int_params[var_name]])
    return node
```

`RANGE_VARIABLE_EXP` nodes store their variable in the FNodeContent **payload**
(not in `args`). The `_transform_generic` method builds a new node from transformed
children via `create_node(node_type, tuple(new_args))`. Since `args` is empty for a
`RANGE_VARIABLE_EXP`, `new_args` is empty, and `create_node` gets no useful
information — it creates a broken node or returns `None`.

The explicit handler reads the variable from the payload directly and performs the
substitution.

### 11.6 Bug fix: whole-array fluent references in goals dropped by `_transform_fluent_exp`

#### What broke and when

Commit **`e00fc444`** ("Handling RangeVariables also in goals", Apr 2026) extended
`_compile` with two changes:

1. `new_problem.clear_goals()` — goals are now explicitly cleared before rewriting.
2. A new `_transform_goals` method that runs every goal through `_transform_expression`.

**Before this commit**, `_compile` cloned the problem and cleared only actions and
axioms. Goals survived in the clone unchanged and were passed directly to the next
stage (`ARRAYS_REMOVING`), which knew how to expand them. The `uti` pipeline (`IPAR →
ARRAYS_REMOVING → INTEGERS_REMOVING → USERTYPE_FLUENTS_REMOVING`) worked correctly
for all existing domains.

**After this commit**, every goal is passed through `_transform_expression` inside
IPAR. For goals that contain array fluents in a *whole-array* role — e.g.,

```pddl
(:goal (= (pancake_stack) (array.mk (0 1 2 3 4))))
```

the expression `Equals(pancake_stack(), ArrayConstant(...))` is processed as follows:

1. `_transform_expression` is not `is_constant`, not `is_variable_exp`, not
   `is_fluent_exp`, not `is_array_read`/`is_array_write`, not a quantifier.
2. It falls through to `_transform_generic`, which recursively transforms each child.
3. The first child is `pancake_stack()` — a `FLUENT_EXP`. `_transform_expression`
   detects `is_fluent_exp()` and delegates to `_transform_fluent_exp`.
4. Inside `_transform_fluent_exp`, `old_fluent.type.is_array_type()` is `True`.
5. `_extract_array_indices` is called on `node.fluent()`. The fluent name is
   `'pancake_stack'` — no `[k]` brackets. The regex `r'\[(.*?)\]'` finds nothing, so
   `indices = []`.
6. The bounds check: `tuple([]) = ()` is tested against `self.domains['pancake_stack']`,
   which contains only element-level tuples like `{(0,), (1,), (2,), (3,), (4,)}`.
   `() not in valid_accesses` → **`return None`**.
7. Back in `_transform_generic`, one arg is `None`. `_handle_none_args` for
   `OperatorKind.EQUALS` with a `None` arg returns `None`.
8. Back in `_transform_goals`: `transformed is None` → **goal not added**.

The compiled problem has zero goals. The planner trivially reports "goal already
reached" with an empty plan — or crashes, depending on the solver.

#### Why the commit was needed (but incomplete)

The motivation was correct: PDDL goals *can* contain `RangeVariable`-typed
`forall`/`exists` quantifiers (e.g., `(forall (?i - (number 0 4)) ...)`) that must
be expanded by IPAR before reaching `ARRAYS_REMOVING`. Without calling
`_transform_expression` on goals, those range-variable quantifiers would survive
unexpanded into later stages that do not understand them.

However, the commit did not account for goals that reference array fluents in a
*non-indexed* role. Such a goal has no range variables — `_transform_expression`
has nothing to substitute — yet it was silently destroyed by the bounds check.

Note that **none of the current PDDL extension domains have `RangeVariable`
quantifiers in their goals**. All existing goals use either:
- Whole-array equality: `(= (fluent) (array.mk ...))`
- Indexed reads: `(= (read (fluent) (idx)) value)` — handled by the `is_array_read()`
  branch, which was not broken

The goal transformation is therefore currently harmless for range-variable goals and
harmful only for whole-array goals. But the infrastructure is correct in principle
and should be kept.

#### Why the Python API version was unaffected

In `PancakeSorting.py` (the Python-API equivalent of the same domain):

```python
for i in range(n):
    pancake_problem.add_goal(Equals(pancake[i], i))
```

Goals are added as individual element equalities using the Python subscript operator.
`pancake[0]` creates a `FLUENT_EXP` whose fluent name is already `'pancake[0]'` — the
bracket is part of the name at construction time. `_extract_array_indices` finds
`indices = [0]`, `(0,)` is in the valid domain, and the substitution succeeds.

There are no whole-array fluent references in the Python API goals, so the bug never
surfaces. This is a consequence of the Python API encouraging element-by-element goal
construction, while the PDDL surface syntax naturally allows whole-array comparisons
that are meant to be expanded later.

#### Alternative approaches considered

**Alternative A — Revert `e00fc444` entirely.**  
Since no current PDDL domain uses `RangeVariable` quantifiers in goals, reverting the
commit would restore the pre-bug state. Goals would be cloned through unchanged and
`ARRAYS_REMOVING` would expand them. Downside: the infrastructure for correctly
handling future domains that *do* use range-variable goal quantifiers would be lost,
requiring re-implementation at that point. Fragile against future breakage.

**Alternative B — Expand whole-array equality at parse time.**  
The PDDL reader could recognise `(= (fluent) (array.mk ...))` in the `:goal` block
and immediately expand it into `n` individual `Equals(read(fluent, k), v_k)` goals,
matching the Python API style. This would avoid the IPAR issue entirely since each
goal would only contain indexed reads. Downside: this moves array-expansion logic
into the parser, breaking the clean separation of concerns (parser produces IR;
compilers reduce it). It would also make the parser sensitive to the number of array
dimensions in a way that is currently handled by `ARRAYS_REMOVING`.

**Alternative C — Detect whole-array fluents in `_transform_goals` and pass through.**  
Check `goal.is_equals() and goal.arg(0).type.is_array_type()` before calling
`_transform_expression`, and if true, add the goal unchanged. More targeted than the
chosen fix, but requires duplicating the array-type detection at the call site rather
than at the point of failure.

#### Chosen fix — guard in `_transform_fluent_exp`

```diff
 if old_fluent.type.is_array_type():
+    # Whole-array reference (e.g. `pancake_stack()` in a goal equality) — no
+    # indices encoded in the name yet.  IPAR has nothing to substitute here;
+    # let ARRAYS_REMOVING expand it into element-wise comparisons.
+    if '[' not in node.fluent().name:
+        return node.fluent()(*new_args)
     index_params = self._extract_array_indices(node.fluent(), int_params, instantiations)
```

**Why this is the right fix:**

- `'[' not in node.fluent().name` is the minimal, unambiguous test for "this fluent
  has not yet had any array index bound into its name". IPAR encodes each concrete
  index directly into the fluent name as `fluent[k]` (see `_transform_array_access`
  and `_extract_array_indices`). A name without brackets means no index has been
  resolved yet — the fluent is still a whole-array reference.

- Returning the fluent unchanged with its (possibly transformed) args is exactly the
  right contract: the `new_args` substitution is still applied (for any action-
  parameter args the fluent might carry), and the rest of the expression tree is
  rebuilt correctly. `ARRAYS_REMOVING` then sees the whole-array expression and
  correctly expands it.

- The fix is local to `_transform_fluent_exp`, the precise point of failure, and does
  not affect any other path. In particular:
  - Indexed fluents like `pancake_stack[2]()` still go through
    `_extract_array_indices` as before.
  - `ARRAY_READ`/`ARRAY_WRITE` nodes are already handled by their own branch in
    `_transform_expression` before `_transform_fluent_exp` is reached.
  - Non-array fluents are unaffected (the guard is inside the
    `if old_fluent.type.is_array_type()` block).

- It is forward-compatible: if a future domain introduces a whole-array fluent as an
  action parameter (e.g., `(move ?s - stack)` where `stack` is an array type), the
  guard still correctly passes it through.

### 11.7 Grounding integer-parametered fluents (`_transform_int_param_fluents`)

#### The problem

The PDDL writer (`pddl_writer.py`, line 544) rejects any fluent whose parameters are
not `UserType`:

```python
if param.type.is_user_type():
    out.write(...)
else:
    raise UPTypeError("PDDL supports only user type parameters")
```

This check applies to **every** fluent kind — boolean, numeric, and object-fluent —
regardless of which compilation stages have already run. The problem arises when a
PDDL domain uses a `(number lo hi)` type as a fluent parameter type:

```pddl
(:types idx - (number 0 4))
(:functions
    (val ?i - idx) - idx   ; idx resolves to IntType(0,4), not UserType
    (tmp ?i - idx) - idx
)
```

The parser resolves `idx` to `IntType(0, 4)` (see §1.11). The resulting fluents have
signature `[i: integer[0, 4]]` — an `IntType` parameter, not a `UserType` parameter.

Before this fix, the compilation pipeline for such a domain (`uti` =
`IPAR → ARRAYS_REMOVING → INTEGERS_REMOVING → USERTYPE_FLUENTS_REMOVING`) silently
passed the integer-parametered fluents through all four stages:

- **IPAR** grounded integer *action* parameters and expanded range variables in
  effects, but left fluent signatures unchanged.
- **ARRAYS_REMOVING** found no array fluents to transform.
- **INTEGERS_REMOVING** converted fluent *return types* from `IntType` to
  `UserType('Number')`, but its `_transform_fluents` method copies `fluent.signature`
  unchanged (line 956 of `integers_remover.py`):
  ```python
  new_fluent = Fluent(fluent.name, number_ut, fluent.signature, new_problem.environment)
  #                                            ↑ integer-typed params survive unchanged
  ```
- **USERTYPE_FLUENTS_REMOVING** saw fluents with integer parameters and failed to
  ground them.

The compiled problem still carried `BOUNDED_INT_FLUENT_PARAMETERS` in its kind —
flagging the problem. When the planner attempted to write the problem to PDDL, the
writer crashed at line 544.

#### Why the fix belongs in IPAR, not INTEGERS_REMOVING

`INTEGERS_REMOVING` converts integer-*valued* fluents to `UserType('Number')` fluents.
Extending it to also enumerate integer-*parametered* fluents would conflate two
distinct operations: value-type lifting and parameter grounding. More importantly,
IPAR already has all the machinery needed — it knows the integer domains of every
parameter type, it enumerates Cartesian products of integer ranges to ground actions,
and its `_transform_expression` tree walk already substitutes concrete integer values
into all call sites. Extending IPAR keeps the grounding logic in one place and
ensures it runs early (before any other compiler sees the integer-parametered fluents).

#### What changed

**New method `_transform_int_param_fluents(problem, new_problem)`:**

For each fluent whose signature contains at least one `IntType` parameter, the method:

1. Collects the indices of integer-typed parameters in the signature.
2. Enumerates all combinations of concrete integer values for those parameters.
3. For each combination, creates a new parameterless fluent whose name encodes the
   integer values: `val[i: int[0,4]]` becomes `val_0`, `val_1`, …, `val_4`.
   Non-integer parameters (user-type object params) are preserved in the new fluent's
   signature unchanged — so a mixed fluent like `score[p: player, i: int[0,4]]`
   becomes `score_0(player p)`, `score_1(player p)`, …, `score_4(player p)`.
4. Rewrites the explicit initial values from the original problem using the grounded
   fluent names.
5. Populates `self._int_param_fluents[fluent.name]` — a dict mapping base fluent name
   to `(int_param_indices, {int_vals_tuple → grounded_fluent})`.

Fluents without integer parameters and array fluents are copied unchanged.

**Modified `_transform_fluent_exp`:**

After the existing array-fluent branch, a new check looks up the base fluent name in
`_int_param_fluents`. If found, it extracts the concrete integer values from the
(already-evaluated) transformed arguments and returns the corresponding grounded
fluent:

```python
if fluent_base_name in self._int_param_fluents:
    int_param_indices, grounded = self._int_param_fluents[fluent_base_name]
    int_vals = tuple(new_args[i].constant_value() for i in int_param_indices)
    int_param_set = set(int_param_indices)
    non_int_args = [new_args[i] for i in range(len(new_args)) if i not in int_param_set]
    grounded_fluent = grounded.get(int_vals)
    if grounded_fluent is None:
        return None
    return grounded_fluent(*non_int_args)
```

**Why `new_args` are always concrete constants at this point:**

`_transform_fluent_exp` transforms its children first (the `for arg in node.args` loop
at the top). For an integer-typed argument, the child is either:
- A parameter expression `?f` — substituted to `Int(v)` by the `is_parameter_exp`
  branch of `_transform_expression`.
- A range variable `?i` — substituted to `Int(v)` by the `is_range_variable_exp`
  branch.
- An arithmetic expression `(f - i)` — its children are recursively substituted,
  and `_transform_generic` calls `.simplify()` on the result, collapsing `Int(3) -
  Int(1)` to `Int(2)`.

By the time `_transform_fluent_exp` runs, every integer-typed argument is guaranteed
to be an `Int` constant.

**Modified `resulting_problem_kind`:**

```python
new_kind.unset_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
```

Added alongside the existing `unset_parameters("BOUNDED_INT_ACTION_PARAMETERS")`.
After IPAR, no fluent retains an integer-typed parameter, so the kind flag is
correctly cleared.

**Modified `_compile`:**

Following the same pattern as `ArraysRemover._compile`, the method now clears fluents
and initial values from the cloned problem before calling `_transform_int_param_fluents`
to rebuild them:

```python
new_problem.clear_fluents()
...
new_problem.initial_values.clear()
...
self._transform_int_param_fluents(problem, new_problem)
```

This ensures the grounded fluents replace the originals cleanly, rather than coexisting
with them in the cloned state.

#### Concrete result for `domain_full_bounds.pddl`

The domain declares `val[i: int[0,4]]` and `tmp[i: int[0,4]]` and uses
`forall (?i - (number 0 ?f))` in effects. After IPAR with the new extension:

| Before IPAR | After IPAR |
|---|---|
| `integer[0,4] val[i=integer[0,4]]` | `val_0`, `val_1`, `val_2`, `val_3`, `val_4` (all parameterless) |
| `integer[0,4] tmp[i=integer[0,4]]` | `tmp_0`, `tmp_1`, `tmp_2`, `tmp_3`, `tmp_4` (all parameterless) |
| 2 actions with forall range effects | 10 grounded actions; `flip-copy_3` has effects `tmp_0 := val_3`, `tmp_1 := val_2`, `tmp_2 := val_1`, `tmp_3 := val_0` |
| `val(0) := 3` in init | `val_0 := 3` in init |

The subsequent stages (INTEGERS_REMOVING, USERTYPE_FLUENTS_REMOVING) see only
parameterless fluents with bounded integer *return* types, which they already handle
correctly. The PDDL writer no longer encounters any `IntType` parameter and succeeds.

### 11.5 `is_constant()` guard in `_add_single_effect`

```python
# Before — crashes when value is not a constant
if (fluent.type.is_int_type() and
        not fluent.type.lower_bound <= value.constant_value() <= fluent.type.upper_bound):

# After — safe
if (fluent.type.is_int_type() and value.is_constant() and
        not fluent.type.lower_bound <= value.constant_value() <= fluent.type.upper_bound):
```

`FNode.constant_value()` has the implementation:

```python
def constant_value(self):
    assert self.is_constant(), "..."
    return self._content.payload
```

Calling it on a non-constant node (e.g., `ArrayRead(pancake_stack, Int(2))`) hits the
assert. The `is_constant()` guard short-circuits the entire condition to `False` when
the value is not a constant literal, skipping the bounds check. The effect is then
added unconditionally, which is correct — the bounds check is a static-pruning
optimisation, not a correctness requirement. A runtime planner will verify bounds
when it evaluates the effect.

---

## 12. New domain files

All new domains are placed under `docs/extensions/domains/`. Each domain has a
`pddl-extension/` subdirectory with `domain.pddl` and numbered instance files
(`i0.pddl`, `i1.pddl`, etc.).

### 12.1 Pancake sorting (`pancake-sorting/pddl-extension/`)

> **Note:** This subsection describes the *initial* formulation of the domain (5
> pancakes, conditional `when` guard). The domain was later extended to 10 pancakes
> and the `forall` effect was simplified to use an **inline range type**; see §12.5
> for the current state.

- **Types:** `pancakes - (number 0 4)` (bounded integer), `stack - (array 5 pancakes)`.
- **State:** a single fluent `(pancake_stack) - stack`.
- **Action:** `flip ?f` reverses the prefix of length `?f + 1` in one shot using a
  `forall (?i - pancakes)` effect with a conditional write:

  ```pddl
  (forall (?i - pancakes)
      (when (<= ?i ?f)
          (write (pancake_stack) (?i) (read (pancake_stack) (- ?f ?i)))
      )
  )
  ```

  This is the canonical motivation for bounded-integer `forall`: a simultaneous
  prefix reversal cannot be expressed in standard PDDL without separate per-position
  actions or complex precondition schemas.

- **Instances:** 5 instances (`i0`–`i4`), each with a different initial ordering of
  the stack.

### 12.2 Labyrinth (`labyrinth/pddl-extension/`)

- **Types:** `idx - (number 0 3)` (row/column index), `grid - (array 4 4 card)`
  (2-D board of cards).
- **State:** `(card_at) - grid` (the board layout), `(robot_at) - card`.
- **Actions:** 4 directional moves (`move_north`, etc.) reading the current and
  adjacent card from the 2-D array; 4 rotation actions (`rotate_col_up`,
  `rotate_col_down`, `rotate_row_left`, `rotate_row_right`) that shift an entire
  row or column using `forall (?r - idx)` with conditional writes.

  Example (rotate column upward):
  ```pddl
  (forall (?r - idx)
      (when (>= ?r 1)
          (write ((card_at) (- ?r 1) ?c) (read (card_at) ?r ?c))
      )
  )
  (write ((card_at) 3 ?c) (read (card_at) 0 ?c))
  ```

  The N-D write syntax `(write ((card_at) ?r ?c) val)` is exercised here. The outer
  group `((card_at) ?r ?c)` is the target (array + indices), and `val` is the value.
  There are two PDDL domain files: `domain.pddl` and `1_domain.pddl` (the latter
  uses a different formulation of the same domain for comparison).

- **Instance:** `i1.pddl` and `1_i1.pddl`.

### 12.3 15-puzzle (`15-puzzle/pddl-extension/`)

- **Types:** `size - (number 0 3)` (row/column), `range - (number 0 15)` (tile
  values 0–15), `puzzle15 - (array 4 4 range)`.
- **State:** `(puzzle) - puzzle15`.
- **Actions:** 4 directional tile moves. Each move reads the blank (0) from an
  adjacent cell and writes the tile to that position:

  ```pddl
  (write ((puzzle) (- ?i 1) ?j) (read (puzzle) ?i ?j))
  (write ((puzzle) ?i ?j) 0)
  ```

- **Instances:** 4 instances (`i0`–`i3`).

### 12.4 Dump trucks (`dump-trucks/pddl-extension/`)

- **Types:** `pckg_set - (set package)`, `loc_set - (set location)`.
- **State:** `(package_at ?l) - pckg_set`, `(package_in ?t) - pckg_set`,
  `(connects ?l) - loc_set`.
- **Actions:** `move_truck` (uses `member` in precondition to check connectivity),
  `load_truck` (uses `member`, `cardinality` for capacity check, `remove`/`add`
  effects), `unload_truck` (uses `union` and `difference` to bulk-transfer packages).

  This domain exercises the full set-operator vocabulary: `member`, `cardinality`,
  `add`, `remove`, `union`, `difference`.

- **Instance:** `i0.pddl`.

- **Syntax corrections applied:**
  - `domain.pddl`: The `unload_truck` effect changed empty-set assignment from `(assign (package_in ?t) ())` to `(assign (package_in ?t) (set.mk ()))`. The bare `()` form was not parseable by the extension grammar; the `set.mk` constructor must be used explicitly.
  - `i0.pddl`: Empty-set initialization lines for `(package_at l2)`, `(package_in t1)`, and `(package_in t2)` were commented out in the original. They are now restored, using the `(set.mk ())` form.

### 12.5 Pancake sorting (`pancake-sorting/pddl-extension/`) — current formulation

The domain was extended from 5 to 10 pancakes and the `forall` effect was simplified
to use an **inline range type** instead of a named type plus `when` guard (see §12.1
for the earlier formulation).

- **Types:** `pancakes - (number 0 9)` (bounded integer 0–9), `stack - (array 10 pancakes)`.
- **Domain:** `domain.pddl` — one action `flip(?f - pancakes)` that reverses the
  sub-stack from position 0 up to position `?f`:
  ```pddl
  (:action flip
      :parameters (?f - pancakes)
      :effect (forall (?i - (number 0 ?f))
          (write (pancake_stack) (?i) (read (pancake_stack) (- ?f ?i)))
      )
  )
  ```
  The forall variable `?i` uses an **inline range type** `(number 0 ?f)` whose upper
  bound is the action parameter `?f`. At compile time `IntParameterActionsRemover`
  grounds `flip` once per value in `[0, 9]`, each time instantiating the forall over
  the correct sub-range — producing 10 grounded `flip_0` … `flip_9` actions.

  This replaces the earlier `(forall (?i - pancakes) (when (<= ?i ?f) ...))` pattern:
  using the inline range type makes the bound explicit at the type level rather than
  encoding it as a conditional guard, and avoids the need for a separate named type
  `pancakes` to serve as the iteration domain.

- **Instances:** `i0`–`i4` (five shuffle permutations of a 10-element stack).

- **`long.pddl`:** A dedicated 10-element problem instance with a harder initial
  permutation `[1, 8, 9, 6, 7, 5, 3, 0, 2, 4]` and goal `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.

- **Test:** `test_pancake.py` parses instances and compiles through the
  `up` pipeline (`INT_PARAMETER_ACTIONS_REMOVING` → `ARRAYS_REMOVING`), asserting
  10 grounded `flip_k` actions are produced.

### 12.6 Count test — party lights (`tests/pddl-extension/`)

- **Files:** `domain_count.pddl`, `problem_count.pddl`, `test_count.py`.
- **Purpose:** Minimal domain that exercises the `count` expression in an action
  precondition and verifies the full parse → compile cycle.
- **Types:** `light - object`.
- **Predicates:** `(lit ?l - light)`, `(party-on)`.
- **Actions:**
  - `turn-on ?l` / `turn-off ?l` — toggle a light.
  - `start-party ?a ?b ?c` — starts the party when at least 2 of the 3 chosen
    lights are on:
    ```pddl
    :precondition (and
        (not (= ?a ?b))
        (not (= ?a ?c))
        (not (= ?b ?c))
        (>= (count (lit ?a) (lit ?b) (lit ?c)) 2)
    )
    ```
    The three distinctness guards are required because `?a`, `?b`, `?c` are
    unconstrained object parameters. Without them the planner can bind all three
    to the same lit light, making `count = 3 >= 2` true with only one light on.
- **Problem:** 3 lights (`l1`, `l2`, `l3`); `l1` and `l2` start lit. Minimal plan:
  `start-party(l1, l2, l3)`.

**Test assertions (`test_count.py`):**
1. A `Count` FNode is present inside `start-party`'s preconditions after parsing.
2. `"COUNTING"` appears in `problem.kind.features` before compilation.
3. After `COUNT_REMOVING`, `"COUNTING"` is absent from the kind and no `Count` node
   survives in any action precondition or goal.

---

## 13. New test files

### 13.1 `docs/extensions/domains/tests/pddl-extension/`

Integration test files for the extension. Placed inside `docs/` rather than `tests/`
because they were written incrementally during development and exercise the
extension-specific pipeline rather than the main test suite.

#### PDDL domain and problem files

| Current filename | Previous filename | Contents |
|---|---|---|
| `domain_array.pddl` | `domain.pddl` (renamed) | 1-D array read/write test domain |
| `domain_arrays2d.pddl` | `domain2d.pddl` (renamed) | 2-D array read/write test domain |
| `domain_sets.pddl` | — | Set `member`/`cardinality`/`union`/`intersect`/`difference` test domain |
| `domain_sets2.pddl` | — | Second set test domain (additional set operators) |
| `domain_count.pddl` | — | `count` expression test domain (party lights, see §12.6) |
| `domain_bounded.pddl` | — | Bounded-integer type test domain (thermostat, see §13.2) |
| `domain_full_bounds.pddl` | — | Integer-parametered fluents without arrays (see §11.7) |

Corresponding problem files follow the same naming: `problem_array.pddl`,
`problem_arrays2d.pddl`, `problem_count.pddl`, `problem_bounded.pddl`,
`problem_full_bounds.pddl`.

**Rename rationale:** `domain.pddl` / `domain2d.pddl` were renamed to
`domain_array.pddl` / `domain_arrays2d.pddl` (and similarly for problem files) to
make the file names self-describing and consistent with the growing set of
`domain_{feature}.pddl` test files. The test scripts `test_2d.py` and `test_sets.py`
were updated to reference the new paths.

**`domain_sets.pddl` correction:** The `dump_basket` action effect was simplified
from `(assign (basket) (difference (basket) (basket)))` to `(assign (basket) set.empty)`,
using the shorthand that was already implemented in the parser.

**`domain_full_bounds.pddl`:** A minimal domain with integer-parametered fluents
`(val ?i - idx)` and `(tmp ?i - idx)` (both of type `idx = integer[0,9]`) and two
actions using `forall (?i - (number 0 ?f))` range effects. This domain exists
specifically to validate the `IntParameterActionsRemover` grounding path described
in §11.7 — it is NOT an array domain, so `ARRAYS_REMOVING` is a no-op, and the
interesting work happens entirely in IPAR.

#### Python test scripts

| Script | What it tests |
|---|---|
| `test_2d.py` | 2-D array parse roundtrip |
| `test_sets.py` | Set operator parse roundtrip |
| `test_count.py` | `count` expression parse + `COUNT_REMOVING` compilation (see §12.6) |
| `test_bounded.py` | Bounded-integer type parse + `BOUNDED_TYPES_REMOVING` compilation (see §13.2) |

### 13.2 Bounded-integer type test — thermostat (`tests/pddl-extension/`)

New files: `domain_bounded.pddl`, `problem_bounded.pddl`, `test_bounded.py`.

**Domain:** a thermostat domain exercising `(number lo hi)` types with a *non-zero*
lower bound:
- `temp - (number 15 25)` — current temperature, range `[15, 25]`.
- `setpoint - (number 18 22)` — target temperature, range `[18, 22]`.
- Fluents: `(current-temp ?r - room) - temp`, `(target-temp ?r - room) - setpoint`.
- Actions: `heat-up ?r` (increments `current-temp` by 1), `cool-down ?r` (decrements
  by 1).

**Problem:** 2 rooms (`kitchen`, `living-room`); kitchen starts at 16 (target 20),
living-room starts at 25 (target 22). Minimal plan: heat kitchen four times, cool
living-room three times.

**Test assertions (`test_bounded.py`):**
1. Parser resolves `(number 15 25)` to `IntType(15, 25)` with the correct bounds.
2. `problem.kind.has_bounded_types()` returns `True` before compilation.
3. `BOUNDED_TYPES_REMOVING` compilation succeeds.
4. `compiled.kind.has_bounded_types()` returns `False` after compilation.
5. Compiled `heat-up` preconditions contain explicit range guards for both the lower
   bound (`15`) and the upper bound (`25`).

This domain specifically tests that *non-zero* lower bounds (here `15`, not `0`) are
preserved correctly through the type resolver and the compilation stage.

---

## 14. `unified_planning/engines/compilers/sets_remover.py`

Two bugs in `SetsRemover` were discovered and fixed in commit **`fe37f780`**
("Fixed recursive call for non-set equality expressions", Apr 2026).

### 14.1 Bug fix: `fluent_name` crashes on fluents with more than one argument

**Location:** `_get_cardinality_fluent` (the helper that mints or retrieves the
`card_X` fluent for `cardinality(f(...))` expressions).

**Original code:**
```python
fluent_name = f'card_{old_fluent.name}_{str(*set_expr.args)}' \
    if set_expr.args else f'card_{old_fluent.name}'
```

`str(*set_expr.args)` unpacks the args tuple as positional arguments to `str()`.
Python's built-in `str()` accepts at most one positional argument; passing two or more
raises `TypeError: str expected at most 1 argument, got N`. The conditional `if
set_expr.args` meant the crash path was only exercised for fluents with *at least one*
argument, while zero-argument fluents used the safe branch.

**Fix:**
```python
fluent_name = f'card_{old_fluent.name}_{"_".join(str(a) for a in set_expr.args)}'
```

Joining with `"_"` handles 0, 1, or N arguments consistently. For a single-argument
fluent the result is identical to the old code.

**Risk of omitting:** Any domain that calls `cardinality` on a set fluent with two or
more parameters (e.g., `(cardinality (connections ?a ?b))`) raises `TypeError` inside
`SetsRemover._compile`.

---

### 14.2 Bug fix: overly restrictive assert and missing recursion in `_transform_equality`

**Location:** `_transform_equality`, which handles the case where the `=` operator
appears in an expression that `SetsRemover` is transforming.

**Original code:**
```python
# Guard before dispatching on set vs. non-set equality
assert (left.is_fluent_exp() and (right.is_fluent_exp() or right.is_parameter_exp() or right.is_constant()) or
        (right.is_fluent_exp() and (left.is_fluent_exp() or left.is_parameter_exp() or left.is_constant()))), \
    f"Expression of the form {node} not supported"
```

This assert required at least one side of the equality to be a fluent expression. It
fires for any non-set equality where neither operand is a fluent — for example:

- Object-parameter equality: `(= ?a ?b)` (the "all distinct" guards in §12.6's
  `start-party` action).
- Pure numeric comparisons: `(= ?i (+ ?j 1))`.

When `SetsRemover` is compiling a problem that *also* uses sets elsewhere, these
innocent equalities reach `_transform_equality` through the recursive `_transform_expression`
walk and trigger the assert.

Additionally, the original `else` branch (reached for non-set equalities after the
assert) simply returned the node unchanged without recursing into its children. This
meant that a sub-expression like `(= ?a (count ...))` — where the right side was a
`Count` node that needed transformation — would be silently passed through without
being expanded.

**Fix:**
```python
# No sets equality — remove the assert, recurse into both operands
else:
    new_left = self._transform_expression(old_problem, new_problem, left)
    new_right = self._transform_expression(old_problem, new_problem, right)
    return em.Equals(new_left, new_right)
```

The assert is removed entirely. The else branch now calls `_transform_expression`
recursively on both sides, which is the correct behaviour: any sub-expression that
uses sets (or any other extension feature handled by the walker) will be properly
expanded, while purely numeric or object expressions are returned unchanged by their
own handlers.

**Risk of omitting:** Any problem that uses sets *and* contains at least one plain
`(= expr expr)` equality where neither operand is a set fluent raises `AssertionError`
during compilation. For the `dump-trucks` domain this manifests when an action
compares object parameters (e.g., checking `?t ≠ ?t2`).

---

## Summary table

| File | Change type | Justification |
|---|---|---|
| `io/up_pddl_reader.py` | Feature | Full PDDL surface syntax for arrays, sets, bounded integers, `count`, and inline range quantifiers (§1.9–§1.10) |
| `model/operators.py` | Feature | New `ARRAY_READ` / `ARRAY_WRITE` operator kinds |
| `model/expression.py` | Feature | Factory methods for the two new FNode types |
| `model/fnode.py` | Feature | String repr + predicate methods for the new node types |
| `model/effect.py` (§5.1) | Bug fix | Skip nested-fluent check for array-write targets |
| `model/effect.py` (§5.2) | Bug fix | Widen assert to accept `RangeVariable` in forall effects |
| `model/problem.py` | Bug fix | Handle array-write targets when computing static fluents |
| `model/transition.py` | Bug fix | Accept `ARRAY_WRITE` as a valid `add_effect` target |
| `model/walkers/simplifier.py` | Feature | Walk handlers for `ARRAY_READ` / `ARRAY_WRITE` |
| `model/walkers/type_checker.py` | Feature | Type-inference handlers for `ARRAY_READ` / `ARRAY_WRITE` |
| `engines/compilers/arrays_remover.py` | Feature + Bug fix | N-D array access transformation; guard against misrouting (see §"Why the compilers…") |
| `engines/compilers/int_parameter_actions_remover.py` | Feature + Bug fix | Integer-range variable substitution in array indices; `is_constant()` guard; whole-array goal pass-through (§11.6); grounding of integer-parametered fluents into parameterless variants (§11.7) |
| `engines/compilers/sets_remover.py` | Bug fix | Fluent name crash on multi-argument fluents (§14.1); removed overly restrictive assert and added recursive transformation for non-set equalities (§14.2) |
| `docs/extensions/domains/pancake-sorting/pddl-extension/` | Domain + Test | Pancake-sorting domain extended to 10 elements; `forall` uses inline range type (§12.5); new `long.pddl` instance |
| `docs/extensions/domains/dump-trucks/pddl-extension/` | Bug fix | Empty-set syntax corrected to `(set.mk ())` in domain and problem files (§12.4) |
| `docs/extensions/domains/tests/pddl-extension/` | Tests | Renamed array test files; new `domain_full_bounds.pddl`, `test_bounded.py`, and thermostat domain files (§13.1–§13.2) |
| `docs/extensions/domains/pddl_reader.py` | Fix | Use `UPPDDLReader` directly for extension domains instead of relying on `PDDLReader` fallback |
