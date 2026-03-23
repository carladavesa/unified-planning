from unified_planning.shortcuts import *
from docs.extensions.domains import compilation_solving
import argparse
import math

# Run: python -m docs.extensions.domains.storytellers.Storytellers --compilation sc --solving fast-downward

# A set of storytellers tell their stories to different audiences.
# The storytellers know different (possibly intersecting) sets of stories.
# The audiences begin having heard none of the stories.
# Entertaining an audience leaves it having heard all the stories a storyteller knows.
# A storyteller might tell stories an audience has already heard, adding nothing to the stories the audience knows.

# --- Parser ---
parser = argparse.ArgumentParser(description="Solve Storytellers")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# --- Instance ---
# Fixed benchmark instance with 5 storytellers, 2 audiences, and 20 stories
n_stories = 20

# --- Problem ---
storytellers_problem = unified_planning.model.Problem('storytellers_problem')

Storyteller = UserType('Storyteller')
st1 = Object('st1', Storyteller)
st2 = Object('st2', Storyteller)
st3 = Object('st3', Storyteller)
st4 = Object('st4', Storyteller)
st5 = Object('st5', Storyteller)

Audience = UserType('Audiences')
a1 = Object('a1', Audience)
a2 = Object('a2', Audience)

storytellers_problem.add_objects([st1, st2, st3, st4, st5, a1, a2])

objects = []
Stories = UserType('Stories')
for i in range(n_stories):
    objects.append(Object(f's{i + 1}', Stories))
storytellers_problem.add_objects(objects)

known = Fluent('known', SetType(Stories), st=Storyteller)
heard = Fluent('heard', SetType(Stories), a=Audience)
story_set = Fluent('story_set', SetType(Stories))

storytellers_problem.add_fluent(known, default_initial_value=set())
storytellers_problem.add_fluent(heard, default_initial_value=set())
storytellers_problem.add_fluent(story_set, default_initial_value=set())

# initial state
storytellers_problem.set_initial_value(story_set, {*objects})

n_per_st = int(n_stories / 5)
split = 0
for st_n in range(5):
    st_objects = []
    st = storytellers_problem.object(f"st{st_n + 1}")
    for n in range(n_per_st):
        st_objects.append(storytellers_problem.object(f's{split + (n + 1)}'))
    split += n_per_st

    storytellers_problem.set_initial_value(known(st), {*st_objects})

# --- Actions ---
entertain = InstantaneousAction('entertain', st=Storyteller, a=Audience)
st = entertain.parameter('st')
a = entertain.parameter('a')
entertain.add_effect(heard(a), SetUnion(heard(a), known(st)))
storytellers_problem.add_action(entertain)

# --- Goals ---
a_var = Variable('a_var', Audience)
storytellers_problem.add_goal(Forall(
        GE(
            SetCardinality(heard(a_var)), math.ceil(n_stories / 2)
        ),
        a_var
    )
)

a_var2 = Variable('a_var2', Audience)
storytellers_problem.add_goal(
   Forall(
       Equals(heard(a_var), heard(a_var2)),
       a_var, a_var2
   )
)

# --- Compile and Solve ---
assert compilation in ['sc', 'sci', 'scin'], f"Unsupported compilation type: {compilation} for this domain!"

compilation_solving.compile_and_solve(storytellers_problem, solving, compilation)