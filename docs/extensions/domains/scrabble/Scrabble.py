import subprocess
from experiments import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Parser
parser = argparse.ArgumentParser(description="Solve Sokoban")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

#instance = subprocess.run(['python3', '/Users/cds26/PycharmProjects/unified-planning/experiments/sokoban/read_instance.py', 'i_1'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#output = instance.stdout.split("---")
#print(instance)
#initial_state = eval(output[0].strip())
#undefined_positions = eval(output[1].strip())
#goal_positions = eval(output[2].strip())
#rows = eval(output[3].strip())
#columns = eval(output[4].strip())

# a board (2-dimensional array) with some tiles of letters. In each action, a subsequent of letters should be
# chosen such that by putting them on the board, it forms a word in the dictionary. A word is an array of letters,
# and the dictionary is a set of arrays of letters.

# ---------------------------------------------------- Problem ---------------------------------------------------------
# Sokoban model with an action per direction of movement.
scrabble_problem = Problem('scrabble_problem')

Letter = UserType('Letter')
none = Object('none', Letter)
T = Object('T', Letter)
O = Object('O', Letter)
P = Object('P', Letter)
L = Object('L', Letter)
E = Object('E', Letter)
scrabble_problem.add_objects([none, T, O, P, L, E])


# board
board = Fluent('board', ArrayType(5, ArrayType(5, Letter)))
# a dictionary is a set of arrays of letters
dictionary = Fluent('dictionary', ArrayType(3, ArrayType(3, Letter)))
# the set of letters from where to choose
hand = Fluent('hand', SetType(Letter))
# the word chosen to put in the board
chosen = Fluent('chosen', SetType(Letter))
time_to_choose = Fluent('time_to_choose')

scrabble_problem.add_fluent(board, default_initial_value=none)
scrabble_problem.add_fluent(dictionary, default_initial_value=none)
scrabble_problem.add_fluent(hand, default_initial_value=set())
scrabble_problem.add_fluent(chosen, default_initial_value=none)
scrabble_problem.add_fluent(time_to_choose, default_initial_value=True)

scrabble_problem.set_initial_value(dictionary, [[T,O,P], [L,E,T], [P,E,T]])
scrabble_problem.set_initial_value(hand, {E,T,P,O})

# triar 3 lletres, no fa falta que formin una paraula
choose_alphabet = InstantaneousAction('choose_alphabet', a=Letter, b=Letter, c=Letter)
a = choose_alphabet.parameter('a')
b = choose_alphabet.parameter('b')
c = choose_alphabet.parameter('c')
# you don't have any pending letters
choose_alphabet.add_precondition(Equals(chosen, set()))
# the letters are in your hand
choose_alphabet.add_precondition(SetSubseteq({a,b,c}, hand))
# we save the 3 letters chosen
choose_alphabet.add_effect(chosen, {a,b,c})
# not choosing anymore
choose_alphabet.add_effect(time_to_choose, False)

# put each letter in a position in the grid
put_alphabet_in_board = InstantaneousAction('put_alphabet_in_board', x=IntType(0,2),
                                            y=IntType(0,2), l=Letter)
x = put_alphabet_in_board.parameter('x')
y = put_alphabet_in_board.parameter('y')
l = put_alphabet_in_board.parameter('l')
# the position in the grid is empty
put_alphabet_in_board.add_precondition(Equals(board[x][y], none))
# one of the letters of the chosen ones
put_alphabet_in_board.add_precondition(SetMember(l, chosen))
# put the letter in a position of the grid
put_alphabet_in_board.add_effect(board[x][y], l)
# remove the letter from the ones chosen
put_alphabet_in_board.add_effect(chosen, SetRemove(l, chosen))

check_words = InstantaneousAction('check_words')
i = RangeVariable('i', 0, 2)
j = RangeVariable('j', 0, 2)
a = Variable('a', Letter)
b = Variable('b', Letter)
c = Variable('c', Letter)
# totes
check_words.add_precondition(Forall(
    # per la casella que estem mirant
    # o no es la que comença (la prèvia no es buida ni esta al borde del grid
    # o les seguents (2) han de tenir sentit...
        # horitzontal
    Or(
        Not(Equals(board[i][j-1], none)), And(Equals(board[i][j], none)),
    And(Not(Equals(board[i][j], none)), Or(Equals(board[i-1][j], none), Equals(i, 0))),
    i,j)))
check_words.add_effect(time_to_choose, True)





assert compilation in ['sets-arrays'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(scrabble_problem, solving, compilation)