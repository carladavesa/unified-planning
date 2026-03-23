"""labyrinth planning domain.

Example:
  python run.py --domain labyrinth --compilation up --solving fast-downward
"""
import math
from typing import Dict, Optional
from unified_planning.model import Action, Expression, Object, RangeVariable
from unified_planning.shortcuts import (
    ArrayType,
    Equals,
    Fluent,
    Int,
    IntType,
    InstantaneousAction,
    MinimizeActionCosts,
    Problem, UserType, Minus, Plus, LE, GE, Forall, Not,
)

from domains.base import Domain

INSTANCES: Dict[str, tuple[int, list[list[int]], list[list[set[str]]]]] = {
    "p4_5_1": (
        4, [[0, 1, 2, 3],[9, 5, 7, 4],[10, 6, 8, 13],[12, 11, 14, 15]], [[{'N', 'E'},{'W', 'E'},{'W', 'S', 'E'},{'N', 'S', 'E'}],[{'E', 'N', 'S', 'W'},{'S', 'E'},{'S', 'W'},{'S', 'W'}],[{'S', 'W'},{'N', 'S', 'W'},{'N', 'W'},{'N', 'E'}],[{'N', 'S'},{'W', 'E'},{'N', 'E'},{'N', 'S', 'W'}]]
    ),
    "p4_5_2": (
        4, [[0, 1, 14, 3], [6, 7, 4, 2], [10, 5, 8, 9], [12, 13, 11, 15]], [[{'E', 'W'}, {'E', 'N', 'W'}, {'S', 'N', 'E'}, {'N', 'S'}], [{'N', 'E'}, {'S', 'E', 'W'}, {'S', 'W'}, {'S', 'N', 'W'}], [{'S', 'N', 'E'}, {'S', 'E'}, {'S', 'N', 'W'}, {'N', 'W'}], [{'S', 'N', 'E'}, {'S', 'W'}, {'E', 'N', 'W'}, {'S', 'E', 'W'}]]
    ),
    "p4_5_3": (
        4, [[0, 1, 14, 3], [4, 5, 2, 7], [10, 11, 8, 6], [12, 13, 9, 15]], [[{'E', 'W'}, {'E', 'W', 'N', 'S'}, {'E', 'W', 'N'}, {'W', 'N', 'S'}], [{'W', 'N', 'S'}, {'E', 'N', 'S'}, {'E', 'S'}, {'E', 'N', 'S'}], [{'E', 'S'}, {'E', 'W', 'N', 'S'}, {'W', 'N'}, {'W', 'N'}], [{'W', 'S'}, {'E', 'W', 'N'}, {'W', 'S'}, {'W', 'S'}]]
    ),
    "p4_5_4": (
        4, [[0, 6, 7, 3], [5, 9, 10, 4], [8, 13, 14, 11], [12, 1, 2, 15]], [[{'W', 'E'}, {'S', 'W'}, {'N', 'W', 'S', 'E'}, {'N', 'W', 'E'}], [{'N', 'W', 'E'}, {'S', 'W'}, {'N', 'S'}, {'S', 'E'}], [{'N', 'S', 'E'}, {'S', 'W'}, {'N', 'S', 'E'}, {'W', 'E'}], [{'N', 'W'}, {'N', 'W', 'S'}, {'N', 'S'}, {'S', 'W'}]]
    ),
    "p4_5_5": (
        4, [[0, 9, 14, 3], [7, 4, 2, 6], [5, 1, 13, 8], [12, 10, 11, 15]], [[{'W', 'S'}, {'W', 'E'}, {'N', 'E'}, {'N', 'W', 'E', 'S'}], [{'N', 'S'}, {'N', 'S'}, {'W', 'E', 'S'}, {'W', 'S'}], [{'N', 'E'}, {'E', 'S'}, {'N', 'E', 'S'}, {'N', 'E'}], [{'N', 'W'}, {'N', 'W', 'S'}, {'N', 'S'}, {'N', 'S'}]]
    ),
    "p4_10_1": (
        4, [[0, 8, 2, 3], [9, 5, 11, 4], [6, 1, 13, 10], [12, 7, 14, 15]], [[{'N', 'E'}, {'N', 'W'}, {'E', 'W', 'S'}, {'N', 'E', 'S'}], [{'N', 'E', 'S', 'W'}, {'E', 'S'}, {'E', 'W'}, {'W', 'S'}], [{'N', 'W', 'S'}, {'E', 'W'}, {'N', 'E'}, {'W', 'S'}], [{'N', 'S'}, {'W', 'S'}, {'N', 'E'}, {'N', 'W', 'S'}]]
    ),
    "p4_10_2": (
        4, [[0, 1, 8, 3], [5, 7, 11, 6], [10, 4, 14, 9], [12, 13, 2, 15]], [[{'E', 'W'}, {'E', 'N', 'W'}, {'S', 'N', 'W'}, {'S', 'N'}], [{'E', 'S'}, {'E', 'S', 'W'}, {'E', 'N', 'W'}, {'E', 'N'}], [{'E', 'N', 'S'}, {'S', 'W'}, {'E', 'N', 'S'}, {'N', 'W'}], [{'E', 'N', 'S'}, {'S', 'W'}, {'S', 'N', 'W'}, {'E', 'S', 'W'}]]
    ),
    "p4_10_3": (
        4, [[0, 13, 9, 3], [14, 2, 1, 4], [10, 7, 5, 6], [12, 11, 8, 15]], [[{'W', 'E'}, {'W', 'E', 'N'}, {'W', 'S'}, {'W', 'N', 'S'}], [{'W', 'E', 'N'}, {'E', 'S'}, {'W', 'E', 'S', 'N'}, {'W', 'N', 'S'}], [{'E', 'S'}, {'E', 'S', 'N'}, {'E', 'S', 'N'}, {'W', 'N'}], [{'W', 'S'}, {'W', 'E', 'S', 'N'}, {'W', 'N'}, {'W', 'S'}]]
    ),
    "p4_10_4": (
        4, [[0, 10, 7, 3], [5, 9, 13, 4], [8, 1, 14, 11], [12, 6, 2, 15]], [[{'E', 'W'}, {'S', 'N'}, {'S', 'E', 'W', 'N'}, {'E', 'W', 'N'}], [{'E', 'W', 'N'}, {'S', 'W'}, {'S', 'W'}, {'S', 'E'}], [{'S', 'E', 'N'}, {'S', 'W', 'N'}, {'S', 'E', 'N'}, {'E', 'W'}], [{'W', 'N'}, {'S', 'W'}, {'S', 'N'}, {'S', 'W'}]]
    ),
    "p4_10_5": (
        4, [[0, 1, 14, 3], [7, 9, 10, 6], [4, 2, 8, 5], [12, 13, 11, 15]], [[{'S', 'W'}, {'S', 'E'}, {'N', 'E'}, {'S', 'W', 'E', 'N'}], [{'S', 'N'}, {'W', 'E'}, {'S', 'W', 'N'}, {'S', 'W'}], [{'S', 'N'}, {'S', 'W', 'E'}, {'N', 'E'}, {'N', 'E'}], [{'N', 'W'}, {'S', 'N', 'E'}, {'S', 'N'}, {'S', 'N'}]]
    ),
    "p4_15_1": (
        4, [[0, 6, 1, 3], [9, 7, 14, 4], [10, 8, 2, 13], [12, 5, 11, 15]], [[{'N', 'E'}, {'W', 'S', 'N'}, {'W', 'E'}, {'S', 'N', 'E'}], [{'W', 'S', 'N', 'E'}, {'W', 'S'}, {'N', 'E'}, {'W', 'S'}], [{'W', 'S'}, {'W', 'N'}, {'W', 'S', 'E'}, {'N', 'E'}], [{'S', 'N'}, {'S', 'E'}, {'W', 'E'}, {'W', 'S', 'N'}]]
    ),
    "p4_15_2": (
        4, [[0, 13, 6, 3], [5, 1, 11, 14], [7, 2, 9, 10], [12, 4, 8, 15]], [[{'E', 'W'}, {'S', 'W'}, {'E', 'N'}, {'S', 'N'}], [{'S', 'E'}, {'E', 'N', 'W'}, {'E', 'N', 'W'}, {'S', 'E', 'N'}], [{'S', 'E', 'W'}, {'S', 'N', 'W'}, {'N', 'W'}, {'S', 'E', 'N'}], [{'S', 'E', 'N'}, {'S', 'W'}, {'S', 'N', 'W'}, {'S', 'E', 'W'}]]
    ),
    "p4_15_3": (
        4, [[0, 1, 5, 3], [7, 11, 8, 14], [10, 13, 9, 6], [12, 2, 4, 15]], [[{'E', 'W'}, {'S', 'E', 'W', 'N'}, {'S', 'E', 'N'}, {'S', 'W', 'N'}], [{'S', 'E', 'N'}, {'S', 'E', 'W', 'N'}, {'W', 'N'}, {'E', 'W', 'N'}], [{'S', 'E'}, {'E', 'W', 'N'}, {'S', 'W'}, {'W', 'N'}], [{'S', 'W'}, {'S', 'E'}, {'S', 'W', 'N'}, {'S', 'W'}]]
    ),
    "p4_15_4": (
        4, [[0, 1, 2, 3], [5, 6, 7, 4], [8, 10, 13, 11], [12, 9, 14, 15]], [[{'E', 'W'}, {'S', 'W', 'N'}, {'S', 'N'}, {'E', 'W', 'N'}], [{'E', 'W', 'N'}, {'S', 'W'}, {'S', 'E', 'W', 'N'}, {'S', 'E'}], [{'S', 'E', 'N'}, {'S', 'N'}, {'S', 'W'}, {'E', 'W'}], [{'W', 'N'}, {'S', 'W'}, {'S', 'E', 'N'}, {'S', 'W'}]]
    ),
    "p4_15_5": (
        4, [[0, 6, 7, 3], [10, 2, 11, 8], [4, 13, 14, 5], [12, 1, 9, 15]], [[{'W', 'S'}, {'W', 'S'}, {'N', 'S'}, {'E', 'N', 'W', 'S'}], [{'N', 'W', 'S'}, {'E', 'W', 'S'}, {'N', 'S'}, {'E', 'N'}], [{'N', 'S'}, {'E', 'N', 'S'}, {'E', 'N'}, {'E', 'N'}], [{'N', 'W'}, {'E', 'S'}, {'E', 'W'}, {'N', 'S'}]]
    ),
    "p4_20_1": (
        4, [[0, 6, 11, 3], [9, 7, 14, 1], [10, 8, 4, 13], [12, 5, 2, 15]], [[{'E', 'N'}, {'W', 'S', 'N'}, {'W', 'E'}, {'S', 'E', 'N'}], [{'W', 'S', 'E', 'N'}, {'W', 'S'}, {'E', 'N'}, {'W', 'E'}], [{'W', 'S'}, {'W', 'N'}, {'W', 'S'}, {'E', 'N'}], [{'S', 'N'}, {'S', 'E'}, {'W', 'S', 'E'}, {'W', 'S', 'N'}]]
    ),
    "p4_20_2": (
        4, [[0, 2, 8, 3], [4, 6, 5, 1], [7, 13, 14, 10], [12, 11, 9, 15]], [[{'W', 'E'}, {'W', 'N', 'S'}, {'W', 'N', 'S'}, {'S', 'N'}], [{'W', 'S'}, {'N', 'E'}, {'S', 'E'}, {'W', 'N', 'E'}], [{'W', 'S', 'E'}, {'W', 'S'}, {'S', 'N', 'E'}, {'S', 'N', 'E'}], [{'S', 'N', 'E'}, {'W', 'N', 'E'}, {'W', 'N'}, {'W', 'S', 'E'}]]
    ),
    "p4_20_3": (
        4, [[0, 7, 4, 3], [5, 14, 2, 13], [10, 1, 8, 6], [12, 11, 9, 15]], [[{'E', 'W'}, {'S', 'N', 'E'}, {'S', 'N', 'W'}, {'S', 'N', 'W'}], [{'S', 'N', 'E'}, {'N', 'E', 'W'}, {'S', 'E'}, {'N', 'E', 'W'}], [{'S', 'E'}, {'S', 'N', 'E', 'W'}, {'N', 'W'}, {'N', 'W'}], [{'S', 'W'}, {'S', 'N', 'E', 'W'}, {'S', 'W'}, {'S', 'W'}]]
    ),
    "p4_20_4": (
        4, [[0, 1, 7, 3], [5, 6, 8, 4], [9, 13, 14, 10], [12, 11, 2, 15]], [[{'E', 'W'}, {'S', 'N', 'W'}, {'S', 'E', 'W', 'N'}, {'E', 'N', 'W'}], [{'E', 'N', 'W'}, {'S', 'W'}, {'S', 'E', 'N'}, {'S', 'E'}], [{'S', 'W'}, {'S', 'W'}, {'S', 'E', 'N'}, {'S', 'N'}], [{'N', 'W'}, {'E', 'W'}, {'S', 'N'}, {'S', 'W'}]]
    ),
    "p4_20_5": (
        4, [[0, 10, 9, 3], [8, 13, 7, 11], [4, 1, 2, 5], [12, 6, 14, 15]], [[{'S', 'W'}, {'S', 'N', 'W'}, {'E', 'W'}, {'S', 'N', 'E', 'W'}], [{'N', 'E'}, {'S', 'N', 'E'}, {'S', 'N'}, {'S', 'N'}], [{'S', 'N'}, {'S', 'E'}, {'S', 'E', 'W'}, {'N', 'E'}], [{'N', 'W'}, {'S', 'W'}, {'N', 'E'}, {'S', 'N'}]]
    ),
    "p5_5_1": (
        5, [[0, 1, 2, 7, 4], [9, 5, 6, 12, 8], [10, 11, 19, 13, 14], [16, 17, 18, 23, 15], [20, 21, 22, 3, 24]], [[{'S', 'N', 'E'}, {'S', 'W'}, {'W', 'E'}, {'S', 'W', 'N', 'E'}, {'S', 'W', 'N'}], [{'S', 'W'}, {'N', 'E'}, {'W', 'N', 'E'}, {'S', 'W', 'N'}, {'W', 'E'}], [{'S', 'E'}, {'S', 'E'}, {'W', 'N'}, {'S', 'W'}, {'S', 'N'}], [{'S', 'W'}, {'W', 'N'}, {'S', 'N', 'E'}, {'W', 'N', 'E'}, {'S', 'N'}], [{'S', 'N', 'E'}, {'S', 'N', 'E'}, {'W', 'E'}, {'S', 'W'}, {'S', 'W'}]]
    ),
    "p5_5_2": (
        5, [[0, 21, 2, 23, 4], [1, 7, 8, 3, 5], [10, 6, 12, 9, 14], [18, 19, 15, 11, 13], [20, 16, 22, 17, 24]], [[{'E', 'S'}, {'S', 'N'}, {'E', 'S', 'W'}, {'E', 'N', 'W'}, {'S', 'W'}], [{'E', 'N', 'W'}, {'S', 'N', 'W'}, {'E', 'S'}, {'E', 'S', 'W'}, {'W', 'N'}], [{'S', 'N', 'W'}, {'E', 'S', 'W'}, {'S', 'N'}, {'S', 'N'}, {'E', 'N', 'S', 'W'}], [{'E', 'S'}, {'S', 'N', 'W'}, {'E', 'N', 'S', 'W'}, {'E', 'W'}, {'S', 'N', 'W'}], [{'S', 'N'}, {'E', 'W'}, {'W', 'N'}, {'E', 'S'}, {'S', 'N'}]]
    ),
    "p5_5_3": (
        5, [[0, 21, 22, 7, 4], [9, 1, 2, 13, 8], [10, 5, 6, 19, 14], [11, 12, 18, 23, 15], [20, 16, 17, 3, 24]], [[{'E', 'W'}, {'S', 'W'}, {'E', 'W'}, {'S', 'W'}, {'N', 'W'}], [{'S', 'W'}, {'S', 'E', 'W'}, {'N', 'W'}, {'S', 'E', 'W'}, {'E', 'N'}], [{'N', 'W'}, {'S', 'N', 'W'}, {'E', 'N'}, {'S', 'N', 'W'}, {'S', 'N', 'W'}], [{'N', 'W'}, {'S', 'E', 'N'}, {'S', 'E', 'W'}, {'S', 'W'}, {'S', 'W'}], [{'E', 'N', 'W'}, {'S', 'N'}, {'E', 'W'}, {'S', 'N', 'W'}, {'S', 'N'}]]
    ),
    "p5_5_4": (
        5, [[0, 6, 7, 8, 4], [5, 11, 12, 13, 9], [10, 16, 17, 18, 14], [23, 19, 15, 21, 22], [20, 1, 2, 3, 24]], [[{'W', 'E'}, {'S', 'N'}, {'W', 'S', 'N'}, {'W', 'E', 'S', 'N'}, {'S', 'N'}], [{'S', 'N'}, {'W', 'E', 'S', 'N'}, {'W', 'S', 'N'}, {'W', 'E'}, {'W', 'S'}], [{'W', 'E'}, {'W', 'S'}, {'E', 'N'}, {'W', 'S'}, {'W', 'S'}], [{'E', 'N'}, {'E', 'N'}, {'W', 'N'}, {'W', 'S'}, {'S', 'N'}], [{'S', 'N'}, {'W', 'E'}, {'W', 'S'}, {'W', 'E', 'N'}, {'W', 'S'}]]
    ),
    "p5_5_5": (
        5, [[0, 6, 2, 3, 4], [11, 7, 8, 9, 5], [15, 12, 13, 14, 10], [18, 19, 21, 16, 17], [20, 1, 22, 23, 24]], [[{'W', 'S'}, {'W', 'S'}, {'W', 'S'}, {'E', 'S'}, {'N', 'S'}], [{'W', 'S'}, {'E', 'W'}, {'N', 'S'}, {'W', 'S'}, {'N', 'S'}], [{'N', 'W'}, {'W', 'S'}, {'N', 'S'}, {'E', 'W', 'S'}, {'N', 'E'}], [{'E', 'W', 'S'}, {'W', 'S'}, {'N', 'W'}, {'N', 'W', 'E', 'S'}, {'E', 'W'}], [{'E', 'S'}, {'W', 'S'}, {'N', 'E'}, {'N', 'W', 'S'}, {'N', 'W', 'S'}]]
    ),
    "p5_10_1": (
        5, [[0, 6, 2, 3, 4], [5, 10, 7, 8, 9], [14, 18, 11, 19, 12], [17, 21, 13, 15, 16], [20, 1, 22, 23, 24]], [[{'S', 'E', 'N'}, {'W', 'E', 'N'}, {'W', 'E'}, {'W', 'S'}, {'W', 'S', 'N'}], [{'E', 'N'}, {'S', 'E'}, {'E', 'W', 'S', 'N'}, {'W', 'E'}, {'W', 'S'}], [{'S', 'N'}, {'S', 'E', 'N'}, {'S', 'E'}, {'W', 'N'}, {'W', 'S', 'N'}], [{'W', 'N'}, {'S', 'E', 'N'}, {'W', 'S'}, {'S', 'N'}, {'W', 'S'}], [{'S', 'E', 'N'}, {'W', 'S'}, {'W', 'E'}, {'W', 'E', 'N'}, {'W', 'S'}]]
    ),
    "p5_10_2": (
        5, [[0, 21, 2, 23, 4], [1, 7, 8, 6, 5], [9, 14, 10, 13, 3], [19, 15, 12, 11, 18], [20, 16, 22, 17, 24]], [[{'E', 'S'}, {'N', 'S'}, {'E', 'W', 'S'}, {'N', 'E', 'W'}, {'W', 'S'}], [{'N', 'E', 'W'}, {'N', 'W', 'S'}, {'E', 'S'}, {'E', 'W', 'S'}, {'N', 'W'}], [{'N', 'S'}, {'N', 'E', 'W', 'S'}, {'N', 'W', 'S'}, {'N', 'W', 'S'}, {'E', 'W', 'S'}], [{'N', 'W', 'S'}, {'N', 'E', 'W', 'S'}, {'N', 'S'}, {'E', 'W'}, {'E', 'S'}], [{'N', 'S'}, {'E', 'W'}, {'N', 'W'}, {'E', 'S'}, {'N', 'S'}]]
    ),
    "p5_10_3": (
        5, [[0, 21, 2, 19, 4], [8, 9, 1, 6, 15], [10, 5, 23, 3, 14], [12, 18, 17, 7, 11], [20, 16, 22, 13, 24]], [[{'W', 'E'}, {'S', 'W'}, {'W', 'N'}, {'W', 'S', 'N'}, {'W', 'N'}], [{'N', 'E'}, {'S', 'W'}, {'S', 'W', 'E'}, {'N', 'E'}, {'S', 'W'}], [{'W', 'N'}, {'W', 'S', 'N'}, {'S', 'W'}, {'W', 'S', 'N'}, {'W', 'S', 'N'}], [{'S', 'N', 'E'}, {'S', 'W', 'E'}, {'W', 'E'}, {'S', 'W'}, {'W', 'N'}], [{'W', 'N', 'E'}, {'S', 'N'}, {'W', 'E'}, {'S', 'W', 'E'}, {'S', 'N'}]]
    ),
    "p5_10_4": (
        5, [[0, 6, 7, 3, 4], [5, 11, 12, 8, 9], [13, 14, 10, 16, 17], [23, 19, 18, 21, 22], [20, 1, 2, 15, 24]], [[{'E', 'W'}, {'S', 'N'}, {'S', 'W', 'N'}, {'E', 'W', 'N'}, {'S', 'N'}], [{'S', 'N'}, {'S', 'E', 'W', 'N'}, {'S', 'W', 'N'}, {'S', 'E', 'W', 'N'}, {'S', 'W'}], [{'E', 'W'}, {'S', 'W'}, {'E', 'W'}, {'S', 'W'}, {'E', 'N'}], [{'E', 'N'}, {'E', 'N'}, {'S', 'W'}, {'S', 'W'}, {'S', 'N'}], [{'S', 'N'}, {'E', 'W'}, {'S', 'W'}, {'N', 'W'}, {'S', 'W'}]]
    ),
    "p5_10_5": (
        5, [[0, 6, 22, 14, 4], [7, 2, 3, 16, 11], [15, 12, 8, 23, 10], [18, 19, 13, 5, 17], [20, 1, 21, 9, 24]], [[{'S', 'W'}, {'S', 'W'}, {'E', 'N'}, {'E', 'S', 'W'}, {'N', 'S'}], [{'E', 'W'}, {'S', 'W'}, {'E', 'S'}, {'E', 'N', 'S', 'W'}, {'S', 'W'}], [{'N', 'W'}, {'S', 'W'}, {'N', 'S'}, {'N', 'S', 'W'}, {'E', 'N'}], [{'E', 'S', 'W'}, {'S', 'W'}, {'N', 'S'}, {'N', 'S'}, {'E', 'W'}], [{'E', 'S'}, {'S', 'W'}, {'N', 'W'}, {'S', 'W'}, {'N', 'S', 'W'}]]
    ),
    "p5_15_1": (
        5, [[0, 6, 22, 23, 4], [9, 5, 2, 7, 3], [14, 18, 10, 8, 12], [17, 11, 13, 19, 16], [20, 1, 21, 15, 24]], [[{'N', 'S', 'E'}, {'W', 'N', 'E'}, {'W', 'E'}, {'W', 'N', 'E'}, {'W', 'N', 'S'}], [{'W', 'S'}, {'N', 'E'}, {'W', 'E'}, {'N', 'W', 'S', 'E'}, {'W', 'S'}], [{'N', 'S'}, {'N', 'S', 'E'}, {'S', 'E'}, {'W', 'E'}, {'W', 'N', 'S'}], [{'W', 'N'}, {'S', 'E'}, {'W', 'S'}, {'W', 'N'}, {'W', 'S'}], [{'N', 'S', 'E'}, {'W', 'S'}, {'N', 'S', 'E'}, {'N', 'S'}, {'W', 'S'}]]
    ),
    "p5_15_2": (
        5, [[0, 21, 8, 17, 4], [1, 7, 14, 23, 5], [3, 9, 11, 6, 13], [12, 22, 10, 19, 15], [20, 16, 2, 18, 24]], [[{'S', 'E'}, {'S', 'N'}, {'S', 'E'}, {'S', 'E'}, {'S', 'W'}], [{'E', 'W', 'N'}, {'S', 'W', 'N'}, {'N', 'S', 'W', 'E'}, {'E', 'W', 'N'}, {'W', 'N'}], [{'E', 'S', 'W'}, {'S', 'N'}, {'E', 'W'}, {'E', 'S', 'W'}, {'S', 'W', 'N'}], [{'S', 'N'}, {'W', 'N'}, {'S', 'W', 'N'}, {'S', 'W', 'N'}, {'N', 'S', 'W', 'E'}], [{'S', 'N'}, {'E', 'W'}, {'E', 'S', 'W'}, {'S', 'E'}, {'S', 'N'}]]
    ),
    "p5_15_3": (
        5, [[0, 1, 7, 3, 4], [9, 5, 23, 15, 8], [10, 18, 17, 13, 14], [12, 16, 22, 19, 11], [20, 21, 2, 6, 24]], [[{'W', 'E'}, {'S', 'W', 'E'}, {'S', 'W'}, {'S', 'N', 'W'}, {'N', 'W'}], [{'S', 'W'}, {'S', 'N', 'W'}, {'S', 'W'}, {'S', 'W'}, {'N', 'E'}], [{'N', 'W'}, {'S', 'W', 'E'}, {'W', 'E'}, {'S', 'W', 'E'}, {'S', 'N', 'W'}], [{'S', 'N', 'E'}, {'S', 'N'}, {'W', 'E'}, {'S', 'N', 'W'}, {'N', 'W'}], [{'N', 'W', 'E'}, {'S', 'W'}, {'N', 'W'}, {'N', 'E'}, {'S', 'N'}]]
    ),
    "p5_15_4": (
        5, [[0, 1, 2, 15, 4], [5, 6, 7, 3, 9], [10, 8, 12, 13, 11], [23, 14, 17, 16, 22], [20, 19, 18, 21, 24]], [[{'E', 'W'}, {'E', 'W'}, {'W', 'S'}, {'N', 'W'}, {'N', 'S'}], [{'N', 'S'}, {'N', 'S'}, {'N', 'W', 'S'}, {'E', 'N', 'W'}, {'W', 'S'}], [{'E', 'W'}, {'E', 'N', 'W', 'S'}, {'N', 'W', 'S'}, {'E', 'W'}, {'E', 'N', 'W', 'S'}], [{'E', 'N'}, {'W', 'S'}, {'E', 'N'}, {'W', 'S'}, {'N', 'S'}], [{'N', 'S'}, {'E', 'N'}, {'W', 'S'}, {'W', 'S'}, {'W', 'S'}]]
    ),
    "p5_15_5": (
        5, [[0, 2, 22, 16, 4], [7, 12, 3, 10, 11], [15, 18, 8, 23, 5], [17, 1, 19, 13, 9], [20, 6, 21, 14, 24]], [[{'W', 'S'}, {'W', 'S'}, {'E', 'N'}, {'E', 'N', 'S', 'W'}, {'N', 'S'}], [{'E', 'W'}, {'W', 'S'}, {'E', 'S'}, {'E', 'N'}, {'W', 'S'}], [{'W', 'N'}, {'E', 'W', 'S'}, {'N', 'S'}, {'W', 'N', 'S'}, {'N', 'S'}], [{'E', 'W'}, {'W', 'S'}, {'W', 'S'}, {'N', 'S'}, {'W', 'S'}], [{'E', 'S'}, {'W', 'S'}, {'W', 'N'}, {'E', 'W', 'S'}, {'W', 'N', 'S'}]]
    ),
    "p5_20_1": (
        5, [[0, 2, 22, 15, 4], [18, 7, 3, 23, 5], [14, 11, 10, 9, 12], [16, 17, 1, 8, 19], [20, 6, 21, 13, 24]], [[{'E', 'S', 'N'}, {'E', 'W'}, {'E', 'W'}, {'S', 'N'}, {'S', 'W', 'N'}], [{'E', 'S', 'N'}, {'E', 'N', 'W', 'S'}, {'W', 'S'}, {'E', 'W', 'N'}, {'E', 'N'}], [{'S', 'N'}, {'E', 'S'}, {'E', 'S'}, {'W', 'S'}, {'S', 'W', 'N'}], [{'W', 'S'}, {'W', 'N'}, {'W', 'S'}, {'E', 'W'}, {'W', 'N'}], [{'E', 'S', 'N'}, {'E', 'W', 'N'}, {'E', 'S', 'N'}, {'W', 'S'}, {'W', 'S'}]]
    ),
    "p5_20_2": (
        5, [[0, 21, 8, 18, 4], [1, 7, 14, 17, 5], [3, 9, 11, 6, 23], [12, 22, 10, 13, 15], [20, 16, 2, 19, 24]], [[{'E', 'S'}, {'N', 'S'}, {'E', 'S'}, {'E', 'S'}, {'W', 'S'}], [{'E', 'N', 'W'}, {'N', 'W', 'S'}, {'E', 'N', 'W', 'S'}, {'E', 'S'}, {'N', 'W'}], [{'E', 'W', 'S'}, {'N', 'S'}, {'E', 'W'}, {'E', 'W', 'S'}, {'E', 'N', 'W'}], [{'N', 'S'}, {'N', 'W'}, {'N', 'W', 'S'}, {'N', 'W', 'S'}, {'E', 'N', 'W', 'S'}], [{'N', 'S'}, {'E', 'W'}, {'E', 'W', 'S'}, {'N', 'W', 'S'}, {'N', 'S'}]]
    ),
    "p5_20_3": (
        5, [[0, 21, 7, 15, 4], [9, 1, 23, 13, 8], [10, 5, 17, 19, 14], [16, 18, 6, 11, 12], [20, 22, 2, 3, 24]], [[{'W', 'E'}, {'W', 'S'}, {'W', 'S'}, {'W', 'S'}, {'W', 'N'}], [{'W', 'S'}, {'W', 'E', 'S'}, {'W', 'S'}, {'W', 'E', 'S'}, {'N', 'E'}], [{'W', 'N'}, {'W', 'N', 'S'}, {'W', 'E'}, {'W', 'N', 'S'}, {'W', 'N', 'S'}], [{'N', 'S'}, {'W', 'E', 'S'}, {'N', 'E'}, {'W', 'N'}, {'S', 'N', 'E'}], [{'W', 'N', 'E'}, {'W', 'E'}, {'W', 'N'}, {'W', 'N', 'S'}, {'N', 'S'}]]
    ),
    "p5_20_4": (
        5, [[0, 1, 2, 15, 4], [5, 6, 7, 3, 9], [10, 8, 12, 13, 11], [14, 16, 18, 22, 23], [20, 19, 17, 21, 24]], [[{'W', 'E'}, {'W', 'E'}, {'W', 'S'}, {'W', 'N'}, {'N', 'S'}], [{'N', 'S'}, {'N', 'S'}, {'W', 'N', 'S'}, {'W', 'E', 'N'}, {'W', 'S'}], [{'W', 'E'}, {'W', 'E', 'N', 'S'}, {'W', 'N', 'S'}, {'W', 'E'}, {'W', 'E', 'N', 'S'}], [{'W', 'S'}, {'W', 'S'}, {'W', 'S'}, {'N', 'S'}, {'E', 'N'}], [{'N', 'S'}, {'E', 'N'}, {'E', 'N'}, {'W', 'S'}, {'W', 'S'}]]
    ),
    "p5_20_5": (
        5, [[0, 2, 22, 16, 4], [10, 12, 7, 18, 3], [1, 11, 23, 5, 15], [17, 8, 19, 13, 9], [20, 6, 21, 14, 24]], [[{'W', 'S'}, {'W', 'S'}, {'N', 'E'}, {'W', 'S', 'N', 'E'}, {'S', 'N'}], [{'N', 'E'}, {'W', 'S'}, {'W', 'E'}, {'W', 'S', 'E'}, {'S', 'E'}], [{'W', 'S'}, {'W', 'S'}, {'W', 'S', 'N'}, {'S', 'N'}, {'W', 'N'}], [{'W', 'E'}, {'S', 'N'}, {'W', 'S'}, {'S', 'N'}, {'W', 'S'}], [{'S', 'E'}, {'W', 'S'}, {'W', 'N'}, {'W', 'S', 'E'}, {'W', 'S', 'N'}]]
    )
}


class LabyrinthDomain(Domain):
    def __init__(self) -> None:
        self._instances = INSTANCES

    def list_instances(self) -> dict[str, dict]:
        return {k: {"n": v[0]} for k, v in self._instances.items()}

    def get_instance(self, instance: Optional[str] = None) -> list[int]:
        if instance and instance in self._instances:
            return self._instances[instance]
        raise ValueError(f"Instance '{instance}' not found!")


    def build_problem(self, instance: str | None = None) -> "Problem":
        n, initial_cards, paths = self.get_instance(instance)
        n_cards = n * n

        # --- Problem ---
        problem = Problem('labyrinth_problem')

        Card = UserType('Card')
        Direction = UserType('Direction')
        N_dir = Object('N', Direction)
        S_dir = Object('S', Direction)
        E_dir = Object('E', Direction)
        W_dir = Object('W', Direction)
        direction_by_name = {'N': N_dir, 'S': S_dir, 'E': E_dir, 'W': W_dir}
        problem.add_objects([N_dir, S_dir, E_dir, W_dir])
        problem.add_objects([Object(f'card_{i}', Card) for i in range(n_cards)])
        card_0 = problem.object('card_0')

        card_at     = Fluent('card_at', ArrayType(n, ArrayType(n, Card)))
        robot_at    = Fluent('robot_at', Card)
        connections = Fluent('connections', c=Card, d=Direction)
        problem.add_fluent(card_at,     default_initial_value=card_0)
        problem.add_fluent(robot_at,    default_initial_value=card_0)
        problem.add_fluent(connections, default_initial_value=False)

        for r in range(n):
            for c in range(n):
                card_obj = problem.object(f'card_{initial_cards[r][c]}')
                problem.set_initial_value(card_at[r][c], card_obj)
                for d in paths[r][c]:
                    problem.set_initial_value(connections(card_obj, direction_by_name[d]), True)

        # --- Actions ---
        move_north = InstantaneousAction('move_north', r=IntType(0, n - 1), c=IntType(0, n - 1))
        r, c = move_north.parameter('r'), move_north.parameter('c')
        move_north.add_precondition(Equals(robot_at, card_at[r][c]))
        move_north.add_precondition(connections(card_at[r][c], N_dir))
        move_north.add_precondition(connections(card_at[r - 1][c], S_dir))
        move_north.add_effect(robot_at, card_at[r - 1][c])

        move_south = InstantaneousAction('move_south', r=IntType(0, n - 1), c=IntType(0, n - 1))
        r, c = move_south.parameter('r'), move_south.parameter('c')
        move_south.add_precondition(Equals(robot_at, card_at[r][c]))
        move_south.add_precondition(connections(card_at[r][c], S_dir))
        move_south.add_precondition(connections(card_at[r + 1][c], N_dir))
        move_south.add_effect(robot_at, card_at[r + 1][c])

        move_east = InstantaneousAction('move_east', r=IntType(0, n - 1), c=IntType(0, n - 1))
        r, c = move_east.parameter('r'), move_east.parameter('c')
        move_east.add_precondition(Equals(robot_at, card_at[r][c]))
        move_east.add_precondition(connections(card_at[r][c], E_dir))
        move_east.add_precondition(connections(card_at[r][c + 1], W_dir))
        move_east.add_effect(robot_at, card_at[r][c + 1])

        move_west = InstantaneousAction('move_west', r=IntType(0, n - 1), c=IntType(0, n - 1))
        r, c = move_west.parameter('r'), move_west.parameter('c')
        move_west.add_precondition(Equals(robot_at, card_at[r][c]))
        move_west.add_precondition(connections(card_at[r][c], W_dir))
        move_west.add_precondition(connections(card_at[r][c - 1], E_dir))
        move_west.add_effect(robot_at, card_at[r][c - 1])

        rotate_col_up = InstantaneousAction('rotate_col_up', c=IntType(0, n - 1))
        c = rotate_col_up.parameter('c')
        all_rows = RangeVariable('all_rows', 0, n - 1)
        rotate_col_up.add_precondition(Forall(Not(Equals(robot_at, card_at[all_rows][c])), all_rows))
        rotated_rows = RangeVariable('rotated_rows', 1, n - 1)
        rotate_col_up.add_effect(card_at[rotated_rows - 1][c], card_at[rotated_rows][c], forall=[rotated_rows])
        rotate_col_up.add_effect(card_at[n - 1][c], card_at[0][c])

        rotate_col_down = InstantaneousAction('rotate_col_down', c=IntType(0, n - 1))
        c = rotate_col_down.parameter('c')
        all_rows = RangeVariable('all_rows', 0, n - 1)
        rotate_col_down.add_precondition(Forall(Not(Equals(robot_at, card_at[all_rows][c])), all_rows))
        rotated_rows = RangeVariable('rotated_rows', 1, n - 1)
        rotate_col_down.add_effect(card_at[rotated_rows][c], card_at[rotated_rows - 1][c], forall=[rotated_rows])
        rotate_col_down.add_effect(card_at[0][c], card_at[n - 1][c])

        rotate_row_left = InstantaneousAction('rotate_row_left', r=IntType(0, n - 1))
        r = rotate_row_left.parameter('r')
        all_cols = RangeVariable('all_cols', 0, n - 1)
        rotate_row_left.add_precondition(Forall(Not(Equals(robot_at, card_at[r][all_cols])), all_cols))
        rotated_cols = RangeVariable('rotated_cols', 0, n - 2)
        rotate_row_left.add_effect(card_at[r][rotated_cols], card_at[r][rotated_cols + 1], forall=[rotated_cols])
        rotate_row_left.add_effect(card_at[r][n - 1], card_at[r][0])

        rotate_row_right = InstantaneousAction('rotate_row_right', r=IntType(0, n - 1))
        r = rotate_row_right.parameter('r')
        all_cols = RangeVariable('all_cols', 0, n - 1)
        rotate_row_right.add_precondition(Forall(Not(Equals(robot_at, card_at[r][all_cols])), all_cols))
        rotated_cols = RangeVariable('rotated_cols', 1, n - 1)
        rotate_row_right.add_effect(card_at[r][rotated_cols], card_at[r][rotated_cols - 1], forall=[rotated_cols])
        rotate_row_right.add_effect(card_at[r][0], card_at[r][n - 1])

        problem.add_actions([
            move_north, move_south, move_east, move_west,
            rotate_col_up, rotate_col_down,
            rotate_row_left, rotate_row_right,
        ])

        # --- Goals ---
        problem.add_goal(Equals(robot_at, card_at[n - 1][n - 1]))
        problem.add_goal(connections(card_at[n - 1][n - 1], S_dir))

        # --- Metric ---
        costs: Dict[Action, Expression] = {
            move_north: Int(1), move_south: Int(1),
            move_east:  Int(1), move_west:  Int(1),
            rotate_col_up: Int(1), rotate_col_down: Int(1),
            rotate_row_left: Int(1), rotate_row_right: Int(1),
        }
        problem.add_quality_metric(MinimizeActionCosts(costs))

        return problem

DOMAIN = LabyrinthDomain()