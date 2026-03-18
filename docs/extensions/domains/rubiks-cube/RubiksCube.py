import random

from experiments import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Parser
parser = argparse.ArgumentParser(description="Solve Rush Hour")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# ---------------------------------------------------- Problem ---------------------------------------------------------

rotations = 5

U, F, R, B, L, D = range(6)

def cubeInit():
    cube = []
    for f in range(6):
        face = []
        for _ in range(3):
            row = []
            for _ in range(3):
                row.append(f)
            face.append(row)
        cube.append(face)
    return cube


def _copy_cube(cube):
    new_cube = []
    for face in cube:
        new_face = []
        for row in face:
            new_row = []
            for cell in row:
                new_row.append(cell)
            new_face.append(new_row)
        new_cube.append(new_face)
    return new_cube


def _rotate_face_cw(cube, face):
    orig = cube
    new = _copy_cube(orig)

    for r in range(3):
        for c in range(3):
            new[face][r][c] = orig[face][2 - c][r]

    if face == U:
        F_, R_, B_, L_ = F, R, B, L
        row = 0
        for i in range(3):
            new[F_][row][i] = orig[L_][row][i]
            new[R_][row][i] = orig[F_][row][i]
            new[B_][row][i] = orig[R_][row][i]
            new[L_][row][i] = orig[B_][row][i]

    elif face == D:
        F_, L_, B_, R_ = F, L, B, R
        row = 2
        for i in range(3):
            new[F_][row][i] = orig[R_][row][i]
            new[R_][row][i] = orig[B_][row][i]
            new[B_][row][i] = orig[L_][row][i]
            new[L_][row][i] = orig[F_][row][i]

    elif face == F:
        U_, R_, D_, L_ = U, R, D, L
        for i in range(3):
            new[R_][i][0] = orig[U_][2][i]
            new[D_][0][i] = orig[R_][i][0]
            new[L_][2-i][2] = orig[D_][0][i]
            new[U_][2][i] = orig[L_][2-i][2]

    elif face == R:
        U_, B_, D_, F_ = U, B, D, F
        for i in range(3):
            new[F_][i][2] = orig[U_][i][2]
            new[D_][i][2] = orig[F_][i][2]
            new[B_][2-i][0] = orig[D_][i][2]
            new[U_][i][2] = orig[B_][2-i][0]

    elif face == B:
        U_, L_, D_, R_ = U, L, D, R
        for i in range(3):
            new[R_][i][2] = orig[U_][0][i]
            new[D_][2][2-i] = orig[R_][i][2]
            new[L_][i][0] = orig[D_][2][i]
            new[U_][0][2-i] = orig[L_][i][0]

    elif face == L:
        U_, F_, D_, B_ = U, F, D, B
        for i in range(3):
            new[F_][i][0] = orig[U_][i][0]
            new[D_][i][0] = orig[F_][i][0]
            new[B_][2-i][2] = orig[D_][i][0]
            new[U_][i][0] = orig[B_][2-i][2]

    else:
        raise ValueError("Impossible Index")
    return new


def move_0(cube):
    return _rotate_face_cw(cube, 0)

def move_0_rev(cube):
    for _ in range(3):
        cube = _rotate_face_cw(cube, 0)
    return cube


def move_1(cube):
    return _rotate_face_cw(cube, 1)

def move_1_rev(cube):
    for _ in range(3):
        cube = _rotate_face_cw(cube, 1)
    return cube


def move_2(cube):
    return _rotate_face_cw(cube, 2)

def move_2_rev(cube):
    for _ in range(3):
        cube = _rotate_face_cw(cube, 2)
    return cube


def move_3(cube):
    return _rotate_face_cw(cube, 3)

def move_3_rev(cube):
    for _ in range(3):
        cube = _rotate_face_cw(cube, 3)
    return cube


def move_4(cube):
    return _rotate_face_cw(cube, 4)

def move_4_rev(cube):
    for _ in range(3):
        cube = _rotate_face_cw(cube, 4)
    return cube


def move_5(cube):
    return _rotate_face_cw(cube, 5)

def move_5_rev(cube):
    for _ in range(3):
        cube = _rotate_face_cw(cube, 5)
    return cube


cube_actions = {
    "0": move_0,
    "0rev": move_0_rev,
    "1": move_1,
    "1rev": move_1_rev,
    "2": move_2,
    "2rev": move_2_rev,
    "3": move_3,
    "3rev": move_3_rev,
    "4": move_4,
    "4rev": move_4_rev,
    "5": move_5,
    "5rev": move_5_rev,
}

def genRandomState(moves, actions):
    move_list = []
    cube = cubeInit()

    for i in range(moves):
        move = random.choice(actions)
        move_list.append(move)
        cube = cube_actions[move](cube)

    return cube, move_list


actions = sorted(x for x in cube_actions.keys())
CC = genRandomState(rotations, actions)
print('Final cube state:', CC[0])
init = CC[0]

############################################################

rubik = Problem("rubik")
cube = Fluent("cube", ArrayType(6, ArrayType(3, ArrayType(3, IntType(0, 5)))))
rubik.add_fluent(cube, default_initial_value=0)
rubik.set_initial_value(cube, init)

def get_neighbors(face):
    neighbors = {
        0: [1, 2, 3, 4],
        1: [0, 2, 5, 4],
        2: [0, 3, 5, 1],
        3: [0, 4, 5, 2],
        4: [0, 1, 5, 3],
        5: [1, 4, 3, 2]
    }
    if face not in neighbors:
        raise ValueError(f"Invalid face index: {face}")

    return neighbors[face]

def rotate_edges_cw(action, cube, face):
    if face == 0:
        F, R, B, L = get_neighbors(0)
        row = 0
        for i in range(3):
            action.add_effect(cube[F][row][i], cube[L][row][i])
            action.add_effect(cube[R][row][i], cube[F][row][i])
            action.add_effect(cube[B][row][i], cube[R][row][i])
            action.add_effect(cube[L][row][i], cube[B][row][i])

    elif face == 5:
        F, L, B, R = get_neighbors(5)
        row = 2
        for i in range(3):
            action.add_effect(cube[F][row][i], cube[R][row][i])
            action.add_effect(cube[R][row][i], cube[B][row][i])
            action.add_effect(cube[B][row][i], cube[L][row][i])
            action.add_effect(cube[L][row][i], cube[F][row][i])

    elif face == 1:
        U, R, D, L = get_neighbors(1)
        for i in range(3):
            action.add_effect(cube[R][i][0], cube[U][2][i])
            action.add_effect(cube[D][0][i], cube[R][i][0])
            action.add_effect(cube[L][2-i][2], cube[D][0][i])
            action.add_effect(cube[U][2][i], cube[L][2-i][2])

    elif face == 2:
        U, B, D, F = get_neighbors(2)
        for i in range(3):
            action.add_effect(cube[F][i][2], cube[U][i][2])
            action.add_effect(cube[D][i][2], cube[F][i][2])
            action.add_effect(cube[B][2-i][0], cube[D][i][2])
            action.add_effect(cube[U][i][2], cube[B][2-i][0])

    elif face == 3:
        U, L, D, R = get_neighbors(3)
        for i in range(3):
            action.add_effect(cube[R][i][2], cube[U][0][i])
            action.add_effect(cube[D][2][2-i], cube[R][i][2])
            action.add_effect(cube[L][i][0], cube[D][2][i])
            action.add_effect(cube[U][0][2-i], cube[L][i][0])

    elif face == 4:
        U, F, D, B = get_neighbors(4)
        for i in range(3):
            action.add_effect(cube[F][i][0], cube[U][i][0])
            action.add_effect(cube[D][i][0], cube[F][i][0])
            action.add_effect(cube[B][2-i][2], cube[D][i][0])
            action.add_effect(cube[U][i][0], cube[B][2-i][2])

    else:
        raise NotImplementedError("rotate_edges_cw supports faces 0,1,2,3,4,5.")


def rotate_face_cw(action, cube, face):
    for r in range(3):
        for c in range(3):
            action.add_effect(cube[face][r][c], cube[face][2 - c][r])


rotate_face_0_cw = InstantaneousAction("rotate_face_0_cw")
rotate_face_cw(rotate_face_0_cw, cube, 0)
rotate_edges_cw(rotate_face_0_cw, cube, 0)
rubik.add_action(rotate_face_0_cw)

rotate_face_5_cw = InstantaneousAction("rotate_face_5_cw")
rotate_face_cw(rotate_face_5_cw, cube, 5)
rotate_edges_cw(rotate_face_5_cw, cube, 5)
rubik.add_action(rotate_face_5_cw)

rotate_face_1_cw = InstantaneousAction("rotate_face_1_cw")
rotate_face_cw(rotate_face_1_cw, cube, 1)
rotate_edges_cw(rotate_face_1_cw, cube, 1)
rubik.add_action(rotate_face_1_cw)

rotate_face_2_cw = InstantaneousAction("rotate_face_2_cw")
rotate_face_cw(rotate_face_2_cw, cube, 2)
rotate_edges_cw(rotate_face_2_cw, cube, 2)
rubik.add_action(rotate_face_2_cw)

rotate_face_3_cw = InstantaneousAction("rotate_face_3_cw")
rotate_face_cw(rotate_face_3_cw, cube, 3)
rotate_edges_cw(rotate_face_3_cw, cube, 3)
rubik.add_action(rotate_face_3_cw)

rotate_face_4_cw = InstantaneousAction("rotate_face_4_cw")
rotate_face_cw(rotate_face_4_cw, cube, 4)
rotate_edges_cw(rotate_face_4_cw, cube, 4)
rubik.add_action(rotate_face_4_cw)


for f in range(1, 6):
    for r in range(3):
        for c in range(3):
            rubik.add_goal(Equals(cube[f][r][c], f))

rubik.add_quality_metric(MinimizeActionCosts({
    rotate_face_0_cw: Int(1),
    rotate_face_1_cw: Int(1),
    rotate_face_2_cw: Int(1),
    rotate_face_3_cw: Int(1),
    rotate_face_4_cw: Int(1),
    rotate_face_5_cw: Int(1),
}))

compilation_solving.compile_and_solve(rubik, solving, compilation)
