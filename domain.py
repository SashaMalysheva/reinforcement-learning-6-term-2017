import numpy as np


def cow_place():
    l = [(0,5), (2, 7), (8, 3), (10, 11), (11, 1)]
    return l

def not_allowed():
    s = set()
    for i in range(13):
        s.add(((12, i),(13, i)))
        s.add(((0, i),(-1, i)))
        if i < 9 and i != 3:
            s.add(((4, i), (5, i)))
            s.add(((5, i), (4, i)))
        if i < 6 and i != 2:
            s.add(((7, i), (8, i)))
            s.add(((8, i), (7, i)))
        if 5 < i != 7:
            s.add(((9, i), (10, i)))
            s.add(((10, i), (9, i)))
        # vertical
        s.add(((i, 0), (i, -1)))
        s.add(((i, 12), (i, 13)))
        if i != 5:
            s.add(((i, 5), (i, 6)))
            s.add(((i, 6), (i, 5)))
        if 3 < i < 9 and i != 5:
            s.add(((i, 9), (i, 8)))
            s.add(((i, 8), (i, 9)))
    return s


def print_domain():
    matrix = np.zeros((13,13))
    cows = cow_place()
    print(cow_place())
    moves = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    for i in range(13):
        for j in range(13):
            for action in range(4):
                s1 = (i, j)
                s2 = (i + moves[action][0], j + moves[action][1])
                if (s1, s2) in not_allowed():
                    matrix[i, j] = action + 1
                if s1 in set(cows):
                    matrix[i, j] = 8
    print(matrix)
