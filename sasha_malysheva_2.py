import numpy as np


def capability(i, v):
    c = 0
    # right
    if (i + 1) % v != 0 and i - 1 < v * v: c += 1
    # left
    if i % v != 0 and i > 0: c += 1
    # down
    if i + v < v * v: c += 1
    # up
    if i - v > -1: c += 1
    return c


def init(v):
    P = np.zeros((v * v, v * v))
    for i in range(v * v):
        c = capability(i, v)
        # right
        if (i + 1) % v != 0 and i - 1 < v * v:
            c -= 1
            if c != 0:
                P[i, i + 1] = np.random.random()
            else:
                P[i, i + 1] = 1 - np.sum(P[i])
        # left
        if i % v != 0 and i > 0:
            c -= 1
            if c != 0:
                P[i, i - 1] = np.random.random()
            else:
                P[i, i - 1] = 1 - np.sum(P[i])
        # down
        if i + v < v * v:
            c -= 1
            if c != 0:
                P[i, i + v] = np.random.random()
            else:
                P[i, i + v] = 1 - np.sum(P[i])
        # up
        if i - v > -1:
            c -= 1
            if c != 0:
                P[i, i - v] = np.random.random()
            else:
                P[i, i - v] = 1 - np.sum(P[i])
    R = np.array([np.random.random() * 20 - 10 for i in range(v * v)])
    R[v * v - 1] = 1000
    return P, R


def value_iteration(P, R, eps=0.001, learning_rate=0.75):
    steps = 0
    v = len(R)
    new_U = np.zeros(v)
    p = np.zeros(v)
    U = np.ones(v)
    while np.max(np.fabs(U - new_U)) > eps:
        steps += 1
        U = np.copy(new_U)
        new_U = np.max(np.multiply(P, U) * learning_rate + np.multiply(P, R), axis=0)
    print('Value_iteration takes ' + str(steps) + ' steps.')
    p = np.argmax(np.multiply(P, U) * learning_rate + np.multiply(P, R), axis=0)
    print(p)
    return p


def init_p(P, R, learning_rate=0.75):
    v = len(R)
    p = np.zeros(v)
    U = np.ones(v)
    U = np.max(np.multiply(P, U) * learning_rate + np.multiply(P, R), axis=0)
    p = np.argmax(np.multiply(P, U) * learning_rate + np.multiply(P, R), axis=0)
    return p


def policy_evaluate(P, R, U, p, eps=0.001, learning_rate=0.75):
    v = len(R)
    steps = 0
    new_U = np.copy(U)
    U = np.ones(v)
    while np.max(np.fabs(U - new_U)) > eps:
        steps += 1
        U = np.copy(new_U)
        for i in range(v):
            new_U[i] = P[i, p[i]] * U[p[i]] * learning_rate + R[p[i]]
    return new_U, steps


def policy_iteration(P, R, eps=0.001, learning_rate=0.75):
    steps = 0
    num = []
    v = len(R)
    U = np.zeros(v)
    p = init_p(P, R)
    old_p = np.zeros(v)
    old_2_p = np.zeros(v)
    changed = True
    while changed:
        U, s = policy_evaluate(P, R, U, p, eps)
        num.append(s)
        steps += 1
        changed = False
        p = np.argmax(np.multiply(P, U) * learning_rate + np.multiply(P, R), axis=0)
        if not np.array_equal(p, old_p) and not np.array_equal(p, old_2_p):
            old_2_p = np.copy(old_p)
            old_p = np.copy(p)
            changed = True
    print('Policy_iteration takes ' + str(steps) + ' steps. With ' + str(num) + ' substeps.')
    print(p)


P, R = init(3)
value_iteration(P, R)
policy_iteration(P, R)
