from Tilecoder import numTilings, tilecode, numTiles
from pylab import *
import matplotlib.pyplot as plt


class MountainCar:
    num_of_action = 3

    def __init__(self):
        self.position = -0.4
        self.velocity = 0

    def sample(self, action):
        if not action in (0, 1, 2):
            print('Invalid action:', action)
        R = -1 if action == 1 else -1.5
        action -= 1
        self.velocity += 0.001 * action - 0.0025 * cos(MountainCar.num_of_action * self.position)
        self.velocity = - 0.07 if self.velocity < -0.07 else self.velocity
        self.velocity = 0.06999999 if self.velocity >= 0.07 else self.velocity
        self.position += self.velocity
        done = self.position >= 0.5
        if self.position < -1.2:
            self.position = -1.2
            self.velocity = 0.0
        return R, done


def egreedy(Qs, epsilon):
    if rand() < epsilon:
        return randint(MountainCar.num_of_action)
    else:
        return argmax(Qs)  # return argmax θt^T ф(S,action)


def expQunderPi(Qs, Epi):
    return Epi * average(Qs) + (1 - Epi) * max(Qs)


def Qs(F, theta):
    # initialize Q[S, action, F]
    Q = np.zeros(MountainCar.num_of_action)
    for a in range(MountainCar.num_of_action):
        # numTilings
        for i in F:
            Q[a] = Q[a] + theta[i + (a * numTiles)]
    return Q


def sarsa_lambda(numEpisodes=200, alpha=0.002, gamma=1, lmbda=0.9):
    history = []
    Epi = Emu = 0
    n = numTiles * MountainCar.num_of_action  # 4 * 9 * 9 * 3
    F = [-1] * numTilings  # 4
    theta = -0.01 * rand(n)
    for episodeNum in range(numEpisodes):
        if (episodeNum + 1) % 1 == 0:
            print("\rEpisode {}/{}.".format(episodeNum + 1, numEpisodes), end="")
            sys.stdout.flush()
        error = step = 0
        S = MountainCar()
        e = np.zeros(n)
        while S is not None:
            # get a list of four tile indices
            tilecode(S.position, S.velocity, F)
            Q = Qs(F, theta)
            action = egreedy(Q, Emu)
            R, done = S.sample(action)
            if done:
                theta += alpha * delta * e
                break
            delta = R - Q[action]
            error += R
            for f in F:
                e[f + (action * numTiles)] = 1
            tilecode(S.position, S.velocity, F)
            Qprime = Qs(F, theta)
            delta += expQunderPi(Qprime, Epi)
            theta += alpha * delta * e
            e *= gamma * lmbda
            step += 1
        history.append((episodeNum, -error))
        #print("Episode: ", episodeNum, "Steps:", step, "Return: ", error)
    return history

history = sarsa_lambda()
plt.scatter(*zip(*history))
show(plt)
