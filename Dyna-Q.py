import random as pr
import sys
import matplotlib.pyplot as plt
import numpy as np
from lion_cow import Game
import random
game = Game(10, 10, 2, True)


def hash(x, y):
    len = game.state_space + 1
    return x * len * 2 ** 13 + y[0] + y[1] * 2 ** 13


def unhash(a):
    len = game.state_space + 1
    y0 = int(a % (2 ** 13))
    a = (a - y0) / (2 ** 13)
    y1 = int(a % len)
    x = int((a - y1) / len)
    return x, (y0, y1)


def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        A = np.ones(nA, dtype=float) * epsilon / nA
        if np.max(Q[state[0],state[1]]) != 0:
            best_action = rargmax(Q[state])
            A[best_action] += (1.0 - epsilon)
        else:
            A /= epsilon
        return A
    return policy_fn


def q_learning(game, num_episodes, discount_factor=0.9, alpha=0.1, epsilon=0.1, n=30):
    history = []
    Q = np.zeros((2 ** 13, game.state_space, len(game.action_space)))
    Model = np.zeros((2 ** 13, game.state_space, len(game.action_space)))
    policy = make_epsilon_greedy_policy(Q, epsilon, len(game.action_space))
    for i_episode in range(num_episodes):
        game.restart()

        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = game.state
        episode = []
        while True:
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, action = game.step(action)
            if state != next_state:
                episode.append((state, action))
                best_next_action = np.argmax(Q[next_state[0], next_state[1]])
                td_target = reward + discount_factor * Q[next_state[0], next_state[1], best_next_action]
                Q[state[0]][state[1], action] += alpha * (td_target - Q[state[0]][state[1], action])
                Model[state[0]][state[1], action] = hash(reward, state)
                sa_in_episode = set([(x[0], x[1]) for x in episode])
                if done:
                    history.append((i_episode, game.time))
                    break
                state = next_state
                for i in range(n):
                    random_state, random_action = random.sample(sa_in_episode, 1)[0][0], random.sample(sa_in_episode, 1)[0][1]
                    random_reward, random_new_state = unhash(Model[random_state[0], random_state[1], random_action])
                    best_next_action = np.argmax(Q[random_new_state[0], random_new_state[1]])
                    td_target = random_reward + discount_factor * Q[random_new_state[0], random_new_state[1], best_next_action]
                    Q[random_state[0]][random_state[1], action] += alpha * (td_target - Q[random_state[0]][random_state[1], action])
    return Q, history


Q, history = q_learning(game, 100)
plt.scatter(*zip(*history))
plt.show()
