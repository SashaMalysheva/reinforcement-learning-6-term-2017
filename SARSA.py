import random as pr
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from lion_cow import Game
game = Game(8, 8, 3)


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


def sarsa(game, num_episodes, discount_factor=0.8, alpha=0.8, epsilon=0.1):
    history = []
    Q =np.zeros((2 ** 13, game.state_space, len(game.action_space)))
    policy = make_epsilon_greedy_policy(Q, epsilon, len(game.action_space))

    for i_episode in range(num_episodes):
        game.restart()

        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = game.state
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        while True:
            next_state, reward, done = game.step(action)
            if state != next_state:
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

                td_target = reward + discount_factor * Q[next_state[0], next_state[1], next_action]
                Q[state[0]][state[1], action] += alpha * (td_target - Q[state[0]][state[1], action])

                if done:
                    history.append((i_episode, game.time))
                    break

                action = next_action
                state = next_state
            else:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    return Q, history


Q, history = sarsa(game, 300)
plt.scatter(*zip(*history))
plt.show()
