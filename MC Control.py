import numpy as np
from collections import defaultdict
from lion_cow import Game
import random as pr
import sys
import matplotlib.pyplot as plt
game = Game(9, 9, 4)


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


def mc_control_epsilon_greedy(num_episodes, discount_factor=0.9, epsilon=0.1):
    history = []
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = np.zeros((2 ** 13, game.state_space, len(game.action_space)))
    policy = make_epsilon_greedy_policy(Q, epsilon, len(game.action_space))

    for i_episode in range(1, num_episodes + 1):
        game.restart()

        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        episode = []
        while True:
            state = game.state
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = game.step(action)
            if state != next_state:
                episode.append((state, action, reward))
                if done:
                    history.append((i_episode, game.time))
                    break
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state[0]][state[1], action] = returns_sum[sa_pair] / returns_count[sa_pair]
    return Q, history


Q, history = mc_control_epsilon_greedy(num_episodes=400)
plt.scatter(*zip(*history))
plt.show()
