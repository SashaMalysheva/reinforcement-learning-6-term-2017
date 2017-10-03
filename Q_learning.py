import random as pr
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from lion_cow import Game
game = Game(18, 13, 1, True)
np.random.seed(100)

def abstractMDP(num_episodes=20, discount_factor=0.9, epsilon=0.1):
    history = []
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = np.zeros((2 ** 13, game.state_space, len(game.action_space)))
    policy = make_epsilon_greedy_policy(Q, epsilon, len(game.action_space))

    for i_episode in range(1, num_episodes + 1):
        game.restart()
        episode = []
        if (i_episode + 1) % 1 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        while True:
            state = game.state
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, action = game.step(action)
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

    V = np.max(Q, axis=2)*0.001
    return V


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


def q_learning(game, num_episodes, discount_factor=0.9, alpha=1, epsilon=0.1):
    history = []
    Q = np.zeros((2 ** 13, game.state_space, len(game.action_space)))
    F = np.zeros((2 ** 13, game.state_space))
    F = abstractMDP()
    policy = make_epsilon_greedy_policy(Q, epsilon, len(game.action_space))

    for i_episode in range(num_episodes):
        game.restart()
        sum_reward = 0
        if (i_episode + 1) % 1 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = game.state
        while True:
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, action = game.step(action)
            sum_reward += reward
            if state != next_state:
                best_next_action = np.argmax(Q[next_state[0], next_state[1]])
                td_target = discount_factor * F[next_state[0], next_state[1]] - F[state[0]][state[1]]
                td_target += reward + discount_factor * Q[next_state[0], next_state[1], best_next_action]
                Q[state[0]][state[1], action] += alpha * (td_target - Q[state[0]][state[1], action])

         #       if Q[next_state[0], next_state[1], best_next_action] != 0:
          #          print(Q[next_state[1], next_state[1], best_next_action], best_next_action, next_state, action, state, Q[state[0]][state[1], action])
                if done:
                    history.append((i_episode, game.time))
                    break
                state = next_state

    return Q, history


Q, history = q_learning(game, 100)
plt.scatter(*zip(*history))
plt.show()
