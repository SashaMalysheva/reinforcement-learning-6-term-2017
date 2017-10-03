import numpy as np
import domain
moves = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def hash(x, y, len):
    len += 1
    return x + y*len

def revwers(a):
    return (a + 1 - (a % 2) * 2) + (a // 4)

class Game():
    def __init__(self, tot_row=3, tot_col=3, num_of_cows=1, stochastic=False):
        self.world_row = tot_row - 1
        self.world_col = tot_col - 1
        self.cell = [0, 0]
        self.cow = 0
        self.stochastic = stochastic
        self.cow_place = []
        self.state_space = (self.world_row + 1) * (self.world_col + 1) + 1
        self.deck = np.zeros((tot_row, tot_col))
        for l in domain.cow_place():
            print(l)
            self.cow_place.append(l)
            self.deck[l[0], l[1]] += 1
        self._copy_cow_place = self.cow_place
        self._copy_deck = np.copy(self.deck)
        self.num_of_cows = len(self.cow_place)
        self._copy_num_of_cows = self.num_of_cows
        self.action_space = moves
        self.time = 0
        self.update()

    def restart(self):
        self.cell = [0, 0]
        self.cow = 0
        self.cow_place = self._copy_cow_place.copy()
        self.deck = np.copy(self._copy_deck)
        self.num_of_cows = self._copy_num_of_cows
        self.time = 0
        m = self.deck.flatten() == 1
        m = np.sum([m[i] * (2 ** i) for i in range(len(m))])
        self.update()

    def change(self, action):
        flag = False
        if 0 <= self.cell[0] + action[0] <= self.world_row:
            self.cell[0] += action[0]
        else:
            flag = True
        if 0 <= self.cell[1] + action[1] <= self.world_col:
            self.cell[1] += action[1]
        else:
            flag = True
        return flag

    def change_domain(self, action):
        flag = False
        x_new = self.cell[0] + action[0]
        y_new = self.cell[1] + action[1]
        if ((self.cell[0], self.cell[1]), (x_new, y_new)) not in domain.not_allowed():
            self.cell = [x_new, y_new]
        else:
            flag = True
        return flag

    def update(self):
        m = 0
        for i in range(len(self.cow_place)):
            m += i * self.state_space + hash(self.cow_place[i][0], self.cow_place[i][1], self.world_col)
        self.state = (m, hash(self.cell[0], self.cell[1], self.world_col))

    def revers_action(self, action):
        if self.stochastic:
            action_probs = np.array([0.7, 0.3])
            actions = np.array([action, revwers(action)])
            action = np.random.choice(actions, p=action_probs)
        return action

    def step(self, action):
        action = self.revers_action(action)
        reward = 0
        done = False
        unknown = self.change_domain(moves[action])
        if not unknown:
            self.time += 1
            if self.cell[0] == 0 and self.cell[1] == 0 and self.cow > 0:
                reward = 100 * self.cow
                self.num_of_cows -= self.cow
                self.cow = 0
                if self.num_of_cows == 0:
                    done = True

            if self.deck[self.cell[0], self.cell[1]] > 0:
                self.cow_place.remove((self.cell[0], self.cell[1]))
                self.cow += self.deck[self.cell[0], self.cell[1]]
                reward = 10
                self.deck[self.cell[0], self.cell[1]] = 0
            self.update()

        return self.state, reward, done, action
