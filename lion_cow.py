import numpy as np
moves = [[0, 1], [1, 0], [-1, 0], [0, -1]]


def hash(x, y, len):
    len += 1
    return x + y*len


class Game():
    def __init__(self, tot_row=3, tot_col=3, num_of_cows=1):
        self.world_row = tot_row - 1
        self.world_col = tot_col - 1
        self.cell = [0, 0]
        self.cow = 0
        self.cow_place = []
        self.state_space = (self.world_row + 1) * (self.world_col + 1) + 1
        self.deck = np.zeros((tot_row, tot_col))
        for i in range(num_of_cows):
            x = np.random.randint(0, self.world_row) + 1
            y = np.random.randint(0, self.world_col) + 1
            print(x, y)
            self.cow_place.append((x, y))
            self.deck[x, y] += 1
        self._copy_cow_place = self.cow_place
        self._copy_deck = np.copy(self.deck)
        self._copy_num_of_cows = num_of_cows
        self.num_of_cows = num_of_cows
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

    def update(self):
        m = 0
        for i in range(len(self.cow_place)):
            m += i * self.state_space + hash(self.cow_place[i][0], self.cow_place[i][1], self.world_col)
        self.state = (m, hash(self.cell[0], self.cell[1], self.world_col))

    def step(self, action):
        reward = 0
        done = False
        unknown = self.change(moves[action])
        if not unknown:
            self.time += 1
            if self.cell[0] == 0 and self.cell[1] == 0 and self.cow > 0:
                reward = 100
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

        return self.state, reward, done