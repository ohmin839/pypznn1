from enum import Enum
import os
import random

class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = -1
    LEFT = -2

    @classmethod
    def get_directions(cls):
        return (Direction.UP,
                Direction.RIGHT,
                Direction.DOWN,
                Direction.LEFT)

    def get_opposite(self):
        return Direction(self.value * -1)

class Puzzle:
    def __init__(self, n):
        self._n = n
        self.reset()
        self._steps = 0

    def reset(self):
        self._grid = Puzzle.get_initial_grid(self._n)
        self._empty_point = (self._n, self._n)

    def can_move(self, direction):
        i, j = self._empty_point[0], self._empty_point[1]

        if direction == Direction.UP:
            i += 1
        elif direction == Direction.RIGHT:
            j -= 1
        elif direction == Direction.DOWN:
            i -= 1
        elif direction == Direction.LEFT:
            j += 1

        return self._grid[i][j] > 0

    def move(self, direction):
        if self.can_move(direction):
            o_i, o_j = self._empty_point[0], self._empty_point[1]
            n_i, n_j = o_i, o_j

            if direction == Direction.UP:
                n_i += 1
            if direction == Direction.RIGHT:
                n_j -= 1
            if direction == Direction.DOWN:
                n_i -= 1
            if direction == Direction.LEFT:
                n_j += 1

            n_v = self._grid[n_i][n_j]
            self._grid[o_i][o_j] = n_v
            self._grid[n_i][n_j] = self.empty_value
            self._empty_point = (n_i, n_j)
            self._steps += 1

        return self

    def has_completed(self):
        return self._grid == Puzzle.get_initial_grid(self._n)

    def move_random(self, prev_direction=None):
        options = [d for d in Direction.get_directions() if self.can_move(d)]
        if prev_direction:
            options.remove(prev_direction.get_opposite())
        next_direction = random.choice(options)
        self.move(next_direction)
        return next_direction

    def shuffle(self, shuffles):
        self.reset()
        prev_direction = None
        for _ in range(shuffles):
            next_direction = self.move_random(prev_direction)
            prev_direction = next_direction        
        self._steps = 0

    def __repr__(self):
        formatted =  (os.linesep*2).join(
            (' '*2).join(
                ' '*2 if i == self.empty_value else f'{i}'.rjust(2) for i in r
                ) for r in self.grid)
        return f'{os.linesep}{formatted}{os.linesep}'

    @property
    def size(self):
        return self._n

    @property
    def empty_value(self):
        return self._n ** 2

    @property
    def grid(self):
        return [r[1:-1] for r in self._grid[1:-1]]

    @property
    def steps(self):
        return self._steps

    @classmethod
    def get_initial_grid(cls, n):
        grid = [[0 for _ in range(n+2)] for _ in range(n+2)]
        for i in range(1, n+1):
            for j in range(1, n+1):
                grid[i][j] = (i-1) * n + j
        return grid
