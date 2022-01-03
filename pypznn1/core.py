from enum import Enum
import numpy as np

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
        self.n = n
        self.empty_value = n**2
        self.reset()

    def reset(self):
        self.grid = Puzzle.get_initial_grid(self.n)
        self.empty_point = (self.n, self.n)

    def can_move(self, direction):
        i, j = self.empty_point[0], self.empty_point[1]

        if direction == Direction.UP:
            i += 1
        elif direction == Direction.RIGHT:
            j -= 1
        elif direction == Direction.DOWN:
            i -= 1
        elif direction == Direction.LEFT:
            j += 1

        if self.grid[i, j] == 0:
            return False
        else:
            return True

    def move(self, direction):
        if self.can_move(direction):
            o_i, o_j = self.empty_point[0], self.empty_point[1]
            n_i, n_j = o_i, o_j

            if direction == Direction.UP:
                n_i += 1
            if direction == Direction.RIGHT:
                n_j -= 1
            if direction == Direction.DOWN:
                n_i -= 1
            if direction == Direction.LEFT:
                n_j += 1

            n = self.grid[n_i, n_j]
            self.grid[o_i, o_j] = n
            self.grid[n_i, n_j] = self.empty_value
            self.empty_point = (n_i, n_j)

        return self

    def has_completed(self):
        return np.array_equal(self.grid, Puzzle.get_initial_grid(self.n))

    def move_random(self, prev_direction=None):
        options = [d for d in Direction.get_directions() if self.can_move(d)]
        if prev_direction:
            options.remove(prev_direction.get_opposite())
        next_direction = np.random.choice(options)
        self.move(next_direction)
        return next_direction

    def shuffle(self, shuffles):
        self.reset()
        prev_direction = None
        for _ in range(shuffles):
            next_direction = self.move_random(prev_direction)
            prev_direction = next_direction        

    def __repr__(self):
        grid = self.grid[1:-1, 1:-1].copy()
        return np.array2string(
                grid,
                formatter={"int": lambda x: "  " if x == self.empty_value else f"{x:2d}"})

    @classmethod
    def get_initial_grid(cls, n):
        grid = np.array(
                list(range(1, n**2+1)),
                dtype=np.int32).reshape(n, n)
        grid = np.pad(
                grid,
                ((1, 1), (1, 1)),
                "constant")
        return grid