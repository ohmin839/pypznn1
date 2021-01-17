import numpy as np
from pznn1.core import Direction, Puzzle

def dir2idx(direction):
    if direction == Direction.UP:
        return 0
    elif direction == Direction.RIGHT:
        return 1
    elif direction == Direction.DOWN:
        return 2
    elif direction == Direction.LEFT:
        return 3

def idx2dir(index):
    if index == 0:
        return Direction.UP
    elif index == 1:
        return Direction.RIGHT
    elif index == 2:
        return Direction.DOWN
    elif index == 3:
        return Direction.LEFT

def grid2tuple(grid):
    return tuple(grid.reshape(-1))

def reverse_shuffle(puzzle, shuffles):
    directions = puzzle.shuffle(shuffles)
    return list(reversed([d.get_opposite() for d in directions]))

