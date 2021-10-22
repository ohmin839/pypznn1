import argparse
import numpy as np
from pypznn1.core import Direction, Puzzle

def play(size, shuffles):
    pz = Puzzle(size)
    pz.shuffle(shuffles)

    steps = 0
    while not pz.has_completed():
        print(pz)

        key = input()
        if key == "w":
            direction = Direction.UP
        elif key == "d":
            direction = Direction.RIGHT
        elif key == "s":
            direction = Direction.DOWN
        elif key == "a":
            direction = Direction.LEFT
        elif key == "q":
            print("Aborted.")
            return
        else:
            continue

        if pz.can_move(direction):
            pz.move(direction)
            steps += 1

    print(pz)
    print(f"Completed!({steps}steps)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NxN Puzzle")
    parser.add_argument("--size", type=int, default=4, help="Specifies the height and width")
    parser.add_argument("--shuffles", type=int, default=80, help="Specifies the count of shuffles")
    args = parser.parse_args()
    play(args.size, args.shuffles)

