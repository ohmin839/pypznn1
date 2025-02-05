import argparse
from .api import Direction, Puzzle

def play(size, shuffles):
    pz = Puzzle(size)
    pz.shuffle(shuffles)

    while not pz.has_completed():
        print(pz)

        key = input(f'{pz.steps}> ')
        if key == 'w':
            direction = Direction.UP
        elif key == 'd':
            direction = Direction.RIGHT
        elif key == 's':
            direction = Direction.DOWN
        elif key == 'a':
            direction = Direction.LEFT
        elif key == 'q':
            print('Aborted.')
            return
        else:
            continue

        if pz.can_move(direction):
            pz.move(direction)

    print(pz)
    print(f'Completed!({pz.steps}steps)')

def main():
    parser = argparse.ArgumentParser(description='NxN Puzzle')
    parser.add_argument('--size', type=int, default=4, help='Specifies the height and width')
    parser.add_argument('--shuffles', type=int, default=128, help='Specifies the count of shuffles')
    args = parser.parse_args()
    play(args.size, args.shuffles)

if __name__ == '__main__':
    main()
