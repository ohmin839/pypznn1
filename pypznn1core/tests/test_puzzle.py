import pytest
from pypznn1core.api import Direction, Puzzle

@pytest.mark.parametrize(
    [
        "size",
        "expected"
    ],
    [
        pytest.param(
            3, [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ], 
        ),
        pytest.param(
            4, [
                [0,  0,  0,  0,  0,  0],
                [0,  1,  2,  3,  4,  0],
                [0,  5,  6,  7,  8,  0],
                [0,  9, 10, 11, 12,  0],
                [0, 13, 14, 15, 16,  0],
                [0,  0,  0,  0,  0,  0], 
            ],
        ),
        pytest.param(
            5, [
                [0,  0,  0,  0,  0,  0,  0],
                [0,  1,  2,  3,  4,  5,  0], 
                [0,  6,  7,  8,  9, 10,  0], 
                [0, 11, 12, 13, 14, 15,  0], 
                [0, 16, 17, 18, 19, 20,  0], 
                [0, 21, 22, 23, 24, 25,  0], 
                [0,  0,  0,  0,  0,  0,  0], 
            ],
        ),
    ]
)
def test_get_initial_grid(size, expected):
    actual = Puzzle.get_initial_grid(size)
    assert expected == actual

@pytest.mark.parametrize(
    [
        "size",
    ],
    [
        pytest.param(3),
        pytest.param(4),
        pytest.param(5),
    ]
)
def test_constructor(size):
    pz = Puzzle(size)
    assert pz.size == size
    assert pz.empty_value == size ** 2
    assert pz.grid == [[i for i in range((r*size)+1, (r+1)*size+1 )] for r in range(size)]
    assert pz.steps == 0
    assert pz.has_completed()
