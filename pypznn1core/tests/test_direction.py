import pytest
from pypznn1core.api import Direction

def test_get_directions():
    actual = Direction.get_directions()
    expected = [
        Direction.UP,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT,
    ]
    expected == actual

@pytest.mark.parametrize(
    [
        "direction",
        "opposite"
    ],
    [
        pytest.param(Direction.UP, Direction.DOWN),
        pytest.param(Direction.RIGHT, Direction.LEFT),
        pytest.param(Direction.DOWN, Direction.UP),
        pytest.param(Direction.LEFT, Direction.RIGHT),
    ]
)
def test_get_opposite(direction, opposite):
    actual = direction.get_opposite()
    assert opposite == actual
