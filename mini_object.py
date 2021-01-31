from typing import Union, List, Dict

class Slope():
    __slots__ = ('dx', 'dy')
    def __init__(self, dx: int, dy:int):
        self.dx = dx
        self.dy = dy

class Point():
    __slots__ = ('x', 'y')
    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)

    def copy(self) -> 'Point':
        x = self.x
        y = self.y
        return Point(x, y)
    
    def distance(self, other: Union['Point', tuple]) -> float:
        if isinstance(other, tuple) and len(other) == 2:
            diff_x = self.x - other[0]
            diff_y = self.y - other[1]
        elif isinstance(other, Point):
            diff_x = self.x - other.x
            diff_y = self.y - other.y
        else:
            raise TypeError('Distance module could only be used on "Point" or "tuple".')
        return (diff_x**2 + diff_y**2)**(1/2)

    def to_dict(self) -> Dict[str, int]:
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'Point':
        return Point(d['x'], d['y'])

    def __eq__(self, other: Union['Point', tuple]) -> bool:
        # accept tuple and Point type
        if isinstance(other, tuple) and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        elif isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        else:
            return False
    
    def __sub__(self, other: Union['Point', tuple]) -> 'Point':
        # accept tuple and Point type
        if isinstance(other, tuple) and len(other) == 2:
            return Point(self.x - other[0], self.y - other[1])
        elif isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            raise TypeError('Point only substract to "tuple" or "Point" type.')

    def __rsub__(self, other: tuple) -> 'Point':
        # reverse sub for tuple minus Point
        return Point(other[0] - self.x, other[1] - self.y)
    
    def __str__(self) -> str:
        return 'Point({}, {})'.format(self.x, self.y)

### These lines are defined such that facing "up" would be L0 ###
# Create 16 lines to be able to "see" around
VISION_16 = (
#   L0            L1             L2             L3
    Slope(-1, 0), Slope(-2, 1),  Slope(-1, 1),  Slope(-1, 2),
#   L4            L5             L6             L7      
    Slope(0, 1),  Slope(1, 2),   Slope(1, 1),   Slope(2, 1),
#   L8            L9             L10            L11
    Slope(1, 0),  Slope(2, -1),  Slope(1, -1),  Slope(1, -2),
#   L12           L13            L14            L15
    Slope(0, -1), Slope(-1, -2), Slope(-1, -1), Slope(-2, -1)
)

# Create 8 lines to be able to "see" around
# Really just VISION_16 without odd numbered lines
VISION_8 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i%2==0])

# Create 4 lines to be able to "see" around
# Really just VISION_16 but removing anything not divisible by 4
VISION_4 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i%4==0])
