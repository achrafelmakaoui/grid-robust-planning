class GridEnvironment:

    def __init__(self, width, height, start, goal, obstacles=None):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles else set()

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos):
        return pos not in self.obstacles

    def neighbors(self, pos):
        x, y = pos
        moves = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        return [p for p in moves if self.in_bounds(p) and self.passable(p)]

    def move_cost(self, a, b):
        return 1  # every step costs 1


def make_easy_grid():
    obs = {(4, y) for y in range(7)}
    return GridEnvironment(10, 10, (0,0), (9,9), obs)


def make_medium_grid():
    obs = set()
    for y in range(11):
        obs.add((5, y))
    for y in range(4, 15):
        obs.add((10, y))
    obs.discard((0,0))
    obs.discard((14,14))
    return GridEnvironment(15, 15, (0,0), (14,14), obs)


def make_hard_grid():
    obs = set()
    hbars = [
        (range(16),  4,  {8,9}),
        (range(4,20),9,  {4,5}),
        (range(15),  14, {13,14}),
        (range(6,20),17, {18,19}),
    ]
    for xs, y, gaps in hbars:
        for x in xs:
            if x not in gaps:
                obs.add((x, y))
    vbars = [
        (4,  range(4,9),  {6}),
        (13, range(9,14), {11}),
        (17, range(4),    {2}),
    ]
    for x, ys, gaps in vbars:
        for y in ys:
            if y not in gaps:
                obs.add((x, y))
    obs.discard((0,0))
    obs.discard((19,19))
    return GridEnvironment(20, 20, (0,0), (19,19), obs)
