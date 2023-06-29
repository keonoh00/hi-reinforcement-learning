import numpy as np


class BlobEnv:
    """
    BlobEnv

    Observation Space:
        1. Relative position of the food to the player (x1, y1)
        2. Relative position of the enemy to the player (x2, y2)

    Action Space:
        0: move up
        1: move down
        2: move left
        3: move right
        4: move right and up
        5: move left and down
        6: move left and up
        7: move right and down
        8: do nothing

    Parameters
    ----------
    area_size : int
        Size of the area.
    """

    def __init__(self, area_size):
        if area_size is None:
            raise ValueError("Area size must be provided.")

        self.observation_space = 2
        self.action_space = 9

        self.area_size = area_size

        self.x = np.random.randint(0, area_size)
        self.y = np.random.randint(0, area_size)

    def __str__(self):
        """
        __str__
        Mostly for debugging purposes, and returns the string representation of the position of the blob.
        """
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        """
        __sub__
        Subtraction
        """
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        """
        __eq__
        Equality
        """
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        """
        action

        0: move up
        1: move down
        2: move left
        3: move right
        4: move right and up
        5: move left and down
        6: move left and up
        7: move right and down
        8: do nothing
        """
        if choice == 0:
            self.move(x=0, y=-1)
        elif choice == 1:
            self.move(x=0, y=1)
        elif choice == 2:
            self.move(x=-1, y=0)
        elif choice == 3:
            self.move(x=1, y=0)
        elif choice == 4:
            self.move(x=1, y=-1)
        elif choice == 5:
            self.move(x=-1, y=1)
        elif choice == 6:
            self.move(x=-1, y=-1)
        elif choice == 7:
            self.move(x=1, y=1)
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x, y):
        """
        move
        """
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # When the blob goes out of bounds, it will be placed on the opposite side of the area.
        if self.x < 0:
            self.x = 0
        elif self.x > self.area_size - 1:
            self.x = self.area_size - 1

        if self.y < 0:
            self.y = 0
        elif self.y > self.area_size - 1:
            self.y = self.area_size - 1

    def reset(self):
        """
        reset
        """
        self.x = np.random.randint(0, self.area_size)
        self.y = np.random.randint(0, self.area_size)

        return np.array((self.x, self.y))

    def step(self, action):
        """
        step
        """
        self.action(action)

        return np.array((self.x, self.y)), self.reward(), self.done()

    def reward(self):
        """
        reward
        """
        if self.x == 0 and self.y == 0:
            return 1
        else:
            return -1

    def done(self):
        """
        done
        """
        if self.x == 0 and self.y == 0:
            return True
        else:
            return False
