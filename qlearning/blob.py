import numpy as np


class Blob:
    def __init__(self, area_size):
        if area_size is None:
            raise ValueError("Area size must be provided.")

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

    def action(self, choice):
        """
        action

        0: move right and up
        1: move left and down
        2: move left and up
        3: move right and down
        """
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

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
