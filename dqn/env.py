import time
import numpy as np
import cv2
from PIL import Image


class Blob:
    """
    Blob

    A blob is a player, food, or enemy.

    Attributes
    ----------
    area_size : int
        The size of the area in which the blob can move.
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

    """

    def __init__(self):
        """
        __init__
        """

        self.SIZE = 10
        self.observation_space = self.SIZE
        self.action_space = 9

        self.RETURN_IMAGES = True
        self.MOVE_PENALTY = 1
        self.ENEMY_PENALTY = 300
        self.FOOD_REWARD = 25
        self.OBSERVATION_SPACE_VALUES = (self.SIZE, self.SIZE, 3)  # 4

        self.PLAYER_N = 1  # player key in dict
        self.FOOD_N = 2  # food key in dict
        self.ENEMY_N = 3  # enemy key in dict

        self.d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        # enemy.move()
        # food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if (
            reward == self.FOOD_REWARD
            or reward == -self.ENEMY_PENALTY
            or self.episode_step >= 200
        ):
            done = True

        return new_observation, reward, done

    def render(self, display_connected=False):
        img = self.get_image()
        img = img.resize(
            (300, 300)
        )  # resizing so we can see our agent in all its glory.
        if display_connected:
            cv2.imshow("image", np.array(img))  # show it!
            cv2.waitKey(1)

        else:
            now = time.time()
            cv2.imwrite(
                f"./blob-{now}.jpg", np.array(img)
            )  # to make it actually show up.

    # FOR CNN #
    def get_image(self):
        env = np.zeros(
            (self.SIZE, self.SIZE, 3), dtype=np.uint8
        )  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[
            self.FOOD_N
        ]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[
            self.ENEMY_N
        ]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[
            self.PLAYER_N
        ]  # sets the player tile to blue
        img = Image.fromarray(
            env, "RGB"
        )  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
