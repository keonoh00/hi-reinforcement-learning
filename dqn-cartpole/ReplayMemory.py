from collections import namedtuple, deque
import random

# Transition is a named tuple representing a single transition in our environment.
# It essentially maps (state, action) pairs to their (next_state, reward) result,
# with the state being the screen difference image as described later on.
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# ReplayMemory is a cyclic buffer of bounded size that holds the transitions observed recently.
# It also implements a .sample() method for selecting a random batch of transitions for training.
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
