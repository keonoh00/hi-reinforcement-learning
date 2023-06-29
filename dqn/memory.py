import random
from collections import deque


class ReplayMemory:
    """
    Replay memory for DQN.
        This allows the agent to learn from past experiences.
        Used directly to train the agent.

    Parameters
    ----------
    capacity : int
        Maximum size of the memory.
    """

    def __init__(self, capacity=100000):
        """
        __init__

        Parameters
        ----------
        capacity : int
            Maximum size of the memory.
        """
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        """
        __len__

        Returns
        -------
        int
            Length of the memory.
        """
        return len(self.memory)

    def push(self, transition):
        """
        push

        Parameters
        ----------
        transition : tuple
            Tuple of (state, action, reward, next_state, done).
        """

        self.memory.append(transition)

    def sample_batch(self, batch_size):
        """
        sample_batch

        Parameters
        ----------
        batch_size : int
            Size of the batch.

        Returns
        -------
        batch : list
            Batch of transitions.
        """

        return random.sample(self.memory, batch_size)
