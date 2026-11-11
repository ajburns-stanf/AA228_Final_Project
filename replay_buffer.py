"""Bounded replay of pixel transitions, sampled as within-episode sequences."""
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=10_000, sequence_length=8, seed=0):
        if capacity < sequence_length or sequence_length < 1:
            raise ValueError("Require capacity >= sequence_length >= 1")
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.transitions = deque(maxlen=capacity)
        self.episode = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.transitions)

    def add(self, obs, action, reward, next_obs, terminated, truncated=False):
        # Copy observations because simulators may reuse image storage.
        self.transitions.append((np.array(obs, copy=True),
                                 np.array(action, dtype=np.float32, copy=True),
                                 float(reward), np.array(next_obs, copy=True),
                                 float(terminated), self.episode))
        if terminated or truncated:
            self.episode += 1

    def sample(self, batch_size):
        """Return (images, actions, rewards, terminated), or None if no window exists.

        T transitions require T+1 images: s0, s1, ..., sT. The agent uses
        adjacent GRU outputs as the current and next features for each action.

        Samples use replacement. Episode truncation prevents crossing a reset,
        but only true termination masks bootstrapping in the agent's update.
        """
        data = list(self.transitions)
        length = self.sequence_length
        starts = [i for i in range(len(data) - length + 1)
                  if data[i][-1] == data[i + length - 1][-1]]
        if not starts:
            return None
        chosen = self.rng.choice(starts, size=batch_size)
        windows = [data[i:i + length] for i in chosen]
        images = np.asarray([[window[0][0]] + [row[3] for row in window]
                             for window in windows])
        actions, rewards, terminated = (
            np.asarray([[row[k] for row in window] for window in windows])
            for k in (1, 2, 4)
        )
        return images, actions, rewards, terminated
