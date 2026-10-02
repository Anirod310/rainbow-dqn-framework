import numpy as np

class ReplayBuffer():
    def __init__(self, config):
        self.transitions = []
        self.mini_batch_size = config['replay_buffer']['mini_batch_size']

    def store_transition(self, observation, action, reward, next_observation):
        self.transitions.append((observation, action, reward, next_observation))

    def select_random_mini_batch(self):
        mini_batch = []
        for loop in range(self.mini_batch_size):
            mini_batch.append((self.transitions[np.random.randint(len(self.transitions))]))

        return mini_batch