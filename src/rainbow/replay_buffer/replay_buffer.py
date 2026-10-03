import numpy as np

class ReplayBuffer():
    def __init__(self, config):
        self.transitions = []
        self.mini_batch_size = config['replay_buffer']['mini_batch_size']

    def store_transition(self, observation, action, reward, next_observation):
        self.transitions.append((observation, action, reward, next_observation))

    def select_random_mini_batch(self):

        if len(self.transitions) < self.mini_batch_size:
            return 
        else:
            indices = np.random.choice(
                len(self.transitions),
                size=self.mini_batch_size,
                replace=False
            )

            mini_batch = []

            for index in indices:
                transition = self.transitions[index]
                mini_batch.append(transition)

            return mini_batch


    
