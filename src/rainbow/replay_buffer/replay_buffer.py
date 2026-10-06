import numpy as np

class ReplayBuffer():
    def __init__(self, config):
        self.memory = []
        self.pos = 0
        self.memory_capacity = config['replay_buffer']['memory_capacity']
        self.mini_batch_size = config['replay_buffer']['mini_batch_size']

    def store_transition(self, observation, action, reward, next_observation, terminated, truncated):
        if(len(self.memory) < self.memory_capacity):
            self.memory.append((observation, action, reward, next_observation, terminated, truncated))
        else:
            self.memory[self.pos] = (observation, action, reward, next_observation, terminated, truncated)
            self.pos += 1
            if self.pos == len(self.memory):
                self.pos = 0

    def select_random_mini_batch(self):

        if len(self.memory) < self.mini_batch_size:
            return 
        else:
            indices = np.random.choice(
                len(self.memory),
                size=self.mini_batch_size,
                replace=False
            )

            mini_batch = []

            for index in indices:
                transition = self.memory[index]
                mini_batch.append(transition)

            return mini_batch


    
