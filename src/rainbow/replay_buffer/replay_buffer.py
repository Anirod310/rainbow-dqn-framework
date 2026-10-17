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
            indices = np.random.randint(0, len(self.memory), size=self.mini_batch_size)

            mini_batch = []

            for index in indices:
                transition = self.memory[index]
                mini_batch.append(transition)

            observation_batch, action_batch, reward_batch, next_observation_batch, terminated_batch, truncated_batch = zip(*mini_batch)

            observation_batch = np.stack(observation_batch)
            next_observation_batch = np.stack(next_observation_batch)

            #To convert them from tuples to arrays
            action_batch = np.asarray(action_batch) 
            reward_batch = np.asarray(reward_batch)
            terminated_batch = np.asarray(terminated_batch)
            truncated_batch = np.asarray(truncated_batch)

            return observation_batch, action_batch, reward_batch, next_observation_batch, terminated_batch, truncated_batch


    
