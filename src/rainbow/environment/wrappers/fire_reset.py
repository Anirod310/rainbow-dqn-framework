import gymnasium as gym

class FireResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = None

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        next_observation, reward, terminated, truncated, info = self.env.step(1)
        self.lives = info["lives"]

        return next_observation, info

    def step(self, action):

        next_observation, reward, terminated, truncated, info = self.env.step(action)

        if(info["lives"] < self.lives and (not (terminated or truncated))):
            next_observation, fire_reward, terminated, truncated, info = self.env.step(1)

        self.lives = info["lives"]
        return next_observation, reward, terminated, truncated, info