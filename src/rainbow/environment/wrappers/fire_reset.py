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

        life_lost = info["lives"] < self.lives

        if(life_lost and (not (terminated or truncated))):
            next_observation, fire_reward, terminated, truncated, info = self.env.step(1)

        self.lives = info["lives"]
        info["life_lost"] = life_lost
        return next_observation, reward, terminated, truncated, info