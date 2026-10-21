from config import config
from ..factory import Factory
import torch

#----------Object construction part---------

factory = Factory(config)

env = factory.create_environment()

online_model = factory.create_online_model()
online_model.load_state_dict(torch.load(config['evaluate']['weights_path']))
online_model.eval()

agent = factory.create_agent()
agent.epsilon = 0
print(f"Using device: {agent.device}")

#-------------------------------------------


num_eval_episodes = config['evaluate']['num_episodes']

average_rewards = 0

with torch.no_grad():

    for episode in range(num_eval_episodes):

        observation, infos = env.reset()
        done = False
        episode_rewards = 0


        while (not done):

            action = agent.select_action(observation)
            
            next_observation, reward, terminated, truncated, info = env.step(action)

            episode_rewards += reward

            if terminated or truncated:
                done = True

            observation = next_observation

        average_rewards += episode_rewards

        if (episode + 1) % 5 == 0:
            average_rewards = average_rewards / 5
            print(f"Episode : {episode+1} | Episode Reward : {episode_rewards} | Average reward over last 5 episodes : {average_rewards}")
            average_rewards = 0



