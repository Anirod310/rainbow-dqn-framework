from config import config
from ..factory import Factory
import torch

#----------Object construction part---------

factory = Factory(config)

seed = factory.set_random_seed()

env = factory.create_environment()
if config['environment']['wrappers']['fire_reset']['enabled']:
    env = factory.create_fire_reset_wrapper()


checkpoint = torch.load(config['evaluate']['checkpoint_path'],
                        weights_only=True)


online_model = factory.create_online_model()
online_model.load_state_dict(checkpoint["model_state_dict"])
online_model.eval()

agent = factory.create_agent()
agent.epsilon = 0
print(f"Using device: {agent.device}")
online_model.to(agent.device)

#-------------------------------------------


num_eval_episodes = config['evaluate']['num_episodes']

average_rewards = 0

print(f"Configuration : seed : {checkpoint['seed']} | Last average training reward : {checkpoint['last_interval_training_episode_average_reward']} | Frames at the last checkpoint saved : {checkpoint['emulator_frames']}\n")

with torch.no_grad():

    for episode in range(num_eval_episodes):

        if episode == 0:
            observation, info = env.reset(seed=seed)
        else:
            observation, info = env.reset()

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
        
    average_rewards = average_rewards / num_eval_episodes

    checkpoint["evaluation_mean_reward"] = average_rewards
    torch.save(checkpoint, config['evaluate']['checkpoint_path'])

    print(f"Average Evaluation reward : {checkpoint['evaluation_mean_reward']}")
    average_rewards = 0

    env.close()

