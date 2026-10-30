from config import config
from ..factory import Factory
from pathlib import Path
import torch

#----------Object construction part---------

factory = Factory(config)

seed = factory.set_random_seed()

env = factory.create_environment()
if config['environment']['wrappers']['fire_reset']['enabled']:
    env = factory.create_fire_reset_wrapper()


checkpoint = torch.load(config['evaluate']['checkpoint_path'],
                        weights_only=False)


online_model, __ = factory.create_model()
online_model.load_state_dict(checkpoint["model_state_dict"])
online_model.eval()

agent = factory.create_agent()
agent.epsilon = 0
print(f"Using device: {agent.device}")
online_model.to(agent.device)

#-------------------------------------------


num_eval_episodes = config['evaluate']['num_episodes']
num_render_episodes = config['evaluate']['num_render_episodes']

checkpoint_name = Path(config['evaluate']['checkpoint_path']).stem

average_rewards = 0

print(f"----------Configuration----------\n"
      f"checkpoint : {checkpoint_name}\n"
      f"seed : {checkpoint['seed']}\n"
      f"Last average training reward : {checkpoint['last_interval_training_episode_average_reward']}\n"
      f"Frames at the last checkpoint saved : {checkpoint['emulator_frames']}\n")

with torch.no_grad():

    if config['environment']['render_mode'] == "human" : 
        for episode in range(10):
        
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
    
            print(f"Episode : {episode+1} | Episode reward : {episode_rewards}")
            average_rewards = 0

    else : 
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

