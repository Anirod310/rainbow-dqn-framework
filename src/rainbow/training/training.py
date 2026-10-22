from config import config
from ..factory import Factory
import torch
import numpy as np

#----------Object construction part---------

factory = Factory(config)

env = factory.create_environment()
if config['environment']['wrappers']['fire_reset']['enabled']:
    env = factory.create_fire_reset_wrapper()

online_model = factory.create_online_model()
target_model = factory.create_target_model()

dummy_tensor = torch.randn(1, 4, 84, 84)#to initialize the model's weights since we use Lazy Linear | Not on device since models are put on device in the learner

online_model(dummy_tensor)
target_model(dummy_tensor)

target_model.load_state_dict(online_model.state_dict())

agent = factory.create_agent()

replay_buffer = factory.create_replay_buffer()

learner = factory.create_learner()

print(f"Using device: {learner.device}")


#----------------Training loop---------------------------

num_train_episodes = config['training']['num_episodes']

average_reward = 0

frame_count = 0

loss = 0

for episode in range(num_train_episodes):

    observation, info = env.reset()

    done = False
    episode_rewards = 0

    while(not done):

        action = agent.select_action(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)
        frame_count += 4 #Because each step takes 4 frames 
        agent.epsilon = max(config['agent']['epsilon_end'], config['agent']['epsilon_start']-frame_count/config['agent']['annealing_duration']*(config['agent']['epsilon_start'] - config['agent']['epsilon_end']))

        episode_rewards += reward
        
        if terminated or truncated:
            done = True

        replay_buffer.store_transition(observation, action, np.sign(reward), next_observation, terminated, truncated)

        observation = next_observation

        if len(replay_buffer.memory) >= 50000:
            observation_batch, action_batch, reward_batch, next_observation_batch, terminated_batch, truncated_batch = replay_buffer.select_random_mini_batch()
            loss = learner.learning_loop(observation_batch, action_batch, reward_batch, next_observation_batch, terminated_batch, truncated_batch)

    average_reward += episode_rewards

    if (episode + 1) % 1000 == 0:
        average_reward = average_reward / 1000
        print(f"Episode : {episode + 1} | Frames : {frame_count} | Episode Rewards : {episode_rewards} | Average Reward over last 1000 episodes : {average_reward} | loss : {loss} | Epsilon : {agent.epsilon}\n")
        average_reward = 0
        torch.save(online_model.state_dict(), config['evaluate']['weights_path'])


torch.save(online_model.state_dict(), config['evaluate']['weights_path'])




