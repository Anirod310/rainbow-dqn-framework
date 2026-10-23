from config import config
from ..factory import Factory
import torch
import numpy as np
import time

#----------Object construction part---------

factory = Factory(config)

seed = factory.set_random_seed()

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

print_interval = max(1, num_train_episodes // 5)

start_time = time.perf_counter()

for episode in range(num_train_episodes):

    if episode == 0:
        observation, info = env.reset(seed=seed)
    else:
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

    elapsed_time = time.perf_counter() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    if (episode + 1) % print_interval == 0:
        average_reward = average_reward / print_interval
        print(f"Episode : {episode + 1} | Frames : {frame_count} | Episode Rewards : {episode_rewards} | Average Reward over last {print_interval} episodes : {average_reward} | loss : {loss} | Epsilon : {agent.epsilon}")
        print(f"Training times : {hours}h {minutes}m {seconds}s\n")
        if(episode+1) != num_train_episodes:
            average_reward = 0

evaluation_mean_reward = 0#Only available after an evaluate session
checkpoint = {
    "model_state_dict": online_model.state_dict(),
    "seed": seed,
    "episodes": num_train_episodes,
    "emulator_frames": frame_count,
    "last_interval_training_episode_average_reward": average_reward,
    "evaluation_mean_reward": evaluation_mean_reward,
    "training_duration_seconds": elapsed_time,
    "config":config
}

torch.save(checkpoint, config['evaluate']['checkpoint_path'])




