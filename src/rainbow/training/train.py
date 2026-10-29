from config import config
from ..factory import Factory
from ..checkpoints import save_checkpoint
import torch
import numpy as np
import time
from tqdm import tqdm

#----------Object construction part---------

factory = Factory(config)

seed = factory.set_random_seed()

env = factory.create_environment()
if config['environment']['wrappers']['fire_reset']['enabled']:
    env = factory.create_fire_reset_wrapper()

online_model, target_model = factory.create_model()

agent = factory.create_agent()

replay_buffer = factory.create_replay_buffer()

learner = factory.create_learner()

print(f"Using device: {learner.device}")


#----------------Training loop---------------------------

num_train_episodes = config['training']['num_episodes']
total_frames = config['training']['total_frames']
frame_count = 0
step_count = 0

pbar = tqdm(total=total_frames, unit="frame", unit_scale=True)
last_frame_count = 0
episodes_in_interval = 0

episode_rewards_sum = 0
best_average_reward = float('-inf')

loss = 0

print_interval = max(1, num_train_episodes // 20)

start_time = time.perf_counter()

try:
    for episode in range(num_train_episodes):
        
        if frame_count >= total_frames:
            if episodes_in_interval > 0:
                average_reward = episode_rewards_sum / episodes_in_interval
                tqdm.write(
                    f"Final partial interval: {episodes_in_interval} completed episodes\n"
                    f"Frames: {frame_count}\n"
                    f"Average reward: {average_reward}"
                )

                if average_reward >  best_average_reward:

                    tqdm.write("best average reward reached ! Checkpoint saved")

                    save_checkpoint(online_model=online_model,
                            seed=seed,
                            episodes=num_train_episodes,
                            frame_count=frame_count,
                            average_reward=average_reward,
                            start_time=start_time,
                            config=config)

                    best_average_reward = average_reward
            else : 
                tqdm.write("Frame limit reached; no completed episodes in the final interval.")

            tqdm.write(f"Frame limit ({total_frames}) reached.")
            break

        if episode == 0:
            observation, info = env.reset(seed=seed)
        else:
            observation, info = env.reset()

        done = False
        episode_rewards = 0

        while(not done):

            action = agent.select_action(observation)

            next_observation, reward, terminated, truncated, info = env.step(action)    

            frame_count +=  config['environment']['frameskip_wrap'] if config['environment']['type'] == "atari" else 1 

            agent.epsilon = max(config['agent']['epsilon_end'], config['agent']['epsilon_start']-frame_count/config['agent']['annealing_duration']*(config['agent']['epsilon_start'] - config['agent']['epsilon_end']))

            episode_rewards += reward
            
            if terminated or truncated:
                done = True

            life_lost = info.get("life_lost", False)#Like this if we disable fire_reset it will just count on terminated

            replay_buffer.store_transition(observation, action, np.sign(reward) if config['environment']['type']=="atari" else reward, next_observation, terminated or life_lost, truncated)#We store life_lost as the end of a session in the learning so the agent understands that loosing a life isnt good

            observation = next_observation

            step_count += 1
            if len(replay_buffer.memory) >= config['training']['replay_learning_start'] and step_count % config['learning']['update_frequency'] == 0:
                observation_batch, action_batch, reward_batch, next_observation_batch, terminated_batch, truncated_batch = replay_buffer.select_random_mini_batch()
                loss = learner.learning_loop(observation_batch, action_batch, reward_batch, next_observation_batch, terminated_batch, truncated_batch)

        episode_rewards_sum += episode_rewards
        episodes_in_interval += 1

        elapsed_time = time.perf_counter() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        pbar.update(frame_count - last_frame_count)
        last_frame_count = frame_count
        
        if (episode + 1) % print_interval == 0:
            average_reward = episode_rewards_sum / print_interval
            tqdm.write(f"Episode : {episode + 1}"
                       f"\nFrames : {frame_count}"
                       f"\nEpisode Rewards : {episode_rewards}\n"
                       f"Average Reward over last {print_interval} episodes : {average_reward}"
                       f"\nloss : {loss}"
                       f"\nEpsilon : {agent.epsilon}"
                       f"\nTraining time : {hours}h {minutes}m {seconds}s")

            if average_reward >  best_average_reward:

                tqdm.write("best average reward reached ! Checkpoint saved")

                save_checkpoint(online_model=online_model,
                        seed=seed,
                        episodes=num_train_episodes,
                        frame_count=frame_count,
                        average_reward=average_reward,
                        start_time=start_time,
                        config=config)

                best_average_reward = average_reward

            
            episode_rewards_sum = 0
            episodes_in_interval = 0


    if average_reward >  best_average_reward:
        save_checkpoint(online_model=online_model,
                        seed=seed,
                        episodes=episode+1,
                        frame_count=frame_count,
                        average_reward=average_reward,
                        start_time=start_time,
                        config=config)

    pbar.close()
    
except KeyboardInterrupt:
    print("\n Training interrupted. Last best checkpoint saved\n")




