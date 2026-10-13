from config import config
from ..factory import Factory
import torch

#----------Object construction part---------

factory = Factory(config)

env = factory.create_environment()

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

for episode in range(num_train_episodes):

    obs, info = env.reset()

    if agent.epsilon > config['agent']['epsilon_end']:
        agent.epsilon = config['agent']['epsilon_start'] - (config['agent']['epsilon_start']-config['agent']['epsilon_end']) * episode / num_train_episodes

    done = False
    episode_rewards = 0

    while(not done):

        action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards += reward
        
        if terminated or truncated:
            done = True

        replay_buffer.store_transition(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs

        if len(replay_buffer.memory)>=config['replay_buffer']['mini_batch_size']:
            mini_batch = replay_buffer.select_random_mini_batch()
            learner.learning_loop(mini_batch)

    average_reward += episode_rewards

    if (episode%100) == 0:
        average_reward = average_reward / 100
        print(f"Episode : {episode} | Episode Rewards : {episode_rewards} | Average Reward over last 100 episodes : {average_reward}\n")
        average_reward = 0




