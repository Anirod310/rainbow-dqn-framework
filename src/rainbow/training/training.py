from config import config
from ..factory import Factory

#----------Object construction part---------

factory = Factory(config)

env = factory.create_environment()

preprocessor = factory.create_preprocessor()

online_model = factory.create_model()
target_model = factory.create_model()

agent = factory.create_agent()

replay_buffer = factory.create_replay_buffer()

learner = factory.create_learner()

print(f"Using device: {learner.device}")

#-------------------------------------------

num_train_episodes = config['training']['num_episodes']

average_reward = 0

for episode in range(num_train_episodes):

    obs, info = env.reset()
    processed_obs = preprocessor.reset(obs)

    if agent.epsilon > config['agent']['epsilon_end']:
        agent.epsilon = config['agent']['epsilon_start'] - (config['agent']['epsilon_start']-config['agent']['epsilon_end']) * episode / num_train_episodes

    done = False
    episode_rewards = 0

    while(not done):

        action = agent.select_action(processed_obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        processed_next_obs = preprocessor.process(next_obs)

        episode_rewards += reward
        
        if terminated or truncated:
            done = True

        replay_buffer.store_transition(processed_obs, action, reward, processed_next_obs, terminated, truncated)

        processed_obs = processed_next_obs

        if len(replay_buffer.memory)>=config['replay_buffer']['mini_batch_size']:
            mini_batch = replay_buffer.select_random_mini_batch()
            learner.learning_loop(mini_batch)

    average_reward += episode_rewards

    if (episode%100) == 0:
        average_reward = average_reward / 100
        print(f"Episode : {episode} | Episode Rewards : {episode_rewards} | Average Reward over last 100 episodes : {average_reward}\n")
        average_reward = 0




