import config as config
from ..factory import Factory

#----------Object construction part---------

factory = Factory(config)

env = factory.create_environment()

preprocessor = factory.create_preprocessor()

model = factory.create_model()

agent = factory.create_agent()

replay_buffer = factory.create_replay_buffer()

learner = factory.create_learner()

#-------------------------------------------

train_episode = config['training']['episode']

for loop in range(train_episode):
    obs, info = env.reset()
    processed_obs = preprocessor.reset(obs)

    done = False

    while(not done):

        action = agent.select_action(processed_obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        processed_next_obs = preprocessor.process(next_obs)

        if terminated or truncated:
            done = True

        replay_buffer.store_transition(processed_obs, action, reward, processed_next_obs, terminated, truncated)

        processed_obs = processed_next_obs

        mini_batch = replay_buffer.select_random_mini_batch()

        learner.learning_loop(mini_batch)


#TODO : select_random_mini_batch() can return nothing while the replay buffer has fewer than 32 transitions. learner.learning_loop(mini_batch) needs to handle that case.
        




