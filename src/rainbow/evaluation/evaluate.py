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
print(f"Using device: {agent.device}")

#-------------------------------------------


num_eval_episodes = config['evaluate']['num_episodes']

with torch.no_grad():

    for episode in range(num_eval_episodes):
        ...



