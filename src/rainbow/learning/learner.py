import torch
import numpy as np

class Learner():
    def __init__(self, agent, target_model, config, device):

        self.agent = agent
        self.target_model = target_model
        self.config = config
        self.device = device

        self.agent.online_model.to(self.device)
        self.target_model.to(self.device)

        opt_config = config['learning']['optimizer']
        self.optimizer = getattr(torch.optim, opt_config['name'])(
            agent.online_model.parameters(),
            **opt_config[opt_config['name']],
        )

        self.loss = torch.nn.HuberLoss() #Not exactly the same loss as the paper's one but still rlly good 
        self.target_update_cpt = 0

    def learning_loop(self, observation_batch, action_batch, reward_batch, next_observation_batch, terminated_batch, truncated_batch):

        observation_batch = observation_batch.astype(np.float32)
        next_observation_batch = next_observation_batch.astype(np.float32)

        #----------Shape Tests------------#
        '''print("Observation batch:", observation_batch.shape, observation_batch.dtype,
            observation_batch.min(), observation_batch.max())

        print("Next observation batch:", next_observation_batch.shape,
            next_observation_batch.dtype, next_observation_batch.min(),
            next_observation_batch.max()) '''
        #---------------------------------#

        q_values = self.agent.online_model(torch.tensor(observation_batch, dtype=torch.float32).to(self.device))

        chosen_q_values = q_values[ #Creates a tensor with only the 32 chosen q value so we go from (32, 4) to (32, )
            torch.arange(len(action_batch), device=self.device),
            torch.tensor(action_batch, dtype=torch.long, device=self.device)
        ].unsqueeze(1)

        with torch.no_grad():

            terminated_tensor_mask = torch.logical_not(torch.tensor(terminated_batch, dtype=torch.bool).unsqueeze(1).to(self.device)).float()#to invert zero and ones

            tensor_reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)

            tensor_next_observation_batch = torch.tensor(next_observation_batch, dtype=torch.float32).to(self.device)

            if self.config['learning']['DDQN']['enabled']:

                next_online_actions_indices = torch.max(self.agent.online_model(tensor_next_observation_batch),dim=1, keepdim=True).indices

                next_target_q_values = self.target_model(tensor_next_observation_batch)
                selected_next_q_values = torch.gather(next_target_q_values, dim=1, index= next_online_actions_indices)

                target_values = tensor_reward_batch + self.config['learning']['gamma'] * selected_next_q_values * terminated_tensor_mask

            else :
                target_values = tensor_reward_batch + self.config['learning']['gamma'] * (torch.max(self.target_model(tensor_next_observation_batch), dim=1).values) * terminated_tensor_mask
                                          

        loss = self.loss(chosen_q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_update_cpt += 1

        if self.target_update_cpt == self.config['learning']['target_network_update_freq']:
            self.update_target_model()
            self.target_update_cpt = 0

        return loss

    def update_target_model(self):
        self.target_model.load_state_dict(self.agent.online_model.state_dict())


        
