import torch

class Learner():
    def __init__(self, agent, target_model, config, device):

        self.agent = agent
        self.target_model = target_model
        self.config = config
        self.device = device

        self.agent.online_model.to(self.device)
        self.target_model.to(self.device)

        self.optimizer = torch.optim.RMSprop(agent.online_model.parameters(), 
                                             lr=config['learning']['learning_rate'], 
                                             alpha=config['learning']['squared_gradient_momentum'], 
                                             eps=config['learning']['min_squared_gradient'], 
                                             momentum=config['learning']['gradient_momentum'], )
        self.loss = torch.nn.HuberLoss() #Not exactly the same loss as the paper's one but still rlly good 
        self.target_update_cpt = 0

    def learning_loop(self, mini_batch):
        q_values = [self.agent.online_model(torch.tensor(transition[0], dtype=torch.float32).unsqueeze(0).to(self.device)) for transition in mini_batch]

        chosen_q_values = torch.stack([q_value[0, transition[1]] for q_value, transition in zip(q_values, mini_batch)])

        with torch.no_grad():
            target_values = torch.tensor([transition[2] + self.config["learning"]["gamma"] * (torch.max(self.target_model(torch.tensor(transition[3], dtype=torch.float32).unsqueeze(0).to(self.device)))
                        if not (transition[4] or transition[5]) else 0) for transition in mini_batch], dtype=torch.float32).to(self.device)

        loss = self.loss(chosen_q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_update_cpt += 1

        if self.target_update_cpt == self.config['learning']['target_network_update_freq']:
            self.update_target_model()
            self.target_update_cpt = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.agent.online_model.state_dict())


        
