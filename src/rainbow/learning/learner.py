import torch

class Learner():
    def __init__(self, agent, config, device):
        self.device = device
        self.config = config
        self.agent = agent
        self.agent.model.to(self.device)
        self.optimizer = torch.optim.RMSprop(agent.model.parameters())
        self.loss = torch.nn.MSELoss()

    def learning_loop(self, mini_batch):
        q_values = [self.agent.model(torch.tensor(transition[0], dtype=torch.float32).unsqueeze(0).to(self.device)) for transition in mini_batch]

        chosen_q_values = torch.stack([q_value[0, transition[1]] for q_value, transition in zip(q_values, mini_batch)])

        target_values = torch.tensor([transition[2] + self.config["learning"]["gamma"] * (torch.max(self.agent.model(torch.tensor(transition[3], dtype=torch.float32).unsqueeze(0).to(self.device)).detach())
                    if not (transition[4] or transition[5]) else 0) for transition in mini_batch], dtype=torch.float32).to(self.device)

        loss = self.loss(chosen_q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        
