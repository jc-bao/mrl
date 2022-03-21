import torch
from torch import nn
import wandb
from abc import ABC, abstractmethod

class BasePolicy(ABC, nn.Module):
    def __init__(self, config, logger = None) -> None:
        super().__init__()
        self.config = config
        self.logger = logger
        self.optimize_times = 0

    def update(self, sample_size, buffer): # TODO ignore warmup
        states, actions, rewards, next_states, gammas = buffer.sample(sample_size)
        result = self.learn(states, actions, rewards, next_states, gammas)
        self.optimize_times += 1
        if not self.logger is None:
            self.logger.log({
                **result,
                'Train/Optimize Times': self.optimize_times
            })

    def torch(self, x):
        if isinstance(x, torch.Tensor): return x
        return torch.FloatTensor(x).to(self.config.device)

    def numpy(self, x):
        return x.cpu().detach().numpy()

    @abstractmethod
    def learn(self, states, actions, rewards, next_states, gammas):
        pass


    @abstractmethod
    def forward(self, state):
        pass