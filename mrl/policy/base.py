import torch
from torch import nn
from abc import ABC, abstractmethod

class BasePolicy(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.optimize_times = 0

    def update(self, sample_size, buffer): # TODO ignore warmup
        states, actions, rewards, next_states, gammas = buffer.sample(sample_size)
        self.learn(states, actions, rewards, next_states, gammas)
        self.optimize_times += 1

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