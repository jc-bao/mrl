import torch
from torch import nn
import numpy as np
import logging
from abc import ABC, abstractmethod


class BasePolicy(ABC, nn.Module):
  def __init__(self, config, normalizer=None, logger=None) -> None:
    super().__init__()
    self.config = config
    self.normalizer = normalizer
    if self.normalizer is None:
      logging.warn('Normalizer not Defined')
    self.logger = logger
    self.optimize_times = 0

  def update(self, sample_size, buffer):  # TODO ignore warmup
    states, actions, rewards, next_states, gammas = buffer.sample(
      sample_size, to_torch=False)
    if not self.normalizer is None:
      states = self.normalizer(states, update=False).astype(np.float32)
      next_states = self.normalizer(
        next_states, update=False).astype(np.float32)
    states, actions, rewards, next_states, gammas = (self.torch(states), self.torch(actions),
                                                     self.torch(rewards), self.torch(
                                                       next_states),
                                                     self.torch(gammas))
    result = self.learn(states, actions, rewards, next_states, gammas)
    self.optimize_times += 1
    if not self.logger is None:
      self.logger.log({
        **result,
        'Train/Optimize Times': self.optimize_times
      })

  def torch(self, x):
    if isinstance(x, torch.Tensor):
      return x
    return torch.FloatTensor(x).to(self.config.device)

  def numpy(self, x):
    return x.cpu().detach().numpy()

  @abstractmethod
  def learn(self, states, actions, rewards, next_states, gammas):
    pass

  @abstractmethod
  def forward(self, state):
    pass
