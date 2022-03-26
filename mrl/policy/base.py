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
    if self.normalizer is None: # TODO move it to the buffer (when store experience)
      logging.warn('Normalizer not Defined')
    self.logger = logger
    self.optimize_times = 0

  def update(self, sample_size, buffer):  # TODO ignore warmup
    states, actions, rewards, next_states, masks= buffer.sample(
      sample_size, to_torch=False)
    if self.normalizer is not None:
      states = self.normalizer(states, update=False).astype(np.float32)
      next_states = self.normalizer(
        next_states, update=False).astype(np.float32)
    states, actions, rewards, next_states, masks = (self.torch(states), self.torch(actions),
                                                     self.torch(rewards), self.torch(
                                                       next_states),
                                                     self.torch(masks))
    result = self.learn(states, actions, rewards, next_states, masks)
    self.optimize_times += 1
    if self.logger is not None:
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
  def learn(self, states, actions, rewards, next_states, masks):
    pass

  @abstractmethod
  def forward(self, state):
    pass
