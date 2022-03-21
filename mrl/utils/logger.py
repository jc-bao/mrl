import numpy as np
import wandb

class WandbLogger:
    def __init__(self, config) -> None:
      self.config = config
      self.wandb = wandb.init(project='debug', name='new lib debug')
      self.steps = 0
      self.last_log_step = -1
      self.data_buffer = {}

    def log(self, data, steps = None):
      for k, v in data.items():
        if k in self.data_buffer.keys():
          if isinstance(v, (np.ndarray, list)):
            # if use list, then record average
            self.data_buffer[k] = np.append(self.data_buffer[k], v)
          else:
            # use float, then record last
            self.data_buffer[k] = v
        else:
          self.data_buffer[k] = np.array(v)
      if not steps is None:
        self.steps = steps
      if self.steps > self.last_log_step + self.config.epoch_len:
        for k, v in self.data_buffer.items():
          if isinstance(v, (np.ndarray, list)):
            # if use list, then record mid point of steps
            self.wandb.log({k:np.mean(v)}, step=self.steps)
          else:
            # use float, then record last
            self.wandb.log({k:v}, step=self.steps)
        self.data_buffer = {}
        self.last_log_step = self.steps