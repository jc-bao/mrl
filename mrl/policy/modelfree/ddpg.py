import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import logging

from mrl.policy import BasePolicy
from mrl.utils import GaussianProcess, ConstantSchedule


class DDPGPolicy(BasePolicy):
  def __init__(self, config, actor, actor_opt, critic, critic_opt, replay_buffer, normalizer=None, logger=None) -> None:
    super().__init__(config, normalizer, logger)
    # parameters
    self.config = config
    self.optimize_times = 0
    self.target_optimize_times = 0
    # networks
    self.actor = actor
    self.actor_target = deepcopy(actor)
    for p in self.actor_target.parameters():
      p.requires_grad = False
    self.actor_opt = actor_opt
    self.critic = critic
    self.critic_target = deepcopy(critic)
    for p in self.critic_target.parameters():
      p.requires_grad = False
    self.critic_opt = critic_opt
    # noise
    self.random_process = GaussianProcess(
      size=(self.config.num_envs, self.config.action_dim,),
      std=ConstantSchedule(config.action_noise))
    # tmp need remove!
    self.replay_buffer = replay_buffer

  def forward(self, state, random_explore=False, greedy=False):
    if self.training:
      if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
        return np.array([self.config.action_space.sample()
                         for _ in range(self.config.num_envs)])
      elif hasattr(self, 'ag_curiosity'):  # TODO 1 curiosity
        state = self.ag_curiosity.relabel_state(state)
    # normalize
    state = flatten_state(state)  # flatten goal environments
    if self.normalizer is not None:
      state = self.normalizer(state, update=self.training)
    # get action
    state = self.torch(state)
    action = self.numpy(self.actor(state))
    action_scale = self.config.max_action
    if self.training and not greedy:
      action = self.action_noise(action)
      if self.config.get('eexplore'):
        eexplore = self.config.eexplore
        if hasattr(self, 'ag_curiosity'):
          eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
        mask = (np.random.random(
                (action.shape[0], 1)) < eexplore).astype(np.float32)
        randoms = np.random.random(action.shape) * \
          (2 * action_scale) - action_scale
        action = mask * randoms + (1 - mask) * action

    return np.clip(action, -action_scale, action_scale)

  def learn(self, states, actions, rewards, next_states, gammas):
    with torch.no_grad():
      q_next = self.critic_target(
        next_states, self.actor_target(next_states))

    target = (rewards + gammas * q_next)
    target = torch.clamp(target, *self.config.clip_target_range)

    q = self.critic(states, actions)
    critic_loss = F.mse_loss(q, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()

    # Grad clipping
    if self.config.grad_norm_clipping > 0.:
      for p in [self.critic.parameters()]:
        clip_coef = self.config.grad_norm_clipping / \
          (1e-6 + torch.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(
        self.critic.parameters(), self.config.grad_value_clipping)

    self.critic_opt.step()

    for p in self.critic.parameters():  # TODO make these lines shorter
      p.requires_grad = False

    a = self.actor(states)
    if self.config.get('policy_opt_noise'):
      noise = torch.randn_like(
        a) * (self.config.policy_opt_noise * self.config.max_action)
      a = (a + noise).clamp(-self.config.max_action, self.config.max_action)

    actor_loss = -self.critic(states, a)[:, -1].mean()
    if self.config.action_l2_regularization:
      actor_loss += self.config.action_l2_regularization * \
        F.mse_loss(a / self.config.max_action, torch.zeros_like(a))

    self.actor_opt.zero_grad()
    actor_loss.backward()

    # Grad clipping
    if self.config.grad_norm_clipping > 0.:
      for p in [self.actor.parameters()]:
        clip_coef = self.config.grad_norm_clipping / \
          (1e-6 + torch.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(
        self.actor_params, self.config.grad_value_clipping)

    self.actor_opt.step()

    for p in self.critic.parameters():
      p.requires_grad = True

    if self.optimize_times % self.config.target_network_update_freq == 0:
      soft_update(self.actor_target, self.actor,
                  self.config.target_network_update_frac)
      soft_update(self.critic_target, self.critic,
                  self.config.target_network_update_frac)
      self.target_optimize_times += 1

    result = {
      'Train/Actor Loss': [self.numpy(actor_loss)],
      'Train/Critic Loss': [self.numpy(critic_loss)],
      'Train/Target Optimize Times': self.target_optimize_times,
    }
    return result

  def action_noise(self, act):
    factor = 1
    if self.config.varied_action_noise:
      n_envs = self.config.num_envs
      factor = np.arange(0., 1. + (1./n_envs), 1. /
                         (n_envs-1)).reshape(n_envs, 1)
    return act + (self.random_process.sample() * self.config.max_action * factor)[:len(act)]


def soft_update(target, src, factor):
  with torch.no_grad():
    for target_param, param in zip(target.parameters(), src.parameters()):
      target_param.data.mul_(1.0 - factor)
      target_param.data.add_(factor * param.data)


def flatten_state(state):
  if isinstance(state, dict):
    obs = state['observation']
    goal = state['desired_goal']
    return np.concatenate((obs, goal), -1)
  return state
