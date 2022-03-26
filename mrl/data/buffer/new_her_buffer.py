import os
from re import L

import numpy as np
import numpy.random as rd
import torch


class ReplayBuffer:
  def __init__(self, config, replay_size):
    self.now_len = 0
    self.next_idx = 0
    self.prev_idx = 0
    self.num_trajs = 0
    self.num_trans = 0
    self.if_full = False
    # basic params
    self.replay_size = config.replay_size
    self.action_dim: int = config.action_dim
    self.state_dim: int = config.state_dim  # note: include goal
    self.num_goals: int = config.num_goals
    self.num_envs: int = config.num_envs
    self.max_env_steps: int = config.max_env_steps
    self.single_goal_dim: int = config.single_goal_dim
    self.goal_dim: int = self.single_goal_dim * self.num_goals
    self.device: str = config.device
    # frame=state, traj_idx, rew, mask, action
    self.frame_size = self.state_dim + self.action_dim + 3
    self.header_size = self.state_dim + self.goal_dim +3

    # store headers TODO make the header update smartly
    self.headers = torch.empty(
      (replay_size//config.max_env_steps*2, self.header_size), dtype=torch.float32, device=self.device)
    # store state
    self.frame = torch.empty(
      (replay_size, self.frame_size), dtype=torch.float32, device=self.device)
    # tmp buffer
    self.frame_tmp = torch.empty(
      (self.num_envs, self.frame_size), dtype=torch.float32, device=self.device)
    self.traj_len_tmp = torch.zeros(
      (self.num_envs,), dtype=torch.int32, device=self.device)

  def parse_header(self, traj_header):
    traj_idx=traj_header[0]
    traj_len=traj_header[1]
    traj_env_goal=traj_header[2]
    traj_achieved_goals=traj_header[3:self.goal_dim + 3]
    traj_final_state=traj_header[self.goal_dim + 3:]
    return traj_idx, traj_len, traj_env_goal, traj_achieved_goals, traj_final_state

  def make_header(self, traj_idx, traj_len, traj_env_goal, traj_achieved_goals, traj_final_state):
    return torch.cat(
      (traj_idx, traj_len, traj_env_goal, traj_achieved_goals, traj_final_state), 
      dim=-1
    )

  def parse_frame(self, frame):
    assert frame.shape[-1] == (self.state_dim + \
                               self.action_dim + 3), 'frame shape error'
    state=frame[..., :self.state_dim]
    traj_idx=frame[self.state_dim]
    rew=frame[self.state_dim+1]
    mask=frame[self.state_dim + 2]
    action=frame[self.state_dim + 3:self.state_dim + 3 + self.action_dim]
    return state, traj_idx, rew, mask, action

  def extend_buffer(self, state, other):
    size=len(other)
    next_idx=self.next_idx + size

    if next_idx > self.replay_size:
      self.buf_state[self.next_idx:self.replay_size]=state[:self.replay_size - self.next_idx]
      self.buf_other[self.next_idx:self.replay_size]=other[:self.replay_size - self.next_idx]
      self.if_full=True

      next_idx=next_idx - self.replay_size
      self.buf_state[0:next_idx]=state[-next_idx:]
      self.buf_other[0:next_idx]=other[-next_idx:]
    else:
      self.buf_state[self.next_idx:next_idx]=state
      self.buf_other[self.next_idx:next_idx]=other
    self.next_idx=next_idx

  def add(self, state, next_state, act, rew, done):
    self.add_tmp(state, next_state, act, rew, done)
    for env_idx in torch.where(done):
      traj = self.frame_tmp[env_idx]
      self.num_trajs += 1
      header = self.make_header(self.num_trajs, traj_env_goal=

  def sample_batch(self, batch_size) -> tuple:
    indices=rd.randint(self.now_len - 1, size=batch_size)
    return (self.buf_other[indices, 0:1],
            self.buf_other[indices, 1:2],
            self.buf_other[indices, 2:],
            self.buf_state[indices],
            self.buf_state[indices + 1])

  def sample_batch_r_m_a_s(self):
    if self.prev_idx <= self.next_idx:
      r=self.buf_other[self.prev_idx:self.next_idx, 0:1]
      m=self.buf_other[self.prev_idx:self.next_idx, 1:2]
      a=self.buf_other[self.prev_idx:self.next_idx, 2:]
      s=self.buf_state[self.prev_idx:self.next_idx]
    else:
      r=torch.vstack((self.buf_other[self.prev_idx:, 0:1],
                        self.buf_other[:self.next_idx, 0:1]))
      m=torch.vstack((self.buf_other[self.prev_idx:, 1:2],
                        self.buf_other[:self.next_idx, 1:2]))
      a=torch.vstack((self.buf_other[self.prev_idx:, 2:],
                        self.buf_other[:self.next_idx, 2:]))
      s=torch.vstack((self.buf_state[self.prev_idx:],
                        self.buf_state[:self.next_idx],))
    self.prev_idx=self.next_idx
    return r, m, a, s  # reward, mask, action, state

  def update_now_len(self):
    self.now_len=self.replay_size if self.if_full else self.next_idx

