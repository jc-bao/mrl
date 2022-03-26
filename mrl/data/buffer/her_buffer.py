# TODO fix shared memory buffer
from xml.dom.minidom import Attr
from mrl.data.buffer.base import BaseBuffer, RingBuffer, BufferManager
import numpy as np
import torch
from attrdict import AttrDict

from mrl.replays.core.shared_buffer import achieved_samples


class HERBuffer():

  def __init__(
      self,
      config: AttrDict,
  ):
    """
    Buffer that does online hindsight relabeling.
    Replaces the old combo of ReplayBuffer + HERBuffer.
    """
    self.config = config
    self.size = self.config.replay_size
    self.max_env_steps = config.max_env_steps
    self.future_warm_up = self.config.future_warm_up
    # TODO manage sub buffers directly from here
    self.basic_item_shapes = AttrDict(
      state=(config.state_dim, np.float32),
      action=(config.action_dim, np.float32),
      next_state=(config.state_dim, np.float32),
      reward=(1, np.float32),
      done=(1, np.bool8),
      previous_ag=(config.goal_dim, np.float32),
      ag=(config.goal_dim, np.float32),
      bg=(config.goal_dim, np.float32),
      dg=(config.goal_dim, np.float32),
    )
    self.extra_item_shapes = AttrDict(
      tidx=(1, np.int32),
      tleft=(1, np.int32),
      success=(1, np.bool8)
    )

    self.buffers = BufferManager(self.size, self.basic_item_shapes)
    self.extra_info = BufferManager(self.size, self.extra_item_shapes)

    self.num_envs = self.config.num_envs
    # TODO make reset smart
    self._subbuffers = [BufferManager(
      limit=self.max_env_steps, items_shape=self.basic_item_shapes) for _ in range(self.num_envs)]

    # HER mode can differ if demo or normal replay buffer TODO change to params, try random goals
    self.fut, self.act, self.ach, self.beh = parse_hindsight_mode(
      self.config.her)

    self.current_trajectory = 0

  def add(self, exp):
    done = np.expand_dims(exp.done, 1)  # format for replay buffer
    reward = np.expand_dims(exp.reward, 1)  # format for replay buffer
    action = exp.action
    # TODO change env protocal to make store more easy
    transitions = []
    for i in range(self.num_envs):
      transitions.append(
        AttrDict(
          state=exp.state['observation'][i],
          action=action[i],
          reward=reward[i],
          done=done[i],
          next_state=exp.next_state['observation'][i],
          previous_ag=exp.state['achieved_goal'][i],
          ag=exp.next_state['achieved_goal'][i],
          dg=exp.state['desired_goal'][i],
          bg=exp.next_state['desired_goal'][i],
        ))
    if hasattr(self, 'ag_curiosity') and self.ag_curiosity.current_goals is not None:
      raise NotImplementedError
      # behavioral = self.ag_curiosity.current_goals
      # # recompute online reward
      # reward = self.env.compute_reward(
      #   ag, behavioral, {'s': state, 'ns': next_state}).reshape(-1, 1)
    else:  # TODO add curiosity back (move it to env)
      for i in range(self.num_envs):
        self._subbuffers[i].add(transitions[i])
    for i in range(self.num_envs):
      if exp.trajectory_over[i]:
        items = self._subbuffers[i].get_all()
        trajectory_len = len(items.reward)
        self.buffers.append_batch(items)
        extra_info = AttrDict(
          tidx=np.ones((trajectory_len, 1), dtype=np.int32) *
          self.current_trajectory,
          tleft=np.arange(trajectory_len, dtype=np.int32)[::-1, np.newaxis],
          # TODO to judge success smartly
          success=np.ones((trajectory_len, 1), dtype=np.bool8) * \
          np.any(np.isclose(items.reward, 0.))
        )
        self.extra_info.append_batch(extra_info)
        self.current_trajectory += 1
        self._subbuffers[i] = BufferManager(
          limit=self.max_env_steps, items_shape=self.basic_item_shapes)

  def sample(self, batch_size, to_torch=True):
    if hasattr(self, 'prioritized_replay'):
      batch_idxs = self.prioritized_replay(batch_size)
    else:
      batch_idxs = (np.random.randint(
        len(self.buffers), size=batch_size))

    trans = self.buffers[batch_idxs]
    trans = self.change_replay_goals(trans, batch_idxs)
    # Recompute reward online
    if hasattr(self, 'goal_reward'):
      trans.reward = self.goal_reward(
        trans.ag, trans.replay_goal, {'s': trans.state, 'ns': trans.next_state}).reshape(-1, 1).astype(np.float32)
    else:
      trans.reward = self.config.compute_reward(trans.ag, trans.replay_goal, {'s': trans.state, 'ns': trans.next_state}).reshape(-1, 1).astype(np.float32)

    trans.state = np.concatenate((trans.state, trans.replay_goal), -1)
    trans.next_state = np.concatenate((trans.next_state, trans.replay_goal), -1)
    gammas = self.config.gamma * (1.-trans.done)

    # TODO move normalizer to buffer
    if hasattr(self, 'state_normalizer'):
      trans.state = self.state_normalizer(
        trans.state, update=False).astype(np.float32)
      trans.next_states = self.state_normalizer(
        trans.next_states, update=False).astype(np.float32)

    # TODO make the buffer torch based
    if to_torch:
      return (self.torch(trans.state), self.torch(trans.action),
              self.torch(trans.reward), self.torch(trans.next_state),
              self.torch(gammas))
    else:
      return (trans.state, trans.action, trans.reward, trans.next_state, gammas)

  def change_replay_goals(self, trans: AttrDict , batch_idxs: np.ndarray):
    # get goal size
    if len(self.buffers) > self.future_warm_up:
      fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = np.random.multinomial(
        len(batch_idxs), [self.fut, self.act, self.ach, self.beh, 1.])
    else:
      fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = len(batch_idxs), 0, 0, 0, 0

    fut_local_idxs, act_local_idxs, ach_local_idxs , beh_local_idxs  = local_idxs = np.cumsum([fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size])

    # assign actual goals
    trans.replay_goal = trans.bg    
    
    # sample future goals
    if fut_batch_size > 0:
      fut_idxs = batch_idxs[:fut_local_idxs]
      tlefts = self.extra_info[fut_idxs].tleft.flatten()
      future_achieved_idx = fut_idxs + np.round(np.random.uniform(size=len(fut_idxs)) * tlefts).astype(np.int32)
      trans.replay_goal[:fut_local_idxs] = self.buffers.ag[future_achieved_idx]

    # sample random desired(actual) goal goals
    if act_batch_size > 0:
      trans.replay_goal[fut_local_idxs:act_local_idxs] = self.buffers.dg[
        np.random.randint(len(self.buffers), size=act_batch_size)]
    # sample random achieved goals
    if ach_batch_size > 0:
      trans.replay_goal[act_local_idxs:ach_local_idxs] = self.buffers.ag[
        np.random.randint(len(self.buffers), size=ach_batch_size)]
    # sample random behavior goals
    if beh_batch_size > 0:
      trans.replay_goal[ach_local_idxs:beh_local_idxs] = self.buffers.bg[
        np.random.randint(len(self.buffers), size=beh_batch_size)]

    return trans

  def __len__(self):
    return self.buffers.len

  def torch(self, x):
    if isinstance(x, torch.Tensor):
      return x
    return torch.FloatTensor(x).to(self.config.device)

  def numpy(self, x):
    return x.cpu().detach().numpy()


def parse_hindsight_mode(hindsight_mode: str):
  """setup relabel mode.
  fut: future version
  act: 

  Args:
      hindsight_mode (str): _description_

  Returns:
      _type_: _description_
  """
  if 'future_' in hindsight_mode:
    _, fut = hindsight_mode.split('_')
    fut = float(fut) / (1. + float(fut))
    act = 0.
    ach = 0.
    beh = 0.
  elif 'futureactual_' in hindsight_mode:
    _, fut, act = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(act))
    fut = float(fut) * non_hindsight_frac
    act = float(act) * non_hindsight_frac
    ach = 0.
    beh = 0.
  elif 'futureachieved_' in hindsight_mode:
    _, fut, ach = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(ach))
    fut = float(fut) * non_hindsight_frac
    act = 0.
    ach = float(ach) * non_hindsight_frac
    beh = 0.
  elif 'rfaa_' in hindsight_mode:
    _, real, fut, act, ach = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach))
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = 0.
  elif 'rfaab_' in hindsight_mode:
    _, real, fut, act, ach, beh = hindsight_mode.split('_')
    denom = (float(real) + float(fut) +
             float(act) + float(ach) + float(beh))
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = float(beh) / denom
  else:
    fut = 0.
    act = 0.
    ach = 0.
    beh = 0.

  return fut, act, ach, beh
