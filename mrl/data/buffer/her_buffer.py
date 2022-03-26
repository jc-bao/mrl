# TODO fix shared memory buffer
from typing import Callable
from mrl.data.buffer.base import BufferManager
import numpy as np
import torch
from attrdict import AttrDict


class HERBuffer():

  def __init__(
      self,
      replay_size: int,
      future_warm_up: int,
      env_params: AttrDict,
      her_params: AttrDict,
      reward_fn: Callable,
      device: str, 
  ):
    """
    Buffer that does online hindsight relabeling.
    Replaces the old combo of ReplayBuffer + HERBuffer.
    """
    self.size = replay_size
    self.env_params = env_params
    self.her_params = her_params
    self.reward_fn = reward_fn
    self.future_warm_up = future_warm_up
    self.device = device
    # TODO manage sub buffers directly from here
    self.basic_item_shapes = AttrDict(
      state=(env_params.state_dim, np.float32),
      action=(env_params.action_dim, np.float32),
      next_state=(env_params.state_dim, np.float32),
      reward=(1, np.float32),
      done=(1, np.bool8),
      previous_ag=(env_params.goal_dim, np.float32),
      ag=(env_params.goal_dim, np.float32),
      bg=(env_params.goal_dim, np.float32),
      dg=(env_params.goal_dim, np.float32),
    )
    self.extra_item_shapes = AttrDict(
      tidx=(1, np.int32),
      tleft=(1, np.int32),
      success=(1, np.bool8)
    )

    self.buffers = BufferManager(self.size, self.basic_item_shapes)
    self.extra_info = BufferManager(self.size, self.extra_item_shapes)

    # TODO make reset smart
    self._subbuffers = [BufferManager(
      limit=self.env_params.max_env_steps, items_shape=self.basic_item_shapes) for _ in range(self.env_params.num_envs)]

    self.current_trajectory = 0

  def add(self, exp):
    done = np.expand_dims(exp.done, 1)  # format for replay buffer
    reward = np.expand_dims(exp.reward, 1)  # format for replay buffer
    action = exp.action
    # TODO change env protocal to make store more easy
    transitions = [AttrDict(
      state=exp.state['observation'][i],
      action=action[i],
      reward=reward[i],
      done=done[i],
      next_state=exp.next_state['observation'][i],
      previous_ag=exp.state['achieved_goal'][i],
      ag=exp.next_state['achieved_goal'][i],
      dg=exp.state['desired_goal'][i],
      bg=exp.next_state['desired_goal'][i],
    ) for i in range(self.env_params.num_envs)]
    if hasattr(self, 'ag_curiosity') and self.ag_curiosity.current_goals is not None:
      raise NotImplementedError
    for i in range(self.env_params.num_envs):
      self._subbuffers[i].add(transitions[i])
    for i in range(self.env_params.num_envs):
      if exp.trajectory_over[i]:
        items = self._subbuffers[i].get_all()
        trajectory_len = len(items.reward)
        self.buffers.append_batch(items)
        extra_info = AttrDict(
          tidx=np.ones((trajectory_len, 1), dtype=np.int32) *
          self.current_trajectory,
          tleft=np.arange(trajectory_len, dtype=np.int32)[::-1, np.newaxis],
          # TODO to judge success smartly
          success=np.ones((trajectory_len, 1), dtype=np.bool8) *
          np.any(np.isclose(items.reward, 0.))
        )
        self.extra_info.append_batch(extra_info)
        self.current_trajectory += 1
        self._subbuffers[i] = BufferManager(
          limit=self.env_params.max_env_steps, items_shape=self.basic_item_shapes)

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
      trans.reward = self.reward_fn(trans.ag, trans.replay_goal, {
        's': trans.state, 'ns': trans.next_state}).reshape(-1, 1).astype(np.float32)

    trans.state = np.concatenate((trans.state, trans.replay_goal), -1)
    trans.next_state = np.concatenate(
      (trans.next_state, trans.replay_goal), -1)
    masks = (1.-trans.done)

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
              self.torch(masks))
    else:
      return (trans.state, trans.action, trans.reward, trans.next_state, masks)

  def change_replay_goals(self, trans: AttrDict, batch_idxs: np.ndarray):
    # get goal size
    if len(self.buffers) > self.future_warm_up:
      fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = np.random.multinomial(
        len(batch_idxs), [self.her_params.fut, self.her_params.act, self.her_params.ach, self.her_params.beh, 1.])
    else:
      fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = len(
        batch_idxs), 0, 0, 0, 0
    fut_local_idxs, act_local_idxs, ach_local_idxs, beh_local_idxs = np.cumsum(
      [fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size])
    assert real_batch_size == len(batch_idxs) - beh_local_idxs
    # assign actual goals
    trans.replay_goal = trans.bg

    # sample future goals
    if fut_batch_size > 0:
      fut_idxs = batch_idxs[:fut_local_idxs]
      tlefts = self.extra_info.tleft[fut_idxs].flatten()
      future_achieved_idx = fut_idxs + \
        np.round(np.random.uniform(size=len(fut_idxs))
                 * tlefts).astype(np.int32)
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
    return torch.FloatTensor(x).to(self.device)

  def numpy(self, x):
    return x.cpu().detach().numpy()


if __name__ == '__main__':
  env_params = AttrDict(
    num_envs=1,
    state_dim=2,
    action_dim=2,
    goal_dim=2,
    max_env_steps=10,
  )
  her_params = AttrDict(
    fut=0.8,
    act=0,
    ach=0,
    beh=0,
  )
  def reward_fn(ag, dg, info):
    return np.sum(np.square(ag - dg), -1)
  buffer = HERBuffer(replay_size=10, future_warm_up=4,
                     env_params=env_params, her_params=her_params, reward_fn=reward_fn, device='cpu')
  for i in range(15):
    exp=AttrDict(
      done = [(i+1)%5==0], 
      trajectory_over = [(i+1)%5==0], 
      state = {
        'observation': np.ones((1,2))*i,
        'achieved_goal': np.ones((1,2))*i,
        'desired_goal': np.ones((1,2))*i,
      }, 
      reward = np.ones((1))*i,
      action = np.ones((1,2))*i,
      next_state = {
        'observation': np.ones((1,2))*i + 1,
        'achieved_goal': np.ones((1,2))*i + 1,
        'desired_goal': np.ones((1,2))*i + 1,
      }, 
    )
    buffer.add(exp)