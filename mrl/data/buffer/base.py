from re import L
import numpy as np
from attrdict import AttrDict
from collections import OrderedDict


class RingBuffer(object):
  def __init__(self, maxlen, shape, dtype=np.float32):
    """
    A buffer object, when full restarts at the initial position

    :param maxlen: (int) the max number of numpy objects to store
    :param shape: (tuple) the shape of the numpy objects you want to store
    :param dtype: (str) the name of the type of the numpy object you want to store
    """
    self.maxlen = maxlen
    self.start = 0 
    self.length = 0
    self.shape = shape
    self.data = np.zeros((maxlen, shape), dtype=dtype)

  def _get_state(self):
    end_idx = self.start + self.length
    indices = range(self.start, end_idx)
    return self.start, self.length, self.data.take(indices, axis=0, mode='wrap')

  def _set_state(self, start, length, data):
    self.start = start
    self.length = length
    self.data[:length] = data
    self.data = np.roll(self.data, start, axis=0)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    if len(idx) > 0:
      assert (idx >= 0).all() and (idx < self.length).all(
      ), f'{idx[np.where(0 <= idx < self.length)]}out of bounds access'
      return self.get_batch(idx)
    else:
      assert ((idx >= 0) and (idx < self.length)), f'{idx} out of bounds access'  
      return self.data[(self.start + idx) % self.maxlen]

  def get_batch(self, idxs):
    return self.data[(self.start + idxs) % self.length]

  def append(self, var):
    if self.length < self.maxlen:
      # We have space, simply increase the length.
      self.length += 1
    elif self.length == self.maxlen:
      # No space, "remove" the first item.
      self.start = (self.start + 1) % self.maxlen
    else:
      # This should never happen.
      raise RuntimeError()

    self.data[(self.start + self.length - 1) % self.maxlen] = var

  def _append_batch_with_space(self, var):
    """
    Append a batch of objects to the buffer, *assuming* there is space.

    :param var: (np.ndarray) the batched objects you wish to add
    """
    len_batch = len(var)
    start_pos = (self.start + self.length) % self.maxlen

    self.data[start_pos : start_pos + len_batch] = var
    
    if self.length < self.maxlen:
      self.length += len_batch
      assert self.length <= self.maxlen, "this should never happen!"
    else:
      self.start = (self.start + len_batch) % self.maxlen

    return np.arange(start_pos, start_pos + len_batch)

  def append_batch(self, var):
    """
    Append a batch of objects to the buffer.

    :param var: (np.ndarray) the batched objects you wish to add
    """
    len_batch = len(var)
    assert len_batch < self.maxlen, 'trying to add a batch that is too big!'
    start_pos = (self.start + self.length) % self.maxlen
    
    if start_pos + len_batch <= self.maxlen:
      # If there is space, add it
      idxs = self._append_batch_with_space(var)
    else:
      # No space, so break it into two batches for which there is space
      first_batch, second_batch = np.split(var, [self.maxlen - start_pos])
      idxs1 = self._append_batch_with_space(first_batch)
      # use append on second call in case len_batch > self.maxlen
      idxs2 = self._append_batch_with_space(second_batch)
      idxs = np.concatenate((idxs1, idxs2))
    return idxs

class BufferManager:
  def __init__(self, limit:int, items_shape:AttrDict) -> None:
    # TODO concate to boost index
    self.limit:int = limit
    self.items_shape:AttrDict = items_shape
    for k, (dim, dtype) in items_shape.items():
      setattr(self, k, RingBuffer(limit, dim, dtype=dtype))
    self.len:int = 0

  def add(self, transition: AttrDict):
    assert self.len < self.limit, 'buffer is full'
    for k, v in transition.items():
      getattr(self, k).append(v)
    self.len += 1

  def append_batch(self, transitions: AttrDict):
    for k, v in transitions.items():
      getattr(self, k).append_batch(v)
    self.len += list(transitions.values())[0].shape[0]

  def __getitem__(self, idxs):
    return AttrDict({k: getattr(self, k).get_batch(idxs) for k in self.items_shape.keys()})

  def get_all(self):
    return AttrDict({k: getattr(self, k).data[:self.len] for k in self.items_shape.keys()})

  def __len__(self):
    return getattr(self, list(self.items_shape.keys())[0]).length
  

# TODO move some function to her buffer
class BaseBuffer():
  def __init__(self, limit, item_shape):
    # basic params
    self.limit = limit
    self.items = list(item_shape.keys()) 
    self.extra_items = ['tidx', 'tleft', 'success']
    extra_item_shape = AttrDict(
      tidx = (1, np.int32), 
      tleft = (1, np.int32),
      success = (1, np.bool8)
    )
    # buffers
    self.buffers = BufferManager(limit, item_shape) 
    self.extra_info = BufferManager(limit, extra_item_shape) 

    self.trajectories = OrderedDict() # trajectory_id(global) --> trajectory_idxs(buffer)
    self.total_trajectory_len = 0 # transitions in buffer
    self.current_trajectory = 0 # total trajectory number (ignore pop out)

  def add_trajectory(self, items: AttrDict):
    trajectory_len = len(items.reward)
    self.buffers.append_batch(items)
    extra_info = AttrDict(
      tidx = np.ones((trajectory_len,1), dtype=np.int32) * self.current_trajectory,
      tleft =  np.arange(trajectory_len, dtype=np.int32)[::-1,np.newaxis],
      # TODO to judge success smartly
      success = np.ones((trajectory_len,1), dtype=np.bool8)*np.any(np.isclose(items.reward, 0.))
    )
    self.extra_info.append_batch(extra_info)

    # add trajectory index, total trajectory numbers
    self.trajectories[self.current_trajectory] = extra_info.tidx
    self.total_trajectory_len += trajectory_len

    # remove trajectories until all remaining trajectories fit in the buffer.
    while self.total_trajectory_len > self.limit:
      # TODO remove at once or just overwrite
      # remove left most item
      _, idxs = self.trajectories.popitem(last=False)
      self.total_trajectory_len -= len(idxs)
    self.current_trajectory += 1

  def sample_trajectories(self, n, group_by_buffer=False, from_m_most_recent=None):
    """
    Samples n full trajectories (optionally from 'from_m_most_recent' trajectories)
    """
    # TODO fix this function
    if from_m_most_recent is not None and (len(self.trajectories) - n >= 0):
      min_idx = max(self.current_trajectory - len(self.trajectories), self.current_trajectory - from_m_most_recent)
      idxs = np.random.randint(min_idx, self.current_trajectory, n)
    else:
      idxs = np.random.randint(self.current_trajectory - len(self.trajectories), self.current_trajectory, n)
    queries = [self.trajectories[i] for i in idxs]
    splits = np.cumsum([len(q) for q in queries[:-1]])
    query = np.concatenate(queries)
    transition = []
    for buf in self.items:
      transition.append(np.split(self.buf.get_batch(query), splits))

    if group_by_buffer:
      return transition

    return list(zip(*transition))

  def sample(self, batch_size, batch_idxs=None):
    """
    sample a random batch from the buffer

    :param batch_size: (int) the number of element to sample for the batch
    :return: (list) the sampled batch
    """
    if self.size == 0:
      return []

    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)
    transitions = self.buffers[batch_idxs]
    transitions.replay_goal = transitions.bg
    return transitions 

  def sample_n_step_transitions(self, batch_size, n_steps, gamma, batch_idxs=None):
    # TODO L1 fix this function
    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)
    # TODO merge this sample function  
    return self.n_step_samples((batch_idxs, n_steps, gamma))

  def sample_future(self, batch_size, batch_idxs=None):
    # TODO merge to single sample function
    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)

    return self.future_samples(batch_idxs)

  def sample_from_goal_buffer(self, buffer, batch_size, batch_idxs=None):
    """buffer is one of 'ag', 'dg', 'bg'"""
    if buffer == 'ag':
      sample_fn = self.achieved_samples
    elif buffer == 'dg':
      sample_fn = self.actual_samples
    elif buffer == 'bg':
      sample_fn = self.behavioral_samples
    else:
      raise NotImplementedError

    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)

    return sample_fn(batch_idxs)

  def __len__(self): return self.size

  def _get_state(self): 
    return {
      'trajectories': self.trajectories, 
      'total_trajectory_len': self.total_trajectory_len,
      'current_trajectory': self.current_trajectory,
      'buffers': self.buffers
    } 

  def _set_state(self, d):
    for k, v in d:
      setattr(self, k, v)

  def future_samples(self, idxs):
    # get original transitions
    transition = self.buffers[idxs] 
    # add random future goals
    extra_info = self.extra_info[idxs]
    tlefts = extra_info.tleft.flatten()
    idxs = idxs + np.round(np.random.uniform(size=len(idxs)) * tlefts).astype(np.int32)
    ags = self.buffers.ag.get_batch(idxs)
    transition.replay_goal = ags
    # TODO make it a dict
    return transition


  def actual_samples(self, idxs):
    # add random achieved goals
    transition = self.buffers[idxs]
    idxs = np.random.choice(len(self.extra_info.tidx), len(idxs))
    dgs = self.buffers.dg.get_batch(idxs)
    transition.replay_goal = dgs
    return transition

  def achieved_samples(self, idxs):
    transition = self.buffers[idxs]
    idxs = np.random.choice(len(self.extra_info.tidx), len(idxs))
    ags = self.buffers.ag.get_batch(idxs)
    transition.replay_goal = ags
    return transition

  def behavioral_samples(self, idxs):
    # TODO merge this functions
    transition = self.buffers[idxs]
    idxs = np.random.choice(len(self.extra_info.tidx), len(idxs))
    bgs = self.buffers.ag.get_batch(idxs)
    transition.replay_goal = bgs
    return transition

  def n_step_samples(self, args):
    """Samples s_t and s_{t+n}, along with the discounted reward in between.
    Assumes buffers include: state, action, reward, next_state, done
    Because some sampled states will not have enough future states, this will
    sometimes return less than num_samples samples.
    """
    # TODO fix this function
    state_idxs, n_steps, gamma = args

    tlefts = self.buffers.tleft.get_batch(state_idxs)

    # prune idxs for which there are not enough future transitions
    good_state_idxs = state_idxs[tlefts >= n_steps - 1]
    potentially_bad_idxs = state_idxs[tlefts < n_steps - 1] # 0 tleft corresp to 1 step
    potentially_bad_delt = tlefts[tlefts < n_steps - 1]
    also_good_state_idxs = potentially_bad_idxs[np.isclose(self.buffers.done.get_batch(potentially_bad_idxs + potentially_bad_delt), 1).reshape(-1)]

    state_idxs = np.concatenate((good_state_idxs, also_good_state_idxs), axis=0)
    t_idxs = self.extra_info.tidx.get_batch(state_idxs)

    # start building the transition, of state, action, n_step reward, n_step next_state, done
    transition = [self.buffers.state.get_batch(state_idxs), self.buffers.action.get_batch(state_idxs)]
    rewards = np.zeros_like(state_idxs, dtype=np.float32).reshape(-1, 1)

    for i in range(n_steps):
      query = state_idxs + i
      r_delta = (gamma ** i) * self.buffers.reward.get_batch(query)
      diff_traj = t_idxs != self.extra_info.tidx.get_batch(query)
      r_delta[diff_traj] *= 0.
      rewards += r_delta

    transition.append(rewards)
    transition.append(self.buffers.next_state.get_batch(query)) # n_step state

    dones = self.buffers.done.get_batch(query)
    dones += diff_traj.astype(np.float32).reshape(-1, 1)
    transition.append(dones)
    return transition

  @property
  def size(self):
    # TODO L1 make it elegant
    return len(self.extra_info.tidx)

  @property
  def num_trajectories(self):
    return len(self.trajectories)
