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
    self.data = np.zeros((maxlen, ) + shape, dtype=dtype)

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
    if idx < 0 or idx >= self.length:
      raise KeyError()
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


class BaseBuffer():
  def __init__(self, limit, item_shape):
    # basic params
    self.limit = limit
    self.buffers = AttrDict()
    # buffers
    for name, shape in item_shape:
      setattr(self.buffers, name, RingBuffer(limit, shape))
    # store trajectory index of each transitions
    self.buffers.tidx = RingBuffer(limit, shape=(), dtype=np.int32)
    self.buffers.tleft = RingBuffer(limit, shape=(), dtype=np.int32)   # record current success goals
    if hasattr(self.buffers, 'bg'):
      # each step if success
      self.buffer.success = RingBuffer(limit, shape=(), dtype=np.float32, data=self._meta.np_success)

    self.trajectories = OrderedDict() # trajectory_id(global) --> trajectory_idxs(buffer)
    self.total_trajectory_len = 0 # transitions in buffer
    self.current_trajectory = 0 # total trajectory number (ignore pop out)

  def add_trajectory(self, *items):
    trajectory_len = len(items[0])

    for name, buffer in self.buffers.items():
      if name == 'tidx':
        idxs = buffer.append_batch(np.ones((trajectory_len,), dtype=np.int32) * self.current_trajectory)
      elif name == 'tleft':
        buffer.append_batch(np.arange(trajectory_len, dtype=np.int32)[::-1])
      elif name == 'reward':
        if self.buffers.get('bg'):
          success = np.any(np.isclose(batcheds, 0.))
          if success:
            self.buffers.success.append_batch(np.ones((trajectory_len,), dtype=np.float32))
          else:
            self.buffers.success.append_batch(np.zeros((trajectory_len,), dtype=np.float32))
        buffer.append_batch(batcheds)
      else:
        buffer.append_batch(batcheds)


    self.trajectories[self.current_trajectory] = idxs
    self.total_trajectory_len += trajectory_len

    # remove trajectories until all remaining trajectories fit in the buffer.
    while self.total_trajectory_len > self.limit:
      # remove left most item
      _, idxs = self.trajectories.popitem(last=False)
      self.total_trajectory_len -= len(idxs)
    self.current_trajectory += 1

  def sample_trajectories(self, n, group_by_buffer=False, from_m_most_recent=None):
    """
    Samples n full trajectories (optionally from 'from_m_most_recent' trajectories)
    """

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

    transition = []
    for buf in self.items:
      item = self.buf.get_batch(batch_idxs)
      transition.append(item)

    return transition

  def sample_slices(self, batch_size, slice_size):
    """Tries to sample slices of length slice_size randomly. Slices must be
    from same trajectory, which may not happen even when oversampled, so it's possible
    (but unlikely, at least for small slice_size) to get a small batch size than requested.
    """
    if self.size == 0:
      return [[] for _ in range(slice_size)]

    b_idxs = np.random.randint(self.size, size=int(batch_size * 1.5))
    first_t = self.buffer_tidx.get_batch(b_idxs - slice_size + 1)
    last_t  = self.buffer_tidx.get_batch(b_idxs)

    batch_idxs = b_idxs[first_t == last_t][:batch_size]

    transitions = []
    for i in range(-slice_size + 1, 1):
      transitions.append([])
      for buf in self.items:
        item = self.buf.get_batch(batch_idxs + i)
        transitions[-1].append(item)

    return transitions

  def sample_n_step_transitions(self, batch_size, n_steps, gamma, batch_idxs=None):
    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)
      
    if self.pool is not None:
      res = self.pool.map(n_step_samples, zip(np.array_split(batch_idxs, self.n_cpu), [n_steps] * self.n_cpu, [gamma] * self.n_cpu))
      res = [np.concatenate(x, 0) for x in zip(*res)]
      return res

    return n_step_samples((batch_idxs, n_steps, gamma))

  def sample_future(self, batch_size, batch_idxs=None):
    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)

    if self.pool is not None:
      res = self.pool.map(future_samples, np.array_split(batch_idxs, self.n_cpu))
      res = [np.concatenate(x, 0) for x in zip(*res)]
      return res

    return future_samples(batch_idxs)

  def sample_from_goal_buffer(self, buffer, batch_size, batch_idxs=None):
    """buffer is one of 'ag', 'dg', 'bg'"""
    if buffer == 'ag':
      sample_fn = achieved_samples
    elif buffer == 'dg':
      sample_fn = actual_samples
    elif buffer == 'bg':
      sample_fn = behavioral_samples
    else:
      raise NotImplementedError

    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)

    if self.pool is not None:
      res = self.pool.map(sample_fn, np.array_split(batch_idxs, self.n_cpu))
      res = [np.concatenate(x, 0) for x in zip(*res)]
      return res

    return sample_fn(batch_idxs)

  def __len__(self): return self.size

  def _get_state(self): 
    d = dict(
      trajectories = self.trajectories,
      total_trajectory_len = self.total_trajectory_len,
      current_trajectory = self.current_trajectory
    )
    for bufname in self.items + ['buffer_tleft', 'buffer_tidx', 'buffer_success']:
      if bufname in self.
        d[bufname] = self.bufname]._get_state()
    return d

  def _set_state(self, d):
    for k in ['trajectories','total_trajectory_len','current_trajectory']:
      self.__dict__[k] = d[k]
    for bufname in self.items + ['buffer_tleft', 'buffer_tidx', 'buffer_success']:
      if bufname in self.
        self.bufname]._set_state(*d[bufname])

  def future_samples(self, idxs):
    # get original transitions
    transition = []
    for buf in BUFF.items:
      item = BUFF[buf].get_batch(idxs)
      transition.append(item)

    # add random future goals
    tlefts = BUFF.buffer_tleft.get_batch(idxs)
    idxs = idxs + np.round(np.random.uniform(size=len(idxs)) * tlefts).astype(np.int32)
    ags = BUFF.buffer_ag.get_batch(idxs)
    transition.append(ags)
    return transition


  def actual_samples(idxs):
    """Assumes there is an 'dg' field, and samples n transitions, pairing each with a random dg
    
    """
    global BUFF

    # get original transitions
    transition = []
    for buf in BUFF.items:
      item = BUFF[buf].get_batch(idxs)
      transition.append(item)

    # add random actual goals
    idxs = np.random.choice(len(BUFF.buffer_tidx), len(idxs))
    dgs = BUFF.buffer_dg.get_batch(idxs)
    transition.append(dgs)

    return transition


  def achieved_samples(idxs):
    """Assumes there is an 'ag' field, and samples n transitions, pairing each with a random ag"""
    global BUFF

    # get original transitions
    transition = []
    for buf in BUFF.items:
      item = BUFF[buf].get_batch(idxs)
      transition.append(item)

    # add random achieved goals
    idxs = np.random.choice(len(BUFF.buffer_tidx), len(idxs))
    ags = BUFF.buffer_ag.get_batch(idxs)
    transition.append(ags)

    return transition


  def behavioral_samples(idxs):
    """Assumes there is an 'bg' field, and samples n transitions, pairing each with a random bg"""
    global BUFF

    # get original transitions
    transition = []
    for buf in BUFF.items:
      item = BUFF[buf].get_batch(idxs)
      transition.append(item)

    # add random achieved goals
    idxs = np.random.choice(len(BUFF.buffer_tidx), len(idxs))
    bgs = BUFF.buffer_bg.get_batch(idxs)
    transition.append(bgs)

    return transition


  def n_step_samples(args):
    """Samples s_t and s_{t+n}, along with the discounted reward in between.
    Assumes buffers include: state, action, reward, next_state, done
    Because some sampled states will not have enough future states, this will
    sometimes return less than num_samples samples.
    """
    state_idxs, n_steps, gamma = args

    global BUFF

    tlefts = BUFF.buffer_tleft.get_batch(state_idxs)

    # prune idxs for which there are not enough future transitions
    good_state_idxs = state_idxs[tlefts >= n_steps - 1]
    potentially_bad_idxs = state_idxs[tlefts < n_steps - 1] # 0 tleft corresp to 1 step
    potentially_bad_delt = tlefts[tlefts < n_steps - 1]
    also_good_state_idxs = potentially_bad_idxs[np.isclose(BUFF.buffer_done.get_batch(potentially_bad_idxs + potentially_bad_delt), 1).reshape(-1)]

    state_idxs = np.concatenate((good_state_idxs, also_good_state_idxs), axis=0)
    t_idxs = BUFF.buffer_tidx.get_batch(state_idxs)

    # start building the transition, of state, action, n_step reward, n_step next_state, done
    transition = [BUFF.buffer_state.get_batch(state_idxs), BUFF.buffer_action.get_batch(state_idxs)]
    rewards = np.zeros_like(state_idxs, dtype=np.float32).reshape(-1, 1)

    for i in range(n_steps):
      query = state_idxs + i
      r_delta = (gamma ** i) * BUFF.buffer_reward.get_batch(query)
      diff_traj = t_idxs != BUFF.buffer_tidx.get_batch(query)
      r_delta[diff_traj] *= 0.
      rewards += r_delta

    transition.append(rewards)
    transition.append(BUFF.buffer_next_state.get_batch(query)) # n_step state

    dones = BUFF.buffer_done.get_batch(query)
    dones += diff_traj.astype(np.float32).reshape(-1, 1)
    transition.append(dones)
    return transition

  @property
  def size(self):
    return len(self.buffer_tidx)

  @property
  def num_trajectories(self):
    return len(self.trajectories)
