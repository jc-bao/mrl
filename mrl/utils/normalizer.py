import numpy as np

class MeanStdNormalizer:
	def __init__(self, read_only=False, clip_before=200.0, clip_after=5.0, epsilon=1e-8):

		self.read_only = read_only
		self.rms = None
		self.clip_before = clip_before
		self.clip_after = clip_after
		self.epsilon = epsilon

	def __call__(self, x, update=True):
		x = np.clip(np.asarray(x), -self.clip_before, self.clip_before)
		if self.rms is None:
			self.rms = RunningMeanStd(shape=(1, ) + x.shape[1:])
		if not self.read_only and update:
			self.rms.update(x)
		return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -self.clip_after, self.clip_after)

	def state_dict(self):
		if self.rms is not None:
			return {'mean': self.rms.mean, 'var': self.rms.var, 'count': self.rms.count}

	def load_state_dict(self, saved):
		self.rms.mean = saved['mean']
		self.rms.var = saved['var']
		self.rms.count = saved['count']


class RunningMeanStd(object):
	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	def __init__(self, epsilon=1e-4, shape=()):
		self.mean = np.zeros(shape, 'float64')
		self.var = np.ones(shape, 'float64')
		self.count = epsilon

	def update(self, x):
		batch_mean = np.mean(x, axis=0, keepdims=True)
		batch_var = np.var(x, axis=0, keepdims=True)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		delta = batch_mean - self.mean
		tot_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / tot_count
		m_a = self.var * (self.count)
		m_b = batch_var * (batch_count)
		M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (tot_count)
		new_var = M2 / (tot_count)

		self.mean = new_mean
		self.var = new_var
		self.count = tot_count
