import numpy as np
from copy import deepcopy
from attrdict import AttrDict


class Collector:
    def __init__(self, policy, env, buffer, config) -> None:
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.config = config
        self.reset_idxs = []
        self.env_steps = 0

    def collect(self, num_steps):
        assert num_steps % self.config.num_envs == 0, 'Make sure per collect number matches env num'
        state = self.env.state
        for j in range(num_steps // self.config.num_envs):
            action = self.policy(state)
            next_state, reward, done, info = self.env.step(action)
            self.env_steps += self.config.num_envs
            if self.reset_idxs:
                # TODO fix env wrapper to replace env module
                self.env.reset(self.reset_idxs)
                for i in self.reset_idxs:
                    done[i] = True
                    if not 'done_observation' in info[i]:
                        if isinstance(next_state, np.ndarray):
                            # CHECK what is done obs
                            info[i].done_observation = next_state[i]
                        else:
                            for key in next_state:
                                info[i].done_observation = {
                                    k: next_state[k][i] for k in next_state}
                next_state = self.env.state
                self.reset_idxs = []

            state, experience = debug_vectorized_experience(
                state, action, next_state, reward, done, info)
            self.buffer.add(experience)  # save experience and score goals


# CHECK what is this used for
def debug_vectorized_experience(state, action, next_state, reward, done, info):
    experience = AttrDict(
        state=state,
        action=action,
        reward=reward,
        info=info
    )
    next_copy = deepcopy(next_state)  # deepcopy handles dict states

    for idx in np.argwhere(done):
        i = idx[0]
        if isinstance(next_copy, np.ndarray):
            next_copy[i] = info[i].done_observation
        else:
            assert isinstance(next_copy, dict)
            for key in next_copy:
                next_copy[key][i] = info[i].done_observation[key]

    experience.next_state = next_copy
    experience.trajectory_over = done
    experience.done = np.array(
        [info[i].terminal_state for i in range(len(done))], dtype=np.float32)
    experience.reset_state = next_state

    return next_state, experience
