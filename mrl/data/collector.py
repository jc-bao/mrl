import numpy as np
from copy import deepcopy
from attrdict import AttrDict
import wandb, torch

class Collector:
    def __init__(self, policy, env, buffer, config) -> None:
        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.config = config
        self.reset_idxs = []
        self.env_steps = 0
        # log info
        self.rewards_per_env = np.zeros((self.config.num_envs, ))
        self.steps_per_env = np.zeros((self.config.num_envs, ))
        self.episode_rewards = []
        self.episode_steps = []

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
            rewards, dones = experience.reward, experience.trajectory_over
            # log info
            self.rewards_per_env += rewards
            self.steps_per_env += 1
            if np.any(dones):
                self.episode_rewards += list(self.rewards_per_env[dones])
                self.episode_steps += list(self.steps_per_env[dones])
                self.rewards_per_env[dones] = 0
                self.steps_per_env[dones] = 0
        if self.config.wandb:
            recent_log_time = (self.env_steps // self.config.epoch_len) * self.config.epoch_len
            if recent_log_time <= self.env_steps and self.env_steps < recent_log_time + self.config.num_envs:
                wandb.log({
                    'Train/Episode_rewards': np.mean(self.episode_rewards), 
                    'Train/Episode_steps': np.mean(self.episode_steps)
                }, step = self.env_steps)
                self.rewards_per_env = np.zeros((self.config.num_envs, ))
                self.steps_per_env = np.zeros((self.config.num_envs, ))
                self.episode_rewards = []
                self.episode_steps = []

    def eval(self, num_epochs):
        episode_rewards, episode_steps = [], []
        discounted_episode_rewards = []
        is_successes = []

        for _ in range(num_epochs//self.config.num_envs):
            state = self.env.reset()
            dones = np.zeros((self.config.num_envs,))
            steps = np.zeros_like(dones)
            is_success = np.zeros_like(dones)
            ep_rewards = [[] for _ in range(self.config.num_envs)]

            while not np.all(dones):
                with torch.no_grad():
                    action = self.policy(state, training=False)
                state, reward, dones_, infos = self.env.step(action)
                for i, (rew, done, info) in enumerate(zip(reward, dones_, infos)):
                    if dones[i]:
                        continue
                    ep_rewards[i].append(rew)
                    steps[i] += 1
                    if done:
                        dones[i] = 1.
                        is_success[i] = info['is_success']

            for ep_reward, step, is_succ in zip(ep_rewards, steps, is_success):
                is_successes.append(is_succ)
                episode_rewards.append(sum(ep_reward))
                discounted_episode_rewards.append(
                    discounted_sum(ep_reward, self.config.gamma))
                episode_steps.append(step)
        if self.config.wandb:
            wandb.log({
                'Test/Success':np.mean(is_successes), 
                'Test/Episode_rewards': np.mean(episode_rewards), 
                'Test/Discounted_episode_rewards': np.mean(discounted_episode_rewards), 
                'Test/Episode_steps': np.mean(episode_steps)
            }, step = self.env_steps)
        print(f'Evaluate: episode_rewards:{np.mean(episode_rewards)}, Success:{np.mean(is_successes)}')
        return AttrDict(
            rewards=np.mean(episode_rewards),
            steps=np.mean(episode_steps)
        )


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


def discounted_sum(lst, discount):
    sum = 0
    gamma = 1
    for i in lst:
        sum += gamma*i
        gamma *= discount
    return sum
