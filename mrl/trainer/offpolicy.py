import numpy as np
from tqdm import tqdm
import time


def offpolicy_trainer(policy, collector, config):
    config.num_epoch = int(config.max_steps // config.epoch_len)
    collector.collect(num_steps=config.warm_up)  # warm up
    for epoch in range(config.num_epoch):  # TODO add tqdm
        config.num_cycles = int(config.epoch_len//config.num_envs)
        for cycle in range(config.num_cycles):
            t = time.time()
            # collect
            collector.collect(num_steps=config.num_envs)
            # train TODO optimize every to update per step
            update_times = collector.env_steps//config.optimize_every - policy.optimize_times
            for _ in range(update_times):
                policy.update(config.batch_size, collector.buffer)


# test block
if __name__ == '__main__':
    from attrdict import AttrDict
    config = AttrDict(
        max_steps=100,
        epoch_len=10,
        num_envs=2,
        seed=123,
        her='future_4',
        optimize_every=2,
        grad_norm_clipping=-1,
        grad_value_clipping=-1,
        policy_opt_noise=0,
        target_network_update_frac=0.9,
        # rl
        gamma=0.9,
        # buffer
        replay_size=100,
        # update
        batch_size=2,
        actor_lr=1e-3,
        critic_lr=1e-3,
        target_network_update_freq = 10, 
        # network
        layers=[512, 512, 512],
        actor_weight_decay=0,
        critic_weight_decay=0,
        device='cuda',
        clip_target_range = [-50, 0], 
        action_l2_regularization=1e-2, 
        # explore
        warm_up=5000,
        future_warm_up=0,
        varied_action_noise=False,
        action_noise=0.2,
        # reward
        sparse_reward_shaping=False,
        slot_based_state=False,  # CHECK
        # env
        max_action=1,
        action_dim=8,
        state_dim=26,
        goal_dim=3,
        never_done=True,
    )
    from mrl.policy import DDPGPolicy
    from mrl.data import Collector, Buffer
    from mrl.modules.env import EnvModule
    from mrl.utils.networks import FCBody, Actor, Critic
    from torch import nn
    import torch

    def make_env():
        import gym
        import panda_gym
        return gym.make('PandaRearrangeBimanual-v0')
    env = EnvModule(make_env, num_envs=config.num_envs, seed=config.seed)
    config.compute_reward = env.compute_reward
    actor = Actor(FCBody(config.state_dim + config.goal_dim, config.layers,
                  nn.LayerNorm), config.action_dim, config.max_action).to(config.device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.actor_lr,
                                 weight_decay=config.actor_weight_decay)
    critic = Critic(FCBody(config.state_dim + config.goal_dim +
                    config.action_dim, config.layers, nn.LayerNorm), 1).to(config.device)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr,
                                  weight_decay=config.critic_weight_decay)
    buffer = Buffer(config, env)
    policy = DDPGPolicy(config, actor, actor_opt, critic, critic_opt, buffer)
    collector = Collector(policy, env, buffer, config)
    offpolicy_trainer(policy, collector, config)
