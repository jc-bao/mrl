import time

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *

import time
import os
import gym
import numpy as np
import torch
import panda_gym


def main(args):
    # parse args
    if args.num_envs is None:
        import multiprocessing as mp
        args.num_envs = max(mp.cpu_count() - 1, 1)
    args.num_eval_envs = args.num_envs
    merge_args_into_config(args, config)
    if config.gamma < 1.:
        config.clip_target_range = (np.round(-(1 / (1 - config.gamma)), 2), 0.)
    if config.gamma == 1:
        config.clip_target_range = (np.round(-args.env_max_step - 5, 2), 0.)
    config.agent_name = make_agent_name(
        config, ['env', 'her', 'seed', 'tb'], prefix=args.prefix)
    # setup modules
    config.update(
        dict(trainer=StandardTrain(),
             evaluation=EpisodicEval(),
             policy=ActorPolicy(),
             logger=Logger(),
             state_normalizer=Normalizer(MeanStdNormalizer()),
             replay=OnlineHERBuffer(),
             action_noise=ContinuousActionNoise(
                 GaussianProcess, std=ConstantSchedule(args.action_noise)),
             algorithm=DDPG()))
    torch.set_num_threads(min(4, args.num_envs))
    torch.set_num_interop_threads(min(4, args.num_envs))
    assert gym.envs.registry.env_specs.get(args.env) is not None
    # make env
    def env(): return gym.make(args.env, render=False)
    config.module_train_env = EnvModule(
        env, num_envs=config.num_envs, seed=config.seed)
    config.module_eval_env = EnvModule(
        env, num_envs=config.num_eval_envs, name='eval_env', seed=config.seed + 1138)
    # actor-critic
    e = config.module_eval_env
    config.actor = PytorchModel(
        'actor', lambda: Actor(\
            AttnBody(e.robot_obs_size, e.obj_obs_size, e.goal_size, args.hidden_size, args.n_attention_blocks, args.n_heads), \
                e.action_dim, e.max_action))
    config.critic = PytorchModel(
        'critic', lambda: Critic(\
            AttnBody(e.robot_obs_size+e.action_dim, e.obj_obs_size, e.goal_size, args.hidden_size, args.n_attention_blocks, args.n_heads), \
                1))
    # fix never done
    if e.goal_env:
        # NOTE: This is important in the standard Goal environments, which are never done
        config.never_done = True
    # setup agent (manager)
    agent = mrl.config_to_agent(config)

    # initial test (redundant)
    num_eps = max(args.num_eval_envs * 3, 10)
    res = np.mean(agent.eval(num_episodes=num_eps).rewards)
    agent.logger.log_color(
        f'Initial test reward ({num_eps} eps):', f'{res:.2f}')

    # main loop
    for epoch in range(int(args.max_steps // args.epoch_len)):
        # train
        t = time.time()
        agent.train(num_steps=args.epoch_len)
        # eval
        res = np.mean(agent.eval(num_episodes=num_eps).rewards)
        # log
        tune.report(iterations=num_eps, reward=res)
        agent.logger.log_color(f'Test reward ({num_eps} eps):', f'{res:.2f}')
        agent.logger.log_color('Epoch time:', '{:.2f}'.format(
            time.time() - t), color='yellow')
        print(f"Saving agent at epoch {epoch}")
        agent.save('checkpoint')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train HER",
                                     formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
    parser.add_argument('--parent_folder', default='./results',
                        type=str, help='where to save progress')
    parser.add_argument('--env', default="PandaRearrangeBimanual-v0",
                        type=str, help="gym environment")
    parser.add_argument('--max_steps', default=int(1e10),
                        type=int, help="maximum number of training steps")
    parser.add_argument('--epoch_len', default=5000,
                        type=int, help='number of steps between evals')
    parser.add_argument('--num_envs', default=16, type=int,
                        help='number of envs (defaults to procs - 1)')
    parser.add_argument('--env_max_step', default=50,
                        type=int, help='max_steps_env_environment')
    config = handover_default_config()
    config.alg = 'ddpg'
    parser = add_config_args(parser, config)
    config = parser.parse_args()

    config['actor_lr'] = tune.sample_from(lambda spec: 10**(np.random.uniform(-4, -1)))
    config['critic_lr'] = config['actor_lr']
    config['optimize_every'] = tune.sample_from(lambda spec: 10**(np.random.randint(1, 101)))
    config['batch_size'] = tune.sample_from(lambda spec: 10**(np.random.randint(2, 4)))
    config['wandb']={
                "project": "handover_subvec",
                "group": "1obj_attn2",
                "api_key": "7566004da241a428aa723461625d1e6c5004b476",
                "log_config": True
            }
    config['prefix'] = f'lr{config.actor_lr}op{config.optimize_every}ba{config.batch_size}'
    config['tb'] = config['prefix']

    analysis=tune.run(
        main,
        metric="reward",
        mode="max",
        name="hebo_exp",
        scheduler=AsyncHyperBandScheduler(),
        search_alg=HEBOSearch(random_state_seed=123, max_concurrent=8),
        num_samples=16,
        config=config, 
        loggers=DEFAULT_LOGGERS + (WandbLogger, ))
