import time

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch

"""
To benchmark implementation on multi-goal Fetch tasks.
"""

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
        'actor', lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm), e.action_dim, e.max_action))
    config.critic = PytorchModel(
        'critic', lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm), 1))
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
        agent.logger.log_color(f'Test reward ({num_eps} eps):', f'{res:.2f}')
        agent.logger.log_color('Epoch time:', '{:.2f}'.format(
            time.time() - t), color='yellow')
        print(f"Saving agent at epoch {epoch}")
        agent.save('checkpoint')


def evaluation_fn(step, width, height):
    time.sleep(0.1)
    return (0.1 + width * step / 100)**(-1) + height * 0.1


def easy_objective(config):
    # Hyperparameters
    width, height = config["width"], config["height"]

    for step in range(config["steps"]):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Feed the score back back to Tune.
        tune.report(iterations=step, mean_loss=intermediate_score)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train HER",
                                     formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument('--parent_folder', default='./results',
                        type=str, help='where to save progress')
    parser.add_argument('--prefix', type=str, default='her',
                        help='Prefix for agent name (subfolder where it is saved)')
    parser.add_argument('--env', default="FetchReach-v1",
                        type=str, help="gym environment")
    parser.add_argument('--max_steps', default=1000000,
                        type=int, help="maximum number of training steps")
    parser.add_argument('--layers', nargs='+', default=(512, 512, 512),
                        type=int, help='hidden layers for actor/critic')
    parser.add_argument('--tb', default='', type=str,
                        help='a tag for the agent name / tensorboard')
    parser.add_argument('--epoch_len', default=5000,
                        type=int, help='number of steps between evals')
    parser.add_argument('--num_envs', default=None, type=int,
                        help='number of envs (defaults to procs - 1)')
    parser.add_argument('--env_max_step', default=50,
                        type=int, help='max_steps_env_environment')
    config = handover_default_config()
    config.alg = 'ddpg'
    parser = add_config_args(parser, config)
    config = parser.parse_args()

    previously_run_params = [
        {
            "width": 10,
            "height": 0,
            "activation": "relu"  # Activation will be relu
        },
        {
            "width": 15,
            "height": -20,
            "activation": "tanh"  # Activation will be tanh
        }
    ]
    known_rewards = [-189, -1144]

    # maximum number of concurrent trials
    max_concurrent = 8

    algo = HEBOSearch(
        # space = space, # If you want to set the space
        points_to_evaluate=previously_run_params,
        evaluated_rewards=known_rewards,
        random_state_seed=123,  # for reproducibility
        max_concurrent=max_concurrent,
    )

    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(
        easy_objective,
        metric="mean_loss",
        mode="min",
        name="hebo_exp_with_warmstart",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=10 if args.smoke_test else 50,
        config={
            "steps": 100,
            "width": tune.uniform(0, 20),
            "height": tune.uniform(-100, 100),
            "activation": tune.choice(["relu", "tanh"])
        })
    print("Best hyperparameters found were: ", analysis.best_config)
