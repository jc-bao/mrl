from tqdm import tqdm
import logging


def offpolicy_trainer(config, policy, collector):
	config.num_epoch = int(config.max_env_steps // config.epoch_len)
	logging.debug('start warm up')
	collector.collect(num_steps=config.warm_up)  # warm up
	for epoch in range(config.num_epoch):  # TODO add tqdm
		config.num_cycles = int(config.epoch_len//config.num_envs)
		for cycle in tqdm(range(config.num_cycles)):
			# collect
			collector.collect(num_steps=config.num_envs)
			# train TODO optimize every to update per step
			update_times = collector.env_steps//config.optimize_every - policy.optimize_times
			for _ in range(update_times):
				policy.update(config.batch_size, collector.buffer)
		logging.debug(f'start evaluate {config.num_eval_epochs} epochs')
		collector.eval(num_epochs=config.num_eval_epochs)


# test block
if __name__ == '__main__':
	from mrl.configs import get_config
	from mrl.policy import DDPGPolicy
	from mrl.data import Collector, HERBuffer
	from mrl.modules.env import EnvModule
	from mrl.utils.networks import FCBody, Actor, Critic
	from mrl.utils.normalizer import MeanStdNormalizer
	from torch import nn
	import torch

	logging.basicConfig(level=logging.WARN)
	logging.debug('finish import')

	config = get_config('debug')
	def make_env():
		import gym
		import panda_gym
		return gym.make('PandaRearrangeBimanual-v0')
	if config.wandb:
		from mrl.utils.logger import WandbLogger
		logger = WandbLogger(config)
	else:
		logger = None
	env = EnvModule(make_env, num_envs=config.num_envs, seed=config.seed)
	eval_env = EnvModule(make_env, num_envs=config.num_eval_envs, seed=config.seed) 
	logging.debug('env create done.')
	config.action_space = env.action_space
	actor = Actor(FCBody(config.state_dim + config.goal_dim, config.layers,
								nn.LayerNorm), config.action_dim, config.max_action).to(config.device)
	actor_opt = torch.optim.Adam(actor.parameters(), lr=config.actor_lr,
															 weight_decay=config.actor_weight_decay)
	critic = Critic(FCBody(config.state_dim + config.goal_dim +
									config.action_dim, config.layers, nn.LayerNorm), 1).to(config.device)
	critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr,
																weight_decay=config.critic_weight_decay)
	buffer = HERBuffer(replay_size=config.replay_size, future_warm_up=config.future_warm_up,
	env_params=config.env_params, her_params=config.her_params, reward_fn = env.compute_reward,
	device = config.device)
	logging.debug('buffer create done.')
	normalizer = MeanStdNormalizer(read_only=False)
	policy = DDPGPolicy(config, actor, actor_opt, critic,
											critic_opt, buffer, normalizer=normalizer, logger=logger)
	logging.debug('policy create done.')
	collector = Collector(config, policy, env, eval_env, buffer, logger=logger)
	logging.debug('collect create done.')
	offpolicy_trainer(config, policy, collector)