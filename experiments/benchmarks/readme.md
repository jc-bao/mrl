# Benchmark Results

## Code TODO

- [ ] Drop out
- [ ] Attention/GAT Model
- [ ] D4PG
- [ ] Auto Curriculum
- [ ] Wandb
- [ ] Optimize every step? Try to collect more, update with more.

## Experinment TODO

- [ ] Update interval
  - [ ] 1 update / 1 step
  - [ ] 10 updates / 10 steps
  - [ ] 100 updates / 100 steps
  - [ ] 1000 updates / 1000 steps
- [ ] Update times
  - [ ] 2000 batch 1Hz
  - [ ] 20000 batch 0.1Hz
  - [ ] 200000 batch 0.01Hz
  - [ ] 2000000 batch 0.001Hz
## Tricks

1. clip target network

## Robotics Environments

### Fetch

<img src="plots/robotics_fetch_pickplace.png" alt="pickplace" width=400/>

```shell
PYTHONPATH=./ python experiments/benchmarks/train_her.py --env FetchPickAndPlace-v1 --num_envs 8 --parent_folder ./results --her future_4
```

* other configs

```python
# training
  config.gamma = 0.98
  config.actor_lr = 1e-3
  config.critic_lr = 1e-3
  config.actor_weight_decay = 0.
  config.action_l2_regularization = 1e-1
  config.target_network_update_freq = 40
  config.target_network_update_frac = 0.05
  config.optimize_every = 1 # num_updates/step
  config.batch_size = 2000
  config.warm_up = 2500 # steps wait to update
  config.initial_explore = 5000 # random explore steps
  config.replay_size = int(1e6)
  config.action_noise = 0.1
  config.eexplore = 0.1 # total random action rate (epsilon-greedy)
  config.go_eexplore = 0.1 # for curiosity
  config.go_reset_percent = 0. # for curiosity
  config.her = 'rfaab_1_4_3_1_1' # future-actual-achieved-behavior
  config.grad_value_clipping = 5.
```