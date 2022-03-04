# Benchmark Results

## Code TODO

- [ ] Drop out
- [ ] Attention/GAT Model
- [ ] D4PG
- [ ] Auto Curriculum
- [ ] Wandb
- [ ] Optimize every step? Try to collect more, update with more.

## Experinment TODO

- [x] Update Interval
  - k updates/ k steps
  
    <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1gzxj87qydpj21bq0niaf9.jpg" alt="image-20220304081310066" style="zoom: 25%;" />
  
    <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1gzxj8sckf9j21co0k4n0f.jpg" alt="image-20220304081344822" style="zoom:25%;" />
  
  - Conclusion:
  
    - Freqent update performs better and not hurt running time.
  
- [ ] Update Barch Size
  - [ ] 1000 batch 0.5 Hz
  - [ ] 100000 batch 0.05Hz
  - [ ] 1000000 batch 0.005Hz
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
  config.action_l2_regularization = 1e-2
  config.target_network_update_freq = 10
  config.target_network_update_frac = 0.05
  config.optimize_every = 2 # num_updates/step
  config.batch_size = 1000
  config.warm_up = 5000 # steps wait to update
  config.initial_explore = 10000 # random explore steps
  config.action_noise = 0.1
  config.eexplore = 0.2 # total random action rate (epsilon-greedy)
  config.go_eexplore = 0.1 # for curiosity
  config.go_reset_percent = 0. # for curiosity
  config.grad_value_clipping = -1
  config.her = 'futureactual_2_2' # future-actual-achieved-behavior
  config.replay_size = int(2.5e6)
  config.activ = 'relu'
```