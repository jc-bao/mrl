# Maximum Entropy Gain Exploration for Long Horizon Multi-goal Reinforcement Learning

## TODO

- [x] profile sample times (each curiosity will calculate current goal again)
  * conclusion: kde takes too much times and process_experience with kde is also slow (kde takes moost of time)


## Run times

| MEGA                                                         | HER                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20220304111555655](https://tva1.sinaimg.cn/large/e6c9d24ely1gzxoic226yj21f20u0gos.jpg) | ![image-20220304114721298](https://tva1.sinaimg.cn/large/e6c9d24ely1gzxpf1alv1j21e40u0jus.jpg) |
| 1. density model call _kde every 100 steps collected<br />2. Process_experience also takes much more time (half of update) |                                                              |


