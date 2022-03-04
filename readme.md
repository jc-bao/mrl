# `mrl`: modular RL

## Modules

* policy: 
  * state -> relabel -> normalize
* curiosity
  * key method: `relabel_state`: get current goal
  * current goal maintain: `_process_experience`


## Installation

There is a `requirements.txt` that was works with venv:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then `pip install` the appropriate version of `Pytorch` by following the instructions here: https://pytorch.org/get-started/locally/.

To run `Mujoco` environments you need to have the Mujoco binaries and a license key. Follow the instructions [here](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key).

To test run:

```
pytest tests
PYTHONPATH=./ python experiments/mega/train_mega.py --env FetchReach-v1 --layers 256 256 --max_steps 5000
```

The first command should have 3/3 success.
The second command should solve the environment in <1 minute (better than -5 avg test reward). 

## Usage

To understand how the code works, read `mrl/agent_base.py`. 

See `tests/test_agent_sac.py` and `experiments/benchmarks` for example usage. The basic outline is as follows:

1. Construct a config object that contains all the agent hyperparameters and modules. There are some existing base configs / convenience methods for creating default SAC/TD3/DDPG agents (see, e.g., the benchmarks code). If you use `argparse` you can use a config object automatically populate the parser using `parser = add_config_args(parser, config)`. 
2. Call `mrl.config_to_agent` on the config to get back an agent. 
3. Use the agent however you want; e.g., call its train/eval methods, save/load, module methods, and so on. 

To add functionality or a new algorithm, you generally just need to define a one or more modules that hook into the agent's lifecycle methods and add them to the config. They automatically hook into the agent's lifecycle methods, so the rest of the code can stay the same. 
