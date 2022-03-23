from numpy import longlong
import yaml
from attrdict import AttrDict
import torch
import logging
import pathlib

def get_config(name):
  folder = pathlib.Path(__file__).parent.resolve()
  with open(folder.joinpath(f'{name}.yaml'), "r") as stream:
    try:
      config = AttrDict(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
      print(exc)
  if not torch.cuda.is_available():
    config.device = 'cpu'
    logging.warn('[Device] set to cpu')
  return config