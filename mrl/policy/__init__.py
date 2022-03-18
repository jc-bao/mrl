"""Policy package."""
# isort:skip_file

from mrl.policy.base import BasePolicy
from mrl.policy.modelfree.ddpg import DDPGPolicy

__all__ = [
    'BasePolicy', 
    'DDPGPolicy'
]