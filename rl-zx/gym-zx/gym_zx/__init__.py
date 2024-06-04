import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='zx-v0',
    entry_point='gym_zx.envs:ZXEnv',
)
