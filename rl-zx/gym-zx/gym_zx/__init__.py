import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)
print("Registering zx-v0 environment")
register(
    id='zx-v0',
    entry_point='gym_zx.envs:ZXEnv',
)
