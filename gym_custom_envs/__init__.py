from gym.envs.registration import register

register(
    id='EncodedCarRacing-v0',
    entry_point='gym_custom_envs.envs:EncodedCarRacing',
)
