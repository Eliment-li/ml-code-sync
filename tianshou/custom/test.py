import gym
import numpy as np
import  heartpole
class CustomEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}

from stable_baselines.common.env_checker import check_env

gym.register('HeartPole-v0', entry_point='heartpole:HeartPole')
heartpole = gym.make('HeartPole-v0')

check_env(heartpole)