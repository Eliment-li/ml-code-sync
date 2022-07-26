import gym
import numpy as np
import torch

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import PGPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor

import warnings
warnings.filterwarnings('ignore')

env = gym.make("CartPole-v0")
test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v0") for _ in range(2)])

# model
net = Net(env.observation_space.shape, hidden_sizes=[16,])
actor = Actor(net, env.action_space.shape)
optim = torch.optim.Adam(actor.parameters(), lr=0.0003)

policy = PGPolicy(actor, optim, dist_fn=torch.distributions.Categorical)
test_collector = Collector(policy, test_envs)

collect_result = test_collector.collect(n_episode=9)
print(collect_result)
print("Rewards of 9 episodes are {}".format(collect_result["rews"]))
print("Average episode reward is {}.".format(collect_result["rew"]))
print("Average episode length is {}.".format(collect_result["len"]))