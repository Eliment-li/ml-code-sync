#Training without trainer
import gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PGPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor

import warnings
warnings.filterwarnings('ignore')

train_env_num = 4
buffer_size = 2000

# Create the environments, used for training and evaluation
env = gym.make("CartPole-v0")
test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v0") for _ in range(2)])
train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v0") for _ in range(train_env_num)])

# Create the Policy instance
net = Net(env.observation_space.shape, hidden_sizes=[16,])
actor = Actor(net, env.action_space.shape)
optim = torch.optim.Adam(actor.parameters(), lr=0.001)
policy = PGPolicy(actor, optim, dist_fn=torch.distributions.Categorical)

# Create the replay buffer and the collector
replaybuffer = VectorReplayBuffer(buffer_size, train_env_num)
test_collector = Collector(policy, test_envs)
train_collector = Collector(policy, train_envs, replaybuffer)

train_collector.reset()
train_envs.reset()
test_collector.reset()
test_envs.reset()
replaybuffer.reset()

from tianshou.trainer import onpolicy_trainer
result = onpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=10,
    step_per_epoch=1,
    repeat_per_collect=1,
    episode_per_test=10,
    step_per_collect=2000,
    batch_size=512,
)
print(result)