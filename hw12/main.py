
import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm

import gym
import random

seed = 543 # Do not change this
def fix(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

env = gym.make('LunarLander-v2')
fix(env, seed) # fix the environment Do not revise this !!!


#env.reset()

#img = plt.imshow(env.render(mode='rgb_array'))


class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)

'''
softmax 求导：https://zhuanlan.zhihu.com/p/25723112
通过softmax函数一作用，就映射成为(0,1)的值，而这些值的累和为1（满足概率的性质），
那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的）结点，作为我们的预测目标！
'''

'''

The agent will acts according to the output of the policy network above. 

agent 根据 policy network 的输出 决定下一步的动作

agent 的主要功能：

1. learn()：update the policy network from log probabilities and rewards.

2. sample()：
After receiving observation from the environment,
utilize policy network to tell which action to take. 
The return values of this function includes action and log probabilities.

'''
class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def forward(self, state):
        return self.network(state)
    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum() # You don't need to revise this to pass simple baseline (but you can)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        #采样
        action = action_dist.sample()
        #获取该样本的log probability, 对数的底为e
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)

def train():
    agent.network.train()  # Switch network into training mode
    EPISODE_PER_BATCH = 5  # update the  agent every 5 episode
    NUM_BATCH = 500        # totally update the agent for 400 time

    avg_total_rewards, avg_final_rewards = [], []

    #prg_bar = tqdm(range(NUM_BATCH))
    for batch in range(NUM_BATCH):

        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []

        # collect trajectory
        for episode in range(EPISODE_PER_BATCH):

            state = env.reset()
            total_reward, total_step = 0, 0
            seq_rewards = []
            while True:

                action, log_prob = agent.sample(state) # at, log(at|st)
                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
                # seq_rewards.append(reward)
                state = next_state
                total_reward += reward
                total_step += 1
                rewards.append(reward) # change here
                # ! IMPORTANT !
                # Current reward implementation: immediate reward,  given action_list : a1, a2, a3 ......
                #                                                         rewards :     r1, r2 ,r3 ......
                # medium：change "rewards" to accumulative decaying reward, given action_list : a1,                           a2,                           a3, ......
                #                                                           rewards :           r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,  r3+0.99*r4+0.99^2*r5+ ......
                # boss : implement Actor-Critic
                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)

                    break
        #end episode

        #print(f"rewards looks like ", np.shape(rewards))
        #print(f"log_probs looks like ", np.shape(log_probs))

        # record training process
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        #prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

        # update agent
        # rewards = np.concatenate(rewards, axis=0)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
        print("logs prob looks like ", torch.stack(log_probs).size())
        print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

if __name__ == '__main__':
    network = PolicyGradientNetwork()
    agent = PolicyGradientAgent(network)
    env = gym.make('LunarLander-v2')
    fix(env, seed) # fix the environment Do not revise this !!!
    train()