import os

import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
#from tqdm.notebook import tqdm
from collections import namedtuple
seed = 543 # Do not change this
def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.use_deterministic_algorithms
  #torch.set_deterministic(True)

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

import gym
import random

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)

        # actor's layer
        self.action_head = nn.Linear(16, 4)

        # critic's layer
        self.value_head = nn.Linear(16, 1)

    def forward(self, state):
        hid = torch.relu(self.fc1(state))
        hid = torch.relu(self.fc2(hid))

        action_prob = F.softmax(self.action_head(hid), dim=-1)
        state_values = self.value_head(hid)

        return action_prob,state_values


from torch.optim.lr_scheduler import StepLR

'''
learn()：update the policy network from log probabilities and rewards.

sample()：After receiving observation from the environment, utilize policy 
network to tell which action to take. The return values of this function includes action and log probabilities.
'''
class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

        #保存 actions
        self.saved_actions = []

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        for (log_prob, value), R in zip(self.saved_actions, rewards):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        #loss = (-log_probs * rewards).sum()  # You don't need to revise this to pass simple baseline (but you can)
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.saved_actions[:]

    def sample(self, state):

        action_prob,state_value = self.network(torch.FloatTensor(state))

        action_dist = Categorical(action_prob)
        action = action_dist.sample()

        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob,state_value

network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)

'''
change "rewards" to accumulative decaying reward
'''
def calcReward(rewards):
    len = rewards.__len__()

    for i in range(len):
        ri = rewards[i]
        j = i + 1
        k = 0.9
        while (j < len):
            ri += rewards[j]*k
            k  =  k*k
            j  += 1

        rewards[i] = ri

def train():
    agent.network.train()  # Switch network into training mode
    EPISODE_PER_BATCH = 5  # update the  agent every 5 episode
    NUM_BATCH = 1000  # totally update the agent for 400 time

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
                action, log_prob,state_value = agent.sample(state)  # at, log(at|st)
                #save action
                agent.saved_actions.append(SavedAction(log_prob, state_value))

                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)  # [log(a1|s1), log(a2|s2), ...., log(at|st)]
                # seq_rewards.append(reward)
                state = next_state
                total_reward += reward
                total_step += 1
                rewards.append(reward)  # change here
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

        #print(f"rewards looks like ", np.shape(rewards))
        #print(f"log_probs looks like ", np.shape(log_probs))
        # record training process
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        #prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

        # update agent

        #accumulative decaying reward
        calcReward(rewards)
        # rewards = np.concatenate(rewards, axis=0)
        #rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + np.finfo(np.float32).eps.item())  # normalize the reward
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
        #print("logs prob looks like ", torch.stack(log_probs).size())
        #print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())
        if batch%10 ==0:
            plt.plot(avg_total_rewards)
            plt.title("Total Rewards")
            plt.show()

    os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
    torch.save(agent.network.state_dict(), 'models/agentNet_ac.pth')

def test():
    fix(env, seed)
    agent.network.eval()  # set the network into evaluation mode
    NUM_OF_TEST = 10  # Do not revise this !!!
    test_total_reward = []
    action_list = []
    for i in range(NUM_OF_TEST):
        actions = []
        state = env.reset()

        img = plt.imshow(env.render(mode='rgb_array'))

        total_reward = 0

        done = False
        while not done:
            action, log_prob,state_value = agent.sample(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)

            total_reward += reward

            img.set_data(env.render(mode='rgb_array'))
            #display.display(plt.gcf())
            #display.clear_output(wait=True)

        print(total_reward)
        test_total_reward.append(total_reward)

        action_list.append(actions)  # save the result of testing
    #torch.save(agent.network.state_dict,'models/agentNet.pth')

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    fix(env, seed)  # fix the environment Do not revise this !!!

    #print(env.observation_space)
    #print(env.action_space)
    #ckpt = torch.load('models/agentNet_ac.pth', map_location='cpu')  # Load your best model
    #agent.network.load_state_dict(ckpt)
    train()
    test()




