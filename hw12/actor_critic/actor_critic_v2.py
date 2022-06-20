import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#env = gym.make("CartPole-v0").unwrapped
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.001


seed = 543 # Do not change this
def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  random.seed(seed)
  torch.use_deterministic_algorithms

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

# Plot duration curve:
# From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

def plot_durations(total_rewards):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(total_rewards)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('total_reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

total_rewards = []
def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        total_reward = 0
        for i in count():
            #env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

            state = next_state
            total_reward += reward
            if done:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                total_rewards.append(total_reward)
                if i %100 == 0:
                    plot_durations(total_rewards)
                break

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        #critic_loss=F.smooth_l1_loss(values, torch.tensor([returns]))

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
        if reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(reward,i))
            break

    os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
    timestr = time.strftime("%m_%d_%H_%M", time.localtime())
    torch.save(actor.state_dict(), 'models/'+timestr+'_actor.pth')
    torch.save(critic.state_dict(), 'models/'+timestr+'_critic.pth')
    # env.close()

actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
def test():
    # set the network into evaluation mode
    actor.eval()
    critic.eval()

    NUM_OF_TEST = 10  # Do not revise this !!!
    test_total_reward = []
    action_list = []
    for i in range(NUM_OF_TEST):
        actions = []
        state = env.reset()

        img = plt.imshow(env.render(mode='rgb_array'))

        total_reward = 0

        done = False
        env.reset()
        while not done:
            env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            actions.append(action)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            total_reward += reward
            state = next_state
            #img.set_data(env.render(mode='rgb_array'))
            #display.display(plt.gcf())
            #display.clear_output(wait=True)

        print(total_reward)
        test_total_reward.append(total_reward)

        action_list.append(actions)  # save the result of testing
    #torch.save(agent.network.state_dict,'models/agentNet.pth')
if __name__ == '__main__':
    fix(env, seed)
    #trainIters(actor, critic, n_iters=8000)
    actor.load_state_dict(torch.load('models/06_19_17_10_actor.pth', map_location='cpu'))
    critic.load_state_dict(torch.load('models/06_19_17_10_critic.pth', map_location='cpu'))
    test()