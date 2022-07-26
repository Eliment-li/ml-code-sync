from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.policy import BasePolicy

class REINFORCEPolicy(BasePolicy):
  """Implementation of REINFORCE algorithm."""
  def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer,):
    super().__init__()
    self.actor = model
    self.optim = optim
    # action distribution
    self.dist_fn = torch.distributions.Categorical

  def forward(self, batch: Batch) -> Batch:
    """Compute action over the given batch data."""
    logits, _ = self.actor(batch.obs)
    dist = self.dist_fn(logits)
    act = dist.sample()
    return Batch(act=act, dist=dist)

  def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
    """Compute the discounted returns for each transition."""
    returns, _ = self.compute_episodic_return(batch, buffer, indices, gamma=0.99, gae_lambda=1.0)
    batch.returns = returns
    return batch

  def learn(self, batch: Batch, batch_size: int, repeat: int) -> Dict[str, List[float]]:
    """Perform the back-propagation."""
    logging_losses = []
    for _ in range(repeat):
        for minibatch in batch.split(batch_size, merge_last=True):
            self.optim.zero_grad()
            result = self(minibatch)
            dist = result.dist
            act = to_torch_as(minibatch.act, result.act)
            ret = to_torch(minibatch.returns, torch.float, result.act.device)
            log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
            loss = -(log_prob * ret).mean()
            loss.backward()
            self.optim.step()
            logging_losses.append(loss.item())
    return {"loss": logging_losses}

from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor
import warnings

def init():
    warnings.filterwarnings('ignore')
    state_shape = 4
    action_shape = 2
    net = Net(state_shape, hidden_sizes=[16, 16], device="cpu")
    actor = Actor(net, action_shape, device="cpu").to("cpu")
    optim = torch.optim.Adam(actor.parameters(), lr=0.0003)
    policy = REINFORCEPolicy(actor, optim)

    #SaveAndLoad
    torch.save(policy.state_dict(), 'policy.pth')
    assert policy.load_state_dict(torch.load('policy.pth'))

if __name__ == '__main__':
    import gym
    from tianshou.data import Batch, ReplayBuffer

    # a buffer is initialised with its maxsize set to 20.
    print("========================================")
    buf = ReplayBuffer(size=12)
    env = gym.make("CartPole-v0")

    warnings.filterwarnings('ignore')
    state_shape = 4
    action_shape = 2
    net = Net(state_shape, hidden_sizes=[16, 16], device="cpu")
    actor = Actor(net, action_shape, device="cpu").to("cpu")
    optim = torch.optim.Adam(actor.parameters(), lr=0.0003)

    policy = REINFORCEPolicy(actor, optim)

    obs = env.reset()
    for i in range(3):
        '''
        obs=obs[np.newaxis, :] 给obs增加维度
        例如obs=[1,  2, 3,  4] 升维后 ->[[1,  2, 3, 4]]
        obs 作为 actor 的输入,必须是二维的
        '''
        act = policy(Batch(obs=obs[np.newaxis, :])).act.item()
        obs_next, rew, done, info = env.step(act)
        # pretend ending at step 3
        done = True if i == 2 else False
        info["id"] = i
        buf.add(Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next, info=info))
        obs_next = obs