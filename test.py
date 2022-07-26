import matplotlib.pyplot as plt
import numpy


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm

import gym
import random
import math
def cate():
    m = Categorical(torch.tensor([ 1, 1, 1, 1 ]))
    action = m.sample()
    log_prob = m.log_prob(action)

    print(action)
    print(log_prob)

    print(math.log(0.25,math.e))

def loop():
    for i in range(10):
        n = 0
        while True:
            n+=1
            if n > 10:
                print('break',i)
                #只打断内层循环
                break
observations = ['alertness', 'hypertension', 'intoxication',
                    'time_since_slept', 'time_elapsed', 'work_done']

def make_heartpole_obs_space():
    lower_obs_bound = {
        'alertness': - np.inf,
        'hypertension': 0,
        'intoxication': 0,
        'time_since_slept': 0,
        'time_elapsed': 0,
        'work_done': - np.inf
    }
    higher_obs_bound = {
        'alertness': np.inf,
        'hypertension': np.inf,
        'intoxication': np.inf,
        'time_since_slept': np.inf,
        'time_elapsed': np.inf,
        'work_done': np.inf
    }

    low = np.array([lower_obs_bound[o] for o in observations])
    high = np.array([higher_obs_bound[o] for o in observations])
    shape = (len(observations),)
    return gym.spaces.Box(low,high,shape)


if __name__ == '__main__':
    observation = numpy.array([1])
    print(observation.shape)