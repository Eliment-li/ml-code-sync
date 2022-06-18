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



if __name__ == '__main__':
    np.finfo(np.float32).eps.item()