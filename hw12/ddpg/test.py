import random

import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def fixed():
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    i = 5
    for i in range(100):
        print(i)

    pass