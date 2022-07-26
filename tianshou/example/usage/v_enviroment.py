'''
Tianshou provides vectorized environment wrapper for a Gym environment.
This wrapper allows you to make use of multiple cpu cores in your server to accelerate the data sampling.

https://colab.research.google.com/drive/1ABk2BgjzvC4DZu1rDxGzd2Uqjo3FRLEy?usp=sharing
'''


from tianshou.env import SubprocVectorEnv
import numpy as np
import gym
import time

def init():

    num_cpus = [1, 2, 5,8]
    for num_cpu in num_cpus:
        env = SubprocVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(num_cpu)])
        env.reset()
        sampled_steps = 0
        time_start = time.time()
        while sampled_steps < 1000:
            act = np.random.choice(2, size=num_cpu)
            obs, rew, done, info = env.step(act)
            if np.sum(done):
                env.reset(np.where(done)[0])
            sampled_steps += num_cpu
        time_used = time.time() - time_start
        print("{}s used to sample 1000 steps if using {} cpus.".format(time_used, num_cpu))

if __name__ == '__main__':
    init()

