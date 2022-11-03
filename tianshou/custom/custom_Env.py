import gym
import numpy
import numpy as np
from gym import spaces
from stable_baselines.common.env_checker import check_env
import imp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

obs = ['alertness', 'hypertension', 'intoxication',
                    'time_since_slept', 'time_elapsed', 'work_done']
state = {'alertness': 0.8071992996183379,
 'hypertension': 0.4560908860014237,
 'intoxication': 0.011758177199201024,
 'time_since_slept': 2.,
 'time_elapsed': 2.,
 'work_done': 0.645391786501011}
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

  low = np.array([lower_obs_bound[o] for o in obs])
  high = np.array([higher_obs_bound[o] for o in obs])
  shape = (len(obs),)
  return gym.spaces.Box(low, high, shape,dtype=np.float64)

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(10)
    # Example for using image as input:
    self.observation_space = make_heartpole_obs_space()
    self.observation = []

  def step(self, action):

    reward=0
    done = False
    info = {}
    observation = numpy.array([1,2,3,4,5,6])
    return observation, reward, done, info

  def reset(self):


    return np.array([state[o] for o in obs])  # reward, done, info can't be included




  def render(self, mode='human'):
    pass
  def close (self):
      pass

  '''Internal implementation'''
  def close_env(self):
      print('close_env')


env = CustomEnv('arg1')
env_space = env.observation_space
# It will check your custom environment and output additional warnings if needed
check_env(env)