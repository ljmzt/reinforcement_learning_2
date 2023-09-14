from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import numpy as np

class EnvWrapper():
    '''
      wrapper around env
      for discrete env:
        scale the reward only
      for continuous env:
        1. if make discrete, then make it discrete
        2. if not, scale the input action (-1,1) to proper action space
      implement reset and step
      NOTE: for lunar lander, better to set reward_scale=0.01 to get a lot better convergence
    '''
    def __init__(self, env, reward_scale=1.0, make_discrete=None):
        self.env = env
        self.env.reset()
        self.reward_scale = reward_scale
        if isinstance(env.action_space, Discrete):
            self.type = 'discrete'
            self.K = env.action_space.n
            pass
        else:
            if make_discrete is None:
                self.type = 'continuous'
                lows, highs = env.action_space.low, env.action_space.high
                self.action_space_means = (highs + lows) / 2.0
                self.action_space_stds = (highs - lows) / 2.0
                self.A = env.action_space.shape[0]
            else:
                self.type = 'dis-continuous'
                self.discrete_actions = np.array([np.linspace(low, high, make_discrete) \
                                                  for low, high in zip(env.action_space.low, env.action_space.high)])
                self.K = make_discrete
                self.A = env.action_space.shape[0]                
        
    def reset(self):
        return self.env.reset()
        
    def step(self, action):
        '''
          for discrete: action is int, from range(K)
          for continuous: action is [A], A is the dimension of action_space
          for dis-continuous: action is [A], A is the dimension of discrete_actions.shape[0], each is from range(K)
        '''
        if self.type == 'discrete':
            pass
        elif self.type == 'continuous':
            action = action * self.action_space_stds + self.action_space_means
        else:
            action = np.take_along_axis(self.discrete_actions, action.reshape(-1,1), axis=1).reshape(-1)
        state, reward, done, info = self.env.step(action)
        reward = reward * self.reward_scale
        return state, reward, done, info
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
