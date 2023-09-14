from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import torch
import numpy as np

class DiagGaussian():
    def __init__(self, mean, logstd):
        '''
           mean is [N, size], normaly from policy
           logstd is [1, size]
        '''
        self.mode = self.mean = mean
        self.logstd = logstd.reshape(1,-1)
        self.std = logstd.exp()
    
    def log_prob(self, x):
        '''
          x is [N, size]
        '''
        k = x.shape[-1]
        log_prob = -0.5 * k * np.log(2.0 * np.pi) - torch.sum(self.logstd) \
                   - 0.5 * torch.square((x - self.mean) / self.std).sum(axis=-1)
        return log_prob
    
    def sample(self):
        return torch.randn(*self.mean.shape).to(self.std.device) * self.std + self.mean

class Policy(nn.Module):
    '''
      policy for the lunar landing gym, which has 1 hidden layer
      if use_critic = False, it won't predict critic, 
        i.e. a pure policy gradient method should be used by setting lam=1.0 in the runner
      for continuous, use 
        init_logstd=0, std = exp(0)=1; 
        init_logstd=-1, std = exp(-1)=0.36
        init_logstd=-2, std = exp(-2)=0.13
    '''
    def __init__(self, env, num_hidden=16, share=True, use_critic=True, init_logstd=0.0):
        super().__init__()
        self.env = env
        self.share = share
        self.use_critic = use_critic
        
        self.actor_latent = nn.Sequential(
            nn.Linear(env.env.observation_space.shape[0], num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh()
        )
        if use_critic and share == False:
            self.value_latent = nn.Sequential(
                nn.Linear(env.env.observation_space.shape[0], num_hidden),
                nn.Tanh(),
                nn.Linear(num_hidden, num_hidden),
                nn.Tanh()
        )
        
        if env.type == 'discrete':
            self.actor_fc = nn.Linear(num_hidden, env.K)
        elif env.type == 'continuous':
            self.actor_fc = nn.Linear(num_hidden, env.A)
            self.logstd = nn.Parameter(torch.ones(env.A) * init_logstd)  # start from exp(0)=1
        elif env.type == 'dis-continuous':
            self.actor_fc = nn.Linear(num_hidden, env.A * env.K)
        
        if use_critic:
            self.critic_fc = nn.Linear(num_hidden, 1)

    def forward(self, state):
        '''
          state is [N, num_state] tensor
          output is a distribution, and value of shape [N]
        '''
        # actor
        x = self.actor_latent(state)
        
        if self.env.type == 'discrete':
            z = F.softmax(self.actor_fc(x), dim=-1)
            dist = Categorical(z)
        elif self.env.type == 'continuous':
            means = self.actor_fc(x)
            dist = DiagGaussian(means, self.logstd)
        elif self.env.type == 'dis-continuous':
            z = self.actor_fc(x).reshape(state.shape[0], self.env.A, self.env.K)
            dist = Categorical(F.softmax(z, dim=-1))

        # critic
        if self.use_critic:
            if self.share == False:
                x = self.value_latent(state)
            value = self.critic_fc(x).squeeze(-1)
        else:
            value = torch.zeros(state.shape[0], device=x.device)

        return dist, value
