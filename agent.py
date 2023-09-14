from torch import optim
import torch
import numpy as np
from torch import nn

class PPOAgent():
    def __init__(self, policy, device='cpu', lr=1e-4, batch_size=1024, K=10, eps=0.2):
        self.device = device
        self.policy = policy.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.K = K
        self.batch_size = batch_size
        self.eval_mode = False
        self.eps = 0.2
    
    def train(self):
        self.eval_mode = False
    
    def eval(self):
        self.eval_mode = True

    def step(self, state):
        # input state is a 1D numpy array, usually from env.step or env.reset
        with torch.no_grad():
            state = torch.tensor(state).reshape(1,-1).to(self.device)
            dist, value = self.policy(state)
            if self.eval_mode:
                action = dist.mode
                log_prob = dist.log_prob(action)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)
        action = action.cpu().numpy()[0]  # convert back to np array
        if self.policy.env.type == 'dis-continuous':
            log_prob = log_prob.sum(axis=-1)
        return action, log_prob, value

    def learn(self, info):
        '''
          the info here should have states, log_probs, advantages, values
        '''
        n = len(info['advantages'])

        ### do K update 
        for _ in range(self.K):

            # loop over each batch
            idx = np.arange(n)
            np.random.shuffle(idx)
            for start in np.arange(0, n, self.batch_size):
                ijk = idx[start:min(start+self.batch_size, n)]
                states = info['states'][ijk, :].to(self.device)
                advantages = info['advantages'][ijk].to(self.device)
                log_probs_old = info['log_probs'][ijk].to(self.device)
                actions = info['actions'][ijk].to(self.device)
                values_old = info['values'][ijk].to(self.device)
                returns = info['returns'][ijk].to(self.device)

                # collect the off policy
                dist, values = self.policy(states)
                log_probs = dist.log_prob(actions)
                if self.policy.env.type == 'dis-continuous':
                    log_probs = log_probs.sum(axis=-1)

                # get the policy gradients
                r = torch.exp(log_probs - log_probs_old)
                r_clipped = torch.clip(r, 1.0 - self.eps, 1.0 + self.eps)
                loss_1 = - (torch.min(r*advantages, r_clipped*advantages)).mean()

                # get the mse of the values
#                values_clipped = values_old + torch.clip(values - values_old, 1.0 - self.eps, 1.0 + self.eps)
#                loss_2 = torch.max(torch.square(values_clipped - returns), torch.square(values - returns)).mean()
                loss_2 = nn.MSELoss()(values, returns)
#                 loss_2 = F.smooth_l1_loss(values, returns)

                loss = loss_1 + 0.5 * loss_2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss_2.item()
