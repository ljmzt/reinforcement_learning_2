import torch
import collections
import numpy as np

class Runner():
    def __init__(self, n_episodes, n_steps, env, agent, gamma=0.99, lam=0.95):
        '''
          set up the runner; it will run for n_episodes, each episode goes for n_steps max
          return the stuffs needed for policy to learn
          this implement GAE version of the advantages
          the policy should return action, log_prob and estimate value for this state
        '''
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.lam = lam
    
    def eval_run(self):
        self.agent.eval()
        state = self.env.reset()
        rewards = 0.0
        for _ in range(self.n_steps):
            action, _, _ = self.agent.step(state)
            state, reward, done, _ = self.env.step(action)
            rewards += reward
            if done:  break
        return rewards
    
    def run(self):
        '''
          input the initial status, it will run according to the params in the init
          this is a training run
        '''
        self.agent.train()
        info = collections.defaultdict(list)
        actual_rewards = 0.0

        for _ in range(self.n_episodes):

            # loop over this episode
            state, rewards, values, dones = self.env.reset(), [], [], []
            for _ in range(self.n_steps):
                info['states'].append(state)
                action, log_prob, value = self.agent.step(state)
                info['actions'].append(action)
                info['log_probs'].append(log_prob.item())
                value = value.item()
                next_state, reward, done, _ = self.env.step(action)

                rewards.append(reward)
                values.append(value)
                dones.append(done)
                state = next_state
                if done:
                    break

            actual_rewards += sum(rewards)

            # compute the GAE
            n_steps = len(rewards)  # get the actual steps
            advantages, last_adv = np.zeros_like(rewards), 0
            values = np.array(values)
            for t in range(n_steps)[::-1]:
                if t == n_steps - 1:
                    _, _, next_value = self.agent.step(next_state)
                    next_value = next_value.item()
                else:
                    next_value = values[t+1]
                delta = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
                adv = delta + self.gamma * self.lam * (1 - dones[t]) * last_adv
                last_adv = adv
                advantages[t] = adv
            returns = advantages + values
            info['returns'].append(returns)
            info['advantages'].append(advantages)
            info['values'].append(values)

        info['states'] = torch.tensor(np.vstack(info['states']))
        info['returns'] = torch.tensor(np.hstack(info['returns'])).float()
        info['advantages'] = torch.tensor(np.hstack(info['advantages'])).float()
        if isinstance(info['actions'][0], np.int64):
            info['actions'] = torch.tensor(info['actions']).long()
        else:
            info['actions'] = torch.tensor(np.vstack(info['actions'])).float()
        info['log_probs'] = torch.tensor(info['log_probs']).float()
        info['values'] = torch.tensor(np.hstack(info['values'])).float()
        return info, actual_rewards/self.n_episodes
