from tqdm import tqdm
import torch
from agent import PPOAgent
from runner import Runner
import numpy as np

class Trainer():
    def __init__(self, env, policy, device='cpu', batch_size=1024, lr=1e-4, n_episodes=10, n_steps=1000, lam=0.95):
        self.agent = PPOAgent(policy, device=device, batch_size=batch_size, lr=lr)
        self.runner = Runner(n_episodes=n_episodes, n_steps=n_steps, env=env, agent=self.agent, lam=lam)
        
    def train(self, save_file, n_epoches=300, eval_everyepoch=10, eval_n_episodes=1, keepstd=False):
        progress_bar = tqdm(range(n_epoches), ncols=250)
        output_rewards, output_mse_loss, output_timesteps, eval_rewards = [], [], [], []
        if keepstd:
            output_stds = []
        
        eval_reward_best = -999.0
        for i, _ in enumerate(progress_bar):
            
            info, actual_reward = self.runner.run()
            mse_loss = self.agent.learn(info)
            
            output_rewards.append(actual_reward)
            output_mse_loss.append(mse_loss)
            output_timesteps.append(len(info['values'])/self.runner.n_episodes)
            if keepstd:
                output_stds.append(np.exp(self.agent.policy.logstd.detach().numpy()))
        
            if i % eval_everyepoch == 0:
                eval_reward = 0.0
                for _ in range(eval_n_episodes):
                    eval_reward += self.runner.eval_run()
                eval_reward /= eval_n_episodes
                if eval_reward > eval_reward_best:
                    with open(save_file,'wb') as fid:
                        torch.save(self.agent.policy.state_dict(), fid)
                eval_rewards.append(eval_reward)
        
            info_string = 'train reward {:.3f} trial steps {} mse value loss {:.3f} eval reward {:.3f}'\
                        .format(output_rewards[-1], output_timesteps[-1], output_mse_loss[-1], eval_rewards[-1])
            progress_bar.set_description(info_string)
        
        outputs = [output_rewards, output_timesteps, output_mse_loss, eval_rewards]
        if keepstd:
            outputs.append(np.vstack(output_stds))
        return outputs
