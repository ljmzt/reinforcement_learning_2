### Purpose
This repository contains files for some RL tests on the lunar lander gym. 

It follows heavily on [Hung-Yi Lee's lecture](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php), [openai's baseline](https://github.com/openai/baselines/tree/master/baselines) (although in a much simplified form), and [this](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/train.py) and [this](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On) github pages.

For papers, the main references on Proximal Policy Optimization (PPO) and Generalized Advantage Estimation (GAE) are attached.

### Main files
1. debug-consistent-routines.ipynb: the main file for debugging routines and classes.

2. full-run.ipynb: the main file for running the training

3. env.py: wrapper for gym env, which allows users to normalize the reward and the action space for better convergence in the training. **Normalizing the reward** for the lunar lander is critical. It also allows the user to discretize the action space for the continuous case.

4. policy.py: implement the fully connected network as actor and critic. One can choose whether the actor and critic share parts of the network or not. In addition, one can also drop the critic, i.e. use pure policy gradient method. For continuous action space, the output distribution for action space is diag Gaussian, and for discrete action space, it is categorical.

5. agent.py: PPO agent that carries the policy and implements the step and train methods.

6. runner.py: runner that couples the agent and event. The main purpose is to prepare the info dictionary that the agent can use for training. The info dictionary contains the states, returns, advantages, actions, log probabilities and estimated values V(s). The notations used in computing the advantages are [s_t, a_t, r_t, s_t+1], where s_t is the state at the current step, a_t is the action taken, r_t is the reward received, and s_t+t is the next state.

7. trainer.py: the usual trainer that calls the runner then the agent for learning, and records some outputs for plotting.

### Technical notes
1. For this specific example, I only need to scale the reward by multiplying 0.01. I didn't scale the advantages by removing the mean then scale to advantages to std 1.0. I think in a more general case, one needs to rescale the advantages as many authors suggest.

2. Exploration is done implicitly by using distribution.sample() for both the discrete and continuous cases. Other examples may be better off to tune this explicitly.

3. Parameters used
| Parameter | Value |
| --------- | ----- |
| a | 1.0 |


### Things to do
1. When one discretizes the continous case, it may be useful to keep the order by following Tang and Agrawal's paper, which is also attached.