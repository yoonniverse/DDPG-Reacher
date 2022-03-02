import numpy as np
import random
import copy
from collections import namedtuple, deque

from net import Actor, Critic

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(local_model, target_model, tau):
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    :param local_model: (nn.module) weights will be copied from
    :param target_model: (nn.module) weights will be copied to
    :param tau: (float) interpolation parameter
    :return: 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DDPGAgent(nn.Module):
    def __init__(self, state_size, action_size, actor_lr=5e-4, critic_lr=5e-4, tau=1e-3, gamma=0.99, buffer_size=int(1e6), batch_size=512):
        super(DDPGAgent, self).__init__()
        """
        Interacts with and learns from the environment.
        :param state_size: (int)
        :param action_size: (int)
        :param actor_lr: (float)
        :param critic_lr: (float)
        :param tau: (float) soft update rate of target parameters
        :param gamma: (float) discount factor
        :param update_freq: (int) update local & target network every n steps
        :param buffer_size: (int)
        :param batch_size: (int) how many samples to use when doing single update
        """
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_agents = 20

        # actor
        self.actor_local = Actor(self.state_size, self.action_size).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor_local)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # critic
        self.critic_local = Critic(self.state_size, self.action_size).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic_local)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr)

        # action noise process for each agent
        self.noise = [OUNoise(action_size) for _ in range(self.n_agents)]

        # replay buffer
        self.buffer = ReplayBuffer(self.state_size, self.action_size, self.buffer_size, self.batch_size)

        # initialize time step
        self.t_step = 0

    def update_buffer(self, experience_dict):
        self.buffer.add(experience_dict)

    def step(self):
        self.t_step += 1
        if len(self.buffer) > self.batch_size:
            experiences = self.buffer.sample()
            self.learn(experiences)

    def act(self, state, agent_idx, add_noise=True):
        """
        produce action from state using actor_local
        :param state: (np.array) [state_size]
        :param agent_idx: (int)
        :param add_noise: (bool) whether to add noise to action for exploration
        :return: action (np.array) [action_size]
        """
        state = torch.from_numpy(state).to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state.unsqueeze(0)).squeeze(0).cpu().numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise[agent_idx].sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """
        reset agent before each episode of training
        """
        for n in self.noise:
            n.reset()

    def learn(self, e):
        """
        Update value parameters using given batch of experience tuples.
        :param e: experience dictionary with keys {state, action, reward, next_state, done}
        """

        # update critic
        with torch.no_grad():
            next_action = self.actor_target(e['next_state'])
            next_q = self.critic_target(e['next_state'], next_action)
            target_q = e['reward'].unsqueeze(-1) + self.gamma * next_q * (1 - e['done'].unsqueeze(-1))  # IMPORTANT: unsqueeze to match dimension
        q = self.critic_local(e['state'], e['action'])
        critic_loss = ((target_q - q) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        action = self.actor_local(e['state'])
        actor_loss = -self.critic_local(e['state'], action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target network towards local network
        soft_update(self.actor_local, self.actor_target, self.tau)
        soft_update(self.critic_local, self.critic_target, self.tau)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu.copy()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:

    def __init__(self, state_size, action_size, buffer_size, batch_size):
        """
        Fixed-size buffer to store experience tuples.
        :param state_size: (int)
        :param action_size: (int)
        :param buffer_size: (int)
        :param batch_size: (int)
        """
        self.memory = {
            'state': np.zeros((buffer_size, state_size), dtype=np.float32),
            'action': np.zeros((buffer_size, action_size), dtype=np.float32),
            'reward': np.zeros(buffer_size, dtype=np.float32),
            'next_state': np.zeros((buffer_size, state_size), dtype=np.float32),
            'done': np.zeros(buffer_size, dtype=np.float32)
        }
        self.memory_keys = set(self.memory.keys())
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def add(self, experience_dict):
        """
        Add a new experience to memory.
        :param experience_dict: experience dictionary with keys {state, action, reward, next_state, done}
        """
        assert self.memory_keys == set(experience_dict.keys())
        for k in self.memory_keys:
            self.memory[k][self.ptr] = experience_dict[k]
        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def reset(self):
        self.ptr = 0
        self.size = 0

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        idx = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=False)
        out = {k: torch.from_numpy(self.memory[k][idx]).to(device=DEVICE) for k in self.memory_keys}
        return out

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return self.size
