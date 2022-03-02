import random
import time

import numpy as np
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from agent import DDPGAgent
from environment import Environment


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(env, agent, n_episodes=2000, print_freq=100):
    """
    :param agent: agent.Agent class instance
    :param env: environment class instance compatible with OpenAI gym
    :param n_episodes: (int) maximum number of training episodes
    :param print_freq: (int) print frequency of episodic score
    :return: scores: (list[float]) scores of last 100 episodes
    """
    scores = []
    for i_episode in range(1, n_episodes + 1):
        t0 = time.time()
        state = env.reset(options={'train_mode': True})
        agent.reset()
        score = 0
        while True:
            action = [agent.act(state[i], i) for i in range(agent.n_agents)]
            next_state, reward, done, _ = env.step(action)
            for i in range(agent.n_agents):
                experience_dict = {'state': state[i], 'action': action[i], 'reward': reward[i], 'next_state': next_state[i], 'done': done[i]}
                agent.update_buffer(experience_dict)
            agent.step()
            state = next_state
            score += np.mean(reward)
            if any(done):
                break
        scores.append(score)
        print(f'\rEpisode {i_episode} | LAST: {scores[-1]:3.2f} | Mean: {np.mean(scores[-100:]):3.2f} | Max: {np.max(scores[-100:]):3.2f} | Min: {np.min(scores[-100:]):3.2f} | Time: {int(time.time()-t0)}',
              end="\n" if i_episode % print_freq == 0 else "")
        if np.mean(scores[-100:]) > 30:
            print('\nSOLVED!')
            env.reset(options={'train_mode': True})
            return scores
    env.reset(options={'train_mode': True})
    return scores


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--env_path', type=str, default='Reacher_Linux/Reacher.x86_64', help='path for unity environment')
    parser.add_argument('--save_path', type=str, default='agent.ckpt', help='save path for trained agent weights')
    parser.add_argument('--actor_lr', type=float, default=1e-3, help='learning rate for updating actor')  # 1e-4
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='learning rate for updating critic')  # 1e-3
    parser.add_argument('--tau', type=float, default=1e-3, help='learning rate for updating target q-network')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor when calculating return')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='maximum number of experiences to save in replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='number of experiences to do one step of update')
    parser.add_argument('--n_episodes', type=int, default=2000, help='total number of episodes to play')
    parser.add_argument('--print_freq', type=int, default=10, help='print training status every ~ steps')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    print(args)

    seed_all(args.seed)

    env = Environment(args.env_path)

    agent = DDPGAgent(
        state_size=env.observation_size[0],
        action_size=env.action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size
    )

    scores = train(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        print_freq=args.print_freq
    )

    torch.save(agent.state_dict(), args.save_path)

    plt.figure(figsize=(20, 20))
    plt.plot(scores)
    plt.title(f'MAX{np.max(scores):3.2f} | LAST{scores[-1]:3.2f}')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
