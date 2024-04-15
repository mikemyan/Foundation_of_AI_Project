import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import moving_average
from collections import namedtuple

# Define the neural network
class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        # Represents neural nets for policy approximation
        self.fc3_1 = nn.Linear(64, 2)
        # Represents neural nets for value function approximation
        self.fc3_2 = nn.Linear(64, 1)
        self.saved_actions = []
        self.rewards = []
    
    def feed_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_scores = self.fc3_1(x)
        state_values = self.fc3_2(x)
        prob_dist_over_actions = F.softmax(action_scores, dim=-1)
        return prob_dist_over_actions, state_values

# Initialize the network
net = ActorCriticNet()

def choose_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    prob_dist_over_actions, state_value = net.feed_forward(state)
    distribution = Categorical(prob_dist_over_actions)
    action = distribution.sample()
    return action.item(), distribution.log_prob(action), state_value

def main():
    optimizer = optim.RMSprop(net.parameters(), lr=0.003)
    env = gym.make('CartPole-v1', render_mode="human")
    num_episodes = 500
    max_steps = 500
    gamma = 0.99
    reward_threshold = 499
    log_interval = 10
    running_reward = []
    for i_episode in range(num_episodes):
        print("Episode: ", i_episode)
        state, info = env.reset()
        log_probs = []
        values = []
        rewards = []
        episode_rewards = []
        break_all_episodes = False
        for t in range(max_steps):
            # Choose an action, and get the log probability and state value
            action, log_prob, state_value = choose_action(state)
            log_probs.append(log_prob)
            values.append(state_value)

            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            episode_rewards.append(reward)
            if t == 499 and not terminated:
                print("Episode {} forced to terminate at time step 500".format(i_episode))
                print("Total reward: ", sum(rewards))
            if sum(rewards) > reward_threshold:
                break_all_episodes = True

            if terminated:
                # Calculate the discounted rewards
                R = 0
                returns = []
                for r in rewards[::-1]:
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

                # Calculate the loss for the policy and value function
                policy_loss = []
                value_loss = []
                for log_prob, value, R in zip(log_probs, values, returns):
                    advantage = R - value.item()
                    policy_loss.append(-log_prob * advantage)
                    value_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))
                optimizer.zero_grad()
                loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
                loss.backward()
                optimizer.step()

                
                if i_episode % 10 == 0:                
                    print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
                    print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, sum(rewards)))
                del log_probs[:]
                del values[:]
                del rewards[:]
                break
        running_reward.append(sum(episode_rewards))
        print("total running reward: ", sum(episode_rewards))
        print("running reward length: ", len(running_reward))
        print("running reward: ", running_reward)
                
        if break_all_episodes:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(sum(rewards), t))
            # break
    env.close()
    fig, ax = plt.subplots()
    ax.plot(running_reward, label='Reward')
    ax.plot(moving_average(running_reward, 10), label='Moving Average', linestyle='--')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    #plt.show()
    plt.savefig('culmulative_rewards_over_time_A2C.png')

if __name__ == '__main__':
    main()
# env = gym.make('CartPole-v1', render_mode="human")

# observation, info = env.reset()
# x = np.array(observation)


# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()