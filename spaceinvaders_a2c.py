import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import moving_average
import random
from collections import deque
import torch.autograd as autograd
import cv2

# Define the neural network
class ActorNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorNet, self).__init__()
        # Define the actor network
        self.actor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding = 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 1),
            nn.ReLU()
        )

        self.actor_output = nn.Sequential(
            nn.Linear(self.output_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def output_size(self, input_shape):
        return self.actor(autograd.Variable(torch.zeros(1, * input_shape))).view(1, -1).size(1)
    
    def feed_forward(self, x):
        # x = x.to(mps_device)
        x = self.actor(x)
        x = x.view(x.size(0), -1)
        action_scores = self.actor_output(x)
        prob_dist_over_actions = F.softmax(action_scores, dim=-1)
        return prob_dist_over_actions

class CriticNet(nn.Module):
    def __init__(self, input_shape):
        super(CriticNet, self).__init__()
        # Define the critic network
        self.critic = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding = 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 1),
            nn.ReLU()
        )

        self.critic_output = nn.Sequential(
            nn.Linear(self.output_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def output_size(self, input_shape):
        return self.critic(autograd.Variable(torch.zeros(1, * input_shape))).view(1, -1).size(1)
    
    def feed_forward(self, x):
        # x = x.to(mps_device)
        x = self.critic(x)
        x = x.view(x.size(0), -1)
        state_values = self.critic_output(x)
        return state_values

# Initialize the network
env = gym.make('ALE/SpaceInvaders-v5')
input_shape = (4, 84, 84)
actor_nn = ActorNet(input_shape, num_actions = env.action_space.n)
critic_nn = CriticNet(input_shape)

# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")

# else:
#     mps_device = torch.device("mps")
#     actor_nn.to(mps_device)
#     critic_nn.to(mps_device)

# Define the optimizers
actor_optimizer = optim.Adam(actor_nn.parameters(), lr=0.0001)
critic_optimizer = optim.Adam(critic_nn.parameters(), lr=0.0005)

def choose_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    prob_dist_over_actions = actor_nn.feed_forward(state)
    distribution = Categorical(prob_dist_over_actions)
    action = distribution.sample()
    return action.item(), distribution.log_prob(action), distribution.entropy()

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

def preprocess_frame(screen, exclude, output):
    # Convert to gray scale
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    # Exclude the top, bottom, left and right of the screen
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]
    # Normalize pixel values
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # Resize the screen
    screen = cv2.resize(screen, (output, output), interpolation = cv2.INTER_AREA)
    return screen

def stack_frame(stacked_frames, frame, is_new):
    if is_new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame
    
    return stacked_frames

def main():
    num_episodes = 5000
    max_steps = 1000
    gamma = 0.99
    reward_threshold = 999
    log_interval = 10
    running_reward = []
    for i_episode in range(num_episodes):
        print("Episode: ", i_episode)
        # state, _ = env.reset()
        state = stack_frames(None, env.reset()[0], True)
        log_probs = []
        entropies = []
        values = []
        rewards = []
        masks = []
        episode_rewards = []
        break_all_episodes = False
        # for t in range(max_steps):
        time_step = 0
        while True:
            # Choose an action, and get the log probability and state value
            action, log_prob, entropy = choose_action(state)
            log_probs.append(log_prob)
            entropies.append(entropy)

            next_state, reward, terminated, truncated, info = env.step(action)
            rewards.append(torch.from_numpy(np.array([reward])))
            episode_rewards.append(reward)
            next_state = stack_frames(state, next_state, False)

            state_value = critic_nn.feed_forward(torch.from_numpy(state).float().unsqueeze(0))
            values.append(state_value)
            masks.append(torch.from_numpy(np.array([1 - terminated])))

            if sum(rewards) > reward_threshold:
                break_all_episodes = True
                break
            # train the network after every 100 time steps
            time_step += 1
            if time_step % 100 == 0:
                next_state_value = critic_nn.feed_forward(torch.from_numpy(next_state).float().unsqueeze(0))
                # Calculate the discounted rewards
                returns = []
                for index in reversed(range(len(rewards))):
                    next_state_value = rewards[index] + gamma * next_state_value * masks[index]
                    returns.insert(0, next_state_value)
                
                log_probs = torch.cat(log_probs)
                values = torch.cat(values)
                returns = torch.cat(returns).detach()

                advantage = returns - values
                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.001 * sum(entropies)
                
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()
                # returns = torch.tensor(returns)
                # returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
                log_probs = []
                values = []
                rewards = []
                masks = []
                entropies = []

            state = next_state
            if terminated:
                break
        running_reward.append(sum(episode_rewards))
        print("total running reward: ", sum(episode_rewards))
        print("running reward length: ", len(running_reward))
                
        if break_all_episodes:
            print("Solved! Running reward is now {}!".format(episode_rewards[-1]))
            break
    env.close()
    fig, ax = plt.subplots()
    ax.plot(running_reward, label='Reward')
    ax.plot(moving_average(running_reward, 10), label='Moving Average', linestyle='--')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    #plt.show()
    plt.savefig('culmulative_rewards_over_time_space_invader_A2C.png')

if __name__ == '__main__':
    main()