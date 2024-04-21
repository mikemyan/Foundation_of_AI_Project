import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as T


# env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
env = gym.make("ALE/SpaceInvaders-v5")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, buffer_size):
        self.memory = deque([], maxlen=buffer_size)


    def push(self, *args):
        """Save a transition to replay mem"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Get a random sample from memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    


class DQN(nn.Module):

    # def __init__(self, n_observations, n_actions):
    #     super(DQN, self).__init__()
    #     self.fc1 = nn.Linear(n_observations, 128)
    #     self.fc2 = nn.Linear(128, 128)
    #     self.fc3 = nn.Linear(128, n_actions)

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        # self.seed = torch.manual_seed(seed)
        # print(n_observations)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Adjust according to the output size of last conv layer

        self.fc1 = nn.Linear(64*7*7, 512)
        # 6 actions
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        """
        mapping a state to action-values.
        ---
        args:
            x: state tensor (grayscale img)
        returns:
            q_values: array of length 6. It corresponds to the action-values for each action given the input state
                q_values=[Q(state, a_1), Q(state, a_2), ..., Q(state, a_6)]
        """

        # forward pass through conv layers
        # print("Input size:", x.size())  # Should show [batch_size, 1, 84, 84]

        x = F.relu(self.conv1(x))
        # print("After Conv1 size:", x.size())  # Check size

        x = F.relu(self.conv2(x))
        # print("After Conv2 size:", x.size())  # Check size

        x = F.relu(self.conv3(x))
        # print("After Conv3 size:", x.size())  # Check size

        # flatten the tensor for the fc layers
        # x = x.view(-1, 128 * 19 * 8)
        x = x.view(-1, 64 * 7 * 7)

        # forward pass through fc layers
        x = F.relu(self.fc1(x))

        return self.fc2(x)



# Define the transformation pipeline
transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.Resize((84, 84)),  # Resize to 84x84 pixels
    T.ToTensor(),        # Converts to tensor and scales to [0, 1]
])

def preprocess(state):
    # Convert numpy array to PyTorch tensor if not already a tensor
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float().to(device)

    # print("Original state shape:", state.shape)  # Debug: Check the shape of the input state

    # Ensure the state is in the correct [C, H, W] format
    if state.dim() == 4:  # Likely [B, H, W, C]
        state = state.squeeze(0)  # Assuming the first dimension is batch size

    if state.dim() == 3 and state.shape[-1] in [1, 3, 4]:  # [H, W, C]
        state = state.permute(2, 0, 1)  # Convert to [C, H, W]

    # print("After permute (if applied) state shape:", state.shape)  # Debug: Check after permute

    # Apply transformations and ensure it's ready for neural network input
    state = transform(state)  # Apply transformation
    state = state.unsqueeze(0)  # Add back the batch dimension if needed

    # print("Final state shape:", state.shape)  # Debug: Final check before return

    return state


#***TRAINING****
# *hyper params and utilities:

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer


BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 996
TAU = 0.003
LR = .0005
BUFFER_SIZE = 50000
# max_timesteps = 1000

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
# print(n_observations)

# input_shape = (4, 84, 84)
input_shape = (1, 84, 84)

policy_net = DQN(input_shape, n_actions).to(device)
target_net = DQN(input_shape, n_actions).to(device)

# policy_net.load_state_dict(torch.load("policy_net_weights.pth"))
# target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR) #amsgrad=True? last arg needed?
memory = ReplayMemory(BUFFER_SIZE)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_scores = []


def plot_scores(scores, show_result=False):
    ### maybe make the score a score / duration_of_episode

    scores_t = torch.tensor(scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


#***TRAINING****
# *actual training loop

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute MSE LOSS
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    
if torch.cuda.is_available():
    num_episodes = 10000
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    total_reward = 0
    
    #print episode
    print(i_episode)
    
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    state = preprocess(state)
    state = state.to(device)

    for t in count():
        action = select_action(state)
        action = torch.tensor([[action]], device=device, dtype=torch.long)
        
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)


        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = preprocess(observation)
            next_state = next_state.to(device)

        # next_state = preprocess(unprocessed_next_state)
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Add the reward of each step to total_reward
        total_reward += reward

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # if done or t > max_timesteps:
        if done:
            episode_scores.append(total_reward)
            
            #TEST
            # print(episode_scores)

            plot_scores(episode_scores)
            break

print('Complete')
plot_scores(episode_scores, show_result=True)
plt.ioff()
plt.savefig('spaceinvadersDQN_training_scores_10000.png')
plt.show()

# torch.save(target_net.state_dict(), "target_net_weights_LUNAR.pth")
# torch.save(policy_net.state_dict(), "policy_net_weights_LUNAR.pth")


#***TESTING***