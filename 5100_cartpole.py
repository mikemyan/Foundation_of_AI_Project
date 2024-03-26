import json
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matrix_mdp
import sys
import matplotlib.pyplot as plt
import time

#######################################
# 1. Initialize transition and reward matrices
# 2. Fill transition and reward matrices with correct values
#######################################

num_states = 5
num_actions = 4

# Create transition and reward matrices below:

T = np.zeros((num_states, num_states, num_actions))
R = np.zeros((num_states, num_states, num_actions))

# Set the entries in T and R below as per the MDP used in the assignment diagram:

'''

YOUR CODE HERE


'''
R[4, 1, 2] = 10
R[4, 2, 3] = 10

p, q, s = 0.5, 0.5, 0.5
r = 1
T[0, 0, 0] = p
T[1, 0, 0] = 1 - p
T[2, 0, 1] = q
T[1, 0, 1] = 1 - q
T[2, 1, 2] = r / 3
T[3, 1, 2] = r / 3
T[4, 1, 2] = r / 3
T[0, 2, 3] = 1 - s
T[4, 2, 3] = s

#######################################
# 3. Map indices to action-labels 
#######################################

# map each index to a human-readable action-label. 
# There are only 2 actions in this case, move left and move right.
A = {0 : "LEFT", 1 : "RIGHT"}


#######################################
# Initialize the gymnasium environment
#######################################


#P_0 = np.array([1, 0, 0, 0, 0])    # This is simply the initial probability distribution which denotes where your agent is, i.e. the start state.

# Import the CartPole gymnasium environment
env=gym.make('CartPole-v1', render_mode = 'human')


#######################################
# 4. Random exploration
#######################################

'''
First, we reset the environment and get the initial observation.

The observation includes cart position, cart velocity, pole angle, and pole angular velocity.

The observation tells you which state you are in - in this case, indices 0-4 map to states S1 - S5.

Since we set P_0 = [1, 0, 0, 0, 0], the initial state is always S1 after env.reset() is called.
'''

observation, info = env.reset()

'''
Below, complete the function for random exploration, i.e. randomly choosing an action at each time-step and executing it.

A random action is simply a random integer between 0 and the number of actions (num_actions not inclusive).
However, you should make sure that the chosen action can actually be taken from the current state.
If it is not a legal move, generate a new random move. You can use the transition probabilities to figure out
whether a given action is valid from the current state.

Avoid hardcoding actions even for states where there is only one action available. That way, your
code is more general, and may be easily adapted to a different environment later.

You will use the following line of code to explore at each time step:

observation, reward, terminated, truncated, info = env.step(action)

The above line of code is used to take one step in the environment using the chosen action.
It takes as input the action chosen by the agent, and returns the next observation (i.e., state),
reward, whether the episode terminated (terminal states), whether the episode was 
truncated (max iterations reached), and additional information.

If at any point the episode is terminated (this happens when we reach a terminal state, 
and the env.step() function returns True for terminated), you should
end the episode in order to reset the environment, and start a new one.

Keep track of the total reward in each episode, and reset the environment when the episode terminates.

Print the average reward obtained over 10000 episodes. 

'''

def random_exploration():

    '''


    YOUR CODE HERE
    

    '''
    observation, info = env.reset()
    total_reward = 0
    total_episodes = 10000
    current_episode = 10000
    while current_episode > 0:
        legal_action = False
        while not legal_action:
            action = np.random.randint(0, num_actions)
            for i in range(len(T)):
                for j in range(len(T[i])):
                    if j == observation and T[i, j, action] != 0:
                        legal_action = True

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            observation, info = env.reset()
            current_episode -= 1
    avg_reward = total_reward / total_episodes
    print(avg_reward)
    return avg_reward


#######################################
# 5 - 7. Policy evaluation 
#        & Plotting V_pi(s)
#######################################

gamma = 0.9

'''
Fill in the following function to evaluate a given policy.

The policy is a dictionary that maps each state to an action.
The key is an integer representing the state, and the value is an integer representing the action.

Initialize the value function V(s) = 0 for all states s.
Perform the Bellman update iteratively to update the value function.
Plot the value of S1 over time (i.e., at each iteration of Bellman update).
Save and insert this plot into the written submission.

Function returns the expected values for the first two states after all iterations.

Functions are called within main function at the end of the file.

'''
    
def evaluate_policy(policy, num_iterations, plot_filename):
    # Initialize value function to 0 for all states
    V = { i : 0 for i in range(num_states) }
    '''

    YOUR CODE HERE

    '''
    # Create a list to store the value of S1 over time
    first_state_value_over_time = [V[0]]
    while num_iterations > 0:
        # Temporary dictionary to store the value of each state in the current iteration
        curr_iter_V = {}
        # Iterate through all states except the terminal states
        for j in range(num_states - 2):
            curr_val = 0
            action = policy[j]
            # Iterate through all possible next states and calculate the value of the current state
            for k in range(num_states):
                if T[k, j, action] != 0:
                    curr_val += T[k, j, action] * (R[k, j, action] + gamma * V[k])
            curr_iter_V[j] = curr_val
            if j == 0:
                first_state_value_over_time.append(curr_val)
        V.update(curr_iter_V)
        num_iterations -= 1
    plt.figure()
    plt.plot(first_state_value_over_time)
    plt.xlabel("Iterations")
    plt.ylabel("V(S1)")
    plt.title("Policy Evaluation")
    plt.savefig(plot_filename) # Save the plot to the filename provided

    return V[0], V[1]


#######################################
# 8. Value Iteration for Best Policy
# 9. Output Best Policy
#######################################

'''
Initialize the value function V(s) = 0 for all states s.

Use value iteration to find the optimal policy for the MDP.

The optimal policy should be encoded as a dictionary that maps each state to an action.
The key is an integer representing the state, and the value is an integer representing the action.

Plot V_opt(S1) over time (i.e., at each iteration of Bellman update).
Please use plt.savefig() to save the plot, and do not use plt.show().
Save and insert this plot into the written submission.

Print the optimal policy after 100 iterations.

'''

def value_iteration(num_iterations, plot_filename):
    V_opt={ i : 0 for i in range(num_states) }
    pi_opt={}

    '''

    YOUR CODE HERE

    '''
    # Create a list to store the value of S1 over time
    first_state_value_over_time = [V_opt[0]]
    while num_iterations > 0:
        # Temporary dictionary to store the value of each state in the current iteration
        curr_iter_V = {}
        # Iterate through all states except the terminal states
        for j in range(num_states - 2):
            actions_vals_arr = []
            # Dictionary to store the value of each action for the current state
            action_val_relations = {}
            # Iterate through all possible actions and calculate the value of the current state
            for action in range(num_actions):
                curr_val = 0
                # Iterate through all possible next states and calculate the value of the current state
                for k in range(num_states):
                    if T[k, j, action] != 0:
                        curr_val += T[k, j, action] * (R[k, j, action] + gamma * V_opt[k])
                actions_vals_arr.append(curr_val)
                action_val_relations[action] = curr_val
            curr_iter_V[j] = max(actions_vals_arr)
            if j == 0:
                first_state_value_over_time.append(curr_iter_V[j])
            # Find the action that gives the maximum value for the current state
            pi_opt[j] = list(action_val_relations.keys())[list(action_val_relations.values()).index(curr_iter_V[j])]
        V_opt.update(curr_iter_V)
        num_iterations -= 1
    plt.figure()
    plt.plot(first_state_value_over_time)
    plt.xlabel("Iterations")
    plt.ylabel("V(S1)")
    plt.title("Value Iteration")
    plt.savefig(plot_filename) # Save the plot to the filename provided
    
    return pi_opt


#######################################
# Main function
#######################################


def main():
    avg_reward = random_exploration()
    
    # Set first policy - S1 : A1, S2 : A3, S3 : A4, S4: no action, S5: no action
    
    policy_1 = { 0 : 0, 1 : 2, 2 : 3}
    V1_1, V2_1 = evaluate_policy(policy_1, 100, "V_pi_1.png")

    # Set second policy - S1 : A2, S2 : A3, S3 : A4, S4: no action, S5: no action
    
    policy_2 = {0 : 1, 1 : 2, 2 : 3}
    V1_2, V2_2 = evaluate_policy(policy_2, 100, "V_pi_2.png")
    
    optimal_policy = value_iteration(100, "V_opt.png")
    
#######################################
# DO NOT CHANGE THE FOLLOWING - AUTOGRADER JSON DUMP
#######################################

    answer = {
        "average_reward": avg_reward,
        "V1_1": V1_1,
        "V2_1": V2_1,
        "V1_2": V1_2,
        "V2_2": V2_2,
        "optimal_policy": optimal_policy,
    }

    with open("answers_MDP.json", "w") as outfile:
        json.dump(answer, outfile)


if __name__ == "__main__":
    main()
