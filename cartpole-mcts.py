# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import os
import time
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import moving_average

import gym
from gym import logger
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from mcts import MCTSAgent

# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #
SEED = 28
EPISODES = 5
ENVIRONMENT = 'CartPole-v1'
LOGGER_LEVEL = logger.WARN
ITERATION_BUDGET = 80
LOOKAHEAD_TARGET = 100
MAX_EPISODE_STEPS = 1000
VIDEO_BASEPATH = './video'
START_CP = 20

# ---------------------------------------------------------------------------- #
#                                   Main loop                                  #
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    random.seed(SEED)
    parser = argparse.ArgumentParser(
        description='Run a Monte Carlo Tree Search agent on the Cartpole environment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', nargs='?', default=ENVIRONMENT,
                        help='The environment to run (only CartPole-v0 is supperted)')
    parser.add_argument('--episodes', nargs='?', default=EPISODES, type=int,
                        help='The number of episodes to run.')
    parser.add_argument('--iteration_budget', nargs='?', default=ITERATION_BUDGET, type=int,
                        help='The number of iterations for each search step. Increasing this should lead to better performance.')
    parser.add_argument('--lookahead_target', nargs='?', default=LOOKAHEAD_TARGET, type=int,
                        help='The target number of steps the agent aims to look forward.')
    parser.add_argument('--max_episode_steps', nargs='?', default=MAX_EPISODE_STEPS, type=int,
                        help='The maximum number of steps to play.')
    parser.add_argument('--video_basepath', nargs='?', default=VIDEO_BASEPATH,
                        help='The basepath where the videos will be stored.')
    parser.add_argument('--start_cp', nargs='?', default=START_CP, type=int,
                        help='The start value of C_p, the value that the agent changes to try to achieve the lookahead target. Decreasing this makes the search tree deeper, increasing this makes the search tree wider.')
    parser.add_argument('--seed', nargs='?', default=SEED, type=int,
                        help='The random seed.')

    args = parser.parse_args()

    logger.set_level(LOGGER_LEVEL)

    #env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    #SEED = 28
    #env.seed(SEED)

    agent = MCTSAgent(ITERATION_BUDGET, ENVIRONMENT)


    reward = 0
    done = False
    rewards = []

    for i in range(args.episodes):
        print("Episode number", i)
        ob = env.reset()
        env._max_episode_steps = args.max_episode_steps
        #video_path = os.path.join(
        #    args.video_basepath, f"output_{timestr}_{i}.mp4")
        #rec = VideoRecorder(env, path=video_path)

        sum_reward = 0
        node = None
        all_nodes = []
        C_p = args.start_cp
        
        while True:
            #print("################")
            #env.render()
            #rec.capture_frame()
            action, node, C_p = agent.act(env.state, n_actions=env.action_space.n, node=node, C_p=C_p, lookahead_target=args.lookahead_target)
            next_env = env.step(action)
            ob = next_env[0]
            reward = next_env[1]
            done = next_env[2]
            #ob, reward, done, _ = env.step(action)
            #print("### observed state: ", ob)
            sum_reward += reward    
            if done:
                break   
        
        print("### sum_reward: ", sum_reward)
        rewards.append(sum_reward)
           
    plt.plot(rewards, label='Reward')
    plt.plot(moving_average(rewards, 10), label='Average', linestyle='--')
    plt.legend()
    plt.show()

    env.close()
