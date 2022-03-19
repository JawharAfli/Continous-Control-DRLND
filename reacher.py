import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment
from config import (
    REACHER_ENVIRONMENT,
    DEVICE,
    MAX_T,
    N_EPISODES,
)

from ddpg_agent import DDPGAgent
from evaluate import evaluate
from model import Actor
from train import ddpg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    device = torch.device(DEVICE)

    env = UnityEnvironment(file_name=REACHER_ENVIRONMENT)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print("Number of agents:", num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print("Size of each action:", action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print(
        "There are {} agents. Each observes a state with length: {}".format(
            states.shape[0], state_size
        )
    )
    print("The state for the first agent looks like:", states[0])

    if args.train:

        scores = ddpg(
            env=env,
            brain_name=brain_name,
            state_size=state_size,
            action_size=action_size,
            random_seed=0,
            n_episodes=N_EPISODES,
            max_t=MAX_T,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.show()
