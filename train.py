from collections import deque

import numpy as np
import torch

from ddpg_agent import DDPGAgent


def ddpg(
    env, brain_name, state_size, action_size, random_seed, n_episodes=2000, max_t=1000
):
    env_info = env.reset(train_mode=True)[brain_name]

    avg_score = []
    scores_deque = deque(maxlen=100)
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)

    env_info = env.reset(train_mode=True)[brain_name]

    states = env_info.vector_observations

    agents = [
        DDPGAgent(state_size, action_size, random_seed)
        for _ in range(len(env_info.agents))
    ]
    actions = [agent.act(states[i]) for i, agent in enumerate(agents)]

    for i_episode in range(1, n_episodes + 1):
        states = env_info.vector_observations
        for agent in agents:
            agent.reset()

        for t in range(max_t):
            actions = [agent.act(states[i]) for i, agent in enumerate(agents)]
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            step_t = zip(agents, states, actions, rewards, next_states, dones)

            for agent, state, actions, reward, next_step, done in step_t:
                agent.memory.add(state, actions, reward, next_step, done)
                if t % 100 == 0:
                    agent.step(state, actions, reward, next_step, done)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        score = np.mean(scores)
        avg_score.append(score)
        scores_deque.append(score)
        average = np.mean(scores_deque)

        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode,
                average,
            ),
            end="\n",
        )

        if np.mean(scores_deque) >= 30.0:
            print(
                f"Enviroment solved in episode={i_episode} avg_score={average:.2f}".format(
                    i_episode=i_episode, avg=average
                )
            )

            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")

            return avg_score

    return avg_score
