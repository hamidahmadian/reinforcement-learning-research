import os

import numpy as np

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines import bench

from utils.gym_wrapper import WrapPyTorch
from utils.agent import agent
from utils.config import *


def run():
    env_id = "BreakoutNoFrameskip-v4"
    env = make_atari(env_id)
    env = bench.Monitor(env, os.path.join("log", env_id))
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)
    env = WrapPyTorch(env)
    model = agent(env=env)

    episode_reward = 0

    observation = env.reset()
    for frame_idx in range(1, MAX_FRAMES + 1):
        epsilon = epsilon_by_frame(frame_idx)

        action = model.get_action(observation, epsilon)

        prev_observation = observation
        observation, reward, done, _ = env.step(action)
        observation = None if done else observation

        model.update(prev_observation, action, reward, observation, frame_idx)
        episode_reward += reward

        if done:
            observation = env.reset()
            model.save_reward(episode_reward)
            episode_reward = 0

        if frame_idx % 10000 == 0:
            model.save_w()
            print('AVG. last 10 rewards is %s' % str(np.mean(model.rewards[-10:])))

    model.save_w()
    env.close()


if __name__ == "__main__":
    run()
