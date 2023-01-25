"""This file just serves as an example on how to configure the zoo
using python scripts instead of yaml files."""
import torch
import numpy as np

from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
# Algorithm set in command, env_kwargs are in command
hyperparams = {
    'CTR-Reach-Goal-v0': dict(
        env_wrapper=[{"gym.wrappers.TimeLimit": {"max_episode_steps": 200}}],
        callback={"rl_zoo3.callbacks.CTRReachCallback": {}},
        n_timesteps=1e6,
        policy='MultiInputPolicy',
        buffer_size=int(1e6),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=False
        ),
        gamma=0.95,
        learning_rate=0.001,
        batch_size=1024,
        policy_kwargs="dict(net_arch=[256, 256, 256])",
        action_noise = NormalActionNoise(mean=0, sigma=0.25)),
    'CTR-Reach-v0': dict(
        env_wrapper=[{"gym.wrappers.TimeLimit": {"max_episode_steps": 150}}],
        callback={"rl_zoo3.callbacks.CTRReachCallback": {}},
        n_envs=1,
        n_timesteps=3e6,
        policy='MlpPolicy',
        gamma=0.95,
        learning_rate=0.0005,
        batch_size=256,
        policy_kwargs="dict(net_arch=[128, 128, 128])",
        #action_noise=NormalActionNoise(mean=np.zeros(6), sigma=np.array([0.0018, 0.0018, 0.0018, 0.025, 0.025, 0.025]))
    )}
