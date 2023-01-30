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
        n_timesteps=3e6,
        policy='MultiInputPolicy',
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=False
        ),
        gamma=0.995,
        learning_rate=0.0067,
        batch_size=16,
        buffer_size=int(1e4),
        tau=0.08,
        train_freq=4,
        policy_kwargs="dict(net_arch=[256, 256])",
        action_noise = NormalActionNoise(mean=0, sigma=0.28),
    ),
    'CTR-Reach-v0': dict(
        env_wrapper=[{"gym.wrappers.TimeLimit": {"max_episode_steps": 200}}],
        callback={"rl_zoo3.callbacks.CTRReachCallback": {}},
        n_envs=1,
        n_timesteps=3e6,
        n_steps=128,
        policy='MlpPolicy',
        gamma=0.999,
        learning_rate=0.03,
        batch_size=128,
        ent_coef=4.85e-7,
        clip_range=0.2,
        n_epochs=5,
        gae_lambda=0.9,
        max_grad_norm=0.8,
        vf_coef=0.514,
        policy_kwargs="dict(net_arch=dict(pi=[64, 64], vf=[64,64]), activation_fn=nn.Tanh, log_std_init=-3.34)",
        use_sde=True,
        sde_sample_freq=16
    )}
