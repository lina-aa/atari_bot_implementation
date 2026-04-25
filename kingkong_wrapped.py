import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from kingkong_bot_v2 import KingKongHeightWrapper1
from kingkong_bot_v3 import KingKongHeightWrapper2
from kingkong_bot_v4 import KingKongHeightWrapper3
from kingkong_bot_v5 import KingKongHeightWrapper4
from kingkong_bot_v6 import KingKongHeightWrapper5
from kingkong_bot_v7 import KingKongHeightWrapper6
from kingkong_bot_v8 import KingKongHeightWrapper7
from kingkong_wrapper_base import (
    LADDER_ON_VALUE,
    LADDER_RAM_ADDR,
    KingKongHeightWrapperBase,
    PLAYER_Y_ADDR,
)

"""PPO training and test entrypoint for ALE King Kong.

Reward-shaping coaches live in kingkong_bot_v*.py; this module re-exports them and helpers
in one place.
"""

NUMBER_OF_ENVS = os.cpu_count() - 4

__all__ = [
    "LADDER_ON_VALUE",
    "LADDER_RAM_ADDR",
    "PLAYER_Y_ADDR",
    "KingKongHeightWrapperBase",
    "KingKongHeightWrapper1",
    "KingKongHeightWrapper2",
    "KingKongHeightWrapper3",
    "KingKongHeightWrapper4",
    "KingKongHeightWrapper5",
    "KingKongHeightWrapper6",
    "KingKongHeightWrapper7",
    "NUMBER_OF_ENVS",
    "make_kingkong_env",
    "train_model",
    "test_model",
]


def make_kingkong_env(
    rank,
    seed=0,
    render_mode=None,
    height_wrapper_cls=KingKongHeightWrapper1,
    terminal_on_life_loss=True,
):
    def _init():
        env = gym.make("ALE/KingKong-v5", render_mode=render_mode)
        env = AtariWrapper(env, terminal_on_life_loss=terminal_on_life_loss)
        env = height_wrapper_cls(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_model(
    save_model_path,
    tensorboard_path,
    training_timestamps,
    n_envs=NUMBER_OF_ENVS,
    seed=0,
    additional_training=False,
    height_wrapper_cls=KingKongHeightWrapper1,
    terminal_on_life_loss=True,
):
    train_env = SubprocVecEnv(
        [
            make_kingkong_env(
                i,
                seed=seed,
                height_wrapper_cls=height_wrapper_cls,
                terminal_on_life_loss=terminal_on_life_loss,
            )
            for i in range(n_envs)
        ]
    )
    train_env = VecFrameStack(train_env, n_stack=4)

    if additional_training:
        if not os.path.exists(save_model_path):
            print(f"Model path {save_model_path} does not exist. Please provide a valid path for additional training.")
            return

        model = PPO.load(save_model_path, env=train_env)

        model.learn(
            total_timesteps=training_timestamps,
            reset_num_timesteps=False,
            tb_log_name="PPO_2",
        )

    else:
        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=0,
            tensorboard_log=tensorboard_path,
            learning_rate=0.00025,
            n_steps=256,
            batch_size=256,
            ent_coef=0.10, # MORE ENTROPY FOR EXPLORATION !!
        )

        model.learn(total_timesteps=training_timestamps, tb_log_name="PPO_1")

    model.save(save_model_path)
    train_env.close()


def test_model(model_path, testing_timestamps, height_wrapper_cls=KingKongHeightWrapper1):
    test_env = DummyVecEnv(
        [
            make_kingkong_env(
                0,
                seed=42,
                render_mode="human",
                height_wrapper_cls=height_wrapper_cls,
                terminal_on_life_loss=False,
            )
        ]
    )
    test_env = VecFrameStack(test_env, n_stack=4)

    model = PPO.load(model_path, env=test_env)
    obs = test_env.reset()

    for _ in range(testing_timestamps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        test_env.render()
        if dones[0]:
            obs = test_env.reset()

    test_env.close()


if __name__ == '__main__':
    model_path = "models/kingkong_ppo_v8.zip"
    log_path = "./logs/ppo_kingkong_v8_logs/"

    train_model(
        model_path,
        log_path,
        2_000_000,
        additional_training=False,
        height_wrapper_cls=KingKongHeightWrapper7,
        terminal_on_life_loss=True,
    )
    # train_model(
    #     model_path,
    #     log_path,
    #     3_000_000,
    #     additional_training=True,
    #     height_wrapper_cls=KingKongHeightWrapper7,
    #     terminal_on_life_loss=False,
    # )
    test_model(model_path, 5000, height_wrapper_cls=KingKongHeightWrapper7)

# v4: jumps well, ignores ladder, lots of in-place jumping
# v5: less random than v4, still weak on ladders
# v6: KingKongHeightWrapper5 — dense delta-Y + milestones + level-up; 2-stage curriculum
# v7: KingKongHeightWrapper6 — simple delta-Y + 64→0 exit; v7.1 is coefficient tuning
# v8: KingKongHeightWrapper7 — stall, ladder loiter, jump-in-place penalty, exit >=, ent 0.10
