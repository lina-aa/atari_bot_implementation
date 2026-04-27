import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
import torch
from atari_bot_implementation.kingkong_bot_v10 import KingKongHeightWrapper8

def make_kingkong_env(rank, seed=0, height_wrapper_cls=KingKongHeightWrapper8):
    def _init():
        env = gym.make("ALE/KingKong-v5")
        env = AtariWrapper(env, terminal_on_life_loss=True)
        env = height_wrapper_cls(env)  # ← Wrapper TUTAJ, zanim wektoryzujemy
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_model(train_env, save_model_path, tensorboard_path, training_timestamps):
    model = PPO(
        "CnnPolicy",          
        train_env, 
        verbose=1, 
        tensorboard_log=tensorboard_path,
        learning_rate=0.0001,
        n_steps=512,
        batch_size=64,
        ent_coef=0.01,
        gae_lambda=0.95,
        clip_range=0.2,
        gamma=0.99,
        device="cuda" #if torch.cuda.is_available() else "cpu",
    )
    model.learn(total_timesteps=training_timestamps)
    model.save(save_model_path)
    train_env.close()

def test_model(model_path, testing_timestamps, height_wrapper_cls=KingKongHeightWrapper8):
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
    save_model_path = 'models/kingkong_ppo_v9_improved.zip'
    tensorboard_path = './logs/ppo_kingkong_v9_logs/'
    training_timestamps = 2000000
    n_envs = 2

    # Prawidłowa kolejność: SubprocVecEnv → VecFrameStack
    train_env = SubprocVecEnv([make_kingkong_env(i) for i in range(n_envs)])
    train_env = VecFrameStack(train_env, n_stack=4)

    train_model(train_env, save_model_path, tensorboard_path, training_timestamps)
    # test_model(save_model_path, testing_timestamps=5000, height_wrapper_cls=KingKongHeightWrapper8)