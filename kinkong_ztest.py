import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch
from kingkong_bot_v10 import KingKongHeightWrapper10
import time


def make_kingkong_env(rank, seed=0, height_wrapper_cls=KingKongHeightWrapper10):
    def _init():
        env = gym.make("ALE/KingKong-v5")
        env = AtariWrapper(env, terminal_on_life_loss=True)
        env = height_wrapper_cls(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_model(train_env, eval_env, save_model_path, tensorboard_path, training_timestamps):

    # 🔥 CALLBACKI
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path="./checkpoints/",
        name_prefix="kingkong"
    )

    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        tensorboard_log=tensorboard_path,
        learning_rate=1e-4,
        n_steps=1024,              
        batch_size=256,            
        ent_coef=0.02,             
        gae_lambda=0.95,
        clip_range=0.2,
        gamma=0.99,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model.learn(
        total_timesteps=training_timestamps,
        callback=[eval_callback, checkpoint_callback]
    )

    model.save(save_model_path)
    train_env.close()


if __name__ == '__main__':
    save_model_path = 'models/kingkong_ppo_v10_new.zip'
    tensorboard_path = './logs/ppo_kingkong_v10_new/'
    
    training_timestamps = 10_000_000  
    n_envs = 4                         

    start = time.time()

    train_env = SubprocVecEnv([make_kingkong_env(i) for i in range(n_envs)])
    train_env = VecFrameStack(train_env, n_stack=4)

    eval_env = SubprocVecEnv([make_kingkong_env(999)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    train_model(train_env, eval_env, save_model_path, tensorboard_path, training_timestamps)

    stop = time.time()
    print(f"Training completed in {stop - start:.2f} seconds.")