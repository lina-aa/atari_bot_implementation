import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper, EpisodicLifeEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os

"""
Bot implementations with a custom wrappers (bot v2 and v3)
"""

class KingKongHeightWrapperBase(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.highest_y = None
        self.steps_without_progress = 0

    def _get_current_y(self):
        ram = self.unwrapped.ale.getRAM()
        return int(ram[0x21])

    def _initialize_tracking(self, current_y):
        if self.highest_y is None:
            self.highest_y = current_y
            self.steps_without_progress = 0

    def step(self, action):
        raise NotImplementedError("Use KingKongHeightWrapper1 or KingKongHeightWrapper2.")

    def reset(self, **kwargs):
        self.highest_y = None
        self.steps_without_progress = 0
        return self.env.reset(**kwargs)


# bot v2 
class KingKongHeightWrapper(KingKongHeightWrapperBase):

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_y = self._get_current_y()
        self._initialize_tracking(current_y)
 
        if current_y < self.highest_y and current_y > 0:
            bonus = (self.highest_y - current_y) * 0.1
            reward += bonus

            self.highest_y = current_y
            
        return obs, reward, terminated, truncated, info

# bot v3 
class KingKongHeightWrapper2(KingKongHeightWrapperBase):

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_y = self._get_current_y()
        self._initialize_tracking(current_y)
 
        if current_y < self.highest_y and current_y > 0:
            bonus = (self.highest_y - current_y) * 0.1
            reward += bonus

            self.highest_y = current_y
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

            if self.steps_without_progress > 500:
                reward -= 0.05
            
        return obs, reward, terminated, truncated, info

def make_kingkong_env(rank, seed=0, render_mode=None):
    def _init():
        env = gym.make("ALE/KingKong-v5", render_mode=render_mode)
        env = AtariWrapper(env)    
        env = EpisodicLifeEnv(env)      
        env = KingKongHeightWrapper1(env)  
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_model(save_model_path, tensorboard_path, training_timestamps, n_envs=4, seed=0, additional_training=False):
    train_env = SubprocVecEnv([make_kingkong_env(i, seed=seed) for i in range(n_envs)])
    train_env = VecFrameStack(train_env, n_stack=4)

    if additional_training:
        if not os.path.exists(save_model_path):
            print(f"Model path {save_model_path} does not exist. Please provide a valid path for additional training.")
            return

        model = PPO.load(model_path, env=train_env)

        model.learn(
            total_timesteps=training_timestamps,
            reset_num_timesteps=False, 
            tb_log_name="PPO_1",     
        )

    else:
        model = PPO(
            "CnnPolicy",          
            train_env, 
            verbose=1, 
            tensorboard_log=tensorboard_path,
            learning_rate=0.00025, 
            n_steps=128,          
            batch_size=256,
            ent_coef=0.01          
        )

        model.learn(total_timesteps=training_timestamps)
        
    model.save(save_model_path)
    train_env.close()


def test_model(model_path, testing_timestamps):
    test_env = DummyVecEnv([make_kingkong_env(0, seed=42, render_mode="human")])
    test_env = VecFrameStack(test_env, n_stack=4)

    model = PPO.load(model_path, env=test_env)
    obs = test_env.reset()

    for _ in range(testing_timestamps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        # if rewards > 0:
        #     print(f"REWARDED -> {rewards}") # basic bomb gives 1.
        test_env.render()
        if dones[0]:
            obs = test_env.reset()

    test_env.close()

if __name__ == '__main__':
    model_path = "models/kingkong_ppo_v3add.zip"
    log_path = "./logs/ppo_kingkong_v3_logs/"

    # train_model(model_path, log_path,  2000000, additional_training=True)
    test_model(model_path, 5000)