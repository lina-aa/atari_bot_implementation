import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper, EpisodicLifeEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

"""
Bot implementation with a custom wrapper
"""

class KingKongHeightWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.highest_y = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        ram = self.unwrapped.ale.getRAM()
        current_y = int(ram[0x21])

        if self.highest_y is None:
            self.highest_y = current_y
            self.steps_without_progress = 0
 
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

    def reset(self, **kwargs):
        print(f"GOT highest_y -> {self.highest_y}")
        self.highest_y = None
        print(f"RESET highest_y -> {self.highest_y}")
        return self.env.reset(**kwargs)
    
class DebugCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_count = 0

    def _on_step(self):
        if any(self.locals.get('dones', [])):
            self.episode_count += 1
            print(f"\n>>> EPIZOD SKOŃCZONY - Łącznie: {self.episode_count} | timesteps: {self.num_timesteps}")
        return True
    
def make_kingkong_env(rank, seed=0, render_mode=None):
    def _init():
        env = gym.make("ALE/KingKong-v5", render_mode=render_mode)
        env = AtariWrapper(env)    
        env = EpisodicLifeEnv(env)      
        env = KingKongHeightWrapper(env)  
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_model(save_model_path, tensorboard_path, training_timestamps, n_envs=4, seed=0):
    train_env = SubprocVecEnv([make_kingkong_env(i, seed=seed) for i in range(n_envs)])
    train_env = VecFrameStack(train_env, n_stack=4)

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

    model.learn(total_timesteps=training_timestamps) # callback=DebugCallback()
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
        if rewards > 0:
            print(f"REWARDED -> {rewards}")
        test_env.render()
        if dones[0]:
            obs = test_env.reset()

    test_env.close()

if __name__ == '__main__':
    # v1 is so stupid he just learned how to jump przez bomby, uczony bez wrapera
    # v2 is even dumber bo stoi se w kącie i na zbawienie czeka, ale był uczony na głupim wrapperze
    train_model("models/kingkong_ppo_v3.zip", "./ppo_kingkong_v3_logs/", 200000)

    model_path = "kingkong_ppo_v3.zip"
    test_model(model_path, 5000)
