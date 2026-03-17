import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

"""
Basic bot implementation using PPO (Bot v1)
"""

def train_model(train_env, save_model_path, tensorboard_path, training_timestamps):

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


def test_model(test_env, model_path, testing_timestamps):

    model = PPO.load(model_path, env=test_env)
    obs = test_env.reset()

    for _ in range(testing_timestamps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        test_env.render()
        if dones:
            obs = test_env.reset()

    test_env.close()

# environment
env_id = "ALE/KingKong-v5"
save_model_path = 'models/kingkong_ppo_v1.zip'
tensorboard_path = './logs/ppo_kingkong_img_logs/'
training_timestamps = 200000

train_env = make_atari_env(env_id, n_envs=4, seed=0)
train_env = VecFrameStack(train_env, n_stack=4)

# loading model
model_path = "models/kingkong_ppo_v1.zip"
testing_timestamps = 5000

test_env = make_atari_env(env_id, n_envs=1, env_kwargs={"render_mode": "human"})
test_env = VecFrameStack(test_env, n_stack=4)

if __name__ == '__main__':
    #bot v1
    train_model(train_env, save_model_path, tensorboard_path, training_timestamps)

    test_model(test_env, model_path, testing_timestamps)


