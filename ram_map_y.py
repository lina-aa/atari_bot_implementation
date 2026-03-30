import gymnasium as gym
import ale_py
import time
from pynput import keyboard

'''
This script allows to manually control the King Kong game using the arrow keys and spacebar
'''

current_action = 0

def on_press(key):
    global current_action
    try:
        if key == keyboard.Key.up: current_action = 2
        elif key == keyboard.Key.down: current_action = 5
        elif key == keyboard.Key.left: current_action = 4
        elif key == keyboard.Key.right: current_action = 3
        elif key == keyboard.Key.space: current_action = 1
    except: pass

def on_release(key):
    global current_action
    current_action = 0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

env = gym.make("ALE/KingKong-v5", render_mode="human", mode=2)
obs, _ = env.reset()

print(f"{'akcja':>6} | {'(100)':>10} | {'(33)':>10}")

while True:
    obs, reward, terminated, truncated, info = env.step(current_action)
    ram = env.unwrapped.ale.getRAM()

    print(f"{current_action:>6} | {ram[100]:>10} | {ram[33]:>10}", end="\r")

    time.sleep(0.05)
    if terminated or truncated:
        obs, _ = env.reset()