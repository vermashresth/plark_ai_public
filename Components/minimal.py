import gym
import gc
# !pip install stable-baselines3
from stable_baselines3 import PPO
import torch
import time
env = gym.make('CartPole-v1')


for i in range(10):
    print(i)
    model = PPO('MlpPolicy', env, verbose=1)
    print("model setup")
    model.learn(total_timesteps=1000)
    print("Memory allocated before: " + str(torch.cuda.memory_allocated()))
    print("GC Start")
    del model
    gc.collect()
    torch.cuda.empty_cache() 
    print("Memory allocated after: " + str(torch.cuda.memory_allocated()))
    print("GC End")
    print(torch.cuda.memory_allocated())
    time.sleep(10)


