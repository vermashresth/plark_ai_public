#!/usr/bin/env python
# coding: utf-8

# Copyright 2020 Montvieux Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
from io import BytesIO
import PIL.Image
from IPython.display import display,clear_output,HTML
from IPython.display import Image as DisplayImage
import base64
import json
from io import StringIO
import ipywidgets as widgets
import sys
import time
import imageio
import numpy as np
import io

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from plark_game import classes
from gym_plark.envs import plark_env, plark_env_guided_reward, plark_env_top_left, plark_env_sonobuoy_deployment, panther_env_reach_top
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse

from stable_baselines.bench import Monitor

from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

import helper 
import datetime
import os

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

#Reads all the game config file paths in /Components/plark-game/plark_game/game_config/
#Well not all, we don't recurse down into 10x10/nn
def read_game_config_paths():
    import glob

    config_file_directories = ["/Components/plark-game/plark_game/game_config/10x10/*.json", 
                               "/Components/plark-game/plark_game/game_config/20x20/*.json", 
                               "/Components/plark-game/plark_game/game_config/30x30/*.json"]
    config_file_paths = []

    for dr in config_file_directories:
        config_file_paths += glob.glob(dr)

    return config_file_paths

log_dir = './logs'
normalize = True
os.makedirs(log_dir, exist_ok=True)

n_eval_episodes = 10
training_steps = 1000

#For all configs, train on one config and then test on all the other configs
#training_configs = ['/Components/plark-game/plark_game/game_config/10x10/balanced.json']
#testing_configs = ['/Components/plark-game/plark_game/game_config/10x10/balanced_max_torps.json']
training_configs = read_game_config_paths()
testing_configs = read_game_config_paths()

for train_config in training_configs:

    print('Training on:', train_config)

    env = panther_env_reach_top.PantherEnvReachTop(config_file_path=train_config, \
                                                   image_based=False)
    env = Monitor(env, log_dir)
    if normalize:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200., gamma=0.95) 

    model = PPO2('MlpPolicy', env, seed=5000)

    #Train
    model.learn(training_steps)

    #Evaluate on all testing configs
    for test_config in testing_configs:

        print('Evaluating on:', test_config)
        
        #If test_config is the same as what was trained on, just skip
        if test_config == train_config:
            continue

        sparse_env = PlarkEnvSparse(config_file_path=test_config, driving_agent='panther', \
                                    image_based=False)

        sparse_env = Monitor(sparse_env, log_dir)

        if normalize:
            sparse_env = DummyVecEnv([lambda: sparse_env])
            sparse_env = VecNormalize(sparse_env, norm_obs=True, norm_reward=False, \
                                      clip_obs=200., gamma=0.95) 

        mean_reward, n_steps = evaluate_policy(model, sparse_env, \
                                               n_eval_episodes=n_eval_episodes, \
                                               deterministic=False, render=False, \
                                               callback=None, reward_threshold=None, \
                                               return_episode_rewards=False)
        print("Mean reward: ", mean_reward)

        #Save model 
        '''
        basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        basepath = '/data/agents/models'
        exp_name = 'test_' + basicdate
        exp_path = os.path.join(basepath, exp_name)

        env.get_attr('driving_agent')

        modeltype = 'PPO2'
        modelplayer = env.get_attr('driving_agent')[0] #env.driving_agent 
        render_height = env.get_attr('render_height')[0] #env.render_height
        render_width =  env.get_attr('render_width')[0] #env.render_width
        image_based = False
        helper.save_model(exp_path,model,modeltype,modelplayer,render_height,render_width,image_based,basicdate)
        '''

#Making the video

#video_path = '/test.mp4'
#basewidth,hsize = helper.make_video(model,env,video_path)

#video = io.open(video_path, 'r+b').read()
#encoded = base64.b64encode(video)
