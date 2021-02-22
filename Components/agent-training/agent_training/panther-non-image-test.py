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

# In[2]:


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
from gym_plark.envs import plark_env,                           plark_env_guided_reward,                           plark_env_top_left,                           plark_env_sonobuoy_deployment,                           panther_env_reach_top

from stable_baselines.bench import Monitor

from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

import helper 
import datetime
import os

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[3]:


display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))


# In[4]:


log_dir = './logs'
normalize = True
os.makedirs(log_dir, exist_ok=True)
env = panther_env_reach_top.PantherEnvReachTop(config_file_path='/Components/plark-game/plark_game/game_config/10x10/balanced.json',image_based=False)
env = Monitor(env, log_dir)
if normalize:
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200., gamma=0.95) 


# In[5]:


# Test that the observations are indeed normalized
# observation, reward, done, info = env.step([0])
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=3)
# observation


# ## Training loop

# In[6]:


n_eval_episodes = 10
#n_eval_episodes = 1000
#training_steps = 10000000 
#training_steps = 100000
training_steps = 10


# In[7]:


model = PPO2('MlpPolicy', env, seed=5000)


# In[8]:


model.learn(training_steps)


# In[14]:


print("****** STARTING EVALUATION *******")

#sparse_env = env
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
sparse_env = PlarkEnvSparse(config_file_path='/Components/plark-game/plark_game/game_config/10x10/balanced.json', 
                            driving_agent='panther',
                            image_based=False)

sparse_env = Monitor(sparse_env, log_dir)

if normalize:
    sparse_env = DummyVecEnv([lambda: sparse_env])
    sparse_env = VecNormalize(sparse_env, norm_obs=True, norm_reward=False, clip_obs=200., gamma=0.95) 

#for nee in [1000]:
#for nee in [10,20,30,40,50,100,250,500]: # 0.892 for 1000
    mean_reward, n_steps = evaluate_policy(model, sparse_env, n_eval_episodes=n_eval_episodes, deterministic=False, render=False, callback=None, reward_threshold=None, return_episode_rewards=False)
    print("%s episodes,Mean Reward is %.3f,Number of steps is %d" % (n_eval_episodes,mean_reward,n_steps))

print("****** EVALUATION FINISHED *******")

# In[10]:


#Save model 
basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

basepath = '/data/agents/models'
exp_name = 'test_' + basicdate
exp_path = os.path.join(basepath, exp_name)

print(exp_path)


# In[11]:


env.get_attr('driving_agent')


# In[12]:


modeltype = 'PPO2'
modelplayer = env.get_attr('driving_agent')[0] #env.driving_agent 
render_height = env.get_attr('render_height')[0] #env.render_height
render_width =  env.get_attr('render_width')[0] #env.render_width
image_based = False
helper.save_model(exp_path,model,modeltype,modelplayer,render_height,render_width,image_based,basicdate)


# # making the video

# In[13]:

video_path = '/test.mp4'
basewidth,hsize = helper.make_video(model,env,video_path)

video = io.open(video_path, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="test" width="'''+str(basewidth)+'''" height="'''+str(hsize)+'''" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))


# In[ ]:




