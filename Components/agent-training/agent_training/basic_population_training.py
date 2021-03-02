import PIL.Image
from IPython.display import display,clear_output,HTML
from IPython.display import Image as DisplayImage
import base64
import json
from io import StringIO
import ipywidgets as widgets
import sys
import time
import datetime
import imageio
import numpy as np
import io
import os
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy

from plark_game import classes
#from gym_plark.envs import plark_env,plark_env_guided_reward,plark_env_top_left
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from gym_plark.envs.plark_env import PlarkEnv
import datetime
# from tqdm import tqdm

from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

from tensorboardX import SummaryWriter


import helper 

# +
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

population_size = 10
steps = 250
iterations = 10000

log_dir_base = './self_play/'
os.makedirs(log_dir_base, exist_ok=True)
config_file_path = '/Components/plark-game/plark_game/game_config/10x10/balanced.json'

basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
basepath = '/data/agents/models'
exp_name = 'test_' + basicdate
policy_panther = 'MlpPolicy'
policy_pelican = 'MlpPolicy'
model_type = 'PPO2'
exp_path = os.path.join(basepath, exp_name)

# +
pelican_env = PlarkEnvSparse(driving_agent='pelican',
                     config_file_path=config_file_path,
                     image_based=False,
                     random_panther_start_position=True,
                     max_illegal_moves_per_turn=1)

panther_env = PlarkEnvSparse(driving_agent='panther',
                     config_file_path=config_file_path,
                     image_based=False,
                     random_panther_start_position=True,
                     max_illegal_moves_per_turn=1)                    
# -

panthers = [helper.make_new_model(model_type, policy_panther, panther_env) for i in range(population_size)]
pelicans = [helper.make_new_model(model_type, policy_pelican, pelican_env) for i in range(population_size)]

# for iteration in tqdm(range(iterations)):
for iteration in range(iterations):
    print("Iteration: " + str(iteration))
    for panther in panthers:
        for pelican in pelicans:
            panther_env.set_pelican(pelican)
            pelican_env.set_panther(panther)
            pelican.learn(steps)
            panther.learn(steps)


# Make video 
video_path =  os.path.join('./', 'test_self_play.mp4') 
basewidth,hsize = helper.make_video(pelicans[2],pelican_env,video_path)
video = io.open(video_path, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="test" width="'''+str(basewidth)+'''" height="'''+str(hsize)+'''" controls>
             <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))

start_time = time.time()
stats = []
for panther in panthers:
    for pelican in pelicans:
        #panther_env.set_pelican(pelican)
        pelican_env.set_panther(panther)
        mean, std = evaluate_policy(pelican, pelican_env, n_eval_episodes=10)
        stats.append({'panther':panther, 
                      'pelican':pelican,
                      'mean':mean,
                      'std':std})
print("--- %s seconds ---" % (time.time() - start_time))
print(stats)

import pandas as pd
df = pd.DataFrame(stats)[:100]


df['mean'].max()


