# +
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
from copy import deepcopy

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from plark_game import classes
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from gym_plark.envs.plark_env import PlarkEnv
import datetime
# -

from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from tensorboardX import SummaryWriter


import helper

# To solve linear programs
# !pip install lemkelcp
import lemkelcp as lcp

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

log_dir_base = './pnm_logs/'
os.makedirs(log_dir_base, exist_ok=True)


def compute_payoff_matrix(pelican,
                          panther,
                          payoffs,
                          pelicans,
                          panthers,
                          trials = 1000,
                          image_based=False,
                          num_parallel_envs=1,
                          config_file_path = '/Components/plark-game/plark_game/game_config/10x10/balanced.json'):

    payoffs = np.pad(payoffs, 
                     [(0, len(pelicans) - payoffs.shape[0]), 
                      (0, len(panthers) - payoffs.shape[1])], 
                     mode='constant')

    # Adding payoff for the last row player        
    for i, opponent in enumerate(panthers):
        print(i)
        print(opponent)
        env = SubprocVecEnv([lambda:PlarkEnvSparse(driving_agent='pelican',
                                                   panther_agent_filepath=opponent,
                                                   config_file_path=config_file_path,
                                                   image_based=image_based,
                                                   random_panther_start_position=True,
                                                   max_illegal_moves_per_turn=3) for _ in range(num_parallel_envs)])            
        
        victory_count, avg_reward = helper.check_victory(pelican, env, trials = trials)
        payoffs[-1, i] = avg_reward

            
    for i, opponent in enumerate(pelicans):
        print(i)
        print(opponent)        
        env = SubprocVecEnv([lambda:PlarkEnvSparse(driving_agent='panther',
                                                   panther_agent_filepath=opponent,
                                                   config_file_path=config_file_path,
                                                   image_based=image_based,
                                                   random_panther_start_position=True,
                                                   max_illegal_moves_per_turn=3) for _ in range(num_parallel_envs)])            

        
        victory_count, avg_reward = helper.check_victory(panther, env, trials = trials)
        
        # Given that we are storing everything in one table, and the value below is now computed 
        # from the perspective of the panther, I assume we need this value to be negative?
        payoffs[i, -1] = -avg_reward        
    return payoffs


def train_agent_against_mixture(driving_agent,
                                keep_instances,
                                policy,
                                exp_path,
                                model,
                                tests,
                                mixture,
                                testing_interval,
                                max_steps,
                                model_type,
                                basicdate,
                                tb_writer,
                                tb_log_name,
                                early_stopping = True,
                                previous_steps = 0,
                                config_file_path = '/Components/plark-game/plark_game/game_config/10x10/balanced.json',
                                image_based=False,
                                num_parallel_envs=1):       
        
    opponents = np.random.choice(tests, size=num_parallel_envs, p=mixture)    
                
    if driving_agent == 'pelican':            
        env = SubprocVecEnv([lambda:PlarkEnvSparse(driving_agent=driving_agent,
                                                           panther_agent_filepath=opponent_model,
                                                           config_file_path=config_file_path,
                                                           image_based=image_based,
                                                           random_panther_start_position=True,
                                                           max_illegal_moves_per_turn=3) for opponent_model in opponents])            
    else:
        env = SubprocVecEnv([lambda:PlarkEnvSparse(driving_agent=driving_agent,
                                                           pelican_agent_filepath=opponent_model,
                                                           config_file_path=config_file_path,
                                                           image_based=image_based,
                                                           random_panther_start_position=True,
                                                           max_illegal_moves_per_turn=3) for opponent_model in opponents])            
        
    agent_filepath, new_steps = train_agent(exp_path,
                                             model,
                                             env,
                                             testing_interval,
                                             testing_interval,
                                             model_type,
                                             basicdate,
                                             tb_writer,
                                             tb_log_name,
                                             early_stopping = True,
                                             previous_steps = previous_steps,
                                             save_model = True)
    
    return agent_filepath, new_steps

def train_agent(exp_path,
                model,
                env,
                testing_interval,
                max_steps,
                model_type,
                basicdate,
                tb_writer,
                tb_log_name,
                early_stopping = True,
                previous_steps = 0,
                save_model = True):
    steps = 0
    logger.info("Beginning training for {} steps".format(max_steps))
    model.set_env(env)

    while steps < max_steps:
        logger.info("Training for {} steps".format(testing_interval))
        model.learn(testing_interval)
        steps = steps + testing_interval
        if early_stopping:
            victory_count, avg_reward = helper.check_victory(model, env, trials = 10)
            if tb_writer is not None and tb_log_name is not None:
                tb_steps = steps + previous_steps
                logger.info("Writing to tensorboard for {} after {} steps".format(tb_log_name, tb_steps))
                tb_writer.add_scalar('{}_avg_reward'.format(tb_log_name), avg_reward, tb_steps)
                tb_writer.add_scalar('{}_victory_count'.format(tb_log_name), victory_count, tb_steps)
            if victory_count > 7:
                logger.info("Stopping training early")
                break # Stopping training as winning
    # Save agent
    logger.info('steps = '+ str(steps))
    agent_filepath = ''
    if save_model:
        basicdate = basicdate + '_steps_' + str(previous_steps + steps)
        agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, model, model_type, env, basicdate)
        agent_filepath = os.path.dirname(agent_filepath)
    return agent_filepath, steps

def run_deep_pnm(exp_name,
                 exp_path,
                 basicdate,
                 pelican_testing_interval = 100,
                 pelican_max_learning_steps = 10000,
                 panther_testing_interval = 100,
                 panther_max_learning_steps = 10000,
                 deep_pnm_iterations = 10000,
                 model_type = 'PPO2',
                 log_to_tb = False,
                 image_based = True,
                 num_parallel_envs = 1,
                 keep_instances = False):

    # Initialize the supports of opponents
    pelicans = []
    panthers = []    
    
    pelican_training_steps = 0
    panther_training_steps = 0
    pelican_model_type = model_type
    panther_model_type = model_type

    pelican_tmp_exp_path = os.path.join(exp_path, 'pelican')
    os.makedirs(pelican_tmp_exp_path, exist_ok = True)
    panther_tmp_exp_path = os.path.join(exp_path, 'panther')
    os.makedirs(panther_tmp_exp_path, exist_ok = True)
        
    if log_to_tb:
        writer = SummaryWriter(exp_path)
        pelican_tb_log_name = 'pelican'
        panther_tb_log_name = 'panther'
    else:
        writer = None
        pelican_tb_log_name = None
        panther_tb_log_name = None

    policy = 'CnnPolicy'
    if image_based is False:
        policy = 'MlpPolicy'

    parallel = False
    # Commented out for debugging
    if model_type.lower() == 'ppo2':
        parallel = True

    log_dir_base = '/deep_pnm/'
    os.makedirs(log_dir_base, exist_ok = True)
    config_file_path = '/Components/plark-game/plark_game/game_config/10x10/balanced.json'

    # Train initial pelican vs default panther
    if parallel:
        pelican_env = SubprocVecEnv([lambda:PlarkEnv(driving_agent = 'pelican',
                                                            config_file_path = config_file_path,
                                                            image_based = image_based,
                                                            random_panther_start_position = True,
                                                            max_illegal_moves_per_turn = 3) for _ in range(num_parallel_envs)])
        pelican_test_env = SubprocVecEnv([lambda:PlarkEnvSparse(driving_agent = 'pelican',
                                                            config_file_path = config_file_path,
                                                            image_based = image_based,
                                                            random_panther_start_position = True,
                                                            max_illegal_moves_per_turn = 3) for _ in range(num_parallel_envs)])
    else:
        pelican_env = PlarkEnv(driving_agent ='pelican',
                                      config_file_path = config_file_path,
                                      image_based = image_based,
                                      random_panther_start_position = True,
                                      max_illegal_moves_per_turn = 3)
        pelican_test_env = PlarkEnvSparse(driving_agent ='pelican',
                                      config_file_path = config_file_path,
                                      image_based = image_based,
                                      random_panther_start_position = True,
                                      max_illegal_moves_per_turn = 3)

    pelican = helper.make_new_model(model_type, policy, pelican_env)
    logger.info('Training initial pelican')
        
    pelican_agent_filepath, steps = train_agent(pelican_tmp_exp_path,
                                                pelican,
                                                pelican_env,
                                                pelican_testing_interval,
                                                pelican_max_learning_steps,
                                                pelican_model_type,
                                                basicdate,
                                                writer,
                                                pelican_tb_log_name,
                                                early_stopping = True,
                                                previous_steps = 0,
                                                save_model = keep_instances)
    pelican_training_steps = pelican_training_steps + steps

    # Train initial panther agent vs default pelican
    if parallel:
        panther_env = SubprocVecEnv([lambda:PlarkEnv(driving_agent = 'panther',
                                                            config_file_path = config_file_path,
                                                            image_based = image_based,
                                                            random_panther_start_position = True,
                                                            max_illegal_moves_per_turn = 3) for _ in range(num_parallel_envs)])
    else:
        panther_env = PlarkEnv(driving_agent = 'panther',
                                      config_file_path = config_file_path,
                                      image_based = image_based,
                                      random_panther_start_position = True,
                                      max_illegal_moves_per_turn = 3)

    panther = helper.make_new_model(model_type, policy, panther_env)
    logger.info('Training initial panther')
    panther_agent_filepath, steps = train_agent(panther_tmp_exp_path,
                                                panther,
                                                panther_env,
                                                panther_testing_interval,
                                                panther_max_learning_steps,
                                                panther_model_type,
                                                basicdate,
                                                writer,
                                                panther_tb_log_name,
                                                early_stopping = True,
                                                previous_steps = 0,
                                                save_model = keep_instances)
    panther_training_steps = panther_training_steps + steps

    pelicans.append(pelican_agent_filepath)
    panthers.append(panther_agent_filepath)

    mixture_pelicans = np.array([1.])
    mixture_panthers = np.array([1.])
    payoffs = np.zeros((1,1))

    # Train agent vs agent
    logger.info('Deep Parallel Nash Memory')

    for i in range(deep_pnm_iterations):
        logger.info('Deep PNM iteration ' + str(i) + ' of ' + str(deep_pnm_iterations))
        logger.info('Training pelican')
        pelican = helper.make_new_model(model_type, policy, pelican_env)
        pelican_agent_filepath, steps = train_agent_against_mixture('pelican',
                                                                    keep_instances,
                                                                    policy,
                                                                    exp_path,
                                                                    pelican,
                                                                    panthers,
                                                                    mixture_pelicans,
                                                                    pelican_testing_interval,
                                                                    pelican_max_learning_steps,
                                                                    pelican_model_type,
                                                                    basicdate,
                                                                    writer,
                                                                    pelican_tb_log_name,
                                                                    previous_steps = pelican_training_steps,
                                                                    num_parallel_envs=num_parallel_envs)
        
        
        pelican_training_steps = pelican_training_steps + steps

        logger.info('Training panther')
        panther = helper.make_new_model(model_type, policy, panther_env)
        panther_agent_filepath, steps = train_agent_against_mixture('panther',
                                                                    keep_instances,
                                                                    policy,
                                                                    exp_path,
                                                                    panther,
                                                                    pelicans,
                                                                    mixture_panthers,
                                                                    panther_testing_interval,
                                                                    panther_max_learning_steps,
                                                                    panther_model_type,
                                                                    basicdate,
                                                                    writer,
                                                                    panther_tb_log_name,
                                                                    previous_steps = panther_training_steps,
                                                                    num_parallel_envs=num_parallel_envs)
        panther_training_steps = panther_training_steps + steps

        pelicans.append(pelican_agent_filepath)
        panthers.append(panther_agent_filepath)


        # Computing the payoff matrices and solving the corresponding LPs
        # Only compute for pelican in the sparse env, that of panther is the negative traspose (game is zero-sum)
        
        payoffs = compute_payoff_matrix(pelican,
                                        panther,
                                        payoffs,
                                        pelicans,
                                        panthers,
                                        trials = 1000,
                                        image_based=False,
                                        num_parallel_envs=num_parallel_envs)      
        
        
        logger.info(payoffs)
        np.save('%s/itartion%d.npy' % (log_dir_base, i), payoffs)
        mixture_pelicans, exit_code1, exit_string1 = lcp.lemkelcp.lemkelcp(payoffs, np.zeros((len(pelicans),)))
        logger.info(mixture_pelicans)
        np.save('%s/mixture1_%d.npy' % (log_dir_base, i), mixture_pelicans)        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Let's double check that we really need the transpose !!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        mixture_panthers, exit_code2, exit_string2 = lcp.lemkelcp.lemkelcp(-payoffs.transpose(), np.zeros((len(panthers),)))
        logger.info(mixture_panthers)
        np.save('%s/mixture2_%d.npy' % (log_dir_base, i), mixture_panthers)                
        if exit_code1 != 0 or exit_code2 != 0:
            print('Cannot solve the LPs...')
            print('Exit code 1 %s' % exit_code1)
            print('Exit code 2 %s' % exit_code2)            
            break

    # Saving final version of the agents
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, pelican_model, pelican_model_type, pelican_env, basicdate)
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, panther_model, panther_model_type, panther_env, basicdate)

    logger.info('Training pelican total steps: ' + str(pelican_training_steps))
    logger.info('Training panther total steps: ' + str(panther_training_steps))
    # Make video
    video_path =  os.path.join(exp_path, 'test_deep_pnm.mp4')
    basewidth,hsize = helper.make_video(pelican_model, pelican_env, video_path)
    return video_path, basewidth, hsize


# +
def main():
    basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    basepath = '/data/agents/models'
    exp_name = 'test_' + basicdate
    exp_path = os.path.join(basepath, exp_name)

    logger.info(exp_path)

    # run_deep_pnm(exp_name,exp_path,basicdate)
    run_deep_pnm(exp_name,
                 exp_path,
                 basicdate,
                  pelican_testing_interval = 250,
                  pelican_max_learning_steps = 250,
                  panther_testing_interval = 250,
                  panther_max_learning_steps = 250,
                  deep_pnm_iterations = 200,
                  model_type = 'PPO2',
                  log_to_tb = True,
                  image_based = False,
                  num_parallel_envs = 5,
                  keep_instances = True)    

if __name__ == '__main__':
    main()
# -




