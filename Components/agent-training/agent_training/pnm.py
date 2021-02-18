# +
# !pip install pycddlib
import datetime
import numpy as np
import os
import copy
import glob
from stable_baselines.common.vec_env import SubprocVecEnv
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from gym_plark.envs.plark_env import PlarkEnv
from tensorboardX import SummaryWriter
import helper
import lp_solve
import tensorflow as tf
from stable_baselines import DQN, PPO2, A2C, ACKTR

tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -

def compute_payoff_matrix(pelican,
                          panther,
                          payoffs,
                          pelicans,
                          panthers,
                          config_file_path,
                          trials = 1000,
                          parallel = False,
                          image_based = False,
                          num_parallel_envs = 1):
    # Resizing the payoff matrix for new strategies
    payoffs = np.pad(payoffs, 
                     [(0, len(pelicans) - payoffs.shape[0]), 
                      (0, len(panthers) - payoffs.shape[1])], 
                     mode = 'constant')

    # Adding payoff for the last row strategy       
    for i, opponent in enumerate(panthers):        
        env = helper.get_envs('pelican', config_file_path,
                              opponents = [opponent],
                              num_envs = num_parallel_envs,
                              image_based = image_based,
                              random_panther_start_position = True,
                              max_illegal_moves_per_turn = 3,
                              sparse = True,
                              vecenv = parallel)                
        victory_count, avg_reward = helper.check_victory(pelican, env, trials = trials // num_parallel_envs)
        payoffs[-1, i] = avg_reward

    # Adding payoff for the last column strategy
    for i, opponent in enumerate(pelicans):
        env = helper.get_envs('panther', config_file_path,
                              opponents = [opponent],
                              num_envs = num_parallel_envs,
                              image_based = image_based,
                              random_panther_start_position = True,
                              max_illegal_moves_per_turn = 3,
                              sparse = True,
                              vecenv = parallel) 
        victory_count, avg_reward = helper.check_victory(panther, env, trials = trials // num_parallel_envs)
        # Given that we are storing everything in one table, and the value below is now computed 
        # from the perspective of the panther, I assume we need this value to be negative?
        payoffs[i, -1] = -avg_reward        
    return payoffs

def train_agent_against_mixture(driving_agent,
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
                                config_file_path,                                
                                early_stopping = True,
                                previous_steps = 0,
                                parallel = False,
                                image_based = False,
                                num_parallel_envs = 1):       
    # If we use parallel envs, we run all the training against different sampled opponents in parallel
    if parallel:
        env = helper.get_envs(driving_agent,
                              config_file_path,
                              opponents = tests,
                              num_envs = num_parallel_envs,
                              image_based = image_based,
                              random_panther_start_position = True,
                              max_illegal_moves_per_turn = 3,
                              sparse = False,
                              vecenv = parallel,
                              mixture = mixture) 
        agent_filepath, new_steps = train_agent(exp_path,
                                                model,
                                                env,
                                                max_steps,
                                                testing_interval,
                                                model_type,
                                                basicdate,
                                                tb_writer,
                                                tb_log_name,
                                                early_stopping = True,
                                                previous_steps = previous_steps)
    # Otherwise we sample different opponents and we train against each of them separately
    else:
        opponents = np.random.choice(tests, size = max_steps // testing_interval, p = mixture)
        for opponent in opponents:
            env = helper.get_envs(driving_agent,
                                  config_file_path,
                                  opponents = [opponent],
                                  num_envs = num_parallel_envs,
                                  image_based = image_based,
                                  random_panther_start_position = True,
                                  max_illegal_moves_per_turn = 3,
                                  sparse = False,
                                  vecenv = parallel) 
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
                                                    previous_steps = previous_steps)    
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
                previous_steps = 0):
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
    basicdate = basicdate + '_steps_' + str(previous_steps + steps)
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, model, model_type, env, basicdate)
    agent_filepath = os.path.dirname(agent_filepath)
    return agent_filepath, steps

def run_pnm(exp_path,
            basicdate,
            pelican_testing_interval = 100,
            pelican_max_learning_steps = 10000,
            panther_testing_interval = 100,
            panther_max_learning_steps = 10000,
            max_pnm_iterations = 10000,
            stopping_eps = 0.001,
            retraining_prob = 0.,
            model_type = 'PPO2',
            log_to_tb = False,
            image_based = True,
            num_parallel_envs = 1):

    pelican_training_steps = 0
    panther_training_steps = 0
    pelican_model_type = model_type
    panther_model_type = model_type

    pelicans_tmp_exp_path = os.path.join(exp_path, 'pelicans_tmp')
    os.makedirs(pelicans_tmp_exp_path, exist_ok = True)
    panthers_tmp_exp_path = os.path.join(exp_path, 'panthers_tmp')
    os.makedirs(panthers_tmp_exp_path, exist_ok = True)
        
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
    if model_type.lower() == 'ppo2':
        parallel = True

    pnm_logs_exp_path = '/data/pnm_logs/test_' + basicdate
    os.makedirs(pnm_logs_exp_path, exist_ok = True)
    config_file_path = '/Components/plark-game/plark_game/game_config/10x10/balanced.json'

    # Train initial pelican vs default panther
    pelican_env = helper.get_envs('pelican',
                                  config_file_path,
                                  num_envs = num_parallel_envs,
                                  image_based = image_based,
                                  random_panther_start_position = True,
                                  max_illegal_moves_per_turn = 3,
                                  sparse = False,
                                  vecenv = parallel) 
    pelican_model = helper.make_new_model(model_type, policy, pelican_env)
    logger.info('Training initial pelican')
    pelican_agent_filepath, steps = train_agent(pelicans_tmp_exp_path,
                                                pelican_model,
                                                pelican_env,
                                                pelican_testing_interval,
                                                pelican_max_learning_steps,
                                                pelican_model_type,
                                                basicdate,
                                                writer,
                                                pelican_tb_log_name,
                                                early_stopping = True,
                                                previous_steps = 0)
    pelican_training_steps = pelican_training_steps + steps

    # Train initial panther agent vs default pelican
    panther_env = helper.get_envs('panther',
                                  config_file_path,
                                  num_envs = num_parallel_envs,
                                  image_based = image_based,
                                  random_panther_start_position = True,
                                  max_illegal_moves_per_turn = 3,
                                  sparse = False,
                                  vecenv = parallel)
    panther_model = helper.make_new_model(model_type, policy, panther_env)
    logger.info('Training initial panther')
    panther_agent_filepath, steps = train_agent(panthers_tmp_exp_path,
                                                panther_model,
                                                panther_env,
                                                panther_testing_interval,
                                                panther_max_learning_steps,
                                                panther_model_type,
                                                basicdate,
                                                writer,
                                                panther_tb_log_name,
                                                early_stopping = True,
                                                previous_steps = 0)
    panther_training_steps = panther_training_steps + steps

    # Initialize the payoffs and sets
    payoffs = np.zeros((1, 1))
    pelicans = []
    panthers = []

    # Train best responses until Nash equilibrium is found or max_iterations are reached
    logger.info('Parallel Nash Memory (PNM)')
    for i in range(max_pnm_iterations):
        logger.info('PNM iteration ' + str(i) + ' of ' + str(max_pnm_iterations))
        pelicans.append(pelican_agent_filepath)
        panthers.append(panther_agent_filepath)

        # Computing the payoff matrices and solving the corresponding LPs
        # Only compute for pelican in the sparse env, that of panther is the negative traspose (game is zero-sum)
        logger.info('Computing payoffs and mixtures')
        payoffs = compute_payoff_matrix(pelican_model,
                                        panther_model,
                                        payoffs,
                                        pelicans,
                                        panthers,
                                        config_file_path,
                                        trials = 100,
                                        parallel = parallel,
                                        image_based = image_based,
                                        num_parallel_envs = num_parallel_envs)
        logger.info(payoffs)
        np.save('%s/payoffs_%d.npy' % (pnm_logs_exp_path, i), payoffs)
        (mixture_pelicans, value_pelicans) = lp_solve.solve_zero_sum_game(payoffs)
        mixture_pelicans /= np.sum(mixture_pelicans)
        logger.info(mixture_pelicans)
        np.save('%s/mixture_pelicans_%d.npy' % (pnm_logs_exp_path, i), mixture_pelicans)
        (mixture_panthers, value_panthers) = lp_solve.solve_zero_sum_game(-payoffs.transpose())
        mixture_panthers /= np.sum(mixture_panthers)
        logger.info(mixture_panthers)
        np.save('%s/mixture_panthers_%d.npy' % (pnm_logs_exp_path, i), mixture_panthers)

        # Check if we found a stable NE, in that case we are done
        br_value_pelican = np.dot(mixture_pelicans, payoffs[-1])
        br_value_panther = np.dot(mixture_panthers, -payoffs[:, -1])
        if i > 0 and abs(br_value_pelican - value_pelicans) < stopping_eps and abs(br_value_panther - value_panthers) < stopping_eps:
            print('Stable Nash Equilibrium found')
            break

        # Train from skratch or retrain an existing model for pelican
        logger.info('Training pelican')
        if np.random.rand(1) < retraining_prob:
            idx = np.random.choice(pelicans, 1, p = mixture_pelicans)[0] 
            idx = glob.glob(idx+"/*.zip")[0]
            print(idx)
            #pelican_model = helper.loadAgent(pelicans[idx], pelican_model_type)
            #pelican_model = helper.loadAgent(idx, pelican_model_type)
            if pelican_model_type.lower() == 'dqn':
                model = DQN.load(idx)            
            elif pelican_model_type.lower() == 'ppo2':
                model = PPO2.load(idx)
            elif pelican_model_type.lower() == 'a2c':
                model = A2C.load(idx)    
            elif pelican_model_type.lower() == 'acktr':
                model = ACKTR.load(idx)
            
            
        else:
            pelican_model = helper.make_new_model(model_type, policy, pelican_env)
        pelican_agent_filepath, steps = train_agent_against_mixture('pelican',
                                                                    policy,
                                                                    pelicans_tmp_exp_path,
                                                                    pelican_model,
                                                                    panthers,
                                                                    mixture_pelicans,
                                                                    pelican_testing_interval,
                                                                    pelican_max_learning_steps,
                                                                    pelican_model_type,
                                                                    basicdate,
                                                                    writer,
                                                                    pelican_tb_log_name,
                                                                    config_file_path,
                                                                    previous_steps = pelican_training_steps,
                                                                    parallel = parallel,
                                                                    image_based = image_based,
                                                                    num_parallel_envs = num_parallel_envs)
        pelican_training_steps = pelican_training_steps + steps

        # Train from skratch or retrain an existing model for panther
        logger.info('Training panther')
        if np.random.rand(1) < retraining_prob:
            idx = np.random.choice(panthers, 1, p = mixture_panthers)[0]            
            idx = glob.glob(idx+"/*.zip")[0]
            #panther_model = helper.loadAgent(panthers[idx], panther_model_type)
            #panther_model = helper.loadAgent(idx, panther_model_type)
            if panther_model_type.lower() == 'dqn':
                model = DQN.load(idx)            
            elif panther_model_type.lower() == 'ppo2':
                model = PPO2.load(idx)
            elif panther_model_type.lower() == 'a2c':
                model = A2C.load(idx)    
            elif panther_model_type.lower() == 'acktr':
                model = ACKTR.load(idx)            
        else:
            panther_model = helper.make_new_model(model_type, policy, panther_env)
        panther_agent_filepath, steps = train_agent_against_mixture('panther',
                                                                    policy,
                                                                    panthers_tmp_exp_path,
                                                                    panther_model,
                                                                    pelicans,
                                                                    mixture_panthers,
                                                                    panther_testing_interval,
                                                                    panther_max_learning_steps,
                                                                    panther_model_type,
                                                                    basicdate,
                                                                    writer,
                                                                    panther_tb_log_name,
                                                                    config_file_path,
                                                                    previous_steps = panther_training_steps,
                                                                    parallel = parallel,
                                                                    image_based = image_based,
                                                                    num_parallel_envs = num_parallel_envs)
        panther_training_steps = panther_training_steps + steps

    logger.info('Training pelican total steps: ' + str(pelican_training_steps))
    logger.info('Training panther total steps: ' + str(panther_training_steps))
    # Make video
    video_path =  os.path.join(exp_path, 'test_pnm.mp4')
    basewidth,hsize = helper.make_video(pelican_model, pelican_env, video_path)

    # Saving final mixture and corresponding agents
    support_pelicans = np.nonzero(mixture_pelicans)[0]
    mixture_pelicans = mixture_pelicans[support_pelicans]
    np.save(exp_path + '/mixture_pelicans.npy', mixture_pelicans)    
    for i, idx in enumerate(mixture_pelicans):
        pelican_model = helper.loadAgent(pelicans[i], pelican_model_type)
        agent_filepath ,_, _= helper.save_model_with_env_settings(pelicans_tmp_exp_path, pelican_model, pelican_model_type, pelican_env, basicdate + "_ps_" + str(i))
    support_panthers = np.nonzero(mixture_panthers)[0]
    mixture_panthers = mixture_panthers[support_panthers]
    np.save(exp_path + '/mixture_panthers.npy', mixture_panthers)    
    for i, idx in enumerate(mixture_panthers):
        panther_model = helper.loadAgent(panthers[i], panther_model_type)
        agent_filepath ,_, _= helper.save_model_with_env_settings(panthers_tmp_exp_path, panther_model, panther_model_type, panther_env, basicdate + "_ps_" + str(i))
    return video_path, basewidth, hsize

def main():
    basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    basepath = '/data/agents/models'
    exp_name = 'test_' + basicdate
    exp_path = os.path.join(basepath, exp_name)
    logger.info(exp_path)

    run_pnm(exp_path,
            basicdate,
            pelican_testing_interval = 250,
            pelican_max_learning_steps = 250,
            panther_testing_interval = 250,
            panther_max_learning_steps = 250,
            max_pnm_iterations = 100,
            stopping_eps = 0.001,
            retraining_prob = 1.0,
            model_type = 'PPO2',
            log_to_tb = True,
            image_based = False,
            num_parallel_envs = 10)    

if __name__ == '__main__':
    main()


