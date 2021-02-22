# +
# #!pip install pycddlib
# #!pip install torch
# #!pip install numba
# #!pip install stable-baselines3

# The following is needed on the DGX:
# !pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
import sys
sys.path.insert(1, '/Components/')

import datetime
import numpy as np
import pandas as pd
import os
import gc
import glob
from tensorboardX import SummaryWriter
import helper
import lp_solve
import torch
import matplotlib.pyplot as plt
import time
# -

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ######################################################################
# PARAMS (in a config file later?)

PAYOFF_MATRIX_TRIALS = 50
MAX_ILLEGAL_MOVES_PER_TURN = 2
NORMALISE = True

""
def get_fig(df):
    # fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    # df[['NE_Payoff', 'Pelican_BR_Payoff', 'Panther_BR_Payoff']].plot(ax=ax1)
    # df[['Pelican_supp_size', 'Panther_supp_size']].plot(kind='bar', ax=ax2)
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    df[['NE_Payoff', 'Pelican_BR_Payoff', 'Panther_BR_Payoff']].plot(ax=ax1,fontsize=6)
    ax1.legend(loc='upper right',prop={'size': 7})
    ax1.set_ylabel('Payoff to Pelican')
    df[['Pelican_supp_size', 'Panther_supp_size']].plot(kind='bar', ax=ax2, rot=0)
    ax2.tick_params(axis='x', which='both', labelsize=6)
    ax2.legend(loc='upper left',prop={'size': 8})



def compute_payoff_matrix(pelican,
                          panther,
                          pelican_env,
                          panther_env,                          
                          payoffs,
                          pelicans,
                          panthers,
                          trials = 1000):
    """
    CHECK:

    - Pelican strategies are rows; panthers are columns
    - Payoffs are all to the panther though ?? 
      If so then we solving game wrong presumably?

    """

    # Resizing the payoff matrix for new strategies
    payoffs = np.pad(payoffs,
                     [(0, len(pelicans) - payoffs.shape[0]),
                      (0, len(panthers) - payoffs.shape[1])],
                     mode = 'constant')

    # Adding payoff for the last row strategy
    for i, opponent in enumerate(panthers):#        
        pelican_env.env_method('set_panther_using_path', opponent)            
        victory_count, avg_reward = helper.check_victory(pelican, pelican_env, trials = trials)
        payoffs[-1, i] = avg_reward

    # Adding payoff for the last column strategy
    for i, opponent in enumerate(pelicans):
        panther_env.env_method('set_pelican_using_path', opponent)        
        victory_count, avg_reward = helper.check_victory(panther, panther_env, trials = trials)
        # Given that we are storing everything in one table, and the value below is now computed
        # from the perspective of the panther, I assume we need this value to be negative?
        payoffs[i, -1] = -avg_reward
    return payoffs

def train_agent_against_mixture(driving_agent,
                                policy,
                                exp_path,
                                model,
                                env,
                                tests,
                                mixture,
                                testing_interval,
                                max_steps,
                                model_type,
                                basicdate,
                                early_stopping = True,
                                previous_steps = 0,
                                parallel = False):

    opponents = np.random.choice(tests, size = max_steps // testing_interval, p = mixture)
    
    # If we use parallel envs, we run all the training against different sampled opponents in parallel
    if parallel:
        # Method to load new opponents via filepath
        setter = 'set_panther_using_path' if driving_agent == 'pelican' else 'set_pelican_using_path'
        for i, opponent in enumerate(opponents):
            env.env_method(setter, opponent, indices = [i])
        
        agent_filepath, new_steps = train_agent(exp_path,
                                                model,
                                                env,
                                                max_steps,
                                                testing_interval,
                                                model_type,
                                                basicdate,
                                                early_stopping = True,
                                                previous_steps = previous_steps)

    # Otherwise we sample different opponents and we train against each of them separately
    else:
        opponents = np.random.choice(tests, size = max_steps // testing_interval, p = mixture)
        for opponent in opponents:
            if driving_agent == 'pelican': 
                env.set_panther_using_path(opponent)
            else:
                env.set_pelican_using_path(opponent)

            agent_filepath, new_steps = train_agent(exp_path,
                                                    model,
                                                    env,
                                                    testing_interval,
                                                    testing_interval,
                                                    model_type,
                                                    basicdate,
                                                    early_stopping = True,
                                                    previous_steps = previous_steps)
            previous_steps += new_steps

    return agent_filepath, new_steps

def train_agent(exp_path,
                model,
                env,
                testing_interval,
                max_steps,
                model_type,
                basicdate,
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
            num_parallel_envs = 1,
            early_stopping=True,
            sparse=False):

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
    if 'ppo' in model_type.lower():
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
                                  max_illegal_moves_per_turn = MAX_ILLEGAL_MOVES_PER_TURN,
                                  sparse = sparse,
                                  vecenv = parallel)
    pelican_model = helper.make_new_model(model_type, policy, pelican_env, n_steps=pelican_testing_interval)
    logger.info('Training initial pelican')
    pelican_agent_filepath, steps = train_agent(pelicans_tmp_exp_path,
                                                pelican_model,
                                                pelican_env,
                                                pelican_testing_interval,
                                                pelican_max_learning_steps,
                                                pelican_model_type,
                                                basicdate,
                                                early_stopping = True,
                                                previous_steps = 0)
    pelican_training_steps = pelican_training_steps + steps

    # Train initial panther agent vs default pelican
    panther_env = helper.get_envs('panther',
                                  config_file_path,
                                  num_envs = num_parallel_envs,
                                  image_based = image_based,
                                  random_panther_start_position = True,
                                  max_illegal_moves_per_turn = MAX_ILLEGAL_MOVES_PER_TURN,
                                  sparse = sparse,
                                  vecenv = parallel)
    panther_model = helper.make_new_model(model_type, policy, panther_env, n_steps=panther_testing_interval)
    logger.info('Training initial panther')
    panther_agent_filepath, steps = train_agent(panthers_tmp_exp_path,
                                                panther_model,
                                                panther_env,
                                                panther_testing_interval,
                                                panther_max_learning_steps,
                                                panther_model_type,
                                                basicdate,
                                                early_stopping = True,
                                                previous_steps = 0)
    panther_training_steps = panther_training_steps + steps

    # Initialize the payoffs and sets
    payoffs = np.zeros((1, 1))
    pelicans = []
    panthers = []
    # Initialize old NE stuff for stopping criterion
    value_to_pelican = 0.
    mixture_pelicans = np.array([1.])
    mixture_panthers = np.array([1.])
    # Create DataFrame for plotting purposes
    df_cols = ["NE_Payoff", "Pelican_BR_Payoff", "Panther_BR_Payoff", "Pelican_supp_size", "Panther_supp_size"]
    df = pd.DataFrame(columns = df_cols)

    # Train best responses until Nash equilibrium is found or max_iterations are reached
    logger.info('Parallel Nash Memory (PNM)')
    for i in range(max_pnm_iterations):
        start = time.time()
        
        logger.info("*********************************************************")
        logger.info('PNM iteration ' + str(i + 1) + ' of ' + str(max_pnm_iterations))
        logger.info("*********************************************************")

        pelicans.append(pelican_agent_filepath)
        panthers.append(panther_agent_filepath)

        # Computing the payoff matrices and solving the corresponding LPs
        # Only compute for pelican in the sparse env, that of panther is the negative traspose (game is zero-sum)
        logger.info('Computing payoffs and mixtures')
        payoffs = compute_payoff_matrix(pelican_model,
                                        panther_model,
                                        pelican_env,
                                        panther_env,
                                        payoffs,
                                        pelicans,
                                        panthers,
                                        trials = PAYOFF_MATRIX_TRIALS)

        #logger.info("Memory allocated before: " + str(torch.cuda.memory_allocated()))
        #logger.info("Clearing GPU memory.")
        #del pelican_model
        #del panther_model
        #gc.collect()
        #torch.cuda.empty_cache()
        #logger.info("Memory allocated after: " + str(torch.cuda.memory_allocated()))

        logger.info("=================================================")
        logger.info("New matrix game:")
        logger.info("As numpy array:")
        logger.info('\n' + str(payoffs))
        logger.info("As dataframe:")
        tmp_df = pd.DataFrame(payoffs).rename_axis('Pelican', axis = 0).rename_axis('Panther', axis = 1)
        logger.info('\n' + str(tmp_df))

        # save payoff matrix
        np.save('%s/payoffs_%d.npy' % (pnm_logs_exp_path, i), payoffs)

        def get_support_size(mixture):
            # return size of the support of mixed strategy mixture
            return sum([1 if m > 0 else 0 for m in mixture])

        # Check if we found a stable NE, in that case we are done (and fitting DF)
        if i > 0:
            # Both BR payoffs (from against last time's NE) in terms of pelican payoff
            br_value_pelican = np.dot(mixture_pelicans, payoffs[-1, :-1])
            br_value_panther = np.dot(mixture_panthers, payoffs[:-1, -1])

            ssize_pelican = get_support_size(mixture_pelicans)
            ssize_panther = get_support_size(mixture_panthers)

            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            logger.info("\n\
                         Pelican BR payoff: %.3f,\n\
                         Value of Game: %.3f,\n\
                         Panther BR payoff: %.3f,\n\
                         Pelican Supp Size: %d,\n\
                         Panther Supp Size: %d,\n" % (
                                                      br_value_pelican, 
                                                      value_to_pelican, 
                                                      br_value_panther,
                                                      ssize_pelican,
                                                      ssize_panther
                                                      ))
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            values = dict(zip(df_cols, [value_to_pelican, br_value_pelican, 
                                                          br_value_panther, 
                                                          ssize_pelican, 
                                                          ssize_panther]))
            df = df.append(values, ignore_index = True)

            # Write to csv file
            df_path =  os.path.join(exp_path, 'values_iter_%02d.csv' % i)
            df.to_csv(df_path, index = False)
            get_fig(df)
            fig_path = os.path.join(exp_path, 'values_iter_%02d.pdf' % i) 
            plt.savefig(fig_path)
            print("==========================================")
            print("WRITTEN DF TO CSV: %s" % df_path)
            print("==========================================")

            # here value_to_pelican is from the last time the subgame was solved
            if early_stopping and\
                    abs(br_value_pelican - value_to_pelican) < stopping_eps and\
                    abs(br_value_panther - value_to_pelican) < stopping_eps:
                print('Stable Nash Equilibrium found')
                break

        # solve game for pelican
        (mixture_pelicans, value_to_pelican) = lp_solve.solve_zero_sum_game(payoffs)
        # with np.printoptions(precision=3):
        logger.info(mixture_pelicans)
        mixture_pelicans /= np.sum(mixture_pelicans)
        # with np.printoptions(precision=3):
        logger.info("After normalisation:")
        logger.info(mixture_pelicans)
        np.save('%s/mixture_pelicans_%d.npy' % (pnm_logs_exp_path, i), mixture_pelicans)

        # solve game for panther
        (mixture_panthers, value_panthers) = lp_solve.solve_zero_sum_game(-payoffs.transpose())
        # with np.printoptions(precision=3):
        logger.info(mixture_panthers)
        mixture_panthers /= np.sum(mixture_panthers)
        # with np.printoptions(precision=3):
        logger.info("After normalisation:")
        logger.info(mixture_panthers)
        np.save('%s/mixture_panthers_%d.npy' % (pnm_logs_exp_path, i), mixture_panthers)

        # end of logging matrix game and solution
        logger.info("=================================================")

        # Train from skratch or retrain an existing model for pelican
        logger.info('Training pelican')
        if np.random.rand(1) < retraining_prob:
            path = np.random.choice(pelicans, 1, p = mixture_pelicans)[0]
            path = glob.glob(path + "/*.zip")[0]
            pelican_model = helper.loadAgent(path, pelican_model_type)
        else:
            pelican_model = helper.make_new_model(model_type, policy, pelican_env, n_steps=pelican_testing_interval)
        pelican_agent_filepath, steps = train_agent_against_mixture('pelican',
                                                                    policy,
                                                                    pelicans_tmp_exp_path,
                                                                    pelican_model,
                                                                    pelican_env,
                                                                    panthers,
                                                                    mixture_pelicans,
                                                                    pelican_testing_interval,
                                                                    pelican_max_learning_steps,
                                                                    pelican_model_type,
                                                                    basicdate,
                                                                    previous_steps = pelican_training_steps,
                                                                    parallel = parallel)
        pelican_training_steps = pelican_training_steps + steps

        # Train from skratch or retrain an existing model for panther
        logger.info('Training panther')
        if np.random.rand(1) < retraining_prob:
            path = np.random.choice(panthers, 1, p = mixture_panthers)[0]
            path = glob.glob(path + "/*.zip")[0]
            panther_model = helper.loadAgent(path, panther_model_type)
        else:
            panther_model = helper.make_new_model(model_type, policy, panther_env, n_steps=panther_testing_interval)
        panther_agent_filepath, steps = train_agent_against_mixture('panther',
                                                                    policy,
                                                                    panthers_tmp_exp_path,
                                                                    panther_model,
                                                                    panther_env,
                                                                    pelicans,
                                                                    mixture_panthers,
                                                                    panther_testing_interval,
                                                                    panther_max_learning_steps,
                                                                    panther_model_type,
                                                                    basicdate,
                                                                    previous_steps = panther_training_steps,
                                                                    parallel = parallel)
        panther_training_steps = panther_training_steps + steps
        
        logger.info("PNM iteration lasted %d: " % (time.time() - start))

        
    logger.info('Training pelican total steps: ' + str(pelican_training_steps))
    logger.info('Training panther total steps: ' + str(panther_training_steps))
    # Store DF for printing
    df_path = os.path.join(exp_path, "values.csv")
    df.to_csv(df_path, index = False)
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
            model_type = 'PPO', # 'PPO' instead of 'PPO2' since we are using torch version
            log_to_tb = True,
            image_based = False,
            num_parallel_envs = 10,
            early_stopping = False,
            sparse = False)

if __name__ == '__main__':
    main()
