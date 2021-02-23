# +
# !pip install pycddlib
# #!pip install torch
# #!pip install numba
# !pip install stable-baselines3
# #!pip install lp_solve
# The following is needed on the DGX:
# #!pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
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

#######################################################################
# PARAMS 
# ######################################################################

TRAINING_STEPS = 100
PAYOFF_MATRIX_TRIALS = 25
MAX_ILLEGAL_MOVES_PER_TURN = 2
NORMALISE = True
MAX_N_OPPONENTS_TO_SAMPLE = 30 # so 28 max for 7 parallel envs
NUM_PARALLEL_ENVS = 7
MODEL_TYPE = 'PPO' # 'PPO' instead of 'PPO2' since we are using torch version
POLICY = 'MlpPolicy'
PARALLEL = True # keep it true while working with PPO

""

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
                          panthers):
    """
    - Pelican strategies are rows; panthers are columns
    - Payoffs are all to the pelican
    """

    # Resizing the payoff matrix for new strategies
    payoffs = np.pad(payoffs,
                     [(0, len(pelicans) - payoffs.shape[0]),
                      (0, len(panthers) - payoffs.shape[1])],
                     mode = 'constant')

    # Adding payoff for the last row strategy
    for i, opponent in enumerate(panthers):#        
        pelican_env.env_method('set_panther_using_path', opponent)            
        victory_prop, avg_reward = helper.check_victory(pelican, pelican_env, trials = PAYOFF_MATRIX_TRIALS)
        payoffs[-1, i] = victory_prop

    # Adding payoff for the last column strategy
    for i, opponent in enumerate(pelicans):
        panther_env.env_method('set_pelican_using_path', opponent)        
        victory_prop, avg_reward = helper.check_victory(panther, panther_env, trials = PAYOFF_MATRIX_TRIALS)
        payoffs[i, -1] = 1 - victory_prop # do in terms of pelican

    return payoffs

def train_agent_against_mixture(driving_agent, # agent that we train
                                exp_path,
                                model, 
                                env,
                                opponent_policy_fpaths, # policies of opponent of driving agent
                                opponent_mixture, # mixture of opponent of driving agent
                                basicdate,
                                previous_steps):

    ################################################################
    # Heuristic to compute number of opponents to sample as mixture
    ################################################################
    # min positive probability
    min_prob = min([pr for pr in opponent_mixture if pr > 0])
    target_n_opponents = NUM_PARALLEL_ENVS * int(1.0/min_prob)
    n_opponents = min(target_n_opponents, MAX_N_OPPONENTS_TO_SAMPLE) 

    if PARALLEL:
        # ensure that n_opponents is a multiple of 
        n_opponents = NUM_PARALLEL_ENVS * round(n_opponents/NUM_PARALLEL_ENVS)

    logger.info("=============================================")
    logger.info("Sampling %d opponents" % n_opponents)
    logger.info("=============================================")

    # sample n_opponents
    opponents = np.random.choice(opponent_policy_fpaths, 
                                 size = n_opponents, 
                                 p = opponent_mixture)

    logger.info("=============================================")
    logger.info("opponents has %d elements" % len(opponents))
    logger.info("=============================================")
    
    # If we use parallel envs, we run all the training against different sampled opponents in parallel
    if PARALLEL:
        # Method to load new opponents via filepath
        setter = 'set_panther_using_path' if driving_agent == 'pelican' else 'set_pelican_using_path'

        for i, opponent in enumerate(opponents):
            # stick this in the right slot, looping back after NUM_PARALLEL_ENVS
            env.env_method(setter, opponent, indices = [i % NUM_PARALLEL_ENVS])
            # when we have filled all NUM_PARALLEL_ENVS, then train
            if i > 0 and (i+1) % NUM_PARALLEL_ENVS == 0:
                logger.info("Beginning parallel training for {} steps".format(TRAINING_STEPS))
                model.set_env(env)
                model.learn(TRAINING_STEPS)
                previous_steps += TRAINING_STEPS

    # Otherwise we sample different opponents and we train against each of them separately
    else:
        for opponent in opponents:
            if driving_agent == 'pelican': 
                env.set_panther_using_path(opponent)
            else:
                env.set_pelican_using_path(opponent)

            logger.info("Beginning sequential training for {} steps".format(TRAINING_STEPS))
            model.set_env(env)
            model.learn(TRAINING_STEPS)
            previous_steps += TRAINING_STEPS
            
    # Save agent
    logger.info('Finished train agent')
    basicdate = basicdate + '_steps_' + str(previous_steps)
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, model, MODEL_TYPE, env, basicdate)
    agent_filepath = os.path.dirname(agent_filepath)

    return agent_filepath, previous_steps

def train_agent(exp_path,
                model,
                env,
                basicdate,
                previous_steps = 0):

    logger.info("Beginning individual training for {} steps".format(TRAINING_STEPS))
    model.set_env(env)
    model.learn(TRAINING_STEPS)
    
    logger.info('Finished train agent')
    basicdate = basicdate + '_steps_' + str(previous_steps)
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, model, MODEL_TYPE, env, basicdate)
    agent_filepath = os.path.dirname(agent_filepath)    
    return agent_filepath, previous_steps + TRAINING_STEPS

def run_pnm(exp_path,
            basicdate,
            max_pnm_iterations,
            stopping_eps,
            retraining_prob,
            sparse
            ):

    pelican_training_steps = 0
    panther_training_steps = 0

    pelicans_tmp_exp_path = os.path.join(exp_path, 'pelicans_tmp')
    os.makedirs(pelicans_tmp_exp_path, exist_ok = True)
    panthers_tmp_exp_path = os.path.join(exp_path, 'panthers_tmp')
    os.makedirs(panthers_tmp_exp_path, exist_ok = True)

    pnm_logs_exp_path = '/data/pnm_logs/test_' + basicdate
    os.makedirs(pnm_logs_exp_path, exist_ok = True)
    config_file_path = '/Components/plark-game/plark_game/game_config/10x10/balanced.json'

    # Train initial pelican vs default panther
    pelican_env = helper.get_envs('pelican',
                                  config_file_path,
                                  num_envs = NUM_PARALLEL_ENVS,
                                  image_based = False, # REMOVE FROM GET_ENVS
                                  random_panther_start_position = True,
                                  max_illegal_moves_per_turn = MAX_ILLEGAL_MOVES_PER_TURN,
                                  sparse = sparse,
                                  vecenv = PARALLEL,
                                  normalise=NORMALISE)
    pelican_model = helper.make_new_model(MODEL_TYPE, POLICY, pelican_env, n_steps=TRAINING_STEPS)
    logger.info('Training initial pelican')
    pelican_agent_filepath, pelican_training_steps = train_agent(pelicans_tmp_exp_path,
                                                pelican_model,
                                                pelican_env,
                                                basicdate,
                                                previous_steps = 0)

    # Train initial panther agent vs default pelican
    panther_env = helper.get_envs('panther',
                                  config_file_path,
                                  num_envs = NUM_PARALLEL_ENVS,
                                  image_based = False, # REMOVE FROM GET_ENVS
                                  random_panther_start_position = True,
                                  max_illegal_moves_per_turn = MAX_ILLEGAL_MOVES_PER_TURN,
                                  sparse = sparse,
                                  vecenv = PARALLEL,
                                  normalise=NORMALISE)
    panther_model = helper.make_new_model(MODEL_TYPE, POLICY, panther_env, n_steps=TRAINING_STEPS)
    logger.info('Training initial panther')
    panther_agent_filepath, panther_training_steps = train_agent(panthers_tmp_exp_path,
                                                panther_model,
                                                panther_env,
                                                basicdate,
                                                previous_steps = 0)

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
                                        panthers)

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
            if abs(br_value_pelican - value_to_pelican) < stopping_eps and\
               abs(br_value_panther - value_to_pelican) < stopping_eps:

                print('Stable Nash Equilibrium found')
                break

        logger.info("SOLVING NEW GAME:")
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
            pelican_model = helper.loadAgent(path, MODEL_TYPE)
        else:
            pelican_model = helper.make_new_model(MODEL_TYPE, POLICY, pelican_env, n_steps=TRAINING_STEPS)
        pelican_agent_filepath, pelican_training_steps = train_agent_against_mixture('pelican',
                                                                    pelicans_tmp_exp_path,
                                                                    pelican_model,
                                                                    pelican_env,
                                                                    panthers,
                                                                    mixture_panthers,
                                                                    basicdate,
                                                                    previous_steps = pelican_training_steps)

        # Train from scratch or retrain an existing model for panther
        logger.info('Training panther')
        if np.random.rand(1) < retraining_prob:
            path = np.random.choice(panthers, 1, p = mixture_panthers)[0]
            path = glob.glob(path + "/*.zip")[0]
            panther_model = helper.loadAgent(path, MODEL_TYPE)
        else:
            panther_model = helper.make_new_model(MODEL_TYPE, POLICY, panther_env, n_steps=TRAINING_STEPS)
        panther_agent_filepath, panther_training_steps = train_agent_against_mixture('panther',
                                                                    panthers_tmp_exp_path,
                                                                    panther_model,
                                                                    panther_env,
                                                                    pelicans,
                                                                    mixture_pelicans,
                                                                    basicdate,
                                                                    previous_steps = panther_training_steps)
        
        logger.info("PNM iteration lasted: %d seconds" % (time.time() - start))
        
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
        pelican_model = helper.loadAgent(pelicans[i], MODEL_TYPE)
        agent_filepath ,_, _= helper.save_model_with_env_settings(pelicans_tmp_exp_path, pelican_model, MODEL_TYPE, pelican_env, basicdate + "_ps_" + str(i))
    support_panthers = np.nonzero(mixture_panthers)[0]
    mixture_panthers = mixture_panthers[support_panthers]
    np.save(exp_path + '/mixture_panthers.npy', mixture_panthers)
    for i, idx in enumerate(mixture_panthers):
        panther_model = helper.loadAgent(panthers[i], MODEL_TYPE)
        agent_filepath ,_, _= helper.save_model_with_env_settings(panthers_tmp_exp_path, panther_model, MODEL_TYPE, panther_env, basicdate + "_ps_" + str(i))
    return video_path, basewidth, hsize

def main():
    basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    basepath = '/data/agents/models'
    exp_name = 'test_' + basicdate
    exp_path = os.path.join(basepath, exp_name)
    logger.info(exp_path)

    run_pnm(exp_path,
            basicdate,
            max_pnm_iterations = 100,
            stopping_eps = 0.001, # required quality of RB-NE 
            retraining_prob = .8, # prob that we bootstrap for best response
            sparse = False)

if __name__ == '__main__':
    main()
