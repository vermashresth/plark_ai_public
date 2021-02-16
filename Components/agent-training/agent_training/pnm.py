import datetime
import numpy as np
import os
from stable_baselines.common.vec_env import SubprocVecEnv
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from gym_plark.envs.plark_env import PlarkEnv
from tensorboardX import SummaryWriter
import helper
import lp_solve
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def compute_payoff_matrix(driving_agent,
                          keep_instances,
                          model_type,
                          policy,
                          payoffs,
                          env,
                          players,
                          opponents,
                          trials = 1000):

    payoffs = np.pad(payoffs, [(0, len(players) - payoffs.shape[0]), (0, len(opponents) - payoffs.shape[1])], mode='constant')
    # Adding payoff for the last row player
    model = players[-1]
    if not keep_instances:
        model = helper.loadAgent(model, model_type)
    if driving_agent == 'pelican':
        env.set_pelican(model)      
    else:
        env.set_panther(model) 
        
    for i, opponent in enumerate(opponents):
        if not keep_instances: # i.e., if we want to load from file...
            opponent = helper.loadAgent(opponent, model_type)
        if driving_agent == 'pelican':
            env.set_panther(opponent)
        else:
            env.set_pelican(opponent)
        victory_count, avg_reward = helper.check_victory(model, env, trials = trials)
        if driving_agent == 'pelican':
            payoffs[-1, i] = avg_reward
        else:
            payoffs[i, -1] = avg_reward

    # Adding payoff for all the old players against the last column player
    opponent = opponents[-1]
    if not keep_instances:
        opponent = helper.loadAgent(opponent, model_type)
    if driving_agent == 'pelican':
        env.set_panther(opponent)        
    else:
        env.set_pelican(opponent)        

    for i, player in enumerate(players):
        if not keep_instances: # i.e., if we want to load from file...
            player = helper.loadAgent(player, model_type)
        if driving_agent == 'pelican':
            env.set_pelican(player)   
        else:
            env.set_panther(player)
        victory_count, avg_reward = helper.check_victory(player, env, trials = trials)
        if driving_agent == 'pelican':
            payoffs[i, -1] = avg_reward
        else:
            payoffs[-1, i] = avg_reward
    return payoffs

def train_agent_against_mixture(driving_agent,
                                keep_instances,
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
                                tb_writer,
                                tb_log_name,
                                early_stopping = True,
                                previous_steps = 0):       

    opponents = np.random.choice(tests, size = max_steps // testing_interval, p = mixture)
    steps = 0

    for opponent_model in opponents:
        if not keep_instances:
            opponent_model = helper.loadAgent(opponent_model, model_type)
        if driving_agent == 'pelican':
            env.set_panther(opponent_model)          
        else:
            env.set_pelican(opponent_model)
            
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
                                                 previous_steps = steps,
                                                 save_model = False)
        steps += new_steps
    if not keep_instances:
        basicdate = basicdate + '_steps_' + str(previous_steps + steps)
        agent_filepath, _, _ = helper.save_model_with_env_settings(exp_path, model, model_type, env, basicdate)
        agent_filepath = os.path.dirname(agent_filepath)
    return agent_filepath, steps

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
    #model.set_env(env)
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

def run_pnm(exp_name,
                 exp_path,
                 basicdate,
                 pelican_testing_interval = 100,
                 pelican_max_learning_steps = 10000,
                 panther_testing_interval = 100,
                 panther_max_learning_steps = 10000,
                 pnm_iterations = 10000,
                 model_type = 'PPO2',
                 log_to_tb = False,
                 image_based = True,
                 num_parallel_envs = 1,
                 keep_instances = False):

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
    #if model_type.lower() == 'ppo2':
    #    parallel = True

    log_dir_base = 'pnm_logs/'
    os.makedirs(log_dir_base, exist_ok = True)
    config_file_path = 'C:\\Users\Jacopo\Documents\plark_ai_public\Components\plark-game\plark_game\game_config\\10x10\\balanced.json'

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

    pelican_model = helper.make_new_model(model_type, policy, pelican_env)
    logger.info('Training initial pelican')
        
    pelican_agent_filepath, steps = train_agent(pelican_tmp_exp_path,
                                                pelican_model,
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

    panther_model = helper.make_new_model(model_type, policy, panther_env)
    logger.info('Training initial panther')
    panther_agent_filepath, steps = train_agent(panther_tmp_exp_path,
                                                panther_model,
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

    # Initialize the payoffs and sets
    payoffs = np.zeros((1,1))
    pelicans = []
    panthers = []

    # Train agent vs agent
    logger.info('Parallel Nash Memory (PNM)')
    for i in range(pnm_iterations):
        logger.info('PNM iteration ' + str(i) + ' of ' + str(pnm_iterations))

        if keep_instances:
            pelicans.append(pelican_model)
            panthers.append(panther_model)
        else:
            pelicans.append(pelican_agent_filepath)
            panthers.append(panther_agent_filepath)

        # Computing the payoff matrices and solving the corresponding LPs
        # Only compute for pelican in the sparse env, that of panther is the negative traspose (game is zero-sum)
        logger.info('Computing payoffs and mixtures')
        payoffs = compute_payoff_matrix('pelican',
                                        keep_instances,
                                        model_type,
                                        policy,
                                        payoffs,
                                        pelican_test_env,
                                        pelicans,
                                        panthers)
        logger.info(payoffs)
        np.save('%s/payoffs_%d.npy' % (log_dir_base, i), payoffs)
        (mixture_pelicans, value_pelicans) = lp_solve.solve_zero_sum_game(payoffs)
        mixture_pelicans /= np.sum(mixture_pelicans)
        logger.info(mixture_pelicans)
        np.save('%s/mixture1_%d.npy' % (log_dir_base, i), mixture_pelicans)
        (mixture_panthers, value_panthers) = lp_solve.solve_zero_sum_game(-payoffs.transpose())
        mixture_panthers /= np.sum(mixture_panthers)
        logger.info(mixture_panthers)
        np.save('%s/mixture2_%d.npy' % (log_dir_base, i), mixture_panthers)

        logger.info('Training pelican')
        pelican_model = helper.make_new_model(model_type, policy, pelican_env)
        pelican_agent_filepath, steps = train_agent_against_mixture('pelican',
                                                                    keep_instances,
                                                                    policy,
                                                                    exp_path,
                                                                    pelican_model,
                                                                    pelican_env,
                                                                    panthers,
                                                                    mixture_pelicans,
                                                                    pelican_testing_interval,
                                                                    pelican_max_learning_steps,
                                                                    pelican_model_type,
                                                                    basicdate,
                                                                    writer,
                                                                    pelican_tb_log_name,
                                                                    previous_steps = pelican_training_steps)
        
        
        pelican_training_steps = pelican_training_steps + steps

        logger.info('Training panther')
        panther_model = helper.make_new_model(model_type, policy, panther_env)
        panther_agent_filepath, steps = train_agent_against_mixture('panther',
                                                                    keep_instances,
                                                                    policy,
                                                                    exp_path,
                                                                    panther_model,
                                                                    panther_env,
                                                                    pelicans,
                                                                    mixture_panthers,
                                                                    panther_testing_interval,
                                                                    panther_max_learning_steps,
                                                                    panther_model_type,
                                                                    basicdate,
                                                                    writer,
                                                                    panther_tb_log_name,
                                                                    previous_steps = panther_training_steps)
        panther_training_steps = panther_training_steps + steps

    # Saving final version of the agents
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, pelican_model, pelican_model_type, pelican_env, basicdate)
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, panther_model, panther_model_type, panther_env, basicdate)

    logger.info('Training pelican total steps: ' + str(pelican_training_steps))
    logger.info('Training panther total steps: ' + str(panther_training_steps))
    # Make video
    video_path =  os.path.join(exp_path, 'test_pnm.mp4')
    basewidth,hsize = helper.make_video(pelican_model, pelican_env, video_path)
    return video_path, basewidth, hsize

def main():
    basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    basepath = 'data/agents/models'
    exp_name = 'test_' + basicdate
    exp_path = os.path.join(basepath, exp_name)

    logger.info(exp_path)

    # run_pnm(exp_name,exp_path,basicdate)
    run_pnm(exp_name,
                 exp_path,
                 basicdate,
                  pelican_testing_interval = 1000,
                  pelican_max_learning_steps = 50000,
                  panther_testing_interval = 1000,
                  panther_max_learning_steps = 50000,
                  pnm_iterations = 200,
                  model_type = 'PPO2',
                  log_to_tb = True,
                  image_based = False,
                  num_parallel_envs = 1,
                  keep_instances = True)    

if __name__ == '__main__':
    main()