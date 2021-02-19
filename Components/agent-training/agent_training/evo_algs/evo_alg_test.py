from gym_plark.envs.plark_env import PlarkEnv
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from plark_game.agents.basic.panther_nn import PantherNN
from agent_training import helper

from deap import creator, base, benchmarks, cma, tools, algorithms

import numpy as np

indv = 0
def evaluate(genome, agent):
    global indv
    print("Evaluating indv:", indv)

    #print("Genome:", genome)

    agent.set_weights(genome)

    env.reset()

    max_num_steps = 200

    reward = 0
    obs = env._observation()
    for step_num in range(max_num_steps):
        action = agent.getAction(obs)    
        obs, r, done, _ = env.step(action)
        reward += r
        if done:
            print("Finished at step num:", step_num)
            break

    print("Reward:", reward)
    indv += 1

    #Make a video
    video_path = '/evo_run.mp4'
    helper.make_video_plark_env(agent, env, video_path, n_steps=max_num_steps)

    #video = io.open(video_path, 'r+b').read()
    #encoded = base64.b64encode(video)

    return [reward]




if __name__ == '__main__':

    #Instantiate the env
    config_file_path = '/Components/plark-game/plark_game/game_config/10x10/nn/balanced_nn.json'
    env = PlarkEnvSparse(config_file_path=config_file_path, image_based=False, 
                            driving_agent='panther')

    num_inputs = len(env._observation())
    num_hidden_layers = 0
    neurons_per_hidden_layer = 0
    panther_agent = PantherNN(num_inputs=num_inputs, num_hidden_layers=num_hidden_layers, 
                                neurons_per_hidden_layer=neurons_per_hidden_layer)  
    num_weights = panther_agent.get_num_weights()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, agent=panther_agent)

    #np.random.seed(108)

    #Initial location of distribution centre
    centroid = [0.0] * num_weights
    #Initial standard deviation of the distribution
    init_sigma = 1.0
    #Number of children to produce at each generation
    #lambda_ = 20 * num_weights
    lambda_ = 2
    strategy = cma.Strategy(centroid=centroid, sigma=init_sigma, lambda_=lambda_)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("Running CMAES...")
    num_gens = 1
    population, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=num_gens, 
                                                      stats=stats, halloffame=hof)
