from gym_plark.envs.plark_env import PlarkEnv
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from plark_game.agents.basic.panther_nn import PantherNN

from deap import creator, base, benchmarks, cma, tools, algorithms

import numpy as np

def evaluate(genome, agent):

    agent.set_weights(genome)

    env.reset()

    max_num_steps = 1

    reward = 0
    obs = env._observation()
    for step_num in range(max_num_steps):
        action = agent.getAction(obs)    
        obs, r, done, _ = env.step(action)
        reward += r

    return [reward]


    #Making the video

    #video_path = '/test.mp4'
    #basewidth,hsize = helper.make_video(model,env,video_path)

    #video = io.open(video_path, 'r+b').read()
    #encoded = base64.b64encode(video)


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
    lambda_ = 50
    strategy = cma.Strategy(centroid=centroid, sigma=init_sigma, lambda_=lambda_)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    num_gens = 250
    algorithms.eaGenerateUpdate(toolbox, ngen=num_gens, stats=stats, halloffame=hof)
