from gym_plark.envs.plark_env import PlarkEnv
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from plark_game.agents.basic.panther_nn import PantherNN

def evaluate(agent):

    env.reset()

    max_num_steps = 1

    obs = env._observation()
    obs = [3.0, 4.0]

    for step_num in range(max_num_steps):
        action = agent.getAction(obs)    
        #obs, reward, done, _ = env.step(action)

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
    panther_agent = PantherNN(num_inputs=2, num_hidden_layers=num_hidden_layers, 
                              neurons_per_hidden_layer=neurons_per_hidden_layer)  
    panther_agent.set_weights([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    #panther_agent.set_weights([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    
    panther_weights = panther_agent.get_weights()
    print(panther_weights)

    exit()

    evaluate(panther_agent)


'''
import random
from deap import creator, base, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)

#NGEN=40
NGEN=1
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
top10 = tools.selBest(population, k=10)
print(top10)
'''
