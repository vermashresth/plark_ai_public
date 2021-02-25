from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from plark_game.agents.basic.panther_nn import PantherNN
from plark_game.agents.basic.pelican_nn import PelicanNN

if __name__ == '__main__':

    #Env variables
    config_file_path = '/Components/plark-game/plark_game/game_config/10x10/nn/nn_coevolution_balanced.json'
    normalise_obs = True

    #Instantiate dummy env and dummy agent
    #I need to do this to ascertain the number of weights needed in the optimisation
    #procedure
    dummy_env = PlarkEnvSparse(config_file_path=config_file_path, image_based=False, 
                               driving_agent='panther', normalise=normalise_obs)

    #Neural net variables
    num_inputs = len(dummy_env._observation())
    num_hidden_layers = 0
    neurons_per_hidden_layer = 0

    panther_dummy_agent = PantherNN(num_inputs=num_inputs, num_hidden_layers=num_hidden_layers, 
                                    neurons_per_hidden_layer=neurons_per_hidden_layer)  
    #I need to figure out how to get rid of the 139 magic number
    pelican_dummy_agent = PelicanNN(num_inputs=139, num_hidden_layers=num_hidden_layers, 
                                    neurons_per_hidden_layer=neurons_per_hidden_layer)  

    #num_panther_weights = panther_dummy_agent.get_num_weights()
    #num_pelican_weights = pelican_dummy_agent.get_num_weights()

    #Let's try instantiating with dummy agents and setting the agents competing against each 
    #other
    dummy_env.reset()

    max_num_steps = 200

    reward = 0
    obs = dummy_env._observation()
    for step_num in range(max_num_steps):
        action = panther_dummy_agent.getAction(obs)    
        print(action)
        obs, r, done, info = dummy_env.step(action)
        reward += r
        if done:
            break
