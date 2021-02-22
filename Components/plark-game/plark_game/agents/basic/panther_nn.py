from plark_game.classes.pantherAgent import *
from plark_game.agents.basic.nn_agent import *
import numpy as np
import torch

class PantherNN(NNAgent):

    def __init__(self, num_inputs, num_hidden_layers=0, neurons_per_hidden_layer=0):
        super().__init__(num_inputs, len(ACTION_LOOKUP), 
                         num_hidden_layers, neurons_per_hidden_layer)

    def action_lookup(self, action):
        return ACTION_LOOKUP[action]
