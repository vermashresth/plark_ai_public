from plark_game.classes.agent import Agent
import numpy as np
import torch

class NNAgent(Agent):

    def __init__(self, num_inputs, num_outputs, 
                 num_hidden_layers=0, neurons_per_hidden_layer=0):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer

        #Build neural net
        self._build_nn()

    def _build_nn(self):

        layers = []

        if self.num_hidden_layers == 0:
            layers.append(torch.nn.Linear(self.num_inputs, self.num_outputs))

        else:
            layers.append(torch.nn.Linear(self.num_inputs, self.neurons_per_hidden_layer))
            #Hidden layers have ReLU activation
            layers.append(torch.nn.ReLU())

            for i in range(self.num_hidden_layers-1):
                layers.append(torch.nn.Linear(self.neurons_per_hidden_layer,
                                              self.neurons_per_hidden_layer))
                layers.append(torch.ReLU())

            layers.append(torch.nn.Linear(self.neurons_per_hidden_layer, self.num_outputs))

        #Final layer goes through a softmax
        layers.append(torch.nn.Softmax(dim=0))

        self.nn = torch.nn.Sequential(*layers)

    #Takes a list, passes through the network and returns a list
    def _forward_pass(self, x):
        x = torch.tensor(x, dtype=torch.float64)
        net_out = self.nn.forward(x)
        return net_out.tolist()
    
    #Randomly sample action from network output probability distribution
    def _sample_action(self, net_out):
        action_nums = list(range(len(net_out)))
        return np.random.choice(action_nums, p=net_out)

    #Get the most probable action from the network output probability distribution
    def _get_most_probable_action(self, net_out):
        return np.argmax(net_out)

    def getAction(self, state):
        assert len(state) == self.num_inputs, "State length: {}, num inputs: {}" \
            .format(len(state), self.num_inputs)

        #Push state through network
        net_out = self._forward_pass(state)

        #Get action from nework output
        #action = self._get_most_probable_action(net_out)
        action = self._sample_action(net_out)

        return action

    #Returns the number of weights
    def get_num_weights(self):
        num_weights = 0
        for layer in self.nn:
            for params in layer.parameters():
                #num_weights += np.prod(params.size())
                num_weights += params.numel()
        return num_weights

    def print_weights(self):
        for layer in self.nn:
            for params in layer.parameters():
                print(params)

    def _set_weights_err_msg(self, weights_len, num_weights_required):
        return "Trying to set {} weights to an NN that requires {} weights" \
            .format(weights_len, num_weights_required)

    #Sets a list of weights
    def set_weights(self, new_weights):

        #Check new weights is of correct size
        num_weights_required = self.get_num_weights()
        assert num_weights_required == len(new_weights), \
                                       self._set_weights_err_msg(len(new_weights), \
                                                                 num_weights_required)

        weight_index = 0
        for layer in self.nn:
            for params in layer.parameters():

                #Slice out new weights
                p_weights = new_weights[weight_index : weight_index + params.numel()] 
                weight_index += params.numel()

                #Resize and set new weights
                params.data = torch.tensor(np.reshape(p_weights, params.size()), \
                                           dtype=torch.float64)

    #Return weights as a 1d list for optimisation rather than as tensors 
    def get_weights(self):
        weights = []
        for layer in self.nn:
            for params in layer.parameters():
                weights += params.flatten().tolist()
        return weights
                

