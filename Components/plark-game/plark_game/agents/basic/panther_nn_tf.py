#THIS CLASS IS DEFUNCT - I AM ONLY KEEPING IT AROUND IN CASE WE HAVE TO GO BACK TO 
#TENSORFLOW
from plark_game.classes.pantherAgent import *
import tensorflow as tf
import numpy as np

class PantherNN(Panther_Agent):

    def __init__(self, num_inputs, num_hidden_layers=0, neurons_per_hidden_layer=0):
        self.num_inputs = num_inputs
        self.num_outputs = len(ACTION_LOOKUP) 
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer

        #tf.random.set_random_seed(5)

        #Build neural net
        self._build_nn()

        print("Built NN")

        #Start tf session
        self.sess = tf.Session()
        #sess = tf.Session(
        #    config=tf.ConfigProto(
        #        inter_op_parallelism_threads=4,
        #        intra_op_parallelism_threads=4
        #    )
        #)
        #init = tf.global_variables_initializer()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print("Ran TF session")


    def _build_nn(self):

        #Network inputs
        self.X = tf.placeholder("float", [1, self.num_inputs])

        #Initialise weights and biases
        self.weights = [] 
        self.biases = []

        if self.num_hidden_layers == 0:
            self.weights.append(tf.Variable(tf.random_normal([self.num_inputs, 
                                                              self.num_outputs])))
            self.biases.append(tf.Variable(tf.random_normal([self.num_outputs])))

        else:
            self.weights.append(tf.Variable(tf.random_normal([self.num_inputs, 
                                                              self.neurons_per_hidden_layer])))
            self.biases.append(tf.Variable(tf.random_normal([self.neurons_per_hidden_layer])))

            for i in range(self.num_hidden_layers-1):
                self.weights.append(
                    tf.Variable(tf.random_normal([self.neurons_per_hidden_layer, 
                                                  self.neurons_per_hidden_layer]))
                )
                self.biases.append(
                    tf.Variable(tf.random_normal([self.neurons_per_hidden_layer]))
                )

            self.weights.append(tf.Variable(tf.random_normal([self.neurons_per_hidden_layer, 
                                                              self.num_outputs])))
            self.biases.append(tf.Variable(tf.random_normal([self.num_outputs])))

    def _forward_pass(self, x):
        out = x
        for i in range(len(self.weights)):
            out = tf.add(tf.matmul(out, self.weights[i]), self.biases[i])
            #Add ReLU to all hidden layers
            if i < (len(self.weights)-1):
                out = tf.nn.relu(out)
            #Add softmax to the output layer
            else:
                out = tf.nn.softmax(out)

        return out
    
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

        net_out = self.sess.run(self._forward_pass(self.X), feed_dict={self.X: [state]})

        action = self._get_most_probable_action(net_out[0])
        #action = self._sample_action(net_out[0])

        return action

    #Returns the number of weights and biases
    def get_num_weights(self):
        num_weights = 0
        for layer_weights in self.weights:
            num_weights += self.sess.run(tf.size(layer_weights))
        for layer_biases in self.biases:
            num_weights += self.sess.run(tf.size(layer_biases))
        return num_weights

    def print_weights(self):
        for i in range(len(self.weights)):
            self.sess.run(tf.print(self.weights[i], summarize=-1))
            self.sess.run(tf.print(self.biases[i], summarize=-1))

    def _set_weights_err_msg(self, weights_len, num_weights_required):
        return "Trying to set {} weights to an NN that requires {} weights" \
            .format(weights_len, num_weights_required)

    #Sets a list of weights and biases
    def set_weights(self, new_weights):

        #Check new weights is of correct size
        num_weights_required = self.get_num_weights()
        assert num_weights_required == len(new_weights), \
                                       self._set_weights_err_msg(len(new_weights), \
                                                                 num_weights_required)

        #Set weights and biases layer by layer
        index = 0
        for i in range(len(self.weights)):

            #Slice out new weights
            w_size = self.sess.run(tf.size(self.weights[i]))
            layer_weights = new_weights[index:index+w_size]
            index += w_size

            #Resize and set new weights
            w_shape = self.sess.run(tf.shape(self.weights[i]))
            self.weights[i] = tf.Variable(np.reshape(layer_weights, w_shape), dtype='float32')

            #Slice out new biases
            b_size = self.sess.run(tf.size(self.biases[i]))
            layer_biases = new_weights[index:index+b_size]
            index += b_size

            #Set new biases
            self.biases[i] = tf.Variable(layer_biases, dtype='float32')

        #Reinitiaise variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    #Return weights and biases as a 1d list for optimisation rather than as tensors 
    def get_weights(self):
        weights = []

        for i in range(len(self.weights)):
            flattened_weights = self.sess.run(tf.reshape(self.weights[i], [-1])) 
            weights += flattened_weights.tolist()
            weights += self.sess.run(self.biases[i]).tolist()

        return weights
