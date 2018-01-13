import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # initialize weights and learning rate
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # define activation function
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
            epoch: number of times passed through the data
        '''
        # number of records
        n_records = features.shape[0]
        # initialize weight steps
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        # save a copy for each weight step for momentum
        delta_weights_i_h_old, delta_weights_h_o_old = delta_weights_i_h.copy(), delta_weights_h_o.copy()
        ### Gradient descent with momentum ###
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h + 0.9 * delta_weights_i_h_old,
                            delta_weights_h_o + 0.9 * delta_weights_h_o_old, n_records)


    def forward_pass_train(self, X):
        '''

            Arguments
            ---------
            X: 1D array of feature values

        '''
        ### Forward pass ###
        ## matmul note
        # inner dimension is the number of input nodes
        # outer dimensions are the number of hidden nodes and 1
        hidden_inputs = np.matmul(self.weights_input_to_hidden.T, X[:, None])
        hidden_outputs = self.activation_function(hidden_inputs)

        ## matmul note
        # inner dimension is the number of hidden nodes
        # outer dimensions are 1 and 1 (effectively a dot product)
        final_inputs = np.matmul(self.weights_hidden_to_output.T, hidden_outputs)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            lr: learning rate

        '''
        ### Backward pass ###

        # identity function has gradient 1
        output_error_term = y - final_outputs
        ## matmul note
        # inner dimension is the number of outputs
        # outer dimensions are the number of hidden nodes and 1
        hidden_error_term = (np.matmul(self.weights_hidden_to_output, output_error_term)
                            * hidden_outputs * (1 - hidden_outputs))
        ## matmul note
        # inner dimension is 1
        # outer dimensions are the number of input nodes and the number of hidden nodes
        delta_weights_i_h += self.lr * np.matmul(X[:, None], hidden_error_term.T)
        # inner dimension is 1
        # outer dimensions are the number of hidden nodes and the number of output nodes
        delta_weights_h_o += self.lr * np.matmul(hidden_outputs, output_error_term.T)

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += delta_weights_h_o / n_records
        self.weights_input_to_hidden += delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 2D array of feature values
        '''
        #### Forward pass ####
        ## matmul note
        # inner dimension is the number of input nodes
        # outer dimensions are the number of hidden nodes and the number of records
        hidden_inputs = np.matmul(self.weights_input_to_hidden.T, features.T)
        hidden_outputs = self.activation_function(hidden_inputs)

        ## matmul note
        # inner dimension is the number of hidden nodes
        # outer dimensions are the number of output nodes and the number of records
        final_inputs = np.matmul(self.weights_hidden_to_output.T, hidden_outputs)
        final_outputs = final_inputs.T

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10000
learning_rate = 0.2
hidden_nodes = 14
output_nodes = 1
