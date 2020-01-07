import numpy as np

class NeuralNetwork():

    layers_number = 2
    neurons = []
    weights = []    
    bias = []    
    LEARNING_RATE = 0.1

    def __init__(self, neurons = [1,1]):
        
        # Check if neurons have right configuration
        self.neurons  = list(map(int, neurons))
        self.layers_number  = neurons.__len__()
        if( self.layers_number < 2 or neurons[0] < 1 or neurons[-1] < 1):            
            import sys
            print('\nError: Wrong neurons/layers configuration. \nNetwork must have at least 1 input and 1 output.', file=sys.stderr)
            self.layers_number = 2
            self.neurons = [1,1]
            print('-> 1 input and 1 output has been set.\n')

        self.inputs = self.neurons[0]
        self.outputs = self.neurons[-1]

        # Generate starting weights and bias
        for i in range(0, self.layers_number-1):
            self.weights.append( 2 * np.random.random(( self.neurons[i], self.neurons[i+1] )) - 1 )
            self.bias.append( 2 * np.random.random(( 1, self.neurons[i+1] )) - 1 )
        
    def feedforward(self, input_arr):
        neurons_values = []
        neurons_values.append(input_arr)
        for i in range(0, self.layers_number-1):
            # Calculate sum
            next_neurons_sums = np.dot(neurons_values[i], self.weights[i]) + self.bias[i]
            # Activation function
            neurons_values.append( self.sigmoid(next_neurons_sums) )

        return neurons_values[-1]

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # Gets sigmoid function as input
    def sigmoid_derivative(self, x):
        return x * (1 - x)



if __name__ == "__main__":

    neural_network = NeuralNetwork([1,1])

    output = neural_network.feedforward([1])
    print(output)