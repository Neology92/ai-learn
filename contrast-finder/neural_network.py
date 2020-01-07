import numpy as np

class NeuralNetwork():

    self.weights = []    
    self.weights = []    

    def __init__(self, neurons = [1,1]):
        
        # Check if neurons have right configuration
        neurons  = list(map(int, neurons))
        if( neurons.__len__() < 2 or neurons[0] < 1 or neurons[-1] < 1):            
            import sys
            print('Wrong neurons/layers configuration. Network must have at least 1 input and 1 output', file=sys.stderr)
            hidden_layers = 0
            neurons = [1,1]

        self.inputs = neurons[0]
        self.outputs = neurons[-1]

        for i in range(0, neurons.__len__()-2):
            self.weights[i] = 2 * np.random.random(( i, i+1 )) - 1
            self.bias[i] = 2 * np.random.random(( 1, i+1 )) - 1
        

    
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # Gets sigmoid function as input
    def sigmoid_derivative(self, x):
        return x * (1 - x)