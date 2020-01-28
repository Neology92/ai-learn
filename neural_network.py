import numpy as np

class NeuralNetwork():

    layers_number = 2
    neurons = []
    weights = []    
    bias = []    
    LEARNING_RATE = 10
    BIAS_LEARNING_RATE = 2

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
        

    def train(self, train_X, train_Y, times):
        for t in range(times):

            # Forward propagation
            # -------------------
            neurons_values = []
            neurons_values.append(train_X)
            for i in range(0, self.layers_number-1):
                # Calculate sum
                next_neurons_sums = np.dot(neurons_values[i], self.weights[i]) + self.bias[i]
                # Activation function
                neurons_values.append( self.sigmoid(next_neurons_sums) )
            output = neurons_values[-1]

            cost = np.sum((output - train_Y)**2)
            # print("(Training...) Cost: ", cost)
            
            # Back propagation
            # -------------------
            errors = [0] * (self.layers_number-1)
            # Calc errors
            errors[-1] = train_Y - output
            for i in range(2, self.layers_number):
                errors[-i] = np.dot(errors[-(i-1)], self.weights[-(i-1)].T )

            # Calc new weights and bias
            for i in range(1, errors.__len__()+1):
                gradient = errors[i-1] * self.d_sigmoid(neurons_values[i])
                
                weights_delta = self.LEARNING_RATE * np.dot(neurons_values[i-1].T, gradient)
                gradient_rows = np.ma.size(gradient, 0)
                bias_delta = self.BIAS_LEARNING_RATE * np.dot( np.ones((1,gradient_rows)) , gradient)
                
                # Update weights and bias
                self.weights[i-1] += weights_delta
                self.bias[i-1] += bias_delta
            # end_for
        # end_for

        return cost
        
    def save(self, name):
        path = f'./trained_networks/{name}' 

                
        np.savez(f'{path}/neurons.npz', *self.neurons)
        np.savez(f'{path}/weights.npz', *self.weights)
        np.savez(f'{path}/bias.npz', *self.bias)

    def load(self, name):
        path = f'./trained_networks/{name}' 
        
        neurons_dict = np.load(f'{path}/neurons.npz')
        weights_dict = np.load(f'{path}/weights.npz')
        bias_dict = np.load(f'{path}/bias.npz')

        neurons = []
        for elem in neurons_dict.values():
            neurons.append(int(elem))

        weights = []
        for elem in weights_dict.values():
            weights.append(elem)

        bias = []
        for elem in bias_dict.values():
            bias.append(elem)

        self.neurons = neurons
        self.weights = weights
        self.bias = bias

    def sigmoid(self, x):
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()

    # Gets sigmoid output as input
    def d_sigmoid(self, y):
        return y * (1 - y)



if __name__ == "__main__":

    neural_network = NeuralNetwork([1,1])

    output = neural_network.feedforward([1])
    print(output)