from create_sets import read_from_files
import numpy as np
import random as rnd


def sigmoid(x):
    # np.clip( x, -10, 10 )
    # print("\nnp.exp(-x)")
    # print(np.exp(-x))
    return 1.0 / (1 + np.exp(-x))

# Gets sigmoid function as input
def sigmoid_derivative(x):
    return x * (1 - x)


X_train, Y_train, X_test, Y_test = read_from_files("./data")

# Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1), copy=False)
X_train = scaler.fit_transform(X_train)
Y_train = np.vstack(( Y_train.T/255, (Y_train.T/255-1)**2 )).T
X_test = scaler.fit_transform(X_test)
Y_test = np.vstack(( Y_test.T/255, (Y_test.T/255-1)**2 )).T

print(np.shape(X_test))
print(np.shape(Y_test))

layer2_bias = 2 * np.random.random((1,3)) - 1
# print("\nLayer2 bias: \n", layer2_bias)
layer2_synaptic_weights = 2 * np.random.random((3,3)) - 1
print("\nLayer 2 Synaptic weights: \n", layer2_synaptic_weights)

output_bias = 2 * np.random.random((1,2)) - 1
print("\nOutput bias: \n", output_bias)
output_synaptic_weights = 2 * np.random.random((3,2)) - 1
print("\nOutput Synaptic weights: \n", output_synaptic_weights)
print("\n====================================================")

LEARNING_RATE = 0.1

for i in range(25000):
    
    input_layer = X_train
    # print("\nSynaptic weights: \n", synaptic_weights)
    layer2 = sigmoid(np.dot(input_layer, layer2_synaptic_weights) + layer2_bias)

    output = sigmoid(np.dot(layer2, output_synaptic_weights) + output_bias)


    cost = np.sum((np.around(output) - Y_train)**2)
    print("\nCost: ")
    print(cost)

    output_error = output - Y_train
    output_gradient = output_error * sigmoid_derivative(output)
    output_weights_delta = LEARNING_RATE * np.dot(layer2.T, output_gradient)
    output_bias_delta = LEARNING_RATE * output_gradient

    layer2_error = np.dot(output_error, output_synaptic_weights.T)
    layer2_gradient = layer2_error * sigmoid_derivative(layer2)
    layer2_delta = LEARNING_RATE * np.dot(input_layer.T, layer2_gradient)
    layer2_bias_delta = LEARNING_RATE * layer2_gradient


    # print("\nlayer2_delta2")
    # print(layer2_delta)
    # print("-----------------------------------")
    # print("\noutput_delta")
    # print(output_delta)

    output_synaptic_weights -= output_weights_delta
    output_bias -= np.dot(np.ones((1, np.ma.size(output_bias_delta,0) )), output_bias_delta)

    layer2_synaptic_weights -= layer2_delta
    layer2_bias -= np.dot(np.ones((1, np.ma.size(layer2_bias_delta,0) )), layer2_bias_delta)

    

# print("\nNew Layer 2 Synaptic weights: \n", layer2_synaptic_weights)
# print("\nNew Output Synaptic weights: \n", output_synaptic_weights)
print("\n============================")
# print("\nLayer2: ")
# print( layer2)
# print("\nNew synaptic weights: \n",synaptic_weights)
print("\nOutput: \n",np.around(output[:3]))

print("\n============================")

input_layer = X_test
layer2 = sigmoid(np.dot(input_layer, layer2_synaptic_weights))
output = sigmoid(np.dot(layer2, output_synaptic_weights))

cost = np.sum((np.around(output) - Y_test)**2)
print("\nCost: ")
print(cost)

# print("\n============================")


# while(True):
#     r = int(input("Get R: "))
#     g = int(input("Get G: "))
#     b = int(input("Get B: "))

#     input_layer = np.array([[r,g,b]])
#     layer2 = sigmoid(np.dot(input_layer, layer2_synaptic_weights))
#     output = sigmoid(np.dot(layer2, output_synaptic_weights))

#     print("Output: ", np.around(output))