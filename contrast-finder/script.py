from create_sets import read_from_files
import numpy as np
import random as rnd


def sigmoid(x):
    return 1 / (1 + np.e**(-x))


X_train, Y_train, X_test, Y_test = read_from_files("./data")

# Normalize data
from sklearn.preprocessing import scale
X_train = scale(X_train)
Y_train = scale(Y_train)
X_test = scale(X_test)
Y_train = scale(Y_train)


synaptic_weights = 2 * np.random.random((3,1)) - 1
print("\nSynaptic weights: \n", synaptic_weights)

for i in range(1):
    
    input_layer = X_train[:4]

    output = sigmoid(np.dot(input_layer, synaptic_weights))
    
print("\nOutput: \n", output)