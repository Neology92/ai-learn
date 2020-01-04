from create_sets import read_from_files
import numpy as np
import random as rnd


def sigmoid(x):
    return 1 / (1 + np.e**(-x))

def sigmoid_derivative(x):
    return x - x**2


X_train, Y_train, X_test, Y_test = read_from_files("./data")

# Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1), copy=False)
X_train = scaler.fit_transform(X_train)
Y_train = np.vstack(( Y_train.T/255, (Y_train.T/255-1)**2 )).T
X_test = scaler.fit_transform(X_test)
Y_test = np.vstack(( Y_test.T/255, (Y_test.T/255-1)**2 )).T


synaptic_weights = 2 * np.random.random((3,2)) - 1
print("\nSynaptic weights: \n", synaptic_weights)

for i in range(1):
    
    input_layer = X_train[:4]
    output = sigmoid(np.dot(input_layer, synaptic_weights))


    cost = np.sum((output - Y_train[:4])**2)
    adjustments = output - Y_train[:4]

    # synaptic_weights -= adjustments
    print("\nCost: \n",cost)
    print("\nAdjustments: \n",adjustments)
    # print("\nNew synaptic weights: \n",synaptic_weights)


    # defect = output - Y_train[:4]
    # print("\nDefect: \n",defect)

    # adjustments = defect * sigmoid_derivative(output)
    
print("\nOutput: \n", output)
