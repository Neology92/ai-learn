from create_sets import read_from_files
import numpy as np
import random as rnd


def sigmoid(x):
    return 1 / (1 + np.e**(-x))


# X_train, Y_train, X_test, Y_test = read_from_files("./data")

synaptic_weights = 2 * np.random.random((3)) - 1

print (synaptic_weights)
