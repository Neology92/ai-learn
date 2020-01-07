from neural_network import NeuralNetwork
import numpy as np
from create_sets import read_from_files


neural_network = NeuralNetwork([3,2,1])


X_train, Y_train, X_test, Y_test = read_from_files("./data")

# Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1), copy=False)
X_train = X_train/255
Y_train = Y_train/255
X_test = X_test/255
Y_test = Y_test/255


cost = 999
while(cost > 0.05):
    cost = neural_network.train(X_train, Y_train, 100)

neural_network.train(X_train, Y_train, 1000)


print("")
print("===============")
print("Testing....")
print("---------------")

output = neural_network.feedforward(X_test)

rounded_output = np.around(output)
cost = np.sum((rounded_output - Y_test)**2)
print("\ncost: ", cost)

# input_arr = np.array([[1,0],[0,1]])
# output = neural_network.feedforward(input_arr)
# print(output)

# train_X = np.array([[1,0,0],
#                     [1,1,1],
#                     [0,1,1],
#                     [0,1,0]])

# train_Y = np.array([[1],
#                     [1],
#                     [0],
#                     [0]])

# while(True):
#     a = list(map( int, input("Get array[3]: ") ))

#     output = neural_network.feedforward(a)
#     print(output)

