from neural_network import NeuralNetwork
import numpy as np
from create_sets import read_from_files


neural_network = NeuralNetwork([3,4,1])


X_train, Y_train, X_test, Y_test = read_from_files("./data/contrast")

# Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1), copy=False)
X_train = X_train/255
Y_train = Y_train/255
X_test = X_test/255
Y_test = Y_test/255


# neural_network.load("contrast")

# print("")
# print("===============")
# print("Testing....")
# print("---------------")

# output = neural_network.feedforward(X_test)

# rounded_output = np.around(output)
# cost = np.sum((rounded_output - Y_test)**2)
# print("\ncost: ", cost)

# exit()

cost = 100
i = 0
while(cost > 50):
    cost = neural_network.train(X_train, Y_train, 100)
    i += 1
    if i%100 == 0:
        print(f"(Training... iteration: {i}) Cost: {cost}")

neural_network.train(X_train, Y_train, 1000)
print(f"(Training... iteration: {i+1000}) Cost: {cost}")


print("")
print("===============")
print("Testing....")
print("---------------")

output = neural_network.feedforward(X_test)

rounded_output = np.around(output)
cost = np.sum((rounded_output - Y_test)**2)
print("\ncost: ", cost)


neural_network.save("contrast")

