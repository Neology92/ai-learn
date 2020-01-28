from neural_network import NeuralNetwork
import numpy as np
from mnist import MNIST


neural_network = NeuralNetwork([784,16,10])

mndata = MNIST('./data/digits_recognition')

X_train, train_labels = mndata.load_training()
X_test, test_labels = mndata.load_testing()


# Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1), copy=False)
X_train = np.array(X_train)/255
X_test = np.array(X_test)/255
Y_train = np.zeros((train_labels.__len__(), 10))
# Y_train = np.zeros((1000, 10))
i = 0
for digit in train_labels:
    Y_train[i][digit] = 1
    i += 1
Y_test = np.zeros((test_labels.__len__(), 10))
# Y_test = np.zeros((1000, 10))
i = 0
for digit in test_labels:
    Y_test[i][digit] = 1
    i += 1

print("====== Start Training ======")
cost = 1
# i = 0
while(cost > 900):
    cost = neural_network.train(X_train, Y_train, 100)
    # i += 1
    # if i == 1:
    print("(Training...) Cost: ", cost)
        # i = 0
neural_network.train(X_train, Y_train, 1000)
print("(Training...) Cost: ", cost)

print("")
print("===============")
print("Testing....")
print("---------------")

output = neural_network.feedforward(X_test)

rounded_output = np.around(output)
cost = np.sum((rounded_output - Y_test)**2)
print("\ncost: ", cost)


neural_network.save("digits_recognition")