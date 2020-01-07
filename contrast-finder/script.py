from neural_network import NeuralNetwork
import numpy as np

if __name__ == "__main__":

    neural_network = NeuralNetwork([3,2,1])

    # input_arr = np.array([[1,0],[0,1]])
    # output = neural_network.feedforward(input_arr)
    # print(output)

    train_X = np.array([[1,0,0],
                        [1,1,1],
                        [0,1,1],
                        [0,1,0]])

    train_Y = np.array([[1],
                        [1],
                        [0],
                        [0]])

    neural_network.train(train_X, train_Y, 100)

    while(True):
        a = list(map( int, input("Get array[3]: ") ))

        output = neural_network.feedforward(a)
        print(output)